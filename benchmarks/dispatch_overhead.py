#!/usr/bin/env python3
"""Measure Metal dispatch overhead in WiLoR's ViT backbone.

Compares wall-clock time of:
1. Full 32-block ViT forward (real WiLoR shapes)
2. Single transformer block (to see per-block cost)
3. Equivalent-FLOP single matmul (dispatch overhead floor)

WiLoR ViT shapes:
- B=1, N=210 tokens, D=1280 embed dim, H=16 heads, head_dim=80
- MLP hidden: 5120 (4x ratio)
- 32 transformer blocks

Per block, the operations are roughly:
- LayerNorm (2 ops: normalize + affine)
- QKV linear (1 matmul + bias)
- SDPA (multiple internal dispatches)
- Output projection (1 matmul + bias)
- Residual add
- LayerNorm (2 ops)
- MLP fc1 (1 matmul + bias)
- GELU activation
- MLP fc2 (1 matmul + bias)
- Residual add

That's ~12-15 Metal dispatches per block, ~384-480 for 32 blocks.

Usage:
    python benchmarks/dispatch_overhead.py
    python benchmarks/dispatch_overhead.py --warmup 10 --iterations 50
"""

import argparse
import time
import json
import platform

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def time_fn(fn, warmup=10, iterations=50):
    """Time a function, returning percentile stats in ms."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Measure
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = sorted(times)
    n = len(times)
    return {
        'n': n,
        'min': round(times[0], 3),
        'p50': round(times[n // 2], 3),
        'p90': round(times[int(n * 0.9)], 3),
        'p95': round(times[int(n * 0.95)], 3),
        'p99': round(times[int(n * 0.99)], 3),
        'max': round(times[-1], 3),
        'mean': round(sum(times) / n, 3),
    }


class SingleBlock(nn.Module):
    """One transformer block with WiLoR shapes."""
    def __init__(self, dim=1280, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

    def __call__(self, x):
        # Attention
        h = self.norm1(x)
        B, N, C = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(0, 3, 2, 1, 4)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        h = h.transpose(0, 2, 1, 3).reshape(B, N, C)
        h = self.proj(h)
        x = x + h
        # MLP
        h = self.norm2(x)
        h = self.fc1(h)
        h = nn.gelu(h)
        h = self.fc2(h)
        x = x + h
        return x


class StackedBlocks(nn.Module):
    """N transformer blocks stacked."""
    def __init__(self, depth=32, dim=1280, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.blocks = [SingleBlock(dim, num_heads, mlp_ratio) for _ in range(depth)]

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def count_block_flops(B=1, N=210, D=1280, mlp_ratio=4):
    """Rough FLOP count for one transformer block."""
    H = D * int(mlp_ratio)  # 5120
    # QKV projection: B*N*D * 3*D * 2
    qkv_flops = B * N * D * 3 * D * 2
    # Attention QK^T: B*heads*N*N*head_dim * 2
    head_dim = D // 16
    attn_flops = B * 16 * N * N * head_dim * 2
    # Attention PV: same as QK^T
    pv_flops = attn_flops
    # Output projection: B*N*D*D*2
    proj_flops = B * N * D * D * 2
    # MLP fc1: B*N*D*H*2
    fc1_flops = B * N * D * H * 2
    # MLP fc2: B*N*H*D*2
    fc2_flops = B * N * H * D * 2
    total = qkv_flops + attn_flops + pv_flops + proj_flops + fc1_flops + fc2_flops
    return total


def main():
    parser = argparse.ArgumentParser(description="Measure Metal dispatch overhead in WiLoR ViT")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--json", type=str, default=None, help="Write results to JSON file")
    args = parser.parse_args()

    B, N, D = 1, 210, 1280
    mlp_ratio = 4
    depth = 32

    print(f"WiLoR ViT dispatch overhead benchmark")
    print(f"  Shape: B={B}, N={N}, D={D}, heads=16, MLP ratio={mlp_ratio}, depth={depth}")
    print(f"  Warmup: {args.warmup}, Iterations: {args.iterations}")
    print(f"  Device: {platform.processor()}")
    print(f"  MLX: {mx.__version__}")
    print()

    block_flops = count_block_flops(B, N, D, mlp_ratio)
    total_flops = block_flops * depth
    print(f"  Per-block FLOPs: {block_flops:,}")
    print(f"  Total 32-block FLOPs: {total_flops:,}")
    print()

    # Create random input (same shape as ViT internal)
    x = mx.random.normal((B, N, D))
    mx.eval(x)

    results = {}

    # --- Test 1: Single transformer block ---
    print("1. Single transformer block...")
    block = SingleBlock(D, 16, mlp_ratio)
    # Initialize weights
    out = block(x)
    mx.eval(out)

    def run_single_block():
        out = block(x)
        mx.eval(out)

    stats = time_fn(run_single_block, args.warmup, args.iterations)
    results['single_block'] = stats
    print(f"   p50={stats['p50']:.3f}ms  p90={stats['p90']:.3f}ms  mean={stats['mean']:.3f}ms")

    # --- Test 2: Full 32-block stack ---
    print("2. Full 32-block ViT backbone...")
    stack = StackedBlocks(depth, D, 16, mlp_ratio)
    out = stack(x)
    mx.eval(out)

    def run_full_stack():
        out = stack(x)
        mx.eval(out)

    stats = time_fn(run_full_stack, args.warmup, args.iterations)
    results['full_32_blocks'] = stats
    print(f"   p50={stats['p50']:.3f}ms  p90={stats['p90']:.3f}ms  mean={stats['mean']:.3f}ms")

    # --- Test 3: Equivalent-FLOP single matmul (dispatch floor) ---
    # One block is dominated by 4 big matmuls:
    #   QKV: (210, 1280) @ (1280, 3840)
    #   Proj: (210, 1280) @ (1280, 1280)
    #   FC1: (210, 1280) @ (1280, 5120)
    #   FC2: (210, 5120) @ (5120, 1280)
    # Total matmul FLOPs per block ≈ the bulk of block_flops
    # For 32 blocks, we'd need a massive single matmul.
    # Instead, measure one big matmul at per-block FLOP scale.
    print("3. Single matmul at per-block FLOP scale...")
    # (210, 1280) @ (1280, 5120) — this is the fc1 shape, largest single op
    a = mx.random.normal((N, D))
    b = mx.random.normal((D, D * mlp_ratio))
    mx.eval(a, b)

    def run_single_matmul():
        out = a @ b
        mx.eval(out)

    stats = time_fn(run_single_matmul, args.warmup, args.iterations)
    results['single_matmul_fc1_shape'] = stats
    print(f"   p50={stats['p50']:.3f}ms  p90={stats['p90']:.3f}ms  mean={stats['mean']:.3f}ms")

    # --- Test 4: 32 sequential matmuls (dispatch overhead of 32 evals) ---
    print("4. 32 sequential matmuls (same shape, 32 dispatches)...")

    def run_32_matmuls():
        for _ in range(32):
            out = a @ b
            mx.eval(out)

    stats = time_fn(run_32_matmuls, args.warmup, args.iterations)
    results['32_sequential_matmuls'] = stats
    print(f"   p50={stats['p50']:.3f}ms  p90={stats['p90']:.3f}ms  mean={stats['mean']:.3f}ms")

    # --- Test 5: 32 matmuls in one graph (single eval) ---
    print("5. 32 chained matmuls in one graph (single dispatch)...")
    sq_a = mx.random.normal((N, D))
    sq_b = mx.random.normal((D, D))
    mx.eval(sq_a, sq_b)

    def run_32_chained_square():
        h = sq_a
        for _ in range(32):
            h = h @ sq_b
        mx.eval(h)

    stats = time_fn(run_32_chained_square, args.warmup, args.iterations)
    results['32_chained_matmuls_single_eval'] = stats
    print(f"   p50={stats['p50']:.3f}ms  p90={stats['p90']:.3f}ms  mean={stats['mean']:.3f}ms")

    # --- Test 6: 32 sequential evals of the same matmul ---
    print("6. 32 sequential evals of (210,1280)@(1280,1280) — pure dispatch overhead...")

    def run_32_eval_overhead():
        for _ in range(32):
            out = sq_a @ sq_b
            mx.eval(out)

    stats = time_fn(run_32_eval_overhead, args.warmup, args.iterations)
    results['32_sequential_evals_square'] = stats
    print(f"   p50={stats['p50']:.3f}ms  p90={stats['p90']:.3f}ms  mean={stats['mean']:.3f}ms")

    # --- Analysis ---
    print()
    print("=== Analysis ===")
    single_block_ms = results['single_block']['p50']
    full_stack_ms = results['full_32_blocks']['p50']
    per_block_from_stack = full_stack_ms / 32

    single_matmul_ms = results['single_matmul_fc1_shape']['p50']

    chained_32_ms = results['32_chained_matmuls_single_eval']['p50']
    sequential_32_ms = results['32_sequential_evals_square']['p50']

    print(f"  Single block:         {single_block_ms:.3f}ms")
    print(f"  Full 32 blocks:       {full_stack_ms:.3f}ms  ({per_block_from_stack:.3f}ms/block)")
    print(f"  Single fc1 matmul:    {single_matmul_ms:.3f}ms")
    print(f"  32 chained (1 eval):  {chained_32_ms:.3f}ms  ({chained_32_ms/32:.3f}ms/matmul)")
    print(f"  32 sequential evals:  {sequential_32_ms:.3f}ms  ({sequential_32_ms/32:.3f}ms/eval)")
    print()

    dispatch_overhead_per_eval = (sequential_32_ms - chained_32_ms) / 32
    print(f"  Estimated dispatch overhead per eval: {dispatch_overhead_per_eval:.3f}ms")
    print(f"  Overhead as % of single block: {dispatch_overhead_per_eval / single_block_ms * 100:.1f}%")

    # Each block does ~12-15 kernel dispatches within one eval.
    # The real question is: how much overhead is inside the graph compilation?
    overhead_in_stack = full_stack_ms - chained_32_ms
    print(f"  Full stack - 32 chained matmuls: {overhead_in_stack:.3f}ms")
    print(f"  This delta includes: LayerNorm, SDPA, GELU, residuals, reshapes,")
    print(f"  and in-graph dispatch overhead for ~12-15 ops/block vs 1 op/block")

    if args.json:
        results['analysis'] = {
            'per_block_from_stack_ms': round(per_block_from_stack, 3),
            'dispatch_overhead_per_eval_ms': round(dispatch_overhead_per_eval, 3),
            'overhead_pct_of_block': round(dispatch_overhead_per_eval / single_block_ms * 100, 1),
            'full_stack_minus_chained_ms': round(overhead_in_stack, 3),
        }
        results['config'] = {
            'B': B, 'N': N, 'D': D, 'depth': depth,
            'mlp_ratio': mlp_ratio, 'heads': 16,
            'warmup': args.warmup, 'iterations': args.iterations,
            'device': platform.processor(),
            'mlx_version': mx.__version__,
        }
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results written to {args.json}")


if __name__ == '__main__':
    main()
