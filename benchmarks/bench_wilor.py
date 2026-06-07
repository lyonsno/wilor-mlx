"""Reproducible benchmark for wilor-mlx vs PyTorch MPS WiLoR.

Usage:
    # MLX benchmark (no torch needed if using pre-converted weights)
    python benchmarks/bench_wilor.py --backend mlx --weights weights/wilor-mlx.safetensors

    # PyTorch MPS benchmark (requires torch + WiLoR-mini)
    python benchmarks/bench_wilor.py --backend pytorch \
        --ckpt pretrained_models/wilor_final.ckpt \
        --mano pretrained_models/MANO_RIGHT.pkl \
        --mean-params pretrained_models/mano_mean_params.npz

    # Both (for comparison table)
    python benchmarks/bench_wilor.py --backend both \
        --weights weights/wilor-mlx.safetensors \
        --ckpt pretrained_models/wilor_final.ckpt \
        --mano pretrained_models/MANO_RIGHT.pkl \
        --mean-params pretrained_models/mano_mean_params.npz
"""

import argparse
import gc
import json
import platform
import subprocess
import time


def get_chip_name():
    try:
        out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        return out
    except Exception:
        return platform.processor() or "unknown"


def bench_mlx(weights_path, warmup=30, iterations=100):
    import mlx.core as mx
    from wilor_mlx import WiLoR
    import numpy as np

    model = WiLoR.from_pretrained(weights_path)

    # Deterministic input
    np.random.seed(42)
    x_np = np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8)
    x = mx.array(x_np)
    mx.eval(x)

    gc.disable()

    # Warmup
    for _ in range(warmup):
        mx.eval(model(x))

    # Benchmark — batch of 10 iterations per measurement
    batch_size = 10
    n_batches = iterations // batch_size
    batch_times = []
    for _ in range(n_batches):
        start = time.perf_counter()
        for _ in range(batch_size):
            mx.eval(model(x))
        batch_times.append((time.perf_counter() - start) / batch_size * 1000)

    gc.enable()

    batch_times.sort()
    return {
        "backend": "mlx",
        "min_ms": round(batch_times[0], 1),
        "p50_ms": round(batch_times[len(batch_times) // 2], 1),
        "p90_ms": round(batch_times[int(len(batch_times) * 0.9)], 1),
        "p95_ms": round(batch_times[int(len(batch_times) * 0.95)], 1),
        "mean_ms": round(sum(batch_times) / len(batch_times), 1),
        "fps": round(1000 / batch_times[len(batch_times) // 2]),
        "iterations": iterations,
        "warmup": warmup,
    }


def bench_pytorch(ckpt_path, mano_path, mean_params_path, warmup=30, iterations=100):
    import torch
    import numpy as np
    import sys

    # Need WiLoR-mini on path
    wilor_mini_path = None
    for p in ["/private/tmp/wilor-mini-stride-bonewright/WiLoR-mini"]:
        if __import__("os").path.exists(p):
            wilor_mini_path = p
            break

    if wilor_mini_path is None:
        print("ERROR: WiLoR-mini not found. Clone it and set the path.")
        return None

    sys.path.insert(0, wilor_mini_path)
    from wilor_mini.models.wilor import WiLor

    model = WiLor(
        mano_model_path=mano_path,
        mano_mean_path=mean_params_path,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model = model.to("mps")

    # Same deterministic input
    np.random.seed(42)
    x_np = np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8)
    x = torch.from_numpy(x_np).float().to("mps")

    gc.disable()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            result = model(x)
            torch.mps.synchronize()

    # Benchmark
    batch_size = 10
    n_batches = iterations // batch_size
    batch_times = []
    with torch.no_grad():
        for _ in range(n_batches):
            start = time.perf_counter()
            for _ in range(batch_size):
                result = model(x)
                torch.mps.synchronize()
            batch_times.append((time.perf_counter() - start) / batch_size * 1000)

    gc.enable()

    batch_times.sort()
    return {
        "backend": "pytorch-mps",
        "min_ms": round(batch_times[0], 1),
        "p50_ms": round(batch_times[len(batch_times) // 2], 1),
        "p90_ms": round(batch_times[int(len(batch_times) * 0.9)], 1),
        "p95_ms": round(batch_times[int(len(batch_times) * 0.95)], 1),
        "mean_ms": round(sum(batch_times) / len(batch_times), 1),
        "fps": round(1000 / batch_times[len(batch_times) // 2]),
        "iterations": iterations,
        "warmup": warmup,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark wilor-mlx vs PyTorch MPS")
    parser.add_argument("--backend", choices=["mlx", "pytorch", "both"], default="mlx")
    parser.add_argument("--weights", default="weights/wilor-mlx.safetensors")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--mano", default=None)
    parser.add_argument("--mean-params", default=None)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    chip = get_chip_name()
    results = {"chip": chip, "benchmarks": []}

    if args.backend in ("mlx", "both"):
        print(f"Benchmarking MLX on {chip}...")
        mlx_result = bench_mlx(args.weights, args.warmup, args.iterations)
        results["benchmarks"].append(mlx_result)

    if args.backend in ("pytorch", "both"):
        if not all([args.ckpt, args.mano, args.mean_params]):
            print("ERROR: --ckpt, --mano, --mean-params required for pytorch backend")
            return
        print(f"Benchmarking PyTorch MPS on {chip}...")
        pt_result = bench_pytorch(args.ckpt, args.mano, args.mean_params,
                                   args.warmup, args.iterations)
        if pt_result:
            results["benchmarks"].append(pt_result)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"WiLoR Benchmark — {chip}")
        print(f"Input: (1, 256, 256, 3) uint8, {args.iterations} iterations")
        print(f"{'='*60}")
        print(f"{'Backend':<15} {'p50':>8} {'p90':>8} {'p95':>8} {'min':>8} {'FPS':>6}")
        print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for b in results["benchmarks"]:
            print(f"{b['backend']:<15} {b['p50_ms']:>7.1f}ms {b['p90_ms']:>7.1f}ms {b['p95_ms']:>7.1f}ms {b['min_ms']:>7.1f}ms {b['fps']:>5}")

        if len(results["benchmarks"]) == 2:
            mlx = next(b for b in results["benchmarks"] if b["backend"] == "mlx")
            pt = next(b for b in results["benchmarks"] if b["backend"] == "pytorch-mps")
            speedup = pt["p50_ms"] / mlx["p50_ms"]
            print(f"\nSpeedup (p50): {speedup:.1f}x")


if __name__ == "__main__":
    main()
