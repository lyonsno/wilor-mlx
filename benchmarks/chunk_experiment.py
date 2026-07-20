#!/usr/bin/env python3
"""WiLoR Burst Butcher — chunked cooperative-GPU inference experiment.

Sidecar-only. Does NOT modify wilor_mlx core. Wraps a loaded WiLoR model
and re-runs the ViT transformer loop in K segments, forcing mx.eval() +
an explicit yield between segments so the browser's Metal command buffers
can interleave instead of being starved by one ~37ms monolithic burst.

Mechanism (per charter):
  - MLX is lazy: the monolithic path builds one giant graph, evaluated once.
  - Splitting the 32-block loop into K groups and calling mx.eval() at each
    group boundary breaks the burst into K shorter Metal submissions.
  - A yield (sleep or mx.synchronize) between segments gives the OS/GPU
    scheduler a window to service the contending consumer (Chrome WebGPU).

Success gates measured here:
  1. Overhead ratio = chunked_wall / monolithic_wall  (target <=2.0)
  3. Correctness: chunked landmarks == monolithic within float tolerance.

Gate 2 (browser frame impact) is measured by the contended harness while the
livingworld product runs; this file also emits per-segment timing so the
contended run can be interpreted.

Usage:
    python benchmarks/chunk_experiment.py \
        --weights weights/wilor-mlx.safetensors --mano-npz weights/mano.npz \
        --segments 4 --yield-mode sleep --yield-ms 0.5
"""

import argparse
import time
import json

import numpy as np
import mlx.core as mx

from wilor_mlx import WiLoR
from wilor_mlx.model import rotmat_to_rotvec
from wilor_mlx.vit import rot6d_to_rotmat


def _yield_gpu(mode, yield_ms):
    """Explicit yield between segments so the contending consumer can run."""
    if mode == "none":
        return
    if mode == "sync":
        # mx.eval already synchronized; this is a no-op placeholder for symmetry.
        return
    if mode == "sleep":
        if yield_ms > 0:
            time.sleep(yield_ms / 1000.0)
        return
    raise ValueError(f"unknown yield mode: {mode}")


def _segment_bounds(depth, segments):
    """Split `depth` blocks into `segments` near-equal contiguous groups."""
    base = depth // segments
    rem = depth % segments
    bounds = []
    start = 0
    for i in range(segments):
        n = base + (1 if i < rem else 0)
        bounds.append((start, start + n))
        start += n
    return bounds


def run_backbone_chunked(backbone, x, segments, yield_mode, yield_ms,
                         seg_times=None):
    """Run WiLoRViT forward with the transformer loop split into `segments`
    groups, forcing mx.eval() + yield between groups.

    Mirrors WiLoRViT.__call__ exactly except for the chunked block loop.
    Returns the same 4-tuple as the monolithic backbone.
    """
    B = x.shape[0]

    # --- identical prologue to WiLoRViT.__call__ ---
    h, (Hp, Wp) = backbone.patch_embed(x)
    h = h + backbone.pos_embed[:, 1:] + backbone.pos_embed[:, :1]

    pose_tokens = backbone.pose_emb(
        mx.broadcast_to(
            backbone.init_hand_pose.reshape(
                1, backbone.num_hand_joints + 1, backbone.joint_rep_dim),
            (B, backbone.num_hand_joints + 1, backbone.joint_rep_dim),
        )
    )
    shape_tokens = mx.broadcast_to(
        backbone.shape_emb(backbone.init_betas)[:, None, :],
        (B, 1, backbone.embed_dim),
    )
    cam_tokens = mx.broadcast_to(
        backbone.cam_emb(backbone.init_cam)[:, None, :],
        (B, 1, backbone.embed_dim),
    )
    h = mx.concatenate([pose_tokens, shape_tokens, cam_tokens, h], axis=1)

    # --- chunked transformer loop: the burst butcher core ---
    bounds = _segment_bounds(len(backbone.blocks), segments)
    for (lo, hi) in bounds:
        t0 = time.perf_counter()
        for bi in range(lo, hi):
            h = backbone.blocks[bi](h)
        mx.eval(h)  # force this segment's Metal submission to complete
        if seg_times is not None:
            seg_times.append((time.perf_counter() - t0) * 1000)
        _yield_gpu(yield_mode, yield_ms)

    # --- identical epilogue to WiLoRViT.__call__ ---
    h = backbone.last_norm(h)
    n_pose = backbone.num_hand_joints + 1
    pose_feat = h[:, :n_pose]
    shape_feat = h[:, n_pose:n_pose + 1]
    cam_feat = h[:, n_pose + 1:n_pose + 2]

    pred_hand_pose = backbone.decpose(pose_feat).reshape(B, -1) + backbone.init_hand_pose
    pred_betas = backbone.decshape(shape_feat).reshape(B, -1) + backbone.init_betas
    pred_cam = backbone.deccam(cam_feat).reshape(B, -1) + backbone.init_cam

    pred_mano_feats = {'hand_pose': pred_hand_pose, 'betas': pred_betas, 'cam': pred_cam}
    pred_hand_pose_rot = rot6d_to_rotmat(pred_hand_pose).reshape(B, n_pose, 3, 3)
    pred_mano_params = {
        'global_orient': pred_hand_pose_rot[:, :1],
        'hand_pose': pred_hand_pose_rot[:, 1:],
        'betas': pred_betas,
    }
    img_feat = h[:, n_pose + 2:]
    img_feat = img_feat.reshape(B, Hp, Wp, -1).transpose(0, 3, 1, 2)
    return pred_mano_params, pred_cam, pred_mano_feats, img_feat


def run_model_chunked(model, x, segments, yield_mode, yield_ms, seg_times=None):
    """Full WiLoR forward using the chunked backbone. Mirrors WiLoR.__call__."""
    x = x.astype(mx.float32) / 255.0
    x = (x - model.IMAGE_MEAN) / model.IMAGE_STD
    B = x.shape[0]
    backbone_input = x[:, :, 32:-32, :]

    temp_mano_params, pred_cam, pred_mano_feats, vit_out = run_backbone_chunked(
        model.backbone, backbone_input, segments, yield_mode, yield_ms, seg_times)

    focal_length = mx.full((B, 2), model.FOCAL_LENGTH)
    temp_vertices, _ = model.mano(
        global_orient=temp_mano_params['global_orient'],
        hand_pose=temp_mano_params['hand_pose'],
        betas=temp_mano_params['betas'], pose2rot=False)

    pred_mano_params = model.refine_net(
        vit_out, temp_vertices, pred_cam, pred_mano_feats, focal_length)

    pred_vertices, pred_keypoints_3d = model.mano(
        global_orient=pred_mano_params['global_orient'],
        hand_pose=pred_mano_params['hand_pose'],
        betas=pred_mano_params['betas'], pose2rot=False)

    pred_mano_params['pred_keypoints_3d'] = pred_keypoints_3d.reshape(B, -1, 3)
    pred_mano_params['pred_vertices'] = pred_vertices.reshape(B, -1, 3)
    if model.mano.faces is not None:
        pred_mano_params['faces'] = model.mano.faces
    pred_mano_params['global_orient'] = rotmat_to_rotvec(pred_mano_params['global_orient'])
    pred_mano_params['hand_pose'] = rotmat_to_rotvec(pred_mano_params['hand_pose'])
    return pred_mano_params


def _pctl(ts, p):
    return sorted(ts)[min(len(ts) - 1, int(len(ts) * p))]


def bench(model, x, run_fn, warmup, iters):
    for _ in range(warmup):
        mx.eval(run_fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(run_fn())
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {
        'min': round(ts[0], 2),
        'p50': round(ts[len(ts) // 2], 2),
        'p95': round(_pctl(ts, 0.95), 2),
        'mean': round(sum(ts) / len(ts), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/wilor-mlx.safetensors")
    ap.add_argument("--mano-npz", default="weights/mano.npz")
    ap.add_argument("--segments", type=int, nargs="+", default=[1, 2, 4, 7, 8])
    ap.add_argument("--yield-mode", choices=["none", "sync", "sleep"], default="sleep")
    ap.add_argument("--yield-ms", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--contended", action="store_true",
                    help="Tag results as contended (product running).")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    model = WiLoR.from_pretrained(weights_path=args.weights, mano_path=args.mano_npz)
    np.random.seed(42)
    x = mx.array(np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8))
    mx.eval(x)

    # Monolithic baseline (the real WiLoR.__call__ path).
    mono = bench(model, x, lambda: model(x), args.warmup, args.iters)

    # Correctness reference from monolithic path.
    ref = model(x)
    ref_kp = np.array(ref['pred_keypoints_3d'])

    results = {
        'contended': args.contended,
        'yield_mode': args.yield_mode,
        'yield_ms': args.yield_ms,
        'monolithic': mono,
        'chunked': [],
    }

    print(f"{'segs':>4} {'p50':>8} {'p95':>8} {'overhead':>9} "
          f"{'seg_p50':>8} {'max_kp_err':>11}")
    print(f"monolithic p50={mono['p50']}ms p95={mono['p95']}ms "
          f"(contended={args.contended})")

    for segs in args.segments:
        seg_times = []
        chunk = bench(
            model, x,
            lambda s=segs, st=seg_times: run_model_chunked(
                model, x, s, args.yield_mode, args.yield_ms, st),
            args.warmup, args.iters)

        # Correctness check on a fresh chunked run.
        chunk_out = run_model_chunked(model, x, segs, args.yield_mode, args.yield_ms)
        chunk_kp = np.array(chunk_out['pred_keypoints_3d'])
        max_err = float(np.abs(chunk_kp - ref_kp).max())

        # Per-segment p50 (segments accumulated across iters; take median).
        per_seg = round(float(np.median(seg_times)), 3) if seg_times else 0.0
        overhead = round(chunk['p50'] / mono['p50'], 3)

        row = {
            'segments': segs, **chunk,
            'overhead_ratio': overhead,
            'seg_p50_ms': per_seg,
            'max_kp_err': max_err,
        }
        results['chunked'].append(row)
        print(f"{segs:>4} {chunk['p50']:>7.2f}m {chunk['p95']:>7.2f}m "
              f"{overhead:>8.2f}x {per_seg:>7.2f}m {max_err:>11.2e}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
