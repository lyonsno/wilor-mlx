"""Chunked cooperative-GPU WiLoR inference (Burst Butcher).

Splits the ViT transformer loop into K near-equal segments, forcing an
`mx.eval()` + optional yield between segments so the monolithic ~37ms Metal
burst is broken into ~7-15ms slices. This lets a contending GPU consumer
(e.g. a browser WebGPU render loop) interleave its command buffers between
segments instead of waiting behind one long burst.

Drop-in for a monolithic call. Output is BITWISE-IDENTICAL to `model(x)` /
`pipeline.pose_model(x)` — eval boundaries change *when* the graph is
submitted, not the math. Uncontended overhead is ~1.01-1.04x with no yield.

Integration (sidecar side, one line at pipeline.py:253):

    from wilor_mlx.chunked import run_model_chunked
    # was: wilor_out = self.pose_model(crop[None])
    wilor_out = run_model_chunked(self.pose_model, crop[None], segments=7)

No product-side contract is required; this holds no state and mirrors
`WiLoR.__call__` / `WiLoRViT.__call__` exactly except for the chunked loop.
"""

import time

import mlx.core as mx

from wilor_mlx.model import rotmat_to_rotvec
from wilor_mlx.vit import rot6d_to_rotmat


def _yield_gpu(mode, yield_ms):
    """Explicit yield between segments so the contending consumer can run."""
    if mode == "sleep" and yield_ms > 0:
        time.sleep(yield_ms / 1000.0)
    # "none"/"sync": mx.eval already synchronized; nothing more to do.


def _segment_bounds(depth, segments):
    """Split `depth` blocks into `segments` near-equal contiguous groups."""
    base, rem = divmod(depth, segments)
    bounds, start = [], 0
    for i in range(segments):
        n = base + (1 if i < rem else 0)
        bounds.append((start, start + n))
        start += n
    return bounds


def run_backbone_chunked(backbone, x, segments=7, yield_mode="none",
                         yield_ms=0.0, seg_times=None):
    """Run WiLoRViT forward with the transformer loop split into `segments`
    groups (mx.eval + yield between groups). Returns the same 4-tuple as
    `WiLoRViT.__call__`.
    """
    B = x.shape[0]

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

    for (lo, hi) in _segment_bounds(len(backbone.blocks), segments):
        t0 = time.perf_counter()
        for bi in range(lo, hi):
            h = backbone.blocks[bi](h)
        mx.eval(h)  # force this segment's Metal submission to complete
        if seg_times is not None:
            seg_times.append((time.perf_counter() - t0) * 1000)
        _yield_gpu(yield_mode, yield_ms)

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


def run_model_chunked(model, x, segments=7, yield_mode="none", yield_ms=0.0,
                      seg_times=None):
    """Full WiLoR forward using the chunked backbone. Drop-in for `model(x)`;
    bitwise-identical output. Mirrors `WiLoR.__call__`.

    Args:
        model: a loaded `WiLoR` instance.
        x: (B, 256, 256, 3) uint8 NHWC RGB (same as `WiLoR.__call__`).
        segments: number of cooperative GPU segments (4-8; 7 is a good default).
        yield_mode: "none" (free, lets Metal schedule) or "sleep" (explicit
            CPU yield of `yield_ms` between segments).
        yield_ms: sleep duration per boundary when yield_mode="sleep".
        seg_times: optional list; appended with per-segment wall ms for profiling.
    """
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
