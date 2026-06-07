"""WiLoR RefineNet ported to MLX.

Takes ViT image features + initial MANO vertices, refines MANO parameters.
Key challenge: grid_sample implemented as pure MLX bilinear interpolation.
"""

import math
import mlx.core as mx
import mlx.nn as nn
from wilor_mlx.vit import rot6d_to_rotmat


def grid_sample(input, grid, align_corners=True):
    """Bilinear grid sampling — pure MLX replacement for F.grid_sample.

    Args:
        input: (B, C, H, W) feature map — NCHW layout
        grid: (B, Hout, Wout, 2) normalized coordinates in [-1, 1]
        align_corners: whether -1 and 1 map to pixel centers of corner pixels
    Returns:
        (B, C, Hout, Wout) sampled features
    """
    B, C, H, W = input.shape

    if align_corners:
        ix = (grid[..., 0] + 1) / 2 * (W - 1)
        iy = (grid[..., 1] + 1) / 2 * (H - 1)
    else:
        ix = ((grid[..., 0] + 1) * W - 1) / 2
        iy = ((grid[..., 1] + 1) * H - 1) / 2

    ix0 = mx.floor(ix).astype(mx.int32)
    iy0 = mx.floor(iy).astype(mx.int32)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    wx1 = ix - ix0.astype(mx.float32)
    wy1 = iy - iy0.astype(mx.float32)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    ix0 = mx.clip(ix0, 0, W - 1)
    ix1 = mx.clip(ix1, 0, W - 1)
    iy0 = mx.clip(iy0, 0, H - 1)
    iy1 = mx.clip(iy1, 0, H - 1)

    def gather_2d(inp, iy, ix):
        B, C, H, W = inp.shape
        idx = (iy * W + ix).reshape(B, 1, -1)
        idx = mx.broadcast_to(idx, (B, C, idx.shape[2]))
        inp_flat = inp.reshape(B, C, H * W)
        result = mx.take_along_axis(inp_flat, idx, axis=2)
        return result.reshape(B, C, iy.shape[1], iy.shape[2])

    v00 = gather_2d(input, iy0, ix0)
    v01 = gather_2d(input, iy0, ix1)
    v10 = gather_2d(input, iy1, ix0)
    v11 = gather_2d(input, iy1, ix1)

    wx0 = mx.expand_dims(wx0, 1)
    wx1 = mx.expand_dims(wx1, 1)
    wy0 = mx.expand_dims(wy0, 1)
    wy1 = mx.expand_dims(wy1, 1)

    return wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11)


def sample_joint_features(img_feat, joint_xy):
    """Sample image features at projected joint locations.

    Args:
        img_feat: (B, C, H, W) feature map — NCHW
        joint_xy: (B, J, 2) 2D joint coordinates in pixel space
    Returns:
        (B, J, C) sampled features
    """
    height, width = img_feat.shape[2], img_feat.shape[3]
    x = joint_xy[:, :, 0] / (width - 1) * 2 - 1
    y = joint_xy[:, :, 1] / (height - 1) * 2 - 1
    # grid_sample expects (B, Hout, Wout, 2)
    g = mx.stack([x, y], axis=2)[:, :, None, :]  # (B, J, 1, 2)
    sampled = grid_sample(img_feat, g, align_corners=True)  # (B, C, J, 1)
    sampled = sampled[:, :, :, 0]  # (B, C, J)
    sampled = sampled.transpose(0, 2, 1)  # (B, J, C)
    return sampled


def perspective_projection(points, translation, focal_length, camera_center=None):
    """Project 3D points to 2D using perspective projection.

    Args:
        points: (B, N, 3)
        translation: (B, 3)
        focal_length: (B, 2)
        camera_center: (B, 2) optional
    Returns:
        (B, N, 2) projected 2D points
    """
    batch_size = points.shape[0]
    if camera_center is None:
        camera_center = mx.zeros((batch_size, 2))

    K = mx.zeros((batch_size, 3, 3))
    # Build K matrix — need to do element-wise since MLX doesn't support item assignment
    # K[:, 0, 0] = focal_length[:, 0]
    # K[:, 1, 1] = focal_length[:, 1]
    # K[:, 2, 2] = 1
    # K[:, 0, 2] = camera_center[:, 0]
    # K[:, 1, 2] = camera_center[:, 1]
    row0 = mx.stack([focal_length[:, 0], mx.zeros(batch_size), camera_center[:, 0]], axis=1)
    row1 = mx.stack([mx.zeros(batch_size), focal_length[:, 1], camera_center[:, 1]], axis=1)
    row2 = mx.stack([mx.zeros(batch_size), mx.zeros(batch_size), mx.ones(batch_size)], axis=1)
    K = mx.stack([row0, row1, row2], axis=1)  # (B, 3, 3)

    # Transform points
    points = points + translation[:, None, :]

    # Perspective division
    projected = points / points[:, :, 2:3]

    # Apply camera intrinsics
    projected = mx.einsum('bij,bkj->bki', K, projected)

    return projected[:, :, :2]


class DeConvNet(nn.Module):
    """Multi-scale deconvolution network.

    Exact architecture from WiLoR checkpoint (feat_dim=1280, upscale=3):
    - first_conv: Conv2d(1280→640, 1x1)
    - branch 0: ConvTranspose2d(640→320) + BN(320)
    - branch 1: ConvTranspose2d(640→320) + BN(320) + ReLU + ConvTranspose2d(320→160) + BN(160)
    Output pyramid: [160-ch (high res), 320-ch (mid), 640-ch (low res)]
    """

    def __init__(self, feat_dim=1280, upscale=3):
        super().__init__()
        self.first_conv = nn.Conv2d(feat_dim, feat_dim // 2,
                                     kernel_size=1, stride=1, padding=0)

        # Branch 0: 640 → 320
        self.branch0_conv = nn.ConvTranspose2d(feat_dim // 2, feat_dim // 4,
                                                kernel_size=4, stride=2, padding=1)
        self.branch0_bn = nn.BatchNorm(feat_dim // 4)

        # Branch 1: 640 → 320 → 160
        self.branch1_conv0 = nn.ConvTranspose2d(feat_dim // 2, feat_dim // 4,
                                                 kernel_size=4, stride=2, padding=1)
        self.branch1_bn0 = nn.BatchNorm(feat_dim // 4)
        self.branch1_conv1 = nn.ConvTranspose2d(feat_dim // 4, feat_dim // 8,
                                                 kernel_size=4, stride=2, padding=1)
        self.branch1_bn1 = nn.BatchNorm(feat_dim // 8)

    def __call__(self, img_feat):
        # img_feat: (B, C, H, W) NCHW → NHWC for MLX conv
        B, C, H, W = img_feat.shape
        x = img_feat.transpose(0, 2, 3, 1)  # NHWC
        x = nn.relu(self.first_conv(x))

        # Low res feature: 640-ch
        feat_low = x.transpose(0, 3, 1, 2)  # NCHW

        # Branch 0: → 320-ch
        b0 = self.branch0_conv(x)
        b0 = self.branch0_bn(b0)
        feat_mid = b0.transpose(0, 3, 1, 2)  # NCHW

        # Branch 1: → 160-ch
        b1 = self.branch1_conv0(x)
        b1 = nn.relu(self.branch1_bn0(b1))
        b1 = self.branch1_conv1(b1)
        b1 = self.branch1_bn1(b1)
        feat_high = b1.transpose(0, 3, 1, 2)  # NCHW

        # Return high → low resolution
        return [feat_high, feat_mid, feat_low]


class RefineNet(nn.Module):
    """WiLoR refinement network."""

    def __init__(self, feat_dim=1280, upscale=3):
        super().__init__()
        self.deconv = DeConvNet(feat_dim=feat_dim, upscale=upscale)
        self.out_dim = feat_dim // 8 + feat_dim // 4 + feat_dim // 2
        self.dec_pose = nn.Linear(self.out_dim, 96)
        self.dec_cam = nn.Linear(self.out_dim, 3)
        self.dec_shape = nn.Linear(self.out_dim, 10)

    def __call__(self, img_feat, verts_3d, pred_cam, pred_mano_feats, focal_length):
        """
        Args:
            img_feat: (B, C, H, W) NCHW image features from ViT
            verts_3d: (B, V, 3) 3D vertices from temp MANO
            pred_cam: (B, 3) predicted camera params
            pred_mano_feats: dict with hand_pose, betas, cam
            focal_length: (B, 2) focal lengths
        Returns:
            pred_mano_params: dict with refined params
        """
        B = img_feat.shape[0]

        img_feats = self.deconv(img_feat)  # list of NCHW tensors, high→low res
        img_feat_sizes = [f.shape[2] for f in img_feats]

        # Project vertices to 2D at each feature map scale
        temp_cams = []
        for size in img_feat_sizes:
            cam = mx.stack([
                pred_cam[:, 1],
                pred_cam[:, 2],
                2 * focal_length[:, 0] / (size * pred_cam[:, 0] + 1e-9)
            ], axis=-1)
            temp_cams.append(cam)

        verts_2d = [
            perspective_projection(
                verts_3d,
                translation=temp_cams[i],
                focal_length=focal_length / img_feat_sizes[i]
            )
            for i in range(len(img_feat_sizes))
        ]

        # Sample features at projected vertex locations, take max across vertices
        vert_feats = []
        for i in range(len(img_feat_sizes)):
            sampled = sample_joint_features(img_feats[i], verts_2d[i])  # (B, V, C)
            vert_feats.append(mx.max(sampled, axis=1))  # (B, C)

        vert_feats = mx.concatenate(vert_feats, axis=-1)  # (B, out_dim)

        # Predict deltas
        delta_pose = self.dec_pose(vert_feats)
        delta_betas = self.dec_shape(vert_feats)
        delta_cam = self.dec_cam(vert_feats)

        pred_hand_pose = pred_mano_feats['hand_pose'] + delta_pose
        pred_betas = pred_mano_feats['betas'] + delta_betas
        pred_cam_out = pred_mano_feats['cam'] + delta_cam

        pred_hand_pose_rot = rot6d_to_rotmat(pred_hand_pose).reshape(B, -1, 3, 3)

        pred_mano_params = {
            'global_orient': pred_hand_pose_rot[:, :1],
            'hand_pose': pred_hand_pose_rot[:, 1:],
            'betas': pred_betas,
            'pred_cam': pred_cam_out,
        }

        return pred_mano_params
