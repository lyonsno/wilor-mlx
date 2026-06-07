"""Full WiLoR hand pose estimation model in MLX.

Combines ViT backbone, MANO hand model, and RefineNet.
Drop-in replacement for wilor_mini.models.wilor.WiLor.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from wilor_mlx.vit import WiLoRViT
from wilor_mlx.mano import MANO
from wilor_mlx.refinenet import RefineNet


def rotmat_to_rotvec(rotmat):
    """Convert rotation matrices to axis-angle (Rodrigues) vectors.
    Simplified version of roma.rotmat_to_rotvec.

    Args:
        rotmat: (..., 3, 3) rotation matrices
    Returns:
        (..., 3) axis-angle vectors
    """
    orig_shape = rotmat.shape[:-2]
    R = rotmat.reshape(-1, 3, 3)
    batch_size = R.shape[0]

    # Angle from trace: cos(angle) = (trace(R) - 1) / 2
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_angle = mx.clip((trace - 1) / 2, -1.0, 1.0)
    angle = mx.arccos(cos_angle)  # (N,)

    # Axis from skew-symmetric part: [R - R^T] / (2 * sin(angle))
    skew = R - R.transpose(0, 2, 1)  # (N, 3, 3)
    axis = mx.stack([
        skew[:, 2, 1],
        skew[:, 0, 2],
        skew[:, 1, 0],
    ], axis=1)  # (N, 3)

    sin_angle = mx.sin(angle)[:, None]
    # Avoid division by zero for small angles
    safe_sin = mx.where(mx.abs(sin_angle) < 1e-6, mx.ones_like(sin_angle), sin_angle)
    axis = axis / (2 * safe_sin)

    rotvec = axis * angle[:, None]
    return rotvec.reshape(*orig_shape, 3)


class WiLoR:
    """Full WiLoR model in MLX.

    Usage:
        model = WiLoR.from_pytorch_checkpoint(ckpt_path, mano_path)
        pred = model(image)  # image: (B, H, W, 3) uint8 NHWC
    """

    def __init__(self):
        self.backbone = WiLoRViT()
        self.refine_net = RefineNet(feat_dim=1280, upscale=3)
        self.mano = MANO()
        self.FOCAL_LENGTH = 5000
        self.IMAGE_SIZE = 256
        self.IMAGE_MEAN = mx.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        self.IMAGE_STD = mx.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)

    def __call__(self, x):
        """Run WiLoR inference.

        Args:
            x: (B, 256, 256, 3) uint8 image in NHWC RGB format
        Returns:
            dict with pred_keypoints_3d, pred_vertices, global_orient, hand_pose,
            betas, pred_cam — all as MLX arrays
        """
        # Preprocess: flip BGR→RGB if needed, normalize
        x = x[..., ::-1].astype(mx.float32) / 255.0
        x = (x - self.IMAGE_MEAN) / self.IMAGE_STD

        batch_size = x.shape[0]

        # Crop to (B, 256, 192, 3) — remove 32px from each side
        backbone_input = x[:, :, 32:-32, :]  # NHWC

        # ViT backbone
        temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.backbone(backbone_input)

        focal_length = mx.full((batch_size, 2), self.FOCAL_LENGTH)

        # Temp MANO forward
        temp_vertices, _ = self.mano(
            global_orient=temp_mano_params['global_orient'],
            hand_pose=temp_mano_params['hand_pose'],
            betas=temp_mano_params['betas'],
            pose2rot=False,
        )

        # RefineNet
        pred_mano_params = self.refine_net(
            vit_out, temp_vertices, pred_cam, pred_mano_feats, focal_length
        )

        # Final MANO forward
        pred_vertices, pred_keypoints_3d = self.mano(
            global_orient=pred_mano_params['global_orient'],
            hand_pose=pred_mano_params['hand_pose'],
            betas=pred_mano_params['betas'],
            pose2rot=False,
        )

        # Convert rotmats to rotvecs for output
        pred_mano_params['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        pred_mano_params['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_mano_params['global_orient'] = rotmat_to_rotvec(pred_mano_params['global_orient'])
        pred_mano_params['hand_pose'] = rotmat_to_rotvec(pred_mano_params['hand_pose'])

        return pred_mano_params

    @staticmethod
    def from_pytorch_checkpoint(ckpt_path, mano_model_path, mano_mean_path):
        """Load a WiLoR model from a PyTorch checkpoint.

        Args:
            ckpt_path: path to wilor_final.ckpt
            mano_model_path: path to MANO model file
            mano_mean_path: path to mano_mean_params.npz
        Returns:
            WiLoR model with loaded weights
        """
        from wilor_mlx.convert import load_pytorch_checkpoint
        model = WiLoR()
        load_pytorch_checkpoint(model, ckpt_path, mano_model_path, mano_mean_path)
        return model
