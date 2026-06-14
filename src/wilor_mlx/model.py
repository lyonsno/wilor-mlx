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

    Limitation: inaccurate near angle=pi (180°) where both sin(angle)
    and the skew-symmetric part approach zero. Roma avoids this via
    quaternion intermediates. For WiLoR hand pose output, joint angles
    are well below 180° so this does not affect inference quality.

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
        model = WiLoR.from_pretrained()
        pred = model(image)  # image: (B, 256, 256, 3) uint8 RGB NHWC
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
            dict with pred_keypoints_3d, pred_vertices, global_orient,
            hand_pose, betas, pred_cam as MLX arrays (batched).
            faces: (1538, 3) int32 triangle indices (unbatched, static topology)
            — present only when MANO faces are loaded.
        """
        # Normalize (ImageNet mean/std, RGB order)
        x = x.astype(mx.float32) / 255.0
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
        if self.mano.faces is not None:
            pred_mano_params['faces'] = self.mano.faces
        pred_mano_params['global_orient'] = rotmat_to_rotvec(pred_mano_params['global_orient'])
        pred_mano_params['hand_pose'] = rotmat_to_rotvec(pred_mano_params['hand_pose'])

        return pred_mano_params

    HF_REPO_ID = "BasinShapers/wilor-mlx"
    HF_WEIGHTS_FILE = "wilor-mlx.safetensors"
    WILOR_MINI_REPO_ID = "warmshao/WiLoR-mini"
    MANO_CACHE_FILE = "mano.npz"

    @staticmethod
    def from_pretrained(weights_path=None, mano_path=None):
        """Load a WiLoR model with automatic weight downloading.

        On first call, downloads model weights from HuggingFace and converts
        MANO data from the WiLoR-mini checkpoint. Everything is cached locally
        for instant loading on subsequent calls.

        First run requires torch (for one-time MANO conversion from PyTorch
        checkpoint). After that, torch is never used again.

        Args:
            weights_path: path to .safetensors file, or None to auto-download
            mano_path: path to mano.npz or MANO_RIGHT.pkl, or None to
                       auto-download and convert from WiLoR-mini checkpoint
        Returns:
            WiLoR model with loaded weights
        """
        from wilor_mlx.convert import load_safetensors_weights, load_mano_npz

        if weights_path is None:
            weights_path = WiLoR._download_weights()

        if mano_path is None:
            mano_path = WiLoR._ensure_mano()

        model = WiLoR()
        load_safetensors_weights(model, weights_path)
        if mano_path.endswith('.npz'):
            load_mano_npz(model, mano_path)
        else:
            from wilor_mlx.convert import load_mano_from_pkl
            load_mano_from_pkl(model.mano, mano_path)
        return model

    @staticmethod
    def _download_weights():
        """Download MLX model weights from HuggingFace."""
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=WiLoR.HF_REPO_ID,
            filename=WiLoR.HF_WEIGHTS_FILE,
        )

    @staticmethod
    def _ensure_mano():
        """Ensure MANO data is available, converting from WiLoR-mini checkpoint if needed."""
        import os

        # Check for cached mano.npz next to the weights
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "wilor-mlx"
        )
        mano_npz_path = os.path.join(cache_dir, WiLoR.MANO_CACHE_FILE)

        if os.path.exists(mano_npz_path):
            # Regenerate if cached file is missing faces (pre-v0.3.0)
            data = np.load(mano_npz_path)
            if 'faces' in data:
                return mano_npz_path
            print("Cached mano.npz is missing faces, regenerating...")

        # Need to convert — download WiLoR-mini files and extract MANO
        print("First run: converting MANO data from WiLoR-mini checkpoint (requires torch)...")

        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for one-time MANO conversion on first run.\n"
                "Install it with: pip install torch\n"
                "After the first run, torch is no longer needed."
            )

        from huggingface_hub import hf_hub_download
        import numpy as np

        # Download from WiLoR-mini's HuggingFace repo
        ckpt_path = hf_hub_download(
            repo_id=WiLoR.WILOR_MINI_REPO_ID,
            subfolder="pretrained_models",
            filename="wilor_final.ckpt",
        )
        mean_path = hf_hub_download(
            repo_id=WiLoR.WILOR_MINI_REPO_ID,
            subfolder="pretrained_models",
            filename="mano_mean_params.npz",
        )

        # Extract MANO arrays from PyTorch checkpoint (plain tensors, no chumpy)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        mano_arrays = {
            "v_template": sd["mano.v_template"].numpy().astype(np.float32),
            "shapedirs": sd["mano.shapedirs"].numpy().astype(np.float32),
            "posedirs": sd["mano.posedirs"].numpy().astype(np.float32),
            "J_regressor": sd["mano.J_regressor"].numpy().astype(np.float32),
            "parents": sd["mano.parents"].numpy().astype(np.int32),
            "lbs_weights": sd["mano.lbs_weights"].numpy().astype(np.float32),
            "extra_joints_idxs": sd["mano.extra_joints_idxs"].numpy().astype(np.int32),
            "joint_map": sd["mano.joint_map"].numpy().astype(np.int32),
            "faces": sd["mano.faces_tensor"].numpy().astype(np.int32),
        }

        # Save init params too
        mean_params = np.load(mean_path)
        mano_arrays["init_hand_pose"] = mean_params["pose"].astype(np.float32)
        mano_arrays["init_betas"] = mean_params["shape"].astype(np.float32)
        mano_arrays["init_cam"] = mean_params["cam"].astype(np.float32)

        os.makedirs(cache_dir, exist_ok=True)
        np.savez(mano_npz_path, **mano_arrays)
        print(f"MANO data cached at {mano_npz_path}")

        return mano_npz_path

    @staticmethod
    def from_pytorch_checkpoint(ckpt_path, mano_model_path, mano_mean_path):
        """Load a WiLoR model from a PyTorch checkpoint.

        Requires torch to be installed (one-time weight conversion).

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
