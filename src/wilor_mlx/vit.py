"""WiLoR ViT backbone ported to MLX.

Architecture: ViT-H/16 variant with 1280 embed dim, 32 layers, 16 heads.
Input: (B, 3, 256, 192) image → crops to (B, 3, 256, 192) → patches 16x12 = 192 tokens.
Adds 16 pose tokens + 1 shape token + 1 cam token = 210 total tokens.
Output: MANO params (global_orient, hand_pose as 3x3 rotmats), pred_cam, pred_mano_feats, img_feat.
"""

import math
import mlx.core as mx
import mlx.nn as nn


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Args:
        x: (..., 6) batch of 6D rotation representations.
    Returns:
        (..., 3, 3) rotation matrices. Caller reshapes as needed.
    """
    # Flatten to (N, 6) for processing, restore batch dims after
    flat = x.reshape(-1, 2, 3)  # (N, 2, 3)
    # Transpose to (N, 3, 2) to match PyTorch's permute(0, 2, 1)
    flat = flat.transpose(0, 2, 1)
    a1 = flat[:, :, 0]  # (N, 3)
    a2 = flat[:, :, 1]  # (N, 3)
    b1 = a1 / (mx.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = mx.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (mx.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = mx.linalg.cross(b1, b2)
    return mx.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding using Conv2d."""

    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=1280):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_shape = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        # MLX Conv2d expects NHWC input
        # PyTorch: Conv2d(3, 1280, kernel_size=16, stride=16, padding=4)
        # The WiLoR config has ratio=1, so stride = patch_size // ratio = 16
        # padding = 4 + 2 * (ratio // 2 - 1) = 4 + 0 = 4
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, padding=4)

    def __call__(self, x):
        # x: (B, H, W, C) in MLX convention (NHWC)
        x = self.proj(x)  # (B, Hp, Wp, embed_dim)
        B, Hp, Wp, C = x.shape
        x = x.reshape(B, Hp * Wp, C)  # (B, num_patches, embed_dim)
        return x, (Hp, Wp)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(0, 3, 2, 1, 4)  # (B, heads, 3, N, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class WiLoRViT(nn.Module):
    """WiLoR ViT backbone.

    Produces initial MANO parameter estimates and image features
    for the RefineNet.
    """

    def __init__(
        self,
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_hand_joints=15,
        joint_rep_dim=6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hand_joints = num_hand_joints
        self.joint_rep_dim = joint_rep_dim
        self.npose = joint_rep_dim * (num_hand_joints + 1)

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional embedding: (1, num_patches + 1, embed_dim) — includes cls token slot
        self.pos_embed = mx.zeros((1, num_patches + 1, embed_dim))

        # Token embeddings for pose/shape/cam
        self.pose_emb = nn.Linear(joint_rep_dim, embed_dim)
        self.shape_emb = nn.Linear(10, embed_dim)
        self.cam_emb = nn.Linear(3, embed_dim)

        # Decode heads
        self.decpose = nn.Linear(embed_dim, 6)
        self.decshape = nn.Linear(embed_dim, 10)
        self.deccam = nn.Linear(embed_dim, 3)

        # Transformer blocks
        self.blocks = [Block(embed_dim, num_heads, mlp_ratio, qkv_bias) for _ in range(depth)]
        self.last_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Mean params (loaded from checkpoint)
        self.init_hand_pose = mx.zeros((1, self.npose))
        self.init_betas = mx.zeros((1, 10))
        self.init_cam = mx.zeros((1, 3))

    def __call__(self, x):
        """
        Args:
            x: (B, H, W, C) image tensor in NHWC format, already preprocessed
               (normalized, cropped to 256x192)
        Returns:
            pred_mano_params: dict with global_orient (B,1,3,3), hand_pose (B,15,3,3), betas (B,10)
            pred_cam: (B, 3)
            pred_mano_feats: dict with hand_pose (B,96), betas (B,10), cam (B,3)
            img_feat: (B, embed_dim, Hp, Wp) — NOTE: NCHW for RefineNet compatibility
        """
        B = x.shape[0]

        # Patch embedding
        x, (Hp, Wp) = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional embedding
        # WiLoR: x = x + pos_embed[:, 1:] + pos_embed[:, :1]
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        # Create and concat pose/shape/cam tokens
        pose_tokens = self.pose_emb(
            mx.broadcast_to(
                self.init_hand_pose.reshape(1, self.num_hand_joints + 1, self.joint_rep_dim),
                (B, self.num_hand_joints + 1, self.joint_rep_dim)
            )
        )  # (B, 16, embed_dim)
        shape_tokens = mx.broadcast_to(
            self.shape_emb(self.init_betas)[:, None, :],
            (B, 1, self.embed_dim)
        )  # (B, 1, embed_dim)
        cam_tokens = mx.broadcast_to(
            self.cam_emb(self.init_cam)[:, None, :],
            (B, 1, self.embed_dim)
        )  # (B, 1, embed_dim)

        x = mx.concatenate([pose_tokens, shape_tokens, cam_tokens, x], axis=1)
        # x: (B, 210, embed_dim)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.last_norm(x)

        # Decode
        n_pose = self.num_hand_joints + 1  # 16
        pose_feat = x[:, :n_pose]           # (B, 16, embed_dim)
        shape_feat = x[:, n_pose:n_pose+1]  # (B, 1, embed_dim)
        cam_feat = x[:, n_pose+1:n_pose+2]  # (B, 1, embed_dim)

        pred_hand_pose = self.decpose(pose_feat).reshape(B, -1) + self.init_hand_pose  # (B, 96)
        pred_betas = self.decshape(shape_feat).reshape(B, -1) + self.init_betas         # (B, 10)
        pred_cam = self.deccam(cam_feat).reshape(B, -1) + self.init_cam                 # (B, 3)

        pred_mano_feats = {
            'hand_pose': pred_hand_pose,
            'betas': pred_betas,
            'cam': pred_cam,
        }

        # rot6d → rotation matrices: (B, 96) → reshape to N*6 → (N, 3, 3) → (B, 16, 3, 3)
        pred_hand_pose_rot = rot6d_to_rotmat(pred_hand_pose).reshape(B, n_pose, 3, 3)
        pred_mano_params = {
            'global_orient': pred_hand_pose_rot[:, :1],   # (B, 1, 3, 3)
            'hand_pose': pred_hand_pose_rot[:, 1:],       # (B, 15, 3, 3)
            'betas': pred_betas,                          # (B, 10)
        }

        # Image features for RefineNet — reshape back to spatial
        img_feat = x[:, n_pose+2:]  # (B, 192, embed_dim)
        img_feat = img_feat.reshape(B, Hp, Wp, -1)  # (B, Hp, Wp, embed_dim) — NHWC
        img_feat = img_feat.transpose(0, 3, 1, 2)   # (B, embed_dim, Hp, Wp) — NCHW for RefineNet

        return pred_mano_params, pred_cam, pred_mano_feats, img_feat
