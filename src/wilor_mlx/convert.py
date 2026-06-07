"""Convert PyTorch WiLoR checkpoint to MLX model weights.

Handles:
- Linear weights: same layout (out×in) in both PyTorch and MLX
- Conv2d weight transposition (PyTorch: OIHW, MLX: OHWI)
- ConvTranspose2d weight transposition (PyTorch: IOHW, MLX: OHWI)
- BatchNorm param mapping
- MANO buffer loading
"""

import pickle
import numpy as np
import mlx.core as mx


def _t(x):
    """Convert numpy/torch tensor to MLX array."""
    if hasattr(x, 'numpy'):
        x = x.numpy()
    return mx.array(np.array(x, dtype=np.float32))


def _t_int(x):
    """Convert to MLX int32 array."""
    if hasattr(x, 'numpy'):
        x = x.numpy()
    return mx.array(np.array(x, dtype=np.int32))


def load_pytorch_checkpoint(model, ckpt_path, mano_model_path, mano_mean_path):
    """Load PyTorch checkpoint into MLX WiLoR model.

    Args:
        model: WiLoR MLX model instance
        ckpt_path: path to wilor_final.ckpt
        mano_model_path: path to MANO_RIGHT.pkl
        mano_mean_path: path to mano_mean_params.npz
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt if not isinstance(ckpt, dict) or 'state_dict' not in ckpt else ckpt['state_dict']

    # --- Backbone (ViT) ---
    bb = model.backbone

    # Buffers
    bb.pos_embed = _t(sd['backbone.pos_embed'])
    bb.init_cam = _t(sd['backbone.init_cam'])
    bb.init_hand_pose = _t(sd['backbone.init_hand_pose'])
    bb.init_betas = _t(sd['backbone.init_betas'])

    # Patch embed conv: PyTorch (O, I, H, W) → MLX (O, H, W, I)
    bb.patch_embed.proj.weight = _t(sd['backbone.patch_embed.proj.weight'].permute(0, 2, 3, 1))
    bb.patch_embed.proj.bias = _t(sd['backbone.patch_embed.proj.bias'])

    # Token embeddings (Linear: same layout in PyTorch and MLX)
    for name in ['pose_emb', 'shape_emb', 'cam_emb', 'decpose', 'decshape', 'deccam']:
        layer = getattr(bb, name)
        layer.weight = _t(sd[f'backbone.{name}.weight'])
        layer.bias = _t(sd[f'backbone.{name}.bias'])

    # Transformer blocks
    for i in range(32):
        blk = bb.blocks[i]
        prefix = f'backbone.blocks.{i}'

        # LayerNorms
        blk.norm1.weight = _t(sd[f'{prefix}.norm1.weight'])
        blk.norm1.bias = _t(sd[f'{prefix}.norm1.bias'])
        blk.norm2.weight = _t(sd[f'{prefix}.norm2.weight'])
        blk.norm2.bias = _t(sd[f'{prefix}.norm2.bias'])

        # Attention
        blk.attn.qkv.weight = _t(sd[f'{prefix}.attn.qkv.weight'])
        blk.attn.qkv.bias = _t(sd[f'{prefix}.attn.qkv.bias'])
        blk.attn.proj.weight = _t(sd[f'{prefix}.attn.proj.weight'])
        blk.attn.proj.bias = _t(sd[f'{prefix}.attn.proj.bias'])

        # MLP
        blk.mlp.fc1.weight = _t(sd[f'{prefix}.mlp.fc1.weight'])
        blk.mlp.fc1.bias = _t(sd[f'{prefix}.mlp.fc1.bias'])
        blk.mlp.fc2.weight = _t(sd[f'{prefix}.mlp.fc2.weight'])
        blk.mlp.fc2.bias = _t(sd[f'{prefix}.mlp.fc2.bias'])

    # Last norm
    bb.last_norm.weight = _t(sd['backbone.last_norm.weight'])
    bb.last_norm.bias = _t(sd['backbone.last_norm.bias'])

    # --- RefineNet ---
    rn = model.refine_net

    # first_conv: Conv2d (O, I, 1, 1) → MLX (O, 1, 1, I)
    rn.deconv.first_conv.weight = _t(sd['refine_net.deconv.first_conv.0.weight'].permute(0, 2, 3, 1))
    rn.deconv.first_conv.bias = _t(sd['refine_net.deconv.first_conv.0.bias'])

    # Deconv branches — hardcoded to match checkpoint structure
    dn = rn.deconv

    def load_conv_transpose(layer, key):
        # PyTorch ConvTranspose2d: (I, O, H, W) → MLX: (O, H, W, I)
        layer.weight = _t(sd[f'{key}.weight'].permute(1, 2, 3, 0))

    def load_bn(layer, key):
        layer.weight = _t(sd[f'{key}.weight'])
        layer.bias = _t(sd[f'{key}.bias'])
        layer.running_mean = _t(sd[f'{key}.running_mean'])
        layer.running_var = _t(sd[f'{key}.running_var'])

    # Branch 0: ConvTranspose2d(640→320) + BN(320)
    load_conv_transpose(dn.branch0_conv, 'refine_net.deconv.deconv.0.0')
    load_bn(dn.branch0_bn, 'refine_net.deconv.deconv.0.1')

    # Branch 1: ConvTranspose2d(640→320) + BN(320) + ConvTranspose2d(320→160) + BN(160)
    load_conv_transpose(dn.branch1_conv0, 'refine_net.deconv.deconv.1.0')
    load_bn(dn.branch1_bn0, 'refine_net.deconv.deconv.1.1')
    load_conv_transpose(dn.branch1_conv1, 'refine_net.deconv.deconv.1.3')
    load_bn(dn.branch1_bn1, 'refine_net.deconv.deconv.1.4')

    # RefineNet decode heads
    for name in ['dec_pose', 'dec_cam', 'dec_shape']:
        layer = getattr(rn, name)
        layer.weight = _t(sd[f'refine_net.{name}.weight'])
        layer.bias = _t(sd[f'refine_net.{name}.bias'])

    # --- MANO ---
    mano = model.mano
    mano.v_template = _t(sd['mano.v_template'])
    mano.shapedirs = _t(sd['mano.shapedirs'])
    mano.posedirs = _t(sd['mano.posedirs'])
    mano.J_regressor = _t(sd['mano.J_regressor'])
    mano.parents = _t_int(sd['mano.parents'])
    mano.lbs_weights = _t(sd['mano.lbs_weights'])
    mano.extra_joints_idxs = _t_int(sd['mano.extra_joints_idxs'])
    mano.joint_map = _t_int(sd['mano.joint_map'])

    # Load mean params
    mean_params = np.load(mano_mean_path)
    model.backbone.init_hand_pose = mx.array(mean_params['pose'].astype(np.float32)).reshape(1, -1)
    model.backbone.init_betas = mx.array(mean_params['shape'].astype(np.float32)).reshape(1, -1)
    model.backbone.init_cam = mx.array(mean_params['cam'].astype(np.float32)).reshape(1, -1)

    # Evaluate all params to materialize
    mx.eval(*_collect_arrays(model))

    print(f"Loaded {len(sd)} parameters from checkpoint")


def _collect_arrays(obj, prefix=""):
    """Recursively collect all mx.array attributes."""
    arrays = []
    if isinstance(obj, mx.array):
        return [obj]
    if isinstance(obj, (list, tuple)):
        for item in obj:
            arrays.extend(_collect_arrays(item))
    elif hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            arrays.extend(_collect_arrays(v, f"{prefix}.{k}"))
    return arrays
