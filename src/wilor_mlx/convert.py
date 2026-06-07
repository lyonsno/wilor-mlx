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

    # Materialize in batches to avoid Metal shared event exhaustion
    arrays = _collect_arrays(model)
    batch_size = 64
    for i in range(0, len(arrays), batch_size):
        mx.eval(*arrays[i:i + batch_size])

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


def load_safetensors_weights(model, weights_path):
    """Load pre-converted MLX weights from a .safetensors file.

    No PyTorch dependency required.
    """
    weights = mx.load(weights_path)

    bb = model.backbone

    # Backbone buffers
    bb.pos_embed = weights['backbone.pos_embed']
    bb.init_cam = weights['backbone.init_cam']
    bb.init_hand_pose = weights['backbone.init_hand_pose']
    bb.init_betas = weights['backbone.init_betas']

    # Patch embed
    bb.patch_embed.proj.weight = weights['backbone.patch_embed.proj.weight']
    bb.patch_embed.proj.bias = weights['backbone.patch_embed.proj.bias']

    # Embedding linears
    for name in ['pose_emb', 'shape_emb', 'cam_emb', 'decpose', 'decshape', 'deccam']:
        layer = getattr(bb, name)
        layer.weight = weights[f'backbone.{name}.weight']
        layer.bias = weights[f'backbone.{name}.bias']

    # Transformer blocks
    for i in range(32):
        blk = bb.blocks[i]
        p = f'backbone.blocks.{i}'
        blk.norm1.weight = weights[f'{p}.norm1.weight']
        blk.norm1.bias = weights[f'{p}.norm1.bias']
        blk.attn.qkv.weight = weights[f'{p}.attn.qkv.weight']
        blk.attn.qkv.bias = weights[f'{p}.attn.qkv.bias']
        blk.attn.proj.weight = weights[f'{p}.attn.proj.weight']
        blk.attn.proj.bias = weights[f'{p}.attn.proj.bias']
        blk.norm2.weight = weights[f'{p}.norm2.weight']
        blk.norm2.bias = weights[f'{p}.norm2.bias']
        blk.mlp.fc1.weight = weights[f'{p}.mlp.fc1.weight']
        blk.mlp.fc1.bias = weights[f'{p}.mlp.fc1.bias']
        blk.mlp.fc2.weight = weights[f'{p}.mlp.fc2.weight']
        blk.mlp.fc2.bias = weights[f'{p}.mlp.fc2.bias']

    bb.last_norm.weight = weights['backbone.last_norm.weight']
    bb.last_norm.bias = weights['backbone.last_norm.bias']

    # RefineNet
    rn = model.refine_net
    dn = rn.deconv
    dn.first_conv.weight = weights['refine_net.deconv.first_conv.weight']
    dn.first_conv.bias = weights['refine_net.deconv.first_conv.bias']

    for name in ['branch0_conv', 'branch0_bn', 'branch1_conv0', 'branch1_bn0',
                 'branch1_conv1', 'branch1_bn1']:
        layer = getattr(dn, name)
        layer.weight = weights[f'refine_net.deconv.{name}.weight']
        bias_key = f'refine_net.deconv.{name}.bias'
        if bias_key in weights:
            layer.bias = weights[bias_key]
        rm_key = f'refine_net.deconv.{name}.running_mean'
        if rm_key in weights:
            layer.running_mean = weights[rm_key]
            layer.running_var = weights[f'refine_net.deconv.{name}.running_var']

    for name in ['dec_pose', 'dec_cam', 'dec_shape']:
        layer = getattr(rn, name)
        layer.weight = weights[f'refine_net.{name}.weight']
        layer.bias = weights[f'refine_net.{name}.bias']

    # MANO
    mano = model.mano
    mano.v_template = weights['mano.v_template']
    mano.shapedirs = weights['mano.shapedirs']
    mano.posedirs = weights['mano.posedirs']
    mano.J_regressor = weights['mano.J_regressor']
    mano.parents = weights['mano.parents'].astype(mx.int32)
    mano.lbs_weights = weights['mano.lbs_weights']
    mano.extra_joints_idxs = weights['mano.extra_joints_idxs'].astype(mx.int32)
    mano.joint_map = weights['mano.joint_map'].astype(mx.int32)

    # Model-level
    model.IMAGE_MEAN = weights['IMAGE_MEAN']
    model.IMAGE_STD = weights['IMAGE_STD']

    # Materialize arrays in small batches to avoid exhausting Metal shared events
    # under concurrent GPU pressure (TRELLIS, other MLX sessions, etc.)
    arrays = _collect_arrays(model)
    batch_size = 64
    for i in range(0, len(arrays), batch_size):
        mx.eval(*arrays[i:i + batch_size])
    print(f"Loaded {len(weights)} arrays from {weights_path}")


def save_mlx_weights(model, output_path):
    """Save model weights as safetensors for torch-free loading."""
    import os
    bb = model.backbone
    all_weights = {}

    # Patch embed
    all_weights['backbone.patch_embed.proj.weight'] = bb.patch_embed.proj.weight
    all_weights['backbone.patch_embed.proj.bias'] = bb.patch_embed.proj.bias

    # Embedding linears
    for name in ['pose_emb', 'shape_emb', 'cam_emb', 'decpose', 'decshape', 'deccam']:
        layer = getattr(bb, name)
        all_weights[f'backbone.{name}.weight'] = layer.weight
        all_weights[f'backbone.{name}.bias'] = layer.bias

    # Blocks
    for i in range(32):
        blk = bb.blocks[i]
        p = f'backbone.blocks.{i}'
        for sub in ['norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
                     'attn.proj.weight', 'attn.proj.bias', 'norm2.weight', 'norm2.bias',
                     'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias']:
            obj = blk
            for attr in sub.split('.'):
                obj = getattr(obj, attr)
            all_weights[f'{p}.{sub}'] = obj

    all_weights['backbone.last_norm.weight'] = bb.last_norm.weight
    all_weights['backbone.last_norm.bias'] = bb.last_norm.bias

    # Backbone buffers
    for attr in ['pos_embed', 'init_cam', 'init_hand_pose', 'init_betas']:
        all_weights[f'backbone.{attr}'] = getattr(bb, attr)

    # RefineNet
    rn = model.refine_net
    dn = rn.deconv
    all_weights['refine_net.deconv.first_conv.weight'] = dn.first_conv.weight
    all_weights['refine_net.deconv.first_conv.bias'] = dn.first_conv.bias
    for name in ['branch0_conv', 'branch0_bn', 'branch1_conv0', 'branch1_bn0',
                 'branch1_conv1', 'branch1_bn1']:
        layer = getattr(dn, name)
        all_weights[f'refine_net.deconv.{name}.weight'] = layer.weight
        if hasattr(layer, 'bias') and layer.bias is not None:
            all_weights[f'refine_net.deconv.{name}.bias'] = layer.bias
        if hasattr(layer, 'running_mean'):
            all_weights[f'refine_net.deconv.{name}.running_mean'] = layer.running_mean
            all_weights[f'refine_net.deconv.{name}.running_var'] = layer.running_var
    for name in ['dec_pose', 'dec_cam', 'dec_shape']:
        layer = getattr(rn, name)
        all_weights[f'refine_net.{name}.weight'] = layer.weight
        all_weights[f'refine_net.{name}.bias'] = layer.bias

    # MANO
    for attr in ['v_template', 'shapedirs', 'posedirs', 'J_regressor', 'lbs_weights',
                 'joint_map', 'extra_joints_idxs', 'parents']:
        all_weights[f'mano.{attr}'] = getattr(model.mano, attr)

    # Model-level
    all_weights['IMAGE_MEAN'] = model.IMAGE_MEAN
    all_weights['IMAGE_STD'] = model.IMAGE_STD

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    mx.save_safetensors(output_path, all_weights)
    total_mb = sum(v.nbytes for v in all_weights.values()) / 1024 / 1024
    print(f"Saved {len(all_weights)} arrays ({total_mb:.0f} MB) to {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python -m wilor_mlx.convert <ckpt> <mano_pkl> <mean_params> <output.safetensors>")
        sys.exit(1)

    from wilor_mlx.model import WiLoR
    model = WiLoR()
    load_pytorch_checkpoint(model, sys.argv[1], sys.argv[2], sys.argv[3])
    save_mlx_weights(model, sys.argv[4])
