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


def load_mano_from_pkl(mano, pkl_path):
    """Load MANO model buffers from the official MANO_RIGHT.pkl file.

    MANO data is not included in wilor-mlx weights due to its
    non-redistributable license. Users must obtain MANO_RIGHT.pkl
    separately by registering at https://mano.is.tue.mpg.de/

    Args:
        mano: MANO model instance
        pkl_path: path to MANO_RIGHT.pkl
    """
    import pickle
    import sys

    # The MANO pkl references chumpy (MPI autodiff lib). We can't install it
    # (build fails on Python 3.14) and can't mock it cleanly (custom __reduce__).
    # Solution: use smplx's loader which handles the chumpy deserialization,
    # OR fall back to loading via the WiLoR checkpoint which already has the
    # arrays in plain numpy form.
    # Simplest: try loading, catch chumpy errors, advise user.
    try:
        with open(pkl_path, 'rb') as f:
            mano_data = pickle.load(f, encoding='latin1')
    except (ModuleNotFoundError, TypeError) as e:
        if 'chumpy' in str(e):
            raise RuntimeError(
                f"Failed to load MANO pkl: {e}\n"
                f"The MANO pkl file requires the 'chumpy' package to deserialize.\n"
                f"Install it with: pip install chumpy\n"
                f"Alternatively, use WiLoR.from_pytorch_checkpoint() which loads\n"
                f"MANO data from the PyTorch checkpoint (already in numpy format)."
            ) from e
        raise

    def to_np(v):
        if isinstance(v, np.ndarray):
            return v
        if hasattr(v, 'r'):
            return np.array(v.r)
        return np.array(v)

    mano.v_template = mx.array(to_np(mano_data['v_template']).astype(np.float32))
    mano.shapedirs = mx.array(to_np(mano_data['shapedirs']).astype(np.float32))
    # posedirs in pkl is (778, 3, 135), WiLoR expects (135, 2334) = (135, 778*3)
    posedirs_raw = to_np(mano_data['posedirs']).astype(np.float32)  # (778, 3, 135)
    mano.posedirs = mx.array(posedirs_raw.reshape(-1, posedirs_raw.shape[-1]).T)  # (135, 2334)
    J_reg = mano_data['J_regressor']
    mano.J_regressor = mx.array(np.array(J_reg.toarray() if hasattr(J_reg, 'toarray') else J_reg, dtype=np.float32))
    mano.parents = mx.array(mano_data['kintree_table'][0].astype(np.int32))
    mano.lbs_weights = mx.array(to_np(mano_data['weights']).astype(np.float32))

    # Extra joint vertex indices (from smplx.vertex_ids['mano'])
    mano.extra_joints_idxs = mx.array(np.array([744, 320, 443, 554, 671], dtype=np.int32))
    mano.joint_map = mx.array(np.array(
        [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
        dtype=np.int32
    ))

    arrays = [v for v in [mano.v_template, mano.shapedirs, mano.posedirs,
              mano.J_regressor, mano.parents, mano.lbs_weights,
              mano.extra_joints_idxs, mano.joint_map] if isinstance(v, mx.array)]
    mx.eval(*arrays)


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

    # MANO data is NOT in the safetensors file — loaded separately via load_mano_from_pkl()

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

    # MANO data is NOT saved — non-redistributable MPI license

    # Model-level
    all_weights['IMAGE_MEAN'] = model.IMAGE_MEAN
    all_weights['IMAGE_STD'] = model.IMAGE_STD

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    mx.save_safetensors(output_path, all_weights)
    total_mb = sum(v.nbytes for v in all_weights.values()) / 1024 / 1024
    print(f"Saved {len(all_weights)} arrays ({total_mb:.0f} MB) to {output_path}")


def save_mano_npz(model, output_path):
    """Save MANO buffers as a separate .npz file for MANO-license-safe loading.

    Users generate this from their own MANO_RIGHT.pkl obtained after MPI registration.
    This file stays local and is never redistributed.
    """
    import os
    mano = model.mano
    mano_arrays = {
        'v_template': np.array(mano.v_template),
        'shapedirs': np.array(mano.shapedirs),
        'posedirs': np.array(mano.posedirs),
        'J_regressor': np.array(mano.J_regressor),
        'parents': np.array(mano.parents),
        'lbs_weights': np.array(mano.lbs_weights),
        'extra_joints_idxs': np.array(mano.extra_joints_idxs),
        'joint_map': np.array(mano.joint_map),
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez(output_path, **mano_arrays)
    total_kb = sum(v.nbytes for v in mano_arrays.values()) / 1024
    print(f"Saved MANO buffers ({total_kb:.0f} KB) to {output_path}")


def load_mano_npz(mano, npz_path):
    """Load MANO buffers from a locally-generated .npz file."""
    data = np.load(npz_path)
    mano.v_template = mx.array(data['v_template'])
    mano.shapedirs = mx.array(data['shapedirs'])
    mano.posedirs = mx.array(data['posedirs'])
    mano.J_regressor = mx.array(data['J_regressor'])
    mano.parents = mx.array(data['parents'].astype(np.int32))
    mano.lbs_weights = mx.array(data['lbs_weights'])
    mano.extra_joints_idxs = mx.array(data['extra_joints_idxs'].astype(np.int32))
    mano.joint_map = mx.array(data['joint_map'].astype(np.int32))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python -m wilor_mlx.convert <ckpt> <mano_pkl> <mean_params> <output_dir>")
        print("  Produces: <output_dir>/wilor-mlx.safetensors + <output_dir>/mano.npz")
        sys.exit(1)

    import os
    from wilor_mlx.model import WiLoR
    model = WiLoR()
    load_pytorch_checkpoint(model, sys.argv[1], sys.argv[2], sys.argv[3])

    out_dir = sys.argv[4]
    save_mlx_weights(model, os.path.join(out_dir, "wilor-mlx.safetensors"))
    save_mano_npz(model, os.path.join(out_dir, "mano.npz"))
