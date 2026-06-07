# wilor-mlx

WiLoR hand pose estimation running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). **4x faster than PyTorch MPS.**

This is a from-scratch MLX port of [WiLoR-mini](https://github.com/abcbdf/WiLoR-mini) — the complete pipeline including ViT backbone, MANO hand model, and RefineNet refinement stage. No PyTorch dependency at inference time.

## Performance

Tested on M4 Max, single-image inference, float32:

| Backend | Full pipeline | FPS |
|---|---|---|
| PyTorch MPS (2.5.0) | 208 ms | ~5 |
| **MLX (wilor-mlx)** | **50 ms** | **~20** |

The entire MLX pipeline runs faster than just the PyTorch ViT backbone alone (55 ms).

## Why so fast?

The speedup comes from MLX's unified memory architecture eliminating CPU↔GPU transfer overhead. PyTorch MPS pays ~100ms in pipeline sync and memory transfers between stages. MLX's lazy evaluation builds one computation graph and executes it without transfers.

## Quick start

```python
from wilor_mlx import WiLoR

model = WiLoR.from_pytorch_checkpoint(
    "path/to/wilor_final.ckpt",
    "path/to/MANO_RIGHT.pkl",
    "path/to/mano_mean_params.npz",
)

# Input: (B, 256, 256, 3) uint8 NHWC RGB image
result = model(image)

# Outputs:
# result['pred_keypoints_3d']  — (B, 21, 3) 3D hand keypoints
# result['pred_vertices']      — (B, 778, 3) MANO mesh vertices
# result['pred_cam']           — (B, 3) weak-perspective camera
# result['global_orient']      — (B, 1, 3) axis-angle global rotation
# result['hand_pose']          — (B, 15, 3) axis-angle finger poses
# result['betas']              — (B, 10) MANO shape parameters
```

## Install

```bash
pip install wilor-mlx
```

Or from source:

```bash
git clone https://github.com/lyonsno/wilor-mlx
cd wilor-mlx
pip install -e .
```

Requires: macOS with Apple Silicon, Python 3.10+, MLX 0.22+.

## Numerical accuracy

Compared against PyTorch WiLoR-mini on identical inputs (float32):

| Output | Max absolute diff |
|---|---|
| pred_vertices (778×3) | 0.024 |
| pred_keypoints_3d (21×3) | 0.024 |

Geometric outputs match within ~1mm — well within visual tolerance for hand tracking.

## What's ported

- ViT-H/16 backbone (1280 dim, 32 layers, 16 heads)
- MANO hand model with LBS (Linear Blend Skinning)
- RefineNet with deconvolution pyramid and grid sampling
- rot6d ↔ rotation matrix ↔ axis-angle conversions
- Full weight conversion from PyTorch checkpoints

## License

MIT. The MANO model has its own license — see [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/).
