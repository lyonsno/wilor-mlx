# wilor-mlx

WiLoR hand pose estimation running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). **4x faster than PyTorch MPS.**

This is a from-scratch MLX port of [WiLoR-mini](https://github.com/abcbdf/WiLoR-mini) (Zhan et al., "WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild") — the complete inference pipeline including ViT backbone, MANO hand model, and RefineNet refinement stage. No PyTorch dependency at inference time.

## Performance

Tested on M4 Max, single-image inference, float32:

| Backend | Full pipeline | FPS |
|---|---|---|
| PyTorch MPS (2.5.0) | 208 ms | ~5 |
| **MLX (wilor-mlx)** | **50 ms** | **~20** |

The entire MLX pipeline runs faster than just the PyTorch MPS ViT backbone alone (55 ms).

### Why so fast?

The speedup comes from MLX's unified memory architecture eliminating CPU↔GPU transfer overhead. PyTorch MPS pays ~100ms in pipeline sync and memory transfers between stages. MLX's lazy evaluation builds one computation graph and executes it without round-trips.

## Install

```bash
git clone https://github.com/lyonsno/wilor-mlx
cd wilor-mlx
pip install -e .

# Weight conversion requires torch (one-time only, not needed for inference)
pip install torch
```

Requires macOS with Apple Silicon, Python 3.10+. MLX is installed automatically.

## Getting the model weights

There are two options:

### Option A: Pre-converted weights (recommended, no torch needed)

Convert the weights once, then load without any PyTorch dependency:

```bash
# One-time conversion (requires torch)
pip install torch
python -m wilor_mlx.convert \
    pretrained_models/wilor_final.ckpt \
    pretrained_models/MANO_RIGHT.pkl \
    pretrained_models/mano_mean_params.npz \
    weights/wilor-mlx.safetensors
```

### Option B: Load from PyTorch checkpoint directly

If you already have the WiLoR-mini pretrained models, you can load them directly (requires `torch`).

### Getting the original WiLoR-mini files

Both options need the original model files. Clone [WiLoR-mini](https://github.com/abcbdf/WiLoR-mini) and follow its setup instructions. You need:

1. **`wilor_final.ckpt`** — the WiLoR model checkpoint
2. **`MANO_RIGHT.pkl`** — the MANO hand model (requires registration at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/))
3. **`mano_mean_params.npz`** — mean MANO parameters

The `detector.pt` file from WiLoR-mini is not needed by wilor-mlx.

## Quick start

```python
from wilor_mlx import WiLoR
import mlx.core as mx
import numpy as np

# Option A: Load from pre-converted weights (no torch needed)
model = WiLoR.from_pretrained("weights/wilor-mlx.safetensors")

# Option B: Load from PyTorch checkpoint (requires torch)
# model = WiLoR.from_pytorch_checkpoint(
#     "pretrained_models/wilor_final.ckpt",
#     "pretrained_models/MANO_RIGHT.pkl",
#     "pretrained_models/mano_mean_params.npz",
# )

# Prepare input: a 256x256 RGB hand crop as uint8
# WiLoR expects a tightly cropped hand image, typically from a hand detector
image = np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8)  # replace with real image
image_mlx = mx.array(image)

# Run inference
result = model(image_mlx)
mx.eval(result)

# Outputs
keypoints_3d = np.array(result['pred_keypoints_3d'])  # (1, 21, 3) — 21 hand keypoints in 3D
vertices = np.array(result['pred_vertices'])            # (1, 778, 3) — MANO mesh vertices
camera = np.array(result['pred_cam'])                   # (1, 3) — weak-perspective camera [s, tx, ty]
```

### Input format

The model expects a **256×256 RGB crop of a hand**, as a `(B, 256, 256, 3)` uint8 MLX array in NHWC layout. The model handles normalization internally (ImageNet mean/std). In a typical pipeline, a hand detector (like YOLO) first finds the hand bounding box in a full frame, then the crop is resized to 256×256 and passed to WiLoR for 3D pose estimation.

### Output format

| Key | Shape | Description |
|---|---|---|
| `pred_keypoints_3d` | (B, 21, 3) | 3D hand joint locations (OpenPose ordering) |
| `pred_vertices` | (B, 778, 3) | MANO mesh vertex positions |
| `pred_cam` | (B, 3) | Weak-perspective camera `[scale, tx, ty]` |
| `global_orient` | (B, 1, 3) | Global wrist rotation (axis-angle) |
| `hand_pose` | (B, 15, 3) | Per-finger joint rotations (axis-angle) |
| `betas` | (B, 10) | MANO shape parameters |

## Numerical accuracy

Compared against PyTorch WiLoR-mini on identical inputs (float32):

| Output | Max abs diff | Notes |
|---|---|---|
| pred_vertices (778×3) | 0.024 | < 1mm — within visual tolerance |
| pred_keypoints_3d (21×3) | 0.024 | < 1mm |
| hand_pose (15×3) | 0.13 | Axis-angle is sensitive near gimbal lock |
| betas (10) | 0.23 | Accumulates through 32 transformer layers |

The geometric outputs that matter for hand tracking (vertices, keypoints) match within ~1mm.

## Architecture

The port includes:

- **ViT-H/16 backbone** — 1280 embed dim, 32 transformer layers, 16 heads. Processes 192 image patches + 18 learnable tokens (pose/shape/camera).
- **MANO hand model** — differentiable hand mesh with Linear Blend Skinning, Rodrigues rotations, and kinematic chain. 778 vertices, 16 joints.
- **RefineNet** — multi-scale deconvolution pyramid that samples ViT features at projected vertex locations via bilinear grid sampling, then refines the initial MANO parameter estimates.
- **Weight converter** — loads PyTorch `.ckpt` files, handles Conv2d NCHW→NHWC transposition, ConvTranspose2d weight layout, and BatchNorm parameter mapping.

## Note on weight conversion

`from_pretrained` loads pre-converted `.safetensors` weights with no PyTorch dependency. `from_pytorch_checkpoint` requires `torch` for one-time conversion. Use the `python -m wilor_mlx.convert` CLI to convert and save weights for future torch-free loading.

## License

MIT. The MANO hand model has its own license — see [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/).
