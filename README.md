# wilor-mlx

WiLoR hand pose estimation running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). **4x faster than PyTorch MPS.**

This is a from-scratch MLX port of [WiLoR-mini](https://github.com/abcbdf/WiLoR-mini) (Zhan et al., "WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild") — the complete inference pipeline including ViT backbone, MANO hand model, and RefineNet refinement stage. No PyTorch dependency at inference time.

## Performance

Tested on Apple M4 Max, single-image inference, float32:

### Isolated benchmark (same input, same machine, back-to-back)

| Backend | p50 | p90 | min | FPS |
|---|---|---|---|---|
| **MLX (wilor-mlx)** | **36 ms** | **36 ms** | **36 ms** | **28** |
| PyTorch MPS (2.5.0) | 50 ms | 51 ms | 49 ms | 20 |

**1.4x faster** in isolated model-stage benchmarks.

### Live sidecar (embedded in [Perceptasia](https://github.com/lyonsno/perceptasia) hand tracking)

| Backend | Model p50 | Model p90 |
|---|---|---|
| **MLX (wilor-mlx)** | **61 ms** | **107 ms** |
| PyTorch MPS (2.5.0) | ~208 ms | — |

**3.4x faster** in the live integration context, where MLX's unified memory eliminates the CPU↔GPU transfer overhead that dominates PyTorch MPS pipeline latency.

### Why the difference?

The isolated benchmark measures pure model compute. In a real application, PyTorch MPS pays ~100ms in pipeline sync and memory transfers between CPU and GPU stages. MLX's lazy evaluation builds one computation graph and executes it on unified memory without round-trips — so the integration speedup is larger than the compute speedup.

Reproduce the benchmark: `python benchmarks/bench_wilor.py --backend mlx --weights weights/wilor-mlx.safetensors`

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

### Option A: Download pre-converted weights (recommended)

Pre-converted weights are on HuggingFace — **no PyTorch needed:**

| Variant | Size | Download |
|---|---|---|
| **float32** (recommended) | 2.4 GB | [wilor-mlx.safetensors](https://huggingface.co/lyonsno/wilor-mlx/resolve/main/wilor-mlx.safetensors) |
| **int4** (5x smaller) | 490 MB | [wilor-mlx-int4.safetensors](https://huggingface.co/lyonsno/wilor-mlx/resolve/main/wilor-mlx-int4.safetensors) |

```bash
# Download with hf CLI
hf download lyonsno/wilor-mlx wilor-mlx.safetensors --local-dir weights/

# Or int4 for faster download
hf download lyonsno/wilor-mlx wilor-mlx-int4.safetensors --local-dir weights/
```

Both variants run at the same speed on Apple Silicon (~36ms). Int4 has slightly lower precision (< 2mm vs < 1mm on hand keypoints). See [model card](https://huggingface.co/lyonsno/wilor-mlx) for full benchmarks.

### MANO hand model (required separately)

The MANO hand model data is **not included** in our weights due to its [non-redistributable license](https://mano.is.tue.mpg.de/license.html) from the Max Planck Institute. You must obtain `MANO_RIGHT.pkl` separately:

1. Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/)
2. Download the MANO model files
3. Place `MANO_RIGHT.pkl` in your project directory

### Option B: Convert from PyTorch checkpoint yourself

If you have the original [WiLoR-mini](https://github.com/abcbdf/WiLoR-mini) pretrained models:

```bash
pip install torch
python -m wilor_mlx.convert \
    pretrained_models/wilor_final.ckpt \
    pretrained_models/MANO_RIGHT.pkl \
    pretrained_models/mano_mean_params.npz \
    weights/wilor-mlx.safetensors
```

The original WiLoR-mini files (`wilor_final.ckpt`, `MANO_RIGHT.pkl`, `mano_mean_params.npz`) can be obtained by cloning [WiLoR-mini](https://github.com/abcbdf/WiLoR-mini) and following its setup instructions. The `detector.pt` file is not needed by wilor-mlx.

## Quick start

```python
from wilor_mlx import WiLoR
import mlx.core as mx
import numpy as np

# Option A: Load from pre-converted weights (no torch needed)
model = WiLoR.from_pretrained(
    "weights/wilor-mlx.safetensors",
    mano_path="MANO_RIGHT.pkl",  # from mano.is.tue.mpg.de
)

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

The wilor-mlx code and distributed weight files are MIT licensed. The weights contain only ViT backbone, RefineNet, and learned embedding parameters.

**The MANO hand model is licensed separately** by the Max Planck Institute under a [non-commercial, non-redistributable research license](https://mano.is.tue.mpg.de/license.html). MANO data is not included in our weights or code. Users must obtain `MANO_RIGHT.pkl` directly from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) after registration.
