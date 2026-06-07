# wilor-mlx

WiLoR hand pose estimation for Apple Silicon, rebuilt end-to-end in [MLX](https://github.com/ml-explore/mlx). No PyTorch dependency at inference time.

A from-scratch MLX port of [WiLoR-mini](https://github.com/abcbdf/WiLoR-mini) (Zhan et al., "WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild") — the complete inference pipeline including ViT backbone, MANO hand model, and RefineNet refinement stage.

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

Reproduce the benchmark: `python benchmarks/bench_wilor.py --backend mlx --weights weights/wilor-mlx.safetensors --mano-npz weights/mano.npz`

## Install

```bash
git clone https://github.com/lyonsno/wilor-mlx
cd wilor-mlx
pip install -e .
pip install torch  # needed once for first-run MANO conversion, not used after
```

Requires macOS with Apple Silicon, Python 3.10+. MLX and other dependencies install automatically.

## How it works

On the first call to `WiLoR.from_pretrained()`, wilor-mlx automatically:

1. Downloads model weights from [HuggingFace](https://huggingface.co/lyonsno/wilor-mlx) (2.4 GB, cached locally)
2. Downloads MANO hand model data from the [WiLoR-mini](https://huggingface.co/warmshao/WiLoR-mini) checkpoint (requires `torch` for one-time conversion)
3. Caches converted MANO data at `~/.cache/wilor-mlx/mano.npz`

After the first run, everything loads from cache and **torch is never used again.**

The MANO hand model is licensed separately by the Max Planck Institute. We do not redistribute MANO data — it is downloaded from the original WiLoR-mini source and converted locally on your machine. See [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) for MANO license terms.

Float32 and int4 weight variants are available on the [model card](https://huggingface.co/lyonsno/wilor-mlx). Both run at the same speed on Apple Silicon.

The original WiLoR-mini files (`wilor_final.ckpt`, `MANO_RIGHT.pkl`, `mano_mean_params.npz`) can be obtained by cloning [WiLoR-mini](https://github.com/abcbdf/WiLoR-mini) and following its setup instructions. The `detector.pt` file is not needed by wilor-mlx.

## Quick start

```python
from wilor_mlx import WiLoR
import mlx.core as mx
import numpy as np

# Load model — everything downloads and caches automatically
model = WiLoR.from_pretrained()

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
