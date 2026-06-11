# wilor-mlx

WiLoR hand pose estimation for Apple Silicon, rebuilt end-to-end in [MLX](https://github.com/ml-explore/mlx).

A from-scratch MLX port of [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) (Zhan et al., "WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild") — the pose/reconstruction model including ViT backbone, MANO hand model, and RefineNet refinement stage. It expects a cropped hand image from a separate detector. First run requires `torch` for a one-time MANO conversion; after that, inference runs purely on MLX.

## Performance

Tested on Apple M4 Max, single-image inference, float32.

### Live sidecar behavior (gesture UI prototype)

The strongest launch evidence is the route we actually use for interaction: camera frame → hand crop → WiLoR-mini pose/reconstruction sidecar → hand-pose event.

On a clean post-reboot M4 Max same-harness smoke over recent 160x120 saved frames from a gesture UI prototype, MLX runs the pose/reconstruction model stage at about 37ms median versus 49ms for PyTorch MPS, and the full saved-frame route at about 49ms versus 60ms. That is roughly a 1.3x model-stage advantage and a 1.2x full-route advantage on the fair comparison denominator we trust most right now.

Larger derived-frame stress tests widen both backends; MLX remained faster in those runs, but we treat those numbers as route/runtime stress evidence rather than the headline model benchmark.

Older app-level telemetry is what pushed us toward MLX in the first place, but clean reruns narrowed the comparison denominator enough that we are not using the old tail story as a fresh universal PyTorch-vs-MLX headline. The current public claim is narrower and stronger: WiLoR-mini now has a native MLX runtime on Apple Silicon, with live sidecar latency low enough to build interaction on.

### Why so consistent?

MLX's lazy evaluation builds a graph that can be evaluated in fewer, fused submissions. That reduces dispatch and synchronization surface area in this short-context ViT workload, which is where our traces suggest the PyTorch MPS tail was coming from. The result is tight, flat latency that makes 3D hand pose viable as a real-time control primitive.

### Local benchmark harness

The repository includes a local benchmark harness for route checks and local reproduction:

```bash
python benchmarks/bench_wilor.py --backend mlx --weights weights/wilor-mlx.safetensors --mano-npz weights/mano.npz
```

We are not using the old app-tail telemetry as the headline claim right now. The strongest current evidence is the same-harness saved-frame route above: it measures the path that actually matters for using hand pose as a real-time input primitive while keeping the PyTorch MPS comparison on the same denominator. Lower-bandwidth M2 Pro/Tahoe validation also shows MLX ahead on archived hand-positive frames, but recent macOS/Metal changes moved both backends enough that we are treating exact M2 Pro numbers as rebaseline work rather than launch headline copy.

## Install

```bash
pip install wilor-mlx
pip install torch  # needed once for first-run MANO conversion, not used after
```

Or from source:

```bash
git clone https://github.com/lyonsno/wilor-mlx
cd wilor-mlx
pip install -e .
```

Requires macOS with Apple Silicon, Python 3.10+. MLX and other dependencies install automatically.

## How it works

On the first call to `WiLoR.from_pretrained()`, wilor-mlx automatically:

1. Downloads model weights from [HuggingFace](https://huggingface.co/BasinShapers/wilor-mlx) (2.4 GB, cached locally)
2. Downloads MANO hand model data from the [WiLoR-mini](https://huggingface.co/warmshao/WiLoR-mini) checkpoint (requires `torch` for one-time conversion)
3. Caches converted MANO data at `~/.cache/wilor-mlx/mano.npz`

After the first run, everything loads from cache and **torch is never used again.**

The MANO hand model is licensed separately by the Max Planck Institute. We do not redistribute MANO data — it is downloaded from the original WiLoR-mini source and converted locally on your machine. See [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) for MANO license terms.

Float32 and int4 weight variants are available on the [model card](https://huggingface.co/BasinShapers/wilor-mlx). Both run at the same speed on Apple Silicon — at these sequence lengths (210 tokens) the model is compute-bound, not memory-bandwidth-bound, so smaller weights don't accelerate inference. Int4 is purely a deployment convenience (2.4 GB → 490 MB).

If you prefer to supply your own MANO data (e.g. obtained directly from [MPI](https://mano.is.tue.mpg.de/)), pass `mano_path=...` to `from_pretrained()`.

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
| pred_vertices (778×3) | 0.006 | Sub-millimeter |
| pred_keypoints_3d (21×3) | 0.006 | Sub-millimeter |
| hand_pose (15×3) | 0.06 | Axis-angle is sensitive near gimbal lock |
| betas (10) | 0.10 | Accumulates through 32 transformer layers |

The geometric outputs that matter for hand tracking (vertices, keypoints) match within sub-millimeter precision.

## Architecture

The port includes:

- **ViT-H/16 backbone** — 1280 embed dim, 32 transformer layers, 16 heads. Processes 192 image patches + 18 learnable tokens (pose/shape/camera).
- **MANO hand model** — differentiable hand mesh with Linear Blend Skinning, Rodrigues rotations, and kinematic chain. 778 vertices, 16 joints.
- **RefineNet** — multi-scale deconvolution pyramid that samples ViT features at projected vertex locations via bilinear grid sampling, then refines the initial MANO parameter estimates.
- **Weight converter** — loads PyTorch `.ckpt` files, handles Conv2d NCHW→NHWC transposition, ConvTranspose2d weight layout, and BatchNorm parameter mapping.

## License

The wilor-mlx code and distributed weight files are MIT licensed. Our weights (on [HuggingFace](https://huggingface.co/BasinShapers/wilor-mlx)) contain only ViT backbone, RefineNet, and learned embedding parameters — no MANO data is bundled or rehosted.

The [MANO hand model](https://mano.is.tue.mpg.de/) is separately licensed by the Max Planck Institute. `WiLoR.from_pretrained()` fetches upstream [WiLoR-mini](https://huggingface.co/warmshao/WiLoR-mini) assets and converts MANO data locally on your machine. If you prefer to obtain MANO directly from MPI, pass `mano_path=...` to use your own copy.
