# wilor-mlx

WiLoR hand pose estimation for Apple Silicon, rebuilt end-to-end in [MLX](https://github.com/ml-explore/mlx).

A from-scratch MLX port of [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) (Zhan et al., "WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild") — complete pipeline from full image to 3D hand pose. Includes both the YOLOv8m-pose hand detector and the WiLoR pose/reconstruction model (ViT backbone, MANO hand model, RefineNet). After first-run setup, the entire pipeline runs purely on MLX with no PyTorch dependency.

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

On the first call to `HandPosePipeline.from_pretrained()`, wilor-mlx automatically:

1. Downloads hand detector weights from [HuggingFace](https://huggingface.co/BasinShapers/wilor-mlx) (107 MB, cached locally)
2. Downloads WiLoR pose model weights (2.4 GB, cached locally)
3. Downloads MANO hand model data from the [WiLoR-mini](https://huggingface.co/warmshao/WiLoR-mini) checkpoint (requires `torch` for one-time conversion)
4. Caches converted MANO data at `~/.cache/wilor-mlx/mano.npz`

After the first run, everything loads from cache and **torch is never used again.**

The MANO hand model is licensed separately by the Max Planck Institute. We do not redistribute MANO data — it is downloaded from the original WiLoR-mini source and converted locally on your machine. See [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) for MANO license terms.

Float32 and int4 weight variants are available on the [model card](https://huggingface.co/BasinShapers/wilor-mlx). Both run at the same speed on Apple Silicon — at these sequence lengths (210 tokens) the model is compute-bound, not memory-bandwidth-bound, so smaller weights don't accelerate inference. Int4 is purely a deployment convenience (2.4 GB → 490 MB).

If you prefer to supply your own MANO data (e.g. obtained directly from [MPI](https://mano.is.tue.mpg.de/)), pass `mano_path=...` to `from_pretrained()`.

## Quick start

```python
from wilor_mlx import HandPosePipeline
import numpy as np

# Load pipeline — detector + pose model download and cache automatically
pipeline = HandPosePipeline.from_pretrained()

# Run on any image — detection + 3D pose in one call
image = np.array(...)  # (H, W, 3) uint8 RGB
hands = pipeline(image)

for hand in hands:
    print(hand.hand_side)     # "left" or "right"
    print(hand.confidence)    # detection confidence
    print(hand.bbox)          # [x1, y1, x2, y2] in pixel coords
    print(hand.keypoints_2d)  # 21 keypoints from detector
    print(hand.keypoints_3d)  # 21 keypoints from WiLoR (3D)
```

The pipeline accepts numpy arrays or MLX arrays. All models download from [HuggingFace](https://huggingface.co/BasinShapers/wilor-mlx) on first use and cache locally.

### Pipeline output

Each detected hand is a `HandPose` with:

| Field | Type | Description |
|---|---|---|
| `hand_side` | `str` | `"left"` or `"right"` |
| `confidence` | `float` | Detection confidence (0–1) |
| `bbox` | `list[float]` | `[x1, y1, x2, y2]` in pixel coords |
| `keypoints_2d` | `list[list[float]]` | 21 keypoints `[x, y]` from detector |
| `keypoints_3d` | `list[list[float]]` or `None` | 21 keypoints `[x, y, z]` from WiLoR |
| `vertices` | `list[list[float]]` or `None` | 778 MANO mesh vertices |

### Options

```python
hands = pipeline(
    image,
    conf_threshold=0.3,      # detection confidence threshold
    iou_threshold=0.5,       # NMS IoU threshold
    include_3d=True,         # run WiLoR for 3D keypoints (default True)
    include_vertices=False,  # include MANO mesh vertices
)
```

### Lower-level API

For pre-cropped hand images or custom detection pipelines, the `WiLoR` model is available directly:

```python
from wilor_mlx import WiLoR
import mlx.core as mx
import numpy as np

model = WiLoR.from_pretrained()

# Input: 256x256 RGB hand crop, uint8 NHWC
crop = mx.array(np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8))
result = model(crop)
mx.eval(result)

keypoints_3d = np.array(result['pred_keypoints_3d'])  # (1, 21, 3)
vertices = np.array(result['pred_vertices'])            # (1, 778, 3)
```

#### WiLoR output format

| Key | Shape | Description |
|---|---|---|
| `pred_keypoints_3d` | (B, 21, 3) | 3D hand joint locations |
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

The pipeline consists of two MLX models:

**Hand Detector** (26.6M params) — YOLOv8m-pose trained on hand detection with left/right classification and 21-keypoint prediction. Backbone: Conv+BN+SiLU, C2f CSP blocks, SPPF. FPN neck with 3 detection scales. Produces bounding boxes, hand side, confidence, and 2D keypoints.

**WiLoR Pose Model** (632M params) — 3D hand pose estimation from cropped hand images:
- **ViT-H/16 backbone** — 1280 embed dim, 32 transformer layers, 16 heads. Processes 192 image patches + 18 learnable tokens (pose/shape/camera).
- **MANO hand model** — differentiable hand mesh with Linear Blend Skinning, Rodrigues rotations, and kinematic chain. 778 vertices, 16 joints.
- **RefineNet** — multi-scale deconvolution pyramid that samples ViT features at projected vertex locations via bilinear grid sampling, then refines the initial MANO parameter estimates.

Both models are ported op-for-op from PyTorch to MLX with sub-pixel numerical parity (max diff < 0.001 for detector, < 0.006 for pose model).

## License

The wilor-mlx code and distributed weight files are MIT licensed. Our weights (on [HuggingFace](https://huggingface.co/BasinShapers/wilor-mlx)) contain only ViT backbone, RefineNet, and learned embedding parameters — no MANO data is bundled or rehosted.

The [MANO hand model](https://mano.is.tue.mpg.de/) is separately licensed by the Max Planck Institute. `WiLoR.from_pretrained()` fetches upstream [WiLoR-mini](https://huggingface.co/warmshao/WiLoR-mini) assets and converts MANO data locally on your machine. If you prefer to obtain MANO directly from MPI, pass `mano_path=...` to use your own copy.
