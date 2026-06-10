# Wave 1 Technical Post — wilor-mlx

Status: final draft, pending operator go
Channels: HuggingFace community post, MLX community, personal social
Gate: operator approval required before any public posting
Target: Tuesday 2026-06-09 12:00–13:00 ET

---

We rebuilt WiLoR-mini end-to-end in MLX for Apple Silicon: ViT-H/16, MANO, and RefineNet, with sub-millimeter geometric parity against PyTorch.

We couldn't find another public WiLoR MLX/CoreML port or MANO-in-MLX implementation, so we're publishing this as a technical priority flag and would love pointers if we missed related work.

Setup is one line:

```python
from wilor_mlx import WiLoR
model = WiLoR.from_pretrained()  # auto-downloads weights + derives MANO locally
```

First run requires `torch` for a one-time conversion of MANO hand model data from the upstream WiLoR-mini checkpoint; after that, inference runs purely on MLX. MANO is separately licensed by MPI and is not bundled — it's fetched from the public upstream source and converted on your machine.

Performance on M4 Max (float32):

- In our live hand-tracking sidecar, MLX holds flat ~61ms with virtually no tail — p99 is ~66ms (8% spread from median). That's the consistency you need to use 3D hand pose as a real-time control primitive, not just a batch inference model.
- Lower-bandwidth M2 Pro/Tahoe validation also shows MLX ahead on archived hand-positive frames, but recent macOS/Metal changes moved both backends enough that we are treating exact M2 Pro numbers as rebaseline work rather than headline copy.

Numerical fidelity: 0.006 max absolute diff on mesh vertices and hand keypoints — sub-millimeter, verified layer-by-layer against PyTorch through all 32 transformer blocks. The remaining divergence is float32 accumulation noise, not a port error.

Float32 and int4 safetensors weights are on Hugging Face. Int4 cuts the download from 2.4GB to 490MB — same inference speed because the model is compute-bound at 210 tokens, not memory-bandwidth-bound.

GitHub: https://github.com/lyonsno/wilor-mlx
Weights: https://huggingface.co/lyonsno/wilor-mlx
