# Wave 1 Technical Post — wilor-mlx

Status: draft, pending operator review
Channels: HuggingFace, MLX community, personal social, direct author note
Gate: operator approval required before any public posting

---

We rebuilt WiLoR-mini end-to-end in MLX for Apple Silicon: ViT-H/16, MANO, and RefineNet, with sub-millimeter geometric parity against PyTorch.

We couldn't find another public WiLoR MLX/CoreML port or MANO-in-MLX implementation, so we're publishing this as a technical priority flag and would love pointers if we missed related work.

Setup is one line:

```python
from wilor_mlx import WiLoR
model = WiLoR.from_pretrained()  # auto-downloads weights + derives MANO locally
```

First run requires `torch` for a one-time conversion of MANO hand model data from the upstream WiLoR-mini checkpoint; after that, inference runs purely on MLX. MANO is separately licensed by MPI and is not bundled — it's fetched from the public upstream source and converted on your machine.

Performance on M4 Max:
- ~1.4x faster than PyTorch MPS in isolated model benchmarks (36ms vs 50ms)
- Much tighter in live sidecar use: ~60ms p50 / ~63ms p90 vs MPS's ~85ms p50 / ~144ms p90. MPS tail latency is 2.3x worse — MLX's unified memory eliminates the CPU↔GPU sync spikes that make MPS unpredictable

Numerical fidelity: 0.006 max absolute diff on mesh vertices and hand keypoints — sub-millimeter, verified layer-by-layer against PyTorch through all 32 transformer blocks. The remaining divergence is float32 accumulation noise, not a port error.

Float32 and int4 safetensors weights are on Hugging Face. Int4 cuts the download from 2.4GB to 490MB with no speed difference on Apple Silicon.

GitHub: https://github.com/lyonsno/wilor-mlx
Weights: https://huggingface.co/lyonsno/wilor-mlx
