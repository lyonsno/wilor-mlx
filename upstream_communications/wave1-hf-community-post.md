# HuggingFace Community Post — wilor-mlx

Status: final draft, pending operator go
Surface: HuggingFace community post on lyonsno/wilor-mlx model page
Target: Tuesday 2026-06-09 12:00 ET
Gate: operator approval required before posting

---

Title: WiLoR hand pose estimation rebuilt end-to-end in MLX for Apple Silicon

---

We rebuilt [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) end-to-end in MLX for Apple Silicon — the full inference pipeline including ViT-H/16 backbone, MANO hand model, and RefineNet refinement stage, with sub-millimeter geometric parity against PyTorch.

We couldn't find another public WiLoR MLX or CoreML port, so we're publishing this as a technical priority flag. If we missed related work, we'd love pointers.

## One-line setup

```python
from wilor_mlx import WiLoR
model = WiLoR.from_pretrained()  # auto-downloads weights, derives MANO locally
```

First run needs `torch` once for MANO conversion from the upstream WiLoR-mini checkpoint. After that, inference is pure MLX — no torch dependency.

## Performance (M4 Max, float32)

**Stable live sidecar window** (embedded in real-time hand tracking):

| Backend | p50 | p90 | p95 | p99 |
|---|---|---|---|---|
| **MLX (wilor-mlx)** | **~61 ms** | **~62 ms** | **~63 ms** | **~66 ms** |
| PyTorch MPS (2.5.0) | ~85 ms | ~144 ms | ~238 ms | ~427 ms |

Flat ~61ms with virtually no tail — 8% spread from p50 to p99. MLX's unified memory means no CPU↔GPU transfer stalls.

**Isolated model benchmark:**

| Backend | p50 | FPS |
|---|---|---|
| **MLX** | **36 ms** | **28** |
| PyTorch MPS | 50 ms | 20 |

1.4x faster in pure model compute. The advantage also reproduced on a lower-bandwidth M2 Pro across 80 archived hand-positive camera frames (~30% faster at p50), confirmed by a reversed measurement-order audit.

## Numerical accuracy

| Output | Max abs diff |
|---|---|
| Mesh vertices (778×3) | 0.006 |
| Hand keypoints (21×3) | 0.006 |

Sub-millimeter. Verified layer-by-layer through all 32 transformer blocks — the residual is float32 accumulation noise, not a port error.

## Weights

Float32 (2.4 GB) and int4 (490 MB) safetensors on this model page. Int4 is a download/storage convenience — same inference speed because the model is compute-bound at 210 tokens, not memory-bandwidth-bound.

## MANO licensing

MANO is separately licensed by the Max Planck Institute. wilor-mlx does not bundle or rehost MANO data — it fetches upstream WiLoR-mini assets and converts locally. You can also supply your own copy via `mano_path=...`.

## Links

- **Code:** https://github.com/lyonsno/wilor-mlx
- **Weights:** https://huggingface.co/lyonsno/wilor-mlx
- **Original:** [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) (Zhan et al., CVPR)
