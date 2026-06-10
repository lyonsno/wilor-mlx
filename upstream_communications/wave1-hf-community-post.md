# HuggingFace Community Post — wilor-mlx

Status: posted and pinned 2026-06-10
Surface: Hugging Face repo discussion on BasinShapers/wilor-mlx
Live: https://huggingface.co/BasinShapers/wilor-mlx/discussions/1
Gate: fulfilled by operator posting

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

The important measurement is the live sidecar route we actually use for interaction: camera frame → hand crop → WiLoR-mini pose/reconstruction → hand-pose event.

On a clean post-reboot M4 Max same-harness smoke over recent 160x120 saved frames from a gesture UI prototype, MLX runs the pose/reconstruction model stage at about 37ms median versus 49ms for PyTorch MPS, and the full saved-frame route at about 49ms versus 60ms. That is roughly a 1.3x model-stage advantage and a 1.2x full-route advantage on the fair comparison denominator we trust most right now.

That latency is low enough to make 3D hand pose plausible as a real-time control primitive, not just a batch inference model. Our traces point to dispatch and synchronization as the main difference, not memory copies: both routes sit on Apple Silicon unified memory, but MLX's lazy graph gives the hot path fewer places for a hitch to land.

Older app-level PyTorch MPS telemetry is what motivated the port; clean reruns moved the comparison denominator enough that we're not using the old tail history as a fresh universal PyTorch-vs-MLX headline.

Larger derived-frame stress tests widen both backends; MLX remained faster in those runs, but we treat those numbers as route/runtime stress evidence rather than the headline model benchmark.

Lower-bandwidth M2 Pro/Tahoe validation also shows MLX ahead on archived hand-positive frames, but recent macOS/Metal changes moved both backends enough that we are treating exact M2 Pro numbers as rebaseline work rather than headline copy.

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
- **Weights:** https://huggingface.co/BasinShapers/wilor-mlx
- **Original:** [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) (Zhan et al., CVPR)
