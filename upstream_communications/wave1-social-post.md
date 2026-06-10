# Social Post — wilor-mlx

Status: final draft, pending operator go
Surface: Twitter/X, LinkedIn
Target: Tuesday 2026-06-09 ~12:30–13:00 ET (30-60 min after HF post)
Gate: operator approval required before posting

---

## Short version (Twitter-length)

Rebuilt WiLoR-mini 3D hand pose estimation end-to-end in MLX for Apple Silicon.

~37ms model-stage latency on M4 Max, with sub-millimeter parity against PyTorch.

One-line setup, no torch at inference time.

GitHub: https://github.com/lyonsno/wilor-mlx
Weights: https://huggingface.co/BasinShapers/wilor-mlx

---

## Longer version (LinkedIn / thread)

Rebuilt WiLoR-mini — 3D hand pose estimation with ViT-H/16, MANO, and RefineNet — end-to-end in MLX for Apple Silicon.

The real win is in live use: inside our hand-tracking sidecar, the clean post-reboot M4 Max same-harness route puts MLX around 37ms median for the pose/reconstruction model stage versus 49ms for PyTorch MPS, and around 49ms versus 60ms for the full saved-frame route.

Older app-level PyTorch MPS telemetry is what motivated the port, but clean reruns moved the comparison denominator enough that I'm not using the old tail history as today's headline.

Sub-millimeter geometric parity (0.006 max diff on mesh vertices), verified layer-by-layer through all 32 transformer blocks. Lower-bandwidth M2 Pro/Tahoe validation also shows MLX ahead, but exact low-end numbers are being rebaselined after recent macOS/Metal changes.

Setup is one line — weights auto-download from Hugging Face, MANO data derives locally from the upstream WiLoR-mini checkpoint (first run needs torch for a one-time MANO conversion; after that, pure MLX).

We couldn't find another public WiLoR MLX/CoreML port, so we're publishing this as a technical priority flag. Pointers to related work welcome.

GitHub: https://github.com/lyonsno/wilor-mlx
Weights: https://huggingface.co/BasinShapers/wilor-mlx
