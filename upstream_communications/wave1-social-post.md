# Social Post — wilor-mlx

Status: final draft, pending operator go
Surface: Twitter/X, LinkedIn
Target: Tuesday 2026-06-09 ~12:30–13:00 ET (30-60 min after HF post)
Gate: operator approval required before posting

---

## Short version (Twitter-length)

Rebuilt WiLoR-mini 3D hand pose estimation end-to-end in MLX for Apple Silicon.

Flat ~61ms in a live hand-tracking sidecar with virtually no tail. Sub-millimeter parity against PyTorch.

One-line setup, no torch at inference time.

GitHub: https://github.com/lyonsno/wilor-mlx
Weights: https://huggingface.co/lyonsno/wilor-mlx

---

## Longer version (LinkedIn / thread)

Rebuilt WiLoR-mini — 3D hand pose estimation with ViT-H/16, MANO, and RefineNet — end-to-end in MLX for Apple Silicon.

The real win is in live use: inside our hand-tracking sidecar, MLX holds flat ~61ms with only 8% spread to p99. PyTorch MPS tails blow up to ~427ms at p99. That consistency is what makes 3D hand pose viable as a real-time control primitive.

Sub-millimeter geometric parity (0.006 max diff on mesh vertices), verified layer-by-layer through all 32 transformer blocks. Lower-bandwidth M2 Pro/Tahoe validation also shows MLX ahead, but exact low-end numbers are being rebaselined after recent macOS/Metal changes.

Setup is one line — weights auto-download from Hugging Face, MANO data derives locally from the upstream WiLoR-mini checkpoint (first run needs torch for a one-time MANO conversion; after that, pure MLX).

We couldn't find another public WiLoR MLX/CoreML port, so we're publishing this as a technical priority flag. Pointers to related work welcome.

GitHub: https://github.com/lyonsno/wilor-mlx
Weights: https://huggingface.co/lyonsno/wilor-mlx
