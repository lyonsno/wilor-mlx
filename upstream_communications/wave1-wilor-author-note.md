# Wave 1 WiLoR Author Note

Status: draft, pending operator review
Target: WiLoR authors (Potamias et al.)
Channel: GitHub issue on rolpotamias/WiLoR, or email if available
Gate: operator approval required before sending

---

Hi WiLoR team,

I built and published an Apple Silicon MLX port of WiLoR-mini — the full inference pipeline including ViT-H/16 backbone, MANO with LBS, and RefineNet:

GitHub: https://github.com/lyonsno/wilor-mlx
Weights: https://huggingface.co/BasinShapers/wilor-mlx

A few things I was careful about:

- **MANO licensing:** wilor-mlx does not bundle or rehost MANO data. On first run it fetches upstream WiLoR-mini assets and converts MANO locally on the user's machine. Users can also supply their own MANO copy.
- **Numerical fidelity:** Sub-millimeter geometric parity — 0.006 max absolute diff on mesh vertices, verified layer-by-layer through all 32 transformer blocks. The residual is float32 cross-implementation accumulation, not a port error.
- **Attribution:** The README and HF model card cite WiLoR-mini and link to the original repo throughout.

If I missed any attribution, licensing, or related-work details, I would really appreciate pointers and will correct them quickly.

I couldn't find another public WiLoR MLX/CoreML port, so I'm publishing this as something that may be useful to Mac/Apple Silicon users. Congrats on the CVPR paper — the model works great.

Best,
Noah
