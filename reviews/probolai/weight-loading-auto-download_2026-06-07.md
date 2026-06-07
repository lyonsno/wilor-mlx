# Probolē: Weight Loading and Auto-Download

Target: `src/wilor_mlx/convert.py`, `src/wilor_mlx/model.py`
Scope: Review of weight conversion, loading, and automatic download/caching pipeline.

## Review targets

- `load_pytorch_checkpoint`: Conv2d weight transposition (OIHW→OHWI). ConvTranspose2d transposition (IOHW→OHWI). Linear weights NOT transposed (same layout in PyTorch and MLX). BatchNorm running_mean/running_var loading. Are all 490 checkpoint keys accounted for?
- `load_safetensors_weights`: Does it load all arrays from the stripped (no-MANO) safetensors? Are backbone buffers (pos_embed, init_cam, init_hand_pose, init_betas) loaded correctly?
- `load_mano_npz`: Does it handle both old-format (no init params) and new-format (with init params) npz files? Int32 casting for index arrays.
- `_ensure_mano`: Auto-download from warmshao/WiLoR-mini — does it download the right files? MANO extraction from PyTorch checkpoint — are the array shapes correct? Cache path at ~/.cache/wilor-mlx/. Does the mean_params loading work?
- `save_mlx_weights`: Does it exclude MANO data? Are all weight keys present?
- Batched eval: Does the 64-batch eval avoid Metal shared event exhaustion?
- `from_pretrained`: Zero-arg path — does the auto-download + auto-convert + cache check work end-to-end? Keyword arg ordering.
