# Probolē: ViT Backbone Correctness

Target: `src/wilor_mlx/vit.py`
Scope: Full review of the MLX ViT-H/16 backbone port.

## Review targets

- `rot6d_to_rotmat`: Does the 6D→rotation matrix conversion match the Zhou et al. formulation? Is the transpose/reshape sequence correct? Edge cases near zero-norm inputs?
- `PatchEmbed`: Conv2d kernel_size/stride/padding — do they match WiLoR-mini's (16, 16) patches with padding=4? NHWC layout correct?
- `Attention`: QKV reshape/transpose for multi-head attention. Does the head_dim split match 1280/16=80? Is the SDPA scale correct?
- `Block`: Residual connections — pre-norm (LN before attention/MLP), not post-norm? Drop path removed (inference only)?
- `WiLoRViT.__call__`: Positional embedding addition `pos_embed[:, 1:] + pos_embed[:, :1]` — this is WiLoR's unusual pos_embed usage. Is it correct? Token concatenation order (pose, shape, cam, patches). Decode head indexing. Image feature reshape back to spatial.
- General: Any silent shape mismatches that would produce wrong results without errors? Any broadcasting traps?
