# Probolē: RefineNet and Grid Sample Correctness

Target: `src/wilor_mlx/refinenet.py`
Scope: Full review of the RefineNet port including the custom grid_sample implementation.

## Review targets

- `grid_sample`: Bilinear interpolation correctness. Coordinate unnormalization for align_corners=True. Clamp-to-edge boundary handling. The gather_2d helper — does take_along_axis produce correct results for 2D spatial indexing? Any off-by-one in floor/ceil?
- `sample_joint_features`: Coordinate normalization from pixel space to [-1,1]. Grid shape construction. Output transpose to (B, J, C).
- `perspective_projection`: Camera matrix K construction via stacking (MLX doesn't support item assignment). Is the projection formula correct? Missing rotation handling (defaults to identity).
- `DeConvNet`: Does the hardcoded architecture match the WiLoR-mini checkpoint? Branch 0 (640→320) and Branch 1 (640→320→160). NCHW↔NHWC conversions — are they consistent? BatchNorm in eval mode?
- `RefineNet.__call__`: Multi-scale feature sampling, max pooling over vertices, delta prediction. Does the concatenation order match out_dim = 160+320+640 = 1120?
