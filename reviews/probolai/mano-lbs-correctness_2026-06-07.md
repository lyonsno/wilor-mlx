# Probolē: MANO/LBS Correctness

Target: `src/wilor_mlx/mano.py`
Scope: Full review of the MANO hand model and Linear Blend Skinning port.

## Review targets

- `batch_rodrigues`: Axis-angle to rotation matrix. Numerical stability for small angles (the `+ 1e-8` in norm). Skew-symmetric matrix K construction.
- `batch_rigid_transform`: Kinematic chain loop — does the parent indexing produce correct forward kinematics? The `rel_joints` computation. The `transform_mat` helper. The F.pad replacement for homogeneous coordinates.
- `lbs`: Full LBS pipeline — shape blend shapes, pose blend shapes, joint regression, skinning. Weight matrix broadcasting. Homogeneous coordinate handling. Does `posedirs` shape (135, 2334) match the expected (J*9-9, V*3)?
- `MANO.__call__`: Joint concatenation, extra joints from vertex indices, joint_map reordering to OpenPose convention. Does the joint_regressor_extra path work?
- General: Any indexing errors in the kinematic chain? Any transpose/reshape bugs that would silently produce wrong mesh vertices?
