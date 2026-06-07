"""MANO hand model and LBS (Linear Blend Skinning) ported to MLX.

Ported from smplx.lbs and wilor_mini.models.mano_wrapper.
"""

import mlx.core as mx


def batch_rodrigues(rot_vecs):
    """Convert batch of axis-angle vectors to rotation matrices.
    Args:
        rot_vecs: (N, 3) axis-angle vectors
    Returns:
        (N, 3, 3) rotation matrices
    """
    batch_size = rot_vecs.shape[0]
    angle = mx.linalg.norm(rot_vecs + 1e-8, axis=1, keepdims=True)  # (N, 1)
    rot_dir = rot_vecs / angle  # (N, 3)

    cos = mx.cos(angle)[:, :, None]  # (N, 1, 1)
    sin = mx.sin(angle)[:, :, None]  # (N, 1, 1)

    rx = rot_dir[:, 0:1]  # (N, 1)
    ry = rot_dir[:, 1:2]
    rz = rot_dir[:, 2:3]
    zeros = mx.zeros((batch_size, 1))

    K = mx.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1)
    K = K.reshape(batch_size, 3, 3)

    ident = mx.eye(3)[None]  # (1, 3, 3)
    rot_mat = ident + sin * K + (1 - cos) * (K @ K)
    return rot_mat


def blend_shapes(betas, shape_disps):
    """Compute per-vertex displacement from shape parameters.
    Args:
        betas: (B, num_betas)
        shape_disps: (V, 3, num_betas)
    Returns:
        (B, V, 3) vertex displacements
    """
    return mx.einsum('bl,mkl->bmk', betas, shape_disps)


def vertices2joints(J_regressor, vertices):
    """Calculate 3D joint locations from vertices.
    Args:
        J_regressor: (J, V) regressor matrix
        vertices: (B, V, 3)
    Returns:
        (B, J, 3)
    """
    return mx.einsum('bik,ji->bjk', vertices, J_regressor)


def transform_mat(R, t):
    """Create batch of 4x4 transformation matrices.
    Args:
        R: (N, 3, 3) rotation matrices
        t: (N, 3, 1) translation vectors
    Returns:
        (N, 4, 4) transformation matrices
    """
    N = R.shape[0]
    # Build [R | t; 0 0 0 1]
    bottom_left = mx.zeros((N, 1, 3))
    bottom_right = mx.ones((N, 1, 1))
    top = mx.concatenate([R, t], axis=2)       # (N, 3, 4)
    bottom = mx.concatenate([bottom_left, bottom_right], axis=2)  # (N, 1, 4)
    return mx.concatenate([top, bottom], axis=1)  # (N, 4, 4)


def batch_rigid_transform(rot_mats, joints, parents):
    """Apply batch of rigid transformations to joints.
    Args:
        rot_mats: (B, N, 3, 3) rotation matrices
        joints: (B, N, 3) joint locations
        parents: (N,) parent indices
    Returns:
        posed_joints: (B, N, 3)
        rel_transforms: (B, N, 4, 4)
    """
    B = rot_mats.shape[0]
    num_joints = rot_mats.shape[1]

    # Convert parents to Python list for indexing
    parents_list = parents.tolist()

    joints = joints[:, :, :, None]  # (B, N, 3, 1)

    rel_joints_list = [joints[:, 0:1]]
    for i in range(1, num_joints):
        pi = int(parents_list[i])
        rel_joints_list.append(joints[:, i:i+1] - joints[:, pi:pi+1])
    rel_joints = mx.concatenate(rel_joints_list, axis=1)  # (B, N, 3, 1)

    # Build per-joint transform matrices
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)
    ).reshape(B, num_joints, 4, 4)

    # Chain transforms along kinematic tree
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, num_joints):
        pi = int(parents_list[i])
        curr_res = transform_chain[pi] @ transforms_mat[:, i]
        transform_chain.append(curr_res)

    transforms = mx.stack(transform_chain, axis=1)  # (B, N, 4, 4)

    posed_joints = transforms[:, :, :3, 3]  # (B, N, 3)

    # Compute relative transforms
    joints_homogen = mx.pad(joints, [(0,0), (0,0), (0,1), (0,0)],
                            constant_values=0)  # (B, N, 4, 1)
    # Set the last element to 1 for homogeneous coords... actually F.pad value=0 then overwrite
    # Wait — PyTorch pads [0,0,0,1] meaning pad last dim by 0 on left and 0 on right,
    # second-to-last dim by 0 on left and 1 on bottom (value=0 for joints_homogen).
    # Let me re-read the PyTorch code:
    # joints_homogen = F.pad(joints, [0, 0, 0, 1])  — pads dim -1 by (0,0) and dim -2 by (0,1)
    # joints is (B, N, 3, 1), so after pad: (B, N, 4, 1) with last row = 0

    # Relative transform subtracts the rest-pose joint contribution
    # rel_transforms = transforms - pad(transforms @ joints_homogen, [3,0,0,0,0,0,0,0])
    Tj = transforms @ joints_homogen  # (B, N, 4, 1)
    # Pad: [3,0,0,0,0,0,0,0] means pad dim -1 by (3,0), dim -2 by (0,0), dim -3 by (0,0), dim -4 by (0,0)
    # So Tj goes from (B,N,4,1) to (B,N,4,4) with 3 zero columns prepended
    Tj_padded = mx.pad(Tj, [(0,0), (0,0), (0,0), (3,0)])  # (B, N, 4, 4)

    rel_transforms = transforms - Tj_padded

    return posed_joints, rel_transforms


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, pose2rot=False):
    """Linear Blend Skinning.
    Args:
        betas: (B, num_betas) shape parameters
        pose: (B, N, 3, 3) rotation matrices (when pose2rot=False)
        v_template: (V, 3) template mesh
        shapedirs: (V, 3, num_betas) shape blend shapes
        posedirs: (P, V*3) pose blend shapes
        J_regressor: (J, V) joint regressor
        parents: (J,) parent indices
        lbs_weights: (V, J) skinning weights
        pose2rot: if True, pose is axis-angle (B, J*3); if False, rotation matrices
    Returns:
        verts: (B, V, 3) posed vertices
        joints: (B, J, 3) joint locations
    """
    batch_size = max(betas.shape[0], pose.shape[0])

    # Shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Joint locations from shaped vertices
    J = vertices2joints(J_regressor, v_shaped)

    # Pose blend shapes
    ident = mx.eye(3)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape(batch_size, -1)
    else:
        pose_feature = (pose[:, 1:].reshape(batch_size, -1, 3, 3) - ident).reshape(batch_size, -1)
        rot_mats = pose.reshape(batch_size, -1, 3, 3)

    pose_offsets = (pose_feature @ posedirs).reshape(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped

    # Rigid transform along kinematic tree
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)

    # Skinning
    num_joints = J_regressor.shape[0]
    W = mx.broadcast_to(lbs_weights[None], (batch_size, *lbs_weights.shape))
    T = (W @ A.reshape(batch_size, num_joints, 16)).reshape(batch_size, -1, 4, 4)

    homogen_coord = mx.ones((batch_size, v_posed.shape[1], 1))
    v_posed_homo = mx.concatenate([v_posed, homogen_coord], axis=2)  # (B, V, 4)
    v_homo = T @ v_posed_homo[:, :, :, None]  # (B, V, 4, 1)

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


class MANO:
    """MANO hand model for MLX.

    Holds the model buffers (template, shape dirs, pose dirs, etc.)
    and runs LBS forward pass.
    """

    def __init__(self):
        # These will be loaded from the checkpoint
        self.v_template = None      # (778, 3)
        self.shapedirs = None       # (778, 3, 10)
        self.posedirs = None        # (135, 2334) = (J*9-9, V*3)
        self.J_regressor = None     # (16, 778)
        self.parents = None         # (16,) int
        self.lbs_weights = None     # (778, 16)

        # Extra joints
        self.joint_regressor_extra = None
        self.extra_joints_idxs = None
        # mano_to_openpose mapping
        self.joint_map = mx.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])

    def __call__(self, global_orient, hand_pose, betas, pose2rot=False, **kwargs):
        """Run MANO forward pass.
        Args:
            global_orient: (B, 1, 3, 3) global orientation
            hand_pose: (B, 15, 3, 3) hand pose
            betas: (B, 10) shape parameters
            pose2rot: whether pose is axis-angle
        Returns:
            vertices: (B, 778, 3)
            joints: (B, 21, 3) in OpenPose order
        """
        full_pose = mx.concatenate([
            global_orient.reshape(-1, 1, 3, 3),
            hand_pose.reshape(-1, 15, 3, 3)
        ], axis=1)  # (B, 16, 3, 3)

        vertices, joints = lbs(
            betas, full_pose,
            self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.parents, self.lbs_weights,
            pose2rot=pose2rot,
        )

        # Extra joints from vertex indices
        extra_joints = vertices[:, self.extra_joints_idxs]
        joints = mx.concatenate([joints, extra_joints], axis=1)
        joints = joints[:, self.joint_map]

        if self.joint_regressor_extra is not None:
            extra = vertices2joints(self.joint_regressor_extra, vertices)
            joints = mx.concatenate([joints, extra], axis=1)

        return vertices, joints
