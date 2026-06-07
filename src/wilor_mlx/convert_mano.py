"""Standalone MANO pkl → npz converter.

Converts MANO_RIGHT.pkl (from mano.is.tue.mpg.de) to a fast-loading
mano.npz file without requiring PyTorch or the WiLoR checkpoint.

Note: The MANO pkl file uses chumpy for some arrays. This converter
includes a chumpy stub that works for most pkl files. If you get
incorrect results, use the full convert CLI instead:
    python -m wilor_mlx.convert <ckpt> <mano_pkl> <mean_params> <output_dir>

Usage:
    python -m wilor_mlx.convert_mano MANO_RIGHT.pkl weights/mano.npz
"""

import sys
import numpy as np


def convert_mano_pkl_to_npz(pkl_path, output_path):
    """Convert MANO_RIGHT.pkl to mano.npz."""
    import pickle
    import os

    # Custom unpickler to handle chumpy references without installing chumpy.
    # Chumpy Ch objects serialize with __reduce__ that calls the class constructor
    # then __setstate__ with a dict containing 'x' (the raw numpy array).
    class _ChStub:
        """Stub for chumpy.ch.Ch and subclasses. Captures the numpy data."""
        def __init__(self, *args, **kwargs):
            self._data = None
        def __setstate__(self, state):
            if isinstance(state, dict) and 'x' in state:
                self._data = np.array(state['x'])
            elif isinstance(state, dict):
                # Try to find any numpy array in the state
                for v in state.values():
                    if isinstance(v, np.ndarray):
                        self._data = v
                        break
        def __array__(self):
            if self._data is not None:
                return self._data
            raise ValueError("ChStub has no array data")

    class _ManoUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith('chumpy'):
                return _ChStub
            return super().find_class(module, name)

    with open(pkl_path, 'rb') as f:
        data = _ManoUnpickler(f, encoding='latin1').load()

    def to_np(v):
        if isinstance(v, np.ndarray):
            return v
        if hasattr(v, 'r'):
            return np.array(v.r)
        return np.array(v)

    # Extract and reshape MANO arrays to match WiLoR's expected format
    posedirs_raw = to_np(data['posedirs']).astype(np.float32)  # (778, 3, 135)
    posedirs = posedirs_raw.reshape(-1, posedirs_raw.shape[-1]).T  # (135, 2334)

    J_regressor = data['J_regressor']
    if hasattr(J_regressor, 'toarray'):
        J_regressor = J_regressor.toarray()
    J_regressor = np.array(J_regressor, dtype=np.float32)

    # shapedirs may come from chumpy stub as flattened — reshape to (778, 3, 10)
    shapedirs = to_np(data['shapedirs']).astype(np.float32)
    if shapedirs.ndim == 1:
        shapedirs = shapedirs.reshape(778, 3, 10)

    mano_arrays = {
        'v_template': to_np(data['v_template']).astype(np.float32),
        'shapedirs': shapedirs,
        'posedirs': posedirs,
        'J_regressor': J_regressor,
        'parents': data['kintree_table'][0].astype(np.int32),
        'lbs_weights': to_np(data['weights']).astype(np.float32),
        # Extra joint vertex indices from smplx.vertex_ids['mano']
        'extra_joints_idxs': np.array([744, 320, 443, 554, 671], dtype=np.int32),
        'joint_map': np.array(
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
            dtype=np.int32,
        ),
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez(output_path, **mano_arrays)
    total_kb = sum(v.nbytes for v in mano_arrays.values()) / 1024
    print(f"Saved MANO buffers ({total_kb:.0f} KB) to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m wilor_mlx.convert_mano <MANO_RIGHT.pkl> <output.npz>")
        print("  Obtain MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/")
        sys.exit(1)

    convert_mano_pkl_to_npz(sys.argv[1], sys.argv[2])
