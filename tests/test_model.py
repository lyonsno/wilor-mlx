"""Tests for WiLoR model loading helpers."""

import os
import platform
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


def test_ensure_mano_returns_cached_npz_with_faces(tmp_path, monkeypatch):
    """A cached MANO npz with faces should not enter the conversion path."""
    from wilor_mlx.model import WiLoR

    monkeypatch.setenv("HOME", str(tmp_path))
    cache_dir = tmp_path / ".cache" / "wilor-mlx"
    cache_dir.mkdir(parents=True)
    mano_npz = cache_dir / WiLoR.MANO_CACHE_FILE
    np.savez(mano_npz, faces=np.zeros((1, 3), dtype=np.int32))

    assert WiLoR._ensure_mano() == str(mano_npz)
