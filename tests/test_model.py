import numpy as np


def test_cached_mano_npz_does_not_shadow_numpy_import(monkeypatch, tmp_path):
    cache_dir = tmp_path / ".cache" / "wilor-mlx"
    cache_dir.mkdir(parents=True)
    mano_path = cache_dir / "mano.npz"
    np.savez(mano_path, faces=np.zeros((1, 3), dtype=np.int32))

    monkeypatch.setenv("HOME", str(tmp_path))

    from wilor_mlx.model import WiLoR

    assert WiLoR._ensure_mano() == str(mano_path)
