"""Tests for YOLOv8m-pose hand detector MLX port.

Verifies:
1. Module imports and model construction
2. Building block output shapes (Conv, C2f, SPPF)
3. Full model output shape: [B, 69, 5376] for 512x512 input
4. Numerical parity with PyTorch reference
5. Weight loading from converted safetensors
"""

import platform
import sys

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# ---------------------------------------------------------------------------
# 1. Module import and model construction
# ---------------------------------------------------------------------------


class TestDetectorConstruction:
    """Verify detector module imports and model builds."""

    def test_import_detector_module(self):
        from wilor_mlx import detector  # noqa: F401

    def test_construct_conv_block(self):
        from wilor_mlx.detector import Conv

        conv = Conv(c1=3, c2=48, k=3, s=2)
        x = mx.zeros((1, 64, 64, 3))
        out = conv(x)
        assert out.shape == (1, 32, 32, 48)

    def test_construct_c2f_block(self):
        from wilor_mlx.detector import C2f

        c2f = C2f(c1=96, c2=96, n=2, shortcut=True)
        x = mx.zeros((1, 32, 32, 96))
        out = c2f(x)
        assert out.shape == (1, 32, 32, 96)

    def test_construct_sppf_block(self):
        from wilor_mlx.detector import SPPF

        sppf = SPPF(c1=576, c2=576, k=5)
        x = mx.zeros((1, 16, 16, 576))
        out = sppf(x)
        assert out.shape == (1, 16, 16, 576)

    def test_construct_full_model(self):
        from wilor_mlx.detector import HandDetector

        model = HandDetector()
        # Should have 23 layers matching YOLOv8m-pose
        assert model is not None


# ---------------------------------------------------------------------------
# 2. Full model output shape
# ---------------------------------------------------------------------------


class TestDetectorOutputShape:
    """Verify model produces correct output dimensions."""

    def test_output_shape_512(self):
        """512x512 input -> [B, 69, 5376] output."""
        from wilor_mlx.detector import HandDetector

        model = HandDetector()
        x = mx.zeros((1, 512, 512, 3), dtype=mx.uint8)
        out = model(x)
        # 69 = 4 (bbox xywh) + 2 (left/right class) + 63 (21 kpts * 3)
        # 5376 = sum of 3 FPN detection grids
        assert out.shape == (1, 69, 5376)

    def test_output_shape_batch(self):
        """Batch dimension is preserved."""
        from wilor_mlx.detector import HandDetector

        model = HandDetector()
        x = mx.zeros((2, 512, 512, 3), dtype=mx.uint8)
        out = model(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 69

    def test_output_channels_breakdown(self):
        """Output has 4 bbox + 2 class + 63 keypoint channels."""
        from wilor_mlx.detector import HandDetector

        model = HandDetector()
        x = mx.zeros((1, 512, 512, 3), dtype=mx.uint8)
        out = model(x)
        nc = 2  # left, right
        nk = 21 * 3  # 21 keypoints, each (x, y, visibility)
        assert out.shape[1] == 4 + nc + nk


# ---------------------------------------------------------------------------
# 3. Weight loading
# ---------------------------------------------------------------------------


class TestWeightLoading:
    """Verify weights can be loaded from converted format."""

    def test_from_pretrained_loads(self):
        """from_pretrained returns a working model."""
        from wilor_mlx.detector import HandDetector

        model = HandDetector.from_pretrained()
        assert model is not None
        x = mx.zeros((1, 512, 512, 3), dtype=mx.uint8)
        out = model(x)
        assert out.shape == (1, 69, 5376)


# ---------------------------------------------------------------------------
# 4. Numerical parity with PyTorch reference
# ---------------------------------------------------------------------------


class TestNumericalParity:
    """Compare MLX output against PyTorch reference on deterministic input."""

    @pytest.fixture(scope="class")
    def reference_output(self):
        """Load saved PyTorch reference output."""
        ref = np.load("/tmp/yolo_hand_ref_output.npy")
        assert ref.shape == (69, 5376)
        return ref

    @pytest.fixture(scope="class")
    def mlx_output(self):
        """Run MLX model on same deterministic input."""
        from wilor_mlx.detector import HandDetector

        model = HandDetector.from_pretrained()
        np.random.seed(42)
        x_np = np.random.randint(0, 256, (1, 512, 512, 3), dtype=np.uint8)
        x = mx.array(x_np)
        out = model(x)
        mx.eval(out)
        return np.array(out[0])

    def test_output_shape_matches(self, reference_output, mlx_output):
        assert mlx_output.shape == reference_output.shape

    def test_bbox_parity(self, reference_output, mlx_output):
        """Bounding box predictions within tolerance."""
        ref_bbox = reference_output[:4, :]
        mlx_bbox = mlx_output[:4, :]
        max_diff = np.abs(ref_bbox - mlx_bbox).max()
        assert max_diff < 1.0, f"bbox max diff {max_diff:.4f} exceeds 1.0 pixel"

    def test_class_score_parity(self, reference_output, mlx_output):
        """Class scores within tolerance."""
        ref_cls = reference_output[4:6, :]
        mlx_cls = mlx_output[4:6, :]
        max_diff = np.abs(ref_cls - mlx_cls).max()
        assert max_diff < 0.01, f"class score max diff {max_diff:.6f} exceeds 0.01"

    def test_keypoint_parity(self, reference_output, mlx_output):
        """Keypoint predictions within tolerance."""
        ref_kps = reference_output[6:, :]
        mlx_kps = mlx_output[6:, :]
        max_diff = np.abs(ref_kps - mlx_kps).max()
        assert max_diff < 1.0, f"keypoint max diff {max_diff:.4f} exceeds 1.0"
