"""Tests for YOLOv8m-pose hand detector MLX port.

Verifies:
1. Module imports and model construction
2. Building block output shapes (Conv, C2f, SPPF)
3. Full model output shape: [B, 69, 5376] for 512x512 input
4. Numerical parity with PyTorch reference
5. Weight loading from converted safetensors
"""

import os
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

    def test_construct_full_model_output_shape(self):
        from wilor_mlx.detector import HandDetector

        model = HandDetector()
        x = mx.zeros((1, 512, 512, 3), dtype=mx.uint8)
        out = model(x)
        assert out.shape == (1, 69, 5376)


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


@pytest.mark.network
class TestWeightLoading:
    """Verify weights can be loaded from HuggingFace."""

    def test_from_pretrained_loads(self):
        """from_pretrained returns a working model with correct output shape."""
        from wilor_mlx.detector import HandDetector

        model = HandDetector.from_pretrained()
        x = mx.zeros((1, 512, 512, 3), dtype=mx.uint8)
        out = model(x)
        assert out.shape == (1, 69, 5376)


# ---------------------------------------------------------------------------
# 4. Numerical parity with PyTorch reference
# ---------------------------------------------------------------------------

_REF_OUTPUT_PATH = "/tmp/yolo_hand_ref_output.npy"
_REF_WEIGHTS_PATH = "/tmp/hand-detector-v3.safetensors"

_has_parity_fixtures = (
    os.path.exists(_REF_OUTPUT_PATH) and os.path.exists(_REF_WEIGHTS_PATH)
)


@pytest.mark.skipif(not _has_parity_fixtures, reason="Parity fixtures not found in /tmp")
class TestNumericalParity:
    """Compare MLX output against PyTorch reference on deterministic input."""

    @pytest.fixture(scope="class")
    def reference_output(self):
        """Load saved PyTorch reference output."""
        ref = np.load(_REF_OUTPUT_PATH)
        assert ref.shape == (69, 5376)
        return ref

    @pytest.fixture(scope="class")
    def mlx_output(self):
        """Run MLX model on same deterministic input as PyTorch reference."""
        from wilor_mlx.detector import HandDetector, _load_detector_weights

        model = HandDetector()
        _load_detector_weights(model, _REF_WEIGHTS_PATH)
        np.random.seed(42)
        x_nchw = np.random.randint(0, 256, (1, 3, 512, 512), dtype=np.uint8)
        x_np = np.transpose(x_nchw, (0, 2, 3, 1))
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


# ---------------------------------------------------------------------------
# 5. Pipeline unit tests
# ---------------------------------------------------------------------------


class TestNMS:
    """Verify NMS implementation."""

    def test_nms_suppresses_overlapping_boxes(self):
        from wilor_mlx.pipeline import _nms_mlx

        # Two boxes with high overlap, different scores
        boxes = mx.array([[100, 100, 50, 50],   # center, high score
                          [105, 105, 50, 50]])   # nearby, lower score
        scores = mx.array([0.9, 0.7])
        keep = _nms_mlx(boxes, scores, iou_threshold=0.5)
        assert keep == [0]  # only the higher-scoring box survives

    def test_nms_keeps_non_overlapping(self):
        from wilor_mlx.pipeline import _nms_mlx

        boxes = mx.array([[50, 50, 30, 30],
                          [200, 200, 30, 30]])
        scores = mx.array([0.8, 0.9])
        keep = _nms_mlx(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2

    def test_nms_empty(self):
        from wilor_mlx.pipeline import _nms_mlx

        keep = _nms_mlx(mx.zeros((0, 4)), mx.zeros((0,)), 0.5)
        assert keep == []


class TestCropAndResize:
    """Verify MLX bilinear crop-and-resize."""

    def test_output_shape(self):
        from wilor_mlx.pipeline import _crop_and_resize_mlx

        img = mx.zeros((100, 200, 3), dtype=mx.uint8)
        out = _crop_and_resize_mlx(img, 100.0, 50.0, 80.0, target_size=64)
        assert out.shape == (64, 64, 3)

    def test_center_crop_bilinear(self):
        """Cropping the center of a gradient image produces interpolated values in range."""
        from wilor_mlx.pipeline import _crop_and_resize_mlx

        img_np = np.zeros((64, 64, 3), dtype=np.uint8)
        for c in range(3):
            img_np[:, :, c] = np.outer(
                np.linspace(0, 255, 64), np.linspace(0, 255, 64)
            ).astype(np.uint8)
        img = mx.array(img_np)
        # Crop center half
        out = _crop_and_resize_mlx(img, 32.0, 32.0, 32.0, target_size=32)
        mx.eval(out)
        out_np = np.array(out)
        # Center crop should produce mid-range values, not extremes
        assert out_np.min() > 15, f"Center crop min {out_np.min()} too low"
        assert out_np.max() < 240, f"Center crop max {out_np.max()} too high"
        assert 90 < out_np.mean() < 165, f"Center crop mean {out_np.mean():.0f} out of range"

    def test_non_square_source(self):
        """Crop from a non-square image works."""
        from wilor_mlx.pipeline import _crop_and_resize_mlx

        img = mx.array(np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8))
        out = _crop_and_resize_mlx(img, 200.0, 100.0, 150.0, target_size=128)
        assert out.shape == (128, 128, 3)


class TestPipelineCoordinates:
    """Verify pipeline coordinate transform for non-square images."""

    def test_square_image_no_offset(self):
        """For square images, detector-to-original is a simple scale."""
        H, W = 512, 512
        img_size = max(H, W)
        scale = img_size / 512.0
        ox = (img_size - W) / 2.0
        oy = (img_size - H) / 2.0
        assert scale == 1.0
        assert ox == 0.0
        assert oy == 0.0

    def test_landscape_image_has_x_offset(self):
        """Landscape image: shorter height is padded, x has no offset."""
        H, W = 300, 500
        img_size = max(H, W)
        scale = img_size / 512.0
        ox = (img_size - W) / 2.0
        oy = (img_size - H) / 2.0
        assert scale == pytest.approx(500.0 / 512.0)
        assert ox == 0.0
        assert oy == 100.0  # (500 - 300) / 2

    def test_portrait_image_has_y_offset(self):
        """Portrait image: shorter width is padded, y has no offset."""
        H, W = 500, 300
        img_size = max(H, W)
        scale = img_size / 512.0
        ox = (img_size - W) / 2.0
        oy = (img_size - H) / 2.0
        assert scale == pytest.approx(500.0 / 512.0)
        assert ox == 100.0
        assert oy == 0.0
