"""Tests for YOLOv8m-pose hand detector and hand pose pipeline.

Verifies:
1. Building block construction and forward pass (non-zero input)
2. Full model output shape and channel layout
3. Weight loading from HuggingFace
4. Numerical parity with PyTorch reference
5. Pipeline: NMS, crop-and-resize, coordinate transforms, mocked end-to-end
"""

import os
import platform
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# ---------------------------------------------------------------------------
# 1. Building blocks
# ---------------------------------------------------------------------------


class TestBuildingBlocks:
    """Verify detector building blocks produce correct shapes on non-zero input."""

    def test_conv_block(self):
        from wilor_mlx.detector import Conv

        conv = Conv(c1=3, c2=48, k=3, s=2)
        x = mx.ones((1, 64, 64, 3))
        out = conv(x)
        assert out.shape == (1, 32, 32, 48)

    def test_c2f_block(self):
        from wilor_mlx.detector import C2f

        c2f = C2f(c1=96, c2=96, n=2, shortcut=True)
        x = mx.ones((1, 32, 32, 96))
        out = c2f(x)
        assert out.shape == (1, 32, 32, 96)

    def test_sppf_block(self):
        from wilor_mlx.detector import SPPF

        sppf = SPPF(c1=576, c2=576, k=5)
        x = mx.ones((1, 16, 16, 576))
        out = sppf(x)
        assert out.shape == (1, 16, 16, 576)


# ---------------------------------------------------------------------------
# 2. Full model output
# ---------------------------------------------------------------------------


class TestDetectorOutput:
    """Verify model produces correct output dimensions."""

    def test_output_shape_and_channels(self):
        """512x512 input -> [B, nc+4+nk, A] with correct channel breakdown."""
        from wilor_mlx.detector import HandDetector

        model = HandDetector()
        x = mx.ones((1, 512, 512, 3), dtype=mx.uint8)
        out = model(x)
        assert out.shape == (1, 69, 5376)
        # Verify channels match model config
        assert out.shape[1] == 4 + model.pose.nc + model.pose.nk

    def test_batch_dimension_preserved(self):
        from wilor_mlx.detector import HandDetector

        model = HandDetector()
        x = mx.ones((2, 512, 512, 3), dtype=mx.uint8)
        out = model(x)
        assert out.shape == (2, 69, 5376)


# ---------------------------------------------------------------------------
# 3. Weight loading
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestWeightLoading:
    """Verify weights can be loaded from HuggingFace."""

    def test_from_pretrained_loads(self):
        from wilor_mlx.detector import HandDetector

        model = HandDetector.from_pretrained()
        x = mx.ones((1, 512, 512, 3), dtype=mx.uint8)
        out = model(x)
        assert out.shape == (1, 69, 5376)


class TestManoCache:
    """Verify MANO cache handling without network or torch side effects."""

    def test_cached_mano_with_faces_returns_without_reconversion(self, tmp_path, monkeypatch):
        """v0.3 cached MANO data already has faces; the fast path must not require torch."""

        from wilor_mlx.model import WiLoR

        cache_dir = tmp_path / ".cache" / "wilor-mlx"
        cache_dir.mkdir(parents=True)
        cached_mano = cache_dir / WiLoR.MANO_CACHE_FILE
        np.savez(cached_mano, faces=np.array([[0, 1, 2]], dtype=np.int32))
        monkeypatch.setenv("HOME", str(tmp_path))

        assert WiLoR._ensure_mano() == str(cached_mano)


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
        ref = np.load(_REF_OUTPUT_PATH)
        assert ref.shape == (69, 5376)
        return ref

    @pytest.fixture(scope="class")
    def mlx_output(self):
        from wilor_mlx.detector import HandDetector, _load_detector_weights

        model = HandDetector()
        _load_detector_weights(model, _REF_WEIGHTS_PATH)
        np.random.seed(42)
        x_nchw = np.random.randint(0, 256, (1, 3, 512, 512), dtype=np.uint8)
        x_np = np.transpose(x_nchw, (0, 2, 3, 1))
        out = model(mx.array(x_np))
        mx.eval(out)
        return np.array(out[0])

    def test_output_shape_matches(self, reference_output, mlx_output):
        assert mlx_output.shape == reference_output.shape

    def test_bbox_parity(self, reference_output, mlx_output):
        max_diff = np.abs(reference_output[:4] - mlx_output[:4]).max()
        assert max_diff < 1.0, f"bbox max diff {max_diff:.4f}"

    def test_class_score_parity(self, reference_output, mlx_output):
        max_diff = np.abs(reference_output[4:6] - mlx_output[4:6]).max()
        assert max_diff < 0.01, f"class score max diff {max_diff:.6f}"

    def test_keypoint_spatial_parity(self, reference_output, mlx_output):
        """Keypoint x/y coordinates within tolerance."""
        ref_kps = reference_output[6:].reshape(21, 3, -1)
        mlx_kps = mlx_output[6:].reshape(21, 3, -1)
        spatial_diff = np.abs(ref_kps[:, :2] - mlx_kps[:, :2]).max()
        assert spatial_diff < 1.0, f"keypoint spatial max diff {spatial_diff:.4f}"

    def test_keypoint_visibility_parity(self, reference_output, mlx_output):
        """Keypoint visibility scores (sigmoid) within tight tolerance."""
        ref_kps = reference_output[6:].reshape(21, 3, -1)
        mlx_kps = mlx_output[6:].reshape(21, 3, -1)
        vis_diff = np.abs(ref_kps[:, 2] - mlx_kps[:, 2]).max()
        assert vis_diff < 0.01, f"keypoint visibility max diff {vis_diff:.6f}"


# ---------------------------------------------------------------------------
# 5. Pipeline: NMS
# ---------------------------------------------------------------------------


class TestNMS:
    def test_suppresses_overlapping(self):
        from wilor_mlx.pipeline import _nms_mlx

        boxes = mx.array([[100, 100, 50, 50], [105, 105, 50, 50]])
        scores = mx.array([0.9, 0.7])
        keep = _nms_mlx(boxes, scores, iou_threshold=0.5)
        assert keep == [0]

    def test_keeps_non_overlapping_in_score_order(self):
        from wilor_mlx.pipeline import _nms_mlx

        boxes = mx.array([[50, 50, 30, 30], [200, 200, 30, 30]])
        scores = mx.array([0.8, 0.9])
        keep = _nms_mlx(boxes, scores, iou_threshold=0.5)
        assert keep == [1, 0]  # highest score first

    def test_empty(self):
        from wilor_mlx.pipeline import _nms_mlx

        assert _nms_mlx(mx.zeros((0, 4)), mx.zeros((0,)), 0.5) == []


# ---------------------------------------------------------------------------
# 6. Pipeline: crop and resize
# ---------------------------------------------------------------------------


class TestCropAndResize:
    def test_output_shape(self):
        from wilor_mlx.pipeline import _crop_and_resize_mlx

        img = mx.ones((100, 200, 3), dtype=mx.uint8)
        out = _crop_and_resize_mlx(img, 100.0, 50.0, 80.0, target_size=64)
        assert out.shape == (64, 64, 3)

    def test_not_constant(self):
        """Bilinear crop of varied input produces non-constant output."""
        from wilor_mlx.pipeline import _crop_and_resize_mlx

        np.random.seed(99)
        img = mx.array(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        out = _crop_and_resize_mlx(img, 32.0, 32.0, 32.0, target_size=32)
        mx.eval(out)
        out_np = np.array(out).astype(float)
        assert out_np.std() > 20, f"Output std {out_np.std():.1f} too low — may be constant"

    def test_pad_value_fills_oob(self):
        """Out-of-bounds pixels use pad_value, not edge clamp."""
        from wilor_mlx.pipeline import _crop_and_resize_mlx

        img = mx.full((10, 10, 3), 200, dtype=mx.uint8)
        # Crop centered at (5, 5) but box_size=30 extends well beyond 10x10
        out = _crop_and_resize_mlx(img, 5.0, 5.0, 30.0, target_size=16, pad_value=114)
        mx.eval(out)
        out_np = np.array(out)
        # Corners should be pad value (114), center should be image value (200)
        assert out_np[0, 0, 0] == 114, f"Corner should be pad 114, got {out_np[0, 0, 0]}"
        assert out_np[8, 8, 0] == 200, f"Center should be image 200, got {out_np[8, 8, 0]}"

    def test_non_square_source(self):
        from wilor_mlx.pipeline import _crop_and_resize_mlx

        img = mx.array(np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8))
        out = _crop_and_resize_mlx(img, 200.0, 100.0, 150.0, target_size=128)
        assert out.shape == (128, 128, 3)


# ---------------------------------------------------------------------------
# 7. Pipeline: mocked end-to-end
# ---------------------------------------------------------------------------


class TestPipelineMocked:
    """Test pipeline with mocked detector and pose model."""

    def _make_mock_detector_output(self, cx, cy, w, h, cls_idx, conf, img_size=512):
        """Create a fake detector output tensor with one detection."""
        A = 5376
        pred = np.zeros((69, A), dtype=np.float32)
        # Put one strong detection at anchor 0
        pred[0, 0] = cx  # bbox cx
        pred[1, 0] = cy  # bbox cy
        pred[2, 0] = w   # bbox w
        pred[3, 0] = h   # bbox h
        # Class scores (pre-sigmoid in detector, but output is post-sigmoid)
        pred[4 + cls_idx, 0] = conf
        # Keypoints: put them near the box center
        for j in range(21):
            pred[6 + j * 3, 0] = cx + (j - 10) * 2  # x
            pred[7 + j * 3, 0] = cy + (j - 10) * 2  # y
            pred[8 + j * 3, 0] = 0.9  # visibility
        return mx.array(pred[np.newaxis])  # (1, 69, A)

    def test_pipeline_returns_hand_pose_and_verifies_wilor_input(self):
        from wilor_mlx.pipeline import HandPosePipeline

        mock_det = MagicMock()
        mock_det.return_value = self._make_mock_detector_output(
            256, 256, 100, 100, cls_idx=1, conf=0.9
        )

        mock_wilor = MagicMock()
        mock_wilor.return_value = {
            "pred_keypoints_3d": mx.zeros((1, 21, 3)),
            "pred_vertices": mx.zeros((1, 778, 3)),
        }

        pipeline = HandPosePipeline(mock_det, mock_wilor)
        image = np.full((512, 512, 3), 128, dtype=np.uint8)
        hands = pipeline(image, conf_threshold=0.5, include_3d=True)

        assert len(hands) == 1
        h = hands[0]
        assert h.hand_side == "right"
        assert h.confidence == pytest.approx(0.9, abs=0.01)
        assert len(h.keypoints_2d) == 21
        assert h.keypoints_3d is not None
        assert len(h.keypoints_3d) == 21

        # Verify WiLoR received a (1, 256, 256, 3) crop
        mock_wilor.assert_called_once()
        wilor_input = mock_wilor.call_args[0][0]
        assert wilor_input.shape == (1, 256, 256, 3)

    def test_pipeline_no_detections(self):
        from wilor_mlx.pipeline import HandPosePipeline

        mock_det = MagicMock()
        mock_det.return_value = mx.zeros((1, 69, 5376))  # all zeros = no confident detections

        pipeline = HandPosePipeline(mock_det, MagicMock())
        hands = pipeline(np.zeros((480, 640, 3), dtype=np.uint8), conf_threshold=0.3)
        assert hands == []

    def test_pipeline_non_square_coordinate_scaling(self):
        """Detections on non-square images have correct letterbox offset."""
        from wilor_mlx.pipeline import HandPosePipeline

        mock_det = MagicMock()
        mock_det.return_value = self._make_mock_detector_output(
            256, 256, 80, 80, cls_idx=1, conf=0.8
        )

        pipeline = HandPosePipeline(mock_det, MagicMock())
        # 300x500 image: max=500, scale=500/512, ox=0, oy=(500-300)/2=100
        H, W = 300, 500
        scale = max(H, W) / 512.0
        oy = (max(H, W) - H) / 2.0
        expected_cx = 256 * scale  # ox=0 for landscape
        expected_cy = 256 * scale - oy

        hands = pipeline(
            np.zeros((H, W, 3), dtype=np.uint8),
            conf_threshold=0.3, include_3d=False,
        )
        assert len(hands) == 1
        bbox_cx = (hands[0].bbox[0] + hands[0].bbox[2]) / 2
        bbox_cy = (hands[0].bbox[1] + hands[0].bbox[3]) / 2
        assert bbox_cx == pytest.approx(expected_cx, abs=2)
        assert bbox_cy == pytest.approx(expected_cy, abs=2)

    def test_pipeline_left_hand_with_vertices(self):
        from wilor_mlx.pipeline import HandPosePipeline

        mock_det = MagicMock()
        mock_det.return_value = self._make_mock_detector_output(
            256, 256, 100, 100, cls_idx=0, conf=0.85  # cls 0 = left
        )

        mock_wilor = MagicMock()
        mock_wilor.return_value = {
            "pred_keypoints_3d": mx.ones((1, 21, 3)),
            "pred_vertices": mx.ones((1, 778, 3)),
        }

        pipeline = HandPosePipeline(mock_det, mock_wilor)
        hands = pipeline(
            np.zeros((512, 512, 3), dtype=np.uint8),
            include_3d=True, include_vertices=True, conf_threshold=0.5,
        )
        assert len(hands) == 1
        assert hands[0].hand_side == "left"
        # 3D keypoints x should be negated for left hand flip-back
        assert hands[0].keypoints_3d[0][0] == pytest.approx(-1.0)
        # Vertices x should also be negated
        assert hands[0].vertices is not None
        assert len(hands[0].vertices) == 778
        assert hands[0].vertices[0][0] == pytest.approx(-1.0)

    def test_pipeline_two_hands(self):
        """Two non-overlapping detections (left + right) both returned."""
        from wilor_mlx.pipeline import HandPosePipeline

        A = 5376
        pred = np.zeros((69, A), dtype=np.float32)
        # Hand 1: right, at (150, 256)
        pred[0, 0] = 150; pred[1, 0] = 256; pred[2, 0] = 80; pred[3, 0] = 80
        pred[5, 0] = 0.9  # cls 1 = right
        for j in range(21):
            pred[6 + j * 3, 0] = 150; pred[7 + j * 3, 0] = 256; pred[8 + j * 3, 0] = 0.8
        # Hand 2: left, at (350, 256)
        pred[0, 1] = 350; pred[1, 1] = 256; pred[2, 1] = 80; pred[3, 1] = 80
        pred[4, 1] = 0.85  # cls 0 = left
        for j in range(21):
            pred[6 + j * 3, 1] = 350; pred[7 + j * 3, 1] = 256; pred[8 + j * 3, 1] = 0.8

        mock_det = MagicMock()
        mock_det.return_value = mx.array(pred[np.newaxis])

        pipeline = HandPosePipeline(mock_det, MagicMock())
        hands = pipeline(
            np.zeros((512, 512, 3), dtype=np.uint8),
            conf_threshold=0.3, include_3d=False,
        )
        assert len(hands) == 2
        sides = {h.hand_side for h in hands}
        assert sides == {"left", "right"}
