"""End-to-end hand pose estimation pipeline.

Full image in → detected hands with 3D pose out.
Combines the YOLOv8m-pose hand detector with the WiLoR pose model.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from wilor_mlx.detector import HandDetector
from wilor_mlx.model import WiLoR


@dataclass
class HandPose:
    """A single detected hand with pose estimation results."""

    hand_side: str  # "left" or "right"
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in pixel coords
    keypoints_2d: list[list[float]]  # 21 keypoints from detector, each [x, y]
    keypoints_3d: list[list[float]] | None = None  # 21 keypoints from WiLoR
    vertices: list[list[float]] | None = None  # 778 MANO mesh vertices


def _nms_mlx(boxes, scores, iou_threshold: float = 0.5):
    """Non-maximum suppression on MLX arrays. boxes: (N, 4) xywh, scores: (N,).

    Returns list of kept indices. Runs on CPU via numpy since candidate
    count is typically <10 for hand detection.
    """
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    if len(boxes_np) == 0:
        return []

    x1 = boxes_np[:, 0] - boxes_np[:, 2] / 2
    y1 = boxes_np[:, 1] - boxes_np[:, 3] / 2
    x2 = boxes_np[:, 0] + boxes_np[:, 2] / 2
    y2 = boxes_np[:, 1] + boxes_np[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores_np)

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


def _crop_and_resize_mlx(image_mx, cx, cy, box_size, target_size=256,
                         pad_value: int | None = None):
    """Crop and bilinear-resize a region from an MLX image to target_size.

    image_mx: (H, W, 3) uint8 MLX array.
    pad_value: if set, out-of-bounds pixels use this value instead of edge clamping.
               Use 114 for YOLO detector letterbox convention.
    Returns: (target_size, target_size, 3) uint8 MLX array.
    """
    H, W = image_mx.shape[:2]
    half = box_size / 2.0

    # Sample grid at pixel centers: maps output pixel i to source (start + (i+0.5)/N * size)
    t = (mx.arange(target_size, dtype=mx.float32) + 0.5) / target_size
    src_y = mx.array(float(cy - half)) + t * mx.array(float(2 * half))
    src_x = mx.array(float(cx - half)) + t * mx.array(float(2 * half))
    grid_y, grid_x = mx.meshgrid(src_y, src_x, indexing="ij")

    # Track out-of-bounds mask before clamping
    if pad_value is not None:
        oob = (grid_x < 0) | (grid_x > W - 1) | (grid_y < 0) | (grid_y > H - 1)

    # Clamp to valid pixel range for gather
    grid_x = mx.clip(grid_x, 0, W - 1 - 1e-6)
    grid_y = mx.clip(grid_y, 0, H - 1 - 1e-6)

    # Bilinear interpolation indices
    x0 = grid_x.astype(mx.int32)
    y0 = grid_y.astype(mx.int32)
    x1 = mx.minimum(x0 + 1, W - 1)
    y1 = mx.minimum(y0 + 1, H - 1)

    # Fractional parts
    fx = grid_x - x0.astype(mx.float32)
    fy = grid_y - y0.astype(mx.float32)

    # Gather corners (cast to float for interpolation)
    img_f = image_mx.astype(mx.float32)
    c00 = img_f[y0, x0]  # (target, target, 3)
    c01 = img_f[y0, x1]
    c10 = img_f[y1, x0]
    c11 = img_f[y1, x1]

    # Bilinear blend
    fx = fx[..., None]
    fy = fy[..., None]
    out = (c00 * (1 - fx) * (1 - fy) +
           c01 * fx * (1 - fy) +
           c10 * (1 - fx) * fy +
           c11 * fx * fy)

    # Replace out-of-bounds pixels with pad value
    if pad_value is not None:
        out = mx.where(oob[..., None], mx.array(float(pad_value)), out)

    return out.astype(mx.uint8)


class HandPosePipeline:
    """End-to-end hand pose estimation.

    Usage:
        pipeline = HandPosePipeline.from_pretrained()
        hands = pipeline(image)  # image: (H, W, 3) uint8 RGB numpy array
    """

    def __init__(self, detector: HandDetector, pose_model: WiLoR):
        self.detector = detector
        self.pose_model = pose_model

    @staticmethod
    def from_pretrained():
        """Load both detector and pose model with auto-downloaded weights."""
        detector = HandDetector.from_pretrained()
        pose_model = WiLoR.from_pretrained()
        return HandPosePipeline(detector, pose_model)

    def __call__(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        include_3d: bool = True,
        include_vertices: bool = False,
    ) -> list[HandPose]:
        """Run hand detection + pose estimation on a full image.

        Args:
            image: (H, W, 3) uint8 RGB numpy array.
            conf_threshold: Minimum detection confidence.
            iou_threshold: NMS IoU threshold.
            include_3d: Include 3D keypoints from WiLoR.
            include_vertices: Include MANO mesh vertices.

        Returns:
            List of HandPose, one per detected hand.
        """
        if isinstance(image, np.ndarray):
            image_mx = mx.array(image)
        else:
            image_mx = image
        H, W = image_mx.shape[:2]

        # Letterbox to 512x512 for detector (gray pad, matching YOLO training)
        det_input = _crop_and_resize_mlx(image_mx, W / 2.0, H / 2.0,
                                          max(H, W), target_size=512,
                                          pad_value=114)
        det_input = det_input[None]  # (1, 512, 512, 3)

        # Run detector — stays on MLX
        raw = self.detector(det_input)  # (1, 69, 5376)

        # Parse on MLX: confidence filter
        pred = raw[0]  # (69, 5376)
        cls_scores = pred[4:6]  # (2, 5376)
        best_score = mx.max(cls_scores, axis=0)  # (5376,)
        mask = best_score >= conf_threshold
        mx.eval(mask)

        # Pull only the filtered candidates to CPU for NMS (typically <10)
        mask_np = np.array(mask)
        if not mask_np.any():
            return []

        # Gather filtered candidates
        indices = mx.array(np.where(mask_np)[0])
        boxes_xywh = pred[:4, indices].T  # (N, 4)
        filt_cls = cls_scores[:, indices].T  # (N, 2)
        filt_scores = mx.max(filt_cls, axis=1)
        filt_classes = mx.argmax(filt_cls, axis=1)
        kpts_filt = pred[6:, indices].T  # (N, 63)
        mx.eval(boxes_xywh, filt_scores, filt_classes, kpts_filt)

        # NMS on CPU (tiny candidate set)
        keep = _nms_mlx(boxes_xywh, filt_scores, iou_threshold)
        if not keep:
            return []

        # Scale from 512x512 detector space to original image coords.
        # The detector input is a max(H,W)-square letterbox resized to 512x512,
        # so the mapping is uniform scale + offset for non-square images.
        img_size = max(float(H), float(W))
        scale = img_size / 512.0
        ox = (img_size - float(W)) / 2.0  # x offset from letterbox
        oy = (img_size - float(H)) / 2.0  # y offset from letterbox

        # Convert kept results to numpy once
        boxes_np = np.array(boxes_xywh)
        scores_np = np.array(filt_scores)
        classes_np = np.array(filt_classes)
        kpts_np = np.array(kpts_filt).reshape(-1, 21, 3)

        results = []
        for idx in keep:
            cx_d, cy_d, bw_d, bh_d = boxes_np[idx]
            score = float(scores_np[idx])
            hand_side = "left" if int(classes_np[idx]) == 0 else "right"
            det_kpts = kpts_np[idx]

            # Map from detector space to original image: x_orig = x_det * scale - ox
            cx_o = cx_d * scale - ox
            cy_o = cy_d * scale - oy
            bw_o = bw_d * scale
            bh_o = bh_d * scale

            bbox_orig = [
                float(cx_o - bw_o / 2),
                float(cy_o - bh_o / 2),
                float(cx_o + bw_o / 2),
                float(cy_o + bh_o / 2),
            ]
            kp2d = [[float(det_kpts[j, 0] * scale - ox),
                     float(det_kpts[j, 1] * scale - oy)]
                    for j in range(21)]

            kp3d = None
            verts = None

            if include_3d or include_vertices:
                # Crop hand in original image coords
                box_size = max(bw_o, bh_o) * 2.5
                crop = _crop_and_resize_mlx(image_mx, cx_o, cy_o, box_size,
                                             target_size=256)

                # Flip left hands (WiLoR is MANO_RIGHT)
                if hand_side == "left":
                    crop = crop[:, ::-1]

                wilor_out = self.pose_model(crop[None])
                mx.eval(wilor_out)

                if include_3d:
                    kp3d_arr = np.array(wilor_out["pred_keypoints_3d"][0])
                    if hand_side == "left":
                        kp3d_arr[:, 0] = -kp3d_arr[:, 0]
                    kp3d = kp3d_arr.tolist()

                if include_vertices:
                    verts_arr = np.array(wilor_out["pred_vertices"][0])
                    if hand_side == "left":
                        verts_arr[:, 0] = -verts_arr[:, 0]
                    verts = verts_arr.tolist()

            results.append(HandPose(
                hand_side=hand_side,
                confidence=score,
                bbox=bbox_orig,
                keypoints_2d=kp2d,
                keypoints_3d=kp3d,
                vertices=verts,
            ))

        return results
