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


def _nms(boxes, scores, iou_threshold: float = 0.5):
    """Non-maximum suppression. boxes: (N, 4) xywh, scores: (N,)."""
    if len(boxes) == 0:
        return []
    # Convert xywh to xyxy
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return keep


def _crop_hand(image_np, bbox_xywh, scale: float = 2.5):
    """Crop and resize hand region to 256x256 for WiLoR.

    Uses the same expansion and resize logic as WiLoR-mini's ViTDetDataset.
    """
    from PIL import Image

    H, W = image_np.shape[:2]
    cx, cy, bw, bh = bbox_xywh
    box_size = max(bw, bh) * scale / 2

    x1 = int(max(0, cx - box_size))
    y1 = int(max(0, cy - box_size))
    x2 = int(min(W, cx + box_size))
    y2 = int(min(H, cy + box_size))

    crop = image_np[y1:y2, x1:x2]
    img = Image.fromarray(crop)
    img = img.resize((256, 256), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


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
        H, W = image.shape[:2]

        # Resize to 512x512 for detector
        from PIL import Image as PILImage
        det_img = PILImage.fromarray(image).resize((512, 512), PILImage.BILINEAR)
        det_np = np.array(det_img, dtype=np.uint8)

        # Run detector
        det_input = mx.array(det_np[np.newaxis])  # (1, 512, 512, 3)
        raw = self.detector(det_input)
        mx.eval(raw)
        pred = np.array(raw[0])  # (69, 5376)

        # Parse detections
        boxes_xywh = pred[:4].T  # (5376, 4)
        cls_scores = pred[4:6].T  # (5376, 2) — left, right
        kpts_raw = pred[6:].T  # (5376, 63)

        # Best class per anchor
        best_cls = np.argmax(cls_scores, axis=1)  # 0=left, 1=right
        best_score = np.max(cls_scores, axis=1)

        # Filter by confidence
        mask = best_score >= conf_threshold
        if not mask.any():
            return []

        boxes = boxes_xywh[mask]
        scores = best_score[mask]
        classes = best_cls[mask]
        kpts = kpts_raw[mask].reshape(-1, 21, 3)

        # NMS
        keep = _nms(boxes, scores, iou_threshold)
        if not keep:
            return []

        # Scale boxes from 512x512 back to original image coords
        sx, sy = W / 512.0, H / 512.0

        results = []
        for idx in keep:
            box = boxes[idx]
            score = float(scores[idx])
            hand_cls = int(classes[idx])
            hand_side = "left" if hand_cls == 0 else "right"
            det_kpts = kpts[idx]  # (21, 3) — x, y, visibility in 512x512

            # Scale bbox to original
            cx, cy, bw, bh = box
            bbox_orig = [
                float((cx - bw / 2) * sx),
                float((cy - bh / 2) * sy),
                float((cx + bw / 2) * sx),
                float((cy + bh / 2) * sy),
            ]

            # Scale detector keypoints to original
            kp2d = [[float(det_kpts[j, 0] * sx), float(det_kpts[j, 1] * sy)]
                    for j in range(21)]

            kp3d = None
            verts = None

            if include_3d or include_vertices:
                # Crop hand for WiLoR
                bbox_orig_xywh = [cx * sx, cy * sy, bw * sx, bh * sy]
                crop = _crop_hand(image, bbox_orig_xywh, scale=2.5)

                # Flip left hands (WiLoR is MANO_RIGHT)
                if hand_side == "left":
                    crop = crop[:, ::-1].copy()

                crop_mx = mx.array(crop[np.newaxis])  # (1, 256, 256, 3)
                wilor_out = self.pose_model(crop_mx)
                mx.eval(wilor_out)

                if include_3d:
                    kp3d_arr = np.array(wilor_out["pred_keypoints_3d"][0])
                    if hand_side == "left":
                        kp3d_arr[:, 0] = -kp3d_arr[:, 0]  # flip x back
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
