"""YOLOv8m-pose hand detector in MLX.

Port of the WiLoR-mini hand detection model (Ultralytics YOLOv8m-pose
trained on hand detection with 2 classes: left/right).

The detector takes a full image and returns bounding boxes, hand side
(left/right), confidence scores, and 21 2D hand keypoints per detection.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _autopad(k: int, p: int | None = None) -> int:
    """Compute same-padding for kernel size k."""
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU (standard YOLOv8 conv block)."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None):
        super().__init__()
        p = _autopad(k, p)
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm(c2, momentum=0.1, eps=1e-5)

    def __call__(self, x):
        return nn.silu(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard YOLOv8 bottleneck: two 3x3 convs with optional residual."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        self.cv1 = Conv(c1, c2, k=3)
        self.cv2 = Conv(c2, c2, k=3)
        self.add = shortcut and c1 == c2

    def __call__(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8 C2f block).

    Split input channels, run through n bottlenecks, concat all
    intermediate outputs, merge with a final 1x1 conv.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False):
        super().__init__()
        self.c = c2 // 2  # hidden channels per branch
        self.cv1 = Conv(c1, 2 * self.c, k=1)
        self.cv2 = Conv((2 + n) * self.c, c2, k=1)
        self.m = [Bottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)]

    def __call__(self, x):
        y = self.cv1(x)
        # Split along channel dim (NHWC layout)
        y1, y2 = y[..., : self.c], y[..., self.c :]
        chunks = [y1, y2]
        out = y2
        for bottleneck in self.m:
            out = bottleneck(out)
            chunks.append(out)
        return self.cv2(mx.concatenate(chunks, axis=-1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (single kernel size, cascaded)."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c_ * 4, c2, k=1)
        self.k = k

    def __call__(self, x):
        x = self.cv1(x)
        p = self.k // 2
        y1 = _max_pool2d(x, self.k, stride=1, padding=p)
        y2 = _max_pool2d(y1, self.k, stride=1, padding=p)
        y3 = _max_pool2d(y2, self.k, stride=1, padding=p)
        return self.cv2(mx.concatenate([x, y1, y2, y3], axis=-1))


def _max_pool2d(x, k: int, stride: int = 1, padding: int = 0):
    """2D max pooling for NHWC layout."""
    if padding > 0:
        x = mx.pad(x, [(0, 0), (padding, padding), (padding, padding), (0, 0)],
                    constant_values=-float("inf"))
    B, H, W, C = x.shape
    oH = (H - k) // stride + 1
    oW = (W - k) // stride + 1
    # Gather windows
    windows = []
    for i in range(k):
        for j in range(k):
            windows.append(x[:, i:i + oH * stride:stride, j:j + oW * stride:stride, :])
    stacked = mx.stack(windows, axis=-1)  # (B, oH, oW, C, k*k)
    return mx.max(stacked, axis=-1)


class DFL(nn.Module):
    """Distribution Focal Loss layer for box regression."""

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, kernel_size=1, bias=False)
        # Weight is fixed arange, not learned
        self.conv.weight = mx.arange(c1, dtype=mx.float32).reshape(1, 1, 1, c1)
        self.c1 = c1

    def __call__(self, x):
        """x: (B, 4 * reg_max, A) -> (B, 4, A)"""
        B, channels, A = x.shape
        # Reshape to (B, 4, reg_max, A), softmax over reg_max, weighted sum
        x = x.reshape(B, 4, self.c1, A)
        x = mx.softmax(x, axis=2)
        # Weighted sum: multiply by [0, 1, ..., reg_max-1]
        weights = mx.arange(self.c1, dtype=mx.float32).reshape(1, 1, self.c1, 1)
        return (x * weights).sum(axis=2)  # (B, 4, A)


# ---------------------------------------------------------------------------
# Detection head
# ---------------------------------------------------------------------------


class Pose(nn.Module):
    """YOLOv8 Pose detection head: bbox + class + keypoints."""

    def __init__(self, nc: int = 2, nk: int = 63, ch: tuple = (192, 384, 576)):
        super().__init__()
        self.nc = nc
        self.nk = nk
        self.no = nc + 64 + nk  # 64 for DFL bbox (4 * reg_max)
        self.nl = len(ch)
        self.reg_max = 16
        self.stride = mx.array([8.0, 16.0, 32.0])

        c2 = max(64, ch[0] // 4)  # bbox branch channels
        c3 = max(nc, ch[0])  # class branch channels (at least nc)
        # Actually c3 for YOLOv8m-pose is max(nc, min(ch))
        # From inspection: cv3[0] in_ch is 192, so c3 = 192
        c3 = max(nc, min(ch))
        c4 = max(nk, ch[0])  # kpt branch — actually just nk output

        # Bbox regression heads (per FPN level)
        self.cv2 = [
            [Conv(c, c2, k=3), Conv(c2, c2, k=3), nn.Conv2d(c2, 64, kernel_size=1)]
            for c in ch
        ]
        # Classification heads
        self.cv3 = [
            [Conv(c, c3, k=3), Conv(c3, c3, k=3), nn.Conv2d(c3, nc, kernel_size=1)]
            for c in ch
        ]
        # Keypoint heads
        self.cv4 = [
            [Conv(c, nk, k=3), Conv(nk, nk, k=3), nn.Conv2d(nk, nk, kernel_size=1)]
            for c in ch
        ]
        self.dfl = DFL(self.reg_max)

    def __call__(self, features: list):
        """features: list of 3 feature maps from FPN levels."""
        outputs = []
        for i, feat in enumerate(features):
            # feat: (B, H, W, C) in NHWC
            # Bbox branch
            box = feat
            for layer in self.cv2[i]:
                box = layer(box) if not isinstance(layer, nn.Conv2d) else layer(box)

            # Class branch
            cls = feat
            for layer in self.cv3[i]:
                cls = layer(cls) if not isinstance(layer, nn.Conv2d) else layer(cls)

            # Keypoint branch
            kpt = feat
            for layer in self.cv4[i]:
                kpt = layer(kpt) if not isinstance(layer, nn.Conv2d) else layer(kpt)

            B, H, W, _ = box.shape
            # Flatten spatial dims: (B, H*W, C) then transpose to (B, C, H*W)
            box = box.reshape(B, H * W, -1).transpose(0, 2, 1)  # (B, 64, A)
            cls = cls.reshape(B, H * W, -1).transpose(0, 2, 1)  # (B, nc, A)
            kpt = kpt.reshape(B, H * W, -1).transpose(0, 2, 1)  # (B, nk, A)

            outputs.append((box, cls, kpt))

        # Concat across FPN levels
        all_box = mx.concatenate([o[0] for o in outputs], axis=2)  # (B, 64, total_A)
        all_cls = mx.concatenate([o[1] for o in outputs], axis=2)  # (B, nc, total_A)
        all_kpt = mx.concatenate([o[2] for o in outputs], axis=2)  # (B, nk, total_A)

        # DFL decode bbox: (B, 64, A) -> (B, 4, A)
        box_decoded = self.dfl(all_box)

        # Sigmoid on class scores
        all_cls = mx.sigmoid(all_cls)

        # Anchor generation and bbox decode
        box_decoded = self._decode_bboxes(box_decoded, features)

        # Decode keypoints
        all_kpt = self._decode_kpts(all_kpt, features)

        return mx.concatenate([box_decoded, all_cls, all_kpt], axis=1)

    def _make_anchors(self, features):
        """Generate anchor points and stride tensors for all FPN levels."""
        anchors = []
        strides = []
        for i, feat in enumerate(features):
            _, H, W, _ = feat.shape
            s = float(self.stride[i])
            # Grid of (x, y) anchor centers
            sx = mx.arange(W, dtype=mx.float32) + 0.5
            sy = mx.arange(H, dtype=mx.float32) + 0.5
            grid_y, grid_x = mx.meshgrid(sy, sx, indexing="ij")
            anchor = mx.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
            anchors.append(anchor)
            strides.append(mx.full((H * W, 1), s))
        return mx.concatenate(anchors, axis=0), mx.concatenate(strides, axis=0)

    def _decode_bboxes(self, box, features):
        """Decode DFL output to xywh bboxes in pixel coords."""
        anchors, strides = self._make_anchors(features)
        # anchors: (A, 2), strides: (A, 1)
        # box: (B, 4, A) — dist2bbox format (left, top, right, bottom distances)
        # Decode: center = anchor * stride, wh from distances
        lt = box[:, :2, :]  # (B, 2, A) — left, top distances
        rb = box[:, 2:, :]  # (B, 2, A) — right, bottom distances

        # anchor centers in pixel space
        ac = (anchors * strides.squeeze(-1).reshape(-1, 1)).T  # (2, A)
        ac = ac[None, :, :]  # (1, 2, A)
        s = strides.squeeze(-1).T[None, :]  # (1, 1, A)

        x1y1 = ac - lt * s
        x2y2 = ac + rb * s
        cx = (x1y1[:, 0:1, :] + x2y2[:, 0:1, :]) / 2
        cy = (x1y1[:, 1:2, :] + x2y2[:, 1:2, :]) / 2
        w = x2y2[:, 0:1, :] - x1y1[:, 0:1, :]
        h = x2y2[:, 1:2, :] - x1y1[:, 1:2, :]
        return mx.concatenate([cx, cy, w, h], axis=1)

    def _decode_kpts(self, kpt, features):
        """Decode keypoint predictions to pixel coordinates."""
        anchors, strides = self._make_anchors(features)
        B = kpt.shape[0]
        A = kpt.shape[2]
        # kpt: (B, 63, A) -> (B, 21, 3, A)
        kpt = kpt.reshape(B, 21, 3, A)
        # xy: decode relative to anchor, visibility: sigmoid
        ac = (anchors * strides.squeeze(-1).reshape(-1, 1)).T  # (2, A)
        s = strides.squeeze(-1).T  # (1, A)
        # x, y decoded
        kpt_x = (kpt[:, :, 0, :] * 2.0 * s[None, None, :] + ac[None, 0:1, :])
        kpt_y = (kpt[:, :, 1, :] * 2.0 * s[None, None, :] + ac[None, 1:2, :])
        kpt_v = mx.sigmoid(kpt[:, :, 2, :])
        decoded = mx.stack([kpt_x, kpt_y, kpt_v], axis=2)  # (B, 21, 3, A)
        return decoded.reshape(B, 63, A)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class HandDetector(nn.Module):
    """YOLOv8m-pose hand detector in MLX.

    Architecture: YOLOv8m backbone + FPN + Pose head.
    Input: (B, H, W, 3) uint8 RGB, NHWC layout.
    Output: (B, 69, A) — 4 bbox + 2 class + 63 keypoints.
    """

    def __init__(self):
        super().__init__()
        # Backbone (layers 0-9)
        self.b0 = Conv(3, 48, k=3, s=2)      # P1/2
        self.b1 = Conv(48, 96, k=3, s=2)     # P2/4
        self.b2 = C2f(96, 96, n=2, shortcut=True)
        self.b3 = Conv(96, 192, k=3, s=2)    # P3/8
        self.b4 = C2f(192, 192, n=4, shortcut=True)
        self.b5 = Conv(192, 384, k=3, s=2)   # P4/16
        self.b6 = C2f(384, 384, n=4, shortcut=True)
        self.b7 = Conv(384, 576, k=3, s=2)   # P5/32
        self.b8 = C2f(576, 576, n=2, shortcut=True)
        self.b9 = SPPF(576, 576, k=5)

        # FPN Head (layers 10-21)
        # Layer 10: Upsample (handled in forward)
        # Layer 11: Concat with b6 output
        self.h12 = C2f(960, 384, n=2, shortcut=False)   # 576 + 384
        # Layer 13: Upsample
        # Layer 14: Concat with b4 output
        self.h15 = C2f(576, 192, n=2, shortcut=False)   # 384 + 192
        self.h16 = Conv(192, 192, k=3, s=2)
        # Layer 17: Concat with h12 output
        self.h18 = C2f(576, 384, n=2, shortcut=False)   # 192 + 384
        self.h19 = Conv(384, 384, k=3, s=2)
        # Layer 20: Concat with b9 output
        self.h21 = C2f(960, 576, n=2, shortcut=False)   # 384 + 576

        # Pose head (layer 22)
        self.pose = Pose(nc=2, nk=63, ch=(192, 384, 576))

    def __call__(self, x):
        # Normalize: uint8 -> float32 [0, 1]
        if x.dtype == mx.uint8:
            x = x.astype(mx.float32) / 255.0

        # Backbone
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        p3 = self.b4(x3)     # P3: stride 8
        x5 = self.b5(p3)
        p4 = self.b6(x5)     # P4: stride 16
        x7 = self.b7(p4)
        x8 = self.b8(x7)
        p5 = self.b9(x8)     # P5: stride 32

        # FPN
        up1 = _upsample_nearest(p5, scale=2)
        f12 = self.h12(mx.concatenate([up1, p4], axis=-1))

        up2 = _upsample_nearest(f12, scale=2)
        f15 = self.h15(mx.concatenate([up2, p3], axis=-1))

        f16 = self.h16(f15)
        f18 = self.h18(mx.concatenate([f16, f12], axis=-1))

        f19 = self.h19(f18)
        f21 = self.h21(mx.concatenate([f19, p5], axis=-1))

        # Pose head takes 3 FPN features
        return self.pose([f15, f18, f21])

    HF_REPO_ID = "BasinShapers/wilor-mlx"
    HF_DETECTOR_FILE = "hand-detector.safetensors"

    @staticmethod
    def from_pretrained(weights_path=None):
        """Load detector with pre-converted MLX weights."""
        if weights_path is None:
            weights_path = HandDetector._download_weights()

        model = HandDetector()
        _load_detector_weights(model, weights_path)
        return model

    @staticmethod
    def _download_weights():
        """Download converted detector weights from HuggingFace."""
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=HandDetector.HF_REPO_ID,
            filename=HandDetector.HF_DETECTOR_FILE,
        )


def _upsample_nearest(x, scale: int = 2):
    """Nearest-neighbor 2x upsample for NHWC layout."""
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    return x.reshape(B, H * scale, W * scale, C)


def _load_detector_weights(model, weights_path):
    """Load safetensors weights into the MLX detector model."""
    import safetensors.numpy

    arrays = safetensors.numpy.load_file(weights_path)
    weights = {k: mx.array(v) for k, v in arrays.items()}
    model.load_weights(list(weights.items()), strict=False)
