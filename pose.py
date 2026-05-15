"""
YOLOv8-nano-pose wrapper — COCO-17 keypoints, Python 3.13 compatible, no TF.

Model (~6MB) is downloaded automatically on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# COCO-17 schema (same order as MoveNet / standard COCO)
# ---------------------------------------------------------------------------

KP_NAMES: dict[int, str] = {
    0:  "nose",
    1:  "left_eye",     2:  "right_eye",
    3:  "left_ear",     4:  "right_ear",
    5:  "left_shoulder",6:  "right_shoulder",
    7:  "left_elbow",   8:  "right_elbow",
    9:  "left_wrist",   10: "right_wrist",
    11: "left_hip",     12: "right_hip",
    13: "left_knee",    14: "right_knee",
    15: "left_ankle",   16: "right_ankle",
}

N_KP = 17

SKELETON_EDGES: list[tuple[int, int]] = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Arms
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COACH_JOINTS: list[int] = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


# ---------------------------------------------------------------------------
# Keypoints
# ---------------------------------------------------------------------------

@dataclass
class Keypoints:
    """COCO-17 landmarks: xy_norm in [0,1], conf in [0,1]."""
    xy_norm: np.ndarray  # (17, 2)
    conf:    np.ndarray  # (17,)

    def visible(self, idx: int, threshold: float = 0.3) -> bool:
        return float(self.conf[idx]) >= threshold

    def to_list(self, threshold: float = 0.3) -> list[dict]:
        return [
            {
                "x": round(float(self.xy_norm[i, 0]), 4),
                "y": round(float(self.xy_norm[i, 1]), 4),
                "v": bool(self.conf[i] >= threshold),
            }
            for i in range(N_KP)
        ]

    def to_coach_text(self, threshold: float = 0.3) -> str:
        lines: list[str] = []
        for i in COACH_JOINTS:
            if self.conf[i] >= threshold:
                x = round(float(self.xy_norm[i, 0]), 2)
                y = round(float(self.xy_norm[i, 1]), 2)
                lines.append(f"  {KP_NAMES[i]}: ({x}, {y})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class YOLOPoseInference:
    """
    YOLOv8-nano-pose — lazy-initialised in the worker thread.
    Not thread-safe; use one instance per session.

    `model` can be "yolov8n-pose.pt" (fast) or "yolov8s-pose.pt" (accurate).
    """

    def __init__(self, model: str = "yolov8n-pose.pt"):
        self._model_name = model
        self._yolo = None

    def _init(self) -> None:
        from ultralytics import YOLO  # noqa: PLC0415
        self._yolo = YOLO(self._model_name)

    def predict(self, bgr_frame: np.ndarray) -> Optional[Keypoints]:
        if self._yolo is None:
            self._init()

        h, w = bgr_frame.shape[:2]
        results = self._yolo(bgr_frame, verbose=False, imgsz=640)

        for result in results:
            if result.keypoints is None or len(result.keypoints) == 0:
                continue
            # Pick the person with the highest detection confidence
            best = 0
            if len(result.keypoints) > 1:
                best = int(result.boxes.conf.argmax())

            kp_xy   = result.keypoints.xy[best].cpu().numpy()    # (17, 2) pixels
            kp_conf = result.keypoints.conf[best].cpu().numpy()   # (17,)

            xy_norm = kp_xy / np.array([w, h], dtype=np.float32)
            return Keypoints(xy_norm=xy_norm, conf=kp_conf)

        return None

    def close(self) -> None:
        self._yolo = None
