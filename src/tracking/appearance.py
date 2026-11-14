from __future__ import annotations

import cv2
import numpy as np

from src.schemas import Box


class HistogramAppearance:
    """Cheap HSV descriptor; intentionally CPU-friendly and training-free."""

    def extract(self, frame: np.ndarray, box: Box) -> np.ndarray | None:
        frame_h, frame_w = frame.shape[:2]
        box = box.clipped(frame_w, frame_h)
        x1, y1 = int(box.x), int(box.y)
        x2, y2 = int(box.x + box.w), int(box.y + box.h)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Focus on the torso to reduce court/background and leg-motion noise.
        h = hsv.shape[0]
        torso = hsv[int(h * 0.15):max(int(h * 0.7), int(h * 0.15) + 1)]
        hist = cv2.calcHist([torso], [0, 1], None, [24, 16], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten().astype(np.float32)

    @staticmethod
    def similarity(reference: np.ndarray | None, candidate: np.ndarray | None) -> float:
        if reference is None or candidate is None:
            return 0.5
        correlation = cv2.compareHist(reference, candidate, cv2.HISTCMP_CORREL)
        return float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))
