from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.config import TrackingConfig
from src.schemas import Box, PersonDetection
from src.tracking.appearance import HistogramAppearance


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class Match:
    index: int
    detection: PersonDetection
    score: float
    appearance: np.ndarray | None
    components: dict[str, float]


class TargetAssociator:
    def __init__(self, config: TrackingConfig, appearance: HistogramAppearance):
        self.config = config
        self.appearance = appearance

    def choose(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
        predicted_box: Box,
        reference_appearance: np.ndarray | None,
        missing_frames: int,
    ) -> Match | None:
        best: Match | None = None
        # Relax the motion gate after a loss, but never remove it completely.
        gate = self.config.max_center_distance_ratio * min(3.0, 1.0 + missing_frames / 10.0)
        normalizer = max(predicted_box.w, predicted_box.h, 1.0)

        for index, detection in enumerate(detections):
            dx = detection.box.cx - predicted_box.cx
            dy = detection.box.cy - predicted_box.cy
            distance_ratio = math.hypot(dx, dy) / normalizer
            if distance_ratio > gate:
                continue

            area_ratio = detection.box.area / max(predicted_box.area, 1.0)
            if area_ratio <= 0:
                continue

            position_score = math.exp(-distance_ratio)
            overlap_score = iou(detection.box, predicted_box)
            size_score = math.exp(-abs(math.log(area_ratio)))
            descriptor = self.appearance.extract(frame, detection.box)
            appearance_score = self.appearance.similarity(reference_appearance, descriptor)

            components = {
                "position": position_score,
                "appearance": appearance_score,
                "iou": overlap_score,
                "size": size_score,
            }
            score = (
                self.config.position_weight * position_score
                + self.config.appearance_weight * appearance_score
                + self.config.iou_weight * overlap_score
                + self.config.size_weight * size_score
            )
            candidate = Match(index, detection, score, descriptor, components)
            if best is None or candidate.score > best.score:
                best = candidate

        if best is None or best.score < self.config.min_match_score:
            return None
        return best
