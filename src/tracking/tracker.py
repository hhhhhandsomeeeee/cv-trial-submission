from __future__ import annotations

import numpy as np

from src.config import TrackingConfig
from src.schemas import Box, PersonDetection, TrackResult, TrackStatus
from src.tracking.appearance import HistogramAppearance
from src.tracking.association import TargetAssociator


class TargetTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.appearance = HistogramAppearance()
        self.associator = TargetAssociator(config, self.appearance)
        self.box: Box | None = None
        self.reference_appearance: np.ndarray | None = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.missing_frames = 0

    def initialize(self, frame: np.ndarray, initial_box: Box) -> None:
        self.box = initial_box.clipped(frame.shape[1], frame.shape[0])
        self.reference_appearance = self.appearance.extract(frame, self.box)
        self.velocity[:] = 0
        self.missing_frames = 0

    def _predict(self) -> Box:
        if self.box is None:
            raise RuntimeError("tracker is not initialized")
        return Box(
            self.box.x + float(self.velocity[0]),
            self.box.y + float(self.velocity[1]),
            self.box.w,
            self.box.h,
        )

    def update(
        self,
        frame_index: int,
        frame: np.ndarray,
        detections: list[PersonDetection],
    ) -> TrackResult:
        predicted = self._predict().clipped(frame.shape[1], frame.shape[0])
        match = self.associator.choose(
            frame,
            detections,
            predicted,
            self.reference_appearance,
            self.missing_frames,
        )

        if match is None:
            self.missing_frames += 1
            self.box = predicted
            confidence = max(0.0, 1.0 - self.missing_frames / self.config.max_missing_frames)
            status = (
                TrackStatus.PREDICTED
                if self.missing_frames <= self.config.max_missing_frames
                else TrackStatus.LOST
            )
            if status is TrackStatus.LOST:
                self.velocity *= 0.8
            return TrackResult(frame_index, self.box, confidence, status)

        old_box = self.box
        was_missing = self.missing_frames > 0
        observed_velocity = np.array(
            [match.detection.box.cx - old_box.cx, match.detection.box.cy - old_box.cy],
            dtype=np.float32,
        )
        beta = self.config.velocity_update_rate
        self.velocity = (1.0 - beta) * self.velocity + beta * observed_velocity
        self.box = match.detection.box
        self.missing_frames = 0

        if match.score >= self.config.appearance_update_threshold and match.appearance is not None:
            if self.reference_appearance is None:
                self.reference_appearance = match.appearance
            else:
                rate = self.config.appearance_update_rate
                self.reference_appearance = cv_normalize(
                    (1.0 - rate) * self.reference_appearance + rate * match.appearance
                )

        return TrackResult(
            frame_index,
            self.box,
            match.score,
            TrackStatus.REACQUIRED if was_missing else TrackStatus.TRACKED,
            match.index,
            match.components,
        )


def cv_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector
