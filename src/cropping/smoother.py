from __future__ import annotations

import numpy as np

from src.config import CropConfig
from src.schemas import CropWindow, TrackStatus


class CropSmoother:
    def __init__(self, config: CropConfig):
        self.config = config
        self.previous: CropWindow | None = None

    def update(self, desired: CropWindow, status: TrackStatus) -> CropWindow:
        if self.previous is None:
            self.previous = desired
            return desired

        alpha = (
            self.config.reacquired_alpha
            if status is TrackStatus.REACQUIRED
            else self.config.smoothing_alpha
        )
        max_step = max(1.0, desired.w * self.config.max_speed_ratio)

        x = self._axis(self.previous.x, desired.x, alpha, max_step)
        y = self._axis(self.previous.y, desired.y, alpha, max_step)
        self.previous = CropWindow(round(x), round(y), desired.w, desired.h)
        return self.previous

    @staticmethod
    def _axis(previous: float, desired: float, alpha: float, max_step: float) -> float:
        smoothed = alpha * desired + (1.0 - alpha) * previous
        return previous + float(np.clip(smoothed - previous, -max_step, max_step))
