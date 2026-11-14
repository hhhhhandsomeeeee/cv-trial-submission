from __future__ import annotations

from src.config import CropConfig
from src.schemas import Box, CropWindow


class CropPlanner:
    def __init__(self, config: CropConfig):
        self.config = config

    def plan(self, frame_shape: tuple[int, ...], target: Box | None) -> CropWindow:
        frame_h, frame_w = frame_shape[:2]
        aspect = 9 / 16
        if frame_w / frame_h >= aspect:
            crop_h = frame_h
            crop_w = min(frame_w, round(crop_h * aspect))
        else:
            crop_w = frame_w
            crop_h = min(frame_h, round(crop_w / aspect))

        if target is None:
            cx, cy = frame_w / 2, frame_h / 2
        else:
            cx = target.cx
            # Put the player's centre slightly below the vertical centre.
            cy = target.cy - (self.config.target_vertical_position - 0.5) * crop_h

        x = round(max(0, min(cx - crop_w / 2, frame_w - crop_w)))
        y = round(max(0, min(cy - crop_h / 2, frame_h - crop_h)))
        return CropWindow(x, y, crop_w, crop_h)
