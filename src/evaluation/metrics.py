from __future__ import annotations

import math
from statistics import mean

from src.schemas import Box, CropWindow


def crop_center_error(crop: CropWindow, target: Box) -> float:
    return abs(crop.cx - target.cx) / max(crop.w, 1)


def target_center_outside_crop(crop: CropWindow, target: Box) -> bool:
    return not (crop.x <= target.cx <= crop.x + crop.w)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def crop_jitter(xs: list[float], crop_widths: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    velocities = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    accelerations = [abs(velocities[i] - velocities[i - 1]) for i in range(1, len(velocities))]
    scale = mean(crop_widths) if crop_widths else 1.0
    return mean(accelerations) / max(scale, 1.0)


def summarise(errors: list[float], outside: list[bool], xs: list[float], widths: list[float]):
    if not errors:
        raise ValueError("no visible reference frames overlap this log")
    return {
        "evaluated_frames": len(errors),
        "mean_center_error": round(mean(errors), 6),
        "p95_center_error": round(percentile(errors, 0.95), 6),
        "max_center_error": round(max(errors), 6),
        "target_center_outside_rate": round(sum(outside) / len(outside), 6),
        "crop_jitter_normalized": round(crop_jitter(xs, widths), 6),
    }
