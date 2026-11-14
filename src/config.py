from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class DetectorConfig:
    model_url: str = "https://tfhub.dev/google/movenet/multipose/lightning/1"
    input_size: int = 256
    confidence_threshold: float = 0.20


@dataclass
class TrackingConfig:
    min_match_score: float = 0.43
    max_missing_frames: int = 20
    position_weight: float = 0.35
    appearance_weight: float = 0.35
    iou_weight: float = 0.20
    size_weight: float = 0.10
    max_center_distance_ratio: float = 2.5
    appearance_update_threshold: float = 0.72
    appearance_update_rate: float = 0.08
    velocity_update_rate: float = 0.35


@dataclass
class CropConfig:
    output_width: int = 1080
    output_height: int = 1920
    smoothing_alpha: float = 0.22
    reacquired_alpha: float = 0.42
    max_speed_ratio: float = 0.045
    target_vertical_position: float = 0.52


@dataclass
class AppConfig:
    detector: DetectorConfig
    tracking: TrackingConfig
    crop: CropConfig

    @classmethod
    def defaults(cls) -> "AppConfig":
        return cls(DetectorConfig(), TrackingConfig(), CropConfig())


def _merge_dataclass(instance: T, values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(instance)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown config keys for {type(instance).__name__}: {sorted(unknown)}")
    for key, value in values.items():
        setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None) -> AppConfig:
    config = AppConfig.defaults()
    if path is None:
        return config
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _merge_dataclass(config.detector, raw.get("detector", {}))
    _merge_dataclass(config.tracking, raw.get("tracking", {}))
    _merge_dataclass(config.crop, raw.get("crop", {}))
    return config
