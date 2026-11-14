from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def clipped(self, frame_w: int, frame_h: int) -> "Box":
        x1 = min(max(self.x, 0.0), float(frame_w))
        y1 = min(max(self.y, 0.0), float(frame_h))
        x2 = min(max(self.x + self.w, x1), float(frame_w))
        y2 = min(max(self.y + self.h, y1), float(frame_h))
        return Box(x1, y1, x2 - x1, y2 - y1)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PersonDetection:
    box: Box
    confidence: float
    keypoints: list[list[float]] | None = None


class TrackStatus(str, Enum):
    INITIALIZED = "initialized"
    TRACKED = "tracked"
    PREDICTED = "predicted"
    LOST = "lost"
    REACQUIRED = "reacquired"


@dataclass
class TrackResult:
    frame_index: int
    box: Box | None
    confidence: float
    status: TrackStatus
    matched_detection_index: int | None = None
    match_components: dict[str, float] | None = None


@dataclass(frozen=True)
class CropWindow:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"cannot serialise {type(value)!r}")
