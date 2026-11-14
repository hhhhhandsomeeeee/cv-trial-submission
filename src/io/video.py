from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.schemas import CropWindow


class VideoReader:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield index, frame
            index += 1

    def close(self) -> None:
        self.cap.release()


class VideoWriter:
    def __init__(self, path: str | Path, fps: float, output_size: tuple[int, int]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.output_size = output_size
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size
        )
        if not self.writer.isOpened():
            raise ValueError(f"cannot create output video: {path}")

    def write_crop(self, frame: np.ndarray, crop: CropWindow) -> None:
        region = frame[crop.y:crop.y + crop.h, crop.x:crop.x + crop.w]
        if region.size == 0:
            raise ValueError(f"empty crop: {crop}")
        self.writer.write(cv2.resize(region, self.output_size, interpolation=cv2.INTER_LINEAR))

    def close(self) -> None:
        self.writer.release()
