import json

import cv2
import numpy as np

from src.config import AppConfig
from src.detection.base import PersonDetector
from src.pipeline import TrackingPipeline
from src.schemas import Box, PersonDetection


class FakeDetector(PersonDetector):
    def detect(self, frame):
        # The white rectangle is the intended target.
        mask = cv2.inRange(frame, np.array([240, 240, 240]), np.array([255, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return [PersonDetection(Box(x, y, w, h), 0.99)]


def test_pipeline_writes_video_and_jsonl(tmp_path):
    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    log = tmp_path / "output.jsonl"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    assert writer.isOpened()
    for index in range(5):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (40 + index * 3, 40), (70 + index * 3, 120), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    config = AppConfig.defaults()
    config.crop.output_width = 90
    config.crop.output_height = 160
    summary = TrackingPipeline(FakeDetector(), config).run(
        source, output, log, Box(40, 40, 31, 81)
    )

    assert summary["frames"] == 5
    assert output.exists() and output.stat().st_size > 0
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    # Integer source pixels may round the crop by < 1 px; encoded output is exact 9:16.
    assert all(abs(row["crop_w"] / row["crop_h"] - 0.5625) < 0.01 for row in rows)
