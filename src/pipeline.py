from __future__ import annotations

import time
from pathlib import Path

import cv2

from src.config import AppConfig
from src.cropping.planner import CropPlanner
from src.cropping.smoother import CropSmoother
from src.detection.base import PersonDetector
from src.io.logging import JsonlLogger
from src.io.video import VideoReader, VideoWriter
from src.schemas import Box, PersonDetection, TrackResult
from src.tracking.tracker import TargetTracker


class TrackingPipeline:
    def __init__(self, detector: PersonDetector, config: AppConfig):
        self.detector = detector
        self.config = config
        self.tracker = TargetTracker(config.tracking)
        self.planner = CropPlanner(config.crop)
        self.smoother = CropSmoother(config.crop)

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        log_path: str | Path,
        initial_box: Box,
        debug_path: str | Path | None = None,
    ) -> dict[str, float | int | str]:
        reader = VideoReader(input_path)
        writer = VideoWriter(
            output_path,
            reader.fps,
            (self.config.crop.output_width, self.config.crop.output_height),
        )
        logger = JsonlLogger(log_path)
        debug_writer = None
        if debug_path:
            debug_writer = VideoWriter(debug_path, reader.fps, (reader.width, reader.height))

        started = time.perf_counter()
        frames = 0
        try:
            for frame_index, frame in reader:
                frame_started = time.perf_counter()
                if frame_index == 0:
                    self.tracker.initialize(frame, initial_box)

                detections = self.detector.detect(frame)
                track = self.tracker.update(frame_index, frame, detections)
                desired = self.planner.plan(frame.shape, track.box)
                crop = self.smoother.update(desired, track.status)
                writer.write_crop(frame, crop)

                processing_ms = (time.perf_counter() - frame_started) * 1000
                logger.write(self._record(frame_index, reader.fps, track, crop, detections, processing_ms))
                if debug_writer is not None:
                    debug_writer.writer.write(self._draw_debug(frame.copy(), track, crop, detections))
                frames += 1
        finally:
            reader.close()
            writer.close()
            logger.close()
            if debug_writer is not None:
                debug_writer.close()

        elapsed = time.perf_counter() - started
        duration = frames / reader.fps if reader.fps else 0.0
        return {
            "frames": frames,
            "video_seconds": round(duration, 3),
            "processing_seconds": round(elapsed, 3),
            "seconds_per_video_minute": round(elapsed / max(duration, 1e-9) * 60, 3),
            "output": str(output_path),
            "log": str(log_path),
        }

    @staticmethod
    def _record(frame_index, fps, track, crop, detections, processing_ms):
        return {
            "frame_index": frame_index,
            "timestamp_s": round(frame_index / fps, 3),
            "track_status": track.status.value,
            "track_confidence": round(track.confidence, 6),
            "target_box": track.box.to_dict() if track.box else None,
            "crop": crop.to_dict(),
            "crop_x": crop.x,
            "crop_y": crop.y,
            "crop_w": crop.w,
            "crop_h": crop.h,
            "matched_detection_index": track.matched_detection_index,
            "match_components": track.match_components,
            "detection_count": len(detections),
            "processing_ms": round(processing_ms, 3),
        }

    @staticmethod
    def _draw_debug(frame, track: TrackResult, crop, detections: list[PersonDetection]):
        for index, detection in enumerate(detections):
            box = detection.box
            color = (0, 255, 255) if index != track.matched_detection_index else (0, 255, 0)
            cv2.rectangle(frame, (round(box.x), round(box.y)),
                          (round(box.x + box.w), round(box.y + box.h)), color, 2)
            cv2.putText(frame, f"{index}:{detection.confidence:.2f}",
                        (round(box.x), max(20, round(box.y) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.rectangle(frame, (crop.x, crop.y), (crop.x + crop.w, crop.y + crop.h), (255, 0, 255), 2)
        cv2.putText(frame, f"{track.status.value} {track.confidence:.2f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame
