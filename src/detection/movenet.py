from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from src.detection.base import PersonDetector
from src.schemas import Box, PersonDetection


class MoveNetMultiPoseDetector(PersonDetector):
    """Thin adapter around TF-Hub MoveNet MultiPose Lightning."""

    def __init__(self, model_url: str, input_size: int = 256, confidence_threshold: float = 0.2):
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.model = hub.load(model_url).signatures["serving_default"]

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = tf.image.resize_with_pad(rgb[None], self.input_size, self.input_size)
        output = self.model(tf.cast(tensor, tf.int32))["output_0"].numpy()[0]

        scale = self.input_size / max(frame_h, frame_w)
        scaled_w, scaled_h = frame_w * scale, frame_h * scale
        pad_x = (self.input_size - scaled_w) / 2
        pad_y = (self.input_size - scaled_h) / 2

        detections: list[PersonDetection] = []
        for person in output:
            score = float(person[55])
            if score < self.confidence_threshold:
                continue

            keypoints_raw = person[:51].reshape(17, 3)
            ymin, xmin, ymax, xmax = person[51:55]
            x1 = (float(xmin) * self.input_size - pad_x) / scale
            y1 = (float(ymin) * self.input_size - pad_y) / scale
            x2 = (float(xmax) * self.input_size - pad_x) / scale
            y2 = (float(ymax) * self.input_size - pad_y) / scale
            box = Box(x1, y1, x2 - x1, y2 - y1).clipped(frame_w, frame_h)
            if box.w < 2 or box.h < 2:
                continue

            keypoints: list[list[float]] = []
            for ky, kx, ks in keypoints_raw:
                px = (float(kx) * self.input_size - pad_x) / scale
                py = (float(ky) * self.input_size - pad_y) / scale
                keypoints.append([px, py, float(ks)])
            detections.append(PersonDetection(box, score, keypoints))

        return detections
