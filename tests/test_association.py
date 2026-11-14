import numpy as np

from src.config import TrackingConfig
from src.schemas import Box, PersonDetection
from src.tracking.appearance import HistogramAppearance
from src.tracking.association import TargetAssociator, iou


def test_iou_identity():
    box = Box(10, 20, 30, 40)
    assert iou(box, box) == 1.0


def test_association_rejects_impossible_jump():
    config = TrackingConfig(max_center_distance_ratio=1.0)
    associator = TargetAssociator(config, HistogramAppearance())
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    result = associator.choose(
        frame,
        [PersonDetection(Box(700, 100, 40, 100), 0.9)],
        Box(100, 100, 40, 100),
        None,
        0,
    )
    assert result is None
