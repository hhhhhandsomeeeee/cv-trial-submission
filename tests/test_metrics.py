from src.evaluation.metrics import crop_center_error, target_center_outside_crop
from src.schemas import Box, CropWindow


def test_center_error_zero_when_aligned():
    crop = CropWindow(100, 0, 200, 356)
    target = Box(190, 50, 20, 100)
    assert crop_center_error(crop, target) == 0
    assert not target_center_outside_crop(crop, target)


def test_center_error_half_at_edge():
    crop = CropWindow(100, 0, 200, 356)
    target = Box(295, 50, 10, 100)
    assert crop_center_error(crop, target) == 0.5
