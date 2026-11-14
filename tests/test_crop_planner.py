from src.config import CropConfig
from src.cropping.planner import CropPlanner
from src.cropping.smoother import CropSmoother
from src.schemas import Box, CropWindow, TrackStatus


def test_landscape_crop_is_nine_sixteen_and_in_bounds():
    crop = CropPlanner(CropConfig()).plan((1080, 1920, 3), Box(1800, 200, 100, 300))
    assert crop.w == round(crop.h * 9 / 16)
    assert crop.x + crop.w <= 1920
    assert crop.y + crop.h <= 1080


def test_smoother_limits_jump():
    config = CropConfig(smoothing_alpha=1.0, max_speed_ratio=0.1)
    smoother = CropSmoother(config)
    smoother.update(CropWindow(0, 0, 100, 178), TrackStatus.TRACKED)
    result = smoother.update(CropWindow(80, 0, 100, 178), TrackStatus.TRACKED)
    assert result.x == 10
