"""python -m pytest FireDepth/tests (needs numpy, scipy and opencv, not the models)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fire_depth import (Detection, FrameResult, bearing, camera_point, detections_json,  # noqa: E402
                        distance_meters, nearest_lit_candle)
from moge_depth import fit_focal_and_shift, view_plane_uv  # noqa: E402


def test_distance_ignores_invalid_depth():
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    depth[40:60, 40:50] = np.nan
    assert distance_meters(Detection("lit_candle", 0.9, (30, 30, 70, 70)), depth) == pytest.approx(2.0)
    assert math.isnan(distance_meters(Detection("lit_candle", 0.9, (30, 30, 70, 70)), np.full((100, 100), np.nan)))


def test_bearing_and_camera_point():
    candle = Detection("lit_candle", 0.9, (300, 220, 340, 260))  # center (320, 240)
    assert bearing(candle, 640) == pytest.approx(0.0)
    assert bearing(Detection("lit_candle", 0.9, (0, 0, 0, 0)), 640) == pytest.approx(-1.0)
    assert camera_point(candle, 2.0, 500, 500, 320, 240) == pytest.approx((0.0, 0.0, 2.0))
    assert camera_point(Detection("lit_candle", 0.9, (560, 240, 580, 240)), 2.0, 500, 500, 320, 240)[0] == pytest.approx(1.0)


def test_nearest_lit_candle():
    unlit = Detection("unlit_candle", 0.99, (0, 0, 10, 10))
    far = Detection("lit_candle", 0.95, (0, 0, 10, 10))
    near = Detection("lit_candle", 0.5, (0, 0, 10, 10))
    assert nearest_lit_candle([unlit, far, near], [0.5, 3.0, 1.0]) == 2
    assert nearest_lit_candle([unlit, far, near], [0.5, math.nan, math.nan]) == 1
    assert nearest_lit_candle([unlit], [1.0]) is None


def test_detections_json():
    fires = FrameResult(7, 1.0, 640, 480, [Detection("lit_candle", 0.9, (300, 220, 340, 260)),
                                           Detection("unlit_candle", 0.8, (0, 0, 10, 10))])
    result = json.loads(detections_json(fires, [1.23456, math.nan], [(1.2, 0.0, 0.01), None], "base_link"))
    assert result["frame"] == "base_link" and result["frame_id"] == 7
    assert result["detections"][0]["distance"] == 1.235 and result["detections"][0]["position"] == [1.2, 0.0, 0.01]
    assert result["detections"][1]["distance"] is None and result["detections"][1]["position"] is None


@pytest.mark.parametrize("given_focal", [True, False])
def test_fit_recovers_shift_and_focal(given_focal):
    height, width, focal, shift = 120, 160, 1.1, 0.7
    uv = view_plane_uv(width, height)
    depth = 2 + np.random.default_rng(0).random((height, width), dtype=np.float32)
    points = np.dstack([uv * depth[..., None] / focal, depth - shift]).astype(np.float32)
    valid = np.ones((height, width), dtype=bool)
    fitted_focal, fitted_shift = fit_focal_and_shift(points, valid, focal if given_focal else None)
    assert fitted_shift == pytest.approx(shift, abs=1e-3)
    assert fitted_focal == pytest.approx(focal, rel=1e-3)
