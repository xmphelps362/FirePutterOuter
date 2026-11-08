import json

import numpy as np
import pytest

from fire_detector import Detection, FireDetector, FrameResult


def test_to_json_schema():
    r = FrameResult(7, 1.5, 1280, 720, [Detection("lit_candle", 0.87654, (1.24, 2.0, 11.0, 22.0))])
    d = json.loads(r.to_json())
    assert set(d) == {"frame_id", "timestamp", "width", "height", "detections"}
    det = d["detections"][0]
    assert det == {"class": "lit_candle", "confidence": 0.877, "box": [1.2, 2.0, 11.0, 22.0], "center": [6.1, 12.0]}


def test_empty_result_json():
    assert json.loads(FrameResult(0, 0.0, 640, 480, []).to_json())["detections"] == []


@pytest.fixture(scope="module")
def det():
    return FireDetector("yolo26n.pt", device=0)


def test_detect_blank_frame(det):
    r = det.detect(np.zeros((720, 1280, 3), np.uint8), frame_id=42)
    assert (r.frame_id, r.width, r.height) == (42, 1280, 720)
    assert r.timestamp > 0


def test_detect_any_size_and_rgb(det):
    frame = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    r = det.detect(frame, frame_id=1, timestamp=123.0, color="rgb")
    assert (r.width, r.height, r.timestamp) == (640, 480, 123.0)
    for d in r.detections:
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 <= x2 <= 640 and 0 <= y1 <= y2 <= 480


@pytest.mark.parametrize("frame,color", [(np.zeros((10, 10, 3), np.uint8), "hsv"),
                                         (np.zeros((720, 1280), np.uint8), "rgb")])
def test_bad_input_rejected(det, frame, color):
    with pytest.raises(ValueError):
        det.detect(frame, 0, color=color)
