from __future__ import annotations

import json
import math
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_DIR / "DepthCode"), str(REPO_DIR / "YoloFire" / "jetson")]

from detect_camera import DEFAULT_MODEL as DEFAULT_FIRE_MODEL  # noqa: E402, F401
from fire_detector import Detection, FireDetector, FrameResult  # noqa: E402
from moge_depth import MogeDepthEstimator  # noqa: E402


def distance_meters(detection: Detection, depth_meters: np.ndarray) -> float:
    # The middle of the box avoids background pixels around the candle's edges.
    x1, y1, x2, y2 = detection.box
    quarter_width, quarter_height = (x2 - x1) / 4, (y2 - y1) / 4
    center = depth_meters[int(y1 + quarter_height):int(y2 - quarter_height) + 1,
                          int(x1 + quarter_width):int(x2 - quarter_width) + 1]
    center = center[np.isfinite(center)]
    return float(np.median(center)) if center.size else math.nan


def bearing(detection: Detection, image_width: int) -> float:
    """-1 at the left edge of the image, 1 at the right edge."""
    return 2 * detection.center[0] / image_width - 1


def camera_point(detection: Detection, distance: float, fx: float, fy: float, cx: float, cy: float) -> tuple:
    """The candle's center in the camera optical frame: z forward, x right, y down."""
    u, v = detection.center
    return (u - cx) * distance / fx, (v - cy) * distance / fy, distance


def nearest_lit_candle(detections: list, distances: list) -> int | None:
    """Index of the closest lit candle, or the most confident one while there is no depth yet."""
    lit = [i for i, detection in enumerate(detections) if detection.cls == "lit_candle"]
    if not lit:
        return None
    measured = [i for i in lit if math.isfinite(distances[i])]
    if measured:
        return min(measured, key=lambda i: distances[i])
    return max(lit, key=lambda i: detections[i].confidence)


def detections_json(fires: FrameResult, distances: list, positions: list, frame: str) -> str:
    """FrameResult.to_json plus each candle's bearing, distance and position in `frame` (null when unknown)."""
    result = json.loads(fires.to_json())
    result["frame"] = frame
    for detection, fire, distance, position in zip(result["detections"], fires.detections, distances, positions):
        detection["bearing"] = round(bearing(fire, fires.width), 3)
        detection["distance"] = round(distance, 3) if math.isfinite(distance) else None
        detection["position"] = None if position is None else [round(value, 3) for value in position]
    return json.dumps(result, separators=(",", ":"))


class FireDepth:
    """YOLO runs on every frame. Depth is slower, so it runs in its own thread on the newest frame,
    and distances come from the newest finished depth map."""

    def __init__(self, fire_model: str, confidence: float) -> None:
        self.fire_detector = FireDetector(fire_model, conf=confidence)
        self.depth_estimator = MogeDepthEstimator()
        self.newest_depth: tuple[int, np.ndarray] | None = None
        self._depth_request: tuple[np.ndarray, int, float | None] | None = None
        self._depth_requested = threading.Event()
        threading.Thread(target=self._estimate_depth_forever, daemon=True).start()

    def measure(self, frame_bgr: np.ndarray, frame_id: int,
                focal_length_per_width: float | None = None) -> tuple[FrameResult, list]:
        """Distances are NaN until the first depth map is ready."""
        self._depth_request = (frame_bgr, frame_id, focal_length_per_width)
        self._depth_requested.set()
        fires = self.fire_detector.detect(frame_bgr, frame_id)
        if self.newest_depth is None:
            return fires, [math.nan] * len(fires.detections)
        depth_meters = self.newest_depth[1]
        return fires, [distance_meters(fire, depth_meters) for fire in fires.detections]

    def _estimate_depth_forever(self) -> None:
        while self._depth_requested.wait():
            self._depth_requested.clear()
            frame_bgr, frame_id, focal_length_per_width = self._depth_request
            try:
                self.newest_depth = frame_id, self.depth_estimator.estimate_meters(frame_bgr, focal_length_per_width)
            except Exception:
                traceback.print_exc()
