import sys
import threading
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_DIR / "DepthCode"), str(REPO_DIR / "YoloFire" / "jetson")]

from depth_estimator import MetricDepthEstimator, focal_length_per_image_width  # noqa: E402, F401
from fire_detector import Detection, FireDetector, FrameResult  # noqa: E402

DEFAULT_FIRE_MODEL = REPO_DIR / "YoloFire" / "export" / "fire.pt"


def distance_meters(detection: Detection, depth_meters: np.ndarray) -> float:
    # The middle of the box avoids background pixels around the candle's edges.
    x1, y1, x2, y2 = detection.box
    quarter_width, quarter_height = (x2 - x1) / 4, (y2 - y1) / 4
    center = depth_meters[int(y1 + quarter_height):int(y2 - quarter_height) + 1,
                          int(x1 + quarter_width):int(x2 - quarter_width) + 1]
    return float(np.median(center))


class FireDepth:
    """YOLO runs on every frame. Depth is about 3x slower, so it runs in its own thread on the newest frame,
    and distances come from the newest finished depth map."""

    def __init__(self, fire_model: str, confidence: float, horizontal_fov_degrees: float,
                 processing_resolution: int) -> None:
        self.fire_detector = FireDetector(fire_model, conf=confidence)
        self.depth_estimator = MetricDepthEstimator(horizontal_fov_degrees, processing_resolution)
        self.newest_depth: tuple[int, np.ndarray] | None = None
        self._depth_request: tuple[np.ndarray, int, float | None] | None = None
        self._depth_requested = threading.Event()
        threading.Thread(target=self._estimate_depth_forever, daemon=True).start()

    def measure(self, frame_bgr: np.ndarray, frame_id: int,
                focal_length_per_width: float | None = None) -> tuple[FrameResult, list[float]]:
        """Distances are NaN until the first depth map is ready."""
        self._depth_request = (frame_bgr, frame_id, focal_length_per_width)
        self._depth_requested.set()
        fires = self.fire_detector.detect(frame_bgr, frame_id)
        if self.newest_depth is None:
            return fires, [float("nan")] * len(fires.detections)
        depth_meters = self.newest_depth[1]
        return fires, [distance_meters(fire, depth_meters) for fire in fires.detections]

    def _estimate_depth_forever(self) -> None:
        while self._depth_requested.wait():
            self._depth_requested.clear()
            frame_bgr, frame_id, focal_length_per_width = self._depth_request
            self.newest_depth = frame_id, self.depth_estimator.estimate_meters(frame_bgr, focal_length_per_width)
