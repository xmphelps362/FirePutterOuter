import math
import os

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

MODEL_NAME = "depth-anything/DA3METRIC-LARGE"
MODEL_FOCAL_LENGTH_PIXELS = 300.0


def focal_length_per_image_width(horizontal_fov_degrees: float) -> float:
    return 0.5 / math.tan(math.radians(horizontal_fov_degrees) / 2)


class MetricDepthEstimator:
    def __init__(self, horizontal_fov_degrees: float, processing_resolution: int) -> None:
        self._model = DepthAnything3.from_pretrained(MODEL_NAME).to("cuda")
        self._model.model.backbone.to(torch.bfloat16)
        self._default_focal_length_per_image_width = focal_length_per_image_width(horizontal_fov_degrees)
        self._processing_resolution = processing_resolution

    def estimate_meters(self, frame_bgr: np.ndarray, focal_length_per_width: float | None = None) -> np.ndarray:
        """Returns depth at the frame's size. Pass fx / image_width from a calibrated camera to override the FOV."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        model_depth = self._model.inference([frame_rgb], process_res=self._processing_resolution).depth[0]
        focal_length_pixels = (focal_length_per_width or self._default_focal_length_per_image_width) * model_depth.shape[1]
        depth_meters = model_depth * focal_length_pixels / MODEL_FOCAL_LENGTH_PIXELS
        return cv2.resize(depth_meters, (frame_bgr.shape[1], frame_bgr.shape[0]))


def to_depth_millimeters(depth_meters: np.ndarray) -> np.ndarray:
    return np.clip(np.round(depth_meters * 1000), 0, np.iinfo(np.uint16).max).astype(np.uint16)
