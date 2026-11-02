import math
import os

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

MODEL_NAME = "depth-anything/DA3METRIC-LARGE"
MODEL_FOCAL_LENGTH_PIXELS = 300.0


class MetricDepthEstimator:
    def __init__(self, horizontal_fov_degrees: float, processing_resolution: int) -> None:
        self._model = DepthAnything3.from_pretrained(MODEL_NAME).to("cuda")
        self._model.model.backbone.to(torch.bfloat16)
        self._focal_length_per_image_width = 0.5 / math.tan(math.radians(horizontal_fov_degrees) / 2)
        self._processing_resolution = processing_resolution

    def estimate_meters(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        model_depth = self._model.inference([frame_rgb], process_res=self._processing_resolution).depth[0]
        focal_length_pixels = self._focal_length_per_image_width * model_depth.shape[1]
        return model_depth * focal_length_pixels / MODEL_FOCAL_LENGTH_PIXELS


def to_depth_millimeters(depth_meters: np.ndarray) -> np.ndarray:
    return np.clip(np.round(depth_meters * 1000), 0, np.iinfo(np.uint16).max).astype(np.uint16)
