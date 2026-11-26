"""MoGe-2 metric depth from ONNX, so the same code runs on the laptop and on the Jetson's Python 3.8.
Takes Microsoft's export (any image size) or export_moge.py's fixed-size one, which TensorRT can run.
The post-processing follows MoGe's MoGeModel.infer()."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "moge-2-vits.onnx"
MODEL_URL = "https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx/resolve/main/model.onnx"
SHIFT_FIT_SIZE = (64, 64)
NUM_TOKENS = 1200  # MoGe-2 was trained with 1200-3600; fewer is faster


def view_plane_uv(width: int, height: int) -> np.ndarray:
    """Pixel centers scaled so the half-diagonal of the image is 1, as MoGe's points are."""
    aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5
    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=np.float32)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=np.float32)
    return np.stack(np.meshgrid(u, v, indexing="xy"), axis=-1)


def fit_focal_and_shift(points: np.ndarray, valid: np.ndarray, focal: float | None = None) -> tuple[float, float]:
    """MoGe predicts points up to an unknown z shift. Solves focal * xy / (z + shift) = uv, fitting focal too
    when it isn't given. Focal is relative to the image half-diagonal."""
    height, width = points.shape[:2]
    keep = cv2.resize(valid.astype(np.uint8), SHIFT_FIT_SIZE, interpolation=cv2.INTER_NEAREST).astype(bool)
    xyz = cv2.resize(points, SHIFT_FIT_SIZE, interpolation=cv2.INTER_NEAREST)[keep]
    uv = cv2.resize(view_plane_uv(width, height), SHIFT_FIT_SIZE, interpolation=cv2.INTER_NEAREST)[keep]
    if len(xyz) < 2:
        return (focal or 1.0), 0.0
    xy, z = xyz[:, :2], xyz[:, 2]

    def best_focal(projected: np.ndarray) -> float:
        return focal if focal is not None else (projected * uv).sum() / np.square(projected).sum()

    def residuals(shift: np.ndarray) -> np.ndarray:
        projected = xy / (z + shift)[:, None]
        return (best_focal(projected) * projected - uv).ravel()

    shift = float(least_squares(residuals, x0=0, ftol=1e-3, method="lm").x[0])
    return float(best_focal(xy / (z + shift)[:, None])), shift


class MogeDepthEstimator:
    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        import onnx
        import onnxruntime

        if not model_path.exists():
            print(f"Downloading MoGe-2 (140 MB) to {model_path}")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, f"{model_path}.part")
            Path(f"{model_path}.part").rename(model_path)
        if hasattr(onnxruntime, "preload_dlls"):
            onnxruntime.preload_dlls()  # loads the CUDA libraries pip installed with onnxruntime-gpu[cuda,cudnn]
        model_path = str(model_path)
        inputs = onnx.load(model_path).graph.input
        fixed_size = len(inputs) == 1
        # TensorRT needs the fixed-size export; the other one has shapes that depend on num_tokens.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if fixed_size:
            providers.insert(0, ("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                                               "trt_engine_cache_path": str(Path(model_path).parent)}))
        self._session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.providers = self._session.get_providers()
        self._input_size = tuple(d.dim_value for d in inputs[0].type.tensor_type.shape.dim[2:]) if fixed_size else None
        self.focal_length_per_width: float | None = None

    def estimate_meters(self, frame_bgr: np.ndarray, focal_length_per_width: float | None = None) -> np.ndarray:
        """Returns depth at the frame's size, NaN where MoGe has no valid depth. Pass fx / image_width from a
        calibrated camera; without it MoGe estimates the focal length, and focal_length_per_width holds it."""
        frame_height, frame_width = frame_bgr.shape[:2]
        height, width = self._input_size or (frame_height, frame_width)
        if abs(width / height - frame_width / frame_height) > 0.01:
            raise ValueError(f"The depth model takes {width}x{height} images, not {frame_width}x{frame_height}.")
        frame_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (width, height)), cv2.COLOR_BGR2RGB)
        feeds = {"image": (frame_rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None]}
        if self._input_size is None:
            feeds["num_tokens"] = np.array(NUM_TOKENS, dtype=np.int64)
        points, _, mask, metric_scale = self._session.run(None, feeds)
        points, valid, metric_scale = points[0], mask[0] > 0.5, float(metric_scale[0])

        # MoGe's focal is relative to the half-diagonal; ours is relative to the width.
        diagonal_per_width = (1 + (height / width) ** 2) ** 0.5
        given_focal = None if focal_length_per_width is None else 2 * focal_length_per_width / diagonal_per_width
        focal, shift = fit_focal_and_shift(points, valid, given_focal)
        self.focal_length_per_width = focal * diagonal_per_width / 2
        depth_meters = (points[..., 2] + shift) * metric_scale
        depth_meters[~valid | (depth_meters <= 0)] = np.nan
        return cv2.resize(depth_meters, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
