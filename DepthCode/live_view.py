import argparse
import functools
import math
import time

import cv2
import numpy as np

from moge_depth import MogeDepthEstimator
from windows_webcam import WindowsWebcam

ESCAPE_KEY = 27
SAVE_KEY = ord("s")
SCALE_BAR_WIDTH = 25
SCALE_LABEL_WIDTH = 45


def to_depth_millimeters(depth_meters: np.ndarray) -> np.ndarray:
    """ROS 16UC1 convention: 0 means no depth."""
    millimeters = np.round(np.nan_to_num(depth_meters, nan=0.0) * 1000)
    return np.clip(millimeters, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def depth_heatmap(depth_meters: np.ndarray, max_display_meters: float) -> np.ndarray:
    nearness = 1.0 - np.clip(np.nan_to_num(depth_meters, nan=max_display_meters) / max_display_meters, 0.0, 1.0)
    return cv2.applyColorMap((nearness * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


@functools.cache
def color_scale(height: int, max_display_meters: float) -> np.ndarray:
    meters_per_row = np.linspace(max_display_meters, 0.0, height)[:, None].repeat(SCALE_BAR_WIDTH, axis=1)
    scale = np.zeros((height, SCALE_BAR_WIDTH + SCALE_LABEL_WIDTH, 3), np.uint8)
    scale[:, :SCALE_BAR_WIDTH] = depth_heatmap(meters_per_row, max_display_meters)
    for meters in range(math.floor(max_display_meters) + 1):
        row = round((1 - meters / max_display_meters) * (height - 1))
        label_y = min(max(row + 5, 12), height - 4)
        cv2.putText(scale, f"{meters} m", (SCALE_BAR_WIDTH + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Live depth heatmap from the Windows webcam.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-display-meters", type=float, default=4.0, help="Depth at the far end of the colormap.")
    arguments = parser.parse_args()

    print("Loading model...")
    depth_estimator = MogeDepthEstimator()
    webcam = WindowsWebcam(arguments.camera_index)
    print("Press S to save a 16-bit depth PNG (millimeters), Esc to quit.")
    depth_meters = None
    while webcam.is_streaming():
        pressed_key = cv2.waitKey(1)
        if pressed_key == ESCAPE_KEY:
            break
        if pressed_key == SAVE_KEY and depth_meters is not None:
            depth_path = time.strftime("depth_%Y%m%d_%H%M%S.png")
            cv2.imwrite(depth_path, to_depth_millimeters(depth_meters))
            print(f"Saved {depth_path}")

        frame_bgr = webcam.take_newest_frame()
        if frame_bgr is None:
            continue
        depth_meters = depth_estimator.estimate_meters(frame_bgr)
        heatmap = depth_heatmap(depth_meters, arguments.max_display_meters)
        cv2.imshow("Depth", np.hstack([heatmap, color_scale(heatmap.shape[0], arguments.max_display_meters)]))


if __name__ == "__main__":
    main()
