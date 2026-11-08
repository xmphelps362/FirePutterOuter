import argparse
import time

import cv2
import numpy as np

from fire_depth import DEFAULT_FIRE_MODEL, FireDepth
from live_view import color_scale, depth_heatmap
from windows_webcam import WindowsWebcam

ESCAPE_KEY = 27
BOX_COLORS = {"lit_candle": (0, 0, 255), "unlit_candle": (255, 200, 0)}


def draw_fires(image: np.ndarray, detections: list, distances: list[float]) -> None:
    for detection, distance in zip(detections, distances):
        x1, y1, x2, y2 = (int(v) for v in detection.box)
        color = BOX_COLORS.get(detection.cls, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{detection.cls} {detection.confidence:.2f} {distance:.2f} m", (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live candle detection with distances from the Windows webcam.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fire-model", default=str(DEFAULT_FIRE_MODEL))
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--processing-resolution", type=int, default=392, help="Lower is faster. Multiple of 14.")
    parser.add_argument("--horizontal-fov-degrees", type=float, default=65.0)
    parser.add_argument("--max-display-meters", type=float, default=4.0)
    arguments = parser.parse_args()

    print("Loading models...")
    fire_depth = FireDepth(arguments.fire_model, arguments.confidence, arguments.horizontal_fov_degrees,
                           arguments.processing_resolution)
    webcam = WindowsWebcam(arguments.camera_index)
    print("Esc to quit.")
    frame_id, previous_frame_time = 0, time.monotonic()
    while webcam.is_streaming() and cv2.waitKey(1) != ESCAPE_KEY:
        frame_bgr = webcam.take_newest_frame()
        if frame_bgr is None:
            continue
        fires, distances = fire_depth.measure(frame_bgr, frame_id)
        frame_id += 1
        if fire_depth.newest_depth is None:
            continue
        for fire, distance in zip(fires.detections, distances):
            print(f"{frame_id} {fire.cls} {fire.confidence:.2f} {distance:.2f} m")

        heatmap = depth_heatmap(fire_depth.newest_depth[1], arguments.max_display_meters)
        draw_fires(frame_bgr, fires.detections, distances)
        draw_fires(heatmap, fires.detections, distances)
        frames_per_second = 1 / (time.monotonic() - previous_frame_time)
        previous_frame_time = time.monotonic()
        cv2.putText(frame_bgr, f"{frames_per_second:.1f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Fire depth", np.hstack([frame_bgr, heatmap, color_scale(heatmap.shape[0], arguments.max_display_meters)]))


if __name__ == "__main__":
    main()
