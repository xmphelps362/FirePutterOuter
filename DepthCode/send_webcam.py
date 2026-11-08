"""Launched by webcam_depth/windows_webcam.py under Windows Python. Requires: py -m pip install opencv-python"""

import socket
import struct
import sys
import time

import cv2

JPEG_LENGTH_HEADER = struct.Struct(">I")
NO_FRAMES_TIMEOUT_SECONDS = 2.0


def main() -> None:
    port, camera_index = int(sys.argv[1]), int(sys.argv[2])
    receiver_socket = socket.create_connection(("127.0.0.1", port))
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    last_frame_time = time.monotonic()
    while time.monotonic() - last_frame_time < NO_FRAMES_TIMEOUT_SECONDS:
        frame_was_read, frame_bgr = camera.read()
        if not frame_was_read:
            continue
        last_frame_time = time.monotonic()
        jpeg_bytes = cv2.imencode(".jpg", frame_bgr)[1].tobytes()
        try:
            receiver_socket.sendall(JPEG_LENGTH_HEADER.pack(len(jpeg_bytes)) + jpeg_bytes)
        except ConnectionError:
            return
    sys.exit(f"No frames from camera {camera_index}.")


if __name__ == "__main__":
    main()
