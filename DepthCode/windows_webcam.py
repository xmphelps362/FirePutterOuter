import socket
import struct
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

FRAME_PORT = 5600
JPEG_LENGTH_HEADER = struct.Struct(">I")
SENDER_SCRIPT = Path(__file__).resolve().parent.parent / "windows" / "send_webcam.py"


class WindowsWebcam:
    """WSL cannot open USB webcams, so frames are captured by a Windows process and sent over TCP."""

    def __init__(self, camera_index: int) -> None:
        self._newest_frame: np.ndarray | None = None
        self._server_socket = socket.create_server(("127.0.0.1", FRAME_PORT))
        sender_script_windows_path = subprocess.run(
            ["wslpath", "-w", str(SENDER_SCRIPT)], capture_output=True, text=True, check=True
        ).stdout.strip()
        # Windows processes reject a WSL (UNC) working directory.
        self._sender = subprocess.Popen(
            ["py.exe", sender_script_windows_path, str(FRAME_PORT), str(camera_index)], cwd="/mnt/c"
        )
        threading.Thread(target=self._receive_frames, daemon=True).start()

    def is_streaming(self) -> bool:
        return self._sender.poll() is None

    def take_newest_frame(self) -> np.ndarray | None:
        frame, self._newest_frame = self._newest_frame, None
        return frame

    def _receive_frames(self) -> None:
        sender_socket, _ = self._server_socket.accept()
        frame_stream = sender_socket.makefile("rb")
        while len(header := frame_stream.read(JPEG_LENGTH_HEADER.size)) == JPEG_LENGTH_HEADER.size:
            jpeg_bytes = frame_stream.read(JPEG_LENGTH_HEADER.unpack(header)[0])
            self._newest_frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
