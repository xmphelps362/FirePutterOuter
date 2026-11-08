# Project Summary: Live Metric Depth with Depth Anything 3 (ENEE408)

## Goal
I'm building a depth-perception pipeline for a **TurtleBot 3** so it can tell how close it is to objects, using a single camera and the monocular depth model **Depth Anything 3 (DA3)** (https://github.com/bytedance-seed/depth-anything-3). **Current phase:** validating the pipeline live on my laptop webcam. Object/proximity detection was built earlier and then **deliberately removed**. Right now the program only shows a live depth heatmap and can save 16-bit depth images. The TurtleBot integration comes later.

## My preferences (please follow these)
- **Keep code as short as possible.** Cut bloat aggressively and don't add features "for later."
- **Names must be intuitive and descriptive** and must not sound AI-generated.
- **No AI-style comments.** Don't narrate or restate the code. Only comment genuinely non-obvious facts.
- **Efficiency matters.** It has to run live.

## Environment
- **Machine:** Windows 11 laptop running **WSL2 Ubuntu**, Python 3.12.3. The project is at `~/ENEE408` (`/home/marks/ENEE408`).
- **GPU:** NVIDIA RTX 3060 Ti with 8 GB VRAM. CUDA works inside WSL (torch 2.14.1+cu130, `cuda available: True`).
- **Python environment:** a venv at `~/ENEE408/.venv`, built with `uv`. The system has no `python3-venv` and no passwordless sudo, so `uv` was installed per-user and needs no sudo.
- **Rebuilding the environment:** `./setup_env.sh` rebuilds everything:
  - It clones DA3 into `~/ENEE408/depth-anything-3` and installs it editable.
  - It installs torch, torchvision, xformers and `requirements.txt`.
  - It warns if the system packages `libsm6 libice6` are missing. OpenCV's Qt window needs them. They are already installed with `sudo apt install libsm6 libice6`.
- **Known DA3 packaging gaps:** DA3's `pyproject.toml` forgets `addict` and `scikit-learn`, so they are listed in `requirements.txt`.
- **Harmless warnings:** an xFormers version-mismatch warning (DA3 falls back to PyTorch attention) and a `gsplat` warning (only needed for Gaussian splatting).
- **Display:** windows open through WSLg (`DISPLAY=:0`).

## Key constraint: the webcam
WSL2 cannot see USB webcams (there is no `/dev/video*`). The workaround:
- **Windows side:** a small script under Windows Python, `windows/send_webcam.py`, captures the webcam with OpenCV (`CAP_DSHOW`). It streams frames over TCP to `127.0.0.1:5600` as length-prefixed JPEGs (a 4-byte big-endian length, then the JPEG bytes).
- **Automatic launch:** the Linux viewer launches that script itself through WSL interop (`py.exe`), so only one command is needed.
- **Windows dependency:** Windows Python has `opencv-python` installed.
- **Networking:** WSL is in NAT mode, and Windows→WSL `localhost` forwarding works.
- **Avoided option:** `usbipd` passthrough was considered and rejected. It needs an admin install, and the WSL kernel usually lacks the UVC driver.

## Model choice and findings
- **Model in use: `DA3METRIC-LARGE`** (Apache 2.0). It outputs metric depth from a single image. It is not in meters directly: `meters = raw * focal_px / 300`.
  - DA3 does **not** apply this conversion for the standalone metric model; only the nested model does it internally.
  - The webcam is uncalibrated, so the focal length comes from an assumed **horizontal FOV of 65°** (`--horizontal-fov-degrees`), scaled to the processed depth-map width.
  - Distances are therefore approximate. A real calibration, or the TurtleBot camera's intrinsics, will fix the scale.
- **Rejected: `DA3NESTED-GIANT-LARGE-1.1`,** the most accurate metric model. It filled all 8 GB of VRAM, ran the GPU at 100%, and did not finish timing a single frame in 10 minutes, so it is unusable live on this GPU. It is also CC BY-NC.
- **Relative-depth models (DA3MONO, DA3-SMALL/BASE):** not used, because metric output is needed.
- **Other model families:** Depth Pro, UniDepth V2, MoGe-2 and Metric3D v2 were mentioned but not tested.
- **API facts from the DA3 source:**
  - `model.inference([rgb_numpy], process_res=N)` accepts numpy RGB arrays.
  - `prediction.depth[0]` is an `[H, W]` float32 array at the processed resolution (longest side ≈ `process_res`, rounded to a multiple of 14).
  - `prediction.conf` is `None` for the metric model.
  - DA3 already calls `.eval()`, runs under `no_grad`/`inference_mode`, and uses autocast to **bfloat16** for the backbone while **deliberately keeping the depth head in fp32** (`torch.autocast(enabled=False)` in `model/da3.py`).
  - Setting the env var `DA3_LOG_LEVEL=WARN` before importing DA3 silences per-frame INFO logs.

## 16-bit weights (done)
- **The obvious approach fails.** Converting the whole model to 16-bit crashes with a dtype mismatch, because the depth head runs in fp32 on purpose.
- **The solution:** convert **only the backbone** to bfloat16 with `model.model.backbone.to(torch.bfloat16)`. That matches the dtype DA3 already computes in, so there is no per-frame casting.
- **Measured against all-fp32, on the same example image at `process_res=392`:**

  | | All fp32 (before) | bf16 backbone (now) |
  |---|---|---|
  | Time per frame | 55 ms | 47 ms |
  | Live webcam FPS | 13.6 | 15.6 |
  | Weight memory | 1337 MB | 728 MB |
  | GPU memory while running | 1467 MB | 858 MB |
  | Depth change | — | 0.26% mean, 0.62% at the 99th percentile |

- **fp16 backbone was worse:** 54 ms/frame and a 1.25% mean change, because fp16 weights get re-cast to bf16 under autocast every frame.

## 16-bit depth output and color scale (done)
- **Saving depth:** pressing **S** in the viewer saves the current depth as a **16-bit PNG in millimeters** (`depth_YYYYMMDD_HHMMSS.png` in the current directory).
  - This is the ROS `16UC1` convention: 0–65,535 mm.
  - It is saved at the model's resolution (e.g. 392×294 from a 640×480 webcam), not resized to the camera's resolution.
  - Verified: the file reads back as uint16 with at most 0.5 mm rounding error.
  - To read it back: `cv2.imread(path, cv2.IMREAD_UNCHANGED) / 1000` gives meters.
- **Color scale:** a bar on the right side of the window maps heatmap colors to meters (0 m at the bottom, `--max-display-meters` at the top, labeled every 1 m).
  - It is built with the same `depth_heatmap` function, so it always matches the heatmap.
  - It is cached with `functools.cache`.
- **Heatmap colors:** TURBO colormap with red = near and dark blue/purple = far. Anything beyond `--max-display-meters` (default 4 m) shows as the farthest color.

## Code reviews done
1. **First review (mine):** led to stale-frame detection, NaN-safe proximity and median smoothing. All of that was later removed when proximity was dropped.
2. **Review using the Superpowers `requesting-code-review` skill:** run with a reviewer subagent, using the skill files downloaded from github.com/obra/superpowers. The plugin itself is enabled in `~/.claude/settings.json`, through the `superpowers-marketplace` entry and `superpowers@superpowers-marketplace`, but it only loads in a new Claude Code session. Verdict: **"With fixes."** Applied:
   - **Important fix:** the viewer used to hang silently if the camera failed to open. Now the sender exits with `No frames from camera N.` and the viewer loop checks `webcam.is_streaming()`.
   - The server is bound to `127.0.0.1` instead of `0.0.0.0`.
   - The log-level setting moved into `depth_estimator.py`.
   - Docstring and comment fixes, and the renames `FRAME_PORT` and `NO_FRAMES_TIMEOUT_SECONDS`.
   - The empty `__init__.py` was removed (namespace package), and `parse_arguments` was inlined into `main`.

   Not applied (open choices):
   - Replacing the argparse flags with constants (saves ~8 lines).
   - Making the window's X button quit; it currently just reopens the window, while Esc and Ctrl+C work.

## How to run
```bash
cd ~/ENEE408 && source .venv/bin/activate && python -m webcam_depth.live_view
```
- **Flags:**
  - `--camera-index N` (default 0)
  - `--processing-resolution N` (default 392; must be a multiple of 14; lower is faster)
  - `--horizontal-fov-degrees F` (default 65)
  - `--max-display-meters M` (default 4)
- **Keys:** **S** saves a 16-bit depth PNG; **Esc** or Ctrl+C quits. The Windows sender exits by itself when the viewer closes.
- **Troubleshooting:**
  - "No frames from camera 0." means another app is using the camera or the index is wrong.
  - "Address already in use" means an old viewer is still running: `pkill -f webcam_depth.live_view`.

## Project layout
```
~/ENEE408/
├── setup_env.sh            # one-command environment setup (no sudo)
├── requirements.txt        # opencv-python, numpy<2, addict, scikit-learn
├── depth-anything-3/       # cloned DA3 repo, installed editable
├── .venv/                  # Python 3.12 venv (~7 GB with CUDA libs)
├── webcam_depth/
│   ├── depth_estimator.py  # model loading, bf16 backbone, meters conversion, mm conversion
│   ├── windows_webcam.py   # TCP receiver + auto-launches the Windows sender
│   └── live_view.py        # heatmap window, color scale, S-to-save
└── windows/
    └── send_webcam.py      # runs under Windows Python; streams the webcam to WSL
```

## Current code

### webcam_depth/depth_estimator.py
```python
import math
import os

# Read when depth_anything_3 is imported; otherwise it logs timing info every frame.
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

MODEL_NAME = "depth-anything/DA3METRIC-LARGE"
# DA3METRIC outputs depth as if the focal length were 300 px; scaling by the real focal length gives meters.
MODEL_FOCAL_LENGTH_PIXELS = 300.0


class MetricDepthEstimator:
    def __init__(self, horizontal_fov_degrees: float, processing_resolution: int) -> None:
        self._model = DepthAnything3.from_pretrained(MODEL_NAME).to("cuda")
        # DA3 already computes the backbone in bfloat16 but keeps the depth head in 32-bit on purpose.
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
```

### webcam_depth/windows_webcam.py
```python
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
```

### webcam_depth/live_view.py
```python
import argparse
import functools
import math
import time

import cv2
import numpy as np

from webcam_depth.depth_estimator import MetricDepthEstimator, to_depth_millimeters
from webcam_depth.windows_webcam import WindowsWebcam

ESCAPE_KEY = 27
SAVE_KEY = ord("s")
SCALE_BAR_WIDTH = 25
SCALE_LABEL_WIDTH = 45


def depth_heatmap(depth_meters: np.ndarray, max_display_meters: float) -> np.ndarray:
    nearness = 1.0 - np.clip(depth_meters / max_display_meters, 0.0, 1.0)
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
    parser.add_argument("--processing-resolution", type=int, default=392, help="Lower is faster. Multiple of 14.")
    parser.add_argument("--horizontal-fov-degrees", type=float, default=65.0)
    parser.add_argument("--max-display-meters", type=float, default=4.0, help="Depth at the far end of the colormap.")
    arguments = parser.parse_args()

    print("Loading model...")
    depth_estimator = MetricDepthEstimator(arguments.horizontal_fov_degrees, arguments.processing_resolution)
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
        frame_width, frame_height = frame_bgr.shape[1], frame_bgr.shape[0]
        heatmap = cv2.resize(depth_heatmap(depth_meters, arguments.max_display_meters), (frame_width, frame_height))
        cv2.imshow("Depth", np.hstack([heatmap, color_scale(frame_height, arguments.max_display_meters)]))


if __name__ == "__main__":
    main()
```

### windows/send_webcam.py
```python
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
```

## Open items and next steps
- **TurtleBot port.** Replace `WindowsWebcam` with a ROS camera subscriber; `MetricDepthEstimator.estimate_meters(frame_bgr)` takes a BGR array and can be reused as is.
  - Use the robot camera's calibrated focal length instead of the 65° FOV guess.
  - Publish depth as `16UC1` in millimeters with `to_depth_millimeters`.
  - Consider resizing saved or published depth to the camera's resolution so it lines up pixel-for-pixel.
- **Proximity logic.** Re-add "nearest object ahead" later: a region of interest, a low-percentile distance, temporal smoothing, and fail-safe handling of stale or invalid data. It existed before and was removed for simplicity.
- **Accuracy.** Calibrate the webcam (checkerboard + OpenCV) to fix the metric scale. This matters more than switching models.
- **Optional cleanups:** argparse → constants; make the window's X button quit.
