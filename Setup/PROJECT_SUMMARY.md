# Project Summary: Candle Detection and Metric Depth (ENEE408I)

## Goal
A **TurtleBot 3** team project (FirePutterOuter): find lit and unlit tealights with YOLO and measure how far away they
are with the monocular depth model **MoGe-2 ViT-S**. The same code runs on my laptop webcam and on the robot's
**Jetson Xavier NX**, for a camera on the robot and a fixed "watch tower" camera. My lab group writes the driving code;
this repo supplies detection, distances and positions.

## My preferences (please follow these)
- **Keep code as short as possible.** Cut bloat aggressively and don't add features "for later."
- **Names must be intuitive and descriptive** and must not sound AI-generated.
- **No AI-style comments.** Don't narrate or restate the code. Only comment genuinely non-obvious facts.
- **Efficiency matters.** It has to run live.

## Environment
- **Laptop:** Windows 11 running **WSL2 Ubuntu**, Python 3.12, NVIDIA RTX 3060 Ti (8 GB). CUDA works inside WSL.
  - `Setup/setup_env.sh` builds `Setup/.venv` with `uv` (no sudo): PyTorch, ONNX Runtime GPU, Ultralytics, OpenCV.
  - OpenCV's window needs `libsm6 libice6`; the script warns if they're missing. Windows open through WSLg.
- **Jetson:** Xavier NX on **JetPack 5.1** (Ubuntu 20.04, **Python 3.8**, Volta GPU, TensorRT 8.5) with **ROS 1
  Noetic**, as in the course labs. `FireDepth/setup_jetson.sh` sets it up.

## Key constraint: the laptop webcam
WSL2 cannot see USB webcams (there is no `/dev/video*`). The workaround:
- **Windows side:** `DepthCode/send_webcam.py`, under Windows Python, captures the webcam with OpenCV (`CAP_DSHOW`). It
  streams frames over TCP to `127.0.0.1:5600` as length-prefixed JPEGs (a 4-byte big-endian length, then the JPEG bytes).
- **Automatic launch:** the Linux viewer launches that script itself through WSL interop (`py.exe`), so only one command
  is needed. Windows Python needs `opencv-python`.
- **Avoided option:** `usbipd` passthrough. It needs an admin install, and the WSL kernel usually lacks the UVC driver.

## Depth model
- **MoGe-2 ViT-S** (35M parameters, MIT) runs from Microsoft's ONNX export through ONNX Runtime, so it works on the
  Jetson's Python 3.8. `DepthCode/moge_depth.py` downloads it (140 MB) on first use.
- The ONNX model leaves out MoGe's post-processing, so `moge_depth.py` redoes it: fit the z shift (and the focal length
  when there's no calibration), then scale to meters. It matches MoGe's PyTorch `infer()` exactly.
- Without a calibrated camera, MoGe estimates the focal length itself.
- `DepthCode/export_moge.py` makes a fixed 640x480 export, which the Jetson can run with TensorRT FP16.
- **Rejected: Depth Anything 3.** `DA3METRIC-LARGE` ran at about 15 fps on the laptop, but it needs Python 3.9+ (JetPack
  5.1 has 3.8), relies on bfloat16 (Volta has none) and would manage about 1 fps on the Xavier NX.

## How to run
- **Laptop:** `Setup/.venv/bin/python FireDepth/fire_depth_view.py` (boxes and distances) or
  `Setup/.venv/bin/python DepthCode/live_view.py` (depth only; **S** saves a 16-bit PNG in millimeters, 0 = no depth).
  Esc quits.
- **Jetson:** see `FireDepth/README.md` (`roslaunch fire_depth fire_depth.launch`).
- **Troubleshooting:** "No frames from camera 0." means another app is using the camera or the index is wrong.
  "Address already in use" means an old viewer is still running.

## Open items
- **Accuracy:** calibrate each webcam (checkerboard + `camera_calibration`) and pass `camera_info_url`.
- **Jetson:** nothing has run on it yet: ROS, usb_cam, ONNX Runtime's TensorRT support and frame rates are unverified.
