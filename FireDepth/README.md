# Fire depth

YOLO finds lit and unlit candles and MoGe-2 ViT-S measures how far away they are, the same code on the laptop and the
Jetson. YOLO runs on every frame. Depth is slower, so it runs in its own thread on the newest frame, and each
detection's distance comes from the newest finished depth map. The distance is the median depth of the middle of the
detection box.

MoGe-2 runs from an ONNX file (`DepthCode/moge_depth.py`), so it works on the Jetson's Python 3.8. It downloads itself
(140 MB) to `DepthCode/models/` the first time. Without a camera calibration, MoGe estimates the focal length itself.
YOLO uses `YoloFire/export/fire.pt`, which is in git.

## WSL (laptop webcam)

```bash
bash Setup/setup_env.sh   # once: Setup/.venv with PyTorch, ONNX Runtime GPU and Ultralytics
Setup/.venv/bin/python FireDepth/fire_depth_view.py
```

This shows the camera and the depth heatmap side by side, with boxes and distances, and prints one line per detection.
Options: `--confidence`, `--max-display-meters`. `DepthCode/live_view.py` shows depth alone and saves it with S.

## ROS 1 Noetic on the Jetson Xavier NX (JetPack 5.1)

### One-time setup

1. Check it's JetPack 5: `cat /etc/nv_tegra_release` should show `R35`.
2. Get the code (HTTPS, so the Jetson needs no SSH key):
   ```bash
   git clone https://github.com/xmphelps362/FirePutterOuter.git && cd FirePutterOuter
   ```
3. Run the setup (internet and sudo, about 15-20 minutes). It runs `YoloFire/jetson/setup_jetson.sh` (Jetson PyTorch,
   ONNX Runtime and YOLO in `YoloFire/jetson/.venv`), installs ROS Noetic and `usb_cam`, links this folder into
   `~/catkin_ws/src` as the `fire_depth` package, builds it and downloads MoGe-2. Just before "Setup done" it prints
   the ONNX Runtime backends; `CUDAExecutionProvider` first means depth will run on the GPU.
   ```bash
   bash FireDepth/setup_jetson.sh
   ```
4. Optional, faster YOLO (several minutes; used automatically once it exists):
   ```bash
   bash YoloFire/jetson/build_engine.sh YoloFire/export/fire.pt
   ```
5. Full power: `sudo nvpmodel -q --verbose` lists the modes; pick the 20 W 6-core one with
   `sudo nvpmodel -m <its number>`, then run `sudo jetson_clocks`.

### Every time

1. Plug the webcam into the Jetson. `v4l2-ctl --list-devices` should show it as `/dev/video0`; otherwise add
   `video_device:=/dev/videoN` to the launch line.
2. Join the TurtleBot's ROS network. `ROS_MASTER_URI` is the same one the Raspberry Pi and Remote PC use (Lab 0).
   Activate the venv first, so the node runs on the Python that has the models.
   ```bash
   source ~/FirePutterOuter/YoloFire/jetson/.venv/bin/activate && source ~/catkin_ws/devel/setup.bash
   export ROS_MASTER_URI=http://<computer running roscore>:11311 ROS_HOSTNAME=<this Jetson's IP>
   ```
3. Measure the webcam's position from the center of the robot: `camera_x` forward and `camera_z` up in meters,
   `camera_pitch` down in radians (0.2 rad is about 11 degrees). Then launch:
   ```bash
   roslaunch fire_depth fire_depth.launch camera_x:=0.05 camera_z:=0.15 camera_pitch:=0.2
   ```
   For the tower camera, positions come out in a `tower` frame on the floor below the camera:
   ```bash
   roslaunch fire_depth fire_depth.launch name:=tower parent_frame:=tower camera_z:=1.5 camera_pitch:=0.6
   ```

The launch file starts `usb_cam` at 640x480, the camera's tf frames and the node, all in the namespace `name`. If
another machine publishes the camera, use `start_camera:=false` and publish it as `/<name>/usb_cam/image_raw` (and
`camera_info`).

### Checking it works

- The startup log should say `Depth on CUDAExecutionProvider` (or `TensorrtExecutionProvider`). `CPUExecutionProvider`
  means the GPU isn't being used.
- From another terminal with the same ROS setup: `rostopic echo /robot/fire/bearing` (and `fire/detections`,
  `fire/target`). `rostopic hz /robot/fire/bearing` is the YOLO rate, `rostopic hz /robot/depth/image` the depth rate.
  View the depth image with `rqt_image_view` on the Remote PC; the Jetson's ROS install doesn't include it.
- Put a lit tealight 1 m and then 2 m straight ahead and compare with `distance` in `fire/detections`. If it's off by a
  steady percentage, calibrate the webcam (`sudo apt install ros-noetic-camera-calibration`, a printed checkerboard,
  `rosrun camera_calibration cameracalibrator.py`) and add `camera_info_url:=file:///path/to/calibration.yaml`.
- Report back the startup log line, the two `rostopic hz` rates and any errors from the setup, `catkin_make` or
  `cv_bridge`.

Publishes, under `/robot/` or `/tower/`:
- `fire/bearing` (`std_msgs/Float32`): where the target candle is across the image, -1 left edge to 1 right edge, NaN
  when no lit candle is in view. The target is the nearest lit candle. Ready for a Lab 4 style P-controller.
- `fire/target` (`geometry_msgs/PointStamped`): the target candle's position in `parent_frame`, once depth is ready.
- `fire/detections` (`std_msgs/String`): every candle in one JSON line, with `class`, `confidence`, `box`, `bearing`,
  `distance` (meters) and `position` (in `frame`), null until depth is ready.
- `depth/image` (`32FC1`, meters, NaN where there is no depth) and `depth/camera_info`, stamped with the frame the depth
  was computed from.

For the driving code, Lab 4 Part C style: steer with `angular.z = -k * bearing` (negative bearing means the candle is
to the left), creep forward slowly, rotate slowly to search while the bearing is NaN, and stop on `fire/target`'s `x`.
The robot camera loses a tealight once it's under the bumper, so finish the approach on the last known position.

### Faster depth with TensorRT (optional)

The downloaded MoGe-2 model takes any image size, which TensorRT can't run, so it runs on ONNX Runtime's CUDA backend.
`DepthCode/export_moge.py` makes a fixed 640x480 one on any PC, no GPU needed (instructions at the top of the file).
It writes `DepthCode/models/moge-2-vits-640x480.onnx` (about 140 MB, too big for git, so pass it on by USB, Drive or
`scp`). On the Jetson, replace the downloaded model with it:

```bash
cp moge-2-vits-640x480.onnx ~/FirePutterOuter/DepthCode/models/moge-2-vits.onnx
```

The node then runs it with TensorRT in FP16. The first start builds the engine, which takes several minutes, and saves it
in `DepthCode/models/`. If the log still says `CUDAExecutionProvider`, this ONNX Runtime has no TensorRT support and
depth runs as before.

Tests (no models needed): `python -m pytest FireDepth/tests`
