# Fire depth

YOLO finds lit and unlit candles and a depth model measures how far away they are. YOLO runs on every frame. Depth is
slower, so it runs in its own thread on the newest frame, and each detection's distance comes from the newest finished
depth map. The distance is the median depth of the middle of the detection box.

- **Laptop (WSL):** Depth Anything 3 on the 3060 Ti (YOLO about 19 fps, depth about 10 fps).
- **Jetson Xavier NX (JetPack 5.1, ROS 1 Noetic):** MoGe-2 ViT-S as an ONNX model. Depth Anything 3 needs Python 3.9+,
  and JetPack 5.1 has Python 3.8.

Both use `YoloFire/export/fire.pt`, which is in git.

## WSL (laptop webcam)

```bash
~/ENEE408/.venv/bin/python FireDepth/fire_depth_view.py
```

This shows the camera and the depth heatmap side by side, with boxes and distances, and prints one line per detection.
Options: `--confidence`, `--processing-resolution` (lower is faster), `--horizontal-fov-degrees`.

## ROS 1 Noetic on the Jetson Xavier NX (JetPack 5.1)

Once, with internet: runs `YoloFire/jetson/setup_jetson.sh` (Jetson PyTorch and ONNX Runtime in
`YoloFire/jetson/.venv`), installs ROS Noetic and `usb_cam`, downloads MoGe-2, and links this folder into
`~/catkin_ws/src` as the `fire_depth` package.

```bash
bash FireDepth/setup_jetson.sh
bash YoloFire/jetson/build_engine.sh YoloFire/export/fire.pt   # optional, faster YOLO
```

Every time (the venv first, so the node runs on the Python that has the models):

```bash
source YoloFire/jetson/.venv/bin/activate && source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://<computer running roscore>:11311 ROS_HOSTNAME=<this Jetson's IP>
roslaunch fire_depth fire_depth.launch camera_x:=0.05 camera_z:=0.15 camera_pitch:=0.2                           # robot
roslaunch fire_depth fire_depth.launch name:=tower parent_frame:=tower camera_z:=1.5 camera_pitch:=0.6           # tower
```

The launch file starts `usb_cam` on `/dev/video0` at 640x480 (`video_device:=`), the camera's tf frames from the
`camera_*` arguments (meters and radians, positive pitch tilts down) and the node, all in the namespace `name`. If
another machine publishes the camera, use `start_camera:=false` and publish it as `/<name>/usb_cam/image_raw` (and
`camera_info`). For the best focal length, calibrate each webcam with `camera_calibration` and pass
`camera_info_url:=file:///path/to/calibration.yaml`. Without it, MoGe estimates it.

Publishes, under `/robot/` or `/tower/`:
- `fire/bearing` (`std_msgs/Float32`): where the target candle is across the image, -1 left edge to 1 right edge, NaN
  when no lit candle is in view. The target is the nearest lit candle. Ready for a Lab 4 style P-controller.
- `fire/target` (`geometry_msgs/PointStamped`): the target candle's position in `parent_frame`, once depth is ready.
- `fire/detections` (`std_msgs/String`): every candle in one JSON line, with `class`, `confidence`, `box`, `bearing`,
  `distance` (meters) and `position` (in `frame`), null until depth is ready.
- `depth/image` (`32FC1`, meters, NaN where there is no depth) and `depth/camera_info`, stamped with the frame the depth
  was computed from.

### Faster depth with TensorRT (optional)

The downloaded MoGe-2 model takes any image size, which TensorRT can't run, so it runs on ONNX Runtime's CUDA backend.
`export_moge.py` makes a fixed 640x480 one on the PC (instructions at the top of the file); copy it over
`FireDepth/models/moge-2-vits.onnx` on the Jetson. The node then runs it with TensorRT in FP16; the first start builds
the engine, which takes several minutes. The log line at startup says which backend is in use.

Tests (no models needed): `python -m pytest FireDepth/tests`
