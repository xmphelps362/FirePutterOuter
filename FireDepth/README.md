# Fire depth

YOLO finds lit and unlit candles, and Depth Anything 3 measures how far away they are. YOLO runs on every frame. Depth
is slower, so it runs in its own thread on the newest frame, and each detection's distance comes from the newest
finished depth map (on a 3060 Ti: YOLO about 19 fps, depth about 10 fps at the same time). The distance is the median
depth of the middle of the detection box.

Both versions need `YoloFire/export/fire.pt`, which is not in git. Copy it from the training machine.

## WSL (laptop webcam)

```bash
~/ENEE408/.venv/bin/python FireDepth/fire_depth_view.py
```

This shows the camera and the depth heatmap side by side, with boxes and distances, and prints one line per detection.
Options: `--confidence`, `--processing-resolution` (lower is faster), `--horizontal-fov-degrees`.

## ROS2 Humble on the Jetson (JetPack 6)

```bash
bash FireDepth/setup_ros_env.sh        # once: Jetson torch, Depth Anything, YOLO in .venv-ros
source .venv-ros/bin/activate && bash YoloFire/jetson/build_engine.sh YoloFire/export/fire.pt   # once, optional: TensorRT YOLO
source /opt/ros/humble/setup.bash
ros2 launch FireDepth/fire_depth.launch.py camera_x:=0.07 camera_z:=0.12 camera_pitch:=0.0
```

The launch file uses `fire.engine` if it exists, otherwise `fire.pt`. The first run downloads the Depth Anything
weights (about 1.4 GB), so the Jetson needs internet once. If depth is too slow, try `processing_resolution:=266`.
YOLO keeps its own rate either way.

Subscribes to `/camera/image_raw` and `/camera/camera_info`. Change them with `image_topic:=` and `camera_info_topic:=`.

Publishes:
- `/fire/detections` (`vision_msgs/Detection3DArray`): each candle's class, score, position and approximate size
  in `target_frame` (default `base_link`).
- `/depth/image` (`32FC1`, meters) and `/depth/camera_info`. Each depth image is stamped with the camera frame it was
  computed from, so `depth_image_proc` can turn it into a point cloud.

Transforms: a candle's pixel center and distance become a 3D point in the camera optical frame through the camera
intrinsics, and tf2 then moves that point into `target_frame`. The launch file publishes `base_link → camera_link →
camera_optical_frame` from the `camera_*` arguments (meters and radians). If the robot's URDF already has camera frames,
pass `publish_camera_tf:=false optical_frame:=<its optical frame>`. Without `camera_info`, the focal length falls back to
`horizontal_fov_degrees` (65).
