#!/usr/bin/env bash
# Run the fire detector on the webcam. Extra flags pass through:
#   bash YoloFire/jetson/run.sh
#   bash YoloFire/jetson/run.sh --source /dev/video1 --conf 0.4
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/.venv/bin/activate"
# stops Ultralytics from pip-installing over the Jetson torch and numpy
export YOLO_AUTOINSTALL=false
exec python3 "$DIR/detect_camera.py" --show "$@"
