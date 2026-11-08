#!/usr/bin/env bash
# Run the fire detector on the webcam with the bundled ONNX model. Extra flags pass through:
#   bash ~/candle-fire/jetson/run.sh
#   bash ~/candle-fire/jetson/run.sh --source /dev/video1 --conf 0.4
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/.venv/bin/activate"
exec python3 "$DIR/detect_camera.py" --show "$@"
