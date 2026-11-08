#!/usr/bin/env bash
# For the ROS2 Humble machine (Ubuntu 22.04, Python 3.10). Run once, then:
#   source /opt/ros/humble/setup.bash && ros2 launch FireDepth/fire_depth.launch.py
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv-ros"
DA3_DIR="$REPO_DIR/depth-anything-3"

sudo apt install -y ros-humble-cv-bridge ros-humble-vision-msgs ros-humble-tf2-ros ros-humble-tf2-geometry-msgs

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
[ -d "$DA3_DIR" ] || git clone https://github.com/bytedance-seed/depth-anything-3 "$DA3_DIR"

# System site packages so the venv sees rclpy, cv_bridge and tf2 from /opt/ros/humble.
uv venv --python /usr/bin/python3.10 --system-site-packages "$VENV_DIR"
export VIRTUAL_ENV="$VENV_DIR"
uv pip install torch torchvision
uv pip install -e "$DA3_DIR"
uv pip install -r "$REPO_DIR/Setup/requirements.txt"

source /opt/ros/humble/setup.bash
"$VENV_DIR/bin/python" -c "import rclpy, cv_bridge, torch, ultralytics, depth_anything_3; print('cuda', torch.cuda.is_available())"
