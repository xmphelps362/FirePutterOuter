#!/usr/bin/env bash
# One-time setup on the Jetson Xavier NX (JetPack 5.1: Ubuntu 20.04, Python 3.8) for the ROS 1 Noetic fire_depth node.
#   bash FireDepth/setup_jetson.sh
# Needs internet and sudo. Safe to run again.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CATKIN_WS="${CATKIN_WS:-$HOME/catkin_ws}"

grep -q "R35" /etc/nv_tegra_release || { echo "ROS Noetic needs JetPack 5 (L4T R35): $(cat /etc/nv_tegra_release)"; exit 1; }

# ROS's apt-source package trusts its key for the ROS repository only and replaces it when ROS rotates it. It goes
# first: an older ROS source with the expired key makes every apt-get update fail, including YoloFire's.
if ! dpkg -s ros-apt-source >/dev/null 2>&1; then
    sudo rm -f /etc/apt/sources.list.d/ros-latest.list
    sudo apt-key del F42ED6FBAB17C654 2>/dev/null || true
    wget -qO /tmp/ros-apt-source.deb \
        https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.3.0/ros-apt-source_1.3.0.focal_all.deb
    sudo dpkg -i /tmp/ros-apt-source.deb
fi

# YOLO's venv (YoloFire/jetson/.venv) has the Jetson torch and ONNX Runtime; the node runs in it too.
bash "$REPO_DIR/YoloFire/jetson/setup_jetson.sh"

sudo apt-get install -y ros-noetic-ros-base ros-noetic-usb-cam ros-noetic-cv-bridge ros-noetic-tf2-ros \
    ros-noetic-tf2-geometry-msgs

mkdir -p "$CATKIN_WS/src"
[ -e "$CATKIN_WS/src/fire_depth" ] || ln -s "$REPO_DIR/FireDepth" "$CATKIN_WS/src/fire_depth"
set +u  # ROS's setup scripts read unset variables
source /opt/ros/noetic/setup.bash
set -u
(cd "$CATKIN_WS" && catkin_make)

"$REPO_DIR/YoloFire/jetson/.venv/bin/pip" install scipy
# Downloads MoGe-2 and checks ONNX Runtime can load it. The image round trip loads cv_bridge's compiled part, which
# `import cv_bridge` alone doesn't, and which breaks if ROS's OpenCV and the venv's don't get along.
PYTHONPATH="$REPO_DIR/DepthCode:${PYTHONPATH:-}" "$REPO_DIR/YoloFire/jetson/.venv/bin/python" - <<'EOF'
import numpy as np
import rospy, tf2_geometry_msgs
from cv_bridge import CvBridge
from moge_depth import MogeDepthEstimator

bridge = CvBridge()
bridge.imgmsg_to_cv2(bridge.cv2_to_imgmsg(np.zeros((2, 2, 3), np.uint8), "rgb8"), "bgr8")
print("Depth backends:", MogeDepthEstimator().providers)
EOF
echo "Setup done. See FireDepth/README.md to run it."
