#!/usr/bin/env bash
# One-time setup on the Jetson Xavier NX (JetPack 5.1: Ubuntu 20.04, Python 3.8) for the ROS 1 Noetic fire_depth node.
#   bash FireDepth/setup_jetson.sh
# Needs internet and sudo. Safe to run again.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CATKIN_WS="${CATKIN_WS:-$HOME/catkin_ws}"

grep -q "R35" /etc/nv_tegra_release || { echo "ROS Noetic needs JetPack 5 (L4T R35): $(cat /etc/nv_tegra_release)"; exit 1; }
# YOLO's venv (YoloFire/jetson/.venv) has the Jetson torch and ONNX Runtime; the node runs in it too.
bash "$REPO_DIR/YoloFire/jetson/setup_jetson.sh"

if [ ! -f /opt/ros/noetic/setup.bash ]; then
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
    wget -qO - https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    sudo apt-get update
fi
sudo apt-get install -y ros-noetic-ros-base ros-noetic-usb-cam ros-noetic-cv-bridge ros-noetic-tf2-ros \
    ros-noetic-tf2-geometry-msgs

mkdir -p "$CATKIN_WS/src"
[ -e "$CATKIN_WS/src/fire_depth" ] || ln -s "$REPO_DIR/FireDepth" "$CATKIN_WS/src/fire_depth"
set +u  # ROS's setup scripts read unset variables
source /opt/ros/noetic/setup.bash
set -u
(cd "$CATKIN_WS" && catkin_make)

"$REPO_DIR/YoloFire/jetson/.venv/bin/pip" install scipy
# Downloads MoGe-2 on first run and checks ONNX Runtime can load it.
PYTHONPATH="$REPO_DIR/DepthCode:${PYTHONPATH:-}" "$REPO_DIR/YoloFire/jetson/.venv/bin/python" -c \
    "import cv_bridge, rospy, tf2_geometry_msgs; from moge_depth import MogeDepthEstimator; print(MogeDepthEstimator().providers)"
echo "Setup done. See FireDepth/README.md to run it."
