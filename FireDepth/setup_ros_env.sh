#!/usr/bin/env bash
# One-time setup on the Jetson (JetPack 6, Ubuntu 22.04, ROS2 Humble). Then:
#   source /opt/ros/humble/setup.bash && ros2 launch FireDepth/fire_depth.launch.py
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv-ros"
DA3_DIR="$REPO_DIR/depth-anything-3"
# Same JetPack 6 wheels as YoloFire/jetson/setup_jetson.sh. Pip's default torch is CPU-only on Jetson.
ASSETS=https://github.com/ultralytics/assets/releases/download/v0.0.0
JETSON_WHEELS=("$ASSETS/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl"
               "$ASSETS/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl"
               "$ASSETS/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl")

grep -q "R36" /etc/nv_tegra_release || { echo "Needs JetPack 6 (L4T R36)."; exit 1; }

sudo apt-get update
sudo apt-get install -y python3-venv libopenblas-dev ros-humble-cv-bridge ros-humble-vision-msgs \
    ros-humble-tf2-ros ros-humble-tf2-geometry-msgs
if ! ldconfig -p | grep -q libcusparseLt; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update
    sudo apt-get install -y libcusparselt0 libcusparselt-dev
fi
[ -d "$DA3_DIR" ] || git clone https://github.com/bytedance-seed/depth-anything-3 "$DA3_DIR"

# System site packages so the venv sees rclpy, cv_bridge, tf2 and TensorRT.
[ -d "$VENV_DIR" ] || python3 -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install -U pip
# DA3's declared dependencies include xformers, which would replace the Jetson torch, so install only what it imports.
pip install --no-deps -e "$DA3_DIR"
pip install addict einops evo huggingface_hub imageio "moviepy==1.0.3" omegaconf plyfile pycolmap safetensors \
    scikit-learn trimesh pillow "ultralytics==8.4.208" onnx onnxslim
pip install "${JETSON_WHEELS[@]}"
# Jetson torch needs numpy 1.x, and opencv-python >= 4.12 needs numpy 2.
pip install "numpy==1.23.5" "opencv-python==4.11.0.86"

source /opt/ros/humble/setup.bash
python - <<'EOF'
import rclpy, cv_bridge, torch, ultralytics, depth_anything_3.api
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch has no CUDA -> wrong wheel for this JetPack"
EOF
echo "For faster YOLO: source $VENV_DIR/bin/activate && bash $REPO_DIR/YoloFire/jetson/build_engine.sh $REPO_DIR/YoloFire/export/fire.pt"
