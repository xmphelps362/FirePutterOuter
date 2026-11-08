#!/usr/bin/env bash
# One-time Jetson setup for JetPack 5 (L4T R35, Python 3.8) or JetPack 6 (R36, Python 3.10).
#   bash ~/candle-fire/jetson/setup_jetson.sh
# Wheels follow the Ultralytics NVIDIA Jetson guide (docs.ultralytics.com/guides/nvidia-jetson/);
# if a link 404s or torch won't import, take the current links for your JetPack from that page.
set -euo pipefail
cd "$(dirname "$0")"

REL="$(cat /etc/nv_tegra_release 2>/dev/null || true)"
ASSETS=https://github.com/ultralytics/assets/releases/download/v0.0.0
if grep -q "R35" <<<"$REL"; then
  JP=5
  WHEELS=("$ASSETS/torch-2.2.0-cp38-cp38-linux_aarch64.whl"
          "$ASSETS/torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl"
          "$ASSETS/onnxruntime_gpu-1.18.0-cp38-cp38-linux_aarch64.whl")
elif grep -q "R36" <<<"$REL"; then
  JP=6
  WHEELS=("$ASSETS/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl"
          "$ASSETS/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl"
          "$ASSETS/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl")
else
  echo "Unsupported Jetson software: $REL"
  echo "Original Jetson Nano (R32 / JetPack 4): use the Docker image ultralytics/ultralytics:latest-jetson-jetpack4"
  exit 1
fi
echo "JetPack $JP detected: $REL"

sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libopenblas-dev v4l-utils

# JetPack 6's torch 2.5 wheel needs cuSPARSELt (missing libcusparseLt.so.0 otherwise)
if [ "$JP" = 6 ] && ! ldconfig -p | grep -q libcusparseLt; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update
  sudo apt-get install -y libcusparselt0 libcusparselt-dev
fi

# --system-site-packages: reuse JetPack's TensorRT python bindings
[ -d .venv ] || python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -U pip
# Same Ultralytics as the PC (tested on Python 3.8 too). Not [export]: that pulls TensorFlow onto aarch64.
pip install "ultralytics==8.4.208" onnx onnxslim
# CUDA-enabled PyTorch + GPU ONNX Runtime for this JetPack (pip's default torch is CPU-only on aarch64)
pip install "${WHEELS[@]}"
# Jetson torch wheels need numpy 1.x, and opencv-python >= 4.12 needs numpy 2 -> pin both together
pip install "numpy==1.23.5" "opencv-python==4.11.0.86"

python3 - <<'EOF'
import numpy, cv2, torch, onnxruntime, ultralytics
print("numpy", numpy.__version__, "| opencv", cv2.__version__, "| ultralytics", ultralytics.__version__)
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
print("onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers())
assert torch.cuda.is_available(), "torch has no CUDA -> wrong wheel for this JetPack"
assert "CUDAExecutionProvider" in onnxruntime.get_available_providers(), "onnxruntime has no GPU support"
EOF
echo "cameras:"; v4l2-ctl --list-devices || true
echo "Setup done. Run: bash $(pwd)/run.sh"
