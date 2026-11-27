#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_VERSION="3.12"
REQUIRED_SYSTEM_PACKAGES=(libsm6 libice6)

missing_system_packages=()
for package in "${REQUIRED_SYSTEM_PACKAGES[@]}"; do
    dpkg -s "$package" >/dev/null 2>&1 || missing_system_packages+=("$package")
done
if [ ${#missing_system_packages[@]} -gt 0 ]; then
    echo "OpenCV windows need: sudo apt install ${missing_system_packages[*]}"
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
export VIRTUAL_ENV="$VENV_DIR"

uv pip install torch torchvision
uv pip install -r "$PROJECT_DIR/requirements.txt"

"$VENV_DIR/bin/python" - <<'EOF'
import cv2, onnxruntime, torch
onnxruntime.preload_dlls()
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()} | opencv {cv2.__version__}")
print("onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers())
EOF

echo "Done. Activate with: source $VENV_DIR/bin/activate"
