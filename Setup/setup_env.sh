#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
DA3_DIR="$PROJECT_DIR/depth-anything-3"
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

if [ ! -d "$DA3_DIR" ]; then
    git clone https://github.com/bytedance-seed/depth-anything-3 "$DA3_DIR"
fi

uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
export VIRTUAL_ENV="$VENV_DIR"

uv pip install torch torchvision xformers
uv pip install -e "$DA3_DIR"
uv pip install -r "$PROJECT_DIR/requirements.txt"

"$VENV_DIR/bin/python" - <<'EOF'
import torch, cv2, depth_anything_3
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()} | opencv {cv2.__version__}")
EOF

echo "Done. Activate with: source $VENV_DIR/bin/activate"
