#!/usr/bin/env bash
# One-time WSL setup: venv + CUDA PyTorch + Ultralytics. Run from the project root.
# Uses uv (no sudo needed). Install uv first if missing: https://docs.astral.sh/uv/
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "uv not found (expected ~/.local/bin/uv)" >&2; exit 1; }

[ -d .venv ] || uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
