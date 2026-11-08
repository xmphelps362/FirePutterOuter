#!/usr/bin/env bash
# Build the TensorRT FP16 engine ON the Jetson (takes several minutes).
#   bash ~/candle-fire/jetson/build_engine.sh ~/candle-fire/export/fire.pt [imgsz]
# -> fire.engine next to fire.pt (also rewrites fire.onnx next to it)
set -euo pipefail
PT="$(realpath "${1:?usage: build_engine.sh path/to/fire.pt [imgsz]}")"   # resolve before cd
cd "$(dirname "$0")"
META="$(dirname "$PT")/meta.json"
if [ -n "${2:-}" ]; then
  IMGSZ="$2"
elif [ -f "$META" ]; then
  IMGSZ="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['imgsz'])" "$META")"
else
  IMGSZ=640; echo "no $META, using imgsz 640"
fi
[ -n "${VIRTUAL_ENV:-}" ] || source .venv/bin/activate
yolo export model="$PT" format=engine half=True imgsz="$IMGSZ" device=0
echo "engine: ${PT%.pt}.engine (imgsz $IMGSZ)"
