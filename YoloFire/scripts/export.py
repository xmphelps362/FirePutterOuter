"""Package a trained model for the Jetson: export/fire.pt, fire.onnx, meta.json.

  python scripts/export.py --weights runs/detect/n640/weights/best.pt --imgsz 640
"""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

TOWER_CONF = 0.3  # start here on the real course; raise only if false alarms appear


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    out = Path("export")
    out.mkdir(exist_ok=True)
    model = YOLO(a.weights)
    shutil.copy(model.export(format="onnx", imgsz=a.imgsz, opset=17, simplify=True), out / "fire.onnx")
    shutil.copy(a.weights, out / "fire.pt")
    meta = {"imgsz": a.imgsz, "names": model.names, "tower_conf": TOWER_CONF, "source_weights": a.weights}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", ", ".join(str(p) for p in sorted(out.iterdir())))


if __name__ == "__main__":
    main()
