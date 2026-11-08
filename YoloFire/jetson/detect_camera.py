"""Live lit/unlit candle detection from the USB webcam, a video file or an image folder.

  python3 detect_camera.py --show                       # uses ../export/fire.onnx, webcam 0
  python3 detect_camera.py --source clip.mp4 --save out.mp4

Prints one JSON line per frame (see FrameResult.to_json). FPS goes to stderr. Ctrl+C or q stops.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fire_detector import FireDetector  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "export" / "fire.onnx"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
MAX_FAILED_READS = 5
COLORS = {"lit_candle": (0, 0, 255), "unlit_candle": (255, 200, 0)}


class Source:
    """Webcam (index or /dev/videoN), video file, or folder of images."""

    def __init__(self, src: str, width: int, height: int):
        self.images, self.cap = None, None
        self.live = src.isdigit() or src.startswith("/dev/video")
        if Path(src).is_dir():
            self.images = sorted(p for p in Path(src).iterdir() if p.suffix.lower() in IMG_EXT)
            self.ok = bool(self.images)
            return
        if self.live:
            self.cap = cv2.VideoCapture(int(src) if src.isdigit() else src, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(src)
        self.ok = self.cap.isOpened()

    def read(self):
        if self.images is not None:
            return (True, cv2.imread(str(self.images.pop(0)))) if self.images else (False, None)
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()


def draw(frame, result):
    for d in result.detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        c = COLORS.get(d.cls, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(frame, f"{d.cls} {d.confidence:.2f}", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return frame


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="default: export/fire.onnx next to this folder")
    ap.add_argument("--source", default="0", help="camera index, /dev/videoN, video file or image folder")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default="0")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", help="write annotated video here")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = forever)")
    a = ap.parse_args(argv)

    if a.show and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("--show: no display found (DISPLAY unset, e.g. over SSH); preview disabled", file=sys.stderr)
        a.show = False
    src = Source(a.source, a.width, a.height)
    if not src.ok:
        print(f"could not open camera/source: {a.source}", file=sys.stderr)
        return 2
    print(f"model: {a.model}", file=sys.stderr)
    det = FireDetector(a.model, imgsz=a.imgsz, conf=a.conf, device=int(a.device) if a.device.isdigit() else a.device)

    writer = None
    frame_id, fails, n_sec, t_sec = 0, 0, 0, time.time()
    try:
        while a.max_frames == 0 or frame_id < a.max_frames:
            ok, frame = src.read()
            if not ok or frame is None:
                if not src.live:
                    break
                fails += 1
                if fails >= MAX_FAILED_READS:
                    print(f"camera stopped delivering frames ({fails} failed reads)", file=sys.stderr)
                    return 1
                time.sleep(0.05)
                continue
            fails = 0
            result = det.detect(frame, frame_id)
            print(result.to_json(), flush=True)
            if a.show or a.save:
                draw(frame, result)
            if a.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                writer.write(frame)
            if a.show:
                cv2.imshow("fire", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_id += 1
            n_sec += 1
            if time.time() - t_sec >= 1.0:
                print(f"fps {n_sec / (time.time() - t_sec):.1f}", file=sys.stderr, flush=True)
                n_sec, t_sec = 0, time.time()
    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        if writer is not None:
            writer.release()
        if a.show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
