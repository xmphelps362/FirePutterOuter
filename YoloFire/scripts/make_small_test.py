"""Build "tower-scale" test sets: k x k mosaics of test photos, each shrunk by 1/k.

Candles end up 1/2 (k=2) or 1/3 (k=3) of their original size on real backgrounds,
a stand-in for the farther tower view until real tower photos exist.

Usage: python scripts/make_small_test.py --k 2 3
Writes data/small_k{k}/{images,labels}/test and data/small_k{k}.yaml
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

W, H = 1280, 720
NAMES = {0: "lit_candle", 1: "unlit_candle"}
MIN_PX = 2.0


def mosaic_labels(labels_per_tile, k):
    """Map YOLO-normalized boxes of tile i into cell (i // k, i % k) of a k x k grid."""
    out = []
    for i, boxes in enumerate(labels_per_tile):
        row, col = divmod(i, k)
        for c, cx, cy, w, h in boxes:
            out.append((c, (col + cx) / k, (row + cy) / k, w / k, h / k))
    return out


def _read_labels(p: Path):
    if not p.exists():
        return []
    boxes = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def build(test_dir: Path, out_dir: Path, k: int) -> int:
    img_dir = Path(test_dir) / "images" / "test"
    lab_dir = Path(test_dir) / "labels" / "test"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    out_img = Path(out_dir) / "images" / "test"
    out_lab = Path(out_dir) / "labels" / "test"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)
    for old in list(out_img.glob("*")) + list(out_lab.glob("*")):
        old.unlink()
    per = k * k
    n = -(-len(imgs) // per)  # ceil
    tw, th = W // k, H // k
    for m in range(n):
        canvas = np.zeros((th * k, tw * k, 3), np.uint8)
        tiles = []
        for i in range(per):
            src = imgs[(m * per + i) % len(imgs)]  # wrap the last partial mosaic
            im = cv2.resize(cv2.imread(str(src)), (tw, th), interpolation=cv2.INTER_AREA)
            r, c = divmod(i, k)
            canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im
            tiles.append(_read_labels(lab_dir / (src.stem + ".txt")))
        canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
        boxes = [b for b in mosaic_labels(tiles, k) if b[3] * W >= MIN_PX and b[4] * H >= MIN_PX]
        name = f"mosaic_k{k}_{m:04d}"
        cv2.imwrite(str(out_img / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out_lab / f"{name}.txt").write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes))
    return n


def preview(out_dir: Path) -> None:
    p = sorted((out_dir / "images" / "test").glob("*.jpg"))[0]
    im = cv2.imread(str(p))
    for c, x, y, w, h in _read_labels(out_dir / "labels" / "test" / (p.stem + ".txt")):
        x1, y1, x2, y2 = int((x - w / 2) * W), int((y - h / 2) * H), int((x + w / 2) * W), int((y + h / 2) * H)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255) if c == 0 else (255, 200, 0), 2)
    cv2.imwrite(str(out_dir / "preview.jpg"), im)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--data", type=Path, default=Path("data"))
    a = ap.parse_args()
    for k in a.k:
        out = a.data / f"small_k{k}"
        n = build(a.data / "detection", out, k)
        preview(out)
        cfg = {"path": str(out.resolve()), "train": "images/test", "val": "images/test",
               "test": "images/test", "names": NAMES}
        (a.data / f"small_k{k}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"k={k}: {n} mosaics -> {out}")


if __name__ == "__main__":
    main()
