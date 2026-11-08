"""Extract the YOLO detection dataset from dataset.zip into data/ and fix data.yaml.

Usage: python scripts/prepare_data.py [--zip PATH] [--out data] [--no-check]
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

DEFAULT_ZIP = Path("/mnt/c/Users/Marks/Downloads/dataset.zip")
NAMES = {0: "lit_candle", 1: "unlit_candle"}
SPLITS = ("train", "val", "test")
EXPECTED = {
    "train": {"images": 556, "lit": 482, "unlit": 796},
    "val": {"images": 123, "lit": 120, "unlit": 142},
    "test": {"images": 118, "lit": 126, "unlit": 166},
}
PREFIX = "dataset/detection/"


def _wanted(name: str) -> str | None:
    """Return the path inside detection/ for a zip entry we keep, else None."""
    if name.startswith("__MACOSX/") or not name.startswith(PREFIX) or name.endswith("/"):
        return None
    rel = name[len(PREFIX):]
    base = rel.rsplit("/", 1)[-1]
    if base.startswith("._") or base == ".DS_Store":
        return None
    if rel.startswith(("images/", "labels/")):
        return rel
    return None


def count(det_dir: Path) -> dict:
    res = {}
    for split in SPLITS:
        imgs = [p for p in (det_dir / "images" / split).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        lit = unlit = 0
        for lab in (det_dir / "labels" / split).glob("*.txt"):
            for line in lab.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "0":
                    lit += 1
                elif parts[0] == "1":
                    unlit += 1
        res[split] = {"images": len(imgs), "lit": lit, "unlit": unlit}
    return res


def write_yaml(det_dir: Path) -> None:
    cfg = {
        "path": str(det_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": NAMES,
    }
    (det_dir / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def prepare(zip_path: Path, out_dir: Path) -> dict:
    det_dir = Path(out_dir) / "detection"
    for sub in ("images", "labels"):
        if (det_dir / sub).exists():
            shutil.rmtree(det_dir / sub)  # idempotent: rebuild from the zip
    det_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            rel = _wanted(info.filename)
            if rel is None:
                continue
            dest = det_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
    write_yaml(det_dir)
    return count(det_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--no-check", action="store_true", help="skip the expected-count check (after adding photos)")
    a = ap.parse_args()
    res = prepare(a.zip, a.out)
    for split in SPLITS:
        r = res[split]
        print(f"{split:5s} images={r['images']:4d} lit={r['lit']:4d} unlit={r['unlit']:4d}")
    print("data.yaml:", (a.out / "detection" / "data.yaml").resolve())
    if not a.no_check and res != EXPECTED:
        print("counts differ from the dataset README; use --no-check if that's expected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
