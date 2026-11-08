import zipfile
from pathlib import Path

import yaml

from prepare_data import prepare

MAC_YAML = """path: /Users/someone/Desktop/dataset/detection
train: images/train
val: images/val
test: images/test
names:
  0: lit_candle
  1: unlit_candle
"""


def make_zip(tmp_path: Path) -> Path:
    zp = tmp_path / "dataset.zip"
    files = {
        "train": [("a", "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n"), ("b", "1 0.3 0.3 0.1 0.1\n")],
        "val": [("c", "0 0.5 0.5 0.1 0.1\n")],
        "test": [("d", "")],
    }
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("dataset/detection/data.yaml", MAC_YAML)
        z.writestr("__MACOSX/dataset/detection/images/train/._a.jpg", b"junk")
        z.writestr("dataset/detection/images/train/._a.jpg", b"junk")
        z.writestr("dataset/README.md", "readme")
        for split, items in files.items():
            for name, lab in items:
                z.writestr(f"dataset/detection/images/{split}/{name}.jpg", b"\xff\xd8fakejpg")
                z.writestr(f"dataset/detection/labels/{split}/{name}.txt", lab)
    return zp


def all_files(d: Path):
    return sorted(p for p in d.rglob("*") if p.is_file())


def test_skips_macosx(tmp_path):
    out = tmp_path / "data"
    prepare(make_zip(tmp_path), out)
    names = [p.name for p in all_files(out)]
    assert not any(n.startswith("._") for n in names)
    assert not (out / "__MACOSX").exists()


def test_data_yaml_path_absolute(tmp_path):
    out = tmp_path / "data"
    prepare(make_zip(tmp_path), out)
    cfg = yaml.safe_load((out / "detection" / "data.yaml").read_text())
    assert cfg["path"] == str((out / "detection").resolve())
    assert cfg["names"] == {0: "lit_candle", 1: "unlit_candle"}
    assert cfg["test"] == "images/test"


def test_counts(tmp_path):
    out = tmp_path / "data"
    res = prepare(make_zip(tmp_path), out)
    assert res["train"] == {"images": 2, "lit": 1, "unlit": 2}
    assert res["val"] == {"images": 1, "lit": 1, "unlit": 0}
    assert res["test"] == {"images": 1, "lit": 0, "unlit": 0}


def test_idempotent(tmp_path):
    out = tmp_path / "data"
    zp = make_zip(tmp_path)
    r1 = prepare(zp, out)
    n1 = len(all_files(out))
    r2 = prepare(zp, out)
    assert r1 == r2
    assert len(all_files(out)) == n1
