import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "jetson" / "detect_camera.py"


@pytest.fixture
def img_dir(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"f{i}.jpg"), np.full((360, 640, 3), 30 * i, np.uint8))
    return tmp_path


def run(*args, env=None):
    return subprocess.run([sys.executable, str(SCRIPT), "--model", "yolo26n.pt", *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=180, env=env)


def test_runs_on_image_dir(img_dir):
    p = run("--source", str(img_dir), "--max-frames", "3")
    assert p.returncode == 0, p.stderr
    lines = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    assert [l["frame_id"] for l in lines] == [0, 1, 2]
    assert all(l["width"] == 640 and l["height"] == 360 for l in lines)


def test_default_model_is_bundled_onnx(img_dir):
    p = subprocess.run([sys.executable, str(SCRIPT), "--source", str(img_dir), "--max-frames", "1", "--device", "cpu"],
                       cwd="/", capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stderr
    assert "fire.onnx" in p.stderr
    assert json.loads(p.stdout.splitlines()[-1])["frame_id"] == 0


def test_bad_source_exits_2():
    p = run("--source", "/dev/video99")
    assert p.returncode == 2
    assert "could not open camera/source: /dev/video99" in p.stderr


def test_show_without_display_disables_preview(img_dir):
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    p = run("--source", str(img_dir), "--max-frames", "2", "--show", env=env)
    assert p.returncode == 0, p.stderr
    assert "no display" in p.stderr
    assert len([l for l in p.stdout.splitlines() if l.startswith("{")]) == 2
