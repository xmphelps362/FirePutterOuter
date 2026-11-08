import cv2
import numpy as np
import pytest

from make_small_test import build, mosaic_labels


def test_mosaic_labels_k2():
    tiles = [[], [], [], [(0, 0.5, 0.5, 0.2, 0.2)]]
    out = mosaic_labels(tiles, 2)
    assert len(out) == 1
    assert out[0] == pytest.approx((0, 0.75, 0.75, 0.1, 0.1))


def test_mosaic_labels_k3_tile0():
    tiles = [[(1, 0.3, 0.6, 0.3, 0.3)]] + [[] for _ in range(8)]
    out = mosaic_labels(tiles, 3)
    assert out[0] == pytest.approx((1, 0.1, 0.2, 0.1, 0.1))


def test_build_outputs_1280x720(tmp_path):
    test_dir = tmp_path / "detection"
    (test_dir / "images" / "test").mkdir(parents=True)
    (test_dir / "labels" / "test").mkdir(parents=True)
    n_boxes = 0
    for i in range(5):
        cv2.imwrite(str(test_dir / "images" / "test" / f"p{i}.jpg"), np.full((720, 1280, 3), i * 40, np.uint8))
        lines = [f"{i % 2} 0.5 0.5 0.1 0.1"] * (i + 1)
        n_boxes += len(lines)
        (test_dir / "labels" / "test" / f"p{i}.txt").write_text("\n".join(lines) + "\n")
    out = tmp_path / "small_k2"
    n = build(test_dir, out, 2)
    assert n == 2
    imgs = sorted((out / "images" / "test").glob("*.jpg"))
    assert len(imgs) == 2
    for p in imgs:
        assert cv2.imread(str(p)).shape == (720, 1280, 3)
    total = sum(len(l.read_text().split("\n")) - 1 for l in (out / "labels" / "test").glob("*.txt"))
    # 5 images in 2 mosaics of 4 -> 3 wrap-around tiles reuse p0,p1,p2 (1+2+3 boxes)
    assert total == n_boxes + 6
