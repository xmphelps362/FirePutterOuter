"""Score a model on the test split and the shrunk tower-scale sets.

  python scripts/evaluate.py --weights runs/detect/n640/weights/best.pt --name n640
  python scripts/evaluate.py --compare reports/*.json      # table -> reports/results.md

Per set: mAP50 per class from Ultralytics val, plus lit-candle recall/precision at fixed
confidences 0.25 and 0.4 (the settings you'd actually run the tower at).
"""
import argparse
import json
from pathlib import Path

import yaml

SETS = {"test": "data/detection/data.yaml", "small_k2": "data/small_k2.yaml", "small_k3": "data/small_k3.yaml"}
CONFS = (0.25, 0.4)
LIT = 0
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def match_predictions(preds, gts, thr=0.5):
    """preds [(conf, box)], gts [box] -> (confs, is_tp), greedy by confidence, one GT per prediction."""
    used, confs, tps = set(), [], []
    for c, box in sorted(preds, key=lambda p: -p[0]):
        best = max(((iou(box, g), i) for i, g in enumerate(gts) if i not in used), default=(0.0, -1))
        hit = best[0] >= thr
        if hit:
            used.add(best[1])
        confs.append(c)
        tps.append(hit)
    return confs, tps


def pr_at(confs, is_tp, n_gt, conf):
    """(precision or None if nothing kept, recall) for predictions with confidence >= conf."""
    kept = [t for c, t in zip(confs, is_tp) if c >= conf]
    return (sum(kept) / len(kept) if kept else None), (sum(kept) / n_gt if n_gt else 0.0)


def lit_scores(model, data_yaml, imgsz):
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    img_dir = Path(cfg["path"]) / cfg["test"]
    lab_dir = Path(str(img_dir).replace("/images/", "/labels/"))
    confs, tps, n_gt = [], [], 0
    for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        r = model.predict(str(img), imgsz=imgsz, conf=min(CONFS), verbose=False)[0]
        h, w = r.orig_shape
        gts = []
        lab = lab_dir / (img.stem + ".txt")
        for line in (lab.read_text().splitlines() if lab.exists() else []):
            k, cx, cy, bw, bh = line.split()
            if int(k) == LIT:
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                gts.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        n_gt += len(gts)
        preds = [(c, b) for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
                 if int(k) == LIT]
        c, t = match_predictions(preds, gts)
        confs += c
        tps += t
    out = {}
    for conf in CONFS:
        p, rec = pr_at(confs, tps, n_gt, conf)
        out[f"lit_R@{conf}"], out[f"lit_P@{conf}"] = round(rec, 4), (None if p is None else round(p, 4))
    return out


def evaluate(weights, imgsz, name):
    from ultralytics import YOLO

    model = YOLO(weights)
    report = {"name": name, "weights": weights, "imgsz": imgsz, "sets": {}}
    for set_name, data in SETS.items():
        m = model.val(data=data, split="test", imgsz=imgsz, plots=False, verbose=False,
                      project=str(Path("runs/val").resolve()), name=f"{name}_{set_name}", exist_ok=True)
        ap50 = {model.names[int(k)]: round(float(m.box.class_result(i)[2]), 4) for i, k in enumerate(m.box.ap_class_index)}
        report["sets"][set_name] = {"mAP50": ap50, **lit_scores(model, data, imgsz)}
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/{name}.json").write_text(json.dumps(report, indent=1))
    return report


def table(reports):
    rows = ["| run | set | lit R @0.25 | lit P @0.25 | lit R @0.4 | lit P @0.4 | lit mAP50 | unlit mAP50 |",
            "|---|---|---|---|---|---|---|---|"]
    for rep in reports:
        for s, v in rep["sets"].items():
            rows.append(f"| {rep['name']} | {s} | {v['lit_R@0.25']} | {v['lit_P@0.25']} | {v['lit_R@0.4']} | "
                        f"{v['lit_P@0.4']} | {v['mAP50'].get('lit_candle')} | {v['mAP50'].get('unlit_candle')} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name")
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    if a.compare:
        t = table([json.loads(Path(p).read_text()) for p in a.compare])
        res = Path("reports/results.md")
        head = res.read_text().split("<!-- table -->")[0] if res.exists() else "# Results\n\n"
        res.write_text(head + "<!-- table -->\n" + t + "\n")
        print(t)
    elif a.weights and a.name:
        print(table([evaluate(a.weights, a.imgsz, a.name)]))
    else:
        ap.error("give --weights and --name, or --compare")


if __name__ == "__main__":
    main()
