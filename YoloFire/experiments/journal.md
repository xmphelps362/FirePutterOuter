# Experiment journal

Append one entry per training run. Before: what you change and what you expect. After: result and what you learned.

## 2026-11-08 — n640 (baseline)
- Change: yolo26n, imgsz 640, train.yaml augmentation (scale 0.7 for tower distance)
- Expect: high scores; same room for train and test
- Result: lit recall 1.00 / 0.99 / 0.99 at conf 0.25 (test / ½ size / ⅓ size), precision 0.98–0.99. Stopped at epoch 98 (best 56)
- Learned: chosen model. Test photos are optimistic; needs real tower photos

## 2026-11-08 — n960
- Change: imgsz 960
- Expect: better on small candles
- Result: no better on lit candles, lower precision on test (0.91 at conf 0.25), about 2× the compute
- Learned: the candles aren't small enough for 960 to help

## 2026-11-08 — s640
- Change: yolo26s (bigger model)
- Expect: slightly better overall
- Result: about the same as n640, about 3× the compute
- Learned: nano is enough for this data
