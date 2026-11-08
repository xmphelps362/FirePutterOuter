# Results

Chosen model: **n640** (yolo26n, 640 input). It ties the bigger runs on lit candles and is the cheapest to run next to
Depth Anything on the Jetson. See `experiments/journal.md` for why.

Caveats: every photo is the same tealight on the same floor, and the test split was also used to pick the model, so these
numbers are optimistic. The shrunk sets (small_k2/k3) reuse a few test photos to fill their last mosaic.

<!-- table -->
| run | set | lit R @0.25 | lit P @0.25 | lit R @0.4 | lit P @0.4 | lit mAP50 | unlit mAP50 |
|---|---|---|---|---|---|---|---|
| n640 | test | 1.0 | 0.9921 | 1.0 | 0.9921 | 0.995 | 0.9923 |
| n640 | small_k2 | 0.9921 | 0.9921 | 0.9921 | 0.9921 | 0.995 | 0.945 |
| n640 | small_k3 | 0.9921 | 0.9843 | 0.9921 | 0.9843 | 0.9948 | 0.961 |
| n960 | test | 1.0 | 0.913 | 1.0 | 0.9403 | 0.995 | 0.9943 |
| n960 | small_k2 | 1.0 | 1.0 | 1.0 | 1.0 | 0.995 | 0.945 |
| n960 | small_k3 | 0.9921 | 0.9921 | 0.9921 | 0.9921 | 0.9948 | 0.9433 |
| s640 | test | 1.0 | 1.0 | 1.0 | 1.0 | 0.9941 | 0.9785 |
| s640 | small_k2 | 1.0 | 1.0 | 1.0 | 1.0 | 0.995 | 0.945 |
| s640 | small_k3 | 0.9921 | 0.969 | 0.9921 | 0.9766 | 0.9949 | 0.9446 |
