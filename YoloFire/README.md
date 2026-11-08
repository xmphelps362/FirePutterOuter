# Candle Fire Detector

YOLO26 model that labels candles `lit_candle` or `unlit_candle`. Trained in WSL, runs on the Jetson GPU from the USB webcam.

## PC (WSL)

```bash
bash scripts/setup_wsl.sh && source .venv/bin/activate
python scripts/prepare_data.py          # dataset.zip from Downloads -> data/detection
python scripts/make_small_test.py       # shrunk test sets (candles at 1/2 and 1/3 size)
yolo train cfg=train.yaml               # -> runs/detect/n640/weights/best.pt
python scripts/evaluate.py --weights runs/detect/n640/weights/best.pt --name n640
python scripts/export.py --weights runs/detect/n640/weights/best.pt   # -> export/
pytest -q
```

Results and the model choice: `reports/results.md`. Log every training run in `experiments/journal.md`.

## Jetson (JetPack 5.1 or 6)

```bash
ssh <user>@<jetson> mkdir -p candle-fire
scp -r export jetson <user>@<jetson>:~/candle-fire/
# on the Jetson:
bash ~/candle-fire/jetson/setup_jetson.sh      # once
bash ~/candle-fire/jetson/run.sh               # webcam + bundled export/fire.onnx, with preview
```

Optional, for more FPS: `bash ~/candle-fire/jetson/build_engine.sh ~/candle-fire/export/fire.pt` (on the Jetson),
then `bash ~/candle-fire/jetson/run.sh --model ~/candle-fire/export/fire.engine`.

It prints one JSON line per frame: `frame_id`, `timestamp`, `width`, `height`, and `detections`
(each with `class`, `confidence`, `box` [x1,y1,x2,y2], `center`). Default `--conf` is 0.3; raise it only if you get
false alarms. `--show` needs a monitor (it's skipped over SSH). Other options: `--source`, `--save out.mp4`.

## Calling it from the Depth Anything code

Only one program can open the webcam, so call the detector on frames you already have:

```python
import sys; sys.path.insert(0, "/home/<user>/candle-fire/jetson")
from fire_detector import FireDetector

fire = FireDetector("/home/<user>/candle-fire/export/fire.engine")
result = fire.detect(frame, frame_id)          # BGR frame; pass color="rgb" if yours is RGB
dh, dw = depth.shape[:2]                       # depth map may be a different size than the frame
for d in result.detections:
    if d.cls == "lit_candle":
        cx, cy = d.center
        distance = depth[min(int(cy * dh / result.height), dh - 1), min(int(cx * dw / result.width), dw - 1)]
```

## Retraining with new photos

Take photos from the real tower position, label them in YOLO format (CVAT or Label Studio), keep a copy outside
`data/` (re-running `prepare_data.py` wipes it), copy them into `data/detection/{images,labels}/{train,val}`, then
`yolo train cfg=train.yaml name=<new name>` and repeat evaluate → export → build_engine.

## Other hardware

- JetPack 5.1 (Python 3.8) and 6 (Python 3.10) are both handled by `setup_jetson.sh`.
- Original Jetson Nano (JetPack 4): `setup_jetson.sh` stops; use the `ultralytics/ultralytics:latest-jetson-jetpack4` Docker image.
- Raspberry Pi robot (later): use `export/fire.onnx`, or `yolo export model=export/fire.pt format=ncnn` for more speed.
