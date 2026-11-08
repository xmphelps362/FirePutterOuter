# Candle Fire Detector — Design

Date: 2026-11-08
Owner: Mark (user). Calls marked *(Claude's call)* were not explicitly decided by the user and can be changed.

## Intent

A robot "fire truck" drives a course (layout not known yet) that has candles on it, some lit and some not.
A **watch tower** camera (USB Logitech webcam on a Jetson) sees the whole course, the robot and the candles.
It must find every candle and say whether it is **lit (fire)** or **unlit**, including flagging "likely fire" spots.
The **robot** runs the same model on its own camera to get a close look and confirm a candle is really burning
before it puts it out.

A separate Depth Anything v3 pipeline (built elsewhere, by the user) handles distance / robot position and
"close enough to extinguish". It is **not** part of this project; this project only has to produce detections that
pipeline and the robot code can easily consume.

## Goals

1. Train a **YOLO26** object detector (Ultralytics) on the existing candle dataset, locally, in **WSL Ubuntu** on the
   user's PC (RTX 3060 Ti, 8 GB).
2. Evaluate it honestly on the held-out test split and pick the best variant.
3. Export it to run on the **Jetson's GPU** with TensorRT.
4. A Jetson runtime script: reads the USB webcam live, runs the model, and outputs per-frame detections
   (class, confidence, box) for other programs to use.
5. Make it easy to add new photos (e.g. from the real tower position once the course exists) and retrain.

## MVP scope (user)

**MVP = the model working live on the USB webcam on the Jetson.** Nothing else is required for the MVP.

- Depth Anything v3 runs **on the same Jetson and the same webcam** (user). So the detector must also be usable as an
  importable module that the depth code calls on frames it already captured (no second camera open), and every
  result carries a frame id. The standalone camera script is just a thin wrapper around that module.
- The robot will be a **Raspberry Pi** (user). Robot runtime is **out of scope for the MVP**. Keeping the ONNX export
  around means an NCNN/ONNX build for the Pi can be added later.

## Non-goals

- Robot (Raspberry Pi) runtime — later, not MVP.
- Depth estimation, robot localisation, navigation, extinguishing logic, tower→robot communication protocol
  beyond a simple detection output.
- The classification (crop) dataset in `dataset/classification/` — detection only.
- Labeling tools / relabeling (the README points to CVAT / Label Studio if needed).

## Anti-goals (would be a failure even if it "works")

- Test scores inflated by leaking near-identical burst frames across splits — keep the existing burst split as is.
- Treating flame reflections, the lighter, hands or metal tins as fire (they are intentionally unlabeled negatives).
- Missing real fire is worse than a false alarm: **lit-candle recall matters most** *(Claude's call, follows from the use case)*.

## Constraints and facts

- **Data** (`C:\Users\Marks\Downloads\dataset.zip`, built on a Mac, contains `__MACOSX/` junk to ignore):
  - `dataset/detection/` — YOLO format, `images/{train,val,test}` + `labels/...`, 1280×720 JPG.
  - Classes: `0: lit_candle`, `1: unlit_candle`.
  - Split by burst: train 556 photos (482 lit / 796 unlit boxes), val 123 (120/142), test 118 (126/166).
  - `detection/data.yaml` has a Mac `path:` (`/Users/nickbetters/...`) — must be rewritten.
  - All photos: same tealight type, same tile floors, not taken from a tower position. Boxes are close but not pixel-perfect.
- `all full size training photos.zip` = the same 820 originals (+ a bundled macOS venv). Not needed for training;
  no extra labels.
- **Training machine**: WSL2 Ubuntu (default distro "Ubuntu"), Python 3.12.3, GPU visible via `nvidia-smi`.
  No torch/ultralytics installed yet. Project lives at `~/candle-fire` with its own venv; do not touch other envs.
- **Jetson**: user believes it is an Orin Nano; design assumes **Orin Nano on JetPack 6** (Python 3.10, CUDA 12, TensorRT 10).
  If it turns out to be an original Nano (JetPack 4, `R32` in `/etc/nv_tegra_release`), the runtime part needs the
  Ultralytics JetPack 4 Docker route instead — out of scope unless that happens.
- **Camera**: USB Logitech webcam on the Jetson (V4L2, e.g. `/dev/video0`). Exact model unknown.
- Tower distance: "not that far", course unknown. Candles will look smaller from the tower than in the photos.

## How it gets made (decided)

- Model family: **YOLO26** via Ultralytics, trained in WSL. (User's choice.)
- One trained model shared by tower and robot; each uses its own confidence threshold
  (tower lower = "possible fire", robot higher = "confirmed") *(Claude's call, user agreed with the plan)*.
- Train for distance: strong scale (zoom-out) augmentation; compare input size 640 vs ~960 *(Claude's call)*.
- Model sizes to try: `yolo26n` first, `yolo26s` too; keep whichever is best on the test split
  given Jetson speed *(Claude's call)*.
- Retraining with new photos must be a single documented command *(Claude's call, user agreed)*.

## Deliverables

```
~/candle-fire/
  docs/                    this design (+ later plan)
  data/                    extracted dataset (gitignored), data.yaml with correct local path
  scripts/
    setup_wsl.sh           venv + torch (CUDA) + ultralytics, sanity GPU check
    prepare_data.sh/.py    unzip from Downloads, drop __MACOSX, fix data.yaml, verify counts
    train.py               train run(s), configurable model/imgsz/epochs; one command to retrain
    evaluate.py            test-split metrics per class (precision/recall/mAP), saves report
    export.py              export best.pt -> ONNX (portable) for the Jetson
  jetson/
    setup_jetson.sh        JetPack 6 deps (Jetson torch/torchvision wheels, ultralytics, onnxruntime-gpu as needed)
    build_engine.sh        build TensorRT FP16 engine ON the Jetson (engines are device-specific)
    fire_detector.py       importable module: FireDetector(engine).detect(frame, frame_id) -> detections
    detect_camera.py       live webcam -> detections (thin wrapper around fire_detector.py)
  README.md                step-by-step: setup, train, evaluate, copy to Jetson, run, add photos & retrain
```

### Runtime script behaviour (`detect_camera.py`)

- Opens the USB camera with OpenCV (V4L2, MJPG, configurable device / resolution).
- Runs the TensorRT engine on the GPU; flags: `--conf`, `--imgsz`, `--device`, `--show`, `--save`.
- Per frame outputs: frame id, timestamp, frame size, list of `{class, confidence, box [x1,y1,x2,y2] px, center}`.
- Output sinks: JSON lines to stdout (default) and optional UDP JSON to `--udp host:port`
  *(Claude's call — simplest for the Depth Anything / robot code to read)*. Optional on-screen preview with boxes.
- Prints FPS.
- Optional `--tile` mode (split frame into overlapping tiles) for when candles are tiny from the tower *(Claude's call)*.

### Tower-scale check *(Claude's call)*

All photos are close-ups and the tower distance is unknown, so `evaluate.py` also scores a **shrunk copy of the test
split** (candles padded/downscaled to ~1/2 and ~1/3 size) to show how detection holds up as candles get smaller.
The 640-vs-960 choice uses both the normal and shrunk test results. Real tower photos replace this once the course exists.

### GPU sharing

Depth Anything shares the Orin Nano GPU, so the ≥ 15 FPS target is for the detector alongside it where possible;
this favours `yolo26n` and 640 input unless 960 is clearly better on the shrunk test.

## Success criteria

- Training + evaluation run end-to-end in WSL on the GPU from the README alone.
- Test-split report per class; target **lit_candle recall ≥ 0.95 at the chosen tower threshold** *(Claude's call)*.
- On the Jetson, the script runs live from the webcam on the GPU at a usable frame rate (aim ≥ 15 FPS) *(Claude's call)*.

## Left to the builder

- Exact hyperparameters, epochs, augmentation values, batch size that fits 8 GB.
- Whether 960 input is worth it after measuring.
- Library versions (pin what works), ONNX opset, FP16 vs INT8 (FP16 default).
- Can't test on the Jetson from here — Jetson scripts must be written carefully and verified by the user.
