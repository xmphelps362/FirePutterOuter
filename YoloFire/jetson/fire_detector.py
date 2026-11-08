"""Lit / unlit candle detector.

    from fire_detector import FireDetector
    det = FireDetector("fire.engine", conf=0.3)
    result = det.detect(frame_bgr, frame_id=n)
    for d in result.detections:
        print(d.cls, d.confidence, d.box, d.center)

Works with .pt, .onnx and TensorRT .engine models. Boxes are in pixels of the frame passed in.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    cls: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2)

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    width: int
    height: int
    detections: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "width": self.width,
            "height": self.height,
            "detections": [
                {
                    "class": d.cls,
                    "confidence": round(float(d.confidence), 3),
                    "box": [round(float(v), 1) for v in d.box],
                    "center": [round(float(v), 1) for v in d.center],
                }
                for d in self.detections
            ],
        }, separators=(",", ":"))


class FireDetector:
    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.3, device=0):
        from ultralytics import YOLO

        self.model = YOLO(model_path, task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float | None = None,
               color: str = "bgr") -> FrameResult:
        if color not in ("bgr", "rgb"):
            raise ValueError("color must be 'bgr' or 'rgb'")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 color frame, got shape {frame.shape}")
        if color == "rgb":
            frame = np.ascontiguousarray(frame[..., ::-1])
        h, w = frame.shape[:2]
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        dets = [
            Detection(res.names[int(k)], float(c), tuple(b))
            for b, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())
        ]
        return FrameResult(frame_id, time.time() if timestamp is None else timestamp, w, h, dets)
