import torch
from ultralytics import YOLO


def test_cuda_and_yolo26():
    assert torch.cuda.is_available()
    model = YOLO("yolo26n.pt")
    assert model.task == "detect"
