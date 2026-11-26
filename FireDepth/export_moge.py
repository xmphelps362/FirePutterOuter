"""Fixed 640x480 MoGe-2 ONNX export, which the Jetson runs with TensorRT FP16 instead of the slower CUDA backend.
Run on the PC (Python 3.10+, PyTorch 2.5+), then copy FireDepth/models/moge-2-vits.onnx over the Jetson's copy:

  git clone https://github.com/microsoft/MoGe ~/MoGe
  pip install huggingface_hub onnx "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
  python FireDepth/export_moge.py ~/MoGe
"""
import os
import sys
from pathlib import Path

import torch

OUTPUT = Path(__file__).resolve().parent / "models" / "moge-2-vits.onnx"
NUM_TOKENS = 1200

os.environ["XFORMERS_DISABLED"] = "1"  # xformers attention can't be exported
sys.path.insert(0, str(Path(sys.argv[1]).expanduser()))
from moge.model.v2 import MoGeModel  # noqa: E402


class FixedTokenMoGe(MoGeModel):
    def forward(self, image: torch.Tensor):
        return super().forward(image, NUM_TOKENS)


model = FixedTokenMoGe.from_pretrained("Ruicheng/moge-2-vits-normal").eval()
model.onnx_compatible_mode = True
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(model, (torch.rand(1, 3, 480, 640),), str(OUTPUT), input_names=["image"],
                  output_names=["points", "normal", "mask", "scale"], opset_version=17, dynamo=False)
print(OUTPUT)
