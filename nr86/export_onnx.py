from __future__ import annotations

from pathlib import Path

import torch

from nr86.config import INPUT_CHANNELS
from nr86.models.student import load_student


def export_onnx(ckpt: Path, onnx_path: Path, height: int, width: int) -> Path:
    model = load_student(ckpt, map_location="cpu")
    model.eval()
    dummy = torch.zeros(1, INPUT_CHANNELS, height, width, dtype=torch.float32)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["rgb"],
        opset_version=17,
    )
    print(f"exported {onnx_path}  shape=1x{INPUT_CHANNELS}x{height}x{width}")
    return onnx_path
