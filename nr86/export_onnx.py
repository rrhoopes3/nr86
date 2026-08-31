from __future__ import annotations

from pathlib import Path

import torch

from nr86.config import INPUT_CHANNELS
from nr86.models.student import load_student


def export_onnx(
    ckpt: Path,
    onnx_path: Path,
    height: int,
    width: int,
    int8: bool = False,
    calib_data: Path | None = None,
) -> Path:
    if int8:
        if calib_data is None:
            raise ValueError("INT8 export needs calib_data (a taught dump)")
        from nr86.quantize import prepare_qdq_from_data

        peek = load_student(ckpt, map_location="cpu")
        model = prepare_qdq_from_data(
            ckpt, calib_data, allow_gn=peek.spec.norm != "none"
        )
    else:
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
    kind = "int8-qdq" if int8 else "fp32"
    print(f"exported {onnx_path}  shape=1x{INPUT_CHANNELS}x{height}x{width}  {kind}")
    return onnx_path
