"""TensorRT-RTX student with the same forward() as ResidualUNet.

This is the thing that can beat 10 ms PyTorch full-frame. run_frame already
calls model(packed); swap the module, keep the skip/mask control flow.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from nr86.config import StudentSpec
from nr86.models.student import load_student


class TrtStudent(nn.Module):
    def __init__(self, engine_path: Path, spec: StudentSpec) -> None:
        super().__init__()
        import tensorrt_rtx as trt

        self.spec = spec
        self.engine_path = Path(engine_path)
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        blob = self.engine_path.read_bytes()
        engine = self._runtime.deserialize_cuda_engine(blob)
        if engine is None:
            raise RuntimeError(f"failed to deserialize {self.engine_path}")
        self._engine = engine
        cfg = engine.create_runtime_config()
        ctx = engine.create_execution_context(cfg)
        if ctx is None:
            raise RuntimeError("failed to create TRT execution context")
        self._ctx = ctx
        self.input_name = None
        self.output_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if int(mode) == int(trt.TensorIOMode.INPUT):
                self.input_name = name
            else:
                self.output_name = name
        if not self.input_name or not self.output_name:
            raise RuntimeError("engine is missing input or output tensor")
        shape = tuple(engine.get_tensor_shape(self.input_name))
        self.in_shape = shape
        self.height = int(shape[-2])
        self.width = int(shape[-1])
        self._in: torch.Tensor | None = None
        self._out: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW, got {tuple(x.shape)}")
        if int(x.shape[-2]) != self.height or int(x.shape[-1]) != self.width:
            raise ValueError(
                f"TRT engine is {self.width}x{self.height}, got {int(x.shape[-1])}x{int(x.shape[-2])}"
            )
        x = x.contiguous()
        if self._in is None or self._in.shape != x.shape or self._in.device != x.device:
            self._in = torch.empty_like(x)
            self._out = torch.empty(x.shape[0], 3, self.height, self.width, device=x.device, dtype=x.dtype)
            self._ctx.set_tensor_address(self.input_name, self._in.data_ptr())
            self._ctx.set_tensor_address(self.output_name, self._out.data_ptr())
        self._in.copy_(x)
        stream = torch.cuda.current_stream().cuda_stream
        if not self._ctx.execute_async_v3(stream):
            raise RuntimeError("TensorRT-RTX execute_async_v3 failed")
        return self._out


def load_trt_student(engine_path: Path, ckpt: Path) -> TrtStudent:
    pt = load_student(ckpt, map_location="cpu")
    return TrtStudent(engine_path, pt.spec)
