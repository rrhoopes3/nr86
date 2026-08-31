"""TensorRT-RTX AOT builder. Optional — needs the NVIDIA SDK.

Follows third_party/TensorRT-RTX/samples/helloWorld/python/hello_world.py:
strongly typed network, serialize engine, JIT on first execute.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from nr86.legal import assert_path_allowed


def build_engine(onnx_path: Path, engine_path: Path, extra: list[str] | None = None) -> Path:
    onnx_path = assert_path_allowed(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cli = shutil.which("tensorrt_rtx") or shutil.which("tensorrt_rtx.exe")
    if cli:
        cmd = [cli, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
        if extra:
            cmd.extend(extra)
        print(" ".join(cmd))
        subprocess.check_call(cmd)
        return engine_path
    return _build_with_python(onnx_path, engine_path)


def _build_with_python(onnx_path: Path, engine_path: Path) -> Path:
    try:
        import tensorrt_rtx as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT-RTX is not installed. Install the NVIDIA SDK and "
            "`pip install tensorrt-rtx`, or put tensorrt_rtx.exe on PATH. "
            "Until then, `python -m nr86 bench` times the PyTorch student."
        ) from exc

    logger = trt.Logger(trt.Logger.INFO)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    builder = trt.Builder(logger)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError(f"ONNX parse failed: {onnx_path}")
    config = builder.create_builder_config()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT-RTX AOT failed")
    engine_path.write_bytes(bytes(serialized))
    print(f"wrote {engine_path}  bytes={engine_path.stat().st_size}")
    return engine_path
