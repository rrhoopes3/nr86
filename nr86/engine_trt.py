"""TensorRT-RTX AOT builder. Optional — needs the NVIDIA SDK.

Follows third_party/TensorRT-RTX/samples/helloWorld/python/hello_world.py:
strongly typed network, serialize engine, JIT on first execute.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

from nr86.legal import assert_path_allowed


def tensorrt_fp16_status() -> dict:
    """Honest availability. Does not invent a millisecond number."""
    cli = shutil.which("tensorrt_rtx") or shutil.which("tensorrt_rtx.exe")
    py = False
    py_ver = None
    try:
        import tensorrt_rtx as trt

        py = True
        py_ver = getattr(trt, "__version__", "present")
    except ImportError:
        pass
    available = bool(cli or py)
    return {
        "available": available,
        "cli": cli,
        "python_module": py,
        "python_version": py_ver,
        "precision": "fp16",
        "mean_ms": None,
        "note": (
            "SDK present. This status check does not time an engine. "
            "Build with `nr86 export` + `nr86 trt` for a real FP16 number."
            if available
            else (
                "TensorRT-RTX SDK is not installed. No FP16 engine number. "
                "Install the SDK and put tensorrt_rtx.exe on PATH, then "
                "`nr86 export` + `nr86 trt`. Do not treat ONNX Runtime as TRT."
            )
        ),
    }


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


def engine_path_for(ckpt: Path, height: int, width: int) -> Path:
    """Engine filename includes a ckpt digest so stale HxW files are not reused."""
    digest = hashlib.sha1(Path(ckpt).read_bytes()).hexdigest()[:10]
    return Path("engines") / f"student_{width}x{height}_{digest}.engine"


def ensure_engine(ckpt: Path, height: int, width: int) -> Path:
    """Export ONNX and build a TRT-RTX engine for this ckpt + HxW if missing."""
    from nr86.export_onnx import export_onnx

    ckpt = Path(ckpt)
    out_dir = Path("engines")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_path_for(ckpt, height, width)
    if engine_path.exists() and engine_path.stat().st_size > 0:
        return engine_path
    onnx_path = engine_path.with_suffix(".onnx")
    export_onnx(ckpt, onnx_path, height, width)
    return build_engine(onnx_path, engine_path)


def bench_fp16(ckpt: Path, height: int, width: int, iters: int = 50) -> dict:
    """Export ONNX, build a TRT-RTX engine, time it with the official CLI."""
    from nr86.export_onnx import export_onnx

    status = tensorrt_fp16_status()
    if not status.get("available"):
        return status
    out_dir = Path("engines")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"student_{width}x{height}.onnx"
    engine_path = out_dir / f"student_{width}x{height}.engine"
    export_onnx(ckpt, onnx_path, height, width)
    build_engine(onnx_path, engine_path)
    cli = status.get("cli") or shutil.which("tensorrt_rtx") or shutil.which("tensorrt_rtx.exe")
    if not cli:
        status["note"] = "Engine built via Python; tensorrt_rtx.exe missing so no CLI timing."
        status["engine"] = str(engine_path)
        status["engine_bytes"] = engine_path.stat().st_size
        return status
    cmd = [
        cli,
        f"--loadEngine={engine_path}",
        f"--iterations={iters}",
        "--warmUp=200",
        "--duration=1",
    ]
    print(" ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print(text)
    mean_ms = _parse_trt_mean_ms(text)
    return {
        "available": True,
        "cli": cli,
        "python_module": status.get("python_module"),
        "python_version": status.get("python_version"),
        "precision": "fp16",
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "height": height,
        "width": width,
        "mean_ms": mean_ms,
        "returncode": proc.returncode,
        "note": (
            f"TensorRT-RTX CLI timed {width}x{height} FP16 engine."
            if mean_ms is not None
            else "Engine built; could not parse a millisecond number from CLI output."
        ),
    }


def _parse_trt_mean_ms(text: str) -> float | None:
    for pat in (
        r"GPU Compute Time.*?mean\s*=\s*([0-9.]+)\s*ms",
        r"Latency.*?mean\s*=\s*([0-9.]+)\s*ms",
        r"mean:\s*([0-9.]+)\s*ms",
        r"Average over \d+ iterations: ([0-9.]+) ms",
    ):
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return round(float(m.group(1)), 3)
    return None
