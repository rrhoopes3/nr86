from __future__ import annotations

import shutil
import subprocess
from typing import Any


def _try_nvidia_smi() -> dict[str, Any] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    line = out.strip().splitlines()[0]
    name, cap, mem, driver = [p.strip() for p in line.split(",")]
    major, minor = (int(x) for x in cap.split("."))
    return {
        "name": name,
        "compute": (major, minor),
        "vram_mib": float(mem),
        "driver": driver,
    }


def _try_torch() -> dict[str, Any]:
    info: dict[str, Any] = {"torch": None, "cuda": False, "torch_device": None}
    try:
        import torch
    except ImportError:
        return info
    info["torch"] = torch.__version__
    info["cuda"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["torch_device"] = torch.cuda.get_device_name(0)
        info["torch_compute"] = tuple(torch.cuda.get_device_capability(0))
        props = torch.cuda.get_device_properties(0)
        info["torch_vram_mib"] = round(props.total_memory / (1024**2), 1)
    return info


def ampere_features(compute: tuple[int, int] | None) -> dict[str, bool]:
    """Ampere (8.6) has INT8 MMA + 2:4 sparsity. FP8 starts at Ada (8.9)."""
    if compute is None:
        return {"fp16": False, "int8": False, "sparsity_2_4": False, "fp8": False}
    major, minor = compute
    sm = major * 10 + minor
    return {
        "fp16": sm >= 70,
        "int8": sm >= 61,
        "sparsity_2_4": sm >= 80,
        "fp8": sm >= 89,
    }


def which_tools() -> dict[str, str | None]:
    return {
        "tensorrt_rtx": shutil.which("tensorrt_rtx") or shutil.which("tensorrt_rtx.exe"),
        "nvcc": shutil.which("nvcc"),
        "cmake": shutil.which("cmake"),
    }


def try_import(name: str) -> str | None:
    try:
        mod = __import__(name)
    except ImportError:
        return None
    return getattr(mod, "__version__", "present")


def doctor() -> dict[str, Any]:
    gpu = _try_nvidia_smi()
    torch_info = _try_torch()
    compute = None
    if gpu:
        compute = gpu["compute"]
    elif torch_info.get("torch_compute"):
        compute = torch_info["torch_compute"]
    feats = ampere_features(compute)
    return {
        "gpu": gpu,
        "torch": torch_info,
        "features": feats,
        "tools": which_tools(),
        "python_mods": {
            "onnx": try_import("onnx"),
            "onnxruntime": try_import("onnxruntime"),
            "tensorrt_rtx": try_import("tensorrt_rtx"),
            "cv2": try_import("cv2"),
        },
        "warning": None
        if feats["fp8"]
        else (
            "This GPU has no FP8 MMA. The leaked NR teacher is an FP8 network; "
            "casting it to FP16 is a compatibility port, not an Ampere engine. "
            "nr86 is a research scaffold aimed at INT8 / sparsity later; "
            "those paths are not implemented yet."
        ),
    }
