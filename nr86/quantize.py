"""INT8 calibration and QDQ export for *our* student.

Not FP8→INT8 of NVIDIA's 148M teacher. GroupNorm stays off the INT8
graph (`smoke_int8` / `ampere_int8`). TensorRT-RTX has no `--int8` flag;
it only fuses QuantizeLinear / DequantizeLinear already in the ONNX.
This file writes those QDQ nodes via fake-quant, plus a min/max JSON
that is documentation, not something the builder reads on its own.

INT4 and 2:4 sparsity are not implemented.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr86.dataset import FrameDataset, pack_input, load_frame
from nr86.models.student import (
    ResidualUNet,
    build_student,
    load_student,
    save_student,
)
from nr86.tiles import iter_tiles


def _scale(lo: float, hi: float) -> float:
    return max(abs(float(lo)), abs(float(hi)), 1e-8) / 127.0


class QDQConv2d(nn.Module):
    """Conv with weight + output fake-quant. Input stays FP so packed RGB is not crushed by mvec range."""

    def __init__(self, conv: nn.Conv2d, out_scale: float) -> None:
        super().__init__()
        self.conv = conv
        w = conv.weight.detach()
        self.w_scale = float(w.abs().max().clamp(min=1e-8) / 127.0)
        self.out_scale = float(max(out_scale, 1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = torch.fake_quantize_per_tensor_affine(
            self.conv.weight, self.w_scale, 0, -128, 127
        )
        y = F.conv2d(
            x,
            wq,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
        return torch.fake_quantize_per_tensor_affine(y, self.out_scale, 0, -128, 127)


def transplant_to_int8(
    src_ckpt: Path,
    out_ckpt: Path,
    preset: str = "smoke_int8",
) -> dict:
    """Copy matching conv weights from a GN smoke/ampere ckpt into a no-norm INT8 graph."""
    src = load_student(src_ckpt, map_location="cpu")
    dst = build_student(preset)
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    copied = 0
    skipped = 0
    for key, tensor in dst_sd.items():
        if key in src_sd and src_sd[key].shape == tensor.shape:
            dst_sd[key] = src_sd[key]
            copied += 1
        else:
            skipped += 1
    dst.load_state_dict(dst_sd)
    save_student(dst, out_ckpt)
    payload = {
        "src": str(src_ckpt),
        "out": str(out_ckpt),
        "preset": preset,
        "copied": copied,
        "skipped": skipped,
        "src_norm": src.spec.norm,
        "dst_norm": dst.spec.norm,
    }
    print(
        f"transplant {src_ckpt} -> {out_ckpt}  copied={copied} skipped={skipped}",
        flush=True,
    )
    return payload


@torch.no_grad()
def calibrate(
    ckpt: Path,
    data: Path,
    out: Path,
    max_tiles: int = 64,
) -> dict:
    model = load_student(ckpt, map_location="cpu")
    model.eval()
    ds = FrameDataset(data, require_teacher=False)
    spec = model.spec
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}

    def hook(name: str):
        def _fn(_m, _inp, output: torch.Tensor) -> None:
            t = output.detach()
            lo = float(t.min().cpu())
            hi = float(t.max().cpu())
            mins[name] = lo if name not in mins else min(mins[name], lo)
            maxs[name] = hi if name not in maxs else max(maxs[name], hi)

        return _fn

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.GroupNorm)):
            handles.append(mod.register_forward_hook(hook(name or "root")))

    n = 0
    for rec in ds.rows:
        frame = load_frame(ds.root, rec)
        x = torch.from_numpy(pack_input(frame)).unsqueeze(0)
        h, w = x.shape[-2:]
        for tile in iter_tiles(h, w, spec.tile, spec.overlap):
            chunk = x[:, :, tile.y0 : tile.y1, tile.x0 : tile.x1]
            if chunk.shape[-2] != spec.tile or chunk.shape[-1] != spec.tile:
                continue
            model(chunk)
            n += 1
            if n >= max_tiles:
                break
        if n >= max_tiles:
            break

    for hnd in handles:
        hnd.remove()

    ranges = {k: {"min": mins[k], "max": maxs[k]} for k in mins}
    payload = {
        "ckpt": str(ckpt),
        "tiles_seen": n,
        "preset": model.spec.name,
        "ranges": ranges,
        "note": (
            "Min/max ranges. Consumed by prepare_qdq / export --int8, not by "
            "tensorrt_rtx.exe itself. No INT4, no 2:4 sparsity."
        ),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}  tensors={len(ranges)}  tiles={n}")
    return payload


def prepare_qdq(
    model: ResidualUNet,
    ranges: dict[str, dict[str, float]],
    allow_gn: bool = False,
) -> ResidualUNet:
    """Replace Conv2d with QDQConv2d using calibrated output scales. Mutates `model`."""
    if model.spec.norm != "none" and not allow_gn:
        raise ValueError(
            f"{model.spec.name} uses norm={model.spec.norm!r}. "
            "QDQ is for smoke_int8 / ampere_int8 (norm=none). "
            "Do not blame the calibrator for GroupNorm."
        )
    if model.spec.norm != "none":
        print(
            f"warning: QDQ on {model.spec.name} (norm={model.spec.norm}); "
            "TensorRT may not fuse. This is a measurement, not the happy path.",
            flush=True,
        )

    def _walk(module: nn.Module, prefix: str) -> None:
        for child_name, child in list(module.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, QDQConv2d):
                continue
            if isinstance(child, nn.Conv2d):
                stats = ranges.get(full, {"min": -1.0, "max": 1.0})
                setattr(module, child_name, QDQConv2d(child, _scale(stats["min"], stats["max"])))
            else:
                _walk(child, full)

    _walk(model, "")
    return model


@torch.no_grad()
def prepare_qdq_from_data(
    ckpt: Path,
    data: Path,
    max_tiles: int = 64,
    allow_gn: bool = False,
) -> ResidualUNet:
    calib_path = Path(ckpt).with_suffix(".calib.json")
    payload = calibrate(ckpt, data, calib_path, max_tiles=max_tiles)
    model = load_student(ckpt, map_location="cpu")
    model.eval()
    return prepare_qdq(model, payload["ranges"], allow_gn=allow_gn)
