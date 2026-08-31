"""INT8 calibration hooks.

TensorRT-RTX wants strongly-typed ONNX. We collect per-tensor min/max on the
student's activations (histogram PTQ) and write a JSON the TRT builder can
consume. Full QDQ graph rewrite lands when `tensorrt_rtx` is installed.

This is not FP8->INT8 of NVIDIA's 148M teacher. It calibrates *our* student.

If INT8 looks worse than FP16 by more than the usual 0.2 dB, suspect
GroupNorm before the calibrator. Use preset `ampere_int8` (norm=none).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nr86.dataset import FrameDataset, pack_input, load_frame
from nr86.models.student import load_student
from nr86.tiles import iter_tiles


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
        "note": "PTQ min/max for our student, not NVIDIA NR tensors.",
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}  tensors={len(ranges)}  tiles={n}")
    return payload
