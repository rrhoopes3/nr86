"""Quality eval: student vs teacher vs identity (cheap color).

If the student does not beat identity, it is fast at doing nothing.
Skip-frame and dirty-tile paths go through runtime.run_frame so the
network is not executed on frames / tiles that should have been skipped.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from nr86.dataset import FrameDataset, apply_ablation, load_frame, pack_input
from nr86.metrics import psnr, ssim
from nr86.models.student import load_student
from nr86.runtime import run_frame


@torch.no_grad()
def evaluate(
    ckpt: Path,
    data: Path,
    max_frames: int = 32,
    every_n: int = 1,
    dirty_tiles: bool = False,
    ablate: str = "none",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_student(ckpt, map_location=device).to(device).eval()
    spec = model.spec
    ds = FrameDataset(data, require_teacher=True)
    n = min(len(ds), max_frames)
    id_psnr: list[float] = []
    st_psnr: list[float] = []
    id_ssim: list[float] = []
    st_ssim: list[float] = []
    fills: list[float] = []
    executed: list[int] = []
    totals: list[int] = []
    paths: list[str] = []
    prev_rgb: np.ndarray | None = None
    prev_out: np.ndarray | None = None

    for i in range(n):
        frame = load_frame(ds.root, ds.rows[i])
        packed = apply_ablation(pack_input(frame), ablate)
        x = torch.from_numpy(packed).unsqueeze(0).to(device)
        pred, stats = run_frame(
            model,
            x,
            color=frame.color,
            mvec=frame.mvec,
            prev_color=prev_rgb,
            prev_out=prev_out,
            frame_index=i,
            every_n=every_n,
            tile=spec.tile,
            overlap=spec.overlap,
            dirty_tiles=dirty_tiles,
        )
        teacher = frame.teacher
        assert teacher is not None
        fills.append(stats.mask_fill)
        executed.append(stats.tiles_executed)
        totals.append(stats.tiles_total)
        paths.append(stats.path)
        id_psnr.append(psnr(frame.color, teacher))
        st_psnr.append(psnr(pred, teacher))
        id_ssim.append(ssim(frame.color, teacher))
        st_ssim.append(ssim(pred, teacher))
        prev_rgb = frame.color
        prev_out = pred

    report = {
        "ckpt": str(ckpt),
        "data": str(data),
        "frames": n,
        "every_n": every_n,
        "dirty_tiles": dirty_tiles,
        "ablate": ablate,
        "identity_psnr": round(float(np.mean(id_psnr)), 3),
        "student_psnr": round(float(np.mean(st_psnr)), 3),
        "delta_psnr": round(float(np.mean(st_psnr) - np.mean(id_psnr)), 3),
        "identity_ssim": round(float(np.mean(id_ssim)), 4),
        "student_ssim": round(float(np.mean(st_ssim)), 4),
        "delta_ssim": round(float(np.mean(st_ssim) - np.mean(id_ssim)), 4),
        "mask_fill_mean": round(float(np.mean(fills)), 3) if fills else None,
        "tiles_executed": int(np.sum(executed)),
        "tiles_total": int(np.sum(totals)),
        "tiles_executed_mean": round(float(np.mean(executed)), 3) if executed else 0.0,
        "paths": dict(Counter(paths)),
        "beats_identity": float(np.mean(st_psnr)) > float(np.mean(id_psnr)) + 0.25,
        "gate": (
            "pass"
            if float(np.mean(st_psnr)) > float(np.mean(id_psnr)) + 0.25
            else "fail: student does not beat identity — fast at doing nothing"
        ),
    }
    print(json.dumps(report, indent=2))
    return report
