"""Quality eval: student vs teacher vs identity (cheap color).

If the student does not beat identity, it is fast at doing nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nr86.dataset import FrameDataset, load_frame, pack_input
from nr86.metrics import psnr, ssim
from nr86.models.student import load_student
from nr86.reproject import composite, fill_ratio, motion_luma_mask, warp_rgb


@torch.no_grad()
def evaluate(
    ckpt: Path,
    data: Path,
    max_frames: int = 32,
    every_n: int = 1,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_student(ckpt, map_location=device).to(device).eval()
    ds = FrameDataset(data, require_teacher=True)
    n = min(len(ds), max_frames)
    id_psnr: list[float] = []
    st_psnr: list[float] = []
    id_ssim: list[float] = []
    st_ssim: list[float] = []
    fills: list[float] = []
    prev_rgb: np.ndarray | None = None
    prev_out: np.ndarray | None = None

    for i in range(n):
        frame = load_frame(ds.root, ds.rows[i])
        x = torch.from_numpy(pack_input(frame)).unsqueeze(0).to(device)
        pred = model(x)[0].permute(1, 2, 0).cpu().numpy()
        color = frame.color
        teacher = frame.teacher
        assert teacher is not None
        if every_n > 1 and prev_out is not None and (i % every_n) != 0:
            warped = warp_rgb(prev_out, frame.mvec)
            mask = motion_luma_mask(color, prev_rgb, frame.mvec)
            pred = composite(pred, warped, mask)
            fills.append(fill_ratio(mask))
        id_psnr.append(psnr(color, teacher))
        st_psnr.append(psnr(pred, teacher))
        id_ssim.append(ssim(color, teacher))
        st_ssim.append(ssim(pred, teacher))
        prev_rgb = color
        prev_out = pred

    report = {
        "ckpt": str(ckpt),
        "data": str(data),
        "frames": n,
        "every_n": every_n,
        "identity_psnr": round(float(np.mean(id_psnr)), 3),
        "student_psnr": round(float(np.mean(st_psnr)), 3),
        "delta_psnr": round(float(np.mean(st_psnr) - np.mean(id_psnr)), 3),
        "identity_ssim": round(float(np.mean(id_ssim)), 4),
        "student_ssim": round(float(np.mean(st_ssim)), 4),
        "delta_ssim": round(float(np.mean(st_ssim) - np.mean(id_ssim)), 4),
        "mask_fill_mean": round(float(np.mean(fills)), 3) if fills else None,
        "beats_identity": float(np.mean(st_psnr)) > float(np.mean(id_psnr)) + 0.25,
        "gate": (
            "pass"
            if float(np.mean(st_psnr)) > float(np.mean(id_psnr)) + 0.25
            else "fail: student does not beat identity — fast at doing nothing"
        ),
    }
    print(json.dumps(report, indent=2))
    return report
