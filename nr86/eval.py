"""Quality eval: student vs teacher vs identity (cheap color).

If the student does not beat identity, it is fast at doing nothing.
Skip-frame and dirty-tile paths go through runtime.FrameRunner so the
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
from nr86.runtime import FrameRunner


@torch.no_grad()
def evaluate(
    ckpt: Path,
    data: Path,
    max_frames: int = 32,
    every_n: int = 1,
    dirty_tiles: bool = False,
    ablate: str = "none",
    use_trt: bool = False,
    engine: Path | None = None,
    offset: int = 0,
    int8: bool = False,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "pytorch"
    if use_trt or engine is not None:
        if device.type != "cuda":
            raise RuntimeError("TensorRT eval needs CUDA")
        from nr86.engine_trt import ensure_engine
        from nr86.trt_student import load_trt_student

        probe = FrameDataset(data, require_teacher=True)
        fr0 = load_frame(probe.root, probe.rows[min(max(0, int(offset)), len(probe) - 1)])
        h, w = fr0.color.shape[:2]
        engine_path = engine or ensure_engine(
            ckpt, h, w, int8=int8, calib_data=data if int8 else None
        )
        model = load_trt_student(engine_path, ckpt)
        backend = "tensorrt_rtx_int8" if int8 else "tensorrt_rtx"
    else:
        model = load_student(ckpt, map_location=device).to(device).eval()
        engine_path = None
    spec = model.spec
    ds = FrameDataset(data, require_teacher=True)
    start = max(0, int(offset))
    n = min(max(0, len(ds) - start), max_frames)
    id_psnr: list[float] = []
    st_psnr: list[float] = []
    id_ssim: list[float] = []
    st_ssim: list[float] = []
    fills: list[float] = []
    executed: list[int] = []
    totals: list[int] = []
    paths: list[str] = []
    runner = FrameRunner(
        model,
        every_n=every_n,
        tile=spec.tile,
        overlap=spec.overlap,
        dirty_tiles=dirty_tiles,
    )

    for i in range(n):
        frame = load_frame(ds.root, ds.rows[start + i])
        packed = apply_ablation(pack_input(frame), ablate)
        x = torch.from_numpy(packed).unsqueeze(0).to(device)
        pred, stats = runner.run(
            x,
            color=frame.color,
            mvec=frame.mvec,
            frame_index=i,
            to_numpy=True,
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

    report = {
        "ckpt": str(ckpt),
        "data": str(data),
        "frames": n,
        "offset": start,
        "every_n": every_n,
        "dirty_tiles": dirty_tiles,
        "ablate": ablate,
        "backend": backend,
        "engine": str(engine_path) if engine_path else None,
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
