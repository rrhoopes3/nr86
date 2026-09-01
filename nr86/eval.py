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

QUIET_DB = 0.25
MOTION_DB = 0.0
OVERLAY_DB = 0.0


STORM_FRAC = 0.30


def infer_regime(paths: dict, fill: float | None, executed_frac: float) -> str:
    """Quiet gameplay, motion storm, or overlay pass-through."""
    if int(paths.get("passthrough") or 0) > 0:
        return "overlay"
    n = sum(int(v) for v in paths.values()) or 1
    storm_n = int(paths.get("storm_identity") or 0) + int(paths.get("storm") or 0)
    # Sustained storm-identity is policy 0.0. A brief hitch on a quiet
    # lobby (few storm frames) does not lower the +0.25 bar.
    if storm_n / n >= STORM_FRAC:
        return "motion"
    if executed_frac >= 0.85 and fill is not None and fill >= 0.10:
        return "motion"
    return "quiet"


def quality_gate(delta: float, regime: str) -> tuple[bool, str]:
    if regime == "overlay":
        need = OVERLAY_DB
        ok = delta >= need
        if ok:
            return True, "pass"
        return False, f"fail: overlay pass-through Δ {delta:+.3f} < 0 (must be identity)"
    need = MOTION_DB if regime == "motion" else QUIET_DB
    ok = delta >= need
    if ok:
        return True, "pass"
    if regime == "motion":
        return False, f"fail: motion-storm Δ {delta:+.3f} < {need:.2f} (must not go negative)"
    return False, "fail: student does not beat identity — fast at doing nothing"


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
    storm: bool = True,
    envelope: dict | None = None,
    envelope_path: Path | None = None,
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
        storm=storm,
        envelope=envelope,
        envelope_path=envelope_path,
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
        "storm": storm,
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
        "executed_frac": round(float(np.sum(executed) / max(float(np.sum(totals)), 1.0)), 4),
        "regime": None,
        "gate_db": None,
        "beats_identity": False,
        "gate": "fail",
    }
    delta = float(report["delta_psnr"])
    regime = infer_regime(report["paths"], report["mask_fill_mean"], report["executed_frac"])
    ok, gate = quality_gate(delta, regime)
    report["regime"] = regime
    report["gate_db"] = {"quiet": QUIET_DB, "motion": MOTION_DB, "overlay": OVERLAY_DB}[regime]
    report["beats_identity"] = ok
    report["gate"] = gate
    print(json.dumps(report, indent=2))
    return report
