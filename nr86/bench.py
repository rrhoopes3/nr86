from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from nr86.config import (
    BUDGET_SKIP_DIRTY_MEAN_MS,
    BUDGET_STUDENT_P95_MS,
    INPUT_CHANNELS,
    PRODUCT_INTERNAL_WH,
    Placement,
)
from nr86.models.student import count_params, load_student
from nr86.placement import internal_size
from nr86.tiles import iter_tiles, pad_to_tile, stitch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_fn(fn, n: int, warmup: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    times: list[float] = []
    if device.type == "cuda":
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        for _ in range(n):
            start_ev.record()
            fn()
            end_ev.record()
            end_ev.synchronize()
            times.append(start_ev.elapsed_time(end_ev))
    else:
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times


@torch.no_grad()
def bench_ckpt(
    ckpt: Path,
    size: str,
    warmup: int = 10,
    iters: int = 50,
    scaling_ratio: float = 0.67,
    every_n: int = 1,
    data: Path | None = None,
    dirty_tiles: bool = False,
    max_frames: int = 32,
    try_trt: bool = False,
    use_trt: bool = False,
    engine: Path | None = None,
    int8: bool = False,
    storm: bool = True,
    envelope: dict | None = None,
    envelope_path=None,
) -> dict:
    if data is not None:
        report = bench_sequence(
            ckpt,
            data,
            warmup=warmup,
            iters=iters,
            every_n=every_n,
            dirty_tiles=dirty_tiles,
            max_frames=max_frames,
            use_trt=use_trt,
            engine=engine,
            int8=int8,
            storm=storm,
            envelope=envelope,
            envelope_path=envelope_path,
        )
    else:
        report = _bench_eager(ckpt, size, warmup, iters, scaling_ratio, every_n)
    if try_trt:
        from nr86.engine_trt import bench_fp16, tensorrt_fp16_status

        status = tensorrt_fp16_status()
        if status.get("available"):
            w_s, h_s = size.lower().split("x")
            iw, ih = internal_size(
                Placement(
                    output_w=int(w_s),
                    output_h=int(h_s),
                    scaling_ratio=scaling_ratio,
                )
            )
            report["tensorrt_fp16"] = bench_fp16(ckpt, height=ih, width=iw)
        else:
            report["tensorrt_fp16"] = status
    print(json.dumps(report, indent=2))
    return report


@torch.no_grad()
def _bench_eager(
    ckpt: Path,
    size: str,
    warmup: int,
    iters: int,
    scaling_ratio: float,
    every_n: int,
) -> dict:
    w_s, h_s = size.lower().split("x")
    p = Placement(
        output_w=int(w_s),
        output_h=int(h_s),
        scaling_ratio=scaling_ratio,
        every_n=every_n,
    )
    iw, ih = internal_size(p)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_student(ckpt, map_location=device).to(device)
    model.eval()
    spec = model.spec
    x = torch.rand(1, INPUT_CHANNELS, ih, iw, device=device)
    x, orig_h, orig_w = pad_to_tile(x, spec.tile)
    h, w = x.shape[-2:]
    tiles = iter_tiles(h, w, spec.tile, spec.overlap)

    def run_tiled() -> torch.Tensor:
        chunks = []
        for t in tiles:
            chunk = x[:, :, t.y0 : t.y1, t.x0 : t.x1]
            chunks.append((t, model(chunk)))
        out = stitch(chunks, h, w, spec.overlap)
        return out[:, :, :orig_h, :orig_w]

    def run_full() -> torch.Tensor:
        return model(x[:, :, :orig_h, :orig_w])

    tiled_times = _time_fn(run_tiled, iters, warmup, device)
    full_times = _time_fn(run_full, iters, warmup, device)
    mean_ms = sum(tiled_times) / len(tiled_times)
    p95 = tiled_times[int(0.95 * (len(tiled_times) - 1))]
    full_mean = sum(full_times) / len(full_times)
    vram = None
    if device.type == "cuda":
        vram = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    return {
        "kind": "eager_pytorch_all_tiles",
        "measured_skip": False,
        "ckpt": str(ckpt),
        "preset": spec.name,
        "params": count_params(model),
        "device": str(device),
        "output": [p.output_w, p.output_h],
        "internal": [iw, ih],
        "scaling_ratio": scaling_ratio,
        "every_n": every_n,
        "tile": spec.tile,
        "n_tiles": len(tiles),
        "tiles_executed": len(tiles),
        "warmup": warmup,
        "iters": iters,
        "mean_ms": round(mean_ms, 3),
        "p95_ms": round(p95, 3),
        "min_ms": round(tiled_times[0], 3),
        "fullframe_mean_ms": round(full_mean, 3),
        "fullframe_min_ms": round(full_times[0], 3),
        "vram_mb": vram,
        "note": (
            "Eager PyTorch: every tile, every iter. --size is OUTPUT resolution; "
            "internal = output * scaling. Do not treat fullframe_mean_ms at "
            "858x482 as a 720p budget. Use --data for skip / dirty-tile ms on "
            "the product tensor (1280x720)."
        ),
    }


def _path_summary(times: list[float]) -> dict | None:
    if not times:
        return None
    s = sorted(times)
    n = len(s)
    return {
        "n": n,
        "mean_ms": round(sum(s) / n, 3),
        "p95_ms": round(s[min(n - 1, int(0.95 * (n - 1)))], 3),
        "max_ms": round(s[-1], 3),
        "min_ms": round(s[0], 3),
    }


@torch.no_grad()
def bench_sequence(
    ckpt: Path,
    data: Path,
    warmup: int = 4,
    iters: int = 8,
    every_n: int = 2,
    dirty_tiles: bool = True,
    max_frames: int = 32,
    use_trt: bool = False,
    engine: Path | None = None,
    int8: bool = False,
    storm: bool = True,
    envelope: dict | None = None,
    envelope_path=None,
) -> dict:
    """Time skip/dirty with prev on device. D2H is off. Color/mvec H2D stays on."""
    from nr86.dataset import FrameDataset, load_frame, pack_input
    from nr86.runtime import FrameRunner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FrameDataset(data, require_teacher=False)
    n = min(len(ds), max_frames)
    frames = [load_frame(ds.root, ds.rows[i]) for i in range(n)]
    backend = "pytorch"
    engine_path = engine
    if use_trt or engine is not None:
        if device.type != "cuda":
            raise RuntimeError("TensorRT student needs CUDA")
        from nr86.engine_trt import ensure_engine
        from nr86.trt_student import load_trt_student

        h, w = frames[0].color.shape[:2]
        engine_path = engine or ensure_engine(
            ckpt, h, w, int8=int8, calib_data=data if int8 else None
        )
        model = load_trt_student(engine_path, ckpt)
        backend = "tensorrt_rtx_int8" if int8 else "tensorrt_rtx"
    else:
        model = load_student(ckpt, map_location=device).to(device).eval()
    spec = model.spec
    params = (
        count_params(load_student(ckpt, map_location="cpu"))
        if backend.startswith("tensorrt_rtx")
        else count_params(model)
    )
    packed = [torch.from_numpy(pack_input(fr)).unsqueeze(0).to(device) for fr in frames]
    fh, fw = frames[0].color.shape[:2]
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

    def run_pass(record: bool) -> tuple[int, int, list[float], dict[str, list[float]], list[float]]:
        runner.reset()
        exec_tiles = 0
        total_tiles = 0
        fills: list[float] = []
        by_path: dict[str, list[float]] = {}
        all_ms: list[float] = []
        for i, (fr, x) in enumerate(zip(frames, packed)):
            if device.type == "cuda" and record:
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                _pred, stats = runner.run(
                    x, color=fr.color, mvec=fr.mvec, frame_index=i, to_numpy=False
                )
                end_ev.record()
                end_ev.synchronize()
                ms = float(start_ev.elapsed_time(end_ev))
                all_ms.append(ms)
                by_path.setdefault(stats.path, []).append(ms)
            else:
                _pred, stats = runner.run(
                    x, color=fr.color, mvec=fr.mvec, frame_index=i, to_numpy=False
                )
            exec_tiles += stats.tiles_executed
            total_tiles += stats.tiles_total
            fills.append(stats.mask_fill)
        return exec_tiles, total_tiles, fills, by_path, all_ms

    for _ in range(warmup):
        run_pass(record=False)
    _sync(device)

    combined: dict[str, list[float]] = {}
    all_frames: list[float] = []
    last_exec = last_total = 0
    last_fills: list[float] = []
    for _ in range(iters):
        exec_tiles, total_tiles, fills, by_path, all_ms = run_pass(record=True)
        last_exec, last_total, last_fills = exec_tiles, total_tiles, fills
        all_frames.extend(all_ms)
        for path, vals in by_path.items():
            combined.setdefault(path, []).extend(vals)

    path_ms = {k: _path_summary(v) for k, v in sorted(combined.items())}
    overall = _path_summary(all_frames)
    student_times: list[float] = []
    for key in ("fullframe", "fullframe_dirty"):
        student_times.extend(combined.get(key) or [])
    student = _path_summary(student_times)
    mean_ms = overall["mean_ms"] if overall else None
    student_p95 = student["p95_ms"] if student else None
    mean_applies = every_n > 1 and dirty_tiles
    mean_pass = (
        mean_ms is not None and mean_ms <= BUDGET_SKIP_DIRTY_MEAN_MS if mean_applies else None
    )
    worst_pass = True if student is None else student_p95 <= BUDGET_STUDENT_P95_MS
    gate_bits = []
    if mean_applies:
        gate_bits.append(
            f"skip+dirty mean {mean_ms} vs {BUDGET_SKIP_DIRTY_MEAN_MS} "
            f"{'pass' if mean_pass else 'fail'}"
        )
    if student is not None:
        gate_bits.append(
            f"student-path p95 {student_p95} vs {BUDGET_STUDENT_P95_MS} "
            f"{'pass' if worst_pass else 'fail'}"
        )
    mean_ok = (not mean_applies) or bool(mean_pass)
    latency_gate = "pass" if mean_ok and worst_pass else "fail: " + "; ".join(gate_bits)
    vram = None
    if device.type == "cuda":
        vram = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    return {
        "kind": "measured_skip_and_dirty_tiles",
        "measured_skip": True,
        "ckpt": str(ckpt),
        "data": str(data),
        "preset": spec.name,
        "backend": backend,
        "engine": str(engine_path) if engine_path else None,
        "params": params,
        "device": str(device),
        "product_internal_wh": list(PRODUCT_INTERNAL_WH),
        "frame_wh": [fw, fh],
        "frames": n,
        "every_n": every_n,
        "dirty_tiles": dirty_tiles,
        "storm": storm,
        "tile": spec.tile,
        "tiles_executed": last_exec,
        "tiles_total": last_total,
        "tiles_executed_mean": round(last_exec / max(n, 1), 3),
        "executed_frac": round(last_exec / max(last_total, 1), 4),
        "mask_fill_mean": round(float(np.mean(last_fills)), 3) if last_fills else None,
        "warmup": warmup,
        "iters": iters,
        "mean_ms": mean_ms,
        "p95_ms": overall["p95_ms"] if overall else None,
        "min_ms": overall["min_ms"] if overall else None,
        "path_ms": path_ms,
        "student_path_ms": student,
        "budget": {
            "skip_dirty_mean_ms": BUDGET_SKIP_DIRTY_MEAN_MS,
            "student_p95_ms": BUDGET_STUDENT_P95_MS,
            "mean_applies": mean_applies,
            "mean_pass": mean_pass,
            "student_p95_pass": worst_pass,
            "latency_gate": latency_gate,
        },
        "vram_mb": vram,
        "note": (
            f"{backend} FrameRunner: prev_color/prev_out stay on GPU. "
            "Color/mvec H2D via pinned host buffers each frame (honest numpy harness). "
            "No D2H of RGB. Mean is all frames; student-path p95 is fullframe + "
            "fullframe_dirty only (storm_identity is not a student path). "
            "10.7 ms eager-858x482 is retired."
        ),
    }
