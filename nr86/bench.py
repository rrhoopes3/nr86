from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from nr86.config import INPUT_CHANNELS, Placement
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
            "Eager PyTorch: every tile, every iter. every_n is recorded only. "
            "Not a compute-saving measurement. Use --data for skip / dirty-tile ms. "
            "TRT-RTX numbers replace this once the SDK is installed."
        ),
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
) -> dict:
    """Time the path that actually skips frames and dirty tiles."""
    from nr86.dataset import FrameDataset, load_frame, pack_input
    from nr86.runtime import run_frame

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
        engine_path = engine or ensure_engine(ckpt, h, w)
        model = load_trt_student(engine_path, ckpt)
        backend = "tensorrt_rtx"
    else:
        model = load_student(ckpt, map_location=device).to(device).eval()
    spec = model.spec
    params = count_params(load_student(ckpt, map_location="cpu")) if backend == "tensorrt_rtx" else count_params(model)
    packed = [
        torch.from_numpy(pack_input(fr)).unsqueeze(0).to(device) for fr in frames
    ]

    def run_pass() -> tuple[int, int, list[float]]:
        prev_rgb = None
        prev_out = None
        exec_tiles = 0
        total_tiles = 0
        fills: list[float] = []
        for i, (fr, x) in enumerate(zip(frames, packed)):
            pred, stats = run_frame(
                model,
                x,
                color=fr.color,
                mvec=fr.mvec,
                prev_color=prev_rgb,
                prev_out=prev_out,
                frame_index=i,
                every_n=every_n,
                tile=spec.tile,
                overlap=spec.overlap,
                dirty_tiles=dirty_tiles,
            )
            exec_tiles += stats.tiles_executed
            total_tiles += stats.tiles_total
            fills.append(stats.mask_fill)
            prev_rgb = fr.color
            prev_out = pred
        return exec_tiles, total_tiles, fills

    last = run_pass()
    times = _time_fn(lambda: run_pass(), iters, warmup, device)
    exec_tiles, total_tiles, fills = last
    seq_mean = sum(times) / len(times)
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
        "frames": n,
        "every_n": every_n,
        "dirty_tiles": dirty_tiles,
        "tile": spec.tile,
        "tiles_executed": exec_tiles,
        "tiles_total": total_tiles,
        "tiles_executed_mean": round(exec_tiles / max(n, 1), 3),
        "executed_frac": round(exec_tiles / max(total_tiles, 1), 4),
        "mask_fill_mean": round(float(np.mean(fills)), 3) if fills else None,
        "warmup": warmup,
        "iters": iters,
        "sequence_mean_ms": round(seq_mean, 3),
        "mean_ms": round(seq_mean / max(n, 1), 3),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))] / max(n, 1), 3),
        "min_ms": round(times[0] / max(n, 1), 3),
        "vram_mb": vram,
        "note": (
            f"{backend} student with actual skip-frame and dirty-tile execution. "
            "mean_ms is per frame. tiles_executed is over one pass of the sequence."
        ),
    }
