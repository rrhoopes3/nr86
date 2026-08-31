from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from nr86.config import INPUT_CHANNELS, Placement
from nr86.models.student import count_params, load_student
from nr86.placement import internal_size
from nr86.tiles import iter_tiles, pad_to_tile, stitch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def bench_ckpt(
    ckpt: Path,
    size: str,
    warmup: int = 10,
    iters: int = 50,
    scaling_ratio: float = 0.67,
    every_n: int = 1,
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

    def time_fn(fn, n: int) -> list[float]:
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

    tiled_times = time_fn(run_tiled, iters)
    full_times = time_fn(run_full, iters)
    mean_ms = sum(tiled_times) / len(tiled_times)
    p95 = tiled_times[int(0.95 * (len(tiled_times) - 1))]
    full_mean = sum(full_times) / len(full_times)
    vram = None
    if device.type == "cuda":
        vram = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    report = {
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
        "warmup": warmup,
        "iters": iters,
        "mean_ms": round(mean_ms, 3),
        "p95_ms": round(p95, 3),
        "min_ms": round(tiled_times[0], 3),
        "fullframe_mean_ms": round(full_mean, 3),
        "fullframe_min_ms": round(full_times[0], 3),
        "vram_mb": vram,
        "note": "PyTorch student. TRT-RTX engine numbers replace this once SDK is installed.",
    }
    print(json.dumps(report, indent=2))
    return report
