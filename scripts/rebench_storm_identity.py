"""Warm-GPU TRT re-bench under storm-identity.

1. Smoke-16 FP16 (shipped engine) on room2 + combat.
2. If clocks look honest (not the post-reboot ~22 ms class), junk-weight
   probe24 FP16 / INT8 on the same clips.

Does not train. Writes:
  results/dxhr-q540-storm-identity-latency.json
  results/dxhr-q540-width-storm.json
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nr86.bench import bench_sequence
from nr86.config import BUDGET_SKIP_DIRTY_MEAN_MS, BUDGET_STUDENT_P95_MS
from nr86.models.student import build_student, count_params, save_student

ROOT = Path(__file__).resolve().parents[1]
ROOM2 = ROOT / "datasets" / "q540-dxhr-room2"
COMBAT = ROOT / "datasets" / "q540-dxhr-combat"
SMOKE_CKPT = ROOT / "runs" / "dxhr-depth-smoke-q540" / "student_best.pt"
SMOKE_ENGINE = ROOT / "engines" / "student_960x540_1ae115ab73.engine"
JUNK = ROOT / "runs" / "_width_probe"
LAT_OUT = ROOT / "results" / "dxhr-q540-storm-identity-latency.json"
WIDTH_OUT = ROOT / "results" / "dxhr-q540-width-storm.json"

# Cold pair we trust (game closed, pre-reboot, all-dirty combat student).
REF_LOBBY = 6.42
REF_COMBAT = 8.004
# Post-reboot contaminated lobby was ~22 ms. Abort well below that.
HONEST_LOBBY_CEILING_MS = 12.0


def _smi() -> dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,power.draw,clocks.current.graphics,"
        "clocks.sm,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(cmd, text=True).strip().split(",")
        keys = ["temp_c", "power_w", "clock_graphics_mhz", "clock_sm_mhz", "util_pct", "mem_mb"]
        out = {}
        for k, v in zip(keys, raw):
            try:
                out[k] = float(v.strip())
            except ValueError:
                out[k] = v.strip()
        return out
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}


def _prime() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(80):
        x = x @ x
    torch.cuda.synchronize()


def _row(ckpt: Path, data: Path, *, int8: bool, every_n: int, dirty: bool, engine=None) -> dict:
    return bench_sequence(
        ckpt,
        data,
        warmup=8,
        iters=8,
        every_n=every_n,
        dirty_tiles=dirty,
        max_frames=32,
        use_trt=True,
        engine=engine,
        int8=int8,
        storm=True,
    )


def _pick(report: dict, *keys: str):
    out = {k: report.get(k) for k in keys}
    return out


KEEP = (
    "mean_ms",
    "p95_ms",
    "min_ms",
    "path_ms",
    "student_path_ms",
    "executed_frac",
    "mask_fill_mean",
    "tiles_executed",
    "tiles_total",
    "backend",
    "engine",
    "budget",
    "storm",
    "every_n",
    "dirty_tiles",
    "frames",
)


def slim(report: dict) -> dict:
    return {k: report.get(k) for k in KEEP}


def main() -> None:
    clocks_idle = _smi()
    print(f"clocks before prime: {clocks_idle}", flush=True)
    _prime()
    clocks_warm = _smi()
    print(f"clocks after prime: {clocks_warm}", flush=True)

    print("=== smoke-16 FP16 room2 skip+dirty ===", flush=True)
    lobby_skip = _row(SMOKE_CKPT, ROOM2, int8=False, every_n=2, dirty=True, engine=SMOKE_ENGINE)
    print(json.dumps(slim(lobby_skip), indent=2), flush=True)

    print("=== smoke-16 FP16 combat skip+dirty ===", flush=True)
    combat_skip = _row(SMOKE_CKPT, COMBAT, int8=False, every_n=2, dirty=True, engine=SMOKE_ENGINE)
    print(json.dumps(slim(combat_skip), indent=2), flush=True)

    print("=== smoke-16 FP16 room2 every-n=1 ===", flush=True)
    lobby_full = _row(SMOKE_CKPT, ROOM2, int8=False, every_n=1, dirty=False, engine=SMOKE_ENGINE)
    print(json.dumps(slim(lobby_full), indent=2), flush=True)

    print("=== smoke-16 FP16 combat every-n=1 ===", flush=True)
    combat_full = _row(SMOKE_CKPT, COMBAT, int8=False, every_n=1, dirty=False, engine=SMOKE_ENGINE)
    print(json.dumps(slim(combat_full), indent=2), flush=True)

    lobby_mean = float(lobby_skip["mean_ms"])
    combat_mean = float(combat_skip["mean_ms"])
    honest = lobby_mean <= HONEST_LOBBY_CEILING_MS
    identity_paths = {}
    for name, rep in (("lobby_skip", lobby_skip), ("combat_skip", combat_skip)):
        pm = rep.get("path_ms") or {}
        identity_paths[name] = {
            k: pm.get(k) for k in ("storm_identity", "warp_clean", "passthrough") if k in pm
        }

    lat = {
        "scene": "warm-GPU storm-identity TRT re-bench (dxhr closed)",
        "ckpt": str(SMOKE_CKPT),
        "engine": str(SMOKE_ENGINE),
        "dxhr_open_during_bench": False,
        "clocks_before_prime": clocks_idle,
        "clocks_after_prime": clocks_warm,
        "clocks_after_smoke": _smi(),
        "reference_cold_all_dirty": {
            "lobby_skip_dirty_mean_ms": REF_LOBBY,
            "combat_skip_dirty_mean_ms": REF_COMBAT,
            "note": "Pre-v0.1; combat ran the student every dirty frame.",
        },
        "lobby_skip": slim(lobby_skip),
        "combat_skip": slim(combat_skip),
        "lobby_every_n_1": slim(lobby_full),
        "combat_every_n_1": slim(combat_full),
        "identity_paths": identity_paths,
        "honest_clocks": honest,
        "honest_rule": f"lobby skip+dirty mean <= {HONEST_LOBBY_CEILING_MS} (contaminated class was ~22 ms)",
        "do_not_quality_claim": ["post_reboot_22ms_bench"],
        "do_not": ["ampere400", "shrink_below_960x540"],
    }
    if honest:
        lat["lesson"] = (
            f"Lobby skip {lobby_mean} vs cold {REF_LOBBY}. "
            f"Combat skip {combat_mean} vs cold all-dirty {REF_COMBAT}. "
            "Storm-identity should make combat cheaper if identity is not 17 ms class."
        )
    else:
        lat["status"] = "contaminated"
        lat["lesson"] = (
            f"Lobby skip {lobby_mean} ms exceeds honesty ceiling "
            f"{HONEST_LOBBY_CEILING_MS}. Same failure class as post-reboot ~22 ms. "
            "Do not cite. Do not run the width map."
        )
        lat["next"] = "retry_when_gpu_idle_and_clocks_boosted"
    LAT_OUT.write_text(json.dumps(lat, indent=2), encoding="utf-8")
    print(json.dumps({k: lat[k] for k in lat if k not in {
        "lobby_skip", "combat_skip", "lobby_every_n_1", "combat_every_n_1"
    }}, indent=2), flush=True)

    if not honest:
        return

    JUNK.mkdir(parents=True, exist_ok=True)
    cells = []
    for preset, int8 in (("probe24", False), ("probe24_int8", True)):
        model = build_student(preset)
        ckpt = JUNK / f"{preset}_junk.pt"
        save_student(model, ckpt)
        print(f"=== {preset} params={count_params(model):,} int8={int8} ===", flush=True)
        skip = _row(ckpt, ROOM2, int8=int8, every_n=2, dirty=True)
        combat = _row(ckpt, COMBAT, int8=int8, every_n=2, dirty=True)
        mean = float(skip["mean_ms"])
        cmean = float(combat["mean_ms"])
        p95 = float((skip.get("student_path_ms") or {}).get("p95_ms") or 0.0)
        cp95 = float((combat.get("student_path_ms") or {}).get("p95_ms") or 0.0)
        # Binding is the clip that still runs the student the most, not
        # all-dirty combat (storm-identity makes that free).
        bind_mean = max(mean, cmean)
        bind_p95 = max(p95, cp95) if (p95 or cp95) else 0.0
        cell = {
            "preset": preset,
            "base": model.spec.base,
            "levels": model.spec.levels,
            "norm": model.spec.norm,
            "precision": "int8_qdq" if int8 else "fp16",
            "params": count_params(model),
            "skip_dirty_mean_ms": mean,
            "combat_skip_dirty_mean_ms": cmean,
            "skip_student_path_p95_ms": p95 or None,
            "combat_student_path_p95_ms": cp95 or None,
            "fits_mean_8_33": bind_mean <= BUDGET_SKIP_DIRTY_MEAN_MS,
            "fits_student_p95_16_67": (bind_p95 <= BUDGET_STUDENT_P95_MS) if bind_p95 else True,
            "binding": "combat_skip" if cmean >= mean else "lobby_skip",
            "lobby_skip": slim(skip),
            "combat_skip": slim(combat),
            "note": (
                "junk weights, timing only. Binding is worst skip+dirty mean "
                "under storm-identity — not all-dirty combat student."
            ),
        }
        cells.append(cell)
        print(json.dumps({k: cell[k] for k in cell if k not in {"lobby_skip", "combat_skip"}}, indent=2), flush=True)

    affordable = [
        c["preset"]
        for c in cells
        if c["fits_mean_8_33"] and c["fits_student_p95_16_67"]
    ]
    width = {
        "resolution": "960x540",
        "weights": "random init, never trained",
        "policy": "storm_identity",
        "cells": cells,
        "affordable_under_8_33_16_67": affordable,
        "clocks_after": _smi(),
        "do_not": ["ampere400", "shrink_below_960x540"],
        "do_not_quality_claim": ["junk_qdq_width_probe"],
        "status": "done",
    }
    if affordable:
        width["next"] = (
            f"Affordable: {affordable}. Train the smallest fitting GN graph "
            "against quiet +0.25 / motion 0.0. Do not train from this file alone."
        )
        width["lesson"] = (
            "Under storm-identity a wider INT8 cell can be gate-legal on ms. "
            "That is still not a quality claim."
        )
    else:
        width["do_not"].insert(1, "grow_width")
        width["lesson"] = (
            "No probe24 cell fits 8.33/16.67 even after storms went free. "
            "Keep grow_width. Do not train probe24."
        )
        width["next"] = "stay_smoke16"
    WIDTH_OUT.write_text(json.dumps(width, indent=2), encoding="utf-8")
    print(json.dumps({k: width[k] for k in width if k != "cells"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
