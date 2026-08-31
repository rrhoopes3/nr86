"""Timing-only (width × precision) map at 960×540 with junk weights.

Do not train these presets. Builds FP16 and QDQ engines for probe24 /
probe32 (3-level, not ampere 32/4) and benches skip+dirty + every-n=1
on room2. Writes results/dxhr-q540-width-precision.json.

INT8 numbers here are timing only — do not cite as quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nr86.bench import bench_sequence
from nr86.config import BUDGET_SKIP_DIRTY_MEAN_MS, BUDGET_STUDENT_P95_MS
from nr86.models.student import build_student, count_params, save_student

ROOT = Path(__file__).resolve().parents[1]
ROOM2 = ROOT / "datasets" / "q540-dxhr-room2"
COMBAT = ROOT / "datasets" / "q540-dxhr-combat"
OUT = ROOT / "results" / "dxhr-q540-width-precision.json"
JUNK = ROOT / "runs" / "_width_probe"


def _row(ckpt: Path, data: Path, *, int8: bool, every_n: int, dirty: bool) -> dict:
    return bench_sequence(
        ckpt,
        data,
        warmup=4,
        iters=8,
        every_n=every_n,
        dirty_tiles=dirty,
        max_frames=32,
        use_trt=True,
        int8=int8,
        storm=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    JUNK.mkdir(parents=True, exist_ok=True)
    cells = []
    for preset, int8 in (
        ("probe24", False),
        ("probe24_int8", True),
        ("probe32", False),
        ("probe32_int8", True),
    ):
        model = build_student(preset)
        ckpt = JUNK / f"{preset}_junk.pt"
        save_student(model, ckpt)
        print(f"=== {preset} params={count_params(model):,} int8={int8} ===", flush=True)
        skip = _row(ckpt, ROOM2, int8=int8, every_n=2, dirty=True)
        combat = _row(ckpt, COMBAT, int8=int8, every_n=2, dirty=True)
        full = _row(ckpt, ROOM2, int8=int8, every_n=1, dirty=False)
        mean = float(skip["mean_ms"])
        combat_mean = float(combat["mean_ms"])
        p95 = float((skip.get("student_path_ms") or {}).get("p95_ms") or skip["p95_ms"])
        combat_p95 = float((combat.get("student_path_ms") or {}).get("p95_ms") or combat["p95_ms"])
        bind_mean = max(mean, combat_mean)
        bind_p95 = max(p95, combat_p95)
        cell = {
            "preset": preset,
            "base": model.spec.base,
            "levels": model.spec.levels,
            "norm": model.spec.norm,
            "precision": "int8_qdq" if int8 else "fp16",
            "params": count_params(model),
            "skip_dirty_mean_ms": mean,
            "combat_skip_dirty_mean_ms": combat_mean,
            "skip_student_path_p95_ms": p95,
            "combat_student_path_p95_ms": combat_p95,
            "every_n_1_mean_ms": float(full["mean_ms"]),
            "fits_mean_8_33": bind_mean <= BUDGET_SKIP_DIRTY_MEAN_MS,
            "fits_student_p95_16_67": bind_p95 <= BUDGET_STUDENT_P95_MS,
            "binding": "combat_all_dirty" if combat_mean >= mean else "lobby_skip",
            "lobby_skip": {
                "mean_ms": skip["mean_ms"],
                "student_path_ms": skip.get("student_path_ms"),
                "path_ms": skip.get("path_ms"),
            },
            "combat_skip": {
                "mean_ms": combat["mean_ms"],
                "student_path_ms": combat.get("student_path_ms"),
                "path_ms": combat.get("path_ms"),
            },
            "note": "junk weights, timing only. Binding constraint is all-dirty combat.",
        }
        cells.append(cell)
        print(json.dumps(cell, indent=2), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "status": "running",
                    "resolution": "960x540",
                    "data": str(ROOM2),
                    "cells": cells,
                    "do_not_quality_claim": ["junk_qdq_width_probe"],
                    "do_not": ["ampere400", "shrink_below_960x540"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    affordable = [
        c["preset"]
        for c in cells
        if c["fits_mean_8_33"] and c["fits_student_p95_16_67"]
    ]
    payload = {
        "resolution": "960x540",
        "data": str(ROOM2),
        "weights": "random init, never trained",
        "cells": cells,
        "affordable_under_8_33_16_67": affordable,
        "lesson": (
            "Width growth stays frozen for training until a cell here fits. "
            "INT8 at 193k was launch-bound; this map is whether base-24/32 "
            "become math-bound enough for QDQ to pay."
        ),
        "do_not": ["ampere400", "shrink_below_960x540"],
        "do_not_quality_claim": ["junk_qdq_width_probe"],
        "status": "done",
    }
    if not affordable:
        payload["do_not"].insert(1, "grow_width")
        payload["lesson"] += " No wider graph fits; keep grow_width."
    else:
        payload["next"] = (
            f"Affordable: {affordable}. Train the smallest fitting GN graph "
            "against the quiet +0.25 / motion 0.0 gate."
        )
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "cells"}, indent=2))


if __name__ == "__main__":
    main()
