"""v0.1 gate table. Writes results/dxhr-q540-v01.json."""

from __future__ import annotations

import json
from pathlib import Path

from nr86.eval import evaluate
from nr86.envelope import load_envelope

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "runs" / "dxhr-city-mix" / "student_best.pt"
OUT = ROOT / "results" / "dxhr-q540-v01.json"

CHECKS = [
    ("room2_skip", "datasets/q540-dxhr-room2", dict(every_n=2, dirty_tiles=True)),
    ("plaza_full", "datasets/q540-dxhr-city", dict(every_n=1, offset=0)),
    ("holdout_last32", "datasets/q540-dxhr-holdout", dict(every_n=1, offset=105)),
    ("lookup_skip", "datasets/q540-dxhr-city", dict(every_n=2, dirty_tiles=True, offset=356)),
    ("lookup_full", "datasets/q540-dxhr-city", dict(every_n=1, offset=356)),
    ("yard_skip", "datasets/q540-dxhr-yard", dict(every_n=2, dirty_tiles=True)),
    ("yard_full", "datasets/q540-dxhr-yard", dict(every_n=1)),
    ("combat1_skip", "datasets/q540-dxhr-combat", dict(every_n=2, dirty_tiles=True)),
    ("combat3_skip", "datasets/q540-dxhr-combat3", dict(every_n=2, dirty_tiles=True)),
    ("combat3_full", "datasets/q540-dxhr-combat3", dict(every_n=1)),
    ("combat2_sv_last32", "datasets/q540-dxhr-combat2", dict(every_n=1, offset=104)),
]


def main() -> None:
    env = load_envelope(ROOT / "results" / "color_envelope.json")
    rows = {}
    for name, data, kw in CHECKS:
        print(f"EVAL {name}", flush=True)
        ev = evaluate(CKPT, ROOT / data, max_frames=32, envelope=env, **kw)
        rows[name] = {
            "delta_psnr": ev["delta_psnr"],
            "regime": ev["regime"],
            "gate_db": ev["gate_db"],
            "gate": ev["gate"],
            "paths": ev.get("paths"),
            "mask_fill_mean": ev.get("mask_fill_mean"),
            "executed_frac": ev.get("executed_frac"),
        }
        print(
            f"{name} {ev['delta_psnr']} {ev['regime']} pass={ev['beats_identity']} {ev.get('paths')}",
            flush=True,
        )
    fails = [k for k, v in rows.items() if v["gate"] != "pass"]
    OUT.write_text(
        json.dumps(
            {
                "version": "v0.1",
                "ckpt": str(CKPT),
                "policy": {
                    "quiet": "+0.25 measured",
                    "motion": "0.0 by policy (storm-identity)",
                    "overlay": "0.0 by policy (color envelope)",
                },
                "rows": rows,
                "fails": fails,
                "status": "measured",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("FAILS", fails, flush=True)


if __name__ == "__main__":
    main()
