"""One overnight experiment. Writes JSON under runs/overnight/ and updates results/overnight-state.json."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE = ROOT / "results" / "overnight-state.json"
RUNS = ROOT / "runs" / "overnight"
SYNTH = ROOT / "datasets" / "synth"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"tick": 0, "done": [], "next": "smoke200", "blocked_on": []}


def _save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def smoke200() -> dict:
    from nr86.bench import bench_ckpt
    from nr86.eval import evaluate
    from nr86.synth import write_synth
    from nr86.train import train

    write_synth(SYNTH, frames=24, size=512, seed=0, hq_scale=2)
    out = RUNS / "smoke200"
    train(SYNTH, out, preset="smoke", steps=200, batch=4, lr=2e-4)
    ckpt = out / "student.pt"
    reports = {
        "full": evaluate(ckpt, SYNTH, max_frames=24),
        "skip_dirty": evaluate(ckpt, SYNTH, max_frames=24, every_n=2, dirty_tiles=True),
        "ablate_rgb": evaluate(ckpt, SYNTH, max_frames=24, ablate="rgb"),
        "ablate_depth": evaluate(ckpt, SYNTH, max_frames=24, ablate="depth"),
        "ablate_mvec": evaluate(ckpt, SYNTH, max_frames=24, ablate="mvec"),
    }
    seq = bench_ckpt(
        ckpt,
        "1280x720",
        warmup=3,
        iters=8,
        every_n=2,
        data=SYNTH,
        dirty_tiles=True,
        max_frames=24,
        try_trt=True,
    )
    eager = bench_ckpt(
        ckpt,
        "1280x720",
        warmup=5,
        iters=20,
        try_trt=True,
    )
    full = reports["full"]
    return {
        "ckpt": str(ckpt),
        "eval": reports,
        "bench_sequence": seq,
        "bench_eager_720p": eager,
        "beats_identity": bool(full.get("beats_identity")),
        "delta_psnr": full.get("delta_psnr"),
    }


def main() -> int:
    from nr86.hw import doctor

    RUNS.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    state["tick"] = int(state.get("tick") or 0) + 1
    state["last_started"] = _utc()
    nxt = state.get("next") or "smoke200"
    payload: dict = {"when": _utc(), "tick": state["tick"], "action": nxt}
    try:
        doc = doctor()
        payload["doctor"] = {
            "gpu": (doc.get("gpu") or {}).get("name"),
            "tensorrt_rtx": (doc.get("tools") or {}).get("tensorrt_rtx"),
            "fp8": (doc.get("features") or {}).get("fp8"),
        }
        if nxt == "smoke200":
            payload["result"] = smoke200()
            state["done"] = list(dict.fromkeys(list(state.get("done") or []) + ["smoke200"]))
            # Do not grow width because smoke already passed identity.
            state["next"] = (
                "real_capture_or_trt_student"
                if payload["result"].get("beats_identity")
                else "diagnose_smoke"
            )
        else:
            payload["result"] = {
                "skipped": True,
                "reason": f"script implements smoke200; agent handles {nxt}",
            }
        payload["ok"] = True
    except Exception as exc:
        payload["ok"] = False
        payload["error"] = str(exc)
        state["last_error"] = str(exc)
    state["last_finished"] = _utc()
    _save_state(state)
    tick_path = RUNS / f"tick-{state['tick']:02d}.json"
    tick_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(tick_path), "next": state.get("next"), "ok": payload.get("ok")}, indent=2))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
