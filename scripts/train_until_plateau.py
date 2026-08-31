"""Train smoke on a dump until held-out ΔPSNR stops improving.

Keeps the smoke preset. Does not grow width, switch to ampere400, or INT8.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from nr86.eval import evaluate
from nr86.train import train

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "q720-dxhr-holdout"
OUT = ROOT / "runs" / "dxhr-depth-smoke"
RESULT = ROOT / "results" / "dxhr-depth-plateau.json"

CHUNK = 200
MAX_STEPS = 8000
MIN_GAIN = 0.02
STALE_LIMIT = 2
TRAIN_FRAMES = 105
EVAL_OFFSET = 105
EVAL_FRAMES = 32


def _brief(ev: dict) -> dict:
    return {
        "identity_psnr": ev["identity_psnr"],
        "student_psnr": ev["student_psnr"],
        "delta_psnr": ev["delta_psnr"],
        "beats_identity": ev["beats_identity"],
        "gate": ev["gate"],
        "paths": ev.get("paths"),
    }


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing taught dump {DATA}")
    OUT.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_delta = float("-inf")
    stale = 0
    total = 0
    resume: Path | None = None
    best_ckpt = OUT / "student_best.pt"
    stop_reason = "max_steps"
    ckpt = OUT / "student.pt"
    if RESULT.exists():
        try:
            prev = json.loads(RESULT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
        if prev.get("history"):
            history = list(prev["history"])
            total = int(history[-1]["steps"])
            best_delta = max(
                float(row["holdout_full"]["delta_psnr"]) for row in history
            )
            if ckpt.exists():
                resume = ckpt
                last_delta = float(history[-1]["holdout_full"]["delta_psnr"])
                if last_delta >= best_delta - 1e-9:
                    shutil.copy2(ckpt, best_ckpt)
            print(f"resuming loop at {total} steps  best={best_delta:+.3f}", flush=True)
    if resume is None and ckpt.exists() and not history:
        resume = ckpt
        total = 200
        print(f"resuming weights {ckpt} as 200-step start", flush=True)

    while total < MAX_STEPS:
        seed = total // CHUNK
        print(f"\n=== chunk {seed + 1}  resume={resume}  seed={seed} ===", flush=True)
        train(
            DATA,
            OUT,
            preset="smoke",
            steps=CHUNK,
            resume=resume,
            skip_eval=True,
            seed=seed,
            data_frames=TRAIN_FRAMES,
        )
        total += CHUNK
        ckpt = OUT / "student.pt"
        resume = ckpt

        hold_full = evaluate(ckpt, DATA, max_frames=EVAL_FRAMES, offset=EVAL_OFFSET, every_n=1)
        hold_skip = evaluate(
            ckpt,
            DATA,
            max_frames=EVAL_FRAMES,
            offset=EVAL_OFFSET,
            every_n=2,
            dirty_tiles=True,
        )
        hold_nodepth = evaluate(
            ckpt,
            DATA,
            max_frames=EVAL_FRAMES,
            offset=EVAL_OFFSET,
            every_n=1,
            ablate="depth",
        )
        train_full = evaluate(ckpt, DATA, max_frames=32, offset=0, every_n=1)

        delta = float(hold_full["delta_psnr"])
        row = {
            "steps": total,
            "holdout_full": _brief(hold_full),
            "holdout_skip_dirty": _brief(hold_skip),
            "holdout_ablate_depth": _brief(hold_nodepth),
            "train_full": _brief(train_full),
        }
        history.append(row)
        RESULT.write_text(
            json.dumps(
                {
                    "preset": "smoke",
                    "train_frames": TRAIN_FRAMES,
                    "eval_offset": EVAL_OFFSET,
                    "eval_frames": EVAL_FRAMES,
                    "chunk": CHUNK,
                    "min_gain_db": MIN_GAIN,
                    "history": history,
                    "best_delta_psnr": max(best_delta, delta) if best_delta != float("-inf") else delta,
                    "status": "running",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"steps={total}  holdout dPSNR={delta:+.3f}  "
            f"skip dPSNR={hold_skip['delta_psnr']:+.3f}  "
            f"no-depth dPSNR={hold_nodepth['delta_psnr']:+.3f}  "
            f"train dPSNR={train_full['delta_psnr']:+.3f}  "
            f"best={best_delta if best_delta != float('-inf') else 'n/a'}",
            flush=True,
        )

        prev_delta = (
            float(history[-2]["holdout_full"]["delta_psnr"])
            if len(history) >= 2
            else float("-inf")
        )
        if delta > best_delta:
            best_delta = delta
            shutil.copy2(ckpt, best_ckpt)
            print(f"new best {best_delta:+.3f} dB -> {best_ckpt}", flush=True)

        if best_delta != float("-inf") and delta < best_delta - 0.25:
            stop_reason = (
                f"overfit: hold-out dropped to {delta:+.3f} from best {best_delta:+.3f}"
            )
            print(stop_reason, flush=True)
            break

        chunk_gain = delta - prev_delta
        if chunk_gain >= MIN_GAIN:
            stale = 0
            print(f"chunk gain {chunk_gain:+.3f} dB (still effective)", flush=True)
            continue

        stale += 1
        print(
            f"chunk gain {chunk_gain:+.3f} dB (stale {stale}/{STALE_LIMIT})",
            flush=True,
        )
        if stale >= STALE_LIMIT:
            stop_reason = (
                f"plateau: last {STALE_LIMIT} chunks gained under {MIN_GAIN:.2f} dB "
                f"(hold-out {delta:+.3f}, best {best_delta:+.3f})"
            )
            break
    else:
        stop_reason = f"max_steps ({MAX_STEPS})"

    if best_ckpt.exists() and (OUT / "student.pt").exists():
        # leave student.pt as last weights; student_best.pt is the peak hold-out
        pass

    payload = {
        "preset": "smoke",
        "train_frames": TRAIN_FRAMES,
        "eval_offset": EVAL_OFFSET,
        "eval_frames": EVAL_FRAMES,
        "chunk": CHUNK,
        "min_gain_db": MIN_GAIN,
        "total_steps": total,
        "best_delta_psnr": None if best_delta == float("-inf") else round(best_delta, 3),
        "best_ckpt": str(best_ckpt) if best_ckpt.exists() else None,
        "stop_reason": stop_reason,
        "history": history,
        "status": "done",
    }
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "history"}, indent=2))
    print(f"wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
