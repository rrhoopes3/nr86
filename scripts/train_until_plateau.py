"""Train smoke on a dump until held-out ΔPSNR stops improving.

Keeps the smoke preset. Does not grow width, switch to ampere400, or INT8.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from nr86.eval import evaluate
from nr86.train import train

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets" / "q720-dxhr-holdout"
DEFAULT_OUT = ROOT / "runs" / "dxhr-depth-smoke"
DEFAULT_RESULT = ROOT / "results" / "dxhr-depth-plateau.json"

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


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    p.add_argument(
        "--init",
        type=Path,
        default=None,
        help="Starting checkpoint when --out has no student.pt (does not resume old history).",
    )
    p.add_argument("--train-frames", type=int, default=TRAIN_FRAMES)
    p.add_argument("--eval-offset", type=int, default=EVAL_OFFSET)
    p.add_argument("--eval-frames", type=int, default=EVAL_FRAMES)
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--preset", default="smoke")
    return p.parse_args()


def main() -> None:
    args = _parse()
    data = args.data
    out = args.out
    result = args.result
    train_frames = args.train_frames
    eval_offset = args.eval_offset
    eval_frames = args.eval_frames
    chunk = args.chunk
    max_steps = args.max_steps
    preset = args.preset

    if not data.exists():
        raise SystemExit(f"missing taught dump {data}")
    out.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_delta = float("-inf")
    stale = 0
    total = 0
    resume: Path | None = None
    best_ckpt = out / "student_best.pt"
    stop_reason = "max_steps"
    ckpt = out / "student.pt"
    if result.exists():
        try:
            prev = json.loads(result.read_text(encoding="utf-8"))
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
        total = 0
        print(f"resuming weights {ckpt} as start (history empty)", flush=True)
    if resume is None and args.init is not None and Path(args.init).exists():
        resume = Path(args.init)
        print(f"init from {resume}", flush=True)

    while total < max_steps:
        seed = total // chunk
        print(f"\n=== chunk {seed + 1}  resume={resume}  seed={seed} ===", flush=True)
        train(
            data,
            out,
            preset=preset,
            steps=chunk,
            resume=resume,
            skip_eval=True,
            seed=seed,
            data_frames=train_frames,
        )
        total += chunk
        ckpt = out / "student.pt"
        resume = ckpt

        hold_full = evaluate(ckpt, data, max_frames=eval_frames, offset=eval_offset, every_n=1)
        hold_skip = evaluate(
            ckpt,
            data,
            max_frames=eval_frames,
            offset=eval_offset,
            every_n=2,
            dirty_tiles=True,
        )
        hold_nodepth = evaluate(
            ckpt,
            data,
            max_frames=eval_frames,
            offset=eval_offset,
            every_n=1,
            ablate="depth",
        )
        train_full = evaluate(ckpt, data, max_frames=32, offset=0, every_n=1)

        delta = float(hold_full["delta_psnr"])
        row = {
            "steps": total,
            "holdout_full": _brief(hold_full),
            "holdout_skip_dirty": _brief(hold_skip),
            "holdout_ablate_depth": _brief(hold_nodepth),
            "train_full": _brief(train_full),
        }
        history.append(row)
        result.write_text(
            json.dumps(
                {
                    "preset": preset,
                    "data": str(data),
                    "out": str(out),
                    "train_frames": train_frames,
                    "eval_offset": eval_offset,
                    "eval_frames": eval_frames,
                    "chunk": chunk,
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
        stop_reason = f"max_steps ({max_steps})"

    if best_ckpt.exists() and (out / "student.pt").exists():
        # leave student.pt as last weights; student_best.pt is the peak hold-out
        pass

    payload = {
        "preset": preset,
        "data": str(data),
        "out": str(out),
        "train_frames": train_frames,
        "eval_offset": eval_offset,
        "eval_frames": eval_frames,
        "chunk": chunk,
        "min_gain_db": MIN_GAIN,
        "total_steps": total,
        "best_delta_psnr": None if best_delta == float("-inf") else round(best_delta, 3),
        "best_ckpt": str(best_ckpt) if best_ckpt.exists() else None,
        "stop_reason": stop_reason,
        "history": history,
        "status": "done",
    }
    result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "history"}, indent=2))
    print(f"wrote {result}", flush=True)


if __name__ == "__main__":
    main()
