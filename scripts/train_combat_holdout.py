"""Retrain smoke on combat + quiet-lobby tiles with HUD-masked loss.

Hold-out is OTHER bursts (Sarif lobby room2, original hold-out last 32).
Last-32 of the same 8s combat clip is same-scene leakage — recorded only.
Leftover capture IDs 000237+ are old leftover IDs, not a second combat burst.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from nr86.dataset import FrameDataset
from nr86.eval import evaluate
from nr86.train import train

ROOT = Path(__file__).resolve().parents[1]
COMBAT = ROOT / "datasets" / "q540-dxhr-combat"
HOLDOUT = ROOT / "datasets" / "q540-dxhr-holdout"
ROOM2 = ROOT / "datasets" / "q540-dxhr-room2"
INIT = ROOT / "runs" / "dxhr-depth-smoke-q540" / "student_best.pt"
OUT = ROOT / "runs" / "dxhr-combat-hud"
RESULT = ROOT / "results" / "dxhr-q540-combat-retrain.json"

CHUNK = 200
MAX_STEPS = 1600
MIN_GAIN = 0.02
STALE_LIMIT = 2
VETO_DB = 0.25
HOLDOUT_TRAIN_FRAMES = 105
HOLDOUT_EVAL_OFFSET = 105


def _brief(ev: dict) -> dict:
    return {
        "identity_psnr": ev["identity_psnr"],
        "student_psnr": ev["student_psnr"],
        "delta_psnr": ev["delta_psnr"],
        "beats_identity": ev["beats_identity"],
        "gate": ev["gate"],
        "paths": ev.get("paths"),
        "mask_fill_mean": ev.get("mask_fill_mean"),
    }


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combat", type=Path, default=COMBAT)
    p.add_argument("--holdout", type=Path, default=HOLDOUT)
    p.add_argument("--room2", type=Path, default=ROOM2)
    p.add_argument("--init", type=Path, default=INIT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--result", type=Path, default=RESULT)
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--preset", default="smoke")
    return p.parse_args()


def _eval_suite(ckpt: Path, args: argparse.Namespace) -> dict:
    n_combat = len(FrameDataset(args.combat, require_teacher=True))
    combat_first = evaluate(ckpt, args.combat, max_frames=32, offset=0, every_n=1)
    combat_last = evaluate(
        ckpt,
        args.combat,
        max_frames=32,
        offset=max(0, n_combat - 32),
        every_n=1,
    )
    room2_full = evaluate(ckpt, args.room2, max_frames=32, offset=0, every_n=1)
    room2_skip = evaluate(
        ckpt, args.room2, max_frames=32, offset=0, every_n=2, dirty_tiles=True
    )
    hold_full = evaluate(
        ckpt, args.holdout, max_frames=32, offset=HOLDOUT_EVAL_OFFSET, every_n=1
    )
    hold_skip = evaluate(
        ckpt,
        args.holdout,
        max_frames=32,
        offset=HOLDOUT_EVAL_OFFSET,
        every_n=2,
        dirty_tiles=True,
    )
    return {
        "combat_first32_full": _brief(combat_first),
        "combat_last32_full_leaky": _brief(combat_last),
        "room2_full": _brief(room2_full),
        "room2_skip_dirty": _brief(room2_skip),
        "holdout_last32_full": _brief(hold_full),
        "holdout_last32_skip_dirty": _brief(hold_skip),
    }


def _veto(row: dict) -> str | None:
    room2 = float(row["room2_skip_dirty"]["delta_psnr"])
    hold = float(row["holdout_last32_full"]["delta_psnr"])
    fails = []
    if room2 < VETO_DB:
        fails.append(f"room2 skip {room2:+.3f} < {VETO_DB}")
    if hold < VETO_DB:
        fails.append(f"holdout last32 {hold:+.3f} < {VETO_DB}")
    return None if not fails else "; ".join(fails)


def main() -> None:
    args = _parse()
    out = args.out
    result = args.result
    out.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_combat = float("-inf")
    stale = 0
    veto_streak = 0
    total = 0
    resume = args.init if args.init.exists() else None
    best_ckpt = out / "student_best.pt"
    stop_reason = "max_steps"
    ckpt = out / "student.pt"

    print(f"baseline eval resume={resume}", flush=True)
    if resume is not None:
        base = _eval_suite(resume, args)
        base["steps"] = 0
        base["veto"] = _veto(base)
        history.append(base)
        result.write_text(
            json.dumps(
                {
                    "preset": args.preset,
                    "train": "combat_all + holdout_first105",
                    "hud_mask": "dxhr",
                    "holdout": "room2 + holdout last32 (other bursts)",
                    "not_holdout": [
                        "combat last32 (same 8s burst)",
                        "leftover capture IDs 000237+",
                    ],
                    "history": history,
                    "status": "running",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"steps=0  combat={base['combat_first32_full']['delta_psnr']:+.3f}  "
            f"room2_skip={base['room2_skip_dirty']['delta_psnr']:+.3f}  "
            f"holdout={base['holdout_last32_full']['delta_psnr']:+.3f}  "
            f"veto={base['veto']}",
            flush=True,
        )

    while total < args.max_steps:
        seed = total // args.chunk
        print(f"\n=== chunk {seed + 1}  resume={resume}  seed={seed} ===", flush=True)
        train(
            args.combat,
            out,
            preset=args.preset,
            steps=args.chunk,
            resume=resume,
            skip_eval=True,
            seed=seed,
            extra=args.holdout,
            extra_frames=HOLDOUT_TRAIN_FRAMES,
            hud_mask="dxhr",
        )
        total += args.chunk
        ckpt = out / "student.pt"
        resume = ckpt
        row = _eval_suite(ckpt, args)
        row["steps"] = total
        row["veto"] = _veto(row)
        history.append(row)
        combat_d = float(row["combat_first32_full"]["delta_psnr"])
        result.write_text(
            json.dumps(
                {
                    "preset": args.preset,
                    "train": "combat_all + holdout_first105",
                    "hud_mask": "dxhr",
                    "out": str(out),
                    "holdout": "room2 + holdout last32 (other bursts)",
                    "not_holdout": [
                        "combat last32 (same 8s burst)",
                        "leftover capture IDs 000237+",
                    ],
                    "history": history,
                    "best_combat_first32_delta": (
                        max(best_combat, combat_d)
                        if best_combat != float("-inf")
                        else combat_d
                    ),
                    "status": "running",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"steps={total}  combat={combat_d:+.3f}  "
            f"room2_skip={row['room2_skip_dirty']['delta_psnr']:+.3f}  "
            f"holdout={row['holdout_last32_full']['delta_psnr']:+.3f}  "
            f"veto={row['veto']}",
            flush=True,
        )

        if row["veto"] is None:
            veto_streak = 0
            if combat_d > best_combat:
                best_combat = combat_d
                shutil.copy2(ckpt, best_ckpt)
                print(f"new best combat {best_combat:+.3f} dB -> {best_ckpt}", flush=True)
        else:
            veto_streak += 1
            print(f"veto ({veto_streak}): {row['veto']}", flush=True)
            if veto_streak >= 2:
                stop_reason = f"burst hold-out vetoed twice: {row['veto']}"
                break

        prev = (
            float(history[-2]["combat_first32_full"]["delta_psnr"])
            if len(history) >= 2
            else float("-inf")
        )
        gain = combat_d - prev
        if gain >= MIN_GAIN:
            stale = 0
            print(f"chunk gain {gain:+.3f} dB (still effective)", flush=True)
            continue
        stale += 1
        print(f"chunk gain {gain:+.3f} dB (stale {stale}/{STALE_LIMIT})", flush=True)
        if stale >= STALE_LIMIT:
            stop_reason = (
                f"plateau: last {STALE_LIMIT} chunks gained under {MIN_GAIN:.2f} dB "
                f"(combat {combat_d:+.3f}, best {best_combat:+.3f})"
            )
            break
    else:
        stop_reason = f"max_steps ({args.max_steps})"

    payload = {
        "preset": args.preset,
        "train": "combat_all + holdout_first105",
        "hud_mask": "dxhr",
        "data": str(args.combat),
        "extra": str(args.holdout),
        "out": str(out),
        "holdout": "room2 + holdout last32 (other bursts)",
        "not_holdout": [
            "combat last32 (same 8s burst)",
            "leftover capture IDs 000237+",
        ],
        "total_steps": total,
        "best_combat_first32_delta": None
        if best_combat == float("-inf")
        else round(best_combat, 3),
        "best_ckpt": str(best_ckpt) if best_ckpt.exists() else None,
        "stop_reason": stop_reason,
        "history": history,
        "status": "done",
        "do_not": ["ampere400", "grow_width", "shrink_below_960x540"],
    }
    result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "history"}, indent=2))
    print(f"wrote {result}", flush=True)


if __name__ == "__main__":
    main()
