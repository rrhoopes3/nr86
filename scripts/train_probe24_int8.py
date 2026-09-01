"""Train probe24_int8 (no GN) on quiet tiles until the v0.1 quiet bar holds.

Timing-legal cell from the storm-identity width map. Does not train
probe24 FP16, does not chase combat storms, does not ingest hangar stills.

Train: Sarif lobby room2 + hold-out first 105.
Quiet score: min(room2 skip, Detroit plaza full). Plaza stays unseen.
Motion / overlay rows are recorded, not optimized.
Stop on quiet pass + confirm, plateau, plaza overfit, or max steps.
If quiet passes, build a QDQ engine and measure TRT quality + skip ms.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from nr86.bench import bench_sequence
from nr86.config import BUDGET_SKIP_DIRTY_MEAN_MS, BUDGET_STUDENT_P95_MS
from nr86.eval import QUIET_DB, evaluate
from nr86.envelope import load_envelope
from nr86.models.student import load_student
from nr86.train import train

ROOT = Path(__file__).resolve().parents[1]
ROOM2 = ROOT / "datasets" / "q540-dxhr-room2"
HOLDOUT = ROOT / "datasets" / "q540-dxhr-holdout"
CITY = ROOT / "datasets" / "q540-dxhr-city"
COMBAT = ROOT / "datasets" / "q540-dxhr-combat"
OUT = ROOT / "runs" / "dxhr-probe24-int8"
RESULT = ROOT / "results" / "dxhr-q540-probe24-int8.json"
STATE = ROOT / "results" / "overnight-state.json"
ENVELOPE = ROOT / "results" / "color_envelope.json"

PRESET = "probe24_int8"
CHUNK = 200
MAX_STEPS = 3200
MIN_GAIN = 0.02
STALE_LIMIT = 2
OVERFIT_DROP = 0.25
HOLDOUT_TRAIN_FRAMES = 105
HOLDOUT_EVAL_OFFSET = 105
SENTINEL = "AGENT_LOOP_TICK_probe24int8"


def _brief(ev: dict) -> dict:
    return {
        "identity_psnr": ev["identity_psnr"],
        "student_psnr": ev["student_psnr"],
        "delta_psnr": ev["delta_psnr"],
        "beats_identity": ev["beats_identity"],
        "gate": ev["gate"],
        "regime": ev.get("regime"),
        "gate_db": ev.get("gate_db"),
        "paths": ev.get("paths"),
        "mask_fill_mean": ev.get("mask_fill_mean"),
        "executed_frac": ev.get("executed_frac"),
    }


def _tick(payload: dict) -> None:
    print(f"{SENTINEL} {json.dumps(payload, separators=(',', ':'))}", flush=True)


def _eval_quiet(ckpt: Path, env: dict | None) -> dict:
    room2 = evaluate(
        ckpt,
        ROOM2,
        max_frames=32,
        every_n=2,
        dirty_tiles=True,
        envelope=env,
    )
    plaza = evaluate(ckpt, CITY, max_frames=32, offset=0, every_n=1, envelope=env)
    hold = evaluate(
        ckpt,
        HOLDOUT,
        max_frames=32,
        offset=HOLDOUT_EVAL_OFFSET,
        every_n=1,
        envelope=env,
    )
    room2_d = float(room2["delta_psnr"])
    plaza_d = float(plaza["delta_psnr"])
    return {
        "room2_skip": _brief(room2),
        "plaza_full": _brief(plaza),
        "holdout_last32": _brief(hold),
        "quiet_score": round(min(room2_d, plaza_d), 3),
        "quiet_pass": room2_d >= QUIET_DB and plaza_d >= QUIET_DB,
    }


def _write(result: Path, payload: dict) -> None:
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _update_state(**kwargs) -> None:
    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    findings = kwargs.pop("findings", None)
    state.update(kwargs)
    if findings:
        merged = dict(state.get("findings") or {})
        merged.update(findings)
        state["findings"] = merged
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _phase2(ckpt: Path, env: dict | None) -> dict:
    """QDQ TRT on the quiet-passing weights. Timing + quality, not a train."""
    print("=== phase2 INT8 TRT on quiet-best ===", flush=True)
    plaza = evaluate(
        ckpt, CITY, max_frames=32, offset=0, every_n=1, envelope=env, use_trt=True, int8=True
    )
    room2 = evaluate(
        ckpt,
        ROOM2,
        max_frames=32,
        every_n=2,
        dirty_tiles=True,
        envelope=env,
        use_trt=True,
        int8=True,
    )
    skip = bench_sequence(
        ckpt,
        ROOM2,
        warmup=8,
        iters=8,
        every_n=2,
        dirty_tiles=True,
        max_frames=32,
        use_trt=True,
        int8=True,
        storm=True,
        envelope=env,
    )
    combat = bench_sequence(
        ckpt,
        COMBAT,
        warmup=8,
        iters=8,
        every_n=2,
        dirty_tiles=True,
        max_frames=32,
        use_trt=True,
        int8=True,
        storm=True,
        envelope=env,
    )
    room2_d = float(room2["delta_psnr"])
    plaza_d = float(plaza["delta_psnr"])
    mean = float(skip["mean_ms"])
    p95 = float((skip.get("student_path_ms") or {}).get("p95_ms") or skip["p95_ms"])
    cmean = float(combat["mean_ms"])
    cp95 = float((combat.get("student_path_ms") or {}).get("p95_ms") or 0.0)
    bind_p95 = max(p95, cp95) if cp95 else p95
    return {
        "ckpt": str(ckpt),
        "plaza_full": _brief(plaza),
        "room2_skip": _brief(room2),
        "quiet_pass": room2_d >= QUIET_DB and plaza_d >= QUIET_DB,
        "lobby_skip": {
            "mean_ms": skip.get("mean_ms"),
            "student_path_ms": skip.get("student_path_ms"),
            "path_ms": skip.get("path_ms"),
            "budget": skip.get("budget"),
        },
        "combat_skip": {
            "mean_ms": combat.get("mean_ms"),
            "student_path_ms": combat.get("student_path_ms"),
            "path_ms": combat.get("path_ms"),
            "budget": combat.get("budget"),
        },
        "fits_mean_8_33": mean <= BUDGET_SKIP_DIRTY_MEAN_MS,
        "fits_student_p95_16_67": bind_p95 <= BUDGET_STUDENT_P95_MS,
        "binding_mean_ms": max(mean, cmean),
        "do_not_quality_claim": []
        if (room2_d >= QUIET_DB and plaza_d >= QUIET_DB)
        else ["int8_trt_missed_quiet"],
    }


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--result", type=Path, default=RESULT)
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--no-phase2", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse()
    out: Path = args.out
    result: Path = args.result
    out.mkdir(parents=True, exist_ok=True)
    env = load_envelope(ENVELOPE) if ENVELOPE.exists() else None

    history: list[dict] = []
    best_score = float("-inf")
    stale = 0
    total = 0
    resume: Path | None = None
    best_ckpt = out / "student_best.pt"
    stop_reason = "max_steps"
    passed_once = False

    _update_state(
        purpose="probe24_int8 quiet train under storm-identity timing",
        next="train_probe24_int8_chunk",
        loop="running",
        loop_name="probe24int8",
        notes=(
            "Training no-GN base-24 from scratch on room2 + holdout105. "
            "Plaza is unseen. Stop when quiet +0.25 holds or the run is stale."
        ),
    )

    while total < args.max_steps:
        seed = total // args.chunk
        print(f"\n=== chunk {seed + 1}  resume={resume}  seed={seed} ===", flush=True)
        train(
            ROOM2,
            out,
            preset=PRESET,
            steps=args.chunk,
            resume=resume,
            skip_eval=True,
            seed=seed,
            extra=HOLDOUT,
            extra_frames=HOLDOUT_TRAIN_FRAMES,
            hud_mask="dxhr",
        )
        total += args.chunk
        ckpt = out / "student.pt"
        loaded = load_student(ckpt)
        if loaded.spec.base != 24 or loaded.spec.norm != "none":
            raise RuntimeError(f"refusing unexpected spec {loaded.spec}")
        resume = ckpt
        row = _eval_quiet(ckpt, env)
        row["steps"] = total
        history.append(row)
        score = float(row["quiet_score"])
        payload = {
            "preset": PRESET,
            "train": "room2 + holdout_first105",
            "hud_mask": "dxhr",
            "not_train": ["plaza", "combat", "hangar", "yard"],
            "quiet_bar": QUIET_DB,
            "out": str(out),
            "history": history,
            "best_quiet_score": max(best_score, score)
            if best_score != float("-inf")
            else score,
            "status": "running",
            "do_not": ["ampere400", "train_probe24_fp16", "chase_hangar_stills"],
        }
        _write(result, payload)
        print(
            f"steps={total}  quiet={score:+.3f}  "
            f"room2_skip={row['room2_skip']['delta_psnr']:+.3f}  "
            f"plaza={row['plaza_full']['delta_psnr']:+.3f}  "
            f"holdout={row['holdout_last32']['delta_psnr']:+.3f}  "
            f"pass={row['quiet_pass']}",
            flush=True,
        )
        _tick(
            {
                "steps": total,
                "quiet_score": score,
                "room2_skip": row["room2_skip"]["delta_psnr"],
                "plaza": row["plaza_full"]["delta_psnr"],
                "quiet_pass": row["quiet_pass"],
                "best": max(best_score, score) if best_score != float("-inf") else score,
            }
        )
        _update_state(
            next="train_probe24_int8_chunk",
            findings={
                "probe24_int8_steps": total,
                "probe24_int8_quiet": score,
                "probe24_int8_room2_skip": row["room2_skip"]["delta_psnr"],
                "probe24_int8_plaza": row["plaza_full"]["delta_psnr"],
                "probe24_int8_quiet_pass": row["quiet_pass"],
            },
        )

        if score > best_score:
            best_score = score
            shutil.copy2(ckpt, best_ckpt)
            print(f"new best quiet {best_score:+.3f} dB -> {best_ckpt}", flush=True)

        if best_score != float("-inf") and score < best_score - OVERFIT_DROP:
            stop_reason = (
                f"overfit: quiet score {score:+.3f} dropped from best {best_score:+.3f}"
            )
            print(stop_reason, flush=True)
            break

        prev = (
            float(history[-2]["quiet_score"]) if len(history) >= 2 else float("-inf")
        )
        gain = score - prev
        if row["quiet_pass"]:
            if passed_once:
                stop_reason = (
                    f"quiet_gates_pass: room2 {row['room2_skip']['delta_psnr']:+.3f} "
                    f"plaza {row['plaza_full']['delta_psnr']:+.3f} confirmed"
                )
                print(stop_reason, flush=True)
                break
            passed_once = True
            stale = 0
            print("quiet gates passed; one confirm chunk", flush=True)
            continue
        if gain >= MIN_GAIN:
            stale = 0
            print(f"chunk gain {gain:+.3f} dB (still useful)", flush=True)
            continue
        stale += 1
        print(f"chunk gain {gain:+.3f} dB (stale {stale}/{STALE_LIMIT})", flush=True)
        if stale >= STALE_LIMIT:
            stop_reason = (
                f"plateau: last {STALE_LIMIT} chunks gained under {MIN_GAIN:.2f} dB "
                f"(quiet {score:+.3f}, best {best_score:+.3f})"
            )
            break
    else:
        stop_reason = f"max_steps ({args.max_steps})"

    quiet_pass = bool(history and any(r["quiet_pass"] for r in history))
    best_row = None
    if history:
        best_row = max(history, key=lambda r: float(r["quiet_score"]))
    phase2 = None
    if (not args.no_phase2) and quiet_pass and best_ckpt.exists():
        try:
            phase2 = _phase2(best_ckpt, env)
        except Exception as exc:
            phase2 = {"error": str(exc)}

    final = {
        "preset": PRESET,
        "train": "room2 + holdout_first105",
        "hud_mask": "dxhr",
        "not_train": ["plaza", "combat", "hangar", "yard"],
        "quiet_bar": QUIET_DB,
        "out": str(out),
        "total_steps": total,
        "best_quiet_score": None if best_score == float("-inf") else round(best_score, 3),
        "best_ckpt": str(best_ckpt) if best_ckpt.exists() else None,
        "best_row": best_row,
        "quiet_pass": quiet_pass,
        "stop_reason": stop_reason,
        "history": history,
        "phase2_int8_trt": phase2,
        "status": "done",
        "do_not": ["ampere400", "train_probe24_fp16", "chase_hangar_stills"],
        "do_not_quality_claim": ["junk_qdq_width_probe"],
    }
    if not quiet_pass:
        final["lesson"] = (
            "no-GN base-24 did not hold quiet +0.25 on plaza and room2 skip. "
            "Do not ship INT8. Stay on smoke-16 GN for quality."
        )
        final["next"] = "stay_smoke16"
    elif phase2 and phase2.get("quiet_pass") and phase2.get("fits_mean_8_33"):
        final["lesson"] = (
            "probe24 INT8 holds quiet +0.25 in PyTorch and in the QDQ engine, "
            "and still fits the skip+dirty mean. First wider graph that is "
            "both timing-legal and quality-legal under storm-identity."
        )
        final["next"] = "consider_v02_probe24_int8"
    elif phase2 and not phase2.get("quiet_pass"):
        final["lesson"] = (
            "PyTorch no-GN cleared quiet +0.25; QDQ TRT dropped it. "
            "Do not quality-claim the INT8 engine."
        )
        final["next"] = "stay_smoke16_or_debug_qdq"
    else:
        final["lesson"] = (
            "PyTorch no-GN cleared quiet +0.25. TRT phase missing or failed; "
            "do not ship until the engine is measured."
        )
        final["next"] = "phase2_int8_trt"
    _write(result, final)
    _update_state(
        loop="done" if final["status"] == "done" else "running",
        next=final.get("next"),
        notes=final.get("lesson") or stop_reason,
        findings={
            "probe24_int8_steps": total,
            "probe24_int8_quiet": final["best_quiet_score"],
            "probe24_int8_quiet_pass": quiet_pass,
            "probe24_int8_stop": stop_reason,
        },
    )
    print(json.dumps({k: final[k] for k in final if k != "history"}, indent=2), flush=True)
    print(f"wrote {result}", flush=True)
    _tick({"done": True, "stop_reason": stop_reason, "quiet_pass": quiet_pass})


if __name__ == "__main__":
    main()
