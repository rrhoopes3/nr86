"""One-shot: inspect → ingest → selfteach → eval a ReShade dump."""

from __future__ import annotations

import json
from pathlib import Path

from nr86.eval import evaluate
from nr86.ingest import ingest
from nr86.inspect_capture import inspect_capture
from nr86.selfteach import selfteach_dataset


def from_dump(
    src: Path,
    raw: Path,
    taught: Path,
    ckpt: Path,
    size: str = "1280x720",
    every_n: int = 2,
    dirty_tiles: bool = True,
    use_trt: bool = False,
    eval_offset: int = 0,
) -> dict:
    src = Path(src)
    report = inspect_capture(src)
    if not report.get("ok"):
        raise RuntimeError("inspect failed: " + "; ".join(report.get("issues") or ["unknown"]))
    n = ingest(src, raw, placeholder=False)
    selfteach_dataset(raw, taught, size)
    ev = evaluate(
        ckpt,
        taught,
        max_frames=min(32, n),
        every_n=every_n,
        dirty_tiles=dirty_tiles,
        use_trt=use_trt,
        offset=eval_offset,
    )
    out = {
        "inspect": {k: report[k] for k in ("frames", "ok", "color_formats", "depth_ok", "prev_color_frames") if k in report},
        "ingested": n,
        "taught": str(taught),
        "size": size,
        "eval": ev,
    }
    print(json.dumps({"from_dump": {"ingested": n, "taught": str(taught), "gate": ev.get("gate")}}, indent=2))
    return out
