"""Copy capture frames whose meta.json mtime is in [after, before)."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--after", required=True, help="ISO local time, inclusive")
    p.add_argument("--before", default=None, help="ISO local time, exclusive")
    args = p.parse_args()
    after = datetime.fromisoformat(args.after).timestamp()
    before = datetime.fromisoformat(args.before).timestamp() if args.before else None

    shutil.rmtree(args.dst, ignore_errors=True)
    args.dst.mkdir(parents=True)
    depth_mins: list[float] = []
    depth_maxs: list[float] = []
    n = 0
    skipped_old = 0
    for meta_p in sorted(args.src.glob("*/meta.json")):
        mt = meta_p.stat().st_mtime
        if mt < after or (before is not None and mt >= before):
            skipped_old += 1
            continue
        folder = meta_p.parent
        dest = args.dst / folder.name
        dest.mkdir()
        for name in ("meta.json", "color.bmp", "color_prev.bmp", "depth.f32"):
            src = folder / name
            if src.exists():
                shutil.copy2(src, dest / name)
        depth = dest / "depth.f32"
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if depth.exists():
            raw = np.fromfile(depth, dtype=np.float32)
            w = int(meta.get("depth_width") or meta.get("width") or 0)
            h = int(meta.get("depth_height") or meta.get("height") or 0)
            if w and h and raw.size == w * h:
                depth_mins.append(float(raw.min()))
                depth_maxs.append(float(raw.max()))
        n += 1
    print(
        json.dumps(
            {
                "dst": str(args.dst),
                "staged": n,
                "skipped_old": skipped_old,
                "depth_frames": len(depth_mins),
                "depth_min": None if not depth_mins else round(min(depth_mins), 4),
                "depth_max": None if not depth_maxs else round(max(depth_maxs), 4),
                "depth_mean_min": None if not depth_mins else round(float(np.mean(depth_mins)), 4),
                "far_plane_frames": sum(1 for lo, hi in zip(depth_mins, depth_maxs) if lo >= 0.999 and hi >= 0.999),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
