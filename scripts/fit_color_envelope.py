"""Fit the overlay pass-through envelope from taught gameplay dumps.

Excludes Smart Vision / overlay tails. Writes results/color_envelope.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nr86.dataset import FrameDataset, load_frame
from nr86.envelope import color_stats, fit_envelope, save_envelope

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = [
    ROOT / "datasets" / "q540-dxhr-room2",
    ROOT / "datasets" / "q540-dxhr-holdout",
    ROOT / "datasets" / "q540-dxhr-combat",
    ROOT / "datasets" / "q540-dxhr-combat3",
    ROOT / "datasets" / "q540-dxhr-hangar",
    ROOT / "datasets" / "q540-dxhr-yard",
    ROOT / "datasets" / "q540-dxhr-city",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "color_envelope.json")
    p.add_argument("--pad", type=float, default=0.03)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--data", type=Path, action="append", default=None)
    args = p.parse_args()
    dumps = [d for d in (args.data or DEFAULTS) if d.exists()]
    if not dumps:
        raise SystemExit("no taught dumps found")
    rows = []
    used = []
    for dump in dumps:
        ds = FrameDataset(dump, require_teacher=False)
        n = 0
        for i, rec in enumerate(ds.rows):
            if i % max(1, args.stride) != 0:
                continue
            # City look-up / yard motion stay in-distribution; combat2 SV does not.
            if dump.name == "q540-dxhr-city" and i > 200:
                continue
            rows.append(color_stats(load_frame(ds.root, rec).color))
            n += 1
        used.append({"data": str(dump), "sampled": n})
    env = fit_envelope(rows, pad=args.pad)
    env["sources"] = used
    env["note"] = (
        "Gameplay color envelope. Overlay / Smart Vision / post effects "
        "that leave these bands pass through as identity."
    )
    save_envelope(env, args.out)
    print(f"wrote {args.out} from {len(rows)} frames across {len(used)} dumps")


if __name__ == "__main__":
    main()
