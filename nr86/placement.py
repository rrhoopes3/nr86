from __future__ import annotations

from dataclasses import dataclass

from nr86.config import Placement


def internal_size(p: Placement) -> tuple[int, int]:
    w = max(1, int(round(p.output_w * p.scaling_ratio)))
    h = max(1, int(round(p.output_h * p.scaling_ratio)))
    return w, h


def pixel_ops(p: Placement) -> float:
    """Relative pixel-ops vs full-frame every-frame at output res."""
    iw, ih = internal_size(p)
    full = float(p.output_w * p.output_h)
    return (iw * ih * p.mask_fill) / (full * p.every_n)


def tile_count(p: Placement) -> int:
    from nr86.tiles import iter_tiles

    iw, ih = internal_size(p)
    return len(iter_tiles(ih, iw, p.tile, p.overlap))


def _row(p: Placement) -> dict:
    iw, ih = internal_size(p)
    leak_full = Placement(
        scaling_ratio=1.0,
        every_n=1,
        mask_fill=1.0,
        tile=p.tile,
        overlap=p.overlap,
        output_w=p.output_w,
        output_h=p.output_h,
    )
    ours = pixel_ops(p)
    theirs = pixel_ops(leak_full)
    return {
        "internal": (iw, ih),
        "scaling_ratio": p.scaling_ratio,
        "every_n": p.every_n,
        "mask_fill": p.mask_fill,
        "tile": p.tile,
        "tiles": tile_count(p),
        "pixel_ops_vs_output": round(ours, 4),
        "cheapness_vs_leak_fullframe": round(theirs / max(ours, 1e-9), 3),
    }


def summarize(p: Placement) -> dict:
    """Average (mask + every-N) and worst-case (full mask, every frame).

    Fast camera motion drives mask_fill → 1 and kills reprojection.
    Size the frame-time budget for worst_case or you get DRG-style cliffs
    exactly when the frame rate matters.
    """
    worst = Placement(
        scaling_ratio=p.scaling_ratio,
        every_n=1,
        mask_fill=1.0,
        tile=p.tile,
        overlap=p.overlap,
        output_w=p.output_w,
        output_h=p.output_h,
    )
    avg = _row(p)
    return {
        "output": (p.output_w, p.output_h),
        **avg,
        "worst_case": _row(worst),
        "note": (
            "Average assumes honest mask + every-N. Worst-case is scaling "
            "alone (~2.2x at 0.67). Budget worst-case milliseconds."
        ),
    }


@dataclass(frozen=True)
class Mask:
    """Binary / soft mask over internal-res pixels. 1 = run the student."""

    fill: float
