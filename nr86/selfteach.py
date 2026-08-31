"""Supersampled self-teacher: reconstruct HQ from a cheap downsample.

Zero NVIDIA bits. Capture (or synth) at high res, then:

- teacher = Lanczos downsample to Quality-input
- color   = box/area downsample (the cheap render the student sees)

That is the quality target. Gate 4 compares the student to this teacher
*and* to the identity (cheap color vs teacher). Fast at doing nothing fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from nr86.dataset import DatasetWriter, Frame, load_frame, load_manifest


def resize_rgb(img: np.ndarray, width: int, height: int, resample: int) -> np.ndarray:
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    out = Image.fromarray(u8, mode="RGB").resize((width, height), resample=resample)
    return np.asarray(out, dtype=np.float32) / 255.0


def resize_map(img: np.ndarray, width: int, height: int, resample: int) -> np.ndarray:
    if img.ndim == 2:
        pil = Image.fromarray(img.astype(np.float32), mode="F")
        return np.asarray(pil.resize((width, height), resample=resample), dtype=np.float32)
    chans = [
        resize_map(img[..., c], width, height, resample) for c in range(img.shape[-1])
    ]
    return np.stack(chans, axis=-1)


def _cheapen(rgb: np.ndarray) -> np.ndarray:
    """Simulate a cheap internal-res present: extra 2x box then bilinear back."""
    h, w, _ = rgb.shape
    if w < 4 or h < 4:
        return rgb
    tiny = resize_rgb(rgb, max(1, w // 2), max(1, h // 2), Image.Resampling.BOX)
    return resize_rgb(tiny, w, h, Image.Resampling.BILINEAR)


def pair_frame(hq: Frame, out_w: int, out_h: int) -> Frame:
    """HQ frame → Quality-input pair. mvec stays normalized (scale-invariant)."""
    teacher = resize_rgb(hq.color, out_w, out_h, Image.Resampling.LANCZOS)
    color = _cheapen(resize_rgb(hq.color, out_w, out_h, Image.Resampling.BOX))
    depth = resize_map(hq.depth, out_w, out_h, Image.Resampling.NEAREST)
    mvec = resize_map(hq.mvec, out_w, out_h, Image.Resampling.BILINEAR)
    return Frame(hq.frame_id, color, depth, mvec, teacher)


def selfteach_dataset(src: Path, out: Path, size: str) -> int:
    import shutil

    w_s, h_s = size.lower().split("x")
    out_w, out_h = int(w_s), int(h_s)
    rows = load_manifest(src)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    writer = DatasetWriter(out)
    n = 0
    for rec in rows:
        hq = load_frame(src, rec)
        paired = pair_frame(hq, out_w, out_h)
        writer.add(
            paired,
            extra={"teacher_kind": "selfteach", "hq_id": rec.get("id"), "size": size},
        )
        n += 1
    print(f"selfteach {n} frames {size} -> {out}")
    return n
