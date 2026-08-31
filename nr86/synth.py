from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

from nr86.dataset import DatasetWriter, Frame
from nr86.selfteach import pair_frame


def _checker(h: int, w: int, cell: int = 32) -> np.ndarray:
    yy, xx = np.indices((h, w))
    board = ((yy // cell) + (xx // cell)) % 2
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[board == 0] = (0.75, 0.62, 0.48)
    img[board == 1] = (0.22, 0.28, 0.36)
    return img


def generate_frame(size: int, t: float, seed: int = 0) -> Frame:
    """A panning checkerboard + a near sphere at `size` (usually HQ)."""
    rng = np.random.default_rng(seed)
    h = w = size
    max_shift = max(8, int(round(48 * (size / 256))))
    shift = min(int(round(t * max_shift)), max_shift)
    cell = max(4, size // 32)
    base = _checker(h, w + max_shift + 1, cell=cell)
    color = base[:, shift : shift + w].copy()
    yy, xx = np.indices((h, w))
    cx = w * (0.35 + 0.3 * np.sin(t * 2 * np.pi))
    cy = h * 0.5
    r = size * 0.18
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    sphere = np.clip(1.0 - dist / r, 0.0, 1.0)
    color += sphere[..., None] * np.array([0.15, 0.35, 0.55], dtype=np.float32)
    color = np.clip(color, 0, 1)
    depth = 0.85 - 0.7 * sphere
    depth += 0.02 * rng.random((h, w)).astype(np.float32)
    depth = np.clip(depth, 0, 1)
    mvec = np.zeros((h, w, 2), dtype=np.float32)
    mvec[..., 0] = (48.0 * (size / 256)) / float(w)
    return Frame(f"{int(round(t * 10000)):06d}", color, depth, mvec, None)


def write_synth(out: Path, frames: int, size: int, seed: int = 0, hq_scale: int = 2) -> int:
    """Render at size*hq_scale, then self-teach down to `size` (quality target)."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    writer = DatasetWriter(out)
    hq_scale = max(int(hq_scale), 2)
    hq = size * hq_scale
    for i in range(frames):
        t = i / max(frames - 1, 1)
        raw = generate_frame(hq, t, seed=seed + i)
        paired = pair_frame(raw, size, size) if hq != size else raw
        if paired.teacher is None:
            from nr86.models.teacher import placeholder_teacher

            paired.teacher = placeholder_teacher(paired.color, paired.depth)
        writer.add(
            paired,
            extra={
                "teacher_kind": "selfteach",
                "hq_scale": hq_scale,
                "teacher_cues": "lanczos+depth_punch",
                "cheap_cues": "box+bilinear+mvec_smear",
            },
        )
    return writer.n_frames
