"""Placeholder teacher — not DLSS 5, not NVIDIA weights.

A cheap image operator so the distill pipeline is real without a leaked
teacher. Replace `teacher_rgb` in the dataset with your own RGB targets
when you have them.
"""

from __future__ import annotations

import numpy as np


def placeholder_teacher(color: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """color/depth float32 HWC RGB in [0,1] → enhanced RGB in [0,1]."""
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("color must be HWC RGB")
    img = color.astype(np.float32)
    z = np.clip(depth.astype(np.float32), 0.0, 1.0)
    if z.ndim == 3:
        z = z[..., 0]
    luma = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    blur = _box_blur(img, k=7)
    contrast = img - blur
    near = 1.0 - z
    out = img + contrast * (0.35 + 0.45 * near[..., None])
    wash = _box_blur(img, k=21)
    cavity = (1.0 - luma) * (0.25 + 0.5 * z)
    out = out + wash * cavity[..., None] * 0.25
    out[..., 0] += 0.02 * near
    out[..., 2] += 0.03 * z
    return np.clip(out, 0.0, 1.0)


def _box_blur(img: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return img
    pad = k // 2
    p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    cs = np.zeros((p.shape[0] + 1, p.shape[1] + 1, p.shape[2]), dtype=np.float32)
    cs[1:, 1:] = p
    np.cumsum(cs, axis=0, out=cs)
    np.cumsum(cs, axis=1, out=cs)
    h, w = img.shape[:2]
    window = (
        cs[k : h + k, k : w + k]
        - cs[0:h, k : w + k]
        - cs[k : h + k, 0:w]
        + cs[0:h, 0:w]
    )
    return window / float(k * k)
