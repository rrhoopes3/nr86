"""Supersampled self-teacher: reconstruct HQ from a cheap downsample.

Zero NVIDIA bits. Capture (or synth) at high res, then:

- teacher = Lanczos downsample, then a depth-aware contrast punch
- color   = box/area downsample, bilinear cheapen, then a small smear along mvec

This is still a synthetic teacher. Depth and motion now change the pair, so
those channels are not inert — but they are not a game-engine teacher either.
Gate 4 compares the student to this teacher *and* to identity (cheap vs teacher).
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


def _depth_punch(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Near pixels get more local contrast. Gives depth a reason to exist."""
    d = np.clip(depth.astype(np.float32), 0.0, 1.0)
    mean = rgb.mean(axis=(0, 1), keepdims=True)
    gain = 1.0 + 0.35 * (1.0 - d)[..., None]
    return np.clip(mean + (rgb - mean) * gain, 0.0, 1.0).astype(np.float32)


def _motion_smear(rgb: np.ndarray, mvec: np.ndarray) -> np.ndarray:
    """Cheap input is smeared along mvec; teacher is not."""
    h, w, _ = rgb.shape
    mag = np.linalg.norm(mvec.astype(np.float32), axis=2)
    if float(mag.max()) < 1e-6:
        return rgb
    try:
        import cv2
    except ImportError:
        return rgb
    dx = mvec[..., 0] * w * 0.35
    dy = mvec[..., 1] * h * 0.35
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    shifted = cv2.remap(
        np.clip(rgb, 0, 1).astype(np.float32),
        (grid_x - dx).astype(np.float32),
        (grid_y - dy).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    amount = np.clip(mag / 0.02, 0.0, 0.5)[..., None]
    return (rgb * (1.0 - amount) + shifted * amount).astype(np.float32)


def pair_frame(hq: Frame, out_w: int, out_h: int) -> Frame:
    """HQ frame → Quality-input pair. mvec stays normalized (scale-invariant)."""
    teacher = resize_rgb(hq.color, out_w, out_h, Image.Resampling.LANCZOS)
    color = _cheapen(resize_rgb(hq.color, out_w, out_h, Image.Resampling.BOX))
    depth = resize_map(hq.depth, out_w, out_h, Image.Resampling.NEAREST)
    mvec = resize_map(hq.mvec, out_w, out_h, Image.Resampling.BILINEAR)
    teacher = _depth_punch(teacher, depth)
    color = _motion_smear(color, mvec)
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
            extra={
                "teacher_kind": "selfteach",
                "hq_id": rec.get("id"),
                "size": size,
                "teacher_cues": "lanczos+depth_punch",
                "cheap_cues": "box+bilinear+mvec_smear",
            },
        )
        n += 1
    print(f"selfteach {n} frames {size} -> {out}")
    return n
