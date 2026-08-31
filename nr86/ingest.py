"""Turn a ReShade capture dump into an nr86 dataset.

Motion vectors: games do not expose them. Consecutive color (and
`color_prev.bmp` from the addon) → Farneback. First frame may be zero;
later frames must not silently be zeros.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from nr86.dataset import DatasetWriter, Frame
from nr86.legal import assert_path_allowed, scan_tree_or_raise
from nr86.reproject import estimate_flow


def _load_color(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_depth(path: Path, h: int, w: int, meta: dict) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == h * w:
        return np.clip(raw.reshape(h, w), 0.0, 1.0)
    # Resized depth vs color (common if depth target != backbuffer).
    declared = meta.get("depth_width"), meta.get("depth_height")
    if declared[0] and declared[1] and raw.size == int(declared[0]) * int(declared[1]):
        src = raw.reshape(int(declared[1]), int(declared[0]))
        from PIL import Image as P

        return np.asarray(
            P.fromarray(src, mode="F").resize((w, h), resample=P.Resampling.NEAREST),
            dtype=np.float32,
        )
    raise ValueError(
        f"depth {path} has {raw.size} floats, expected {h*w} "
        f"(or depth_width*depth_height from meta). Re-dump after updating the addon."
    )


def ingest(
    src: Path,
    out: Path,
    placeholder: bool = False,
) -> int:
    src = Path(src)
    scan_tree_or_raise(src)
    metas = sorted(src.glob("**/meta.json"))
    if not metas:
        raise FileNotFoundError(f"no meta.json under {src}")
    writer = DatasetWriter(out)
    prev_color: np.ndarray | None = None
    n_flow = 0
    n_zero = 0
    for meta_p in metas:
        assert_path_allowed(meta_p)
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        folder = meta_p.parent
        color_p = folder / meta.get("color", "color.bmp")
        if not color_p.exists():
            alt = folder / "color.png"
            color_p = alt if alt.exists() else color_p
        color = _load_color(color_p)
        h, w, _ = color.shape
        depth_name = meta.get("depth")
        depth_p = folder / depth_name if depth_name else folder / "depth.f32"
        if depth_p.exists():
            depth = _load_depth(depth_p, h, w, meta)
        else:
            depth = np.zeros((h, w), dtype=np.float32)
        prev_p = folder / meta.get("prev_color", "color_prev.bmp")
        if prev_p.exists():
            prev_color = _load_color(prev_p)
        mvec = np.zeros((h, w, 2), dtype=np.float32)
        source = "zero"
        mvec_p = folder / meta.get("mvec", "mvec.f32")
        if mvec_p.exists():
            mv = np.fromfile(mvec_p, dtype=np.float32)
            if mv.size == h * w * 2:
                mvec = mv.reshape(h, w, 2)
                source = "file"
        elif prev_color is not None:
            mvec, source = estimate_flow(prev_color, color)
        if source.startswith("zero"):
            n_zero += 1
        else:
            n_flow += 1
        teacher = None
        t_p = folder / "teacher.png"
        if t_p.exists():
            teacher = _load_color(t_p)
        elif placeholder:
            from nr86.models.teacher import placeholder_teacher

            teacher = placeholder_teacher(color, depth)
        fid = meta.get("id", folder.name)
        writer.add(
            Frame(str(fid), color, depth, mvec, teacher),
            extra={
                "mvec_source": source,
                "color_format": meta.get("color_format"),
                "depth_format": meta.get("depth_format"),
                "swap_note": meta.get("note"),
            },
        )
        prev_color = color
    print(f"ingested {writer.n_frames} frames -> {out}  mvec={n_flow} zero={n_zero}")
    if n_flow == 0:
        print(
            "WARNING: every frame has zero motion vectors. Burst-capture (F9) "
            "or drop color_prev.bmp so Farneback can run. The 13x placement "
            "number degrades to scaling-only (~2.2x) without mvec."
        )
    if not placeholder:
        print(
            "No teacher written (ingest is raw). Run `nr86 selfteach` on this "
            "dataset after a high-res capture — that is the quality target."
        )
    return writer.n_frames
