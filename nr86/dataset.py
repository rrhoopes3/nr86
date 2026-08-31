from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from nr86.legal import assert_path_allowed, scan_tree_or_raise


@dataclass
class Frame:
    frame_id: str
    color: np.ndarray  # HWC float32 RGB
    depth: np.ndarray  # HW float32
    mvec: np.ndarray  # HW2 float32
    teacher: np.ndarray | None = None


def _read_png_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _write_png_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="RGB").save(path)


def pack_input(frame: Frame) -> np.ndarray:
    """Return 6×H×W float32."""
    h, w, _ = frame.color.shape
    depth = frame.depth.astype(np.float32)
    if depth.shape != (h, w):
        raise ValueError(f"depth {depth.shape} != color {(h, w)}")
    mvec = frame.mvec.astype(np.float32)
    if mvec.shape != (h, w, 2):
        raise ValueError(f"mvec {mvec.shape} != {(h, w, 2)}")
    packed = np.concatenate(
        [
            np.moveaxis(frame.color, 2, 0),
            depth[None, ...],
            np.moveaxis(mvec, 2, 0),
        ],
        axis=0,
    )
    return packed.astype(np.float32)


class DatasetWriter:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.jsonl"
        self._count = 0
        if self.manifest.exists():
            self._count = sum(1 for _ in self.manifest.open("r", encoding="utf-8"))

    def add(self, frame: Frame, extra: dict | None = None) -> None:
        fid = frame.frame_id
        color_p = self.frames_dir / f"{fid}_color.png"
        depth_p = self.frames_dir / f"{fid}_depth.npy"
        mvec_p = self.frames_dir / f"{fid}_mvec.npy"
        _write_png_rgb(color_p, frame.color)
        np.save(depth_p, frame.depth.astype(np.float32))
        np.save(mvec_p, frame.mvec.astype(np.float32))
        rec: dict = {
            "id": fid,
            "color": str(color_p.relative_to(self.root)).replace("\\", "/"),
            "depth": str(depth_p.relative_to(self.root)).replace("\\", "/"),
            "mvec": str(mvec_p.relative_to(self.root)).replace("\\", "/"),
            "teacher": None,
        }
        if frame.teacher is not None:
            t_p = self.frames_dir / f"{fid}_teacher.png"
            _write_png_rgb(t_p, frame.teacher)
            rec["teacher"] = str(t_p.relative_to(self.root)).replace("\\", "/")
        if extra:
            rec.update(extra)
        with self.manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self._count += 1

    @property
    def n_frames(self) -> int:
        return self._count


def load_manifest(root: Path) -> list[dict]:
    root = Path(root)
    scan_tree_or_raise(root)
    path = root / "manifest.jsonl"
    assert_path_allowed(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_frame(root: Path, rec: dict) -> Frame:
    root = Path(root)
    color = _read_png_rgb(root / rec["color"])
    depth = np.load(root / rec["depth"])
    mvec = np.load(root / rec["mvec"])
    teacher = None
    if rec.get("teacher"):
        teacher = _read_png_rgb(root / rec["teacher"])
    return Frame(rec["id"], color, depth, mvec, teacher)


class FrameDataset:
    def __init__(self, root: Path, require_teacher: bool = True) -> None:
        self.root = Path(root)
        self.rows = load_manifest(self.root)
        if require_teacher:
            self.rows = [r for r in self.rows if rec_has_teacher(r)]
        if not self.rows:
            raise FileNotFoundError(f"no frames in {self.root}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        frame = load_frame(self.root, self.rows[idx])
        x = pack_input(frame)
        if frame.teacher is None:
            raise RuntimeError(f"frame {frame.frame_id} has no teacher RGB")
        y = np.moveaxis(frame.teacher, 2, 0)
        return x, y.astype(np.float32)


def rec_has_teacher(rec: dict) -> bool:
    return bool(rec.get("teacher"))
