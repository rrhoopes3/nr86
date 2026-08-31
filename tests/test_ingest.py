"""Capture dump → ingest. First frame writes prev_color: null."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from nr86.dataset import load_frame, load_manifest
from nr86.ingest import ingest
from nr86.inspect_capture import inspect_capture


def _write_png(path: Path, rgb: np.ndarray) -> None:
    u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="RGB").save(path)


def _write_dump(root: Path) -> Path:
    dump = root / "dump"
    h = w = 16
    depth = np.full((h, w), 0.5, dtype=np.float32)
    a = np.zeros((h, w, 3), dtype=np.float32)
    a[:, :8] = (0.8, 0.2, 0.1)
    b = np.zeros((h, w, 3), dtype=np.float32)
    b[:, 8:] = (0.8, 0.2, 0.1)

    f0 = dump / "000000"
    f0.mkdir(parents=True)
    _write_png(f0 / "color.png", a)
    depth.tofile(f0 / "depth.f32")
    (f0 / "meta.json").write_text(
        json.dumps(
            {
                "id": "000000",
                "width": w,
                "height": h,
                "color": "color.png",
                "color_format": "r8g8b8",
                "depth": "depth.f32",
                "depth_format": "d32_float",
                "depth_width": w,
                "depth_height": h,
                "prev_color": None,
            }
        ),
        encoding="utf-8",
    )

    f1 = dump / "000001"
    f1.mkdir()
    _write_png(f1 / "color.png", b)
    _write_png(f1 / "color_prev.png", a)
    depth.tofile(f1 / "depth.f32")
    (f1 / "meta.json").write_text(
        json.dumps(
            {
                "id": "000001",
                "width": w,
                "height": h,
                "color": "color.png",
                "color_format": "r8g8b8",
                "depth": "depth.f32",
                "depth_format": "d32_float",
                "depth_width": w,
                "depth_height": h,
                "prev_color": "color_prev.png",
            }
        ),
        encoding="utf-8",
    )
    return dump


def test_inspect_tolerates_null_prev_color(tmp_path: Path):
    dump = _write_dump(tmp_path)
    report = inspect_capture(dump)
    assert report["ok"] is True
    assert report["frames"] == 2
    assert report["prev_color_frames"] == 1


def test_ingest_first_frame_null_prev_color(tmp_path: Path):
    dump = _write_dump(tmp_path)
    out = tmp_path / "ds"
    n = ingest(dump, out)
    assert n == 2
    rows = load_manifest(out)
    f0 = load_frame(out, rows[0])
    f1 = load_frame(out, rows[1])
    assert f0.color.shape == (16, 16, 3)
    assert f0.mvec.shape == (16, 16, 2)
    assert float(np.abs(f0.mvec).max()) == 0.0
    assert rows[0]["mvec_source"] == "zero"
    assert rows[1]["mvec_source"] in ("farneback", "zero_no_cv2", "file")
    assert f1.color.shape == (16, 16, 3)


def test_from_dump_inspect_ingest_selfteach_eval(tmp_path: Path):
    from nr86.from_dump import from_dump
    from nr86.models.student import build_student, save_student

    dump = _write_dump(tmp_path)
    ckpt = tmp_path / "student.pt"
    save_student(build_student("smoke"), ckpt)
    out = from_dump(
        dump,
        tmp_path / "raw",
        tmp_path / "taught",
        ckpt,
        size="16x16",
        every_n=1,
        dirty_tiles=False,
    )
    assert out["ingested"] == 2
    assert out["eval"]["frames"] == 2
    assert out["eval"]["backend"] == "pytorch"
