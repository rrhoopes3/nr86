from pathlib import Path

import numpy as np

from nr86.inspect_capture import inspect_capture
from nr86.metrics import psnr, ssim
from nr86.reproject import fill_ratio, motion_luma_mask, warp_rgb
from nr86.selfteach import pair_frame
from nr86.synth import generate_frame


def test_psnr_identity():
    x = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    assert psnr(x, x) >= 98.0


def test_ssim_identity():
    x = np.random.default_rng(0).random((48, 48, 3)).astype(np.float32)
    assert ssim(x, x) > 0.99


def test_warp_zero_mvec_is_identity():
    img = np.random.default_rng(1).random((40, 50, 3)).astype(np.float32)
    mvec = np.zeros((40, 50, 2), dtype=np.float32)
    out = warp_rgb(img, mvec)
    assert np.allclose(out, img, atol=1e-5)


def test_selfteach_pair_changes_res_and_makes_a_task():
    hq = generate_frame(64, 0.3, seed=2)
    pair = pair_frame(hq, 32, 32)
    assert pair.color.shape == (32, 32, 3)
    assert pair.teacher.shape == (32, 32, 3)
    assert pair.depth.shape == (32, 32)
    assert pair.mvec.shape == (32, 32, 2)
    assert float(np.mean(np.abs(pair.color - pair.teacher))) > 1e-4


def test_motion_mask_fill_is_one_on_big_flow():
    color = np.zeros((16, 16, 3), dtype=np.float32)
    mvec = np.ones((16, 16, 2), dtype=np.float32) * 0.05
    mask = motion_luma_mask(color, color, mvec, motion_norm=0.004)
    assert fill_ratio(mask) == 1.0


def test_inspect_flags_bad_depth_size(tmp_path: Path):
    folder = tmp_path / "000000"
    folder.mkdir()
    (folder / "color.bmp").write_bytes(b"BM")
    (folder / "depth.f32").write_bytes(b"\x00" * 12)
    (folder / "meta.json").write_text(
        '{"id":"000000","width":4,"height":4,"color":"color.bmp",'
        '"color_format":"b8g8r8a8","depth":"depth.f32",'
        '"depth_format":"d32_float","depth_width":4,"depth_height":4}',
        encoding="utf-8",
    )
    report = inspect_capture(tmp_path)
    assert report["ok"] is False
    assert any("depth.f32" in i for i in report["issues"])
