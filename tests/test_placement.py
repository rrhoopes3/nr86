from nr86.config import Placement
from nr86.placement import pixel_ops, summarize
from nr86.tiles import hann2d, iter_tiles, stitch
import torch


def test_tiles_cover_full_image():
    tiles = iter_tiles(200, 300, tile=64, overlap=16)
    covered = torch.zeros(200, 300)
    for t in tiles:
        covered[t.y0 : t.y1, t.x0 : t.x1] = 1
        assert t.y1 - t.y0 <= 64
        assert t.x1 - t.x0 <= 64
    assert float(covered.min()) == 1.0


def test_hann_interior_is_one_without_overlap():
    w = hann2d(32, 32, overlap=0)
    assert float(w.min()) == 1.0


def test_hann_keeps_unfaded_border():
    w = hann2d(32, 32, overlap=8, fade_top=False, fade_left=False, fade_bottom=True, fade_right=True)
    assert float(w[0, 0]) == 1.0
    assert float(w[-1, -1]) < 1.0


def test_stitch_identity():
    h, w, tile, ov = 96, 80, 48, 8
    x = torch.arange(h * w, dtype=torch.float32).reshape(1, 1, h, w)
    chunks = []
    for t in iter_tiles(h, w, tile, ov):
        chunks.append((t, x[:, :, t.y0 : t.y1, t.x0 : t.x1]))
    y = stitch(chunks, h, w, ov)
    assert torch.allclose(x, y, atol=1e-5)


def test_placement_quality_is_much_cheaper_than_leak_fullframe():
    leak = Placement(
        scaling_ratio=1.0,
        every_n=1,
        mask_fill=1.0,
        output_w=1920,
        output_h=1080,
        tile=256,
        overlap=16,
    )
    ours = Placement(
        scaling_ratio=0.67,
        every_n=2,
        mask_fill=0.35,
        tile=256,
        overlap=16,
        output_w=1920,
        output_h=1080,
    )
    ratio = pixel_ops(leak) / pixel_ops(ours)
    assert ratio >= 4.0
    s = summarize(ours)
    assert s["cheapness_vs_leak_fullframe"] >= 4.0
    worst = s["worst_case"]
    assert worst["every_n"] == 1
    assert worst["mask_fill"] == 1.0
    assert 2.0 <= worst["cheapness_vs_leak_fullframe"] <= 2.5
    assert worst["cheapness_vs_leak_fullframe"] < s["cheapness_vs_leak_fullframe"]
