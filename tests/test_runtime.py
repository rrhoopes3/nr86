from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nr86.dataset import apply_ablation
from nr86.reproject import fill_ratio, residual_mask, warp_rgb
from nr86.runtime import FrameRunner, run_frame


class CountingNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x[:, :3].clamp(0, 1)


def _packed(color: np.ndarray, depth: np.ndarray, mvec: np.ndarray) -> torch.Tensor:
    x = np.concatenate(
        [
            np.moveaxis(color, 2, 0),
            depth[None, ...],
            np.moveaxis(mvec, 2, 0),
        ],
        axis=0,
    )
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)


def test_skip_frame_does_not_run_student():
    net = CountingNet()
    h = w = 16
    color = np.full((h, w, 3), 0.4, dtype=np.float32)
    depth = np.zeros((h, w), dtype=np.float32)
    mvec = np.zeros((h, w, 2), dtype=np.float32)
    x = _packed(color, depth, mvec)
    pred0, s0 = run_frame(
        net,
        x,
        color=color,
        mvec=mvec,
        prev_color=None,
        prev_out=None,
        frame_index=0,
        every_n=2,
        tile=8,
        overlap=0,
        dirty_tiles=False,
    )
    assert s0.ran_student is True
    assert s0.tiles_executed == s0.tiles_total
    calls_after_first = net.calls
    pred1, s1 = run_frame(
        net,
        x,
        color=color,
        mvec=mvec,
        prev_color=color,
        prev_out=pred0,
        frame_index=1,
        every_n=2,
        tile=8,
        overlap=0,
        dirty_tiles=False,
    )
    assert s1.ran_student is False
    assert s1.tiles_executed == 0
    assert net.calls == calls_after_first
    assert pred1.shape == color.shape


def test_dirty_tiles_skip_clean_regions():
    net = CountingNet()
    h = w = 16
    color = np.full((h, w, 3), 0.4, dtype=np.float32)
    depth = np.zeros((h, w), dtype=np.float32)
    mvec = np.zeros((h, w, 2), dtype=np.float32)
    x = _packed(color, depth, mvec)
    pred0, _ = run_frame(
        net,
        x,
        color=color,
        mvec=mvec,
        prev_color=None,
        prev_out=None,
        frame_index=0,
        every_n=2,
        tile=8,
        overlap=0,
        dirty_tiles=True,
    )
    calls_after_first = net.calls
    _pred1, s1 = run_frame(
        net,
        x,
        color=color,
        mvec=mvec,
        prev_color=color,
        prev_out=pred0,
        frame_index=1,
        every_n=2,
        tile=8,
        overlap=0,
        dirty_tiles=True,
    )
    assert s1.tiles_executed == 0
    assert s1.path == "warp_clean"
    assert net.calls == calls_after_first


def test_student_frame_is_always_fullframe():
    net = CountingNet()
    h = w = 16
    prev = np.full((h, w, 3), 0.2, dtype=np.float32)
    color = prev.copy()
    color[:8, :8] = 0.9
    depth = np.zeros((h, w), dtype=np.float32)
    mvec = np.zeros((h, w, 2), dtype=np.float32)
    x = _packed(color, depth, mvec)
    _pred, stats = run_frame(
        net,
        x,
        color=color,
        mvec=mvec,
        prev_color=prev,
        prev_out=prev,
        frame_index=0,
        every_n=1,
        tile=8,
        overlap=0,
        dirty_tiles=True,
    )
    assert stats.path == "fullframe"
    assert net.calls == 1
    assert stats.tiles_executed == stats.tiles_total


def test_residual_mask_ignores_pure_camera_pan():
    color = np.full((16, 16, 3), 0.4, dtype=np.float32)
    mvec = np.ones((16, 16, 2), dtype=np.float32) * 0.05
    warped = warp_rgb(color, mvec)
    mask = residual_mask(color, warped)
    assert fill_ratio(mask) < 0.05


def test_residual_mask_flags_unexplained_change():
    color = np.zeros((16, 16, 3), dtype=np.float32)
    color[:, 8:] = 1.0
    prev = np.zeros((16, 16, 3), dtype=np.float32)
    warped = warp_rgb(prev, np.zeros((16, 16, 2), dtype=np.float32))
    mask = residual_mask(color, warped)
    assert fill_ratio(mask) > 0.4


def test_hybrid_skip_slot_still_runs_when_dirty():
    net = CountingNet()
    h = w = 16
    prev = np.full((h, w, 3), 0.2, dtype=np.float32)
    color = prev.copy()
    color[:8, :8] = 0.9
    depth = np.zeros((h, w), dtype=np.float32)
    mvec = np.zeros((h, w, 2), dtype=np.float32)
    x = _packed(color, depth, mvec)
    _pred, stats = run_frame(
        net,
        x,
        color=color,
        mvec=mvec,
        prev_color=prev,
        prev_out=prev,
        frame_index=1,
        every_n=2,
        tile=8,
        overlap=0,
        dirty_tiles=True,
    )
    assert stats.ran_student is True
    assert stats.tiles_executed >= 1
    assert stats.path == "fullframe_dirty"
    assert net.calls == 1


def test_frame_runner_holds_prev_on_cpu():
    net = CountingNet()
    h = w = 16
    color = np.full((h, w, 3), 0.4, dtype=np.float32)
    depth = np.zeros((h, w), dtype=np.float32)
    mvec = np.zeros((h, w, 2), dtype=np.float32)
    x = _packed(color, depth, mvec)
    runner = FrameRunner(net, every_n=2, tile=8, overlap=0, dirty_tiles=True)
    _pred0, s0 = runner.run(x, color=color, mvec=mvec, frame_index=0)
    assert s0.path == "fullframe"
    calls = net.calls
    _pred1, s1 = runner.run(x, color=color, mvec=mvec, frame_index=1)
    assert s1.path == "warp_clean"
    assert s1.ran_student is False
    assert net.calls == calls


def test_storm_drops_warp_after_sustained_fill():
    net = CountingNet()
    runner = FrameRunner(
        net,
        every_n=2,
        tile=8,
        overlap=0,
        dirty_tiles=True,
        storm_k=3,
        storm_fill=0.1,
        storm_luma=0.02,
    )
    h = w = 16
    depth = np.zeros((h, w), dtype=np.float32)
    mvec = np.zeros((h, w, 2), dtype=np.float32)
    paths: list[str] = []
    for i in range(6):
        color = np.full((h, w, 3), float(i % 2), dtype=np.float32)
        x = _packed(color, depth, mvec)
        _pred, stats = runner.run(x, color=color, mvec=mvec, frame_index=i)
        paths.append(stats.path)
    assert "storm" in paths
    assert paths[-1] == "storm"
    assert net.calls >= 1

    still = np.full((h, w, 3), 1.0, dtype=np.float32)
    x = _packed(still, depth, mvec)
    _pred, stats = runner.run(x, color=still, mvec=mvec, frame_index=6)
    assert stats.path != "storm"


def test_ablation_zeros_channels():
    packed = np.ones((6, 4, 4), dtype=np.float32)
    rgb = apply_ablation(packed, "rgb")
    assert float(rgb[3:].max()) == 0.0
    assert float(rgb[:3].min()) == 1.0
    depth = apply_ablation(packed, "depth")
    assert float(depth[3].max()) == 0.0
    assert float(depth[4].min()) == 1.0
    mvec = apply_ablation(packed, "mvec")
    assert float(mvec[4:].max()) == 0.0
    assert float(mvec[3].min()) == 1.0
