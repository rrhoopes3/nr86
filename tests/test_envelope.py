from __future__ import annotations

import numpy as np

from nr86.envelope import color_stats, fit_envelope, in_envelope
from nr86.eval import infer_regime, quality_gate
from nr86.runtime import FrameRunner


class CountingNet:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.calls += 1
        return x[:, :3].clamp(0, 1)


def _gray(h=16, w=16, v=0.4):
    return np.full((h, w, 3), v, dtype=np.float32)


def _smart_vision(h=16, w=16):
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = 0.85
    rgb[..., 1] = 0.05
    rgb[..., 2] = 0.75
    return rgb


def test_envelope_rejects_channel_split():
    rows = [color_stats(_gray(v=v)) for v in (0.25, 0.35, 0.45, 0.55)]
    env = fit_envelope(rows, pad=0.03)
    assert in_envelope(_gray(v=0.4), env)
    assert not in_envelope(_smart_vision(), env)


def test_passthrough_skips_student():
    import torch

    rows = [color_stats(_gray(v=v)) for v in (0.3, 0.4, 0.5)]
    env = fit_envelope(rows, pad=0.02)
    net = CountingNet()
    runner = FrameRunner(
        net, every_n=1, tile=8, overlap=0, dirty_tiles=False, envelope=env
    )
    color = _smart_vision()
    packed = torch.from_numpy(
        np.concatenate(
            [
                np.moveaxis(color, 2, 0),
                np.zeros((1, 16, 16), dtype=np.float32),
                np.zeros((2, 16, 16), dtype=np.float32),
            ],
            axis=0,
        )
    ).unsqueeze(0)
    pred, stats = runner.run(
        packed, color=color, mvec=np.zeros((16, 16, 2), dtype=np.float32), frame_index=0
    )
    assert stats.path == "passthrough"
    assert stats.ran_student is False
    assert net.calls == 0
    assert np.allclose(pred, color, atol=1e-5)


def test_quiet_and_motion_gates():
    assert infer_regime({"warp_clean": 12, "fullframe": 20}, 0.04, 0.6) == "quiet"
    assert infer_regime({"storm_identity": 29, "fullframe": 3}, 0.25, 0.09) == "motion"
    assert infer_regime({"fullframe": 32}, 0.4, 1.0) == "motion"
    assert infer_regime({"fullframe": 29, "storm_identity": 3}, 0.107, 0.91) == "motion"
    assert infer_regime(
        {"storm_identity": 9, "warp_clean": 9, "fullframe": 12, "fullframe_dirty": 2},
        0.135,
        0.44,
    ) == "quiet"
    assert infer_regime({"passthrough": 32}, 0.0, 0.0) == "overlay"
    ok, _gate = quality_gate(0.0, "overlay")
    assert ok
    ok, gate = quality_gate(0.04, "motion")
    assert ok and gate == "pass"
    ok, gate = quality_gate(-0.01, "motion")
    assert not ok
    ok, gate = quality_gate(0.04, "quiet")
    assert not ok
    ok, gate = quality_gate(0.30, "quiet")
    assert ok
