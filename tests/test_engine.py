from pathlib import Path

import numpy as np
import torch
import pytest

from nr86.dataset import FrameDataset, load_frame, load_manifest
from nr86.legal import LegalBlock, assert_path_allowed
from nr86.models.student import build_student, count_params
from nr86.synth import write_synth


def test_legal_blocks_leak_names(tmp_path: Path):
    p = tmp_path / "nvngx_dlssnr.dll"
    p.write_bytes(b"nope")
    with pytest.raises(LegalBlock):
        assert_path_allowed(p)


def test_presets_param_bands():
    smoke = count_params(build_student("smoke"))
    ampere = count_params(build_student("ampere"))
    target = count_params(build_student("target"))
    assert smoke < 2_000_000
    assert 2_000_000 < ampere < 20_000_000
    assert 10_000_000 < target < 50_000_000
    assert smoke < ampere < target


def test_student_residual_shape():
    m = build_student("smoke")
    x = torch.rand(2, 6, 128, 128)
    y = m(x)
    assert y.shape == (2, 3, 128, 128)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0


def test_synth_roundtrip(tmp_path: Path):
    root = tmp_path / "ds"
    n = write_synth(root, frames=4, size=128, hq_scale=2)
    assert n == 4
    ds = FrameDataset(root)
    x, y = ds[0]
    assert x.shape == (6, 128, 128)
    assert y.shape == (3, 128, 128)
    rec = load_manifest(root)[0]
    fr = load_frame(root, rec)
    assert rec["teacher"]
    assert rec.get("teacher_kind") == "selfteach"
    # Cheap box vs Lanczos must differ or there is no quality task.
    assert float(np.mean(np.abs(fr.color - fr.teacher))) > 1e-4


def test_int8_preset_has_no_groupnorm():
    m = build_student("ampere_int8")
    assert not any(isinstance(mod, torch.nn.GroupNorm) for mod in m.modules())
    m2 = build_student("ampere")
    assert any(isinstance(mod, torch.nn.GroupNorm) for mod in m2.modules())
