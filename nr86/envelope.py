"""Color-stat envelope for overlay / vision-mode pass-through.

Smart Vision and other full-frame post effects leave the training
distribution of per-frame RGB mean/std. The student then degrades
the frame. Identity is the correct product output for those states.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def color_stats(color: np.ndarray) -> dict[str, float]:
    rgb = np.asarray(color, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB, got {rgb.shape}")
    mean = rgb.reshape(-1, 3).mean(axis=0)
    std = rgb.reshape(-1, 3).std(axis=0)
    return {
        "mean_r": float(mean[0]),
        "mean_g": float(mean[1]),
        "mean_b": float(mean[2]),
        "std_r": float(std[0]),
        "std_g": float(std[1]),
        "std_b": float(std[2]),
        "mean_spread": float(mean.max() - mean.min()),
    }


def fit_envelope(stats_rows: list[dict[str, float]], pad: float = 0.03) -> dict:
    keys = ("mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "mean_spread")
    env: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = [row[k] for row in stats_rows]
        lo, hi = min(vals), max(vals)
        env[k] = {"lo": round(lo - pad, 4), "hi": round(hi + pad, 4)}
    env["pad"] = pad
    env["n_frames"] = len(stats_rows)
    return env


def in_envelope(color: np.ndarray, env: dict) -> bool:
    st = color_stats(color)
    for k in ("mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "mean_spread"):
        band = env.get(k)
        if not isinstance(band, dict):
            continue
        if st[k] < float(band["lo"]) or st[k] > float(band["hi"]):
            return False
    return True


DEFAULT_ENVELOPE_PATH = Path("results") / "color_envelope.json"


def load_envelope(path: Path | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_envelope(
    path: Path | None = None,
    *,
    enabled: bool = True,
    explicit: dict | None = None,
) -> dict | None:
    """Eval/bench default: use results/color_envelope.json when present."""
    if not enabled:
        return None
    if explicit is not None:
        return explicit
    return load_envelope(path or DEFAULT_ENVELOPE_PATH)


def save_envelope(env: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2), encoding="utf-8")
    return path
