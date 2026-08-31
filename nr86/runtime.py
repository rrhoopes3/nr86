"""Measured execution: skip frames and dirty tiles.

placement.py is a cost model. This module skips the network and reports
tiles_executed / milliseconds.

Control flow (do not invert):

- Student frame: always full-frame ``model(packed)``, then residual composite.
  The Python per-tile loop is launch-bound on a 3090. Do not use it here.
- Skip slot + dirty_tiles: warp, then consult the residual mask. Clean →
  warp only. Any dirty tile → full-frame student + composite. Never return
  warp-only *before* reading the mask.
- Skip slot without dirty_tiles: warp only (explicit).

If skip+dirty PyTorch ms is worse than full-frame every frame, keep
every_n=1 until the student itself is TensorRT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from nr86.reproject import composite, fill_ratio, residual_mask, warp_rgb
from nr86.tiles import Tile, iter_tiles


@dataclass
class ExecStats:
    tiles_total: int
    tiles_executed: int
    ran_student: bool
    mask_fill: float
    path: str


def tile_dirty(mask: np.ndarray, tile: Tile, thresh: float = 0.02) -> bool:
    patch = mask[tile.y0 : tile.y1, tile.x0 : tile.x1]
    if patch.size == 0:
        return True
    return float(np.mean(patch)) >= thresh


@torch.no_grad()
def run_frame(
    model: torch.nn.Module,
    packed: torch.Tensor,
    *,
    color: np.ndarray,
    mvec: np.ndarray,
    prev_color: np.ndarray | None,
    prev_out: np.ndarray | None,
    frame_index: int,
    every_n: int,
    tile: int,
    overlap: int,
    dirty_tiles: bool,
    tile_thresh: float = 0.02,
) -> tuple[np.ndarray, ExecStats]:
    """packed is 1x6xHxW on the model device. color/mvec/prev are HWC numpy."""
    _n, _c, h, w = packed.shape
    tiles = iter_tiles(h, w, tile, overlap)
    n_tiles = len(tiles)

    warped_in = warp_rgb(prev_color, mvec) if prev_color is not None else None
    warped_out = warp_rgb(prev_out, mvec) if prev_out is not None else color
    mask = residual_mask(color, warped_in)
    fill = fill_ratio(mask)
    skip_slot = every_n > 1 and frame_index % every_n != 0 and prev_out is not None

    if skip_slot and not dirty_tiles:
        return warped_out, ExecStats(n_tiles, 0, False, fill, "warp_skip")

    if skip_slot and dirty_tiles:
        dirty = any(tile_dirty(mask, t, tile_thresh) for t in tiles)
        if not dirty:
            return warped_out, ExecStats(n_tiles, 0, False, fill, "warp_clean")
        pred = model(packed)[0].permute(1, 2, 0).cpu().numpy()
        pred = composite(pred, warped_out, mask)
        return np.clip(pred, 0.0, 1.0), ExecStats(n_tiles, n_tiles, True, fill, "fullframe_dirty")

    pred = model(packed)[0].permute(1, 2, 0).cpu().numpy()
    if prev_out is not None:
        pred = composite(pred, warped_out, mask)
    return np.clip(pred, 0.0, 1.0), ExecStats(n_tiles, n_tiles, True, fill, "fullframe")
