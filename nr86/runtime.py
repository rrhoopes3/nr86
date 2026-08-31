"""Measured execution: skip frames and dirty tiles.

placement.py is a cost model. This module is the thing that actually
skips the network and reports tiles_executed / milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from nr86.reproject import composite, fill_ratio, residual_mask, warp_rgb
from nr86.tiles import Tile, iter_tiles, stitch


@dataclass
class ExecStats:
    tiles_total: int
    tiles_executed: int
    ran_student: bool
    mask_fill: float


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
    device = packed.device
    _n, _c, h, w = packed.shape
    tiles = iter_tiles(h, w, tile, overlap)

    warped_in = warp_rgb(prev_color, mvec) if prev_color is not None else None
    warped_out = warp_rgb(prev_out, mvec) if prev_out is not None else color
    mask = residual_mask(color, warped_in)
    fill = fill_ratio(mask)

    skip_network = every_n > 1 and frame_index % every_n != 0 and prev_out is not None
    if skip_network:
        return warped_out, ExecStats(len(tiles), 0, False, fill)

    if dirty_tiles and prev_out is not None:
        chunks = []
        executed = 0
        for t in tiles:
            if tile_dirty(mask, t, tile_thresh):
                chunk = packed[:, :, t.y0 : t.y1, t.x0 : t.x1]
                chunks.append((t, model(chunk)))
                executed += 1
            else:
                reuse = torch.from_numpy(
                    np.moveaxis(warped_out[t.y0 : t.y1, t.x0 : t.x1], 2, 0)[None]
                ).to(device=device, dtype=packed.dtype)
                chunks.append((t, reuse))
        out = stitch(chunks, h, w, overlap)[0].permute(1, 2, 0).cpu().numpy()
        return np.clip(out, 0.0, 1.0), ExecStats(len(tiles), executed, executed > 0, fill)

    pred = model(packed)[0].permute(1, 2, 0).cpu().numpy()
    if prev_out is not None:
        pred = composite(pred, warped_out, mask)
    return np.clip(pred, 0.0, 1.0), ExecStats(len(tiles), len(tiles), True, fill)
