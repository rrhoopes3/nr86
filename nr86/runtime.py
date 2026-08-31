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

CUDA path keeps warp/mask/composite on GPU so a TensorRT student can
beat 10 ms PyTorch full-frame. CPU path stays numpy for tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from nr86.reproject import (
    composite,
    composite_nchw,
    fill_ratio,
    residual_mask,
    residual_mask_nchw,
    warp_nchw,
    warp_rgb,
)
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


def _hwc_to_nchw(img: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(img))
    if t.ndim == 2:
        t = t[None, None]
    elif t.ndim == 3 and t.shape[-1] in (2, 3):
        t = t.permute(2, 0, 1).unsqueeze(0)
    else:
        t = t.unsqueeze(0)
    return t.to(device=device, dtype=dtype, non_blocking=True)


def _nchw_to_hwc(t: torch.Tensor) -> np.ndarray:
    return t[0].permute(1, 2, 0).clamp(0.0, 1.0).detach().cpu().numpy()


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
    if packed.device.type == "cuda":
        return _run_frame_cuda(
            model,
            packed,
            color=color,
            mvec=mvec,
            prev_color=prev_color,
            prev_out=prev_out,
            frame_index=frame_index,
            every_n=every_n,
            tile=tile,
            overlap=overlap,
            dirty_tiles=dirty_tiles,
            tile_thresh=tile_thresh,
        )
    return _run_frame_cpu(
        model,
        packed,
        color=color,
        mvec=mvec,
        prev_color=prev_color,
        prev_out=prev_out,
        frame_index=frame_index,
        every_n=every_n,
        tile=tile,
        overlap=overlap,
        dirty_tiles=dirty_tiles,
        tile_thresh=tile_thresh,
    )


def _run_frame_cpu(
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
    tile_thresh: float,
) -> tuple[np.ndarray, ExecStats]:
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


def _run_frame_cuda(
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
    tile_thresh: float,
) -> tuple[np.ndarray, ExecStats]:
    device = packed.device
    dtype = packed.dtype
    _n, _c, h, w = packed.shape
    n_tiles = len(iter_tiles(h, w, tile, overlap))
    color_t = _hwc_to_nchw(color, device, dtype)
    mvec_t = _hwc_to_nchw(mvec, device, dtype)
    prev_c = _hwc_to_nchw(prev_color, device, dtype) if prev_color is not None else None
    prev_o = _hwc_to_nchw(prev_out, device, dtype) if prev_out is not None else None
    warped_in = warp_nchw(prev_c, mvec_t) if prev_c is not None else None
    warped_out = warp_nchw(prev_o, mvec_t) if prev_o is not None else color_t
    mask = residual_mask_nchw(color_t, warped_in)
    fill = float(mask.float().mean().item())
    skip_slot = every_n > 1 and frame_index % every_n != 0 and prev_out is not None

    if skip_slot and not dirty_tiles:
        return _nchw_to_hwc(warped_out), ExecStats(n_tiles, 0, False, fill, "warp_skip")

    if skip_slot and dirty_tiles:
        if float(mask.float().mean().item()) < tile_thresh:
            return _nchw_to_hwc(warped_out), ExecStats(n_tiles, 0, False, fill, "warp_clean")
        pred = model(packed)
        pred = composite_nchw(pred, warped_out, mask)
        return _nchw_to_hwc(pred), ExecStats(n_tiles, n_tiles, True, fill, "fullframe_dirty")

    pred = model(packed)
    if prev_out is not None:
        pred = composite_nchw(pred, warped_out, mask)
    return _nchw_to_hwc(pred), ExecStats(n_tiles, n_tiles, True, fill, "fullframe")
