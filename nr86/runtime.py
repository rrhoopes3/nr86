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

CUDA ``FrameRunner`` keeps prev_color / prev_out on device, pins color/mvec
staging, computes mask fill once, and D2Hs only when the caller asks
(eval dumps). Packed input is already on GPU; H2D of color/mvec is the
honest numpy-harness "copies on". A real hook would bind GPU textures.
CPU path stays numpy for tests.
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


def nchw_to_hwc(t: torch.Tensor) -> np.ndarray:
    return t[0].permute(1, 2, 0).clamp(0.0, 1.0).detach().cpu().numpy()


def _blit_hwc_to_pinned(host: torch.Tensor, img: np.ndarray) -> None:
    src = np.ascontiguousarray(img, dtype=np.float32)
    dst = host.numpy()
    channels = int(dst.shape[1])
    if src.ndim == 2:
        np.copyto(dst[0, 0], src)
        return
    for c in range(channels):
        np.copyto(dst[0, c], src[..., c])


class FrameRunner:
    """Stateful runner. Prev tensors stay on device across frames."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        every_n: int,
        tile: int,
        overlap: int,
        dirty_tiles: bool,
        tile_thresh: float = 0.02,
    ) -> None:
        self.model = model
        self.every_n = every_n
        self.tile = tile
        self.overlap = overlap
        self.dirty_tiles = dirty_tiles
        self.tile_thresh = tile_thresh
        self._have_prev = False
        self._n_tiles: int | None = None
        self._host_color: torch.Tensor | None = None
        self._host_mvec: torch.Tensor | None = None
        self._dev_color: torch.Tensor | None = None
        self._dev_mvec: torch.Tensor | None = None
        self._prev_color: torch.Tensor | None = None
        self._prev_out: torch.Tensor | None = None
        self._prev_color_np: np.ndarray | None = None
        self._prev_out_np: np.ndarray | None = None

    def reset(self) -> None:
        self._have_prev = False
        self._prev_color_np = None
        self._prev_out_np = None

    def seed(self, prev_color: np.ndarray, prev_out: np.ndarray, device: torch.device, dtype: torch.dtype) -> None:
        """Load numpy prev into device buffers (functional run_frame API)."""
        h, w = prev_color.shape[:2]
        self._ensure(h, w, device, dtype)
        if device.type == "cuda":
            _blit_hwc_to_pinned(self._host_color, prev_color)
            self._prev_color.copy_(self._host_color, non_blocking=True)
            _blit_hwc_to_pinned(self._host_color, prev_out)
            self._prev_out.copy_(self._host_color, non_blocking=True)
        else:
            self._prev_color_np = np.ascontiguousarray(prev_color)
            self._prev_out_np = np.ascontiguousarray(prev_out)
        self._have_prev = True

    def run(
        self,
        packed: torch.Tensor,
        *,
        color: np.ndarray,
        mvec: np.ndarray,
        frame_index: int,
        to_numpy: bool = True,
    ) -> tuple[np.ndarray | torch.Tensor, ExecStats]:
        if packed.device.type != "cuda":
            return self._run_cpu(packed, color=color, mvec=mvec, frame_index=frame_index)
        return self._run_cuda(
            packed, color=color, mvec=mvec, frame_index=frame_index, to_numpy=to_numpy
        )

    def _ensure(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self._dev_color is not None
            and self._dev_color.shape[-2:] == (h, w)
            and self._dev_color.device == device
        ):
            return
        self._n_tiles = len(iter_tiles(h, w, self.tile, self.overlap))
        if device.type == "cuda":
            self._host_color = torch.empty(1, 3, h, w, dtype=torch.float32, pin_memory=True)
            self._host_mvec = torch.empty(1, 2, h, w, dtype=torch.float32, pin_memory=True)
            self._dev_color = torch.empty(1, 3, h, w, dtype=dtype, device=device)
            self._dev_mvec = torch.empty(1, 2, h, w, dtype=dtype, device=device)
            self._prev_color = torch.empty(1, 3, h, w, dtype=dtype, device=device)
            self._prev_out = torch.empty(1, 3, h, w, dtype=dtype, device=device)
        else:
            self._host_color = None
            self._host_mvec = None
            self._dev_color = None
            self._dev_mvec = None
            self._prev_color = None
            self._prev_out = None

    def _run_cpu(
        self,
        packed: torch.Tensor,
        *,
        color: np.ndarray,
        mvec: np.ndarray,
        frame_index: int,
    ) -> tuple[np.ndarray, ExecStats]:
        pred, stats = _run_frame_cpu(
            self.model,
            packed,
            color=color,
            mvec=mvec,
            prev_color=self._prev_color_np,
            prev_out=self._prev_out_np,
            frame_index=frame_index,
            every_n=self.every_n,
            tile=self.tile,
            overlap=self.overlap,
            dirty_tiles=self.dirty_tiles,
            tile_thresh=self.tile_thresh,
        )
        self._prev_color_np = color
        self._prev_out_np = pred
        self._have_prev = True
        return pred, stats

    def _run_cuda(
        self,
        packed: torch.Tensor,
        *,
        color: np.ndarray,
        mvec: np.ndarray,
        frame_index: int,
        to_numpy: bool,
    ) -> tuple[np.ndarray | torch.Tensor, ExecStats]:
        device = packed.device
        dtype = packed.dtype
        h, w = color.shape[:2]
        self._ensure(h, w, device, dtype)
        n_tiles = int(self._n_tiles or 0)
        _blit_hwc_to_pinned(self._host_color, color)
        self._dev_color.copy_(self._host_color, non_blocking=True)
        _blit_hwc_to_pinned(self._host_mvec, mvec)
        self._dev_mvec.copy_(self._host_mvec, non_blocking=True)
        color_t = self._dev_color
        mvec_t = self._dev_mvec
        warped_in = warp_nchw(self._prev_color, mvec_t) if self._have_prev else None
        warped_out = warp_nchw(self._prev_out, mvec_t) if self._have_prev else color_t
        mask = residual_mask_nchw(color_t, warped_in)
        fill = float(mask.float().mean().item())
        skip_slot = self.every_n > 1 and frame_index % self.every_n != 0 and self._have_prev

        if skip_slot and not self.dirty_tiles:
            out = warped_out
            stats = ExecStats(n_tiles, 0, False, fill, "warp_skip")
        elif skip_slot and self.dirty_tiles and fill < self.tile_thresh:
            out = warped_out
            stats = ExecStats(n_tiles, 0, False, fill, "warp_clean")
        elif skip_slot and self.dirty_tiles:
            pred = self.model(packed)
            out = composite_nchw(pred, warped_out, mask)
            stats = ExecStats(n_tiles, n_tiles, True, fill, "fullframe_dirty")
        else:
            pred = self.model(packed)
            out = composite_nchw(pred, warped_out, mask) if self._have_prev else pred
            stats = ExecStats(n_tiles, n_tiles, True, fill, "fullframe")

        self._prev_color.copy_(color_t)
        self._prev_out.copy_(out)
        self._have_prev = True
        if to_numpy:
            return nchw_to_hwc(out), stats
        return out, stats


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
    to_numpy: bool = True,
) -> tuple[np.ndarray | torch.Tensor, ExecStats]:
    """packed is 1x6xHxW on the model device. color/mvec/prev are HWC numpy.

    Prefer ``FrameRunner`` for sequences so prev stays on the GPU. This
    functional wrapper still accepts numpy prev (tests, one-shot calls).
    """
    runner = FrameRunner(
        model,
        every_n=every_n,
        tile=tile,
        overlap=overlap,
        dirty_tiles=dirty_tiles,
        tile_thresh=tile_thresh,
    )
    if prev_color is not None and prev_out is not None:
        runner.seed(prev_color, prev_out, packed.device, packed.dtype)
    return runner.run(
        packed,
        color=color,
        mvec=mvec,
        frame_index=frame_index,
        to_numpy=to_numpy if packed.device.type == "cuda" else True,
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
