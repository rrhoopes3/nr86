from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Tile:
    y0: int
    x0: int
    y1: int
    x1: int


def iter_tiles(height: int, width: int, tile: int, overlap: int) -> list[Tile]:
    if tile <= 0:
        raise ValueError("tile must be > 0")
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap must be in [0, tile)")
    stride = tile - overlap
    tiles: list[Tile] = []
    y = 0
    while True:
        y1 = min(y + tile, height)
        y0 = y1 - tile
        if y0 < 0:
            y0 = 0
            y1 = min(tile, height)
        x = 0
        while True:
            x1 = min(x + tile, width)
            x0 = x1 - tile
            if x0 < 0:
                x0 = 0
                x1 = min(tile, width)
            tiles.append(Tile(y0, x0, y1, x1))
            if x1 >= width:
                break
            x += stride
        if y1 >= height:
            break
        y += stride
    # Dedup identical pads on tiny images.
    uniq: list[Tile] = []
    seen: set[Tile] = set()
    for t in tiles:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def hann2d(
    h: int,
    w: int,
    overlap: int,
    *,
    fade_top: bool = True,
    fade_bottom: bool = True,
    fade_left: bool = True,
    fade_right: bool = True,
) -> np.ndarray:
    """Separable Hann window. Image-border sides should not fade (keep 1.0)."""
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)
    if overlap > 0:
        n = min(overlap, max(h // 2, 1), max(w // 2, 1))
        if n > 1:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n, dtype=np.float32))
            if fade_top:
                wy[:n] = ramp
            if fade_bottom:
                wy[-n:] = ramp[::-1]
            if fade_left:
                wx[:n] = ramp
            if fade_right:
                wx[-n:] = ramp[::-1]
    return np.outer(wy, wx).astype(np.float32)


def stitch(
    tiles: list[tuple[Tile, torch.Tensor]],
    height: int,
    width: int,
    overlap: int,
) -> torch.Tensor:
    """Blend tiled NCHW outputs back to a full tensor. All tiles same C."""
    if not tiles:
        raise ValueError("no tiles")
    c = tiles[0][1].shape[1]
    device = tiles[0][1].device
    dtype = tiles[0][1].dtype
    acc = torch.zeros(1, c, height, width, device=device, dtype=dtype)
    weight = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
    for tile, chunk in tiles:
        th, tw = chunk.shape[-2:]
        win = torch.from_numpy(
            hann2d(
                th,
                tw,
                overlap,
                fade_top=tile.y0 > 0,
                fade_bottom=tile.y1 < height,
                fade_left=tile.x0 > 0,
                fade_right=tile.x1 < width,
            )
        ).to(device=device, dtype=dtype)
        acc[:, :, tile.y0 : tile.y1, tile.x0 : tile.x1] += chunk * win
        weight[:, :, tile.y0 : tile.y1, tile.x0 : tile.x1] += win
    return acc / weight.clamp_min(1e-6)


def pad_to_tile(x: torch.Tensor, tile: int) -> tuple[torch.Tensor, int, int]:
    """Pad NCHW on the bottom-right so H,W >= tile."""
    h, w = x.shape[-2:]
    ph = max(0, tile - h)
    pw = max(0, tile - w)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return x, h, w
