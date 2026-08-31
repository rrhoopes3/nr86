"""Static HUD ignore boxes. DXHR cannot hide the full HUD.

Normalized (x0, y0, x1, y1) in [0, 1]. Training multiplies L1 by the
keep-mask so the student is not graded on HP / comms / minimap / ammo.
Center prompts and the M1 marker move — those are not boxed.
"""

from __future__ import annotations

import numpy as np

# Fractions of the Quality-input frame (same on 1080p and 960x540).
DXHR_IGNORE = (
    (0.00, 0.00, 0.22, 0.10),  # HP / energy
    (0.70, 0.00, 1.00, 0.18),  # comms portrait
    (0.00, 0.72, 0.22, 1.00),  # minimap
    (0.82, 0.86, 1.00, 1.00),  # ammo
)

PRESETS = {"dxhr": DXHR_IGNORE, "none": ()}


def keep_mask(height: int, width: int, boxes: tuple = DXHR_IGNORE) -> np.ndarray:
    """1 = train this pixel, 0 = ignore."""
    m = np.ones((height, width), dtype=np.float32)
    for x0, y0, x1, y1 in boxes:
        xa = int(round(x0 * width))
        xb = int(round(x1 * width))
        ya = int(round(y0 * height))
        yb = int(round(y1 * height))
        m[ya:yb, xa:xb] = 0.0
    return m


def tile_keep(
    height: int,
    width: int,
    y0: int,
    x0: int,
    tile_h: int,
    tile_w: int,
    boxes: tuple = DXHR_IGNORE,
) -> np.ndarray:
    full = keep_mask(height, width, boxes)
    return full[y0 : y0 + tile_h, x0 : x0 + tile_w]
