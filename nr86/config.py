from __future__ import annotations

from dataclasses import dataclass


INPUT_CHANNELS = 6  # RGB + depth + mvec_xy
OUTPUT_CHANNELS = 3

# Filenames we will not open. See LEGAL.md.
BLOCKED_NAMES = frozenset(
    {
        "nvngx_dlssnr.dll",
        "dlssnr_ampere.addon64",
        "dlssnr_ampere.addon",
        "renodx-dlss5.addon64",
        "renodx-dlss5.addon",
    }
)
BLOCKED_SUBSTRINGS = (
    "dlssnr-sm86",
    "dlssnr_ampere",
    "nvngx_dlssnr",
    "dlssnr.dll",
)


@dataclass(frozen=True)
class StudentSpec:
    """Residual UNet width. `base` is the first encoder channel count."""

    name: str
    base: int
    levels: int
    tile: int
    overlap: int
    # "gn" is fine for FP16. GroupNorm quantizes poorly and breaks TRT QDQ
    # fusion — use "none" before blaming the calibrator.
    norm: str = "gn"

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(self.base * (2**i) for i in range(self.levels))


PRESETS: dict[str, StudentSpec] = {
    "smoke": StudentSpec(name="smoke", base=16, levels=3, tile=128, overlap=16, norm="gn"),
    "smoke_int8": StudentSpec(
        name="smoke_int8", base=16, levels=3, tile=128, overlap=16, norm="none"
    ),
    "ampere": StudentSpec(name="ampere", base=32, levels=4, tile=256, overlap=16, norm="gn"),
    "ampere_int8": StudentSpec(
        name="ampere_int8", base=32, levels=4, tile=256, overlap=16, norm="none"
    ),
    "target": StudentSpec(name="target", base=64, levels=4, tile=384, overlap=24, norm="gn"),
}


@dataclass
class Placement:
    scaling_ratio: float = 0.67
    every_n: int = 2
    mask_fill: float = 0.35
    tile: int = 256
    overlap: int = 16
    output_w: int = 1920
    output_h: int = 1080


# Product tensor: Quality-input for 1080p output. Taught dumps are 1280x720.
# Do not judge 720p TRT against eager PyTorch at 858x482 (that number was
# `bench --size 1280x720` applying scaling 0.67 a second time).
PRODUCT_OUTPUT_WH = (1920, 1080)
PRODUCT_INTERNAL_WH = (1280, 720)
# Absolute leftover at 60 Hz / 120 Hz — not "beat a historical PyTorch line".
# skip+dirty mean must fit in a 120 Hz frame. Student-path p95 must not
# exceed a 60 Hz frame (an 11 ms spike every dirty frame is a stutter).
BUDGET_SKIP_DIRTY_MEAN_MS = round(1000.0 / 120.0, 3)  # 8.333
BUDGET_STUDENT_P95_MS = round(1000.0 / 60.0, 3)  # 16.667

