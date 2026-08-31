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
