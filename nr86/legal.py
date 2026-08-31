from __future__ import annotations

from pathlib import Path

from nr86.config import BLOCKED_NAMES, BLOCKED_SUBSTRINGS


class LegalBlock(RuntimeError):
    """Raised when a path looks like a leaked NVIDIA NR blob."""


def assert_path_allowed(path: Path | str) -> Path:
    p = Path(path)
    blob = str(p).lower().replace("\\", "/")
    name = p.name.lower()
    if name in BLOCKED_NAMES or any(s in blob for s in BLOCKED_SUBSTRINGS):
        raise LegalBlock(
            f"nr86 will not open {p}. That looks like a leaked NVIDIA Neural "
            "Rendering blob (weights, cubins, or the Ampere PTX swap). "
            "See LEGAL.md. Feed color/depth/mvec (and optional teacher PNG), "
            "not nvngx_dlssnr.dll."
        )
    return p


def scan_tree_or_raise(root: Path | str) -> None:
    root = Path(root)
    if not root.exists():
        return
    for child in root.rglob("*"):
        if child.is_file():
            assert_path_allowed(child)
