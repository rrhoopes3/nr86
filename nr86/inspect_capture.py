"""Validate a raw nr86_capture dump before training on it."""

from __future__ import annotations

import json
from pathlib import Path

from nr86.legal import scan_tree_or_raise


def inspect_capture(src: Path) -> dict:
    src = Path(src)
    scan_tree_or_raise(src)
    metas = sorted(src.glob("**/meta.json"))
    if not metas:
        raise FileNotFoundError(f"no meta.json under {src}")
    issues: list[str] = []
    formats: set[str] = set()
    depth_formats: set[str] = set()
    prev_ok = 0
    depth_ok = 0
    for meta_p in metas:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        folder = meta_p.parent
        color = folder / meta.get("color", "color.bmp")
        if not color.exists():
            issues.append(f"{folder.name}: missing color")
        cf = meta.get("color_format") or "unspecified"
        formats.add(str(cf))
        if cf == "unspecified":
            issues.append(f"{folder.name}: color_format missing — rebuild the capture addon")
        depth_name = meta.get("depth")
        if depth_name:
            dp = folder / depth_name
            if not dp.exists():
                issues.append(f"{folder.name}: depth listed but missing")
            else:
                nbytes = dp.stat().st_size
                w = int(meta.get("depth_width") or meta.get("width") or 0)
                h = int(meta.get("depth_height") or meta.get("height") or 0)
                expect = w * h * 4
                if w and h and nbytes != expect:
                    issues.append(
                        f"{folder.name}: depth.f32 is {nbytes} bytes, expected {expect} "
                        f"({w}x{h} float32). Format was {meta.get('depth_format')}."
                    )
                else:
                    depth_ok += 1
            depth_formats.add(str(meta.get("depth_format") or "unspecified"))
        prev = folder / meta.get("prev_color", "color_prev.bmp")
        if prev.exists():
            prev_ok += 1
        note = meta.get("note") or ""
        if "post-ui" in note.lower() or "post_ui" in note.lower():
            pass
    report = {
        "frames": len(metas),
        "color_formats": sorted(formats),
        "depth_formats": sorted(depth_formats),
        "depth_ok": depth_ok,
        "prev_color_frames": prev_ok,
        "issues": issues,
        "hud": (
            "This dump is post-UI LDR (reshade_finish_effects). Hide the HUD "
            "in-game. Real DLSS runs on linear HDR pre-UI — acceptable for "
            "the self-teacher, not a shipping hook."
        ),
        "ok": len(issues) == 0,
    }
    return report
