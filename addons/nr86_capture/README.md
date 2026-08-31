# nr86_capture

ReShade add-on: **color + depth + previous color**. Not a NVIDIA NR dump.

Needs ReShade with full add-on support and Generic Depth. **Offline
single-player only — no anti-cheat.**

**Hide the HUD.** This hook is `reshade_finish_effects` (post-UI,
display-referred LDR). Real DLSS runs on linear HDR pre-UI. Fine for the
self-teacher; not a shipping path.

F10 = one frame. **F9 burst** writes `color_prev.bmp` so ingest can
Farneback motion vectors. Frame 0 writes `"prev_color": null` (valid).
Without a burst, mvec is zero and the placement **cost model** degrades
to scaling-only (~2.2×). That 13× figure is not a measured millisecond
saving.

Supported color: BGRA8, RGBA8, BGRX, RGB10A2, RGBA16F, RGBA32F.
Supported depth: D32F, D24S8, D16, D32FS8 (converted to float32).
`python -m nr86 inspect --src <dump>` before you trust a dataset.

```powershell
cmake -S addons/nr86_capture -B addons/nr86_capture/build -G "Visual Studio 17 2022" -A x64
cmake --build addons/nr86_capture/build --config Release
```

Output: `addons/nr86_capture/build/Release/nr86_capture.addon64`

```powershell
python -m nr86 inspect --src "<game>\nr86_capture"
python -m nr86 ingest --src "<game>\nr86_capture" --out datasets/raw
python -m nr86 selfteach --data datasets/raw --out datasets/q720 --size 1280x720
```

`ingest` no longer invents a placeholder teacher. Self-teach is the
quality target: capture high, downsample cheap vs Lanczos.
