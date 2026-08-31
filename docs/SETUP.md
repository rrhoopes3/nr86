# Setup

## Python engine

Python 3.10+, a CUDA PyTorch wheel, this repo.

```powershell
python -m pip install -e ".[dev]"
python -m nr86 doctor
```

Optional:

```powershell
python -m pip install -e ".[onnx]"   # export checks
python -m pip install -e ".[trt]"    # TensorRT-RTX Python (needs NVIDIA SDK)
python -m pip install -e ".[cv]"     # Farneback mvec for captures that lack it
```

TensorRT for RTX itself: [developer download](https://developer.nvidia.com/tensorrt-rtx).
After install, `tensorrt_rtx.exe --help` should work and `python -c "import tensorrt_rtx"`
should import. Put the SDK `lib` directory on `PATH`.

## Third-party SDKs

```powershell
git submodule update --init --depth 1
powershell -File scripts/fetch_third_party.ps1
```

RTXNS is the Neural Shading sample tree. Building its Donut samples needs
VS 2022, CMake, and a Vulkan SDK with `VK_NV_cooperative_vector`. You do
**not** need that to run `python -m nr86`.

## Capture addon (MSVC)

ReShade with **full add-on support**, offline single-player, no anti-cheat.

```powershell
cmake -S addons/nr86_capture -B addons/nr86_capture/build -G "Visual Studio 17 2022" -A x64
cmake --build addons/nr86_capture/build --config Release
```

Copy `addons/nr86_capture/nr86_capture.addon64` (built locally) next to the game exe.
Enable Generic Depth. ReShade must have **full add-on support**.
**Hide the HUD.** F10 = one frame; **F9 burst** so `color_prev.bmp` exists
for Farneback. Then:

```powershell
python -m nr86 from-dump --src "D:\Games\SomeGame\nr86_capture" --ckpt runs\overnight\smoke200\student.pt --use-trt
# same as inspect → ingest → selfteach 1280x720 → eval --every-n 2 --dirty-tiles
python -m nr86 eval --ckpt runs\overnight\smoke200\student.pt --data datasets/q720 --ablate depth
python -m nr86 bench --ckpt runs\overnight\smoke200\student.pt --data datasets/q720 --every-n 2 --dirty-tiles --use-trt
```

`cv` extra (`opencv`) is required for mvec. Without it, every-N placement
is a cost-model lie. Frame 0 of a capture dump has `"prev_color": null`;
that is valid and must ingest.

**Raw retention.** A DXHR burst is ~0.5 GB raw and a taught city dump
can be ~10 GB. `B:` hit 9 MB free mid-selfteach; staging deletion
saved that run. After a taught set evals and is copied to
`datasets/q540-…`, delete `datasets/_dxhr_*` staging and the matching
`datasets/raw-dxhr-*` (or move raw off `B:`). Do not keep raw on the
same volume as overnight trains. The game `nr86_capture\` folder can
stay; leftover IDs are not new bursts — stage by `meta.json` mtime
only (`scripts/_stage_by_mtime.py`).

## Driver notes (Ampere, not Win 11)

HAGS on, current Game Ready driver, NVIDIA App overlay off if it fights
ReShade, SmartScreen on unsigned injectors is expected. Keep Smooth Motion /
OptiScaler **off** while measuring this engine so you do not mix frame gen
into the ms numbers.
