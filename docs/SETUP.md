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

Copy `nr86_capture.addon64` next to the game exe. Enable Generic Depth.
**Hide the HUD.** F10 = one frame; **F9 burst** so `color_prev.bmp` exists
for Farneback. Then:

```powershell
python -m nr86 inspect --src "D:\Games\SomeGame\nr86_capture"
python -m nr86 ingest --src "D:\Games\SomeGame\nr86_capture" --out datasets/raw
python -m nr86 selfteach --data datasets/raw --out datasets/q720 --size 1280x720
python -m nr86 eval --ckpt runs/smoke/student.pt --data datasets/q720
python -m nr86 eval --ckpt runs/smoke/student.pt --data datasets/q720 --ablate depth
python -m nr86 bench --ckpt runs/smoke/student.pt --data datasets/q720 --every-n 2 --dirty-tiles
```

`cv` extra (`opencv`) is required for mvec. Without it, every-N placement
is a cost-model lie. Frame 0 of a capture dump has `"prev_color": null`;
that is valid and must ingest.

## Driver notes (Ampere, not Win 11)

HAGS on, current Game Ready driver, NVIDIA App overlay off if it fights
ReShade, SmartScreen on unsigned injectors is expected. Keep Smooth Motion /
OptiScaler **off** while measuring this engine so you do not mix frame gen
into the ms numbers.
