# nr86

Research scaffold for an Ampere (`sm_86`) neural-rendering student.
Not a working DLSS workaround, not a shipping neural-rendering engine,
and not an INT8 product yet.

The hardware thesis is still the right one: a 3090 has no FP8 MMA. The
leaked DLSS 5 Neural Rendering path is an FP8 teacher JIT-compiled as
FP16. This repo is the other job — a student *you own*, at Quality-input
size, with a mask and every-Nth reuse, aimed at 3rd-gen tensor cores.
INT8 / INT4 / 2:4 sparsity are **roadmap**, not implementation.

## What you get today

| Piece | Status |
| --- | --- |
| Hardware doctor (sm, VRAM, FP8/INT8/sparsity) | working |
| Capture → ingest (first-frame `prev_color: null` is valid) | working |
| Self-teacher (Lanczos + depth punch / cheap + mvec smear) | working — still synthetic |
| PSNR/SSIM eval vs teacher **and** identity | working (`nr86 eval`) |
| Residual-after-warp mask; skip-frame + dirty-tile runtime | working — measured tiles + ms |
| Placement: average **and** worst-case | **cost model**, not measured |
| Residual UNet (`gn` FP16 / `none` reserved for later INT8) | working |
| TensorRT-RTX FP16 bench | path exists; needs the SDK |
| INT8 QDQ / INT4 / 2:4 / RTXNS | **not implemented** |

Pulled in as git submodules / vendored headers (open or official only):

- [NVIDIA/TensorRT-RTX](https://github.com/NVIDIA/TensorRT-RTX) — AOT/JIT samples, Apache-2.0
- [NVIDIA-RTX/RTXNS](https://github.com/NVIDIA-RTX/RTXNS) — Neural Shading / `VK_NV_cooperative_vector`
- [crosire/reshade](https://github.com/crosire/reshade) `include/` — capture addon API, BSD-3-Clause

Not pulled: leaked `nvngx_dlssnr.dll`, Discord Ampere addons, DLSS5-Feeder, OptiScaler.
See [LEGAL.md](LEGAL.md).

## Quick start (this machine: 3090, 24 GB, sm_86)

```powershell
cd "B:\Rando Apps\Nvda DLSS5 workaround"
python -m pip install -e ".[dev]"
python -m nr86 doctor
python -m nr86 synth --out datasets/synth --frames 24 --size 512
python -m nr86 train --data datasets/synth --preset smoke --steps 40 --out runs/smoke
python -m nr86 eval --ckpt runs/smoke/student.pt --data datasets/synth
python -m nr86 eval --ckpt runs/smoke/student.pt --data datasets/synth --every-n 2 --dirty-tiles --ablate none
python -m nr86 bench --ckpt runs/smoke/student.pt --size 1280x720 --try-trt
python -m nr86 bench --ckpt runs/smoke/student.pt --data datasets/synth --every-n 2 --dirty-tiles
python -m nr86 place --preset ampere --size 1920x1080
```

`eval` must beat identity. `place` is a pixel-ops model (~13× average vs
~2.2× worst-case). That ratio is **not** a measured millisecond saving.
`bench --data` reports `tiles_executed` and CUDA-event `mean_ms`.

## Why the 30-series tweak is not the product

| Existing 30xx addon | This project |
| --- | --- |
| Makes Blackwell cubins *run* on sm_86 | Makes a student *fast* on sm_86 |
| Approx-FP16 of an FP8 148M teacher | Intended INT8 later; today FP16 PyTorch |
| Full-frame, every frame, post-upscale | Internal res, mask, every 2nd frame, tiles |
| Driver JIT of patched PTX | Your ONNX → TensorRT-RTX (when the SDK is there) |

24 GB VRAM is the one thing that does not suck: a 148M teacher plus
activations fits. Distilling a 20–40M student on one 3090 is a week, not a
cluster. The smoke preset is minutes.

## Repo layout

```
nr86/                 Python engine (dataset, student, distill, runtime, bench)
addons/nr86_capture/  ReShade addon: dump color/depth for offline training
shaders/              CoopVec MLP stub (RTXNS track — postponed)
third_party/          TensorRT-RTX, RTXNS, ReShade headers
docs/                 architecture + measurement protocol
```

## Next (defensible result, not another DLL drop)

1. TensorRT-RTX SDK on PATH — a real FP16 720p engine number.
2. One offline title, HUD off, F9 burst. `inspect` → `ingest` → `selfteach`.
3. `eval` must beat identity on that capture. Ablate RGB / depth / mvec.
4. Only then QDQ INT8 on `ampere_int8`. Postpone INT4, 2:4, and RTXNS.

The milestone that turns this from a scaffold into systems research:
**this student beats identity on real captures and runs in X ms on a 3090.**

Do not grow toward 20–40M until the quality gate passes. Do not put this
on multiplayer.
