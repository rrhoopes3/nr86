# nr86

Neural rendering engine for Ampere (`sm_86`). Distill an INT8 student you
own; tiles, masks, ScalingRatio on 3rd-gen tensor cores. Not a DLSS installer.

Built for a GeForce RTX 3090. The leaked DLSS 5 Neural Rendering path is an
FP8 teacher JIT-compiled as FP16. That is why a 3090 can *show* Hogwarts at
~30 FPS DLAA+NR and why Cyberpunk / DRG still fall off a cliff. This repo
is the other job: an engine that actually uses 3rd-gen tensor cores
(INT8 / INT4 / 2:4 sparsity), at Quality-input size, with tiling and a mask.

Win 11 is not the problem. FP8-cast-to-FP16 is the problem.

## What you get today

| Piece | Status |
| --- | --- |
| Hardware doctor (sm, VRAM, FP8/INT8/sparsity) | working |
| Self-teacher (HQ capture → cheap + Lanczos teacher) | working — this is the quality target |
| PSNR/SSIM eval vs teacher **and** identity | working (`nr86 eval`) |
| Farneback mvec + warp / every-N composite | working |
| Placement: average **and** worst-case | working |
| Residual UNet (`gn` FP16 / `none` INT8) | working |
| Capture addon: BGRA/RGBA/RGB10A2, depth formats, prev-color | rebuilt |
| INT8 / TensorRT-RTX | scaffolding (SDK not on PATH yet) |
| CoopVec / RTXNS | different product (neural shading), stub only |

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
python -m nr86 bench --ckpt runs/smoke/student.pt --size 1280x720
python -m nr86 place --preset ampere --size 1920x1080
```

`eval` must beat identity. `place` prints worst-case (~2.2× on a camera
swing) next to the 13× average. Budget the worst-case row.

## Why the 30-series tweak is not the product

| Existing 30xx addon | This project |
| --- | --- |
| Makes Blackwell cubins *run* on sm_86 | Makes a student *fast* on sm_86 |
| Approx-FP16 of an FP8 148M teacher | INT8 W8A8 (then INT4 + 2:4) on 3rd-gen tensor cores |
| Full-frame, every frame, post-upscale | Internal res via ScalingRatio, mask, every 2nd frame, tiles |
| Driver JIT of patched PTX | Your ONNX → TensorRT-RTX, or RTXNS in-shader |

24 GB VRAM is the one thing that does not suck: a 148M teacher plus
activations fits. Distilling a 20–40M student on one 3090 is a week, not a
cluster. The smoke preset is minutes.

## Repo layout

```
nr86/                 Python engine (dataset, student, distill, bench, TRT)
addons/nr86_capture/  ReShade addon: dump color/depth for offline training
shaders/              CoopVec MLP stub (RTXNS track)
third_party/          TensorRT-RTX, RTXNS, ReShade headers
docs/                 architecture + measurement protocol
```

## Next (days, not another DLL drop)

1. TensorRT-RTX SDK on PATH — every current ms number is a proxy.
2. One offline title, HUD off, highest res the game will do, F9 burst.
   `nr86 inspect` the dump, then `ingest` + `selfteach --size 1280x720`.
3. `eval` must beat identity before any INT8 work. Then `ampere_int8`
   (no GroupNorm). If INT8 does not beat the leak numbers, cut pixels.

Do not grow toward 20–40M until the quality gate passes. Do not put this
on multiplayer.
