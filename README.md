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
| Capture → ingest (`nr86 from-dump`; first-frame `prev_color: null` is valid) | working — 32-bit and 64-bit addons |
| Self-teacher (Lanczos + depth punch / cheap + mvec smear) | working — still a resample teacher |
| PSNR/SSIM eval vs teacher **and** identity | working (`nr86 eval`, `--use-trt`, `--offset`) |
| Residual-after-warp mask; skip-frame + dirty-tile runtime | working — measured tiles + ms |
| Placement: average **and** worst-case | **cost model**, not measured |
| Residual UNet (`gn` FP16 / `none` reserved for later INT8) | working — smoke only |
| TensorRT-RTX FP16 student in `run_frame` | working on this 3090 |
| INT8 QDQ / INT4 / 2:4 / RTXNS | **not implemented** |

Pulled in as git submodules / vendored headers (open or official only):

- [NVIDIA/TensorRT-RTX](https://github.com/NVIDIA/TensorRT-RTX) — AOT/JIT samples, Apache-2.0
- [NVIDIA-RTX/RTXNS](https://github.com/NVIDIA-RTX/RTXNS) — Neural Shading / `VK_NV_cooperative_vector`
- [crosire/reshade](https://github.com/crosire/reshade) `include/` — capture addon API, BSD-3-Clause

Not pulled: leaked `nvngx_dlssnr.dll`, Discord Ampere addons, DLSS5-Feeder, OptiScaler.
See [LEGAL.md](LEGAL.md).

## Measured on this 3090 (honest)

**Synth** (24 frames, 512², smoke 193k): identity 22.21 → 25.79 dB, **+3.58 dB**.
Hybrid skip+dirty holds that after the warp-without-mask bug (−2.06 dB).
PyTorch full-frame ~10 ms. TRT skip+dirty **6.48 ms**. CLI 5.35 ms at 858×482
had H2D/D2H **disabled** — do not quote it as a full pass.

**Deus Ex: Human Revolution** (1920×1080 dump → 1280×720 teach, HUD on):
synth weights **−8.76 dB**. Same-scene train/eval with no depth **+2.16 dB** —
that did not hold out (−2.16 dB) once Generic Depth was a real buffer.
Smoke trained on a depth dump, last-32 hold-out **+0.50 dB** at 3200 steps.
A later Sarif HQ lobby burst (unseen room) **+1.55 dB** full-frame /
**+1.14 dB** skip+dirty. Zeroing depth on that room is **−3.96 dB**.
720p TRT `run_frame` skip+dirty **14.3 ms** (copies on) vs PyTorch 24.8 ms.
Student-only TRT **11.4 ms**. None of those are under the ~10.7 ms line
(that line was eager PyTorch at 858×482).

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
python -m nr86 bench --ckpt runs/smoke/student.pt --data datasets/synth --every-n 2 --dirty-tiles --use-trt
python -m nr86 from-dump --src "D:\Games\SomeGame\nr86_capture" --ckpt runs\overnight\smoke200\student.pt --use-trt
python -m nr86 eval --ckpt runs\dxhr-smoke200\student.pt --data datasets\q720-dxhr --offset 200
python -m nr86 place --preset ampere --size 1920x1080
```

`eval` must beat identity. `place` is a pixel-ops model (~13× average vs
~2.2× worst-case). That ratio is **not** a measured millisecond saving.
`bench --data` reports `tiles_executed` and CUDA-event `mean_ms`.

## Why the 30-series tweak is not the product

| Existing 30xx addon | This project |
| --- | --- |
| Makes Blackwell cubins *run* on sm_86 | Makes a student *fast* on sm_86 |
| Approx-FP16 of an FP8 148M teacher | Intended INT8 later; today FP16 |
| Full-frame, every frame, post-upscale | Internal res, mask, every 2nd frame |
| Driver JIT of patched PTX | Your ONNX → TensorRT-RTX |

24 GB VRAM is the one thing that does not suck: a 148M teacher plus
activations fits. Distilling a 20–40M student on one 3090 is a week, not a
cluster. The smoke preset is minutes. **Do not grow width.** Held-out DXHR
clears +0.25 dB; 720p `run_frame` TRT does not yet beat ~10.7 ms.

## Repo layout

```
nr86/                 Python engine (dataset, student, distill, runtime, bench)
addons/nr86_capture/  ReShade addon: dump color/depth for offline training
shaders/              CoopVec MLP stub (RTXNS track — postponed)
third_party/          TensorRT-RTX, RTXNS, ReShade headers
docs/                 architecture + measurement protocol
```

## Next

Keep smoke. Cut the Python `run_frame` HWC round-trip so 720p TRT can
approach the student-only 11.4 ms. INT8 / Ampere / width growth stay off
until a 720p `run_frame` mean (copies on) is under the ~10.7 ms line.
Another quiet lobby F9 will not teach much; harder motion would.

Do not put this on multiplayer.
