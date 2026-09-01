# nr86

**v0.1** — measured, gated, weights-owned Ampere student. Not a DLSS
DLL drop. Not a 720p product. Not INT8 yet.

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
| Residual-after-warp mask; skip + dirty tiles; storm-identity | working — v0.1 policy |
| Placement: average **and** worst-case | **cost model**, not measured |
| Residual UNet (`gn` FP16 / `smoke_int8` is same width, `norm=none`) | working — smoke only |
| TensorRT-RTX FP16 student in `run_frame` | working on this 3090 |
| INT8 QDQ export + hashed TRT engine | working — not a win yet (see `results/dxhr-q540-int8.json`) |
| INT4 / 2:4 / RTXNS | **not implemented** |

Pulled in as git submodules / vendored headers (open or official only):

- [NVIDIA/TensorRT-RTX](https://github.com/NVIDIA/TensorRT-RTX) — AOT/JIT samples, Apache-2.0
- [NVIDIA-RTX/RTXNS](https://github.com/NVIDIA-RTX/RTXNS) — Neural Shading / `VK_NV_cooperative_vector`
- [crosire/reshade](https://github.com/crosire/reshade) `include/` — capture addon API, BSD-3-Clause

Not pulled: leaked `nvngx_dlssnr.dll`, Discord Ampere addons, DLSS5-Feeder, OptiScaler.
See [LEGAL.md](LEGAL.md).

## v0.1 on this 3090 (honest)

Shipping graph: **smoke 193k GN, 960×540**, city-mix weights
(`runs/dxhr-city-mix/student_best.pt`). Storm-identity after 3 frames
with residual fill ≥ 0.05; exit when fill < 0.02 for 3 frames. Overlay
pass-through when color stats leave the training envelope.

| Scene | Path | ΔPSNR | Regime | Gate |
| --- | --- | --- | --- | --- |
| Sarif lobby | skip+dirty | **+1.119** | quiet ≥ +0.25 | pass |
| Detroit plaza (unseen space) | full | **+1.037** | quiet ≥ +0.25 | pass |
| City look-up | skip+dirty | **+0.005** | motion 0.0 policy | pass |
| Factory yard | skip+dirty | **+0.155** | motion 0.0 policy | pass |
| Warehouse combat | skip+dirty | **+0.159** | motion 0.0 policy | pass |
| Unseen combat3 | skip+dirty | **+0.143** | motion 0.0 policy | pass |
| Smart Vision last32 | full | **0.000** | overlay 0.0 policy | pass |

Latency **cold** (dxhr.exe closed), 960×540 TRT FP16, skip+dirty:
lobby mean was **6.42 ms** / combat all-dirty **8.004 ms** before
storm-identity. Student-path p95 is `fullframe` + `fullframe_dirty`
only. Game-open every-n=1 ~11.4 ms is clocks. 720p still misses
8.33/16.67 — v0.1 is the 540p student, not the product tensor.

Synth +3.58 dB / 6.48 ms remains a synth number. ~13× is a cost model.
Do not quote CLI TRT times as a full H2D+compute pass.

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
`bench --data` reports per-path `mean_ms` / `p95_ms` and the two-sided
latency gate on 1280×720.

## Why the 30-series tweak is not the product

| Existing 30xx addon | This project |
| --- | --- |
| Makes Blackwell cubins *run* on sm_86 | Makes a student *fast* on sm_86 |
| Approx-FP16 of an FP8 148M teacher | Intended INT8 later; today FP16 |
| Full-frame, every frame, post-upscale | Internal res, mask, every 2nd frame |
| Driver JIT of patched PTX | Your ONNX → TensorRT-RTX |

24 GB VRAM is the one thing that does not suck: a 148M teacher plus
activations fits. Distilling a 20–40M student on one 3090 is a week, not a
cluster. The smoke preset is minutes. **Do not train wider until a
storm-identity timing map says the ms fit.** Quiet ≥ +0.25 measured;
storms and overlays are 0.0 by policy. Latency cold, 960×540.

## Repo layout

```
nr86/                 Python engine (dataset, student, distill, runtime, bench)
addons/nr86_capture/  ReShade addon: dump color/depth for offline training
shaders/              CoopVec MLP stub (RTXNS track — postponed)
third_party/          TensorRT-RTX, RTXNS, ReShade headers
docs/                 architecture + measurement protocol
```

## Next

v0.1 is the 193k student with storm-identity. Next is a **junk-weight
base-24 INT8 re-bench** under that policy (quiet-scene mean, not
all-dirty combat). Train wider only if that envelope says yes.
Temporal (9-ch warped_out) stays optional and composes under the
identity floor.

Do not put this on multiplayer.
