# Architecture

Two tracks share one capture contract. Neither track loads NVIDIA’s NR DLL.
This is a research scaffold: the skip/dirty-tile runtime is real; INT8,
INT4, 2:4 sparsity, and RTXNS are not.

```
 game (4K/DLAA, HUD off)  or  synth at 2x
      │
      ▼
 selfteach: box-downsample + mvec smear = cheap color
            Lanczos + depth punch      = teacher
            Farneback                  = mvec  (Streamline later)
      │
      ├──────────── TensorRT-RTX track ─────────────┐
      │  residual UNet (gn for FP16, none later)    │
      │  ScalingRatio · residual-after-warp mask    │
      │  every-N skip · dirty tiles (measured)      │
      │  eval must beat identity PSNR               │
      │                                                │
      └──────────── RTXNS / CoopVec track ────────────┘
         postponed. A 6→32→32→3 MLP is neural shading,
         not a small NR reconstructor.
```

## Quality target (not optional)

The function is: **reconstruct the high-quality downsample from the cheap one**.
That is classic SR distillation. It needs no NVIDIA bits.

The self-teacher is still synthetic. Lanczos vs box+bilinear is mostly a
resampling correction. Depth punch and mvec smear give those channels a
reason to exist; they are not a game-engine teacher. Validate on a real
capture and ablate (`nr86 eval --ablate rgb|depth|mvec`).

`python -m nr86 eval --ckpt … --data …` reports student PSNR/SSIM against
the teacher **and** against identity (cheap color vs teacher). If the
student does not beat identity by ≥ 0.25 dB, it is fast at doing nothing.
Do not grow toward 20–40M until this gate passes.

Capture at the highest res the game will do (4K / DLAA / TAAU off). Hide
the HUD. The addon hooks `reshade_finish_effects` — post-UI, display-referred
LDR. Real DLSS runs on linear HDR pre-UI. Acceptable for the self-teacher;
not a shipping hook.

## Capture contract

| File | Layout |
| --- | --- |
| `{id}_color.png` | RGB uint8, cheap / Quality-input |
| `{id}_depth.npy` | float32 `H×W`, 0..1 |
| `{id}_mvec.npy` | float32 `H×W×2`, `dx/W`, `dy/H` |
| `{id}_teacher.png` | HQ downsample at the same res |
| `manifest.jsonl` | includes `teacher_kind`, `mvec_source` |

Channel pack (`N×6×H×W`): 0–2 color, 3 depth, 4–5 mvec.

Games do not expose motion vectors to ReShade. Burst-capture (F9) writes
`color_prev.bmp`; ingest runs Farneback. Frame 0 writes `"prev_color": null`;
ingest treats that as “no previous,” not a path. Streamline mvec is still
future. Without mvec, every-N + mask silently become “scaling only.”

## Placement: cost model, not a measurement

```
avg   ≈ 0.67² × mask_fill / every_n     →  ~0.079  (~13× cheaper)  [model]
worst = 0.67² × 1.0 / 1                 →  ~0.45   (~2.2× cheaper) [model]
```

`python -m nr86 place` prints both rows with `"kind": "cost_model"`.
Size the frame-time budget for **worst-case**. Measured savings come from
`nr86 bench --data --every-n 2 --dirty-tiles` (`tiles_executed`, `mean_ms`).

## Temporal reuse

`nr86/reproject.py` warps the previous color with mvec, then dirties pixels
where `|luma(current) − luma(warped prev)|` is large. Comparing to the
*unwarped* previous frame marks most of the screen dirty on a camera pan
even when reprojection worked.

`nr86/runtime.py` is the compute-saving path:

- skip-frame: `frame_index % every_n != 0` → warp previous output, **0 tiles**
- dirty-tile: on student frames, run only tiles whose mask fill ≥ threshold

Eval and `bench --data` both go through that path.

## INT8 / GroupNorm

`ConvGN` is fine for FP16. GroupNorm quantizes poorly and tends to break
TensorRT QDQ fusion. Preset `ampere_int8` is the same width with `norm=none`.
`nr86 calibrate` still writes min/max JSON only. The TensorRT builder does
not consume it. No QDQ graph, no INT4, no 2:4 sparsity.

## RTXNS track

Postponed. A 6→32→32→3 CoopVec MLP is neural shading, not a 148M post
filter. Keep it that way.

## What we deliberately do not reconstruct

NVIDIA’s NR resource blob is out of scope. The student graph is
`nr86/models/student.py`.
