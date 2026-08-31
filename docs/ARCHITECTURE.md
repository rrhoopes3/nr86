# Architecture

Two tracks share one capture contract. Neither track loads NVIDIA’s NR DLL.

```
 game (4K/DLAA, HUD off)  or  synth at 2x
      │
      ▼
 selfteach: box-downsample = cheap color
            Lanczos        = teacher
            Farneback      = mvec  (Streamline later)
      │
      ├──────────── TensorRT-RTX track ─────────────┐
      │  residual UNet (gn for FP16, none for INT8) │
      │  ScalingRatio · mask · every-N + warp       │
      │  eval must beat identity PSNR               │
      │                                                │
      └──────────── RTXNS / CoopVec track ────────────┘
         6→32→32→3 MLP in the shader stages
         a legally clean *different product*, not a
         small NR reconstructor
```

## Quality target (not optional)

The function is: **reconstruct the high-quality downsample from the cheap one**.
That is classic SR distillation. It needs no NVIDIA bits.

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
`color_prev.bmp`; ingest runs Farneback. Streamline mvec is still future.
Without mvec, every-N + mask silently become “scaling only” (~2.2×, not 13×).

## Placement: average vs worst case

```
avg   ≈ 0.67² × mask_fill / every_n     →  ~0.079  (~13× cheaper)
worst = 0.67² × 1.0 / 1                 →  ~0.45   (~2.2× cheaper)
```

Fast camera motion drives `mask_fill → 1` and breaks reprojection.
`python -m nr86 place` prints both rows. Size the frame-time budget for
**worst-case** or you get DRG-style cliffs when the frame rate matters.

`nr86/reproject.py` warps the previous student output with mvec and
composites through the motion/luma mask. That is what every-2nd-frame
actually *is*.

## INT8 / GroupNorm

`ConvGN` is fine for FP16. GroupNorm quantizes poorly and tends to break
TensorRT QDQ fusion. Preset `ampere_int8` is the same width with `norm=none`.
If INT8 calibration looks bad, blame the norm layers before the calibrator.

At 720p internal on 24 GB, a single static full-frame TRT engine may beat
batched 256² tiles. Bench both. The 62 ms eager-tile number is launch tax.

## RTXNS track

A 6→32→32→3 CoopVec MLP is neural shading, not a 148M post filter. The
filing story is real. The quality story is not comparable. Keep it that way.

## What we deliberately do not reconstruct

NVIDIA’s NR resource blob is out of scope. The student graph is
`nr86/models/student.py`.
