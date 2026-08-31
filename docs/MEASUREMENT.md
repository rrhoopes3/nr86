# Measurement protocol

Do not quote FPS from a Discord screenshot. Record GPU time **and**
PSNR/SSIM vs the self-teacher. Latency without a quality target is
optimizing an undefined function.

This repo is a research scaffold. Placement ~13× is a cost model.
PyTorch eager-tile benches execute every tile. The number that can
be defended is: student ΔPSNR vs identity, plus `tiles_executed`,
per-path `mean_ms` / `p95_ms` from `FrameRunner`, and the two-sided
latency gate on the **1280×720** product tensor.

## Doctor first

```powershell
python -m nr86 doctor
```

Expected on this box: `NVIDIA GeForce RTX 3090`, compute `8.6`, 24576 MiB,
`fp8=no`, `int8=yes`, `sparsity_2_4=yes`. TensorRT-RTX is installed;
`eval --use-trt` / `bench --use-trt` run the student as an engine.
Synth skip+dirty 512² was 6.48 ms on this 3090 — still not a capture number.

## Always log these columns

| Field | Example |
| --- | --- |
| preset | smoke / ampere / ampere_int8 / target |
| input_wh | Quality-input tensor size |
| tile / fullframe | both, in TRT especially |
| scaling_ratio / every_n / mask_fill | average **and** worst-case (model) |
| tiles_executed / tiles_total | from `eval` / `bench --data` |
| precision | fp16 (int8 not shipped) |
| mean_ms / p95_ms / path_ms | CUDA events per frame; warp_clean vs fullframe_dirty |
| identity_psnr, student_psnr, delta_psnr | gate 4 |
| ablate | none / rgb / depth / mvec |
| beats_identity | must be true |

## Go / no-go gates

Run in order. Stop at the first fail.

1. **Smoke** — `synth` (2× HQ → self-teach) + `train smoke` + `bench`.
   Gate: identity pass on synth. Tiled eager PyTorch is launch-bound;
   ignore it. `--size` is **output** resolution (internal = output × 0.67).
   Do not treat 858×482 eager ms as a 720p budget.
2. **Placement** — `python -m nr86 place --preset ampere --size 1920x1080`.
   Gate: average cheapness ≥ 4× vs leak full-frame **as a model**.
   **Also** read `worst_case`: it must be ~2.2× (scaling only). Budget
   that row. Do not quote 13× as measured.
3. **Quality (required)** — `python -m nr86 eval --ckpt … --data …`.
   Gate: `student_psnr >= identity_psnr + 0.25` and `beats_identity=true`.
   Use the self-teacher, not a placeholder enhancer. If this fails, shrink
   the problem (input res) before growing the student.
4. **Capture → ingest** — `nr86 from-dump --src <nr86_capture> --ckpt …`
   (inspect + ingest + selfteach + eval). First frame `"prev_color": null`
   is valid. Gate: ingest does not crash; later burst frames have Farneback
   or file mvec.
5. **Measured skip / dirty tiles** —
   `nr86 bench --data <set> --every-n 2 --dirty-tiles`.
   Product tensor is 1280×720 (1080p Quality-input). Gate (both):
   skip+dirty **mean** ≤ 8.33 ms **and** student-path **p95**
   (`fullframe` + `fullframe_dirty`) ≤ 16.67 ms. A blended mean that
   hides 11 ms dirty spikes is not a pass. Then
   `eval --every-n 2 --dirty-tiles --ablate …`.
6. **Ampere student + INT8** — only after 3–5 **and** the 720p
   latency gate. If FP16 student-only (~11.4 ms) still busts the mean
   budget, cut pixels (smaller internal) then INT8, then a shallower
   net. Width growth stays last. INT4 / 2:4 stay postponed.

## What not to compare against

30–35 FPS Hogwarts 1080p DLAA+NR on a 3090 is the **ceiling of the FP16-cast
leak**, not the floor of this engine. DRG 138→4 is the leak at the wrong
placement, usually on a camera swing — that is the worst-case row.
