# Measurement protocol

Do not quote FPS from a Discord screenshot. Record GPU time **and**
PSNR/SSIM vs the self-teacher. Latency without a quality target is
optimizing an undefined function.

## Doctor first

```powershell
python -m nr86 doctor
```

Expected on this box: `NVIDIA GeForce RTX 3090`, compute `8.6`, 24576 MiB,
`fp8=no`, `int8=yes`, `sparsity_2_4=yes`. TensorRT-RTX on PATH is still
the real engine; every current ms number is a PyTorch proxy until then.

## Always log these columns

| Field | Example |
| --- | --- |
| preset | smoke / ampere / ampere_int8 / target |
| input_wh | Quality-input tensor size |
| tile / fullframe | both, in TRT especially |
| scaling_ratio / every_n / mask_fill | average **and** worst-case |
| precision | fp16 / int8 |
| mean_ms, fullframe_mean_ms | CUDA events |
| identity_psnr, student_psnr, delta_psnr | gate 4 |
| beats_identity | must be true |

## Go / no-go gates

Run in order. Stop at the first fail.

1. **Smoke** — `synth` (2× HQ → self-teach) + `train smoke` + `bench`.
   Gate: PyTorch `fullframe_mean_ms` at Quality-input 720p is a few ms
   on a 3090. Tiled eager PyTorch is launch-bound; ignore it. TRT is the
   number that matters once the SDK is installed.
2. **Placement** — `python -m nr86 place --preset ampere --size 1920x1080`.
   Gate: average cheapness ≥ 4× vs leak full-frame. **Also** read
   `worst_case`: it must be ~2.2× (scaling only). Budget that row.
3. **Quality (required)** — `python -m nr86 eval --ckpt … --data …`.
   Gate: `student_psnr >= identity_psnr + 0.25` and `beats_identity=true`.
   Use the self-teacher, not a placeholder enhancer. If this fails, shrink
   the problem (input res) before growing the student.
4. **Capture validation** — `python -m nr86 inspect --src <game>/nr86_capture`
   on the first real title, before building a dataset. Gate: `ok=true`,
   `color_format` is a known layout, depth byte count matches
   `depth_width*depth_height*4`, and burst frames have `prev_color`.
5. **Ampere student + INT8** — only after 3 and 4. Gate: 720p TRT
   full-frame *and* tiled `mean_ms`. Then INT8 on `ampere_int8` (no GN).
   INT8 ≤ 0.65× fp16 ms or stop chasing precision and cut pixels.

## What not to compare against

30–35 FPS Hogwarts 1080p DLAA+NR on a 3090 is the **ceiling of the FP16-cast
leak**, not the floor of this engine. DRG 138→4 is the leak at the wrong
placement, usually on a camera swing — that is the worst-case row.
