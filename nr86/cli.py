from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nr86.config import PRESETS, Placement
from nr86.hw import doctor
from nr86.legal import LegalBlock
from nr86.models.student import build_student, count_params
from nr86.placement import summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nr86", description="Ampere sm_86 neural-rendering engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="GPU / SDK / FP8-vs-INT8 report")

    p_s = sub.add_parser("synth", help="HQ synth then self-teach down to --size")
    p_s.add_argument("--out", type=Path, required=True)
    p_s.add_argument("--frames", type=int, default=24)
    p_s.add_argument("--size", type=int, default=512)
    p_s.add_argument("--hq-scale", type=int, default=2)
    p_s.add_argument("--seed", type=int, default=0)

    p_st = sub.add_parser("selfteach", help="HQ dataset -> Quality-input teacher pairs")
    p_st.add_argument("--data", type=Path, required=True)
    p_st.add_argument("--out", type=Path, required=True)
    p_st.add_argument("--size", default="1280x720")

    p_t = sub.add_parser("train", help="Distill student on tiles")
    p_t.add_argument("--data", type=Path, required=True)
    p_t.add_argument("--out", type=Path, required=True)
    p_t.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    p_t.add_argument("--steps", type=int, default=40)
    p_t.add_argument("--batch", type=int, default=4)
    p_t.add_argument("--lr", type=float, default=2e-4)
    p_t.add_argument("--resume", type=Path, default=None)
    p_t.add_argument("--skip-eval", action="store_true")
    p_t.add_argument("--data-frames", type=int, default=None, help="Train on the first N frames only")
    p_t.add_argument("--extra", type=Path, default=None, help="Second taught dump mixed into the tile pool")
    p_t.add_argument("--extra-frames", type=int, default=None)
    p_t.add_argument("--hud-mask", choices=["none", "dxhr"], default="none")

    p_ev = sub.add_parser("eval", help="PSNR/SSIM vs teacher and vs identity")
    p_ev.add_argument("--ckpt", type=Path, required=True)
    p_ev.add_argument("--data", type=Path, required=True)
    p_ev.add_argument("--max-frames", type=int, default=32)
    p_ev.add_argument("--every-n", type=int, default=1)
    p_ev.add_argument("--dirty-tiles", action="store_true")
    p_ev.add_argument(
        "--ablate",
        choices=["none", "rgb", "depth", "mvec"],
        default="none",
        help="Zero input channels: rgb keeps color only; depth/mvec drop that cue",
    )
    p_ev.add_argument("--use-trt", action="store_true")
    p_ev.add_argument("--engine", type=Path, default=None)
    p_ev.add_argument("--offset", type=int, default=0, help="Skip this many frames (hold-out start)")
    p_ev.add_argument("--int8", action="store_true", help="Build/use a QDQ INT8 TensorRT engine")
    p_ev.add_argument(
        "--no-storm",
        action="store_true",
        help="Disable storm mode (bare full-frame after sustained dirty fill)",
    )

    p_e = sub.add_parser("export", help="Export student ONNX")
    p_e.add_argument("--ckpt", type=Path, required=True)
    p_e.add_argument("--onnx", type=Path, required=True)
    p_e.add_argument("--height", type=int, default=256)
    p_e.add_argument("--width", type=int, default=256)
    p_e.add_argument("--int8", action="store_true", help="Insert QDQ from calibration ranges")
    p_e.add_argument("--data", type=Path, default=None, help="Dump used to calibrate INT8 scales")

    p_q = sub.add_parser("calibrate", help="PTQ min/max on our student")
    p_q.add_argument("--ckpt", type=Path, required=True)
    p_q.add_argument("--data", type=Path, required=True)
    p_q.add_argument("--out", type=Path, required=True)

    p_b = sub.add_parser("bench", help="CUDA-event tile + full-frame bench")
    p_b.add_argument("--ckpt", type=Path, required=True)
    p_b.add_argument("--size", default="1280x720")
    p_b.add_argument("--scaling", type=float, default=0.67)
    p_b.add_argument("--every-n", type=int, default=1)
    p_b.add_argument("--iters", type=int, default=50)
    p_b.add_argument("--warmup", type=int, default=10)
    p_b.add_argument("--data", type=Path, default=None, help="Sequence: measure skip / dirty tiles")
    p_b.add_argument("--dirty-tiles", action="store_true")
    p_b.add_argument("--max-frames", type=int, default=32)
    p_b.add_argument("--try-trt", action="store_true", help="Report TensorRT-RTX FP16 availability")
    p_b.add_argument("--use-trt", action="store_true", help="Run the student as a TensorRT-RTX engine")
    p_b.add_argument("--engine", type=Path, default=None)
    p_b.add_argument("--int8", action="store_true", help="Build/use a QDQ INT8 TensorRT engine")
    p_b.add_argument(
        "--no-storm",
        action="store_true",
        help="Disable storm mode (bare full-frame after sustained dirty fill)",
    )

    p_p = sub.add_parser("place", help="Pixel-ops cost (average and worst-case)")
    p_p.add_argument("--preset", choices=sorted(PRESETS), default="ampere")
    p_p.add_argument("--size", default="1920x1080")
    p_p.add_argument("--scaling", type=float, default=0.67)
    p_p.add_argument("--every-n", type=int, default=2)
    p_p.add_argument("--mask-fill", type=float, default=0.35)

    p_v = sub.add_parser("preview", help="Contact sheet of color/teacher/student")
    p_v.add_argument("--data", type=Path, required=True)
    p_v.add_argument("--out", type=Path, required=True)
    p_v.add_argument("--ckpt", type=Path, default=None)

    p_i = sub.add_parser("ingest", help="Ingest ReShade dumps (not NVIDIA blobs)")
    p_i.add_argument("--src", type=Path, required=True)
    p_i.add_argument("--out", type=Path, required=True)
    p_i.add_argument("--placeholder", action="store_true", help="Use the fake enhancer (not the quality target)")

    p_ins = sub.add_parser("inspect", help="Validate a raw capture folder")
    p_ins.add_argument("--src", type=Path, required=True)

    p_fd = sub.add_parser("from-dump", help="Inspect + ingest + selfteach + eval a capture dump")
    p_fd.add_argument("--src", type=Path, required=True)
    p_fd.add_argument("--ckpt", type=Path, required=True)
    p_fd.add_argument("--raw", type=Path, default=Path("datasets/raw"))
    p_fd.add_argument("--taught", type=Path, default=Path("datasets/q720"))
    p_fd.add_argument("--size", default="1280x720")
    p_fd.add_argument("--every-n", type=int, default=2)
    p_fd.add_argument("--dirty-tiles", action="store_true", default=True)
    p_fd.add_argument("--use-trt", action="store_true")
    p_fd.add_argument("--eval-offset", type=int, default=0, help="Hold-out: eval starts at this frame")

    p_tr = sub.add_parser("trt", help="Build TensorRT-RTX engine from ONNX")
    p_tr.add_argument("--onnx", type=Path, required=True)
    p_tr.add_argument("--engine", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except LegalBlock as exc:
        print(exc, file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "doctor":
        report = doctor()
        print(json.dumps(report, indent=2))
        print("\npresets:")
        for name, spec in PRESETS.items():
            n = count_params(build_student(spec))
            print(
                f"  {name:12s}  base={spec.base:3d}  levels={spec.levels}  "
                f"tile={spec.tile:3d}  norm={spec.norm:4s}  params={n:,}"
            )
        return 0
    if args.cmd == "synth":
        from nr86.synth import write_synth

        n = write_synth(args.out, args.frames, args.size, args.seed, args.hq_scale)
        print(f"wrote {n} frames -> {args.out}")
        return 0
    if args.cmd == "selfteach":
        from nr86.selfteach import selfteach_dataset

        selfteach_dataset(args.data, args.out, args.size)
        return 0
    if args.cmd == "train":
        from nr86.train import train

        train(
            args.data,
            args.out,
            args.preset,
            args.steps,
            args.batch,
            args.lr,
            resume=args.resume,
            skip_eval=args.skip_eval,
            data_frames=args.data_frames,
            extra=args.extra,
            extra_frames=args.extra_frames,
            hud_mask=args.hud_mask,
        )
        return 0
    if args.cmd == "eval":
        from nr86.eval import evaluate

        evaluate(
            args.ckpt,
            args.data,
            args.max_frames,
            args.every_n,
            dirty_tiles=args.dirty_tiles,
            ablate=args.ablate,
            use_trt=args.use_trt,
            engine=args.engine,
            offset=args.offset,
            int8=args.int8,
            storm=not args.no_storm,
        )
        return 0
    if args.cmd == "export":
        from nr86.export_onnx import export_onnx

        export_onnx(
            args.ckpt,
            args.onnx,
            args.height,
            args.width,
            int8=args.int8,
            calib_data=args.data,
        )
        return 0
    if args.cmd == "calibrate":
        from nr86.quantize import calibrate

        calibrate(args.ckpt, args.data, args.out)
        return 0
    if args.cmd == "bench":
        from nr86.bench import bench_ckpt

        bench_ckpt(
            args.ckpt,
            args.size,
            warmup=args.warmup,
            iters=args.iters,
            scaling_ratio=args.scaling,
            every_n=args.every_n,
            data=args.data,
            dirty_tiles=args.dirty_tiles,
            max_frames=args.max_frames,
            try_trt=args.try_trt,
            use_trt=args.use_trt,
            engine=args.engine,
            int8=args.int8,
            storm=not args.no_storm,
        )
        return 0
    if args.cmd == "place":
        w, h = args.size.lower().split("x")
        spec = PRESETS[args.preset]
        p = Placement(
            scaling_ratio=args.scaling,
            every_n=args.every_n,
            mask_fill=args.mask_fill,
            tile=spec.tile,
            overlap=spec.overlap,
            output_w=int(w),
            output_h=int(h),
        )
        print(json.dumps(summarize(p), indent=2))
        return 0
    if args.cmd == "preview":
        from nr86.preview import contact_sheet

        contact_sheet(args.data, args.ckpt, args.out)
        return 0
    if args.cmd == "ingest":
        from nr86.ingest import ingest

        ingest(args.src, args.out, placeholder=args.placeholder)
        return 0
    if args.cmd == "inspect":
        from nr86.inspect_capture import inspect_capture

        print(json.dumps(inspect_capture(args.src), indent=2))
        return 0
    if args.cmd == "from-dump":
        from nr86.from_dump import from_dump

        from_dump(
            args.src,
            args.raw,
            args.taught,
            args.ckpt,
            size=args.size,
            every_n=args.every_n,
            dirty_tiles=args.dirty_tiles,
            use_trt=args.use_trt,
            eval_offset=args.eval_offset,
        )
        return 0
    if args.cmd == "trt":
        from nr86.engine_trt import build_engine

        build_engine(args.onnx, args.engine)
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
