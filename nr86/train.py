from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nr86.dataset import FrameDataset
from nr86.models.student import build_student, count_params, load_student, save_student


class TileTorchDataset(Dataset):
    def __init__(
        self,
        root: Path,
        tile: int,
        epoch_tiles: int,
        seed: int = 0,
        offset: int = 0,
        max_frames: int | None = None,
        extra: Path | None = None,
        extra_frames: int | None = None,
        mix: list[Path] | None = None,
        hud: str = "none",
    ) -> None:
        from nr86.hud_mask import PRESETS as HUD_PRESETS

        self.pools = [
            FrameDataset(root, require_teacher=True, offset=offset, max_frames=max_frames)
        ]
        if extra is not None:
            self.pools.append(
                FrameDataset(extra, require_teacher=True, offset=0, max_frames=extra_frames)
            )
        for mix_p in mix or []:
            self.pools.append(FrameDataset(mix_p, require_teacher=True))
        weights = np.array([len(p) for p in self.pools], dtype=np.float64)
        self.pool_p = weights / weights.sum()
        self.tile = tile
        self.epoch_tiles = epoch_tiles
        self.rng = np.random.default_rng(seed)
        self.hud_boxes = HUD_PRESETS[hud]

    def __len__(self) -> int:
        return self.epoch_tiles

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from nr86.hud_mask import tile_keep

        pool = self.pools[int(self.rng.choice(len(self.pools), p=self.pool_p))]
        fi = int(self.rng.integers(0, len(pool)))
        x_np, y_np = pool[fi]
        _c, h, w = x_np.shape
        if h < self.tile or w < self.tile:
            pad_h = max(0, self.tile - h)
            pad_w = max(0, self.tile - w)
            x_np = np.pad(x_np, ((0, 0), (0, pad_h), (0, pad_w)))
            y_np = np.pad(y_np, ((0, 0), (0, pad_h), (0, pad_w)))
            h, w = x_np.shape[-2:]
        y0 = int(self.rng.integers(0, h - self.tile + 1))
        x0 = int(self.rng.integers(0, w - self.tile + 1))
        x = x_np[:, y0 : y0 + self.tile, x0 : x0 + self.tile]
        y = y_np[:, y0 : y0 + self.tile, x0 : x0 + self.tile]
        keep = tile_keep(h, w, y0, x0, self.tile, self.tile, self.hud_boxes)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(keep)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(
    data: Path,
    out: Path,
    preset: str = "smoke",
    steps: int = 40,
    batch: int = 4,
    lr: float = 2e-4,
    seed: int = 0,
    resume: Path | None = None,
    skip_eval: bool = False,
    data_offset: int = 0,
    data_frames: int | None = None,
    extra: Path | None = None,
    extra_frames: int | None = None,
    mix: list[Path] | None = None,
    hud_mask: str = "none",
) -> dict:
    from nr86.config import PRESETS

    spec = PRESETS[preset]
    device = pick_device()
    if resume is not None and Path(resume).exists():
        model = load_student(resume, map_location=device).to(device)
    else:
        model = build_student(spec).to(device)
    ds = TileTorchDataset(
        data,
        tile=spec.tile,
        epoch_tiles=max(steps * batch, spec.tile),
        seed=seed,
        offset=data_offset,
        max_frames=data_frames,
        extra=extra,
        extra_frames=extra_frames,
        mix=mix,
        hud=hud_mask,
    )
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    it = iter(loader)
    last_loss = 0.0
    for step in range(steps):
        try:
            x, y, keep = next(it)
        except StopIteration:
            it = iter(loader)
            x, y, keep = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        keep = keep.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            pred = model(x)
            err = torch.abs(pred - y) * keep.unsqueeze(1)
            loss = err.sum() / keep.sum().clamp(min=1.0) / pred.shape[1]
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        last_loss = float(loss.detach().cpu())
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            print(f"step {step + 1}/{steps}  l1={last_loss:.4f}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "student.pt"
    save_student(model.cpu(), ckpt)
    meta = {
        "preset": preset,
        "params": count_params(model),
        "steps": steps,
        "loss": last_loss,
        "device": str(device),
        "ckpt": str(ckpt),
        "tile": spec.tile,
    }
    (out / "train.json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"saved {ckpt}  params={meta['params']}")
    if not skip_eval:
        try:
            from nr86.eval import evaluate

            ev = evaluate(ckpt, data, max_frames=min(8, len(ds.pools[0])))
            meta["eval"] = ev
            (out / "train.json").write_text(
                __import__("json").dumps(meta, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            print(f"eval skipped: {exc}")
    return meta
