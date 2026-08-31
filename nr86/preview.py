from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from nr86.dataset import load_frame, load_manifest
from nr86.models.student import load_student
from nr86.dataset import pack_input
import torch


def contact_sheet(data: Path, ckpt: Path | None, out: Path, n: int = 6) -> Path:
    rows = load_manifest(data)[:n]
    images = []
    model = None
    device = torch.device("cpu")
    if ckpt:
        model = load_student(ckpt, map_location="cpu")
        model.eval()
    for rec in rows:
        frame = load_frame(data, rec)
        cols = [frame.color]
        if frame.teacher is not None:
            cols.append(frame.teacher)
        if model is not None:
            x = torch.from_numpy(pack_input(frame)).unsqueeze(0)
            with torch.no_grad():
                pred = model(x)[0].permute(1, 2, 0).cpu().numpy()
            cols.append(np.clip(pred, 0, 1))
        images.append(np.concatenate(cols, axis=1))
    sheet = np.concatenate(images, axis=0)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(sheet * 255, 0, 255).astype(np.uint8)).save(out)
    print(f"wrote {out}")
    return out
