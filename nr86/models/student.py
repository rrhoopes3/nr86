from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr86.config import INPUT_CHANNELS, OUTPUT_CHANNELS, PRESETS, StudentSpec


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, norm: str) -> None:
        super().__init__()
        use_bias = norm == "none"
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=use_bias)
        self.norm: nn.Module | None
        if norm == "gn":
            groups = 8 if c_out >= 8 else 1
            while c_out % groups != 0:
                groups -= 1
            self.norm = nn.GroupNorm(groups, c_out)
        elif norm == "none":
            self.norm = None
        else:
            raise ValueError(f"unknown norm {norm!r}, expected gn|none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        if self.norm is not None:
            h = self.norm(h)
        return F.relu(h, inplace=True)


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int, norm: str) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.block = nn.Sequential(ConvBlock(c_in, c_out, norm), ConvBlock(c_out, c_out, norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, c_in: int, skip_c: int, c_out: int, norm: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(c_in + skip_c, c_out, norm),
            ConvBlock(c_out, c_out, norm),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ResidualUNet(nn.Module):
    """Student: packed 6-ch input → residual RGB. ONNX-friendly ops only."""

    def __init__(self, spec: StudentSpec) -> None:
        super().__init__()
        self.spec = spec
        chs = spec.channels
        n = spec.norm
        self.inc = nn.Sequential(ConvBlock(INPUT_CHANNELS, chs[0], n), ConvBlock(chs[0], chs[0], n))
        self.enc = nn.ModuleList(Down(chs[i], chs[i + 1], n) for i in range(len(chs) - 1))
        self.bot = nn.Sequential(ConvBlock(chs[-1], chs[-1], n), ConvBlock(chs[-1], chs[-1], n))
        self.dec = nn.ModuleList(
            Up(chs[i + 1], chs[i], chs[i], n) for i in reversed(range(len(chs) - 1))
        )
        self.head = nn.Conv2d(chs[0], OUTPUT_CHANNELS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        color = x[:, :3]
        h = self.inc(x)
        skips = [h]
        for down in self.enc:
            h = down(h)
            skips.append(h)
        h = self.bot(h)
        for up, skip in zip(self.dec, reversed(skips[:-1])):
            h = up(h, skip)
        delta = torch.tanh(self.head(h))
        return (color + 0.5 * delta).clamp(0.0, 1.0)


def spec_from(preset: str | StudentSpec) -> StudentSpec:
    if isinstance(preset, StudentSpec):
        return preset
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}, expected {sorted(PRESETS)}")
    return PRESETS[preset]


def build_student(preset: str | StudentSpec) -> ResidualUNet:
    return ResidualUNet(spec_from(preset))


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def save_student(model: ResidualUNet, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "spec": model.spec,
            "state": model.state_dict(),
        },
        path,
    )


def load_student(path: Path, map_location: str | torch.device = "cpu") -> ResidualUNet:
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    spec = blob["spec"]
    if not getattr(spec, "norm", None):
        spec = StudentSpec(
            name=spec.name,
            base=spec.base,
            levels=spec.levels,
            tile=spec.tile,
            overlap=spec.overlap,
            norm="gn",
        )
    model = ResidualUNet(spec)
    model.load_state_dict(blob["state"])
    return model
