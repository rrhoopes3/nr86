from __future__ import annotations

import numpy as np


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """pred/target float HWC or CHW in [0,1]."""
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean SSIM on luma. Small, dependency-free, good enough for gates."""
    a = _luma(pred)
    b = _luma(target)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = _box(a, 7)
    mu_b = _box(b, 7)
    sig_a = _box(a * a, 7) - mu_a * mu_a
    sig_b = _box(b * b, 7) - mu_b * mu_b
    sig_ab = _box(a * b, 7) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def _luma(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float64)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.moveaxis(x, 0, -1)
    if x.ndim == 3:
        x = x[..., :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    return x


def _box(img: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    p = np.pad(img, pad, mode="edge")
    cs = np.pad(p, ((1, 0), (1, 0)))
    cs = cs.cumsum(0).cumsum(1)
    h, w = img.shape
    window = cs[k : h + k, k : w + k] - cs[0:h, k : w + k] - cs[k : h + k, 0:w] + cs[0:h, 0:w]
    return window / float(k * k)
