"""Motion vectors and temporal reuse.

Games do not expose mvec through ReShade. Until a Streamline hook exists
(LEGAL.md marks that future), consecutive color frames → Farneback flow.

Dirty pixels are residuals *after* warp. Raw motion magnitude dirties the
whole screen on a camera pan even when reprojection worked.
"""

from __future__ import annotations

import numpy as np


def estimate_flow(prev: np.ndarray, cur: np.ndarray) -> tuple[np.ndarray, str]:
    """Normalized flow (dx/W, dy/H) from prev → cur. Source tag for the manifest."""
    h, w = cur.shape[:2]
    try:
        import cv2
    except ImportError:
        return np.zeros((h, w, 2), dtype=np.float32), "zero_no_cv2"
    a = (np.clip(prev, 0, 1).mean(axis=2) * 255.0).astype(np.uint8)
    b = (np.clip(cur, 0, 1).mean(axis=2) * 255.0).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = flow.astype(np.float32)
    flow[..., 0] /= float(w)
    flow[..., 1] /= float(h)
    return flow, "farneback"


def warp_rgb(image: np.ndarray, mvec: np.ndarray) -> np.ndarray:
    """Backward-warp HWC RGB. dest (x,y) samples src (x - dx*W, y - dy*H)."""
    h, w = image.shape[:2]
    try:
        import cv2
    except ImportError:
        return _warp_numpy(image, mvec)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    map_x = (grid_x - mvec[..., 0] * w).astype(np.float32)
    map_y = (grid_y - mvec[..., 1] * h).astype(np.float32)
    return cv2.remap(
        np.clip(image, 0, 1).astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _warp_numpy(image: np.ndarray, mvec: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    ys, xs = np.indices((h, w))
    src_x = np.clip(xs - mvec[..., 0] * w, 0, w - 1)
    src_y = np.clip(ys - mvec[..., 1] * h, 0, h - 1)
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (src_x - x0)[..., None]
    wy = (src_y - y0)[..., None]
    ia = image[y0, x0]
    ib = image[y0, x1]
    ic = image[y1, x0]
    id_ = image[y1, x1]
    return (ia * (1 - wx) * (1 - wy) + ib * wx * (1 - wy) + ic * (1 - wx) * wy + id_ * wx * wy).astype(
        np.float32
    )


def residual_mask(
    color: np.ndarray,
    warped_prev: np.ndarray | None,
    luma_eps: float = 0.02,
) -> np.ndarray:
    """True where warp failed (run the student). Compare to *warped* previous."""
    if warped_prev is None:
        return np.ones(color.shape[:2], dtype=bool)
    luma = color @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    prev = warped_prev @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.abs(luma - prev) >= luma_eps


# Back-compat name used by older tests / call sites.
def motion_luma_mask(
    color: np.ndarray,
    warped_prev: np.ndarray | None,
    mvec: np.ndarray | None = None,
    motion_norm: float = 0.004,
    luma_eps: float = 0.02,
) -> np.ndarray:
    del mvec, motion_norm
    return residual_mask(color, warped_prev, luma_eps=luma_eps)


def composite(
    student_rgb: np.ndarray,
    warped_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    m = mask.astype(np.float32)[..., None]
    return student_rgb * m + warped_rgb * (1.0 - m)


def fill_ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask.astype(np.float32)))


def warp_nchw(image: "torch.Tensor", mvec: "torch.Tensor") -> "torch.Tensor":
    """Backward-warp NCHW RGB. mvec is N2HW, dx/W and dy/H."""
    import torch
    import torch.nn.functional as F

    _n, _c, h, w = image.shape
    ys = torch.linspace(-1.0, 1.0, h, device=image.device, dtype=image.dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=image.device, dtype=image.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    off_x = mvec[:, 0] * (2.0 * w / max(w - 1, 1))
    off_y = mvec[:, 1] * (2.0 * h / max(h - 1, 1))
    grid = torch.stack([grid_x - off_x, grid_y - off_y], dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


def residual_mask_nchw(color: "torch.Tensor", warped_prev: "torch.Tensor | None", luma_eps: float = 0.02) -> "torch.Tensor":
    import torch

    if warped_prev is None:
        n, _c, h, w = color.shape
        return torch.ones(n, 1, h, w, device=color.device, dtype=torch.bool)
    wts = color.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    luma = (color * wts).sum(dim=1, keepdim=True)
    prev = (warped_prev * wts).sum(dim=1, keepdim=True)
    return (luma - prev).abs() >= luma_eps


def composite_nchw(student: "torch.Tensor", warped: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    m = mask.to(dtype=student.dtype)
    if m.ndim == 3:
        m = m.unsqueeze(1)
    if m.shape[1] != 1:
        m = m[:, :1]
    return student * m + warped * (1.0 - m)
