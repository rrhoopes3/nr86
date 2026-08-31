# Legal line

This repo is an Ampere (`sm_86`) neural-rendering **engine**: capture of
color / depth / motion vectors, a student network you own, INT8 TensorRT-RTX
placement, and a cooperative-vector shader track.

It is **not** a DLSS 5 installer, not a PTX swapper, and not a weight dumper.

## What this repo will not do

nr86 will refuse to open, parse, patch, or redistribute:

- `nvngx_dlssnr.dll` (leaked Neural Rendering runtime)
- `dlssnr_ampere.addon64` and `dlssnr-sm86-patched` fatbins / PTX
- any tool whose job is “hash-match the DLSSNR cubins and swap PTX”
- NVIDIA’s 148M FP8 teacher weights, even quantized or distilled *as those weights*

Private tinkering with a leak on your own machine is what community addons
already do. A derived INT8 engine you redistribute, or a product that ships
NVIDIA’s weights, is a different conversation. Distilling a student you own,
or training a neural shader that never touches `nvngx_dlssnr.dll`, is the
version this repo is for.

## What you *can* feed it

| Allowed | Not allowed |
| --- | --- |
| Color / depth dumps from *your* game via the ReShade capture addon | Dumping tensors out of NVIDIA resource blobs |
| **Self-teacher:** same scene captured at high res, Lanczos-downsampled to Quality-input. Zero NVIDIA bits. This is the quality target. | The leak as a teacher (`.dll`, cubins, PTX, 153 tensors) |
| Farneback mvec from consecutive color frames (and later official Streamline mvec) | Discord Ampere PTX swaps |
| Official NVIDIA SDKs: RTXNS, TensorRT-RTX, Streamline, NGX headers | Leaked NBA 2K27 / Discord “NR addon” packages |
| Official `nvngx_dlss.dll` from a game you own, used only as a motion-vector / jitter source via public Streamline APIs (future) | Multiplayer / anti-cheat titles |

## Two tracks

1. **TensorRT-RTX student (this is the 3090 performance bet).** Train a 5–40M
   INT8 network on 224–512 tiles, run at Quality-input size, mask, every-Nth
   frame. Weights are yours.
2. **RTX Neural Shading / `VK_NV_cooperative_vector`.** Specified down to RTX 20.
   Lives in the shader stages. This is the path you can talk about in a filing.

## Anti-cheat

Do not inject ReShade, DXGI proxies, or this capture addon into anything with
anti-cheat. Offline single-player only.
