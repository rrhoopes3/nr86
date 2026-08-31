# third_party

| Path | Upstream | Why it is here |
| --- | --- | --- |
| `TensorRT-RTX/` | https://github.com/NVIDIA/TensorRT-RTX | Official AOT/JIT samples. Our builder follows `samples/helloWorld/python/hello_world.py`. |
| `RTXNS/` | https://github.com/NVIDIA-RTX/RTXNS | Official Neural Shading SDK. CoopVec MLP pattern used by `shaders/student_mlp.slang`. |
| `reshade-sdk/` | headers copied from https://github.com/crosire/reshade `include/` | Header-only ReShade add-on API for `addons/nr86_capture`. |
| `reshade/` | sparse clone (gitignored) | Full examples (`09-depth`) for local reference. |

Do not add leaked NR runtimes, Discord Ampere addons, or OptiScaler here.

Refresh:

```powershell
git submodule update --init --depth 1
powershell -File scripts/fetch_third_party.ps1
```
