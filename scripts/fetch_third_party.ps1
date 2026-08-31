# Fetch optional full ReShade tree (headers are already vendored in reshade-sdk).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

git submodule update --init --depth 1

$reshade = Join-Path $root "third_party\reshade"
if (-not (Test-Path (Join-Path $reshade "include\reshade.hpp"))) {
    if (Test-Path $reshade) { Remove-Item -Recurse -Force $reshade }
    git clone --depth 1 --filter=blob:none --sparse https://github.com/crosire/reshade.git third_party/reshade
    Set-Location $reshade
    git sparse-checkout set include examples/09-depth examples/07-texture_dump
    Set-Location $root
}

$sdkInc = Join-Path $root "third_party\reshade-sdk\include"
New-Item -ItemType Directory -Force -Path $sdkInc | Out-Null
Copy-Item (Join-Path $reshade "include\*.hpp") $sdkInc -Force
Copy-Item (Join-Path $reshade "LICENSE.md") (Join-Path $root "third_party\reshade-sdk\LICENSE.md") -Force
Write-Host "third_party ready."
