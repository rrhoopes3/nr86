# Shader-stage track

`student_mlp.slang` is the wiring for a 6→32→32→3 residual MLP using
NVIDIA RTX Neural Shading (`third_party/RTXNS`) and
`VK_NV_cooperative_vector`.

Build the RTXNS samples first (Vulkan path, driver ≥ 572.16). Then point
their `InferenceMLP` at weights exported from a tiny student you trained
here — not at a dumped NR blob.

This track is the one that is legal, specified, and does not depend on
fall DLSS 5 shipping for Ampere.
