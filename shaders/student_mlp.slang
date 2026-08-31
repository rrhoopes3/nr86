// nr86 cooperative-vector student (shader-stage track)
//
// This is NOT NVIDIA's 148M FP8 teacher and it never loads nvngx_dlssnr.dll.
// Wire it to third_party/RTXNS InferenceMLP (NVIDIA RTX SDK license) when you
// build the RTXNS samples. Layout: 6 inputs (rgb, linearized depth, mvec)
// → residual rgb. Weights are yours.
//
// VK_NV_cooperative_vector is specified down to RTX 20.

module nr86;

struct StudentLayout
{
    static const int Inputs = 6;
    static const int Hidden = 32;
    static const int HiddenLayers = 2;
    static const int Outputs = 3;
}

// Host packs weights into one ByteAddressBuffer in RTXNS InferenceOptimal layout.
// Pixel shader / compute: CoopVec<half, 6> from color/depth/mvec → tanh residual
// added to the source color. See docs/ARCHITECTURE.md.
