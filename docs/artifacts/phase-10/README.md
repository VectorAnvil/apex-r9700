# Phase 10: Shared Q8_1 Gate/Up Experiment

This compact bundle records the default-off gfx1201 Q6_K gate/up experiment.
It quantized the common F32 activation once and reused the resulting Q8_1
scratch buffer for the existing gate and up MMQ launches, while retaining the
normal F32 SwiGLU node.

The backend and dispatch gates passed: both devices observed one Q8_1
quantization with two Q6_K MMQ launches, and the `m=256` shape used the normal
two-quantizer fallback. The timing and whole-PP gates rejected the candidate,
so it was not registered as an optimization.

This archive intentionally excludes build trees, binaries, raw rocprofv3
traces, HSACO/code objects, and complete capability dumps. The JSON summaries
retain their source paths and hashes for those materials.

See [ARTIFACT_MANIFEST.md](ARTIFACT_MANIFEST.md) for file hashes.
