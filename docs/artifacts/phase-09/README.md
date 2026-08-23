# Phase 9: Gate/Up Fusion Feasibility

This result root proves a source-attributed PP512 gate/up opportunity without
implementing it. The Phase 7 recorder establishes 64 exact adjacent gate/up
pairs per device and capture state, six trace repeats, shared F32 input, Q6_K
weights, Q8_1 MMQ layout, and a 38.304%-class combined PP share.

The feasible future design is a default-off, exact-shape-gated paired Q6_K MMQ
plus SwiGLU dispatcher with baseline fallback. It must preserve pinned
`ggml_swiglu_split` F32 numerical semantics; SuperSonic BF16 rounding is not
adopted. The current result contains plans and provenance only, not a candidate
or performance claim.

See [ARTIFACT_MANIFEST.md](ARTIFACT_MANIFEST.md) and
[evidence/PROCEDURE.md](evidence/PROCEDURE.md).

The scripts under `tools/` are provenance-preserving archival copies. Their
relative paths target the original result-local root; execute the originals
from `results_phase09_gate_up_fusion_feasibility_20260808/`, not these copies.
