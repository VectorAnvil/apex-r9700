# Required Findings

Each finding is an object with `status` (`supported` or `unsupported`) and a
nonempty evidence list. Required keys are graph adjacency, shared input/layouts,
tensor-parallel ownership, numerical SwiGLU contract, existing fusion API fit,
implementation boundary, default-off rollback selector, and resource,
correctness, dispatch, timing, and whole-PP plans. Phase 9 creates no candidate
build regardless of outcome.
