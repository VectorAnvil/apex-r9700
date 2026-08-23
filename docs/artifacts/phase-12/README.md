# Phase 12: Producer-to-D4-Q8_1 Feasibility And Evaluation

This compact bundle preserves both Phase 12 steps:

1. Source/trace feasibility selected the exact PP512 Q6_K
   SwiGLU-to-MMQ-D4 `ffn_down` boundary.
2. A separately frozen, default-off candidate was implemented, validated, and
   rejected because its exact chain regressed on both gfx1201 devices.

The candidate passed dispatch, 18/18 correctness rows, fallback, graph replay,
tensor-parallel selection, resource recording, and the `1.005222x` whole-PP
gate. Exact-chain results were `0.995165x` and `0.992729x`, below the required
`1.010x`; no registration or performance claim was made.

Raw worktrees, builds, traces, logs, HSACO, profiler failures, and intermediate
attempts remain in the ignored Phase 12 result roots. The tracked bundle keeps
the immutable tasks, compact gate records, final decision, source patch, and
hashes needed to audit the result.
