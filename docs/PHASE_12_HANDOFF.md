# Phase 12 Handoff

## Outcome

Phase 12 is complete. The exact PP512 Q6_K SwiGLU-to-MMQ-D4 `ffn_down`
candidate was implemented and **rejected** by its immutable exact-chain timing
gate. It is not registered.

The final candidate passed provenance, both-device dispatch, 18/18 CPU-oracle
rows, fallback, normal graph replay, tensor-parallel registered dispatch,
resource recording, and whole PP. The exact chain was slower on both devices:
`0.995165x` on GPU 0 and `0.992729x` on GPU 1, below the required `1.010x`.

Whole PP512 measured `815.209 -> 819.466 tok/s`, or `1.005222x`, with stable
0.272%/0.376% spreads. The contract deliberately requires both operation and
whole-workload gains, so this does not rescue the candidate.

## Durable Findings

- Registered Qwen PP selected 128 fused producer calls on each R9700 and
  removed the same number of ordinary quantizers.
- Real tensor-parallel allocator reuse requires memory checks at the actual
  execution boundary; a synthetic whole-graph test did not reveal this.
- The fused producer has no LDS, private segment, scratch, or spills. Its low
  resource use did not translate into an exact-chain gain.
- gfx1201 `OccupancyPercent,FETCH_SIZE` attempts failed inside
  ROCprofiler-SDK on both devices and remain explicitly unavailable.
- Demote the same one-wave SwiGLU-to-D4 producer in prior-result ranking.

The complete failed-experiment account is in
`evidence/phase-12/Q6K_SWIGLU_D4_FFN_DOWN_FINDING.md`.

## Next Boundary

Phase 13 should begin as feasibility only. Use the registered normal-graph
attribution and direct-P2P implementation to identify one material row-sharded
MMQ whose partial output is followed by an all-reduce. Prove tile independence,
current synchronization/ownership, and overlap headroom before considering an
Iris-style GEMM/all-reduce producer-consumer design.

Do not port Iris code, mutate the P2P path, or benchmark an overlap candidate
until an exact operation, graph boundary, both-device timeline, and rollback-
safe contract are frozen.
