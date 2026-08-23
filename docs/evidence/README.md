# Optimization Findings

This directory contains human-readable optimization findings derived from the
immutable artifacts under `docs/apex-r9700/artifacts/`.

Every evaluated candidate gets a phase-scoped finding, including candidates
that are rejected, incorrect, neutral, unstable, or infeasible. Each finding
must identify the hypothesis, exact patch and workload, correctness result,
performance and variance evidence, decision, and what did not work. Missing
end-to-end results must be stated explicitly rather than inferred from kernel
measurements.

## Findings

- FA-S1: `fa-s1/FA1_STABILITY_ADJUDICATION.md` - accepted FA-1 after graph,
  cold-start, monitoring, and process-order adjudication plus a clean
  24-process long-depth rerun; the restored path remains unchanged.
- FA-1 serving promotion: `fa-s1/FA1_VIVI_SERVING_PROMOTION.md` - pins and
  validates the full-context Qwen3.8 server and records the healthy production
  endpoint, exact hashes, serving smoke, and rollback boundary.
- Phase 4: `phase-04/Q6K_FOUR_WAVE_FINDING.md` - rejected unconditional
  four-wave Q6_K MMV dispatch.
- Phase 5: `phase-05/Q6K_SHAPE_GATED_FOUR_WAVE_FINDING.md` - rejected exact
  shape-gated four-wave Q6_K MMV dispatch, including an incorrect v1 selector.
- Phase 6: `phase-06/Q6K_FUSED_TWELVE_WAVE_FINDING.md` - rejected fused-only
  twelve-wave Q6_K MMV candidate.
- Phase 7: `phase-07/Q6K_PP_MMQ_ATTRIBUTION_FINDING.md` - PP Q6_K MMQ
  attribution and resource finding.
- Phase 8: `phase-08/Q6K_PP_MMQ_Y64_FINDING.md` - statically rejected MMQ
  tile-height candidate.
- Phase 9: `phase-09/Q6K_PP_GATE_UP_FUSION_FEASIBILITY.md` - exact gate/up
  adjacency, shared-input, and fusion-boundary feasibility finding.
- Phase 10: `phase-10/Q6K_PP_SHARED_Q8_GATE_UP.md` - rejected shared-Q8_1
  gate/up dispatcher, with explicit `WHAT_DIDNT_WORK` analysis.
- Phase 11: `phase-11/Q6K_DECODE_LANE_UTILIZATION_AUDIT.md` - rejected the
  lighttransport-style idle-lane hypothesis without a source change or
  benchmark, while preserving terminal-wave underfill and prior failures.
- Phase 11 follow-up: `phase-11/Q6K_VEC_DOT_LOAD_ISA_AUDIT.md` - confirmed
  packed/coalesced gfx1201 Q6_K and Q8_1 payload loads and rejected an explicit
  q6k_dot4-style vectorization candidate without building or benchmarking.
- Phase 12: `phase-12/Q6K_PP_PRODUCER_TO_D4_Q81_FEASIBILITY.md` - source/trace
  feasibility and selection of a later SwiGLU-to-MMQ-D4 `ffn_down` candidate
  contract, with no implementation or performance claim.
- Phase 12 evaluation: `phase-12/Q6K_SWIGLU_D4_FFN_DOWN_FINDING.md` - rejected
  producer-to-D4 candidate with complete parity, graph, resource, variance,
  and `WHAT_DIDNT_WORK` evidence.
- Phase 13: `phase-13/Q6K_MMA_FLOAT_CONVERSION_FINDING.md` - accepted and
  registered one-line Q6_K accumulator conversion fix, including untrimmed
  PP/TG results, paired kernel trace, ISA/resources, and failed procedures.
- Phase 12 evaluation: `phase-12/Q6K_SWIGLU_D4_FFN_DOWN_FINDING.md` - rejected
  the correct producer-to-D4 candidate after stable exact-chain regressions.
- Phase 13: `phase-13/Q6K_MMA_FLOAT_CONVERSION_FINDING.md` - accepted and
  registered the one-line Q6_K integer-accumulator-to-F32 fix after a
  `1.277663x` PP512 gain, flat TG128, and complete promotion gates.
- Phase 14A: `phase-14/POST_CAST_PP_REBASELINE_FINDING.md` - refreshed the
  registered post-cast PP hotspot, resource, device-balance, graph, and
  communication evidence without creating a candidate.
- Phase 15: `phase-15/Q6K_PAIRED_Y64_W4_FINDING.md` - rejected the structurally
  valid paired Y64/W4 geometry despite its 33.6% LDS reduction.
- Phase 16: `phase-16/Q6K_TG_K2_PIPELINE_FINDING.md` - records the real outer-K
  scheduling gap, global K2 codegen failure, fused-only K2 regression, and
  invalid trailing-ADD microbench attempts.
- Phase 17: `phase-17/Q6K_Q81_NATIVE_DOT8_FINDING.md` - exact static arithmetic
  and gfx1201 ISA proof for the rejected half-utilized native-dot8 route.
- Phase 18: `phase-18/Q6K_FUSED_K2_DOT8_FINDING.md` - rejected the actual
  stacked fused K2 plus dot8 production kernel at its pre-GPU resource gate.
- Phase 19: `phase-19/Q6K_MTP_SMALL_N_FINDING.md` - real target-backed MTP
  verification trace that rejects the N=2-4 boundary and selects the existing
  one-wave DP4A N=5 MMVQ path for a candidate-free successor audit.
- Phase 20: `phase-20/Q6K_MTP_N5_R2_FEASIBILITY.md` - source/ISA feasibility
  proof for two-row Q8_1 reuse and the fail-closed MTP verification marker,
  authorizing a separate R2 candidate without claiming a speedup.
- Phase 21: `phase-21/DFLASH_DDTREE_FEASIBILITY_STUDY.md` - source, algorithm,
  hybrid-state, tensor-split, memory, and experiment feasibility study that
  keeps DDTree in a separate major-work branch and selects linear DFlash
  instrumentation before any tree candidate.
