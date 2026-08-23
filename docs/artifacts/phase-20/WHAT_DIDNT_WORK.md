# What Didn't Work

## Phase 20B R2 build

The separately frozen two-row N=5 R2 code object achieved its structural load
goal but not its bounded live-state goal: 30 global loads and 20 `v_dot4` came
with 80 VGPR and 34 SGPR, above the immutable 64/32 limits. Generic Q6_K N=1-5
unfused fallback encodings and metadata were byte-identical, so no fallback
regression was involved. The hard static failure intentionally prevented GPU
correctness, timing, graph, trace, communication, and counter work. No
promotion occurred.

- The fixed-width WMMA premise remains false. N=5 already uses one-wave DP4A
  MMVQ.
- One-wave-per-column repeats the five shared Q6 field loads across five waves,
  increasing the load model from 25 to 45 per row.
- K partitioning removes no VMEM or DP4A work and adds partial reductions.
- R4 has no bounded production code-object precedent and may require LDS or an
  unacceptable twenty-accumulator live set. It is not bundled with R2.
- A shape-only N=5 selector is unsafe for ordinary and mixed multi-slot server
  batches. Context type also identifies the draft context, not the hot target
  verification graph.
- Achieved occupancy and bandwidth remain unavailable. Static allocation and
  modeled load counts are not reported as achieved hardware behavior.
- The candidate-free Phase 20 audit performed no source edit, candidate build,
  GPU launch, benchmark, or registration. Its separate Phase 20B build then
  rejected R2 at static resources, still with no GPU work or promotion.
