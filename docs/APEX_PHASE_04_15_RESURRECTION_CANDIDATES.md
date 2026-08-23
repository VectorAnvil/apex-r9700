# Apex Phases 04–15 Resurrection Candidates

This file contains only Classes C, D, E, and unresolved F candidates. Inclusion means unresolved evidence, not authorization to implement or benchmark. The minimum-rerun shortlist is narrower than this inventory.

## Priority 1 — measured positive whole workload (Class C)

### C11 — SwiGLU→D4 `ffn_down` producer

- Historical PP: `815.208808→819.466109 tok/s`, `1.005222x`.
- Veto: exact chain `.995165x/.992729x` against a mandatory `1.01x` gate.
- Why unresolved: it already cleared the old whole-PP threshold, but predates the Phase 13 cast and current production stack. No serving stability/model-behavior trial exists.
- Resurrection question: does the recovered patch still deliver a repeatable PP/latency benefit on the current registered stack without harming TG or service behavior?

### C07 — shared-Q8_1 gate/up dispatcher

- Historical PP: approximately `814.891→817.798 tok/s`, `1.003627x`.
- Veto: isolated `.999896x` and PP below the arbitrary `1.005x` promotion floor.
- Why unresolved: the measured whole workload was positive, but small; C13 materially changed the Q6 MMQ cost balance.
- Resurrection question: is the effect repeatable and useful on the present baseline, or does it disappear after C13?

## Priority 2 — positive local evidence without workload trial (Class D)

### C03 — corrected shape-gated four-wave MMV

- Selected exact-shape gains: `1.0558–1.1096x` across devices.
- Veto: one stable fallback row at `.9878x` plus several variance violations; TG was never run.
- Why unresolved: it targets TG N=1, while neither PP nor a microbenchmark-average answers actual TG128 impact. The dispatch was already shape-gated.
- Resurrection question: does selected-shape dispatch improve current TG128 enough to outweigh its fallback exposure?

### C01 — global four-wave MMV

- Positive evidence: several middle shapes gained 5–11%.
- Negative evidence: `M24` and `M512` suffered large exact regressions; fused `ffn_up` was flat.
- Relationship: C03 is the intended shape-gated refinement. C01 should not be rerun globally; its evidence is retained to define which shapes C03 must exclude.
- Current role: lineage/evidence donor, not a separate minimum rerun.

## Priority 3 — performance UNKNOWN after static/predictive screening (Class E)

### C10 — RMSNorm→shared D4 gate/up producer

- Evidence: exact adjacency on both GPUs over six repeats/64 layers; 38.304% connected PP share; producer fusion would differ materially from C07.
- Missing: implementation, numerical correctness, resources, and all performance.
- Constraint: this is new implementation work, not a cheap historical rerun. It should wait until C07/C11 adjudication establishes whether sub-1% PP gains matter in production.

### C08 — monolithic gate+up+SwiGLU MMQ

- Evidence: material pair and intermediate-traffic opportunity.
- Missing: everything after design; resource risk was predicted from a heavy baseline kernel.
- Constraint: C13 raised VGPR allocation and changed MMQ timing, so old resource predictions are stale but potentially more concerning. Requires a fresh design contract, not immediate resurrection.

### C12 — tiled MMQ/all-reduce overlap

- Evidence: proposal was displaced by C13, not technically disproved.
- Missing: exact overlap opportunity on the current service, implementation, and all performance.
- Constraint: Phase 14A found only `1.298%` device imbalance and did not expose copy kernels as the leading cost. Current production tracing should first prove a communication bubble.

### C15 — Stream-K PP

- Evidence: retained independent backlog item; never isolated in Apex.
- Missing: current-shape scheduling analysis, implementation, correctness, and performance.
- Constraint: C13 changed the kernel’s dependency/resource profile; any future contract must start from the current baseline.

### C05 — fused-only 16-wave MMV

- Evidence: 16 waves shares C04’s two K-loop count and predicts more reduction work.
- Missing: actual candidate performance.
- Current status: unresolved but low information value relative to a C03 TG trial; no minimum rerun proposed.

### C06 — Y64/W8

- Evidence: exact proposed implementation violates the writeback invariant.
- Missing: performance because it could not be validly built.
- Current status: no rerun of this exact configuration. The valid paired Y64/W4 design was C14 and was measured.

### C09 — lighttransport-style decode port

- Evidence: the alleged within-wave idle-lane mechanism does not map to llama.cpp’s wave32 Q6 block ownership.
- Missing: performance of an actual port.
- Current status: no minimum rerun unless a concrete port introduces a mechanism beyond the disproven lane premise.

## Unresolved Class F

### C14 — paired Y64/W4

- Local evidence: combined exact operation approximately `.975x` on both GPUs.
- Whole PP: `.998916x`; mean delta smaller than candidate spread.
- Missing: TG, because the exact-operation gate stopped execution.
- Current status: no PP rerun. A TG-only rerun would only be justified if source dispatch can confine this PP geometry to a TG-relevant path, which the historical global MMQ candidate did not establish.

### C04 — fused-only 12-wave

- Exact evidence: `1.0000x/1.0033x`, effectively flat.
- Missing: TG/E2E.
- Current status: no minimum rerun; C03 has stronger local evidence for answering the same TG family question.

## Minimum resurrection shortlist

Only three historical candidates currently justify GPU reruns without first authorizing substantial new implementation:

1. C11 on the current production baseline.
2. C07 on the current production baseline.
3. C03 as a TG128 workload trial, using only the corrected v2 dispatch.

No work is authorized by this list.
