# Phase 20 Q6_K MTP N=5 R2 Feasibility

## Result

Phase 20 authorized a separately frozen R2 experiment for the dominant Q6_K
N=5 target-verification path. Phase 20B built it and rejected it at the static
resource gate. This remains a no-speedup result: no GPU workload was launched.

## Registered dataflow

One physical wave32 computes one output row and all five columns. Per Q6_K
block, the compiler emits five shared Q6 field loads (`ql`, `qh`, two scale
bytes, and `d`), twenty Q8_1 loads for five distinct activation columns, ten
DP4As, five FP32 accumulator chains, and five wave reductions.

| Registered N=5 resource | Value |
|---|---:|
| Workgroup | 32 x 1 x 1 |
| Rows / columns | 1 / 5 |
| VGPR / SGPR | 50 / 32 |
| LDS / private / spills | 0 / 0 / 0 |
| Global loads | 25 |
| DP4A | 10 |

The Q6 loads are already shared optimally across columns. Phase 11's packed
load finding remains binding, and the ISA groups VMEM before dependent dot
work. There is no new column-load or K2 opportunity.

## Selected R2 mechanism

Two adjacent weight rows consume the same five Q8_1 columns. Two registered
blocks therefore execute 50 loads per K-block pair. A single-wave R2 kernel can
load ten Q6 fields and twenty Q8 fields, then compute the same 20 DP4As and ten
reductions: 30 loads, a modeled 40% reduction.

The registered `mul_mat_vec_q_moe<Q6_K,2>` code object proves this reuse is
available to LLVM. Its two-row, one-column body emits fourteen relevant loads
instead of eighteen, with no LDS or barriers. It uses 40 VGPR, 26 SGPR, no
private memory, and no spills.

The combined N=5 R2 resource projection was deliberately bounded at 55-64
VGPR and 32 SGPR. Ten accumulators were the primary risk. The Phase 20B R2
symbol reached 80 VGPR and 34 SGPR, despite retaining 30 global loads, 20
`v_dot4`, zero barriers, zero LDS/private allocation, and zero spills. It
therefore failed before correctness or timing.

All Phase 19 N=5 M values are even: 24, 512, 3072, 5120, 6144, 8704, and
124160. K values 3072, 5120, 8704, and 10240 are Q6_K-block aligned. The
candidate may require even local M and must fall back for odd tails.

## Alternatives rejected

- One wave per column repeats the five Q6 fields across five waves, increasing
  the row load model from 25 to 45. Large M already supplies wave parallelism.
- K partitioning preserves all VMEM and DP4A work and adds partial reductions.
- R4 may reduce more activation loads but has no bounded live-set precedent;
  it could require LDS/barriers or twenty accumulators. It is excluded.

## Verification provenance

The hot N=5 graph belongs to target verification, not the MTP draft context.
`LLAMA_CONTEXT_TYPE_MTP` is therefore the wrong selector. The minimum safe path
is server per-row provenance, an all-rows verification decode mode, separation
in graph reuse identity, a backend-visible MUL_MAT tensor flag, and an exact
HIP selector. Ordinary, prompt, mixed multi-slot, odd-M, non-Q6, non-N5, and
non-gfx1201 cases must retain baseline machine code and dispatch.

## Decision

Close Phase 20B as `complete_reject_no_promotion`. The two variants built and
generic Q6_K N=1-5 unfused machine encodings plus AMDHSA metadata remained
identical, but the dedicated R2 symbol breached both resource limits. Do not
treat the 40% load model as achieved bandwidth or throughput. All GPU-stage
gates remain unrun by fail-fast design.

## What Didn't Work

- Column and K partitioning move or repeat work rather than remove it.
- R4 is not bounded well enough to bundle with R2.
- Shape-only N=5 and MTP-context-type selectors are unsafe.
- Achieved occupancy and bandwidth remain unavailable.
- R2's ten-accumulator live state exceeded both register ceilings.
- Phase 20B generated no performance result; registration is unchanged.

Compact artifacts are under `docs/apex-r9700/artifacts/phase-20/`. Source
copies, the full registered metadata, ISA slices, and static models remain in
the ignored Phase 20 root.
