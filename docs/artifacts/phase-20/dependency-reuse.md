# N=5 Dependency and Reuse Audit

## Registered path

One 32-thread wave computes one output row and all five activation columns.
All lanes cooperate on one 256-element Q6_K block per loop iteration. The Q6
weight fields are identical for the five columns, while each column has a
different Q8_1 activation packet.

The registered ISA has exactly the expected load split:

- five Q6 loads: `ql`, `qh`, two scale bytes, and `d`;
- twenty Q8_1 loads: two packed `qs` and two `ds` values for each column;
- ten DP4As: two per column;
- five independent FP32 accumulator chains and five wave reductions.

The compiler already hoists the Q6 fields across columns and groups VMEM before
the dependent dot sequence. Phase 11's packed-load result remains binding;
there is no vector-width candidate here.

## Two-row reuse

Two adjacent output rows use different Q6 weights but the same five Q8_1
activation columns. Two registered one-row blocks therefore execute, per K
iteration, ten Q6 loads plus forty Q8 loads. A one-wave R2 specialization can
retain each column's Q8 packet while consuming row 0 and row 1, reducing the
model to ten Q6 plus twenty Q8 loads. DP4A and reduction counts do not change.

This is not compiler wishful thinking. The registered
`mul_mat_vec_q_moe<Q6_K,2>` kernel uses the same `vec_dot_q6_K_q8_1` helper for
two rows. Its code object emits fourteen relevant loads rather than eighteen:
ten Q6 fields plus four shared Q8 fields. It uses no LDS, barriers, private
memory, or spills.

## Risks

R2 doubles the N=5 accumulator count from five to ten and retains a second Q6
packet long enough to consume the common activation values. The bounded
projection is 55-64 VGPR, but only a candidate code object can establish the
actual allocation. Phase 20B must stop before GPU work if VGPR exceeds 64 or
if private memory or spills appear.

The generic kernel does not guard speculative row loads with `nrows_x`.
Phase 20B must select R2 only when the local M slice is even. All Phase 19 N=5
M values are even; odd or mixed cases remain on the byte-identical baseline.

R4 is excluded. It has no matching bounded code-object precedent and could
require LDS or an unacceptable twenty-accumulator live set.
