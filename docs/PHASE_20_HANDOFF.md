# Phase 20 Handoff

Phase 20 completed a candidate-free source and gfx1201 ISA audit of the Q6_K
N=5 MMVQ path selected by the real MTP trace. Phase 20B then built the
fail-closed one-wave, two-row R2 specialization and rejected it at the frozen
static resource gate, before GPU work.

The registered kernel computes one output row and five columns per wave. LLVM
already shares the five Q6 weight fields across columns, so column partitioning
is not useful. Adjacent output rows, however, use the same five Q8_1 activation
packets. The selected model is:

```text
                            Two baseline rows    R2 candidate model
Q6 field loads                      10                  10
Q8_1 activation loads               40                  20
Total global loads                  50                  30
DP4A                                20                  20
Wave reductions                     10                  10
LDS / barriers                     0 / 0               0 / 0
```

This 40% modeled load reduction has a registered compiler witness. The existing
Q6_K two-row MoE kernel emits fourteen relevant loads instead of eighteen:
ten Q6 fields plus one shared four-load Q8_1 packet. It uses 40 VGPR, 26 SGPR,
and no LDS, private memory, or spills.

The N=5 R2 live-set projection was 55-64 VGPR because it needs ten
accumulators. The actual R2 code object retained the planned 30 global loads,
20 `v_dot4`, zero barriers, zero LDS/private memory/spills, but used 80 VGPR
and 34 SGPR. Both exceed the immutable 64 VGPR / 32 SGPR ceilings. Phase 20B
is therefore `complete_reject_no_promotion`; correctness, traces, graph,
server, TG, PP, communication, and counter gates were not run.

An MTP-only selector is possible but not shape-only. Server row provenance
must produce an all-rows verification decode mode, that mode must enter graph
reuse identity, a backend-visible MUL_MAT tensor flag must survive scheduling,
and HIP must require exact gfx1201 + Q6_K + N=5 + even M + verification. Mixed
or ordinary batches remain byte-identical fallbacks.

Both Phase 20B variants built. Generic Q6_K N=1-5 unfused machine encodings
and AMDHSA metadata are identical between plumbing baseline and candidate, so
the rejection is isolated to the dedicated R2 symbol. No registered asset was
modified; the Phase 13 library remains
`facd1354c4eba6afec9af0b22694e6ca11bf2b2bd976368158f225d6692c4311`.
The DFlash/DDTree feasibility study is the next document review.
