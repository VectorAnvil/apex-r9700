# Phase 16 Handoff

Phase 16 found a genuine outer-K scheduling gap, but neither resulting K2
candidate is promotable. The global candidate failed its static gate because
the unfused specialization retained no future-block overlap. The fused-only
candidate retained partial overlap but raised VGPR from 35 to 60 and regressed
the exact fused `8704x1x5120` graph.

Correctness passed 84/84. The accepted six-sample full-graph result was:

```text
GPU 0: 134.255 -> 136.630 us, 0.982617x
GPU 1: 134.150 -> 139.900 us, 0.958899x
dual:  134.2025 -> 138.2650 us, 0.970618x
```

GPU 1 was stable at 0.328%/0.393% spread. GPU 0 baseline spread was 4.432%,
which independently failed the frozen stability limit. No TG128, PP512,
profiler, graph-count, or counter stage followed the hard exact-gate failure.
The registered Phase 13 library remains unchanged.

Phase 17 should be a static-only register microkernel/codegen feasibility gate
for exact Q6_K x Q8_1 dot8 arithmetic. Keep it independent of K2/K4. Prove the
four-term signed-nibble identity exhaustively, compile gfx1201 code, and compare
instruction count, dependencies, and live resources with the registered two-
DP4A implementation. No production candidate or GPU benchmark is permitted
unless that static gate passes.
