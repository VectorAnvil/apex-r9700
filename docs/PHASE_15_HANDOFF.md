# Phase 15 Handoff

Paired RDNA4 `mmq_y=64,nwarps=4` was valid but rejected. It cut Q6_K dynamic
LDS from 57,856 B to 38,400 B and workgroups from eight to four wave32 waves,
while static VGPR stayed 252/253 and runtime VGPR stayed 256. Static SGPR rose
from 27/30 to 30/31; private memory, scratch, and spills remained zero.

Correctness passed 96/96, Q6_K dispatch counts stayed 1,984, both graph traces
matched at 514 launches, and only Q6_K MMQ executed. Performance failed:
gate/up was about `0.993x`, ffn_down `0.958x`, combined exact work `0.975x`,
and whole PP512 `1042.838 -> 1041.707 tok/s` (`0.998916x`). The registered
Phase 13 library remains unchanged.

Phase 16 switches to TG. Freeze a measurement-first task for registered N=1
Q6_K MMVQ K2/K4 software-pipeline feasibility. Compare source and generated
gfx1201 ISA/load scheduling against the proposed four-block pipeline. Do not
build or benchmark a candidate unless static analysis identifies repeated or
serialized loads/instructions that a real pipeline can overlap. Preserve the
existing fused-gate path and exact tails.

Stream-K remains a later PP backlog item. Do not combine it with Phase 16.
The separate MTP narrow-DP4A idea still requires a real Qwen3.6 verification
N=2/3/4 shape trace after ordinary N=1 TG work closes.
