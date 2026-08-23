# Q6_K TG K2 Pipeline - 0.9706x Exact Fused Result

## Summary

Phase 16 investigated a software-pipelined K2 schedule for the ordinary N=1
Q6_K MMVQ path on two Radeon AI PRO R9700 GPUs. The work was split into three
immutable stages so that source feasibility, generated code, and performance
could not be conflated:

1. Phase 16A found a real outer-K scheduling gap in the registered kernel.
2. Phase 16B rejected a global fused/unfused K2 candidate before GPU execution
   because LLVM retained no future-block overlap in the unfused specialization.
3. Phase 16C tested a fused-only candidate. It passed static and correctness
   gates, but full-graph exact timing regressed to `0.970618x` combined.

The candidate is rejected and was not registered. TG128, PP512, graph, trace,
and counter stages were not run after the mandatory exact gate failed.

## Phase 16A: Static Feasibility

The registered gfx1201 Q6_K N=1 path uses eight wave32 waves. Each wave starts
at one Q6_K superblock and advances by eight blocks. Material K dimensions
therefore give each wave the following outer-loop work:

| K | Q6_K blocks | Blocks per wave |
|---:|---:|---:|
| 3072 | 12 | 1-2 |
| 5120 | 20 | 2-3 |
| 8704 | 34 | 4-5 |

The inner QR6 loop was already unrolled and Phase 11 had already proved that
ql, qh, scales, and Q8_1 accesses were packed/coalesced. The new finding was
temporal: generated ISA issued no future outer-K packet before completing the
current block's unpack, DP4A, scale, accumulation, and loop backedge. A guarded
`kbx+8` packet could be addressed independently and safely. This authorized K2
only; it did not reopen vector-width or wave-count experiments.

No source edit, build, GPU launch, or benchmark occurred in Phase 16A.

## Phase 16B: Global K2 Codegen

The isolated candidate added a one-packet lookahead for gfx1201 Q6_K N=1 and
preserved the original K1 fallback. Fresh code objects showed different LLVM
outcomes:

- Fused: next-K Q8, primary-Q6, and gate-Q6 loads were scheduled after current
  primary DP4A but before current gate DP4A. This was a real partial overlap.
- Unfused: the compiler retained one nine-load packet per iteration. No
  future-block overlap survived.

The frozen task required the pipeline in both target symbols, so Phase 16B was
rejected before GPU execution. Resources also exposed the risk:

| Specialization | Baseline VGPR/SGPR/LDS | K2 VGPR/SGPR/LDS | Spills/private |
|---|---:|---:|---:|
| Unfused | 26 / 26 / 896 B | 28 / 24 / 896 B | 0 / 0 |
| Fused | 35 / 42 / 1,792 B | 60 / 42 / 1,792 B | 0 / 0 |

The global candidate was not benchmarked. Its positive fused codegen evidence
authorized a new, narrower Phase 16C task instead of weakening Phase 16B.

## Phase 16C: Fused-Only K2

The candidate was guarded by gfx1201/RDNA4, Q6_K, N=1, and compile-time
`has_fusion`. The unfused Q6_K N=1 machine-code slice remained byte-identical
to baseline with SHA-256 `baa2f465...f03387d` and identical resource metadata.
The fused symbol retained the partial next-K-load/current-gate-compute overlap,
used 60 VGPR and 42 SGPR, and had no private memory or spills.

Correctness passed 84/84 rows: three repetitions, two GPUs, two variants, and
seven cases covering fused and unfused K=3072/5120/8704 plus a Q4_K fallback.
The identical test-only harness enforced NMSE <= `0.0005`.

## Exact Full-Graph Result

The accepted benchmark replayed the complete fused
`M=8704,N=1,K=5120` graph once per timed iteration with normal HIP graphs
enabled. Each value below is an untrimmed mean of six measurements from three
AB/BA rounds.

| GPU | Baseline us | Candidate us | Speedup | Baseline spread | Candidate spread |
|---:|---:|---:|---:|---:|---:|
| 0 | 134.255 | 136.630 | 0.982617x | 4.432% | 1.405% |
| 1 | 134.150 | 139.900 | 0.958899x | 0.328% | 0.393% |
| Combined | 134.2025 | 138.2650 | 0.970618x | - | - |

The candidate failed the required `>=1.02x` combined speedup and the rule that
neither device regress. GPU 0 baseline spread also exceeded the `3.5%` limit.
GPU 1 is independently decisive: both variants were stable and the candidate
was about 4.1% slower.

## What Didn't Work

- A global K2 source schedule did not survive LLVM code generation in the
  unfused specialization.
- The fused schedule did create partial overlap, but increased VGPR from 35 to
  60. The measured result shows that this trade was unfavorable.
- Correctness attempts v1 and v2 selected no fused rows. A dimension-only
  selector produced the accepted 84/84 result in v3.
- Microbench v1 added the exact case only to the evaluation list, while perf
  mode uses a separate list.
- Microbench v2 and v3 reported about 3 us because the stock perf harness
  duplicated only the trailing output `ADD` node 335,749 times. The 102 kB/run
  output exposed the error. Both attempts are retained as invalid procedure
  evidence and were excluded from the decision.
- An identical test-only fix in both worktrees set `n_runs=1` for whole-graph
  tests and replayed the original complete graph. Only v4 is gating evidence.
- TG128, PP512, profiler traces, graph-count comparison, and dynamic counters
  were deliberately skipped after the exact gate failed. Achieved occupancy
  and bandwidth remain unavailable.

## Decision And Next Step

Reject fused-only K2 and demote both global and fused K2 scheduling for this
registered workload. Do not try K4 as a rescue: common K=5120 gives only two or
three blocks per wave, and K2 already raised the fused live state by 25 VGPR.

The next independent TG feasibility question is native dot8 arithmetic, not
scheduling. Before any integration or GPU timing, compare a register-only exact
Q6_K x Q8_1 decomposition against the current two-DP4A implementation. With
`l=low4`, `h=high2-2`, `p=x&0xF`, and signed `r=x>>4`, the exact expansion is:

```text
l*p + 16*l*r + 16*h*p + 256*h*r
```

The signs are plus; signed `h` and `r` carry negative values. Reject dot8
statically if four dot8 operations plus nibble extraction, packing, coefficient
combination, and added live state are clearly worse than the current two DP4A
operations and unpack sequence.
