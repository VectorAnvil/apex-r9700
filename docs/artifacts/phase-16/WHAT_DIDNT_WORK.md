# What Didn't Work

- The global K2 candidate produced future-block overlap only in the fused
  symbol; LLVM collapsed the unfused schedule back to one packet per loop.
- Fused-only K2 raised VGPR from 35 to 60 and measured `0.970618x` combined on
  the exact full graph. It is rejected and was not registered.
- Correctness selectors v1/v2 found no fused rows; dimension-only v3 passed
  84/84 rows.
- Microbench v1 lacked a perf-list case. V2/v3 repeated only the final ADD and
  yielded invalid 3-us results. The full-graph v4 result is the sole timing
  evidence.
- GPU 0 baseline spread was 4.432%, above the frozen 3.5% limit. Stable GPU 1
  still showed a decisive `0.958899x` regression.
- TG128, PP512, graph/trace, and counter stages were skipped after fail-fast
  rejection. Achieved occupancy and bandwidth remain unavailable.
