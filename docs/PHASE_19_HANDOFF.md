# Phase 19 Handoff

Phase 19 traced one deterministic 128-token request through the real
target-backed Qwen3.6 MTP server path on both R9700s. It built no optimization
candidate and left the registered Phase 13 library unchanged.

The proposed N=2-4 narrow-DP4A boundary is rejected:

```text
Logical N    Calls    Summed duration    Share of request GPU kernel time
N=1          2,562       375.611 ms       9.5941%
N=2              0         0.000 ms       0.0000%
N=3              0         0.000 ms       0.0000%
N=4          2,018       103.618 ms       2.6467%
N=5         39,390     2,228.050 ms      56.9103%
```

N=2-4 failed the frozen materiality gate: 2.6467% combined versus 10%
required, 2.4150% on GPU 0 and 2.9181% on GPU 1 versus 5% required on each.
Every request-window Q6_K dispatch joined exactly: 43,970 matched, zero
unmatched, ambiguous, or route mismatches.

The important correction is architectural. Registered gfx1201 already routes
N=2-5 Q6_K work through narrow DP4A MMVQ, not fixed-width WMMA/MMQ or dense
GEMM. N=5 uses one wave32 workgroup, 50 VGPR, 32 SGPR, no LDS, private memory,
or spills, 25 global loads, and 10 DP4A instructions. There are no padded
16-column WMMA lanes to remove.

Phase 20 should audit the actual N=5 MMVQ dataflow and work organization. It
must also define how an MTP verification marker reaches the HIP dispatch
boundary; the current backend sees only shape and type, so a shape-only N=5
selector would also affect ordinary small batches. Do not carry forward K2,
K4, dot8, or a forced-MMQ premise.
