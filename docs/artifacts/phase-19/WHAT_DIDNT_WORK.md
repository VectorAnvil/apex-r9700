# What Didn't Work

- The proposed N=2-4 selector misses the measured workload. N=2 and N=3 did
  not occur during the frozen request, and N=4 was only 2.6467% of summed MTP
  GPU kernel duration, below both combined and per-device materiality gates.
- N=5 dominated at 56.9103%. It was deliberately accounted as fallback under
  the frozen contract and cannot be silently folded into the rejected N=2-4
  task after seeing the data.
- The fixed-width-WMMA waste premise does not match this registered build.
  Q6_K N=2-5 already routes to native DP4A MMVQ, with one wave for N>1.
- The N=5 code object is not obviously resource-broken: 50 VGPR, 32 SGPR,
  zero LDS/private/spills, 25 global loads, and 10 DP4As. A successor needs a
  genuinely different row/column work organization, not a route-name change.
- The target verification graph has no MTP marker at the HIP dispatch layer.
  A shape-only selector would also affect ordinary small batches and multi-slot
  serving unless context metadata is deliberately propagated.
- rocprofv3 successfully flushed both raw databases after SIGINT but did not
  exit. The runner retained the complete output, then killed the processes;
  both `-9` return codes are preserved as procedure failures.
- Achieved occupancy and bandwidth remain unavailable. Static allocation is
  not reported as an achieved counter.
- No optimization candidate, correctness test, benchmark comparison, or
  registration was performed in Phase 19.
