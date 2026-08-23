# Phase 14A Handoff

The promoted Phase 13 build is stable at `1042.144 tok/s` PP512 across three
normal-graph processes (`0.0492%` spread), with 514 graph capture/replay
witnesses. A fresh exact six-repeat trace join shows Q6_K MMQ still dominates
at `77.7567%` of summed GPU dispatch duration. Gate, up, and down together are
`51.6601%`; Q6_K device imbalance is only `1.2980%`.

Phase 15 is authorized for the paired gfx1201 geometry `mmq_y=64,nwarps=4` on
top of the registered cast baseline. It satisfies the WMMA invariant
`4 * tile_C::I(16) == 64`. It is not the invalid Phase 8 Y64/eight-wave
configuration and must never be tested against the pre-cast kernel.

Freeze a new immutable Phase 15 task before editing. Change only the host and
device RDNA4 MMQ Y/warp selectors needed for the paired geometry. Do not mix
Stream-K, launch thresholds, arithmetic, fusion, or other quant types into the
candidate. Require both-device Q6_K correctness, exact dispatch witness,
normal graph replay, tensor/P2P equivalence, promoted-vs-candidate resources
and ISA, stable exact-operation timing, and three interleaved PP process pairs.
TG128 is a no-regression witness, not the target.

After Phase 15 closes, switch to TG. First audit and, only if codegen shows a
real scheduling gap, test Q6_K N=1 K2/K4 software-pipelined MMVQ. Separately,
trace real MTP verification shapes before considering narrow Q6_K DP4A for
N=2/3/4.
