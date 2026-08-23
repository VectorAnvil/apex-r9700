# Phase 18 Handoff

Phase 18 built the actual fused-only K2 plus native-dot8 llama.cpp candidate.
The stack is mechanically valid and K2's partial load/compute overlap survived
in generated gfx1201 ISA. The unfused Q6_K N=1 symbol remained byte-identical.

The full fused result nevertheless failed before GPU execution:

```text
                       K2 DP4A    K2 + dot8
VGPR                      60          66
SGPR                      42          42
LDS bytes               1792        1792
global loads              30          30
dot instructions           8          32
ISA instruction lines    918        1095
```

The frozen VGPR limit was 64. No correctness, microbenchmark, TG128, PP512, or
registration stage followed the failure. The registered Phase 13 library is
unchanged.

The next TG work should return to measured workload boundaries. A high-value
option is a trace-only Phase 19 of real Qwen3.6 MTP verification to establish
whether N=2/3/4 Q6_K operations exist and whether narrow DP4A avoids wasted
MMQ width. Keep Stream-K as a separate PP backlog item.
