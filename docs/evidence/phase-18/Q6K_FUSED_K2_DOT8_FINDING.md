# Phase 18 Fused K2 plus Native Dot8 Finding

## Result

The user-authorized stacked rescue is rejected at its immutable static gate.
The actual fused Q6_K N=1 llama.cpp specialization compiled successfully with
both K2 packet scheduling and exact native dot8 arithmetic, but fused VGPR rose
from 60 to 66, above the frozen limit of 64. No GPU workload was launched.

## What stacked

The mechanisms are compatible at the source and ISA levels:

- current-primary dot8 compute occurs at `0xE3D8C-0xE3DE0`;
- next-K packet VMEM occurs at `0xE3F98-0xE4098`;
- current-gate dot8 compute follows at `0xE434C-0xE43A0`.

That is the same useful partial overlap found in Phase 16C: next-block loads
are issued after current primary work and before current gate work. The
unfused symbol remained byte-identical with machine-code SHA-256
`baa2f465a63dff11fce78d648b074ec64143b07cf3bab41ec0749e0a6f03387d`.

## Code-object comparison

| Metric | Fused K2 DP4A | Fused K2 + dot8 |
|---|---:|---:|
| VGPR | 60 | 66 |
| SGPR | 42 | 42 |
| Static LDS | 1,792 B | 1,792 B |
| Private/spills | 0/0 | 0/0 |
| Global loads | 30 | 30 |
| DP4A | 8 | 0 |
| Native dot8 | 0 | 32 |
| ISA instruction lines | 918 | 1,095 |

The K2 overlap therefore did not make dot8 cheaper. It exposed the same
four-reduction arithmetic inside both primary and gate streams, increasing
live state and the mask/shift/OR dependency chain without reducing VMEM.

## Decision

The candidate fails `VGPR <= 64` at 66 VGPR. Per the predeclared stop rule,
correctness, exact timing, TG128, and registration were prohibited. This is a
scientifically useful negative result: K2 overlap survived, so the rejection
is about the combined arithmetic/resource cost rather than a failed compiler
schedule.

## What Didn't Work

- K2 preserved its partial prefetch overlap but could not hide four times as
  many dot instructions plus nibble packing and recombination.
- Global-load count stayed at 30, so the stack did not buy less memory work.
- VGPR increased `60 -> 66`, crossing the immutable pre-GPU ceiling.
- Fused ISA grew by 177 instruction lines; ALU waits and integer bit
  manipulation also rose sharply.
- The fail-closed selector worked and the unfused kernel stayed byte-identical,
  but that does not rescue the targeted fused regression risk.
- No GPU result is claimed. The registered Phase 13 build was not modified.

Compact evidence is in `docs/apex-r9700/artifacts/phase-18/`; full source,
patches, build logs, HSACO, and ISA remain under the ignored Phase 18 root.
