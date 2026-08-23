# Phase 17 Q6_K x Q8_1 Native Dot8 Finding

## Result

Phase 17 is `complete_no_go`. The corrected arithmetic is exact and LLVM emits
native gfx1201 `v_dot8_i32_iu4`, but the Q8_1 operand makes the construction
structurally worse than the registered signed-int8 DP4A helper.

No llama.cpp candidate was built and no GPU workload was launched in this
phase. The registered Phase 13 library remained unchanged.

## Exact arithmetic

For one independently scaled four-value term, let `l` be the unsigned Q6 low
nibble, `h0 = high2 - 2`, and split signed Q8_1 byte `a` into unsigned low
nibble `p` and signed high nibble `r`:

```text
q6 = l + 16*h0
a  = p + 16*r
q6*a = l*p + 16*l*r + 16*h0*p + 256*h0*r
```

The v2 host oracle passed all 131,072 scalar cases across both QR6 terms and
all four lane positions, plus packed-vector, adversarial, independent-scale,
and float-order cases. The two QR6 terms cannot be merged before applying
their distinct `scales[4*i]` and `d8[i]` factors.

An earlier frozen v1 used `H=sign4(high2 xor 2)`, which is not `high2-2`.
Its counterexample is retained as failed procedure evidence and is excluded
from the v2 decision.

## Code generation

The matched register-only gfx1201 probe produced:

| Metric | Registered arithmetic | Native-dot8 probe |
|---|---:|---:|
| Productive dot instructions | 2 DP4A | 8 dot8 |
| Productive lanes per dot | 4/4 | 4/8 |
| Function size | 372 B | 876 B |
| Counted instructions | 60 | 158 |
| ALU waits | 1 | 16 |
| VGPR | 2 | 6 |
| SGPR | 16 | 22 |
| LDS/private/spills | 0/0/0 | 0/0/0 |

The compiler emitted all eight native dot8 operations rather than scalarizing
them. That confirms instruction availability, but also confirms the cost:
Q8_1 int8 must be decomposed into two i4 operands, and each independently
scaled four-value term requires four dot8 reductions with four padded lanes.

## Decision

The frozen lane-efficiency and dependency gates fail. Phase 17 does not
authorize an independent dot8 integration or benchmark.

## What Didn't Work

- Dot8 is i4 by i4, not Q6 i4 by Q8 int8. Activation nibble decomposition is
  unavoidable for an exact result.
- Independent QR6 scale boundaries prevent filling all eight lanes with the
  two four-value terms.
- Native instruction selection worked, but the emitted dependency chain grew
  sharply and required four times as many dot instructions.
- The v1 signed-rebias formula was wrong and was superseded only after its
  failure was preserved.
- The first v2 standalone compile referenced llama.cpp-only byte helpers; the
  fixed probe embedded equivalent operations and preserved the failed log.
- No production integration or GPU timing followed the hard static no-go.

Compact evidence is in `docs/apex-r9700/artifacts/phase-17/`; the full probe,
HSACO, disassembly, intrinsic manifest, and failed attempts remain under the
ignored `results_phase17_q6k_dot8_static_v2_20260809/` root.
