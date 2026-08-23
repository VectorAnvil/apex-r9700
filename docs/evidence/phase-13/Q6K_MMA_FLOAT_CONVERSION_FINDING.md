# Q6_K MMA Float-Conversion Fix

## Summary

**Accepted and registered.** A single explicit cast of the integer WMMA
accumulator to F32 before Q6_K scale multiplication improved the registered
dual-R9700 Qwen3.6 27B Q6_K PP512 workload by `1.277663x` while leaving TG128
flat at `0.999895x`.

- PP512: `814.363 -> 1040.482 tok/s`, `+27.77%`.
- TG128: `36.7308 -> 36.7270 tok/s`, effectively unchanged.
- Profiled Q6_K MMQ time: `1.438788 -> 0.889639 s`, `1.617272x`.
- Correctness: 24/24 Q6_K CPU-oracle executions passed on both devices.
- Graphs: 514 `hipGraphLaunch` calls observed in the candidate runtime trace.

The registered source and `libggml-hip.so` now include the accepted fix. An
independent post-promotion PP512 smoke measured `1039.410 tok/s`.

## Change

The only substantive Q6_K source change is in
`vec_dot_q6_K_q8_1_mma`:

```cpp
- C.x[l] * sc[k01/4] * x_df[...] * dB
+ ((float) C.x[l]) * sc[k01/4] * x_df[...] * dB
```

This is the Q6-only change from draft llama.cpp PR #25940, commit
`68adcd0bf582ba3c9a345ceaecac2c991f1e710a`. The PR's Q2_K and dispatch
changes were excluded.

## Evaluation

| Gate | Baseline | Candidate | Result |
|---|---:|---:|---|
| PP512 mean | 814.363 tok/s | 1040.482 tok/s | PASS, `1.277663x` |
| PP512 spread | 0.235% | 1.016% | PASS |
| TG128 mean | 36.7308 tok/s | 36.7270 tok/s | PASS, `0.999895x` |
| TG128 spread | 0.161% | 0.088% | PASS |
| Q6_K CPU oracle | 12/12 | 12/12 | PASS |
| Q6_K dispatches | 1,984 | 1,984 | PASS |
| direct-copy dispatches | 1,240 | 1,240 | PASS |
| HIP graph launches | n/a | 514 | PASS |

All performance samples are untrimmed. The whole-workload schedule alternated
three baseline and three candidate processes with normal graph settings and
the registered tensor split.

## Codegen And Resources

The static ISA change agrees with the proposed LLVM failure mode:

| Instruction/resource | Baseline | Candidate |
|---|---:|---:|
| `v_cvt_f32_i32` | 1,824 | 3,456 |
| `v_mul_lo_u32` | 2,461 | 637 |
| `v_mul_f32` | 457 | 1,223 |
| runtime VGPR allocation | 232 | 256 |
| runtime SGPR allocation | 128 | 128 |
| private/scratch | 0 B | 0 B |
| VGPR/SGPR spills | 0 / 0 | 0 / 0 |
| waves per workgroup | 8 | 8 |
| dynamic LDS | 57,856 B | 57,856 B |

The cast increases VGPR use but removes most integer multiplies from the Q6_K
code object and is substantially faster. Geometry, wave count, LDS, dispatch
counts, and P2P-copy counts are unchanged.

The opcode table counts the complete Q6_K template code object. Restricting
the comparison to the two active MMQ-128 symbols gives the same direction:
`v_mul_lo_u32` falls from 349 to 93 while `v_cvt_f32_i32` rises from 256 to
320; the focused disassembly contracts from 5,597 to 5,150 instruction lines.

Achieved occupancy and `FETCH_SIZE` bandwidth remain **unavailable**. Both
bounded, nonmultiplexed gfx1201 PMC attempts aborted in ROCprofiler-SDK with
`std::out_of_range: unordered_map::at`. Static allocation is not reported as
achieved occupancy, and modeled traffic is not reported as hardware bandwidth.

## WHAT_DIDNT_WORK

The registered production build did not include `test-backend-ops`, so a
matched result-local no-cast baseline test build was required. Four setup-only
correctness attempts selected zero rows while resolving the old sealed Phase 4
harness and stock test dimensions. Those attempts launched no failing test row
and remain preserved; the successful gate used the stock Q6_K `N=8` and `N=64`
MMA cases for 24 total both-device executions.

The first counter attempt also selected no performance row. The corrected
counter attempt executed the confirmed Q6_K oracle case, reproduced the SDK
exception on both devices, and was terminated by the frozen 20-second guard.

## Provenance

- Frozen task SHA-256:
  `55f2366ddad3fc7ba56bac30e36f595cb67c011aa27711a8cd6f734c83b4a4c2`.
- Candidate patch SHA-256:
  `e1c70a02535e516383217ac53154915e701dd9338d1a3dcef37900645dcf5690`.
- Baseline/candidate Q6_K HSACO SHA-256:
  `9336c932...eff3e` / `d8d464fa...de6a`.
- Promoted `libggml-hip.so` SHA-256:
  `facd1354c4eba6afec9af0b22694e6ca11bf2b2bd976368158f225d6692c4311`.
- Authentic same-revision pre-cast rollback library SHA-256:
  `7f58e700141b23d9e14a567ed6b22a71f53b09aeda2176a705dca974bbd55dda`.
- Raw evidence:
  `results_phase13_q6k_mma_float_cast_20260808/`.
- Compact evidence: `docs/apex-r9700/artifacts/phase-13/`.
