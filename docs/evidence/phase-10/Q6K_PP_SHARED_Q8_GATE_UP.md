# Q6_K PP Shared-Q8_1 Gate/Up Dispatcher

## Finding

**Rejected by the immutable Phase 10 performance gate.** The default-off
gfx1201 Q6_K PP512 dispatcher was functionally correct and selected as
intended, but removing one of two activation quantizations did not produce the
required stable performance gain.

The exact operation gate measured `0.999896x` combined gate/up speedup with
`0.006008` relative spread. The required gain was at least `1.010x`. Normal
graph whole-PP measured `1.003627x` with `0.002589` relative spread across
three processes per selector state. The required gain was at least `1.005x`.
Both results are stable enough to evaluate; neither clears its predeclared
threshold. There is no registration and no performance claim.

## Scope

- Workload: registered Qwen3.6 Q6_K PP512 workload on two gfx1201 R9700 GPUs.
- Exact per-device projections: `blk.<layer>.ffn_gate.weight` and
  `blk.<layer>.ffn_up.weight`, logical `M=8704,N=512,K=5120`.
- Candidate: opt-in
  `GGML_CUDA_Q6K_SHARED_Q8_GATE_UP=1` quantizes the common F32 activation to
  Q8_1 once, then uses the existing independent Q6_K MMQ projection launches.
- Baseline: the same candidate binary with the selector unset/not `1`.
- Retained behavior: independent weights, existing F32 SwiGLU,
  normal HIP graph capture/replay, and per-device tensor-parallel ownership.

## Validation

| Gate | Result | Status |
|---|---:|---|
| Candidate dispatch, devices 0 and 1 | Two quantize dispatches reduced to one per exact pair | PASS |
| Fallback dispatch, N=256 | Two quantize dispatches retained | PASS |
| CPU oracle / F32 SwiGLU parity | Passed within NMSE `0.0005` contract | PASS |
| Exact gate/up timing, 3 untrimmed rounds | `0.999896x`, spread `0.006008` | FAIL, needs `>=1.010x` |
| Normal-graph whole PP, 3 processes/variant | `1.003627x`, spread `0.002589` | FAIL, needs `>=1.005x` |

The same candidate binary SHA-256 was used with the selector off and on. No
samples were trimmed. The result is therefore a real negative result, rather
than an unsupported timing comparison.

## Resources

Static MMQ resource evidence on both devices was unchanged at 229 VGPR, 27
SGPR, 57,856 bytes LDS, zero private segment bytes, zero spills, and eight
waves per workgroup. This candidate reuses the existing MMQ kernel; it does
not claim a new kernel resource improvement.

Dynamic occupancy is **unavailable**. ROCm 7.2 lists `OccupancyPercent` for
gfx1201, but the four prior exact nonmultiplexed captures timed out after 120
seconds. Phase 10 did not repeat a known non-producing capture for a change
that does not alter the GPU MMQ kernel.

Achieved bandwidth is **unavailable**. ROCm 7.2 lists `FETCH_SIZE`, but prior
exact nonmultiplexed captures produced no joinable counter value. Any
test-backend-ops GB/s calculation remains a timing diagnostic, not a hardware
counter measurement.

## WHAT_DIDNT_WORK

The hypothesis was that avoiding the duplicate Q8_1 activation quantization
would yield enough savings in the 38.304% gate/up PP hotspot to clear a
meaningful operation and whole-workload gain. It did remove exactly one
quantize dispatch per selected pair on both devices, and it did not damage
fallback behavior or numerical correctness. It still did not speed up the
combined gate/up operation: `0.999896x` is effectively flat and slightly
slower than baseline. The `1.003627x` whole-PP result is also below the
precommitted `1.005x` floor.

This is not evidence that quantization reuse is broken. It is evidence that,
at this exact shape and implementation boundary, the removed launch and Q8_1
write are not a large enough bottleneck to justify promotion. Do not loosen
the thresholds, report the whole-PP delta as a gain, or register this
dispatcher. The experiment remains recorded to demote the same idea in future
candidate selection.

## Provenance

- Immutable task fingerprint:
  `e5690099bba3770b565276c0d33a186bd49a2fba8decb7696d85c125f380d907`.
- Finalizer decision: `rejected`; failures were only the exact-operation and
  whole-PP performance gates.
- Result-local evidence:
  `results_phase10_shared_q8_gate_up_20260808/`.
- Pinned llama.cpp source commit:
  `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`.
