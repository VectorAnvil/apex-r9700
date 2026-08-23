# Phase 9 Handoff

## Outcome

**Feasible candidate contract, not integrated.** Source and Phase 7 exact
attribution establish 64 adjacent gate/up pairs per device and capture state
for Qwen3.6 Q6_K PP512. Each projection is `(M,N,K)=(8704,512,5120)` and the
combined attributed share is 38.304% of PP duration. Immutable task fingerprint:
`49cfade573e77d500a701cc8e411867c4e82478e7e770f62274819d73ec475b3`.

## What Was Established

- Gate and up consume the same normalized F32 activation, have distinct Q6_K
  weights, then feed the existing F32 `ggml_swiglu_split` contract.
- Existing graph recognition of gate/up/GLU is real, but its MMVQ backend is
  explicitly N=1-only. It cannot service PP N=512 MMQ.
- The two present MMQ calls independently quantize the shared activation. A
  paired MMQ design may remove duplicate quantization and intermediate traffic.
- Tensor-parallel ownership remains per device; no combined cross-device output
  or all-reduce reorder is authorized.

This is dataflow feasibility, not proof that a monolithic fused kernel is
resource-feasible. Phase 7 measured 229/230 VGPR, 27/30 SGPR, and 57,856 bytes
of dynamic LDS for the current MMQ variants. A naive two-accumulator design
has little obvious headroom and may spill or reduce occupancy.

## Phase 10 Boundary

Only after explicit approval, create one default-off, exact-shape gfx1201
Q6_K shared-Q8_1 paired dispatcher for `(8704,512,5120)`. Quantize the common
activation once, reuse that buffer for the two proven MMQ launches, and retain
the existing F32 GLU path and baseline fallback. It requires a new MMQ-specific
API, not extension of the MMV fusion API.

Do not combine this with a single-kernel projection+SwiGLU experiment. That is
a separate future candidate only if a sub-tiled accumulator and LDS plan fits
gfx1201 without unbounded spill or occupancy loss.

Before any promotion or performance claim, require: both-device exact dispatch
and fallback witnesses; CPU-oracle/parity evidence; static resources plus
dynamic gfx1201 occupancy when feasible; variance-aware same-build timing; and
three normal-graph whole-PP processes. Graph capture/replay and existing
tensor-parallel ownership must remain valid. No registration or E2E claim
without every gate passing.

## Not Done

Phase 9 made no source edit, build, GPU workload, counter capture, or candidate
registration. Do not reopen rejected y64 or prior TG wave-count ideas.
