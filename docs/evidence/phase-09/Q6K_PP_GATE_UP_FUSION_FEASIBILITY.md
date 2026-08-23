# Q6_K PP Gate/Up Fusion Feasibility

## Decision

**Feasible for a later candidate contract; not implemented.** Phase 9
established that the exact Q6_K PP512 `ffn_gate` and `ffn_up` pair has a
source-level path to a paired MMQ plus SwiGLU design. The immutable task
fingerprint is
`49cfade573e77d500a701cc8e411867c4e82478e7e770f62274819d73ec475b3`;
the feasibility finalizer returned `feasible_candidate_contract` with no
missing or unsupported findings.

This is not a kernel, benchmark, or performance result. It authorizes only a
future, separately approved immutable candidate contract.

## Exact Evidence

- Registered workload: Qwen3.6 Q6_K PP512, normal graphs, per-device logical
  shape `M=8704,N=512,K=5120`.
- `ffn_gate` and `ffn_up` account for 955,179,820 ns and 959,724,245 ns of
  summed Phase 2 PP duration: 1,914,904,065 / 4,999,245,044 = 38.3039%.
- The positional proof found 64 exact adjacent gate/up pairs per device and
  capture state, with six captures per device. Each pair uses the same
  normalized F32 RHS and `Q8_1_MMQ` layout, with matching `32x8` blocks and
  `68x4` grids.
- Gate and up retain separate Q6_K tensors and per-device tensor-parallel
  ownership. A later implementation must preserve downstream all-reduce order.

## Design Boundary

The existing GGML fusion recognition is useful provenance but is not reusable:
its MMV path requires destination `N=1`, whereas this PP MMQ pair is `N=512`.
A later candidate therefore needs a new, default-off MMQ-specific paired
projection/dispatcher with one activation quantization, two Q6_K weight
streams, and the pinned F32 `ggml_swiglu_split` result contract. It must remain
normal-graph capture/replay safe and provide a rollback to the current separate
dispatches.

The modeled intermediate upper bound is 71,303,168 bytes per layer/device. It
is a metadata-derived model, not a measured bandwidth or speedup claim. Exact
trace placement of the repeated quantization remains unavailable; source proves
independent calls only.

The current MMQ resource footprint is the main implementation risk: Phase 7
recorded 229/230 VGPR, 27/30 SGPR, and 57,856 bytes of dynamic LDS. Phase 9 did
not establish that two projection accumulators fit concurrently. A future
static gate should compare a lower-risk shared-Q8_1 paired dispatcher against a
sub-tiled monolithic design and reject the latter before GPU work if it implies
spills or an unacceptable occupancy bound.

## Required Future Gates

A future candidate needs a new task and must establish dual-GPU CPU-oracle
parity, exact paired dispatch selection, static code-object and dynamic gfx1201
resource evidence, same-build three-round interleaved timings without trimming,
and three normal-graph PP processes for the whole-workload gate. Achieved
bandwidth may be reported only from an exact reliable runtime counter; otherwise
it remains unavailable.

## Not Done

Phase 9 made no source edit, build, GPU workload, graph execution, kernel
integration, candidate registration, or performance claim. SuperSonic and Zinc
remain methodology references only.

## Provenance

The result-local root
`results_phase09_gate_up_fusion_feasibility_20260808/` retains the immutable
task, findings, pair proof, source feasibility record, source hashes, and
final result. The task links the raw Phase 2 PP trace, Phase 7 baseline and
attribution, Phase 8 static result, pinned source commit
`259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`, and registered P2P patch hash.
