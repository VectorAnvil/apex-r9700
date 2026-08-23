# Phase 19 Q6_K MTP Small-N Finding

## Result

The proposed N=2-4 Q6_K narrow-DP4A experiment is not authorized. A real
target-backed Qwen3.6 MTP request produced material N=5 work, while N=2-4
accounted for only 2.6467% of summed request GPU-kernel duration and failed
the predeclared materiality threshold on both devices.

This phase made no source optimization, candidate build, performance A/B
claim, or registration change.

## Workload and provenance

The registered server ran the target model itself with `draft-mtp`,
`spec-draft-n-max=4`, tensor split across both gfx1201 devices, and no separate
draft model. One fixed non-streaming request predicted 128 tokens with
temperature zero and seed 1234. Runtime/model evidence confirms one embedded
next-token-prediction layer and execution of the model's `nextn` tensors.

The request drafted 159 tokens and accepted 87, for 54.717% acceptance and a
mean draft length of 3.17. The normal-graph run reported 56.8569 tokens/s; this
is a workload witness, not a baseline/candidate optimization result.

## Exact attribution

The passive host recorder identified logical M/N/K, Q6_K type, operation,
device, and launch sequence. A separate profiler-induced no-graph trace joined
all 43,970 request-window Q6_K dispatches in order, with zero unmatched,
ambiguous, or mismatched rows.

| N | Calls | Duration | Share of summed request GPU kernel time |
|---:|---:|---:|---:|
| 1 | 2,562 | 375.611 ms | 9.5941% |
| 2 | 0 | 0 | 0% |
| 3 | 0 | 0 | 0% |
| 4 | 2,018 | 103.618 ms | 2.6467% |
| 5 | 39,390 | 2,228.050 ms | 56.9103% |

All Q6_K widths used `mul_mat_vec_q`; no request-span hipBLAS, rocBLAS,
or Q6_K MMQ symbol was present. N=2-4 measured 2.4150% of GPU 0 and 2.9181%
of GPU 1 summed kernel duration, below the frozen 5% per-device floor.

## Corrected mechanism

The initial theory expected a fixed 16-column WMMA kernel to waste most of its
work at N=2-4. That is not the registered gfx1201 route. For every N greater
than one, the current Q6_K specialization uses one physical wave32 and DP4A.
The active N=5 symbol has:

| Resource | N=5 value |
|---|---:|
| Workgroup | 32 x 1 x 1 |
| Physical waves | 1 |
| VGPR / SGPR | 50 / 32 |
| LDS / private / spills | 0 / 0 / 0 |
| Global loads | 25 |
| DP4A | 10 |
| ISA instruction lines | 436 |

The N=5 path is nevertheless a strong successor boundary because it owns
56.9103% of the summed request kernel duration. Its largest operation classes
are ffn gate/up/down and attention projections on both devices. Any successor
must improve this existing narrow-MMVQ organization rather than merely switch
the workload to DP4A.

## Graph and communication evidence

The separate registered normal-graph lane recorded 526 begin/end captures,
526 graph instantiations, and 11,242 graph launches. The server reported 38
graph reuses. These counts witness normal graph behavior only.

The no-graph trace attributed `ggml_ar_hip_kernel<float>` shares of 27.4692%
on GPU 0 and 12.3135% on GPU 1. These peer synchronization kernels can overlap
and spin, so their summed dispatch duration is not a wall-time communication
fraction. Achieved occupancy and bandwidth remain unavailable; no static
allocation is substituted for those counters.

## Decision

Close Phase 19 as `complete_no_go_for_n2_to_n4`. Do not build the proposed
N=2-4 candidate for this `draft_n_max=4` workload. Phase 20 may perform a
candidate-free N=5 source/ISA design audit, including explicit propagation of
an MTP verification marker before any MTP-only selector is considered.

## What Didn't Work

- N=2 and N=3 did not occur in the request window; N=4 was only 2.6467%.
- The fixed-width WMMA waste premise did not transfer. N=2-5 already use
  one-wave DP4A MMVQ.
- A shape-only selector cannot be called MTP-only. The HIP backend currently
  receives shape and type but no MTP/context marker.
- The profiler and HIP runtime databases flushed successfully after SIGINT,
  but both injected processes failed to terminate and were killed with return
  code -9. Raw databases and shutdown logs are preserved.
- No achieved occupancy or bandwidth counters were obtained.
- No candidate, benchmark comparison, or registration occurred.

Compact evidence is in `docs/apex-r9700/artifacts/phase-19/`; raw databases,
logs, recorder rows, code objects, and ISA remain in the ignored Phase 19 root.
