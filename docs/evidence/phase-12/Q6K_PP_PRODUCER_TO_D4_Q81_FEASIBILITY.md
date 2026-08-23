# Q6_K PP Producer-to-D4-Q8_1 Feasibility

## Finding

**Feasible for a new candidate contract; no implementation or speedup claim.**
The selected boundary is the exact PP512 dense-FFN chain
`SwiGLU -> block_q8_1_mmq(D4) -> ffn_down` at logical MMQ shape
`M=5120,N=512,K=8704`.

Phase 12 inspected frozen source and existing traces only. It changed no
llama.cpp source, built no candidate, and ran no GPU or performance workload.

## Exact Trace Join

The Phase 7 ordered Q6_K MMQ host sequence was joined to the Phase 2 raw PP
trace on both physical devices. All six repeats and 64 layers per repeat have
the same adjacent sequence:

1. `unary_gated_op_kernel<op_silu,float>` produces F32 `ffn_swiglu-*`.
2. `quantize_mmq_q8_1<layout0>` writes the Q6_K MMQ D4 activation buffer.
3. `mul_mat_q<GGML_TYPE_Q6_K,128,false>` executes `ffn_down`.

The join covers 384 complete instances per device with zero ambiguity or
unmatched rows. Across both devices, the observed components are:

| Component | Calls | Time | Share of summed PP time |
|---|---:|---:|---:|
| F32 SwiGLU producer | 768 | 55.514 ms | 1.110% |
| D4 Q8_1 MMQ quantizer | 768 | 14.957 ms | 0.299% |
| Q6_K `ffn_down` MMQ | 768 | 919.979 ms | 18.402% |
| Observed chain | 768 | 990.450 ms | 19.812% |

Producer and quantizer time is context, not wholly removable time. The chain
share is not a predicted candidate speedup.

## Actual Layout

PP512 does not use the ordinary MMVQ `block_q8_1` layout. Q6_K MMQ selects
`MMQ_Q8_1_DS_LAYOUT_D4`: each 144-byte `block_q8_1_mmq` contains 128 int8
values and four F32 scales, transposed for contiguous shared-memory loads.

Today `ggml_cuda_mul_mat_q` asserts an F32 RHS, allocates a backend-private
pool buffer, launches the D4 quantizer, calls MMQ, and releases the buffer on
return. A future candidate therefore needs both:

- an exact-shape fused SwiGLU-plus-D4-packing producer with 128-value amax
  reductions and baseline rounding/scale behavior;
- a private MMQ entrypoint that consumes the prequantized buffer without
  launching its own quantizer.

Changing the graph tensor to ordinary Q8_1 or redirecting the current SwiGLU
output pointer is not valid.

## Materialization Model

The F32 `ffn_swiglu` tensor is contiguous `[8704,512]`, or 17,825,792 bytes
per layer per device. It has one consumer. Direct D4 production can logically
avoid one full F32 write and one full F32 quantizer read: 35,651,584 bytes per
layer per device. It must still write 5,013,504 bytes of D4 Q8 data, excluding
the current small allocation guard.

These are logical write/read bytes, not cache traffic or achieved bandwidth.
The model does not assume that every byte reaches VRAM or that its cost is
fully removable.

## Unselected Boundary

`RMSNorm+scale -> shared D4 Q8_1 -> gate/up` is source- and trace-feasible and
connects to 38.304% of summed PP time. Its F32 tensor is `[5120,512]`; direct
production could logically avoid 31,457,280 F32 write/read bytes per layer per
device. It ranks second because it has less modeled staging traffic, two
consumers, a harder whole-row RMS reduction plus 128-value packing problem,
and a closely related negative history.

Phase 10 already shared one D4 quantization across gate/up but retained the
F32 producer. That candidate was flat at `0.999896x` exact-pair and
`1.003627x` whole-PP. RMSNorm-direct remains different, but the launch-only
subcase must not be retried.

## WHAT_DIDNT_WORK

Neither boundary is a simple Q8_1 graph-output change. The existing consumer
requires a private, transposed MMQ D4 format and owns its scratch allocation.
Both direct producers require a new fusion boundary and reduction/packing
kernel design.

The larger gate/up consumer share did not override the predeclared ranking.
Its lower modeled staging bytes, two-consumer lifetime, more complex reduction
structure, and Phase 10 partial negative place it behind the uniquely joined
SwiGLU/down chain. No benchmark was run and no performance result may be
inferred from the selected rank.

## Provenance

- Immutable task fingerprint:
  `8da43026d4a90c0fbd78470e733b2dfe10542fa539a95032e275a81036b65aeb`.
- Registered llama.cpp commit:
  `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`.
- Phase 2 PP trace SHA-256:
  `8563e42c0964c572530a170c05b13262eccee0483b68d8fd2bea6136e7c913bd`.
- Phase 7 host attribution SHA-256:
  `2ba146c852a193388784ac50a12b528a0764b9fefa9b7f0907374c5a1088320b`.
- Result-local evidence:
  `results_phase12_producer_q81_feasibility_20260808/`.
