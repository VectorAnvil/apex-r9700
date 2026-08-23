# Phase 9 Gate/Up Fusion Feasibility

## Source And Workload Facts

Pinned source is `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`. Phase 7's exact
PP join identifies Qwen3.6 Q6_K gate and up as separate material MMQ groups:
each is logical `(M,N,K) = (8704,512,5120)`, has 768 dual-device calls, and
takes 955,179,820 ns (gate) or 959,724,245 ns (up).

`src/models/qwen3.cpp:34-36` creates independent `ffn_gate`, `ffn_up`, and
`ffn_down` tensors. Each gate/up tensor is laid out `[n_embd, n_ff]`; the
observed model values are K=5120 and M=8704. `qwen3.cpp:104-109` normalizes
the common FFN input and calls `build_ffn` with `LLM_FFN_SILU` and
`LLM_FFN_PAR`.

`src/llama-graph.cpp:1585-1608` first builds `up * cur`, then independently
builds `gate * cur` for the parallel gate. `llama-graph.cpp:1649-1652` emits
`ggml_swiglu_split(gate, up)`: SiLU is applied to gate and multiplied by up.
Both projections therefore read the same F32 normalized activation, but have
separate Q6_K weight tensors and outputs before GLU.

## Existing Fusion Boundary

The executor recognizes adjacent `MUL_MAT, MUL_MAT, GLU` subgraphs at
`ggml-cuda.cu:3530-3557` and can pass the gate weight as fusion data to MMVQ.
The fused MMV admission check at `ggml-cuda.cu:1702-1728` rejects ordinary
`MUL_MAT` unless destination N is exactly one. `mmvq.cu:793-823` independently
asserts that fusion is only supported for `c_ncols_dst == 1`. Thus it is a
decode/vector implementation, not a reusable N=512 MMQ implementation.

For PP, `ggml-cuda.cu:1739-1767` selects MMQ after MMVQ is not selected;
Q6_K's MMQ path quantizes the F32 activation to Q8_1 inside each independent
`ggml_cuda_mul_mat_q` call (`mmq.cu:114-144`). The current pair therefore
quantizes and reads the shared activation twice.

## Tensor Parallel And Graphs

The Phase 7 evidence records 768 dual-device gate calls and 768 dual-device
up calls. This is consistent with independent per-device tensor-parallel
shards; a fused candidate must retain the existing per-device output ownership
and downstream all-reduce order. It cannot assume one combined cross-device
output buffer.

Normal HIP graph execution is part of the frozen workload. The CUDA-named
abstraction is used by the HIP build: graph update/instantiate occurs at
`ggml-cuda.cu:2484-2502`, and graph launch at `3964`. Any candidate must be
graph-capture safe, have stable allocations/launch geometry, and be default
off until parity and graph replay are proven.

## Conclusion

The graph adjacency and shared activation make a paired Q6_K MMQ + SwiGLU
candidate plausible. It cannot reuse the existing `MUL_MAT_VEC` fusion API:
that API is deliberately N=1-only and uses MMVQ launch/accumulator semantics.
A Phase 9 candidate would require an MMQ-specific paired-projection kernel or
paired dispatcher, one activation quantization, two Q6_K weight streams, and a
SwiGLU output contract. It must first be shape-gated to the observed
`K=5120,N=512,M=8704`, preserve separate weights and tensor-parallel behavior,
and be validated outside this source-only phase.

No source edit, build, GPU workload, graph execution, or candidate claim was
made in this phase.
