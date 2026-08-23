# Phase 10 source plan: shared Q8_1 for the Q6_K gate/up pair

## Scope and conclusion

This plan is for a deliberately narrow HIP/gfx1201 experiment at the registered
Qwen3.6 Q6_K PP shape:

```
weight (each): Q6_K [K=5120, M=8704]
activation:    F32  [K=5120, N=512]
output (each): F32  [M=8704, N=512]
operation:     gate = W_gate * x; up = W_up * x; out = silu(gate) * up
```

The minimal viable change is **not** a dual-accumulator or SwiGLU kernel. It
is a backend-only pair dispatch which:

1. quantizes the common F32 activation once into one temporary Q8_1-MMQ buffer;
2. invokes the existing Q6_K MMQ implementation twice against that buffer;
3. preserves the two existing F32 projection tensors; and
4. leaves the existing F32 `GGML_OP_GLU` node to run normally.

It removes exactly one activation quantization launch and its duplicate Q8_1
write. The two existing MMQ launches, their output stores, the GLU launch, and
all tensor-parallel ownership/all-reduce behavior remain unchanged. It is a
reasonable Phase 10 implementation candidate, but performance is unproven.

## Source evidence

Registered source: `/home/adam/workspaces/ChatGPT/llama-lab/experiments/llama-hip-p2p-25197`
at `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`.

Relevant implementation facts:

- `ggml/src/ggml-cuda/mmq.cu:77-150` implements the ordinary non-ID MMQ path.
  It allocates `src1_q8_1` from `ctx.pool()`, calls
  `quantize_mmq_q8_1_cuda`, then constructs `mmq_args` and calls
  `ggml_cuda_mul_mat_q_switch_type`.
- `mmq.cu:130-144` proves this occurs separately for every ordinary
  `ggml_cuda_mul_mat_q` call. The Q8_1 buffer is therefore the correct
  sharing boundary.
- `quantize.cu:392-426` makes the quantized layout depend on
  `mmq_get_q8_1_ds_layout(type_src0)`. Sharing is valid here only when both
  weights are the same MMQ source type, specifically Q6_K.
- `mmq.cu:77-125` shows the existing output type is F32 and the normal
  non-ID API has no persistent Q8 input parameter. A small internal helper is
  needed; calling `ggml_cuda_mul_mat_q` twice cannot reuse its local buffer.
- `ggml-cuda.cu:3038-3750` is the graph-time fusion dispatch. It can launch
  work for a node and return the number of following nodes to skip.
- `ggml-cuda.cu:3522-3557` recognizes adjacent `MUL_MAT, MUL_MAT, GLU` but
  only invokes the existing MMV fusion after the `N == 1` restriction in
  `ggml_cuda_should_fuse_mul_mat_vec_q` (`1702-1728`). It deliberately does
  not apply to this `N=512` MMQ workload.
- `ggml-cuda.cu:2818-2862` and `ggml-impl.h:714-741` provide existing
  topological/edge/alias checks for the exact gate/up/GLU graph shape.
- `common.cuh:1151-1191` defines `ggml_cuda_pool_alloc`; `ggml-cuda.cu:361-635`
  provides the legacy and VMM pool behavior. Existing transient allocations
  are already used during graph capture, so the new buffer must follow that
  established local RAII lifetime rather than introduce a persistent cache.
- `ggml-cuda.cu:2434-2523, 3925-3972` records the normal HIP/CUDA graph capture
  and replay path. A same-stream sequence of quantize -> gate MMQ -> up MMQ
  captures naturally as three nodes.
- `common.cuh:78-90` represents AMD architecture dynamically and identifies
  RDNA4 with `GGML_CUDA_CC_IS_RDNA4(cc)`. The selector must use that runtime
  value, not assume Instinct/CDNA behavior.

## Minimal backend implementation

### 1. Factor the existing ordinary-MMQ body

In `ggml/src/ggml-cuda/mmq.cu`, keep `ggml_cuda_mul_mat_q` as the generic
baseline entry point. Factor only its non-ID (`ids == nullptr`) body into two
internal helpers:

```
prepare_q8_1_mmq(ctx, src0, src1, q8_alloc, q8_view)
launch_mmq_from_q8_1(ctx, src0, src1, q8_view, dst)
```

`prepare_q8_1_mmq` should retain the baseline's:

- `ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING)`;
- byte count including `get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq)`;
- source strides (`s11`, `s12`, `s13`);
- Q6_K-selected `quantize_mmq_q8_1_cuda` call; and
- `CUDA_CHECK(cudaGetLastError())` after the quantize launch.

`launch_mmq_from_q8_1` should retain the baseline's Q8 stride calculation,
`mmq_args` construction, `use_stream_k` policy, and
`ggml_cuda_mul_mat_q_switch_type(ctx, args, stream)` call. It must take each
weight tensor and destination tensor independently, so their own source and
destination strides are not silently assumed identical.

Do not generalize this helper to native FP4, `MUL_MAT_ID`, or a cross-call
cache. The candidate only needs ordinary Q6_K `MUL_MAT` at one exact shape.

### 2. Add one narrow pair API

Declare in `ggml/src/ggml-cuda/mmq.cuh` and define in `mmq.cu` an internal
backend API such as:

```
bool ggml_cuda_mul_mat_q_pair_shared_q8(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * gate_weight, const ggml_tensor * up_weight,
    const ggml_tensor * common_input,
    ggml_tensor * gate_dst, ggml_tensor * up_dst);
```

The helper returns `false` without launches unless all preconditions hold. On
success it owns one local `ggml_cuda_pool_alloc<char>` through this sequence:

```
quantize(common_input, q8);
launch_mmq_from_q8_1(gate_weight, common_input, q8, gate_dst);
launch_mmq_from_q8_1(up_weight,   common_input, q8, up_dst);
return true;
```

Retain the existing `ggml_cuda_mul_mat_q` implementation for every baseline
call and all failed pair checks. No global state, graph-owned allocation, new
tensor type, or public ggml API is necessary.

### 3. Graph/backend recognition, not graph rewriting

At the beginning of `ggml_cuda_try_fuse` in `ggml-cuda.cu`, before the current
MMV gate/up fusion block, add a focused recognizer for three consecutive nodes:

```
MUL_MAT, MUL_MAT, GLU
```

It should identify `gate` and `up` first from `glu->src[0]` and `glu->src[1]`,
not infer ownership from node order or names. The Phase 9 recorder observed
gate then up, while high-level graph construction can list independent nodes in
another order. After deriving roles from the GLU edges, the exact experimental
selector may additionally require the registered tensor-name pattern. The
recognizer may use the existing
`ggml_cuda_can_fuse(cgraph, i, { MUL_MAT, MUL_MAT, GLU }, {})` as its
conservative graph/alias guard, then independently validate the exact pairing
below. If it launches the pair helper successfully, it must return **1**:

- node `i` is executed by the pair helper;
- node `i + 1` is skipped, preventing a second quantize/MMQ launch; and
- node `i + 2` (`GGML_OP_GLU`) executes through ordinary backend dispatch.

Returning `2` would incorrectly skip SwiGLU. Returning `1` retains both F32
projection outputs as real tensors for the normal GLU node and any sanctioned
downstream consumers.

### 4. Exact selector contract

The recognizer and pair API should both require all of the following. Any
failure returns zero from the recognizer and executes the unchanged baseline.

- Explicit opt-in: `GGML_CUDA_Q6K_SHARED_Q8_GATE_UP=1`; absent, empty, or any
  other value is off. `GGML_CUDA_DISABLE_FUSION` remains a master rollback.
- HIP build and exact dynamic architecture: `GGML_USE_HIP` and runtime CC equal
  to `GGML_CUDA_CC_RDNA4 + 1`, which the backend parser derives from `gfx1201`.
- Ordinary `GGML_OP_MUL_MAT` only; no `MUL_MAT_ID`, IDs, biases, scales,
  views, or GLU-swapped mode. Require `ggml_get_glu_op(glu) ==
  GGML_GLU_OP_SWIGLU` and the un-swapped parameter.
- The two matmuls' RHS pointer is identical (`gate->src[1] == up->src[1]`),
  which is stronger than matching layouts and proves a common activation.
- After roles are established from GLU edges, require weight names matching
  `blk.<layer>.ffn_gate.weight` and `blk.<layer>.ffn_up.weight`, plus common
  activation `attn_post_norm-<layer>`. The focused test harness must assign the
  same names to its exact case; nonmatching or unnamed graphs fall back.
- Both weights are `GGML_TYPE_Q6_K`; common input and both destinations are
  `GGML_TYPE_F32`; normal MMQ selection remains true dynamically.
- Exact logical shapes and no batching: each weight `ne=[5120,8704,1,1]`,
  common input `ne=[5120,512,1,1]`, and each result `ne=[8704,512,1,1]`.
  Check expected contiguous/layout/stride requirements explicitly rather than
  using the trace's launch geometry as a proxy.
- `gate` and `up` are exactly the first two nodes' outputs and feed the GLU in
  correct gate/up operand order. Reject all other consumers/layouts unless
  later evidence deliberately broadens the contract.

The runtime CC check is intentionally exact gfx1201, not a generic RDNA4 gate
or an assumed Instinct counter/occupancy model. Existing MMQ code maps the HIP
architecture string dynamically and selects its WMMA path through backend
capability logic.

## Pool lifetime and HIP graph capture

`ggml_cuda_pool_alloc` releases the allocation when the helper scope ends.
That is correct only because all three launches use the same stream and the
Q8 buffer is consumed before the scope ends. It exactly follows the lifetime
of the present single-MMQ path.

- During ordinary execution: stream order makes both MMQ reads happen after
  quantization and before any later stream work may reuse the cached pool
  allocation.
- During capture: one quantization node and two existing MMQ nodes capture in
  order with the same address. The pool's existing legacy/VMM behavior already
  supports transient launch buffers captured by the current MMQ path. Do not
  add a separate persistent allocation or synchronize inside the helper.
- Do not allocate another pool buffer between quantization and the second MMQ
  launch. Keep the sole RAII allocator live across both launches.
- The candidate must be tested both with normal graph capture/replay and with
  `GGML_CUDA_DISABLE_GRAPHS=1`. Correctness must not rely on accidental pool
  address stability across graph updates.

There is no cross-device buffer sharing: `ggml_cuda_try_fuse` runs in each
backend context, so each tensor-parallel device allocates, quantizes, and
consumes its own local activation shard. Existing output tensor ownership and
subsequent all-reduce ordering remain untouched.

## Numerical contract and outputs

The pair API changes only duplicated preparation of the same F32 RHS. Each
existing Q6_K MMQ retains its current quantized RHS byte representation and
its existing kernel. The projected gate/up outputs remain F32. The original
F32 `GGML_OP_GLU` performs `silu(gate) * up` unchanged.

Consequently, the intended result is bit-identical to the baseline for the
two projection tensors and the final GLU output. This is a test requirement,
not an assumption: Phase 10 must compare candidate output to baseline and
the established CPU/oracle path at the exact workload, with no BF16 rounding
or SuperSonic-style epilogue introduced.

The selector must never expose the temporary Q8_1 buffer as a `ggml_tensor`.
It is backend-private scratch; only the pre-existing F32 `gate_dst` and
`up_dst` tensors are written and subsequently read by GLU.

## Static feasibility and risks

Feasible:

- Q6_K uses the Q8_1 MMQ quantizer, and both exact projections have the same
  source type and identical input pointer/shape.
- The current backend already has the needed dispatch primitive
  (`ggml_cuda_mul_mat_q_switch_type`) once a Q8 buffer is supplied.
- The graph execution loop supports running two nodes from one fusion hook and
  skipping exactly one subsequent node.
- The plan adds no MMQ register, LDS, workgroup, or wave-count pressure because
  it launches unchanged kernels.

Risks to gate and measure:

- The duplicate quantization may be a small fraction of the pair's time;
  removing it can yield no material PP improvement despite a correct result.
- The current generic graph fusion predicate treats the full three-node chain
  as elidable through GLU. The new recognizer must preserve the GLU node and
  should add dedicated tests for node order, source edges, external uses, and
  aliasing rather than blindly reuse a predicate meant for full fusion.
- Pairing must remain local to one device/context. Do not infer a shared
  activation buffer across the tensor-parallel devices from matching shapes.
- `src0` padding-clear behavior in the ordinary path must be retained for both
  weights if the helper is ever broadened beyond persistent model weights.

## Rollback, witness, and Phase 10 validation

Default off is the rollback mechanism. The candidate is active only with
`GGML_CUDA_Q6K_SHARED_Q8_GATE_UP=1`; unset it (or set any value other than
`1`) to return to the two independent baseline calls without recompiling.
`GGML_CUDA_DISABLE_FUSION=1` is an additional, already-established rollback.

Add a concise debug-only or opt-in dispatch witness that records, per device,
the dynamic CC/RDNA4 decision, matched layer, exact dimensions, one quantize
launch, two Q6_K MMQ launches, and retained F32 GLU. Do not add always-on
per-layer logging to performance runs.

Before accepting a result, Phase 10 needs:

1. focused graph-recognizer tests, including positive exact pair and rejected
   input/type/shape/order/GLU variants;
2. a clean HIP build and GPU smoke test with the selector on/off;
3. exact-output parity (projection and final result when observable), normal
   graph replay, and graphs-disabled parity;
4. same-build isolated pair timing across three stable rounds, reporting the
   duplicate quantization time separately if attribution permits; and
5. normal graph, three-process whole-PP comparison against the registered
   baseline, with an explicit unstable/no-gain rejection path.

No source or build was changed while producing this plan.

## Provenance hashes

```
259f2e2a531af9ed3efa7f66adaa5eb5b53da95f  source commit
550b1f1baf1db0e349b0974754e4810bfb90fc9e7c97873622daf06f02e8a63c  ggml/src/ggml-cuda/mmq.cu
6d153a9d6f293a4ff5f11e7886a48bf765b21d74075d73b2097a2b2a9149de6f  ggml/src/ggml-cuda/mmq.cuh
7e80dfe33ecb54cfde58782b0982a6b25955ddecefb6f63352fa349f1a6ccda9  ggml/src/ggml-cuda/ggml-cuda.cu
a977b50f7479df092bf3c441ba88e4519803b5f0df5a640fd7fb0874e64b9b4c  ggml/src/ggml-cuda/common.cuh
838cff1fd45ab483f3f86d24f23d997833b4af7a0b945a80f4da0e055def6565  ggml/src/ggml-cuda/quantize.cu
017f8592498246d68f8b66e35b7d0de2d1d315440ebaeeaffdcaa8477d02fd8f  ggml/src/ggml-cuda/quantize.cuh
2ed56e264202906d107e26d08eabb242d3107b026ebfb78096fa1e5f94bdbbb8  ggml/src/ggml-impl.h
e427d8c6767d2e0900689d4747177c3acc756b6de1dbfbb4df90ee9c49a1904d  ggml/src/ggml.c
```
