# SuperSonic Low-Bit HIP Reference for Phase 3

## Scope

This is a read-only review of `DeanoC/SuperSonic` `main` at commit
`b667b645efce909d5a8fee8a227ad17f0db050af` (2026-08-01). Nothing was built,
ported, or integrated. The reviewed tree was a shallow temporary clone, not a
dependency or fork.

Primary sources:

- [`full_attention_4b.hip`](https://github.com/DeanoC/SuperSonic/blob/b667b645efce909d5a8fee8a227ad17f0db050af/kernels/full_attention_4b.hip)
- [`full_attention_bridge_4b.cpp`](https://github.com/DeanoC/SuperSonic/blob/b667b645efce909d5a8fee8a227ad17f0db050af/kernels/full_attention_bridge_4b.cpp)
- [`prefill_engine.rs`](https://github.com/DeanoC/SuperSonic/blob/b667b645efce909d5a8fee8a227ad17f0db050af/crates/runtime/src/prefill_engine.rs)
- [`prefill_ffi.rs`](https://github.com/DeanoC/SuperSonic/blob/b667b645efce909d5a8fee8a227ad17f0db050af/crates/kernel-ffi/src/prefill_ffi.rs)
- [`Qwen3.6 optimization log`](https://github.com/DeanoC/SuperSonic/blob/b667b645efce909d5a8fee8a227ad17f0db050af/docs/optimization/qwen36-100tok-algorithms-and-optimizations.md)
- [`parity and A/B log`](https://github.com/DeanoC/SuperSonic/blob/b667b645efce909d5a8fee8a227ad17f0db050af/docs/optimization/qwen36-lucebox-parity-log.md)
- [`gfx1201 support matrix`](https://github.com/DeanoC/SuperSonic/blob/b667b645efce909d5a8fee8a227ad17f0db050af/docs/supported-matrix.md)

The pinned tree contains no tracked `LICENSE`, `COPYING`, or `NOTICE` file and
no detected SPDX declaration. Treat it as unlicensed unless upstream provides
clear terms. Apex may study the ideas and experimental method, but must not
copy or port SuperSonic code.

## Compatibility Boundary

The requested kernels are implemented in SuperSonic's Qwen3.5
`full_attention_4b` path and reused by parts of its Qwen3.6 speculative-prefill
work. They are not equivalent implementations in SuperSonic's Qwen3.6 MoE
persistent-kernel path at this revision. Its principal documented workload is
gfx1100, Qwen3.6-27B Q4_K_M with a Q8 draft, and a DFlash verifier batch of
`m=16`. Apex's evidence is ordinary llama.cpp Q6_K PP512 and TG128 on gfx1201,
with TG dispatches operating at the template's `ncols_y=1` path.

SuperSonic performance results are therefore hypothesis evidence only. A
candidate may activate only after Apex establishes the matching llama.cpp
operation, exact shape, types and layouts, phase share, source/dispatch path,
and code-object target on the registered gfx1201 workload.

## Phase 2 Activation Matrix

| SuperSonic reference | Apex Phase 2 evidence | Phase 3 status |
|---|---|---|
| Q6_K MMQ MLP-down | Q6_K `mul_mat_q<14,128,...>` is 85.01% of summed PP dispatch duration, but its operation ownership and M/N/K are not captured. | Conditional. First attribute a significant PP dispatch to MLP-down and capture the exact shape/layout. |
| Exact `m=16` specialization | Apex profiled PP512 and TG1-style kernel families, not SuperSonic's verifier `m=16,n=5120,k=17408` hot shape. | Inactive until an exact recurring llama.cpp shape is measured. Do not bake SuperSonic dimensions into Apex. |
| Paired gate/up plus SwiGLU | Q6_K matrix work and SiLU-family kernels exist, but Phase 2 does not prove sibling gate/up ownership, compatible layouts, or adjacency. | Inactive until correlation/source evidence establishes the pair and the materialization boundary. |
| Q4/Q6 residual-add epilogue | PP has a 2.087% add-broadcast aggregate, but it is not attributed to an adjacent residual after a dominant Q6_K projection. | Inactive until operation attribution and dispatch adjacency prove the removable launch and its share. |
| Fused Q6_K lm-head argmax | No significant lm-head or argmax dispatch was identified. SuperSonic's path returns 16 greedy IDs rather than full logits. | Inactive. It is not the selected TG Q6_K MMV hotspot and is invalid when full logits/top-k are required. |
| gfx12 BF16/i8 WMMA adapter | gfx1201 is relevant, but SuperSonic provides no checked-in gfx1201 HSACO, resource report, or spill evidence for these paths. | Research reference only. Inspect Apex's actual gfx1201 code object and prove layout and numerical parity first. |
| Launch-bound/register experiments | Phase 2 has no joined code-object resource evidence for the selected kernel. | Inactive until VGPR/SGPR/LDS/private/spill and occupancy evidence is captured for the exact candidate. |
| Rollback/parity-gated A/B method | Directly compatible with Apex's Hipfire-derived task/eval and attempt-history plan. | Adapt the method now; this does not activate any SuperSonic kernel idea. |

The generic Q6_K overlap is not enough to promote any implementation. It is
enough to retain the ideas for source-attributed follow-up after the first
evidence-selected Q6_K MMV loop.

## Candidate Concepts

### Specialized Q6_K MMQ MLP-down

SuperSonic dispatches `maybe_matmul_q6_k_mmq_mlp_down` through
`matmul_mmq_q8_1_q6_k` and `matmul_mmq_q8_1_q6_k_device` to
`supersonic_qwen35_matmul_mmq_q8_1_q6_k_kernel` or its gfx12 variant. The path
requires batch 1, raw GGML Q6_K weights, no AWQ/native-int4 side metadata,
`K % 256 == 0`, and i8 WMMA support. BF16 activations are first quantized into
Q8_1 blocks.

Its MMQ tile is M=16, N=128, K=256 with 256 threads and
`__launch_bounds__(256, 2)`. A hot exact route removes bounds checks and
redundant scale loads only for M=16 and N divisible by 128. gfx12-native
dispatch is separately capability- and rollback-gated.

Useful Apex adaptation: after PP operation attribution, compare the actual
llama.cpp Q6_K block decode, activation representation, tile, and matrix
orientation. Specialize only a measured exact shape. Do not assume the
documented DFlash `m=16,n=5120,k=17408` shape or its gfx1100 speedup applies.

### Paired Gate/Up and SwiGLU

SuperSonic has both a paired quantized matmul and a paired matmul with a SwiGLU
epilogue. The route requires batch 1, M=16, each output width divisible by 16,
K divisible by 256, matching raw GGML quantization for both weights, and no
incompatible side metadata. Its important numerical rule is to round the gate
and up accumulators to BF16 before evaluating `silu(gate) * up`, matching the
unfused materialization semantics.

The repository's history is deliberately mixed: one fixed-qtype paired route
survived same-build parity and full-suite A/B, while another fused GGML attempt
regressed sharply and was rejected because synchronization/timing behavior
changed. Preserve both outcomes. Fusion is not a result; it is a candidate
whose launch removal, dependency boundary, and end-to-end behavior must be
measured.

Apex activation requires two source-attributed gate/up dispatches with
compatible shapes/types/layouts, an adjacent SwiGLU boundary, meaningful
combined phase share, and an oracle capable of checking the precise rounding
contract.

### Q4/Q6 Residual-Add Epilogues

SuperSonic folds a residual addition into exact-M=16 low-bit projection
epilogues. Q4-family paths use the templated GGML dequant-WMMA kernel with an
`ADD_RESIDUAL` mode; Q6_K uses a residual form of its Q8_1-by-Q6_K MMQ path.
Its parity log records byte-exact Q6_K residual output against MMQ followed by
the separate element add.

For Apex, the optimization becomes eligible only when correlation evidence
shows that a specific Q4/Q6 projection is immediately followed by the
compatible residual launch and quantifies the removable time. The current PP
add-broadcast percentage does not establish that relationship.

### Fused Q6_K LM-Head Argmax

SuperSonic's `maybe_matmul_q6_k_lm_head_argmax` route launches a per-tile Q6_K
M=16 argmax kernel followed by a 256-thread per-row reduction. It requires
batch 1, M=16, N divisible by 16, K divisible by 256, raw Q6_K weights, and no
AWQ side scale. It returns 16 greedy token IDs and deliberately bypasses full
logit materialization; the full-logit route remains necessary for top-k or
probing.

This is high risk and currently inactive. SuperSonic's own history includes
lm-head experiments rejected for token/round-count changes and only a small
suite contribution after promotion. Apex must first prove an lm-head plus
argmax hotspot, greedy-only semantics, exact vocabulary/K shape, and token
parity. It must never be substituted for the Phase 3 TG M=1 Q6_K MMV target by
name similarity.

## gfx12 WMMA and Compiler Evidence

SuperSonic implements source-level gfx1200/gfx1201 adapters for the gfx12 BF16
and i8 WMMA builtins. The BF16 adapter remaps wave32 lane fragments before
calling `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12`; the i8 adapter
repackages four source words into the two-word gfx12 operands before calling
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`. Runtime dispatch checks
the detected device family and keeps separate disable switches.

This is a useful warning about operand mapping, not a portable adapter for
llama.cpp. SuperSonic describes gfx1201 as a narrow bring-up lane and does not
provide checked-in gfx1201 disassembly or VGPR/SGPR/LDS/private-segment/spill
evidence for these kernels. Before considering an Apex adapter:

1. dynamically verify the GPU, compiler, builtin availability, wave size, and
   loaded code-object target;
2. prove that llama.cpp's fragments, Q6_K block layout, scale handling, and
   accumulation semantics match the proposed mapping;
3. inspect the exact HSACO symbol, ISA, registers, LDS, private segment, and
   spills with the Phase 3 argv-only inspector; and
4. require independent numerical parity before timing.

SuperSonic also records a narrow launch-bound success: changing one generic
M=16 qtype path from `(32,8)` to `(32,4)` improved its gfx1100 suite, while the
same experiment was flat or worse for fused gate/up and Q6_K MMQ and was
reverted. Apex should change launch bounds only for one exact gfx1201 kernel
after resource and occupancy evidence. Lower occupancy, lower register count,
or fewer resident blocks are not performance conclusions by themselves.

## Parity, Rollback, and A/B Contract

The immediately reusable part is the experiment discipline:

1. Keep every candidate disabled by default or behind a narrow rollback
   switch so the same build can run baseline and candidate paths.
2. Freeze model, prompt/input, seed, generation limits, environment, compiler,
   GPU assignment, harness, oracle, tolerances, and measurement procedure.
3. For fusion/epilogue work, first compare low-level output against the exact
   unfused composition, including intermediate BF16 rounding where required.
4. Run normalized end-to-end output/token parity before performance. Derive
   thresholds and budgets from an accepted reference, never the candidate.
5. Use warmup, cooldown, raw samples, and interleaved A/B/B/A or A/B/A runs.
   Report median, variance, uncertainty, and prompt-level results. An unstable
   run cannot win or establish a durable no-win conclusion.
6. Promote only on stable correctness-valid full-suite improvement. Keep an
   immediate rollback path until Phase 4 promotion is complete.
7. Record full Git SHA, dirty paths, diff SHA-256, build command, binary and
   code-object hashes, toolchain/GPU identity, raw samples, parity output, and
   rejection reason.
8. Append every result to prior-attempt history. Demote a failed idea only
   when architecture, source/build, exact kernel, shape/types/layouts,
   harness, inputs, toolchain, and candidate fingerprint match. Preserve
   compile, parity, performance, and instability failures as distinct states.

This extends the Hipfire-derived `suggest -> task -> eval` contract. A
SuperSonic-inspired suggestion must name the exact Apex evidence that activates
it, the removable launches or instruction/resource hypothesis, allowed files,
rollback control, parity contract, and rejection boundary.

## Phase 3 Use

The first Phase 3 target remains llama.cpp's TG
`mul_mat_vec_q<(ggml_type)14,1,false/true,false>` family. SuperSonic does not
justify changing that selection.

After the first loop, the order for evaluating a SuperSonic-inspired idea is:

1. enrich rocprofv3 evidence with exact operation ownership, shape/types, and
   dispatch adjacency under normal workload conditions;
2. create an immutable, evidence-linked task/eval contract and rollback path;
3. capture the exact gfx1201 code-object resources and ISA when the hypothesis
   involves WMMA, launch bounds, or register pressure;
4. implement only one independently derived candidate in an isolated pinned
   llama.cpp worktree; and
5. run low-level parity, end-to-end parity, and interleaved variance-aware A/B
   before retaining or rejecting it.

Until those gates pass, preserve the reference and do not port it.
