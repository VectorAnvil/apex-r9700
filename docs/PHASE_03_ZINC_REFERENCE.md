# Zinc RDNA4 K-Parallel Reference for Phase 3

## Scope

This is a read-only review of [`fusion44/zinc`](https://github.com/fusion44/zinc)
at commit [`8c6b5988ac1335b0ea156f575f453bc1891b3362`](https://github.com/fusion44/zinc/tree/8c6b5988ac1335b0ea156f575f453bc1891b3362)
(2026-05-19). Zinc is MIT licensed. Nothing was built, run, ported, or
integrated into Apex or llama.cpp.

Primary sources:

- [`dmmv_q6k_batch_kpar.comp`](https://github.com/fusion44/zinc/blob/8c6b5988ac1335b0ea156f575f453bc1891b3362/src/shaders/dmmv_q6k_batch_kpar.comp)
- [`dmmv_q4k_batch_kpar.comp`](https://github.com/fusion44/zinc/blob/8c6b5988ac1335b0ea156f575f453bc1891b3362/src/shaders/dmmv_q4k_batch_kpar.comp)
- [`dmmv.zig`](https://github.com/fusion44/zinc/blob/8c6b5988ac1335b0ea156f575f453bc1891b3362/src/compute/dmmv.zig)
- [`forward.zig`](https://github.com/fusion44/zinc/blob/8c6b5988ac1335b0ea156f575f453bc1891b3362/src/compute/forward.zig)
- [`RDNA4_BATCHED_PREFILL_2X.md`](https://github.com/fusion44/zinc/blob/8c6b5988ac1335b0ea156f575f453bc1891b3362/docs/RDNA4_BATCHED_PREFILL_2X.md)
- [`RDNA4_PERFORMANCE_JOURNEY.md`](https://github.com/fusion44/zinc/blob/8c6b5988ac1335b0ea156f575f453bc1891b3362/docs/RDNA4_PERFORMANCE_JOURNEY.md)

## Verdict

Zinc is a strong Phase 3 reference for a batched Q6_K projection on gfx1201,
but it is not a drop-in implementation for llama.cpp's HIP MMQ path.

Phase 2 already answers the family-level hotspot question: the two PP
`mul_mat_q<(ggml_type)14,128,...>` variants account for **85.011%** of summed
PP dispatch duration across both R9700 agents. Therefore custom GGML Q6_K
matrix work is an actual PP hotspot, not a speculative target.

That evidence does not establish that Zinc's operation corresponds to the hot
llama.cpp dispatch. Zinc implements a Vulkan GLSL batched matrix-vector path in
which columns are accumulated in one workgroup, while the observed llama.cpp
kernel is a HIP/CUDA MMQ template. Phase 2 does not yet join the PP symbols to
an exact projection, M/N/K, activation layout, dispatch call, or loaded code
object. A port remains blocked until Phase 3 proves those facts and compares
Zinc's strategy with the existing llama.cpp MMQ algorithm.

The first Phase 3 target remains the evidence-selected TG Q6_K MMV family at
70.903% of summed TG dispatch duration. Zinc is a conditional PP follow-up,
not authority to switch targets or optimize two families at once.

One Phase 2 PP trace row for the non-fused Q6_K MMQ variant records a `32x8x1`
workgroup, `1280x32x1` grid, zero LDS, 232 VGPRs, and 128 SGPRs. These are
profiler-reported launch/resource facts for that dispatch, not logical M/N/K or
operation ownership. They already contrast with Zinc's 64-thread one-row
scheme and make source/code-object inspection a prerequisite to any strategy
comparison. The Phase 2 run also disabled HIP graphs for profiler stability, so
its throughput is diagnostic and cannot be reused as a Phase 3 baseline.

## Activation Matrix

| Reference | Current Apex evidence | Decision |
|---|---|---|
| Q6_K batched K-parallel DMMV | PP Q6_K MMQ is 85.011%, but its exact operation and shape are unresolved. | High-value PP reference. Activate only after exact source, operation, shape, layout, launch, and code-object attribution. |
| Q4_K batched K-parallel DMMV | The registered workload is Q6_K and no Q4_K PP hotspot was selected. | Inactive for this workload. Preserve only as the source of the measured column-width sweep and shared algorithm pattern. |
| Wave64 one-row workgroup | Zinc requests wave64 on detected wave64 devices and retains a cross-subgroup fallback. | Hypothesis only. Query the actual gfx1201 HIP wavefront and compiled kernel metadata; do not hard-code wave64 from the GPU name. |
| `MAX_COLS=40` | Zinc reports a Q4_K R9700/Qwen3-8B sweep; Q6_K mirrors the constant without its own checked-in sweep. | Do not transfer 40 to Apex. Retune on the exact Q6_K shape while measuring VGPRs, private segment, spills, occupancy evidence, and end-to-end variance. |
| Validate mode | Zinc replays a trusted per-token path and compares outputs, with additional intermediate validators in current source. | Adapt the methodology to the immutable Apex harness and task/eval contract; do not import Zinc runtime code. |

## Q6_K Shader Design

`dmmv_q6k_batch_kpar.comp` uses a 64-thread workgroup and dispatches `(M,1,1)`,
one workgroup per output row. The 64 lanes cooperate over K rather than assigning
one complete row to each lane:

1. A Q6_K block contains 256 values in 210 bytes: 128 bytes of low bits, 64
   bytes of high bits, 16 signed byte scales, and one fp16 block scale.
2. Sixteen lanes decode one block; four 16-lane groups cover four blocks in
   parallel per outer iteration.
3. Each lane reconstructs signed six-bit values from the low and high bit
   planes, applies the sub-block scale and block scale, and multiplies f32
   activation values.
4. Each lane retains `float sums[MAX_COLS]` and accumulates every active batch
   column in registers while the quantized weights are resident.
5. `subgroupAdd` reduces each column. On a requested RDNA wave64 one subgroup
   spans the workgroup; shared storage and elected subgroup results merge
   wave32 or SIMD16 subgroups on other devices.
6. Results are written column-major as `y[c*M + row]`.

The shader assumes K is divisible by 256; a remainder would be ignored. The
host must keep `num_cols <= MAX_COLS`, and Zinc chunks the token dimension to
enforce this. These shape and bounds requirements are part of the algorithm,
not incidental launch details.

## Q4_K and Register Pressure

`dmmv_q4k_batch_kpar.comp` uses the same workgroup mapping, four-block
parallelism, register accumulator array, subgroup reduction, and fallback
merge. Its checked-in comments report this five-sample median prefill sweep for
Qwen3-8B Q4_K_M on an R9700 with a 105-token prompt:

| `MAX_COLS` | Recorded tok/s | Recorded interpretation |
|---:|---:|---|
| 16 | 155 | Seven chunks; repeated weight reads dominate. |
| 32 | 173 | Better amortization. |
| 35 | 176 | Incremental improvement. |
| 40 | 187 | Peak; chunks of 40, 40, and 25. |
| 44 | 179 | VGPR pressure begins to hurt. |
| 48 | 165 | Higher VGPR pressure. |
| 64 | 153 | Scratch spilling. |

This is upstream recorded evidence, not an Apex reproduction. More
importantly, it is a Q4_K result from Zinc's Vulkan runtime and a smaller model,
not a Q6_K result for llama.cpp HIP or Qwen3.6-27B. The Q6 shader shares
`MAX_COLS=40`, but the pinned tree does not provide a separate Q6_K width sweep.

The reusable finding is the tradeoff: widening the token chunk amortizes
quantized weight reads until per-thread column accumulators raise register
pressure, reduce residency, or spill. Phase 3 must sweep a bounded width set on
the actual gfx1201 code object and retain the resource metadata and raw timings.
No source-level register estimate may substitute for compiler evidence.

## Host Dispatch and Current Priority

Zinc compiles the Q4_K and Q6_K GLSL shaders to SPIR-V and creates three-binding
pipelines using dynamically selected wave64 options. The option requests
subgroup size 64 and full subgroups only when Zinc's device configuration
reports a 64-lane wave. Pipeline creation can fail closed, and the shader's
shared merge covers smaller subgroup widths.

In `dispatchProjectionBatched`, K-parallel work is chunked to 40 columns and
launched as one workgroup per output row. A single default-on switch currently
controls both Q4_K and Q6_K selection, despite its Q4-specific name. Current
code first routes eligible Q4_K projections with at least 16 tokens to a tiled
`mul_mm_q4k` implementation; the K-parallel Q4 shader is a smaller-shape or
fallback path. Q6_K projections continue to use the K-parallel batch path when
the required pipeline is available.

This current-code distinction matters more than older prose that presents the
two K-parallel shaders as equal primary paths. It also reinforces the Apex gate:
the right comparison for a llama.cpp PP hotspot may be an existing tiled MMQ,
not a serial DMMV implementation like Zinc's original baseline.

## Validate-Mode Methodology

Zinc's most reusable contribution is the debugging boundary around the kernel.
`ZINC_BATCHED_PREFILL=validate`:

1. runs the candidate batched path;
2. snapshots the last-token logits;
3. resets mutable inference state to a fresh request;
4. replays the trusted per-token path;
5. compares the full last-token output with an explicit tolerance; and
6. leaves the trusted replay authoritative for subsequent decode.

The documented incident is instructive: the logits matched while generated
text was wrong, which redirected investigation from the shader to stale sampler
state. Current source also contains targeted Qwen3.6 dense-prefill validation
that captures boundaries such as normalized input, pre/post hidden state,
gate, up, SwiGLU, and down outputs for a selected layer and bounded token count.
This intermediate capture localizes the first divergence more effectively than
a final-output check alone.

Apex should adapt the pattern as follows:

- freeze a llama.cpp-owned oracle, inputs, tolerance, and reset procedure;
- compare candidate and reference intermediate tensors at named operation
  boundaries, then compare final outputs and tokens;
- verify that reset covers graph, KV, sampler, stream, and scratch state;
- keep the reference result authoritative after a validation replay;
- hash all captures and retain bounded max-absolute/RMS error evidence; and
- perform performance measurement in a separate clean run because validation
  replay overhead invalidates timing.

Zinc's static regression tests confirm shader markers and subgroup fallback
plumbing, but they are not numerical GPU tests. Its throughput and parity
figures remain upstream claims until reproduced under an Apex fingerprint.

## Port Gate

Before proposing any Zinc-inspired Q6_K PP task, Phase 3 must establish all of
the following:

1. The raw PP dispatch is joined to the exact llama.cpp launch and source path
   with stronger-than-name-only evidence.
2. The owning operation is known, including projection role and whether the
   work can legally be treated as batched independent columns.
3. Exact M/N/K, quant and activation datatypes, byte layouts, strides, batch
   organization, alignment, and edge shapes are captured from the real run.
4. Actual wavefront, workgroup, target, VGPR/SGPR, LDS/private segment, spill,
   and relevant ISA evidence is extracted from the loaded gfx1201 code object.
5. The existing llama.cpp MMQ implementation is the fresh comparison baseline,
   with normal graph settings and repeated variance-aware measurements.
6. A llama.cpp-owned isolated harness covers the exact Q6_K decode, scale,
   accumulation, output, and rounding contract before a candidate is compiled.
7. Any independently implemented candidate uses a rollback switch, explicit
   shape guard, fail-closed fallback, correctness-first evaluation, and exact
   source/binary/diff provenance.

Only after these gates pass should Apex ask whether K-parallel weight reuse,
column chunking, or a different tiling strategy is appropriate. Sharing the
GGML Q6_K byte format and the gfx1201 device is necessary context, not proof of
algorithmic compatibility.

## Explicit Non-Adoptions

- Do not add Zinc, Vulkan, GLSL, SPIR-V, or a foreign shader runtime to Apex or
  llama.cpp.
- Do not copy or port the shaders during this reference phase.
- Do not assume wave64, `MAX_COLS=40`, K divisibility, or Zinc's output layout
  for the llama.cpp workload.
- Do not treat upstream benchmark comments as Apex performance evidence.
- Do not replace the selected TG Q6_K Phase 3 loop with this PP follow-up.
- Do not begin the PP port until exact llama.cpp profiling and source/shape
  attribution establish the corresponding operation.
