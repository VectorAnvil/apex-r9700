# Fluke and FlyDSL Reference for Phase 3

## Scope

This is a read-only review of two repositories:

- [`BonsonW/fluke`](https://github.com/BonsonW/fluke/tree/7460d64f3d638223dd4c33084abb48e448056b54)
  at commit `7460d64f3d638223dd4c33084abb48e448056b54` (2026-07-18).
- [`ROCm/FlyDSL`](https://github.com/ROCm/FlyDSL/tree/2d65dea1380876f4faa68159cadfec1eeaa598ab)
  at commit `2d65dea1380876f4faa68159cadfec1eeaa598ab` (2026-08-08).

Both were shallow-cloned to temporary directories. Nothing was built, run,
ported, or integrated into Apex or llama.cpp.

Primary implementation and deployment sources:

- [`Fluke RDNA4 exporter`](https://github.com/BonsonW/fluke/blob/7460d64f3d638223dd4c33084abb48e448056b54/fly/rdna4/_fp8_export.py)
- [`Fluke HIP dispatch`](https://github.com/BonsonW/fluke/blob/7460d64f3d638223dd4c33084abb48e448056b54/src/fused_hip.c)
- [`Fluke public C ABI`](https://github.com/BonsonW/fluke/blob/7460d64f3d638223dd4c33084abb48e448056b54/include/fluke/fluke.h)
- [`FlyDSL gfx120x WMMA atom`](https://github.com/ROCm/FlyDSL/blob/2d65dea1380876f4faa68159cadfec1eeaa598ab/include/flydsl/Dialect/FlyROCDL/IR/MmaAtom.td)
- [`FlyDSL gfx120x lowering`](https://github.com/ROCm/FlyDSL/blob/2d65dea1380876f4faa68159cadfec1eeaa598ab/lib/Dialect/FlyROCDL/GFX120X/MmaAtom.cpp)
- [`FlyDSL dynamic device detection`](https://github.com/ROCm/FlyDSL/blob/2d65dea1380876f4faa68159cadfec1eeaa598ab/python/flydsl/runtime/device.py)
- [`FlyDSL AOT documentation`](https://github.com/ROCm/FlyDSL/blob/2d65dea1380876f4faa68159cadfec1eeaa598ab/docs/quickstart.rst)

FlyDSL is Apache-2.0 licensed. The pinned Fluke tree contains no tracked
`LICENSE`, `COPYING`, or `NOTICE` file and no detected SPDX declaration. Treat
Fluke code as unlicensed unless upstream supplies clear terms. Apex may study
its design and measurements, but must not copy or port its code. Reusable code
must come from licensed FlyDSL sources or be independently implemented.

## Verdict

Fluke is a useful Phase 3 reference because it combines gfx1201-oriented
FlyDSL kernels with a small deployment boundary. It does not change the first
Phase 3 target and is not a Q6_K implementation.

Working prioritization:

| Use | Rating | Meaning |
|---|---:|---|
| Current gfx1201 Q6_K workload | 7/10 reference | Strong architecture, layout, pipeline, fusion, and deployment ideas; incompatible FP8 arithmetic and weight layout prevent direct reuse. |
| Apex kernel prototyping | 8.5/10 reference | FlyDSL can express exact-shape gfx1201 experiments and expose generated code objects, provided Apex retains its own oracle, evaluator, and provenance. |
| Future FP8 Qwen workload | Potentially high | Fluke's FP8 E4M3 WMMA path becomes more directly relevant only after a real FP8 model/build is registered and profiled on R9700. |

The Phase 2 profiler remains the activation authority. Similar operation names,
RDNA4 support, or a promising standalone kernel do not establish a candidate.

## Phase 2 Activation Matrix

| Reference concept | Current Apex evidence | Status and activation gate |
|---|---|---|
| gfx1201 wave32 WMMA patterns | The registered binary targets gfx1201, but the selected hotspot is custom GGML Q6_K MMV, not FP8 dense GEMM. No joined HSACO/ISA resource evidence exists yet. | Pattern reference only. Activate for one source-attributed candidate after exact code-object target, types, fragment ABI, and numerical contract are proven. |
| Preshuffled weight layout | Fluke expects FP8 B as `[N/16,K/16,2,16,8]`; llama.cpp stores GGML Q6_K blocks with different quant metadata and decode rules. | Inactive. Require a measured compatible representation and include offline transform, storage, load, and fallback costs. Never reinterpret Q6_K bytes as a Fluke layout. |
| Register-direct GEMM | Fluke's FP8 base streams GMEM to registers and WMMA without LDS. Phase 2 proves Q6_K MMQ/MMV heat but not that the selected Q6_K path has compatible fragments or is limited by LDS. | Conditional hypothesis only after exact source, shape, ISA, register, LDS, private-segment, spill, and occupancy evidence. |
| Software-pipelined K loop | The selected TG MMV accounts for 70.903% of summed TG dispatch duration; its exact shape and bottleneck are not yet frozen. | Eligible as an independently derived Q6_K experiment only if the isolated harness and ISA show useful load/compute overlap without damaging register occupancy. |
| Dual GEMM plus SiLU | PP contains dominant Q6_K matrix work and SiLU-family kernels, but Phase 2 does not prove sibling gate/up ownership, compatible layouts, or adjacency. | Inactive. Require two source-attributed projections, exact matching shapes/types/layouts, adjacent SwiGLU materialization, combined share, and rounding-aware parity. |
| QKV GEMM plus RoPE | Phase 2 identifies no material source-attributed QKV-plus-RoPE boundary. Current work is Q6_K, not FP8. | Inactive. Require a significant adjacent QKV projection/RoPE pair and exact head, rotary, table, layout, position, and output semantics. |
| Per-token quant fusion | TG `quantize_q8_1` is 2.698% and RMSNorm is 1.892%, but these are GGML Q8_1/f32 operations, not Fluke FP8 E4M3 semantics. | Secondary only. Require source/shape/scale attribution and proof that one compatible producer-consumer fusion removes a measured launch without changing GGML rounding or block layout. |
| AOT HSACO/C ABI deployment | Phase 3 needs isolated, pinned candidate builds but does not need a second production kernel runtime. | Useful prototype pattern after the llama.cpp-owned harness is frozen. Production integration remains out of scope; require reproducible export, hashes, ABI tests, dynamic arch selection, and fail-closed fallback. |
| Future FP8 Qwen | No FP8 weights, activations, dense GEMM, or conversion path was observed in the registered Q6_K workload. | Separate future lane. Register a true FP8 model/build and capture normal-condition PP/TG evidence before considering any kernel. |

PP Q6_K `mul_mat_q<14,128,...>` accounts for 85.011% of summed PP dispatch
duration, while TG Q6_K `mul_mat_vec_q<14,1,...>` accounts for 70.903% of TG.
That proves the importance of custom GGML low-bit matrix work, not compatibility
with Fluke's FP8 GEMM.

## Fluke Kernel Patterns

### Preshuffle and Register-Direct GEMM

[`rdna_fp8_preshuffle_gemm.py`](https://github.com/BonsonW/fluke/blob/7460d64f3d638223dd4c33084abb48e448056b54/fly/rdna4/gemm/rdna_fp8_preshuffle_gemm.py)
implements FP8 E4M3FN input and weights, f32 accumulation, and fp16 output.
Activations use one scale per token and weights one scale per output channel.
The host transforms B from `[K,N]` into
`[N/16,K/16,KLane=2,NLane=16,KPack=8]`, allowing each wave32 lane to load the
bytes expected by the raw gfx12 FP8 WMMA intrinsic.

The hot loop loads raw A and preshuffled B directly from global memory into
8-byte register vectors. It uses no LDS. The K loop has a load-first prologue,
compile-time inner unrolling, loop-carried register fragments, and a final
epilogue tile. Defaults vary with M: M tiles 32 or 64, N tiles 128 or 256, K
tile 32, and one to four waves. M/N/K must exactly divide the chosen tiles.

This is a valuable implementation sketch for testing whether a measured shape
benefits from removing LDS traffic. It is not proof that register-direct is
best for Q6_K, whose block scales and bit unpacking can change both the register
budget and optimal staging strategy.

### Dual GEMM and SiLU

[`rdna_fp8_dual_gemm_silu.py`](https://github.com/BonsonW/fluke/blob/7460d64f3d638223dd4c33084abb48e448056b54/fly/rdna4/dual_gemm_silu/rdna_fp8_dual_gemm_silu.py)
shares the A fragment while maintaining separate gate and up B fragments and
f32 accumulator sets. Its register epilogue applies
`silu(A @ B_gate) * (A @ B_up)` and stores fp16. It uses the same preshuffle,
tile divisibility, wave32, and software-pipeline structure as the base GEMM.

This reinforces SuperSonic's gate/up hypothesis through a different kernel
authoring mechanism. It does not relax the SuperSonic activation gate: Apex
still needs source-attributed sibling projections, adjacency, compatible
layouts, phase share, and exact materialization/rounding parity.

### GEMM and RoPE

[`rdna_fp8_gemm_rotary.py`](https://github.com/BonsonW/fluke/blob/7460d64f3d638223dd4c33084abb48e448056b54/fly/rdna4/rotary/rdna_fp8_gemm_rotary.py)
keeps QKV WMMA accumulators in registers, applies rotate-half RoPE to Q and K,
and writes V unchanged. It requires `N == 3 * nhead * head_dim`, head-aligned N
tiles, and compatible rotary pairs within its accumulator tile.

The reference has important contract risks: the documentation says sequence
length must be a power of two while the implementation uses modulo, and some
rotary-dimension assumptions are implicit rather than asserted. Any Apex task
must freeze the exact llama.cpp Qwen rotary convention, positions, table
precision/layout, QKV organization, partial rotary behavior, and edge shapes.

### Per-Token Quantization

Fluke's FlyDSL export currently emits dual-GEMM-plus-SiLU to fp16 and then a
separate FP16-to-FP8 per-token quantization kernel. It is explicitly a two
dispatch variant, not a fused dual-GEMM-plus-quant kernel. The quantizer uses
one 256-thread workgroup per row, reduces the row amax, writes `amax / 448`, and
packs E4M3FN bytes. The public C ABI does not expose this quantizer.

Fluke's C/HIP path separately demonstrates fused RMSNorm plus per-token INT8 or
software-E4M3FN quantization. Preserve the broader producer-plus-quant fusion
idea, but do not record an implementation that is absent. For today's workload,
GGML's measured Q8_1 conversion and its block metadata remain the correctness
authority.

## FlyDSL gfx1201 Boundary

Official FlyDSL provides dynamic target discovery through
`flydsl.runtime.device`, classifies `gfx120*` as RDNA wave32, and lowers for an
explicit AMDGPU chip target. Its typed `gfx120x.wmma` atom has a dedicated v8
operand ABI and device/MLIR tests, but at the pinned revision the typed atom
accepts only 16x16x16 f16 or bf16 inputs with f32 accumulation. Negative tests
reject FP8 through that typed atom.

Official FlyDSL also contains
[`rdna_fp8_preshuffle_gemm.py`](https://github.com/ROCm/FlyDSL/blob/2d65dea1380876f4faa68159cadfec1eeaa598ab/kernels/gemm/rdna_fp8_preshuffle_gemm.py),
which uses the lower-level `rocdl.wmma_f32_16x16x16_fp8_fp8` route on RDNA4.
This is source evidence for a native gfx12 FP8 intrinsic path, not blanket proof
that every FlyDSL WMMA abstraction, tile, or generated object works on Apex's
ROCm 7.2/gfx1201 environment. It must be compile-tested, numerically tested,
and inspected on the actual R9700. gfx1250-only TDM, MX-scale, FP4, and wider
WMMA paths are not gfx1201 capabilities.

FlyDSL's checked-in gfx1201 tests cover the typed f16/bf16 atom and RDNA GEMM.
The repository also quarantines unrelated RMSNorm SmoothQuant tests on gfx1201,
so repository-wide feature maturity must not be inferred from one working
kernel.

## AOT HSACO and C ABI Pattern

Official FlyDSL's documented AOT deployment is a target-sensitive Python cache:
precompile into `FLYDSL_RUNTIME_CACHE_DIR`, then use
`FLYDSL_RUNTIME_RUN_ONLY=1` to forbid a cache miss from JIT-compiling. The
cached artifact contains compiled MLIR/entry metadata and recreates FlyDSL's
MLIR execution engine. It is not a documented stable standalone HSACO C ABI.

Fluke builds its own deployment layer:

1. compile separately for concrete `gfx1200` and `gfx1201` targets;
2. extract an ELF HSACO from FlyDSL-generated MLIR text;
3. generate a C launch header with baked geometry;
4. embed each HSACO into a library and load it with `hipModuleLoadData`;
5. strip feature suffixes from `gcnArchName`, select the exact per-chip image,
   and return a fail-closed fp16 fallback for an unsupported target or shape;
6. expose ATen-free C calls for QKV-plus-RoPE and gated-MLP kernels.

At the pinned revision this path is specialized to model dimensions
`d_model=512`, `dim_feedforward=2048`, `nhead=8`, and `head_dim=64`; generated
artifacts are gitignored and no committed RDNA4 benchmark results establish
performance. A stale comment in `fly/common.py` claims a generic gfx12 artifact,
while the exporter and runtime correctly use concrete per-chip objects.

If Apex later adopts this deployment shape, derive it from licensed interfaces
and treat the compiler/exporter/runtime ABI as a versioned component. Persist
the FlyDSL and LLVM/MLIR commits, Python/package versions, target, compile and
link arguments, kernel metadata, source/IR/HSACO/header hashes, symbol, launch
geometry, C signature, layout descriptor, and parity results. Load only an
allowlisted exact target and fail closed on any mismatch.

## Prototype Contract

FlyDSL can be used only after the first Phase 3 llama.cpp-owned Q6_K harness,
oracle, inputs, and metric are immutable. A prototype task must:

1. cite the exact rocprofv3 dispatch and source/shape evidence that activates
   the experiment;
2. keep the GGML reference, data generation, tolerances, timing, and scoring
   outside the FlyDSL candidate's write boundary;
3. compile for explicit `gfx1201` in an isolated pinned environment and retain
   all IR and code-object provenance;
4. validate the generated symbol, target, VGPR/SGPR, LDS/private segment,
   spills, and ISA before drawing conclusions from source structure;
5. compare against the same immutable llama.cpp computation, not only a
   PyTorch or Fluke reference;
6. run low-level correctness, end-to-end token/output parity, and interleaved
   variance-aware A/B with a rollback path; and
7. append success, no-win, compile, correctness, and instability results under
   an exact fingerprint.

A FlyDSL success may justify an independently implemented llama.cpp candidate
or a later deployment design. It does not by itself authorize adding Python,
FlyDSL, Fluke, a foreign HSACO, or a new runtime dependency to production.

## Future FP8 Qwen Gate

FP8 Qwen on R9700 is a separate workload, not an alternate interpretation of
the Q6_K profile. Before using Fluke as a more direct kernel source:

1. register the exact FP8 model representation, conversion/calibration lineage,
   llama.cpp or alternate runtime, binary, and normal-condition benchmark;
2. confirm actual gfx1201 FP8 WMMA compilation and ISA on the installed ROCm
   toolchain rather than assuming an Instinct path or a gfx1250 feature;
3. profile PP and TG separately and attribute exact M/N/K, dtypes, scale
   granularity, layouts, conversion/preshuffle cost, and operation ownership;
4. select one material operation using the same hotspot and variance rules as
   the Q6_K work; and
5. establish model-quality, numerical, token, and performance baselines before
   testing representation or fusion changes.

Until that lane exists, Fluke/FlyDSL stays bookmarked and the first Phase 3
target remains llama.cpp's Q6_K TG MMV family.
