# hipBLASLt Reference and Dense GEMM Attribution for Phase 3

## Scope

This is a read-only review and evidence check. It does not integrate
hipBLASLt, alter the Phase 2 capture adapter, rerun the model, or change the
evidence-selected Q6_K Phase 3 target.

The current upstream source reviewed is `ROCm/rocm-libraries` `develop` at
commit `ec658115b36f2158aaafb58810a2085394dac1cf`. This is the maintained
`projects/hipblaslt` implementation, not the stale `jichangjichang/hipBLASLt`
fork. The relevant upstream roots are:

- [`projects/hipblaslt`](https://github.com/ROCm/rocm-libraries/tree/ec658115b36f2158aaafb58810a2085394dac1cf/projects/hipblaslt)
- [`utilities/QuickTune`](https://github.com/ROCm/rocm-libraries/tree/ec658115b36f2158aaafb58810a2085394dac1cf/projects/hipblaslt/utilities/QuickTune)
- [`hipblaslt.h`](https://github.com/ROCm/rocm-libraries/blob/ec658115b36f2158aaafb58810a2085394dac1cf/projects/hipblaslt/library/include/hipblaslt/hipblaslt.h)
- [`tensile_host.cpp`](https://github.com/ROCm/rocm-libraries/blob/ec658115b36f2158aaafb58810a2085394dac1cf/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/tensile_host.cpp)
- [`gfx1201 logic`](https://github.com/ROCm/rocm-libraries/tree/ec658115b36f2158aaafb58810a2085394dac1cf/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1201)

The captured host uses ROCm 7.2.0 with packaged `hipblaslt` 1.2.1 and
`rocblas` 5.2.0. Current upstream behavior must not be assumed available in
that installed release; every candidate must pin the library actually used.

## Current Workload Verdict

The complete Phase 2 `kernel_stats.csv` files, not only Apex's top-N report,
contain no rocBLAS, hipBLAS, hipBLASLt, Tensile, `Cijk`, or equivalent dense
library GEMM kernel symbol in PP or TG.

| Phase | Custom GGML quantized kernels | Custom GGML other | Vendor dense GEMM | Runtime copy/fill | Significant GEMM M/N/K and types |
|---|---:|---:|---:|---:|---|
| PP | 86.700927% | 13.159400% | 0% | 0.137100% | None observed; not applicable |
| TG | 73.600000% | 26.316800% | 0% | 0.076987% | None observed; not applicable |

The custom-quantized bucket includes GGML Q6_K matrix kernels and explicit
quantization kernels. Its dominant matrix work is:

| Phase | Custom GGML kernel | Calls | Total GPU time | Phase share |
|---|---|---:|---:|---:|
| PP | `mul_mat_q<(ggml_type)14, 128, false>` | 4,800 | 4,049,485.081 us | 81.00% |
| PP | `mul_mat_q<(ggml_type)14, 128, true>` | 1,152 | 200,402.724 us | 4.01% |
| TG | `mul_mat_vec_q<(ggml_type)14, 1, false, false>` | 473,058 | 13,526,356.105 us | 41.06% |
| TG | `mul_mat_vec_q<(ggml_type)14, 1, true, false>` | 82,048 | 9,830,506.983 us | 29.84% |

This agrees with the pinned llama.cpp dispatch policy. The registered build
has `GGML_CUDA_FORCE_CUBLAS=OFF` and `GGML_CUDA_FORCE_MMQ=OFF`; the RDNA4 path
in `ggml/src/ggml-cuda/mmq.cu` currently selects MMQ consistently over
dequantization plus hipBLAS. llama.cpp still contains fallback
`hipblasGemmEx`, batched, and strided-batched paths, but they are not a
significant hotspot in this captured Q6_K workload.

Therefore hipBLASLt is not a data-supported optimization target for the first
Phase 3 loop. The custom Q6_K `mul_mat_vec_q` family remains primary.

## Evidence

- PP raw kernel trace:
  `results_phase02_qwen36_p2p_nographs_20260808/pp/raw/cornelius/2490453_kernel_trace.csv`
- PP SHA-256:
  `8563e42c0964c572530a170c05b13262eccee0483b68d8fd2bea6136e7c913bd`
- TG raw kernel trace:
  `results_phase02_qwen36_p2p_nographs_20260808/tg/raw/cornelius/2490710_kernel_trace.csv`
- TG SHA-256:
  `dd4fe80729863c7485267189d618dbbd72195978feb95d2ed20f45a0d53dfa98`
- Complete aggregate files:
  `pp/raw/cornelius/2490453_kernel_stats.csv` and
  `tg/raw/cornelius/2490710_kernel_stats.csv`

The `hip_api_trace.csv` schema contains only domain, function, process,
thread, correlation ID, and start/end timestamps. It can connect a HIP runtime
launch to a dispatch, but it contains no library-call argument payload. The
kernel trace adds launch geometry and resource fields, not BLAS M/N/K or
datatypes. Dimensions and types must never be guessed from a Tensile or code
object symbol.

The correct result for this Phase 2 capture is:

```text
dense_gemm_hotspots: none observed
M/N/K: unavailable because no significant dense GEMM was observed
A/B/C/D/compute datatypes: unavailable for the same reason
```

This is not proof that no library path exists anywhere in llama.cpp. It is
evidence that none is significant in these two phase-isolated runs.

## Attribution Contract

Future normalized profiles should classify every dispatch into one of:

- `custom_ggml_quantized`: `mul_mat_q`, `mul_mat_vec_q`, quantize,
  dequantize, and other explicit GGML quantized families;
- `custom_ggml_other`: GGML attention, normalization, state-space,
  collective, and elementwise kernels;
- `vendor_dense_gemm`: rocBLAS, hipBLAS, hipBLASLt, Tensile, or another
  verified dense GEMM provider;
- `runtime_other`: ROCclr/HIP copy, fill, and runtime kernels; or
- `unknown`.

Classification must retain the exact raw and demangled symbol, provider,
operation, confidence, evidence, phase, agent, calls, and phase duration. A
vendor-sounding kernel name is not enough to claim a specific library API or
GEMM shape.

Treat a vendor dense GEMM as significant when it is in the phase top-N or
accounts for at least 1% of phase GPU dispatch duration. Preserve all smaller
records, but do not activate a tuning branch from noise alone.

## Future Shape and Type Capture

If a later normal-condition PP or TG trace contains significant vendor GEMM,
perform a separate diagnostic capture. Library logging perturbs performance,
so it must not be the timing baseline.

For a hipBLAS/rocBLAS path, use official rocBLAS logging with adapter-owned
absolute output paths:

- `ROCBLAS_LAYER=2` for reproducible `rocblas-bench` command records;
- `ROCBLAS_LAYER=4` for de-duplicated profile YAML and call counts; and
- optionally `ROCBLAS_LAYER=8` with trace logging to record the selected GEMM
  backend when supported.

For a verified hipBLASLt path, `HIPBLASLT_LOG_MASK=32` and
`HIPBLASLT_LOG_FILE=<adapter-owned-path>` emit `hipblaslt-bench` problem
records suitable for offline tuning.

Each significant problem record must include, when the API exposes it:

- API and backend;
- M, N, K;
- transpose/layout for A and B;
- lda/ldb/ldc/ldd;
- batch count and A/B/C/D strides;
- A/B/C/D, compute, scale, and bias datatypes;
- alpha, beta, epilogue/activation, and bias configuration;
- algorithm/solution index and workspace limit;
- phase, call count, agent/device, and stream/thread/correlation where
  available; and
- binary, source/build, ROCm, rocBLAS/hipBLAS/hipBLASLt, log, and trace
  SHA-256 provenance.

Join a library call to a dispatch only with an explicit correlation or other
unambiguous evidence. Otherwise keep the call phase-attributed but
`dispatch_link=unlinked`. Separate PP/TG processes make phase attribution
authoritative without inventing a per-dispatch join.

## Current hipBLASLt Candidate Paths

These are conditional Phase 3 paths, not active recommendations for the
current Q6_K hotspot.

### QuickTune and offline tuning

`projects/hipblaslt/utilities/QuickTune` extracts log-generated
`hipblaslt-bench` commands, removes duplicates while retaining occurrence
counts, searches the installed kernel pool, writes raw/reproduction logs and
`tuning_result.csv`, and applies results through
`HIPBLASLT_TUNING_OVERRIDE_FILE`.

The lower-level offline flow uses:

- `HIPBLASLT_TUNING_FILE` to write winning solution indices;
- `HIPBLASLT_TUNING_OVERRIDE_FILE` to apply them; and
- `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` to constrain tuning workspace.

Solution indices are specific to a library release and device architecture.
Pin and hash the override file; retune after a library, target, problem, or
code-object change. QuickTune's estimated gain covers hipBLASLt GEMM time, not
llama.cpp end-to-end performance.

### Algorithm selection

The maintained API exposes `hipblasLtMatmulAlgoGetHeuristic`, workspace and
compute-unit preference hints, solution enumeration, support checks, and
explicit algorithm selection. For a significant exact problem, compare the
default heuristic with independently measured supported solutions under the
same workspace and correctness contract. A heuristic rank is not a measured
winner.

### Stream-K

Current upstream exposes `HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT`
with `OFF`, `ON`, and `AUTO` modes for supported StreamK=5 hybrid kernels.
`ON` requests the dynamic SK4 work-queue path; `OFF` uses the static SK3 path
by default; `AUTO` lets the library choose per launch. Compute-unit targets
are heuristic hints, not exact occupancy controls.

Only test Stream-K when the exact M/N/K/type/layout problem has supported
StreamK solutions. Benchmark all applicable modes with stable repeated
samples; do not assume Stream-K is faster.

### gfx1201 implementations

Current `ROCm/rocm-libraries` contains dedicated gfx1201 Equality and
GridBased solution logic for multiple FP16, BF16, FP32, FP8/BF8, bias, and
activation combinations. Runtime selection detects gfx1201 and loads matching
code-object modules. This demonstrates a real Radeon-specific path, but only
the installed library's solution set and support query determine whether a
captured problem is covered.

## Activation Gate

Do not activate hipBLASLt tuning unless all of the following hold:

1. A normal-condition phase trace shows a significant verified dense GEMM.
2. A separate library log records its exact M/N/K, layouts, types, batch,
   epilogue, and call count.
3. The installed hipBLASLt version and gfx1201 solution support are queried
   and pinned.
4. The tuning task preserves a fresh uninstrumented baseline and immutable
   correctness contract.
5. Candidate solution/override artifacts, workspace, binary, and library
   provenance are hashed and retained.
6. The result is rechecked in the real llama.cpp workload after isolated GEMM
   tuning.

Never use QuickTune's optional `amd-smi --perf-determinism` procedure; Apex
does not change GPU power state. Never use Phase 2's profiler-disabled-graphs
throughput as the tuning baseline. Do not replace a Q6_K GGML kernel with a
dense GEMM merely because hipBLASLt has gfx1201 implementations.

## Decision

No dense GEMM candidate is activated for Phase 3 from the current evidence.
Preserve hipBLASLt QuickTune, offline tuning, measured algorithm selection,
Stream-K, and gfx1201 solution logic as conditional paths only if a later
profile passes the activation gate.
