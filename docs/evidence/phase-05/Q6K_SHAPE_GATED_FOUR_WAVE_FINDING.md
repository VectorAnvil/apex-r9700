# Rejected Experiment: Shape-Gated Four-Wave Q6_K MMV on 2 x R9700

## Summary

I tested a shape-gated launch policy for the custom GGML Q6_K `N=1`
matrix-vector kernel on two Radeon AI PRO R9700 GPUs. The candidate selected
four wave32 waves only for unfused `3072x1x5120`, `5120x1x3072`, and
`6144x1x5120`; every other shape retained eight waves.

The result was **rejected**. Runtime traces proved the intended selection in
all 36 variant/device/case records, and all 36 CPU-oracle correctness rows
passed. The three selected shapes retained favorable medians, but the frozen
gate required every series to be stable and every unchanged shape to remain at
least `0.99x`. Five rows failed those requirements, including an `M=512`
fallback median of `0.9878x` on device 0.

- Prompt processing: not measured; the exact-slice gate failed first.
- Token generation: not measured end to end; the exact-slice gate failed first.
- Kernel correctness: 36/36 rows passed at NMSE `<=0.0005`.
- Dispatch correctness: 36/36 records passed.
- Decision: reject; do not register or promote this candidate.

## Hardware And Software

- GPUs: 2 x AMD Radeon AI PRO R9700 32 GB
- Target: `gfx1201`, RDNA4, wave32
- OS: Ubuntu 24.04
- ROCm/HIP: 7.2.26015
- llama.cpp: `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`
  (build 9940)
- Existing P2P/AllReduce patch: retained byte-identically
- Model workload: Huihui Qwen3.6-27B Q6_K
- Registered comparison: profile 5, suite 2, model 2, build 2; not run

Both isolated builds used Release mode, HIP, `AMDGPU_TARGETS=gfx1201`, native
optimizations, HIP graphs enabled, CUDA flash attention enabled, RPC disabled,
and the same CPU-oracle harness. The registered source, build, database, model,
and services were not modified.

## Why This Was Targeted

Phase 2 showed custom GGML Q6_K MMV variants at 70.903% of summed TG kernel
duration, with no significant rocBLAS, hipBLAS, or hipBLASLt dense GEMM.
Phase 4 rejected a global four-wave policy because small shapes regressed, but
three exact unfused runtime slices improved stably on both GPUs. Phase 5 tested
whether an exact selector could retain only those wins.

Fused `ffn_up`, logical `8704x1x5120`, remained on eight waves. It accounts for
29.842% of diagnostic TG dispatch duration and received no useful benefit from
the Phase 4 four-wave experiment.

## Optimization

The selector was frozen on actual dispatch parameters:

```text
Q6_K, N=1, RDNA4, no IDs, no fusion
four waves only for (M,K) = (3072,5120), (5120,3072), (6144,5120)
eight waves for every other shape
```

The corrected implementation adds an architecture-independent compile-time
`force_four_waves` specialization. The default specialization continues to
derive its wave count through upstream `calc_nwarps()`.

- Source: `ggml/src/ggml-cuda/mmvq.cu`
- Corrected patch SHA-256:
  `b052fb327bdcdb3f3ac79e4ea7345af5d7cd7b6539dc31f1f6eb6b4c43ff4a16`
- Rollback: remove the candidate patch; upstream eight-wave selection remains
  unchanged.

## Dispatch And Correctness

A bounded `rocprofv3` pass was used only as a runtime dispatch witness. Its
durations were excluded from evaluation. Full mangled Q6_K symbols and
workgroup dimensions established:

| Variant | Cases | Required | Observed |
|---|---|---:|---:|
| Baseline | all nine, both GPUs | 8 waves | 8 waves |
| Candidate selected | three unfused tuples, both GPUs | 4 waves | 4 waves |
| Candidate fallback | six excluded tuples, both GPUs | 8 waves | 8 waves |

The deterministic GGML harness then produced all 36 required baseline and
candidate CPU-oracle rows with maximum NMSE `0.0005`. All passed.

## Benchmark Methodology

Three fixed-order rounds used
`D0-A,D1-A,D0-B,D1-B,D1-A,D0-A`. Each device/case received six baseline and
three candidate process samples. Timings were unprofiled, used normal graph
settings, and were evaluated per shape and physical device.

The frozen policy required relative range/median `<=3.5%`; each selected shape
had to be stable and `>=1.01x` on both GPUs; each fallback had to be stable and
`>=0.99x`. No outlier was removed and no cross-shape/device average was used.

## Kernel-Level Results

| Exact case | Device | Baseline us | Candidate us | Speedup | Gate |
|---|---:|---:|---:|---:|---|
| `24x1x5120`, SSM | 0 | 5.970 | 6.000 | 0.9950x | pass |
| `24x1x5120`, SSM | 1 | 5.940 | 5.980 | 0.9933x | pass |
| `512x1x5120`, attn K/V | 0 | 8.880 | 8.990 | 0.9878x | fail: unstable baseline and regression |
| `512x1x5120`, attn K/V | 1 | 9.620 | 9.650 | 0.9969x | pass |
| `3072x1x5120`, selected | 0 | 23.090 | 21.870 | 1.0558x | pass |
| `3072x1x5120`, selected | 1 | 23.095 | 21.700 | 1.0643x | fail: unstable candidate |
| `5120x1x3072`, selected | 0 | 25.420 | 23.260 | 1.0929x | pass |
| `5120x1x3072`, selected | 1 | 25.610 | 23.080 | 1.1096x | pass |
| `5120x1x5120`, fallback | 0 | 34.460 | 34.550 | 0.9974x | pass |
| `5120x1x5120`, fallback | 1 | 34.970 | 35.110 | 0.9960x | pass |
| `5120x1x8704`, fallback | 0 | 52.375 | 52.460 | 0.9984x | pass |
| `5120x1x8704`, fallback | 1 | 52.235 | 52.260 | 0.9995x | pass |
| `6144x1x5120`, selected | 0 | 40.445 | 37.850 | 1.0686x | fail: unstable baseline |
| `6144x1x5120`, selected | 1 | 40.590 | 37.560 | 1.0807x | pass |
| fused `8704x1x5120` | 0 | 3.005 | 3.010 | 0.9983x | pass |
| fused `8704x1x5120` | 1 | 3.010 | 3.010 | 1.0000x | fail: unstable candidate |
| `124160x1x5120`, lm head | 0 | 852.975 | 852.870 | 1.0001x | pass |
| `124160x1x5120`, lm head | 1 | 856.170 | 855.680 | 1.0006x | fail: unstable baseline |

The unstable rows retained isolated values of 366.25 us for device-0 baseline
`attn_q`, 103.66 us for device-1 candidate `attn_gate`, 3.17 us for device-1
candidate fused `ffn_up`, and 7474.32 us for device-1 baseline lm head. The
device-0 `attn_kv` baseline range was 8.63-9.10 us and its candidate median
also missed the regression floor.

## GPU Resources

| gfx1201 specialization | Threads | LDS | VGPR | SGPR | Private/spills |
|---|---:|---:|---:|---:|---:|
| Baseline unfused eight | 256 | 896 B | 26 | 26 | 0 |
| Candidate unfused four | 128 | 384 B | 26 | 26 | 0 |
| Candidate unfused fallback eight | 256 | 896 B | 26 | 26 | 0 |
| Candidate fused fallback eight | 256 | 1792 B | 35 | 42 | 0 |

These facts confirm the intended specializations and fallback resources. They
are provenance evidence, not a causal explanation of timing.

## What Did Not Work

| Experiment or assumption | Result | Learning |
|---|---:|---|
| Architecture-dependent integer template default (v1) | incorrect | Host launch and gfx1201 specialization could disagree; the excluded `M=24` trace timed out. Use an architecture-independent specialization selector. |
| Corrected exact shape gate (v2) | rejected | Local median wins did not satisfy the all-series stability and regression contract. |
| Treating unchanged fallback code as automatically neutral | invalid | Device-0 `M=512` measured 0.9878x and must fail even though resources matched baseline. |
| Removing isolated samples | prohibited | Several favorable medians remained ineligible because the frozen spread limit was exceeded. |
| Full-model validation after local wins | not permitted | Kernel wins cannot bypass a failed exact gate. |

The v1 implementation failure is preserved separately from the v2 performance
result so future agents do not repeat the unsafe template-default mechanism.

## Caveats

- This result applies to the pinned Q6_K `N=1` workload on two gfx1201 R9700s.
- It is a kernel gate, not a PP/TG throughput claim.
- Unstable rows were preserved rather than retried or trimmed.
- The result rejects this candidate under the frozen policy; it does not show
  that every possible exact dispatch scheme is slower.
- Other architectures, ROCm versions, quantizations, or single-GPU layouts may
  behave differently.

## Reproduction And Evidence

Portable evidence is under `docs/apex-r9700/artifacts/phase-05/`. Raw builds,
HSACO, traces, activity snapshots, stdout/stderr, and both implementation
attempts remain under `results_phase05_q6k_shape_gated_20260808/`.

Key files are `task.json`, `candidate.patch`, `dispatch-selection.json`,
`gate-manifest.json`, `gate-result.json`, `code-object.json`, `e2e.json`, and
`attempts.jsonl`.

## Next Step

Do not promote or rerun this shape-gated candidate. The next independent
hypothesis should target the exact fused `ffn_up` hotspot while leaving all
unfused paths untouched. A conservative first candidate is a fused-only
12-wave specialization for `8704x1x5120`, with 16 waves retained as a separate
later candidate rather than swept in the same phase. It requires dynamic
gfx1201 limit checks, CPU parity, dispatch proof, resource inspection, and the
same per-device variance gate.
