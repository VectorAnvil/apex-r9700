# Rejected Experiment: Global Four-Wave Q6_K MMV Dispatch on 2 x R9700

## Summary

I tested a global workgroup reduction for the custom GGML Q6_K matrix-vector
kernel used by a dual Radeon AI PRO R9700 llama.cpp workload. The candidate
changed Q6_K `N=1` dispatch from eight wave32 warps to four.

The result was **rejected**. All 36 CPU-oracle correctness rows passed, and
some middle-sized unfused shapes improved by 4.8% to 11.3%, but small shapes
regressed by 6.7% to 51.9%. The exact fused `ffn_up` hotspot was flat at about
`0.9983x` on both GPUs. The frozen exact-slice gate therefore prohibited a
full-model Llama Lab benchmark or candidate registration.

- Prompt processing: not measured; the promotion gate failed first.
- Token generation: not measured end to end; the promotion gate failed first.
- Kernel correctness: 36/36 rows passed at NMSE `<=0.0005`.
- Decision: reject the unconditional four-wave policy.

This negative result is retained so the same global launch policy is demoted
in future suggestion history.

## Hardware And Software

- GPUs: 2 x AMD Radeon AI PRO R9700 32 GB
- GPU target: `gfx1201`, RDNA4, wave32
- OS: Ubuntu 24.04
- ROCm/HIP: 7.2.26015
- llama.cpp source: `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`
  (build 9940)
- Existing P2P/AllReduce patch: retained unchanged
- Model workload: Huihui Qwen3.6-27B Q6_K
- Registered comparison target: Llama Lab profile 5, suite 2, model 2,
  build 2; not executed because the exact gate failed

Both isolated builds used Release mode, HIP, `AMDGPU_TARGETS=gfx1201`, native
optimizations, HIP graphs enabled, CUDA flash attention enabled, RPC disabled,
tests enabled, and identical harness source. The only candidate-versus-baseline
source change was `ggml/src/ggml-cuda/mmvq.cu`.

## Why This Was Targeted

The Phase 2 `rocprofv3` trace showed the two custom GGML Q6_K MVM variants at
70.903% of summed token-generation kernel duration. No significant rocBLAS,
hipBLAS, or hipBLASLt dense GEMM dispatch was present.

Phase 3 then found promising full-model-shape microbenchmark results after
reducing the Q6_K `N=1` workgroup from 256 to 128 threads. Runtime attribution
later showed that the registered two-GPU tensor-parallel workload uses
different per-device slices. In particular, fused `ffn_up` is logically
`M=8704,N=1,K=5120` per device and accounts for 29.842% of the diagnostic TG
dispatch duration. Phase 4 tested those actual slices instead of promoting the
full-shape result.

## Optimization

Baseline `calc_nwarps()` selected eight warps for Q6_K when `ncols_dst == 1`.
The candidate selected four unconditionally for the same Q6_K path:

```text
ggml/src/ggml-cuda/mmvq.cu
Q6_K, ncols_dst=1: 8 wave32 warps -> 4 wave32 warps
workgroup size:     256 threads   -> 128 threads
```

The hypothesis was that a smaller workgroup would lower reduction and launch
cost on gfx1201. The candidate did not change quantization, weights, reference
outputs, inputs, timers, or acceptance thresholds.

Patch SHA-256:
`8003dba642a37092a6fb1a9601d13d0c4dd1c7338a2d9c532bbbf4dd9026ad62`.

## Correctness Validation

The independent GGML harness used deterministic inputs and a CPU oracle. Nine
exact runtime-attributed shapes were evaluated for baseline and candidate on
both physical GPUs, producing 36 required correctness rows.

| Check | Result |
|---|---:|
| Required correctness rows | 36/36 pass |
| Maximum accepted NMSE | 0.0005 |
| Baseline/candidate harness identity | byte-identical |
| Unrelated source differences | none |

Correctness passing was necessary but did not override performance regression
or stability gates.

## Benchmark Methodology

Three fixed-order rounds used the schedule
`D0-A,D1-A,D0-B,D1-B,D1-A,D0-A`. Each device/case therefore had six baseline
and three candidate process-level samples. Results below are median kernel
times; speedup is baseline divided by candidate.

The frozen policy required:

- relative range divided by median `<=3.5%` for stability;
- every exact case `>=0.99x` on each device;
- a win threshold of `>=1.01x`;
- fused `ffn_up >=1.01x` on both devices before E2E.

Normal graph settings were retained. Idle guards preserved failed snapshots
and waited for both target GPUs to pass the unchanged 5% activity/VRAM limit.
No cross-shape or cross-device average was used.

## Kernel-Level Results

| Exact case | Device | Baseline us | Candidate us | Speedup | Status |
|---|---:|---:|---:|---:|---|
| `24x1x5120`, SSM | 0 | 5.955 | 6.380 | 0.9334x | stable regression |
| `24x1x5120`, SSM | 1 | 5.945 | 6.370 | 0.9333x | stable regression |
| `512x1x5120`, attn K/V | 0 | 8.870 | 16.860 | 0.5261x | stable regression |
| `512x1x5120`, attn K/V | 1 | 9.620 | 19.990 | 0.4812x | stable regression |
| `3072x1x5120`, attn gate | 0 | 23.130 | 21.710 | 1.0654x | stable win |
| `3072x1x5120`, attn gate | 1 | 23.105 | 21.610 | 1.0692x | stable win |
| `5120x1x3072` | 0 | 25.375 | 23.100 | 1.0985x | stable win |
| `5120x1x3072` | 1 | 25.640 | 23.040 | 1.1128x | stable win |
| `5120x1x5120`, attn QKV | 0 | 34.425 | 32.350 | 1.0641x | unstable candidate |
| `5120x1x5120`, attn QKV | 1 | 35.095 | 32.410 | 1.0828x | stable win |
| `5120x1x8704`, FFN down | 0 | 52.360 | 49.870 | 1.0499x | unstable baseline |
| `5120x1x8704`, FFN down | 1 | 52.185 | 49.800 | 1.0479x | stable win |
| `6144x1x5120`, attn Q | 0 | 40.345 | 37.740 | 1.0690x | stable win |
| `6144x1x5120`, attn Q | 1 | 40.525 | 37.490 | 1.0810x | stable win |
| `8704x1x5120`, fused FFN up | 0 | 3.015 | 3.020 | 0.9983x | stable, no win |
| `8704x1x5120`, fused FFN up | 1 | 3.005 | 3.010 | 0.9983x | stable, no win |
| `124160x1x5120`, lm head | 0 | 852.675 | 845.350 | 1.0087x | unstable baseline; no win |
| `124160x1x5120`, lm head | 1 | 854.455 | 848.220 | 1.0074x | stable, no win |

Device 0 contained three isolated outliers: candidate attn QKV at 87.41 us,
baseline FFN down at 122.52 us, and baseline lm head at 13743.58 us. Those
series remain unstable; their medians are reported but cannot establish wins.

## GPU Resource Changes

Fresh code objects confirmed the intended gfx1201 specialization compiled.

| Variant | Workgroup | Wave | LDS false/true | VGPR false/true | SGPR false/true | Private/spills |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 256 | 32 | 896/1792 B | 26/35 | 26/42 | 0 |
| Four-wave candidate | 128 | 32 | 384/768 B | 26/35 | 26/42 | 0 |

The resource reduction proves that the intended launch geometry was emitted;
it does not explain or outweigh the measured shape-dependent regressions.

## What Did Not Work

| Experiment or assumption | Result | What was learned |
|---|---:|---|
| Unconditional Q6_K eight-to-four-wave dispatch | rejected | Launch width is strongly shape-dependent; one global choice cannot preserve the small cases. |
| Small `M=24` slice | -6.7% on both GPUs | Four waves added enough cost to violate the 0.99x regression floor. |
| Small `M=512` slice | -47.4% / -51.9% | This is the decisive failure and must remain on the eight-wave path. |
| Exact fused `ffn_up` hotspot | about -0.17% on both GPUs | The high-share fused path received no useful benefit from the workgroup reduction. |
| Promoting from Phase 3 full-shape microbenchmarks | invalid basis | Tensor-parallel runtime slices differed from the original GGUF-derived shapes. Runtime attribution was required. |
| Treating isolated device-0 medians as wins | unstable | Preserved outliers made attn QKV, FFN down, and lm head ineligible as promotion evidence. |
| Aggregating wins and regressions across shapes | prohibited | A weighted average could conceal severe exact-shape regressions and was intentionally not calculated. |

The candidate is not merely "not yet proven." Its unconditional form is a
known failed idea and should not be suggested again under a different name.

## Caveats

- This result applies to Q6_K `N=1` MMV on two gfx1201 R9700 GPUs under the
  pinned ROCm and llama.cpp revisions.
- The test was an exact-slice kernel gate, not a full-model throughput result.
- The diagnostic Phase 2 trace used disabled graphs for profiler compatibility;
  only its attribution and duration shares were used to select cases.
- The test does not reject shape-specific four-wave dispatch. It rejects the
  unconditional policy.
- Results may differ on other architectures, ROCm releases, quantizations,
  model shapes, or single-GPU execution.

## Reproduction And Evidence

Portable inputs and results are under:

```text
docs/apex-r9700/artifacts/phase-04/
```

Key files:

- `candidate.patch`: exact rejected source change.
- `harness.patch`: exact-shape CPU-oracle harness.
- `task.json`: immutable thresholds, cases, devices, and provenance.
- `gate-manifest.json`: correctness invocations and timing schedule.
- `gate-result.json`: raw samples, medians, variance, and decision fields.
- `code-object.json`: gfx1201 resource metadata.
- `attempts.jsonl`: safety aborts, procedure revision, and parser recovery.

Raw worktrees, builds, logs, activity snapshots, and HSACO remain under
`results_phase04_q6k_nwarps4_20260808/`.

## Next Step

Test a new dispatch candidate that selects four waves only for the three exact
stable wins: unfused `3072x1x5120`, `5120x1x3072`, and `6144x1x5120`.
Everything else, especially small, unstable, fused, and output-head cases,
must retain eight waves. That is a distinct candidate with a new immutable
contract, not a retry of this patch.
