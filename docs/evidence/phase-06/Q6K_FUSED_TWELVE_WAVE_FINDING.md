# Phase 6: Fused Q6_K Twelve-Wave `ffn_up` Evaluation

## Summary

Phase 6 evaluates exactly one candidate: a fused-only twelve-wave Q6_K `N=1`
MMV specialization for gfx1201 `ffn_up` `M=8704,N=1,K=5120`. All unfused,
ID/MoE, unsupported-fusion, and nonmatching paths retain the upstream
eight-wave behavior.

**Decision: reject.** This is a kernel-level rejection; no PP/TG promotion
claim is made.

- Correctness: 36/36 CPU-oracle rows passed.
- Dispatch: 36/36 runtime witness records passed.
- Exact-gate timing: rejected.
- Counter diagnostic: unavailable after four timeouts.
- Promotion/E2E: not run; the exact gate failed.

## Hardware And Software

- Two dynamically identified Radeon AI PRO R9700 `gfx1201` GPUs.
- `rocminfo` records wavefront 32 and a 1024-thread maximum workgroup on each;
  the candidate 384-thread launch is legal.
- `rocprofv3` 1.1.0, ROCm 7.2.0; llama.cpp
  `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`.

Raw identities, `rocminfo`, and separate `rocprofv3 --list-avail` output for
both GPUs are in `results_phase06_q6k_fused_nwarps12_20260808/capabilities/`.
Advertised counters are capability evidence only; no Instinct counter table or
inferred counter semantics are used.

## Candidate And Rationale

Phase 2 attributed 29.842% of summed diagnostic TG dispatch duration to fused
Q6_K `ffn_up`. At `K=5120`, `QK_K=256` yields 20 blocks: eight waves need
three K-loop advances and twelve need two. This is a hypothesis, not a causal
conclusion; cross-wave reduction and occupancy may erase or reverse it.

The selector is Q6_K, `N=1`, RDNA4, no IDs, `M=8704,K=5120`, nonnull gate,
SwiGLU, and no x/gate bias or lane scale. The architecture-independent
compile-time `force_twelve_waves` specialization preserves upstream
device-side `calc_nwarps()` behavior as fallback. Unknown fusion state fails
closed to eight waves.

## Dispatch And Correctness

`rocprofv3` is a runtime dispatch witness only; its durations are excluded from
performance fields.

| Variant | Required behavior | Witness status |
|---|---|---|
| Baseline | eight waves, all nine slices | pass |
| Candidate exact fused `8704x1x5120` | twelve waves | pass |
| Candidate other eight slices | eight-wave fallback | pass |

All 36 deterministic CPU-oracle rows passed at the frozen NMSE threshold.

## Static Compiler Resources

AMDHSA code-object metadata is the compiler-resource source of truth; trace
resource columns are dispatch observations only.

| Specialization | Threads | VGPR | SGPR | LDS | Private | VGPR/SGPR spills |
|---|---:|---:|---:|---:|---:|---:|
| Fused eight-wave baseline/fallback | 256 | 35 | 42 | 1792 B | 0 B | 0 / 0 |
| Fused twelve-wave candidate | 384 | 35 | 42 | 2816 B | 0 B | 0 / 0 |

The preserved evidence retains raw/demangled symbols, HSACO and raw-metadata
hashes, tool version, launch bounds, and dispatch-join quality. These compiled
facts are not a performance explanation.

## Resource And Counter Contract

Every scorecard value carries `metric_class` and `measurement_scope`:
`static_code_object`, `derived_bound`, `unprofiled_microbenchmark`,
`runtime_counter`, or `unavailable`. Derived occupancy fails closed unless all
CU/SIMD, register-allocation, LDS-allocation, workgroup, and slot inputs are
recorded. The combined occupancy bound is unavailable because register-file
sizes/allocation granularities, LDS allocation granularity, and architectural
workgroup-slot limits were not exposed for gfx1201.

Only a serialized exact-symbol, nonmultiplexed capture with documented
semantics can supply counter values. Bandwidth requires exact-scope measured
bytes and matching GPU elapsed time; tensor dimensions, clocks, and profiler
throughput are not substitutes. The frozen nonmultiplexed
`OccupancyPercent+FETCH_SIZE` diagnostic attempted exact baseline/candidate
captures on both GPUs. All four timed out at 120 seconds before producing a
joinable result. Occupancy and read/total bandwidth are therefore unavailable.

## Benchmark Results

Three interleaved normal-graph rounds require relative range/median `<=3.5%`,
fused `ffn_up >=1.01x` on both GPUs, and every fallback `>=0.99x`. No outlier
is removed and no cross-device/shape aggregate is used.

| Exact case | Device | Baseline | Candidate | Speedup | Gate |
|---|---:|---:|---:|---:|---|
| SSM `24x1x5120` | 0 | 5.985 | 6.020 | 0.9942x | pass |
| attn K/V `512x1x5120` | 0 | 8.925 | 8.940 | 0.9983x | fail: baseline spread 3.70% |
| attn gate `3072x1x5120` | 0 | 23.090 | 23.130 | 0.9983x | pass |
| `5120x1x3072` | 0 | 25.360 | 25.520 | 0.9937x | pass |
| attn QKV `5120x1x5120` | 0 | 34.425 | 34.530 | 0.9970x | pass |
| ffn down `5120x1x8704` | 0 | 52.420 | 52.470 | 0.9990x | pass |
| attn Q `6144x1x5120` | 0 | 40.435 | 40.470 | 0.9991x | pass |
| fused ffn up `8704x1x5120` | 0 | 3.010 | 3.010 | 1.0000x | fail: misses 1.01x |
| lm head `124160x1x5120` | 0 | 853.430 | 854.040 | 0.9993x | pass |
| SSM `24x1x5120` | 1 | 5.955 | 6.000 | 0.9925x | pass |
| attn K/V `512x1x5120` | 1 | 9.600 | 9.640 | 0.9959x | pass |
| attn gate `3072x1x5120` | 1 | 23.040 | 23.080 | 0.9983x | pass |
| `5120x1x3072` | 1 | 25.610 | 25.690 | 0.9969x | pass |
| attn QKV `5120x1x5120` | 1 | 35.105 | 35.080 | 1.0007x | pass |
| ffn down `5120x1x8704` | 1 | 52.320 | 52.320 | 1.0000x | pass |
| attn Q `6144x1x5120` | 1 | 40.595 | 40.550 | 1.0011x | pass |
| fused ffn up `8704x1x5120` | 1 | 3.010 | 3.000 | 1.0033x | fail: misses 1.01x |
| lm head `124160x1x5120` | 1 | 855.320 | 855.670 | 0.9996x | pass |

The exact fused row was stable on both GPUs: device 0 baseline/candidate
spreads were 2.33%/0.33%, and device 1 1.00%/0.00%. It nevertheless missed the
required 1.01x target on both devices. Apart from device-0 `attn_kv`, whose
3.70% baseline spread additionally fails, all excluded rows passed.

## What Did Not Work

| Experiment or assumption | Result | Learning |
|---|---|---|
| Twelve waves for exact fused `ffn_up` | rejected | A 3-to-2 K-loop reduction produced only 1.0000x/1.0033x, below the 1.01x requirement. |
| Counter diagnosis | unavailable | All four frozen exact nonmultiplexed captures timed out at 120 seconds without a join. |
| Derived occupancy minimum from partial limits | prohibited | Required gfx1201 allocation/resource/slot inputs were absent. |
| Sixteen-wave follow-up | demoted | It has the same two K-loop iterations as twelve waves and more reduction work. |

## Caveats And Next Step

This is an exact Q6_K `N=1` kernel gate, not yet a PP/TG claim. A 36/36
dispatch witness proves selection, not speed or counter availability. Preserve
all incorrect, unstable, neutral, and regressing rows. The loop reduction did
not help. VGPR/SGPR/private/spills stayed 35/42/0/0 while LDS rose from 1792 B
to 2816 B and cross-wave reductions rose from seven to eleven. Causality
remains unresolved without counters. Reject twelve waves and demote sixteen
waves; stop at Phase 6.
