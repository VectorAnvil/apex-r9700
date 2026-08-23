# DKQ=256 RDNA4 Flash Attention FA-0 finding

Status: FA-0 timing and dispatch proof complete. Dynamic hardware counters are
unavailable because the installed profiler crashes reproducibly.

## Frozen identities and correctness

- registered Apex `259f2e2` plus registered P2P and Phase 13 patches;
- issue-era rocWMMA `c588c4f47683e73ad2d69f50480bec6cc85fd0f7`;
- native tile base `a7a6d0d269c896218b6c78e0933bd6a17519d3f6`;
- PR `#26419` head `d76c0046947c8b3fe92949fffeb634bbe7cc5d40`.

Registered Apex and native base passed their full FA suites on both GPUs.
rocWMMA passed all 224 target-relevant DKQ=256 cases across both GPUs; its
deprecated full suite reaches an unrelated unsupported `hsk=192` path on
gfx1201. PR `#26419` passed its full suite on GPU 1 and 2,919/2,920 on GPU 0,
with one `hsk=192` tolerance miss. The exact missed case then passed 20/20
repetitions, while repeated PR DKQ=256 coverage passed 1,260/1,260 cases.

## Runtime dispatch and exact active resources

Graph-disabled rocprof traces at PP512/depth 65,536 prove these active paths:

| Path | Exact active symbol | Calls/GPU | Workgroup | Trace LDS | Trace scratch |
| --- | --- | ---: | ---: | ---: | ---: |
| registered | tile `<256,256,16,2,false>` | 2,080 | 256 threads / 8 waves | 37,888 B | 0 B |
| native | tile `<256,256,16,2,false>` | 2,080 | 256 threads / 8 waves | 37,888 B | 0 B |
| rocWMMA | `<256,16,4,64,float,false>` | 2,080 | 128 threads / 4 waves | 25,600 B | 0 B |
| PR MMA | `<256,256,8,8,false,false>` | 2,080 | 128 threads / 4 waves | 0 B static field | 808 B |

The initially inspected rocWMMA `<256,32,4,64,half,false>` and native tile
`<256,256,4,8,false>` symbols are compiled but inactive for this workload.
Exact code-object records for the dispatched symbols are:

| Path | Compiled LDS | VGPR | SGPR | Private | VGPR spills | SGPR spills | WMMA ISA occurrences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rocWMMA | 25,344 B | 211 | 44 | 0 B | 0 | 0 | 128 |
| tile | 37,888 B | 242 | 48 | 0 B | 0 | 0 | 0 |
| PR MMA | 0 B static; 34,944 B dynamic by launch formula | 256 | 107 | 808 B | 284 | 13 | 768 |

PR MMA contains 394 static `scratch_load` and 232 static `scratch_store`
occurrences. rocWMMA and tile contain none. On a 64 KiB LDS CU, LDS alone
permits at most two resident rocWMMA workgroups and one tile or PR workgroup.
This bounds rocWMMA and tile at eight waves and PR at four waves by LDS. It is
not achieved occupancy, and it shows why LDS alone cannot explain rocWMMA
versus tile: rocWMMA also has a different `ncols=16` decomposition, far fewer
compiled VGPRs, WMMA, and a cheap separate combine pass.

## Timing result

The normal-graph matrix completed 144/144 independent processes with rotated
build order and no trimming. Median tokens/second are:

| Workload/depth | registered | rocWMMA | native tile | PR `#26419` | rocWMMA vs tile | PR vs tile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PP512 / 0 | 1,043.04 | 1,090.88 | 1,083.36 | 1,085.30 | +0.7% | +0.2% |
| PP512 / 16k | 785.05 | 951.07 | 820.32 | 841.10 | +15.9% | +2.5% |
| PP512 / 65k | 485.59 | 727.11 | 498.56 | 496.28 | +45.8% | -0.5% |
| PP512 / 127k | 329.78 | 567.47 | 336.10 | 334.18 | +68.8% | -0.6% |
| PP4096 / 0 | 1,004.64 | 1,080.89 | 1,054.90 | 1,059.42 | +2.5% | +0.4% |
| PP4096 / 16k | 802.49 | 975.14 | 829.09 | 851.14 | +17.6% | +2.7% |
| PP4096 / 65k | 493.00 | 743.38 | 502.19 | 500.67 | +48.0% | -0.3% |
| PP4096 / 127k | 331.87 | 577.21 | 336.01 | 336.95 | +71.8% | +0.3% |
| TG128 / 0 | 36.335 | 36.590 | 36.630 | 36.433 | -0.1% | -0.5% |
| TG128 / 16k | 35.388 | 35.134 | 35.531 | 35.514 | -1.1% | 0.0% |
| TG128 / 65k | 32.300 | 31.700 | 32.415 | 32.485 | -2.2% | +0.2% |
| TG128 / 127k | 29.285 | 28.273 | 29.326 | 29.329 | -3.6% | 0.0% |

All PP groups have less than 1.4% min-to-max spread. The graph-disabled trace
shows the same mechanism ordering: summed primary-attention duration across
both GPUs is about 67.2 s for tile, 66.8 s for PR MMA, and 25.6 s for rocWMMA;
the rocWMMA combine pass adds about 0.11 s. Trace timing is not compared with
normal-graph throughput.

## Counter limitation and residual decision

Three ROCm 7.2 `rocprofiler-sdk` 1.1.0 counter probes were attempted: all
metrics dual-GPU, `OccupancyPercent` alone on two selected dual-GPU dispatches,
and `OccupancyPercent` alone on one single-GPU dispatch at depth 16k. All three
aborted with `std::out_of_range: unordered_map::at` and one incomplete dispatch.
Achieved occupancy, utilization, cache behavior, and dynamic stalls are
therefore `unavailable`; resource metadata is not relabeled as counter data.

PR `#26419` selects native MMA and bypasses K/V LDS for DKQ=256, but it remains
statistically tied with tile at deep context and about 41% slower than rocWMMA
in 127k PP throughput. The next isolated mechanism should restore or adapt the
exact rocWMMA `<256,16,4,64,float,false>` plus combine design in current source.

One global gate remains unresolved: registered Apex TG128/127k produced frozen
samples 25.759, 29.285, and 29.314 t/s (12.1% spread), and a separate fourth
adjudication run fell to 13.958 t/s. No sample was removed. All native, PR, and
rocWMMA TG groups and every PP group remain below 1% spread except rocWMMA
PP512/16k at 1.393%.

Raw worktrees, builds, correctness logs, traces, failed counter logs, matrix
rows, hashes, HSACOs, metadata, and ISA remain under ignored
`results_fa00_long_context_20260814/`.
