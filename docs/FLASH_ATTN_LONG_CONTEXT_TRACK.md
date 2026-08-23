# Apex RDNA4 Flash Attention / Long-Context Track

Branch: `hazyumps/vllm-gfx1201-qwen3.6-256k`

This track is separate from the Q6_K WMMA and speculative-decoding tracks. Its
first objective is measurement, not kernel rewriting: determine where the
llama.cpp Flash Attention paths spend LDS to reduce memory latency on gfx1201
but lose more from reduced residency and latency hiding than staging saves.

## Pinned upstream evidence

- `hazyumps/vllm-gfx1201-qwen3.6-256k` main at
  `df4c6ee8622f796620ca2086808b1ed951660ef1`. Its AITER Triton attention change
  reduces `num_stages` from 2 to 1, reducing the reported LDS footprint from
  32 KiB to 16 KiB per workgroup and increasing resident workgroups per CU from
  two to four. At about 175k prompt tokens, measured decode rises from 37.9 t/s
  to 59-63 t/s.
- llama.cpp issue `#26220`, opened 2026-07-28. It compares old rocWMMA commit
  `c588c4f` with native FA commit `42fc2430`; PP4096 at depth 126,976 falls from
  777.27 to 394.81 t/s while TG64 rises slightly.
- llama.cpp PR `#26419`, current head
  `d76c0046947c8b3fe92949fffeb634bbe7cc5d40`. It enables native MMA for head
  dimension 256 and bypasses LDS for K/V when `DKQ > 128`. The current head also
  restores an end-of-iteration barrier for LDS-resident `tile_mask`, fixing a
  write-after-read race. The author's Qwen3.6 tests report +13.6%, +19.1%, and
  +21.8% PP4096 at depths 16k, 65k, and 127k with flat TG. A separate dual-R9700
  test still measured the PR about 31% behind restored rocWMMA at 65k.

Do not replace these identities with floating branch names in evidence. Record
new upstream heads separately if the experiment is refreshed.

## Source map

Inspect and map at minimum:

- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`
- `ggml/src/ggml-cuda/fattn.cu`
- `ggml/src/ggml-cuda/mma.cuh`

For every active attention specialization, record:

`tile shape x head dimension x bytes per element x staging depth`

and reconcile that static accounting with code-object and launch metadata for:

- static and dynamic LDS per workgroup;
- threads, waves, and workgroups;
- VGPR, SGPR, private segment, and spills;
- theoretical residency limits from LDS, waves, VGPR, and SGPR independently.

The minimum causal chain to test is:

`LDS/workgroup -> resident workgroups/CU -> active waves -> memory-latency hiding -> PP performance versus depth`

## Phase FA-0: frozen comparison

Create isolated llama.cpp worktrees under a new ignored results root. Do not
modify the registered Llama Lab source or build.

Compare four builds. The fourth identity is required because source chronology
alone did not establish the registered runtime path. The FA-0 trace later proved
that registered Apex `259f2e2` dispatches the same DKQ=256 tile specialization
as the newer native base, while remaining a distinct integration generation:

1. the current registered Apex llama.cpp baseline, preserving the registered
   Phase 13 target library and runtime configuration;
2. the issue-era old rocWMMA path at
   `c588c4f47683e73ad2d69f50480bec6cc85fd0f7`;
3. the native tile-kernel PR base at
   `a7a6d0d269c896218b6c78e0933bd6a17519d3f6`;
4. PR `#26419` at `d76c0046947c8b3fe92949fffeb634bbe7cc5d40`.

The old rocWMMA path may instead use a source-equivalent restoration whose
patch, base revision, and code-object identity are recorded exactly.

If the four source generations cannot share the same model, flags, graph mode,
KV types, split mode, and tensor ownership, report the comparison as unavailable
rather than treating unlike configurations as an A/B result.

Run the complete matrix on both R9700 devices under the existing registered
two-GPU ownership and communication configuration:

| Workload | Depths |
| --- | --- |
| PP512 | 0, 16,384, 65,536, 126,976 |
| PP4096 | 0, 16,384, 65,536, 126,976 |
| TG128 | 0, 16,384, 65,536, 126,976 |

Use alternating build order, at least three independent processes per cell,
no outlier removal, and report every sample, median, relative spread, and ratio.
Capture normal graph behavior separately from profiling runs if rocprofv3
changes graph execution.

## Profiling contract

First capture kernel traces to prove which FA symbol handles each cell and to
join exact dispatch counts and durations. Then use the smallest supported,
nonmultiplexed counter groups needed to obtain or explicitly mark unavailable:

- LDS per workgroup and limiting resource;
- waves per workgroup, resident workgroups/CU, and achieved occupancy;
- WMMA and VALU utilization;
- memory stalls and cache/global-load behavior;
- K/V load path and bytes, distinguishing LDS staging from direct VRAM;
- barriers, private allocation, and spills.

Static LDS or compiler metadata must not be presented as achieved occupancy.
Profiler timeouts, unsupported counters, and unjoinable dispatches remain
`unavailable`; do not substitute estimates.

## Gates before optimization

No new kernel candidate is authorized until FA-0 establishes all of the
following:

- the exact active symbol for every PP/TG shape and depth;
- correctness and graph parity for every build;
- reproducible long-context performance deltas within 5% process spread;
- static resource accounting reconciled with launch metadata;
- dynamic occupancy and stall evidence, or a documented tool limitation;
- a localized remaining gap between PR `#26419` and rocWMMA.

The first candidate should change one mechanism only: K/V staging, staging
depth, tile shape, or dispatch selection. Do not mix Flash Attention work with
Q6_K matrix multiplication, MTP token-count sweeps, KV precision changes, or
communication tuning.

## Deferred follow-up

After attention is no longer the long-context limiter, return to speculative
decode as a separate experiment. The pinned vLLM report observes about 3.2
accepted tokens per engine step through roughly 243k context and proposes
testing 2, 4, and 5 speculative tokens. That evidence motivates a later sweep;
it is not part of FA-0 and is not assumed to transfer directly to llama.cpp.

## FA-0 outcome

FA-0 completed 144/144 normal-graph timing processes: four builds, three
workloads, four depths, and three independent rotated-order samples per cell.
All target workloads completed successfully. The exact grouped samples,
medians, spreads, and ratios are recorded in
`evidence/fa-00/FA0_MATRIX_RESULTS.tsv`.

The central result is stronger than the upstream report. Relative to the
native tile base, issue-era rocWMMA is:

| Cell | rocWMMA versus native |
| --- | ---: |
| PP512 / 16k | +15.9% |
| PP512 / 65k | +45.8% |
| PP512 / 127k | +68.8% |
| PP4096 / 16k | +17.6% |
| PP4096 / 65k | +48.0% |
| PP4096 / 127k | +71.8% |

PR `#26419` does not close that gap in this Apex configuration. Against the
native tile base it is +2.5%/+2.7% at 16k, -0.5%/-0.3% at 65k, and
-0.6%/+0.3% at 127k for PP512/PP4096 respectively. Its TG128 result is flat
against tile, while rocWMMA is 2.2% slower at 65k and 3.6% slower at 127k.

Trace evidence corrects the initial static map. PP512 at depth 65,536 dispatches:

| Build | Active attention path | Calls per GPU | Runtime resource signature |
| --- | --- | ---: | --- |
| registered Apex | tile `<256,256,16,2,false>` | 2,080 | 37,888 B LDS, 248 VGPR field, no scratch |
| native base | tile `<256,256,16,2,false>` | 2,080 | 37,888 B LDS, 248 VGPR field, no scratch |
| rocWMMA | `<256,16,4,64,float,false>` plus combine | 2,080 each | 25,600 B LDS, 248 VGPR field, no scratch |
| PR `#26419` | MMA `<256,256,8,8,false,false>` | 2,080 | 808 B scratch, 256 VGPR field; 34,944 B dynamic LDS by launch formula |

Aggregate primary-attention duration across both GPUs was about 67.2 s for
tile, 66.8 s for PR MMA, and 25.6 s for rocWMMA. The rocWMMA combine kernel
added only about 0.11 s. These are profiler-distorted, graph-disabled durations
used for path comparison, not production throughput.

The active rocWMMA specialization has 25,344 B compiled static LDS, 211 VGPRs,
44 SGPRs, no private segment, and no spills. Its trace requests a 25,600 B LDS
block and launches four waves per workgroup. Tile requests 37,888 B and launches
eight waves per workgroup. PR MMA launches four waves, still requests about
34,944 B dynamic LDS, reaches 256 VGPRs, and has an 808 B private segment with
284 VGPR and 13 SGPR spills. On a 64 KiB LDS CU this bounds rocWMMA at two
workgroups by LDS, versus one for tile and PR; achieved occupancy remains
unavailable.

ROCm 7.2 `rocprofiler-sdk` 1.1.0 counter collection is broken on this host. An
all-metric dual-GPU probe, an `OccupancyPercent`-only two-dispatch dual-GPU
probe, and an `OccupancyPercent`-only one-dispatch single-GPU probe all aborted
with `std::out_of_range: unordered_map::at` and left one incomplete dispatch.
Failed logs are preserved under the ignored FA-0 result root. Static and trace
resource fields are not substituted for achieved occupancy.

One unrelated PR full-suite `hsk=192` tolerance failure occurred once on GPU 0
and did not recur in 20 exact-case repetitions. The target DKQ=256 PR path
passed 1,260/1,260 repeated cases across both GPUs. A separate registered-Apex
TG128/127k instability remains: the frozen samples were 25.759, 29.285, and
29.314 t/s, and a fourth adjudication run fell to 13.958 t/s. No sample was
removed. This blocks the global spread gate but does not weaken the PP finding,
whose worst three-process spread is 1.393%.

The next mechanism should be a current-source restoration or adaptation of the
exact active rocWMMA `<256,16,4,64,float,false>` path, including its separate
combine step, rather than another isolated K/V LDS-bypass change. Keep that
candidate separate from Q6_K, speculation, KV precision, and P2P work.

## FA-1 outcome: current-source rocWMMA restoration

FA-1 restored the issue-era rocWMMA implementation onto current llama.cpp
`4695f001fece1660d8bb1b3748f50726ddcc100b` in an isolated worktree. The
device implementation in `fattn-wmma-f16.cu/.cuh` is byte-for-byte source
from `c588c4f47683e73ad2d69f50480bec6cc85fd0f7`. The current-source adaptation
adds only its build plumbing and a deliberately narrow dispatch:

- RDNA4 with rocWMMA enabled;
- DKQ=256;
- more than two Q rows, so current batch-1/2 decode dispatch is untouched;
- K length aligned to `FATTN_KQ_STRIDE`.

All other head dimensions and decode shapes remain on current dispatch. The
exact 778-line FA-only transplant is frozen in
`evidence/fa-01/FA1_CURRENT_ROCWMMA_TRANSPLANT.patch`. It excludes the
pre-existing Apex direct-P2P AllReduce and Q6_K integration changes, which were
applied identically to control and candidate.

The final candidate compiled under ROCm 7.2 for gfx1201 without adapting the
restored device kernel to current APIs. It passed 126/126 DKQ=256
`FLASH_ATTN_EXT` cases on each physical R9700 after the final dispatch
correction. The matched current-native control also passed 126/126 on each GPU.

Final trace gates establish the runtime split:

| Work | Current-native | Final FA-1 candidate |
| --- | --- | --- |
| PP512 | tile `<256,256,16,2,false>` | rocWMMA `<256,16,4,64,float,false>` plus combine |
| TG timed decode | tile `<256,256,1,2,false>` | tile `<256,256,1,2,false>` |
| TG depth preparation | tile `<256,256,16,2,false>` | rocWMMA `<256,16,4,64,float,false>` plus combine |

The final PP trace therefore retains the FA-0 resource mechanism: rocWMMA
requests 25,600 B LDS and four waves/workgroup, versus current-native tile at
37,888 B and eight waves/workgroup. A final corrected-candidate PP512/16k
trace measured 939.07 t/s and launched the exact requested specialization.

The first full current-source matrix completed 72/72 normal-graph processes:
two builds, three workloads, four depths, three independent alternating-order
samples. Its candidate dispatch still inherited the historical batch-1/2
vector fallback; those TG rows are diagnostic and superseded below. The PP
predicate and device path are identical to the final candidate, so its PP
results remain the final FA-1 PP evidence:

| Cell | Current-native median | FA-1 median | Gain |
| --- | ---: | ---: | ---: |
| PP512 / d0 | 1076.37 | 1078.10 | +0.2% |
| PP512 / 16k | 812.51 | 940.52 | +15.8% |
| PP512 / 65k | 495.15 | 719.69 | +45.3% |
| PP512 / 127k | 333.49 | 559.58 | +67.8% |
| PP4096 / d0 | 1043.64 | 1071.17 | +2.6% |
| PP4096 / 16k | 822.52 | 965.31 | +17.4% |
| PP4096 / 65k | 500.10 | 734.90 | +47.0% |
| PP4096 / 127k | 333.82 | 543.01 | +62.7% |

No sample was removed. Several first-pass long-depth samples collapsed for one
or both variants, so FA-1 does not claim the FA-0 sub-5% global spread gate:
PP512 spreads reached 58.8%/11.6% at 65k and 50.6%/60.3% at 127k for
native/candidate. The other two samples in each affected group agree closely,
and PP4096/65k is especially tight at 0.17%/0.07%, but the outliers remain part
of the evidence. Exact samples are in
`evidence/fa-01/FA1_MATRIX_V1_RESULTS.tsv`.

The initial dispatch's TG regression was localized by trace, not attributed to
the rocWMMA PP kernel: it changed timed decode from current tile to the old
vector fallback. Tightening the predicate to Q rows greater than two restored
exact decode-kernel parity. Three corrected candidate processes at d0, 16k,
and 65k are flat to slightly positive against the frozen native samples:

| TG128 depth | Native median | Corrected FA-1 median | Delta |
| --- | ---: | ---: | ---: |
| d0 | 36.567 | 36.628 | +0.17% |
| 16k | 35.470 | 35.581 | +0.31% |
| 65k | 32.339 | 32.372 | +0.10% |

At 127k, the separate TG instability recurred: corrected-candidate samples
were 29.080, 21.312, and 21.977 t/s versus native 29.297, 29.347, and
29.341 t/s. The corrected trace proves both timed decode paths use the same
tile specialization, so this is retained as an unresolved depth/setup or graph
interaction rather than evidence against the PP decomposition. Exact corrected
TG samples are in `evidence/fa-01/FA1_TG_DISPATCH_V2_RESULTS.tsv`.

FA-1 therefore succeeds as a current-source transplantation of the specific
long-context PP mechanism: it reproduces FA-0's depth-dependent gains while
leaving shallow PP essentially flat and current decode dispatch intact.
FA-S1 cleared the promotion block without changing the kernel. Graph-on/off,
cold-start, monitored/unmonitored, and repeated in-process probes localized the
occasional collapse to an intermittent early process/device-state effect after
long idle rather than the restored rocWMMA path. A preconditioned clean rerun
completed all 24 affected-cell processes with a worst spread of `0.804%`.
FA-1 measured `+68.06%` for PP512/127k and `+70.51%` for PP4096/127k while
TG128/127k was `-0.10%`. The exact adjudication and samples are in
`evidence/fa-s1/`. Broken rocprof counter collection remains out of scope; no
counter-fix detour or further FA tuning was taken.
