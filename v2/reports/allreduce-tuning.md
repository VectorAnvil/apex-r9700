> Historical research record. Statements about the selected service are as of this report, not current deployment status. See [the V2 index](../README.md). Local raw evidence, private runtime files and logs are not included; local artifact paths are provenance references, not downloadable links.

# Upstream AllReduce tuning on dual R9700

Status: source audit and standalone screens complete; threshold8MiB rejected after the full model curve. Chunk2MiB also completed exact model checks, with only small unconfirmed differences. Keep allreduce defaults; no useful model gain is established. Adam requested investigating the upstream path used with Direct-P2P disabled. This worktree starts at frozen cbc3985. Qualified scheduling/Q6 remains the control; the separate packed-Q8 integration passed exact model checks but added no useful model gain.

## Verified implementation

Downloaded current upstream commit81ff93ea1d48c0482508d30e75b814df99c25bf0 (2026-09-30T16:09:08Z) into an isolated evidence directory. Its allreduce.cu matches our file after removing only the gated APEX P2P/fused-boundary hooks and blank lines. The internal dispatcher and communication-mode selection sections also match. HIP vendor mappings and HIP CMakeLists.txt are byte-identical. There is no newer allreduce.cu implementation missing from the pinned baseline in this comparison. Other backend changes are retained in the downloaded diff and must not be mistaken for communication changes.

The enabling upstream change is https://github.com/ggml-org/llama.cpp/pull/27825, merged September15. It maps host-pinned allocation and device-pointer APIs and emulates nanosleep with AMD S_SLEEP. Its reported gains are against the older meta-backend butterfly route on other configurations; do not count them as new gains available to our already-enabled implementation. The earlier symptom described in the thread was meta-backend fallback, not Apple's Metal backend.

Current trial PID2379983 was read without changing it: GGML_CUDA_AR_BF16_THRESHOLD=1, APEX_V2_DIRECT_P2P=0, APEX_V2_FUSED_BOUNDARY=0, no copy-threshold/chunk overrides. Startup records initialized internal two-GPU pipelines. Prior Core profiler evidence proves actual kernel execution:95,906 ggml_cuda_ar_kernel<float,BF16> calls per card and16,900 ggml_cuda_ar_add_kernel<float,BF16> calls per card. Startup alone does not prove every reduction uses that path; the new instrumented check must count unsupported/fallback calls too.

This is pinned-host-memory transport over PCIe. It is not direct peer-VRAM transport. Small tensors are exchanged and summed by one kernel per GPU on the compute stream. Large tensors use chunked D2H/H2D copy streams and a final local add on the compute stream. Neither allreduce message size nor call count necessarily grows with retained KV depth; active row count, speculation and model graph structure determine those. Measure the full context curve to capture changing compute/communication overlap.

## Tuning leads and discriminators

| Lead | Current behavior / source | Minimal experiment | Required signal |
| --- | --- | --- | --- |
| Small-message launch geometry | allreduce.cu: GGML_CUDA_AR_KERNEL_BLOCKS=8;256threads; per-block host arrival flag | Isolated compile-time2/4/8/16 block variants at observed decode/MTP sizes; keep transport and BF16 rounding fixed | Lower whole collective latency on both GPUs, then improved model TG without PP regression |
| Slot reuse and host waiting | ggml_cuda_ar_acquire_slot; ring size2; cudaEventSynchronize for both ranks on wrap | Count/time wait calls first; then separate ring4/8 experiment preserving distinct staging buffers and all lifetime dependencies | Reduced enqueue gaps on the critical path; correctness under long repeated and alternating-size reductions |
| Copy-engine crossover | GGML_CUDA_AR_COPY_THRESHOLD=1MiB, measured in actual wire bytes | Process-local512KiB/1MiB/2MiB and selected kernel-only diagnostic; observe actual route counts | Better prefill/cached-append wall time for identical work, not just a favorable microbenchmark |
| Chunk size | GGML_CUDA_AR_COPY_CHUNK_BYTES default clamp(wire_bytes/4,512KiB,2MiB);256KiB minimum | Process-local256KiB/512KiB/1MiB/2MiB against heuristic at actual prefill shapes | Lower copy/event overhead without losing transfer overlap |
| Rank launch skew and delayed reduction | ggml-backend-meta.cpp comm_allreduce; per-GPU launch order; ggml-cuda.cu dispatcher | HIP API/kernel timelines aligned to stage boundaries, plus tensor-size/call histograms | Identify whether first-rank kernel time is mostly waiting for late peer work; avoid optimizing a misleading sum of GPU durations |

The previous Core trace averages about63.1us for rank1 small-reduction kernels versus12.3us for rank2, over a mixed whole-process profile. That asymmetry is a lead for arrival skew/overlap, not proof one GPU transports slower or a forecast of recoverable time. Those intervals overlap with other work and contain in-kernel peer waits.

## Execution plan and gates

1. Build a standalone two-GPU collective screen using the existing backend communication entry points; verify F32 inputs against the current BF16-wire/F32-sum arithmetic on both GPUs. Use representative decode/MTP rows and prefill rows, crossover boundaries and alternating small/large histories. Record exact arrays, path, logical bytes, enqueue time and total completion time. Infer no model speed from this alone.
2. Instrument an isolated library to collect actual model reduction sizes, modes, fallback counts and slot-wait time. Keep diagnostic timings separate from clean A/B timings. Include fresh and accumulated request histories because the previous P2P regression depended on history.
3. Screen existing process-local copy settings first. Then change one launch/ring parameter at a time in separate baseline worktrees/builds, with default behavior preserved. Keep all fences and buffer-lifetime dependencies until a separately verified replacement exists.
4. Test promising choices against qualified scheduling/Q6 at8K/32K/64K/128K/192K/near262K, matched MTP2, model, prompt grid, checkpoints and output work. Measure PP, cached TG, cached small appends, history edit/restore, and complete outputs. Interactions with packed-Q8 remain a separate composition gate. Require repeated order-balanced measurements for small gains.

GGML_CUDA_AR_BF16_THRESHOLD stays1 in transport comparisons. BF16 versus F32 changed outputs in prior testing; do not mix a precision change with a claimed transport improvement. RCCL is a separate provider and may be evaluated only using existing installed dependencies and an isolated build; no ROCm/RCCL/driver installation or system change is authorized.

The initial source audit was read-only. The subsequent standalone screen temporarily paused the idle selected trial and restored its unchanged configuration. Evidence: root SOURCE_AUDIT.json, upstream/MANIFEST.json, pinned source copies and diffs. Source locations: ggml/src/ggml-cuda/allreduce.cu, ggml/src/ggml-cuda/ggml-cuda.cu, ggml/src/ggml-backend-meta.cpp and ggml/src/ggml-cuda/vendors/hip.h.


## Completed settings screen

Harness commit1473867 uses the qualified scheduling/Q6 libraries unchanged. It calls the public communication backend directly with two F32 tensors, fixed BF16 transport, internal provider and Direct-P2P/fused-boundary disabled. Seven process-local settings were run in forward then reverse order, with eighteen representative element counts, four timing repeats per shape and64 timed consecutive collectives per repeat. These shapes are not yet a measured model histogram. Timing includes host submission and completion on both GPUs, with no concurrent inference or build job.

All14 processes passed:756 full-array arithmetic cases (both ranks active or either inactive),448 alternating small/large history cases, each checked on both cards, plus sampled output checks after every72-call warmup/timed sequence. All required communication calls returned handled=true. This excludes generic fallback at the public call boundary; copy versus kernel route is inferred from the unchanged dispatch and fixed precision, not separately instrumented. There were1,008 timing points. Baseline library and executable hashes matched before and after the screen.

Pooled medians across both process orders, microseconds per whole collective:

| Wire bytes | Default | Candidate | Candidate latency | Latency reduction |
| --- | ---: | --- | ---: | ---: |
| 1MiB | 384.10 | Copy threshold2MiB | 171.42 | 55.37% |
| 1.25MiB | 479.95 | Copy threshold2MiB | 225.64 | 52.99% |
| 1MiB | 384.10 | Fixed copy chunk2MiB | 348.83 | 9.18% |
| 1.25MiB | 479.95 | Fixed copy chunk2MiB | 373.95 | 22.08% |
| 2MiB | 641.91 | Fixed copy chunk2MiB | 519.07 | 19.14% |
| 2.5MiB | 711.98 | Fixed copy chunk2MiB | 644.76 | 9.44% |
| 5MiB | 1113.32 | Fixed copy chunk2MiB | 1064.55 | 4.38% |

Each listed benefit also appeared separately in forward and reverse process order. Raising the threshold keeps messages below2MiB on the kernel path. This suggests the default switches to the copy pipeline prematurely on this setup in the standalone workload. Conversely, lowering the threshold to512KiB made a512KiB collective roughly175% slower (94.54us to259.65us), and256KiB copy chunks were poor at large sizes. Reject those settings for the next model screen.

Decode-sized messages use the same small kernel across these settings. Their observed small timing variation is not evidence of a tuning gain. The threshold and chunk changes are not additive, and the combination has not been tested. Transport overlap with model compute may change the ranking. Do not translate the above percentages into PP or TG gains.

Next: collect actual model message-size and path histograms, then compare default versus threshold2MiB, chunk2MiB and a separately validated combination across the full context curve and cached edit/restore history with matched MTP2. The qualified Q6 build remains the control. If the kernel path still wins at larger sizes, screen additional thresholds separately before choosing one. Profile slot waits and rank arrival skew before changing ring depth or block count.

Evidence: screen/PASSED.json, results.json, per-process logs/configuration, build.json and RESTORED_INTEGRITY.json. Run session72063 finished with exit0. Independent verification confirmed the original selected Vivi trial restored as PID2397473, healthy, with unchanged argv, binaries, mapped libraries and APEX gates. No production configuration or baseline artifact was modified.


## Model-path diagnostic and extended crossover screen

The isolated diagnostic build replaces only allreduce.cu and ggml-cuda.cu objects in a copy of the qualified Q6 build. Pure instrumentation commit e797ebe adds optional per-pipeline message/route counts, per-rank CPU event-wait duration, and public dispatcher handled/fallback logging. It does not change transport or arithmetic. The performance comparison uses the original qualified binaries, without this instrumentation.

Three matched requests (8K cold,8K cached repeat,32K growing prefix) produced identical full first-step logits,128-token rollouts, cache/work counts and draft acceptance. All28,272 public collective calls were handled by the internal provider; no fallback occurred. There were8,580 copy-path calls carrying5MiB each and19,692 kernel-path calls carrying10/20/30KiB. Therefore the original threshold2MiB microbenchmark win does not apply to these ordinary model requests: none used wire messages between1MiB and2MiB. Small cached appends still need separate coverage.

Diagnostic slot-event waits summed to roughly26.56s on rank0 and2.93s on rank1 over the full run. These are CPU synchronization waits for events after queued model work, not isolated transport cost or a forecast of removable time. Increasing ring depth could affect launch overlap, but this observation alone does not justify removing waits or fences. Diagnostic logging can perturb scheduling.

That actual5MiB message size justified an additional standalone screen: default, fixed2MiB chunks, and threshold8MiB, each in forward and reverse order. All6 processes passed324 full-array arithmetic cases and192 alternating-history cases, each on both cards, plus repeated-output samples and432 timing points. At5MiB, forward medians were1112.82us default,1065.63us chunk2MiB,826.95us threshold8MiB; reverse medians1107.46us,1058.92us,829.91us. Keeping5MiB messages on the chunked kernel path reduced collective latency by about25%, materially more than the chunk-size-only change. This is still standalone latency, not PP gain.

Diagnostic session41624 and extended screen66582 finished with exit0. The selected original Vivi trial was independently verified restored after each, most recently PID2407541. Evidence: model-diagnostic/PASSED.json, comparisons.json, diagnostic/server.log, screen-extended/PASSED.json and both RESTORED_INTEGRITY.json files.

The next clean model run is now scoped to default versus threshold8MiB, using identical unchanged qualified Q6 binaries and fixed BF16. It includes six context depths8K through261632, two cached repeats each,1/16/64/128/256-token appends at32K/64K/128K/192K, and near-full-history edit/restore. Forty requests per lane, matched MTP2. Full logits,128-token rollouts, work/acceptance counts and checkpoint reuse are checked. A small preliminary gain needs reverse-order repetition. Slot-wait/block-count tuning and packed-Q8 integration remain pending separate experiments.


## Completed full-model threshold8MiB comparison: reject

Session45017 completed with exit0. The unchanged qualified scheduling/Q6 binaries were used for both lanes, with fixed BF16 transport and MTP2. Eighty requests covered six depths, cached repeats, twenty small-append cases per lane, and near-full history edit/restore. All40 paired first-step full-vocabulary logits,128-token rollouts, work counts and draft acceptance counts matched exactly. All26 within-lane repeat/restore comparisons matched. Both lanes reused233472 tokens for each deep edit/restore, processing28160 new tokens.

| Context | Default PP tok/s | Threshold8MiB PP tok/s | PP change | Default cached TG | Threshold8MiB cached TG |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | 1132.75 | 1120.69 | -1.06% | 65.84 | 65.27 |
| 32768 | 1057.83 | 1045.38 | -1.18% | 66.40 | 66.42 |
| 65536 | 911.15 | 889.17 | -2.41% | 61.28 | 61.29 |
| 131072 | 740.76 | 718.00 | -3.07% | 58.33 | 58.28 |
| 196608 | 589.27 | 572.17 | -2.90% | 53.58 | 53.54 |
| 261632 | 490.82 | 477.05 | -2.81% | 48.63 | 48.54 |

PP is growing-prefix throughput on identical processed/reused token counts; only8K is a cold request. Cached TG averages two128-token windows at each depth. The run is one forward process order, not a population-level significance claim. No reverse-order repeat is warranted for promotion: this candidate shows no useful model gain, and PP is slower at every depth. Reject threshold8MiB as an optimization for this workload. Do not reinterpret the25% isolated collective latency reduction as a model speedup.

Small append timing is also not a compelling win: mixed small changes, mostly slower at deep context. The first one-token append at each depth reprocessed a512-token cached tail plus the new token; later16/64/128/256-token appends reused the full prefix. Append prompt_ms measures the server prompt phase, not full client-observed TTFT. The precision/transport change did not cause an observed output or checkpoint regression.

Why the isolated win disappears is not established by this experiment. Compute/copy overlap, rank readiness and the larger number of kernel chunks are candidates for a targeted timeline; do not claim one is proven. Further copy-chunk tuning and small-message launch/ring experiments remain distinct leads. The2MiB threshold has no effect on the ordinary5MiB prefill and10/20/30KiB decode messages observed, so it is not a general PP/TG solution.

Evidence: model-curve-threshold8m/PASSED.json, comparisons.json, ANALYSIS.json, per-lane results/configuration/runtime/logits and RESTORED_INTEGRITY.json. The selected original Vivi trial was restored unchanged as PID2430872 and independently verified healthy. The allreduce settings remain at their original defaults.


## Follow-up external research and provider precision audit

A public dual-gfx1201 report describes RCCL collective deadlock, with a reported NCCL_PROTO=Simple workaround. A maintainer attributes it to gfx12 LL protocol selection and missing ordering fixes in that protocol; the discussion links ROCm/rccl PR2187. This is an RCCL-specific lead, not evidence our internal transport suffers the same failure. Our28,272-call diagnostic observed no provider fallback or hang. References: https://github.com/ROCm/rocm-systems/issues/5480 and https://github.com/ROCm/rocm-systems/issues/5480#issuecomment-5360352004 . No protocol flag or library was changed.

Pinned vLLM source at ed3f6d1a56272a966bd9e514cb262c0ce3ee6822 restricts its standard custom-allreduce enablement to gfx94/gfx95, so its custom path is not direct evidence of a supported gfx1201 replacement. Source: https://github.com/vllm-project/vllm/blob/ed3f6d1a56272a966bd9e514cb262c0ce3ee6822/vllm/platforms/rocm.py . Pin records and retrieved source are in external-research/MANIFEST.json.

Read-only local inventory found installed RCCL headers and librccl.so.1.0.70200 under /opt/rocm-7.2.0. This makes a future isolated build using the existing library possible in principle; it does not prove the provider works on this host. No dependency install, rebuild, or system change was performed.

The stock NCCL/RCCL function in our pinned ggml-cuda.cu is not numerically equivalent to the internal BF16-wire path. With two GPUs it reduces ne<32768 directly in FP32; larger tensors are converted to BF16, reduced into a BF16 output, then converted to FP32. Internal mode with BF16_THRESHOLD=1 rounds inputs to BF16 but adds the two values into FP32. Thus a provider switch changes both small-message input precision and large-message result rounding. For example, representable BF16 inputs1 and2^-9 have an exact FP32 sum1.001953125; storing the sum in BF16 rounds it again. This is a source-level numerical distinction, not a measured RCCL model result.

Do not call a stock RCCL comparison a precision-preserving transport A/B. A future compatibility variant would need to match the existing rounding contract, for example BF16 exchange followed by local FP32 addition, before attributing output or speed differences to transport alone. That is a larger change than testing internal ring depths4/8 or copy chunks2MiB. Preserve all existing event and buffer-lifetime fences in those smaller experiments. The pipeline's ring size controls host staging/event arrays, while the large copy buffers have separate read/add completion fences; changing only ring depth does not eliminate those dependencies.


Additional read-only checks: installed rccl.h declares NCCL version2.27.7; provider functionality was not exercised. RCCL PR2187 was still open when inspected, so do not describe the proposed protocol change as a merged fix. Reference: https://github.com/ROCm/rccl/pull/2187 .

The pinned vLLM rocm.py sets GPU_PINNED_MIN_XFER_SIZE to4194304 and describes the units as KB, intending4GiB, to keep mmap weight sources on the HIP staging path and avoid queue suspension caused by page-registration MMU notifications. That is a separate checkpoint-transfer lead, not an established explanation. Follow-up release-source audit found our ROCm7.2 tag uses MiB for explicit overrides and that rectangular host copies can bypass the threshold. Do not adopt that numeric recipe or call it4GiB on this stack without verifying the actual runtime. See CHECKPOINT_TRANSFER_LEAD_20260930.md for local source routing, the units correction and minimal diagnostic. No environment setting was changed. Explicitly pinned internal allreduce buffers are a different path.

## Small-message launch geometry lead

Source-level calculation for the measured BF16-wire decode messages: HIP uses16-byte vector copies, while the small kernel always launches8 blocks of256 threads. A10KiB tensor contains640 vectors, so only3 blocks carry vector data and5 still participate in host-flag synchronization without payload.20KiB uses5 payload-bearing blocks;30KiB uses8. These exact message sizes have no scalar tail. This is redundant launched work, not a measured performance loss.

A minimal isolated alternative can clamp the grid to the blocks needed by the current chunk while retaining maximum8-block arrival storage, the2-slot ring and every system fence. Both ranks must use the same grid and stripe mapping. Keep the scalar tail assigned to block0 and ensure at least one block. Compare against fixed2/4/8/16 geometry only as separate variants; do not combine with larger ring depth until independent benefit and exact arithmetic are established. Potential benefit is decode/MTP latency; prefill's copy route remains unchanged. The existing collective harness should cover the exact observed sizes, inactive ranks, odd tails and alternating histories before full-curve model testing.

The ring-depth experiment is separately prepared at20260930-allreduce-ring, baselinecbc3985, pure patche8fd1a8. It preserves existing synchronization and has a queued independent-buffer correctness screen. No ring binary has been built while model session13600 is running.

## Completed chunk2MiB model comparison: defer, retain defaults

Session13600 completed with exit0. The initial port preflight refused an immediate restart after the preceding experiment; no service was stopped by that refused attempt. A read-only check confirmed the port free before the successful run. Eighty model requests used unchanged qualified scheduling/Q6 binaries, matched MTP2 and fixed BF16 transport. All40 paired full first-step logits,128-token outputs, work counts and draft acceptance matched exactly. All26 within-history checks passed. Deep edit/restore reused233472 tokens and processed28160 in both lanes.

| Context | Default PP tok/s | Chunk2MiB PP tok/s | PP change | Default cached TG | Chunk2MiB cached TG |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | 1131.84 | 1132.14 | +0.026% | 65.86 | 65.36 |
| 32768 | 1058.26 | 1066.18 | +0.748% | 66.25 | 66.29 |
| 65536 | 910.22 | 913.03 | +0.309% | 61.18 | 61.27 |
| 131072 | 740.16 | 741.58 | +0.191% | 58.25 | 58.27 |
| 196608 | 589.14 | 590.38 | +0.210% | 53.44 | 53.51 |
| 261632 | 491.24 | 491.50 | +0.053% | 48.52 | 48.55 |

This is one forward process order. Growing-prefix PP shows a small positive difference, strongest at32K, but nearly disappears at the workload's maximum depth. Cached TG is essentially unchanged except8K, which is0.763% slower despite using the same small-message kernel. Small-append latencies are mixed; near-full edit PP is unchanged, restore differs by0.065%. No statistical improvement or regression is established. Do not change defaults or claim the standalone4% collective gain as a model benefit. Keep this as a low-priority unconfirmed candidate; reverse-order support is prepared if confirmation later becomes worthwhile. Prioritize the separate ring/geometry and large-message peer-copy leads for now.

PP is measured on growing cached prefixes, not cold-full-context input except8K. Cached TG averages two128-token windows; append prompt_ms is not client TTFT. Evidence: model-curve-chunk2m/PASSED.json, ANALYSIS.json, comparisons.json, per-lane logs/results/runtime and RESTORED_INTEGRITY.json. Independent verification confirmed original selected grouped MTP1 Vivi restored unchanged and healthy as PID2513894. No production configuration or dependency change occurred.

## Subsequent isolated kernel experiments

| Experiment | Completed evidence | Decision |
| --- | --- | --- |
| Ring2/4/8 | 1080 exact full-array/history cases on both ranks;576 timings in both orders. No useful gain at actual model sizes; ring8 slower at30KiB. | Keep ring2. No model gain claimed. |
| Empty-block elimination | 1134 exact cases;10/20KiB faster. Full six-depth model:21 exact pairs,26 history checks, checkpoint coverage intact; near262K PP490.31->490.26,TG48.58->48.51. | Do not retain. Too few MTP2 reductions affected. |
| Fixed blocks8/4/2/1 | 1890 exact cases;720 timings.4 blocks improve30KiB by~12.7%, but regress160KiB and near1MiB. | Reject globally smaller grids. |
| Size-based4 blocks<=32KiB,8 above | 1134 exact cases including queued4/8 transitions;432 timings.30KiB~21->18.3-18.6us. Severe large-message regression removed;160KiB still1.4-2.9% slower in this screen. | Model validation pending. |
| Large BF16-to-F32 peer copy | 1134 exact cases including queued source reuse, inactive ranks, tails and outer chunks;432 timings.5MiB~1113-1118->607us, about45.5% latency reduction in both orders. | Both model orders passed42 exact pairs and52 history checks;24 completed tasks exact. PP+3.84-9.41%; cached TG up to1.86% slower. Retain optional prefill candidate. |

All worktrees start from frozen cbc3985; separate binaries replace only allreduce.cu within the qualified scheduling/Q6 build and verify reused-object and baseline hashes. The original selected grouped MTP1 trial is restored and independently checked after each GPU job. Current live state is in the peer-copy experiment's ACTIVE_HANDOFF.json; source reports are under each corresponding20260930-allreduce-* /src/apex directory.

The new peer-copy candidate changes only large BF16-wire/FP32-result transport. It does not enable the old small-message APEX P2P kernel or add BF16 result rounding. Both copy completions are ordered before compute-stream add and pooled source reuse, and prior scratch-reader completion remains enforced. No host/dependency changes. Installed rocprofv3 tracing is prepared after the model run to distinguish actual SDMA versus shader-copy routing; API name alone is insufficient.

Adam confirmed x8/x8 is the expected lane split with both GPU slots populated. Do not present that width as a defect. A separate read-only speed observation remains: first upstream path32GT/s, second16GT/s, both reporting32GT/s maximum. Cause not established, no host changes made, and every A/B retains the same hardware configuration.

## Final peer-copy lead result and review boundary

Peer-copy model64991, reverse30394, completed-task23771 and diagnostic trace28185 all completed successfully. Two-order pooled near262K PP490.00 ->508.80 tok/s (+3.84%), cached TG48.53 ->48.32 (-0.43%). Across the curve PP+3.84-9.41%, while cached TG has a repeatable32K cost of1.86% and smaller64K/128K costs.42 model pairs and52 within-history checks are exact;24/24 completed objective tasks match full messages, first-step logits and work counts. Deep checkpoints retain232960 tokens. The trace associates all2456 chunks at actual5MiB with HSA copy-engine calls; smaller chunks can use shader copies. No profiled timing is used as a speed result.

Retain this isolated peer-on build as an optional prefill-oriented candidate, preserving the peer-off qualified scheduling/Q6 option for generation. It does not activate the old small-message P2P patch. The selected original grouped MTP1 trial was independently restored unchanged as PID2617073. No GPU test remains active. Adam requested finishing this lead and reviewing findings; size-grid model, event-overlap changes and other research are deferred. Full consolidated review: ../../../20260930-allreduce-peer-copy/src/apex/DISCOVERY_REVIEW_20260930.md . Peer implementation/results: ../../../20260930-allreduce-peer-copy/src/apex/ALLREDUCE_PEER_COPY_20260930.md .
