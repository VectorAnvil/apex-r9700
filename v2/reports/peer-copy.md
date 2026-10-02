> Historical research record. Statements about the selected service are as of this report, not current deployment status. See [the V2 index](../README.md). Local raw evidence, private runtime files and logs are not included; local artifact paths are provenance references, not downloadable links.

# Large AllReduce peer-copy experiment

Complete: standalone, both model/history orders, completed-task compatibility and transport trace passed. RETAIN as an optional qualified experimental prefill improvement; selected serving setup unchanged. Baselinecbc3985 worktree, transport change075c19f plus scope restriction644bc89. APEX_V2_AR_PEER_COPY defaults0; separate build-peer0 and build-peer1. Enables in-process HIP peer access on both selected devices; fails explicitly if peer access is unavailable. No driver, system, global environment or production changes.

Scope: only large BF16-wire/FP32-destination reductions, which match the measured Q6 model path. All other datatype combinations retain original host staging. Small reductions retain original8-block/ring2 implementation. Original copy threshold, chunk heuristic and32MiB outer chunker remain. Retain original host allocations to avoid conflating allocator footprint with transport. Peer-copy API use is not proof of SDMA execution; a runtime trace would be needed for that claim.

Ordering: zero inactive data before source-ready events; record both source-ready events before either peer stream waits on its peer. Each copy stream waits for its own previous scratch reader, then copies its peer's prepared BF16 contribution into local scratch. Record both copy completions before either compute stream waits on them. Each compute stream waits for BOTH completions before the unchanged FP32 add kernel. This protects peer source reads from pooled BF16 buffer reuse, including queued calls. Record original scratch-reader completion and event-ring ker completion. No host-wide synchronization added; original acquire_slot waits remain.

Source was written against local upstream-derived allreduce.cu, without importing external fork code. Motivation and contrasts are documented in ../20260930-allreduce-geometry/src/apex/ALLREDUCE_EXTERNAL_LEADS_20260930.md and the allreduce-tuning RDNA_BOOSTS audit. The old APEX P2P kernel is still disabled: this uses a different large-message transfer mechanism and preserves FP32 sum output.

Validation queue: first finish geometry model85368, independent restoration, then the already-prepared block-count sweep. No concurrent builds/GPU timings. This work then builds peer0/1 by replacing only allreduce.cu in qualified scheduling/Q6 binaries, with all reused object and baseline hashes verified. Standalone screen is qualified/peer0/peer1 in both orders,1134 exact full-array/history cases on both ranks including294 queued checks,432 timings, inactive ranks, scalar/vector tails and >32MiB outer-copy coverage. Initialization log must match the selected peer-copy mode. CPU oracle models BF16 rounding plus FP32 sum exactly. Native FP16/BF16-only and unrounded FP32 transport are deliberately left on the original path.

Model/history gates and whole-curve speed validation are required before any adoption. Correctness and speed claims below are limited to the completed standalone screen. Useful transfer-only gains advance to model testing; precision/output/work counts must remain exact, and small gains need reverse-order confirmation.

## Completed standalone screen

Build96908 and screen85915 exited0. All1134 full-array/history checks passed on both ranks, including294 queued-history cases, and432 timing points. Peer-enable log matched the selected variant and all loaded libraries matched their build directories. Original grouped MTP1 trial independently restored as PID2559916.

| Wire bytes | Qualified us forward/reverse | Rebuilt peer-off | Peer-on |
| --- | --- | --- | --- |
| 10240 | 19.478/19.489 | 19.364/19.459 | 19.731/19.370 |
| 20480 | 19.851/19.617 | 20.266/20.483 | 19.584/19.865 |
| 30720 | 21.348/21.215 | 20.824/21.098 | 20.405/20.898 |
| 1048576 | 381.710/384.132 | 381.807/383.426 | 277.276/277.057 |
| 1310720 | 479.685/479.911 | 479.865/480.056 | 335.838/337.471 |
| 2097152 | 640.296/634.986 | 639.969/640.356 | 435.945/435.339 |
| 2621440 | 712.047/712.391 | 709.806/710.423 | 475.473/475.995 |
| 5242880 | 1117.723/1112.839 | 1114.384/1115.639 | 606.710/607.422 |

At actual5MiB prefill size, whole-collective latency dropped about45.5% in both orders. Small-message path unchanged with mixed small timing variation. Advance peer-copy before size-grid to the six-depth21-pair model/history curve; its larger transport saving offers a clearer discriminator. This does not establish a model throughput gain or identify the actual copy engine used. Source-buffer lifetime and precision gates passed only in the tested workloads; whole-model checkpoint/parity validation remains necessary.

## Prepared transport trace

Source-only review of pinned ROCm/clr rocm-7.2.0 shows size-sensitive peer-copy routing: KernelBlitManager::copyBuffer compares peer transfer size against sdma_p2p_threshold_; settings derive that from ROC_P2P_SDMA_SIZE in KiB, default1024. The5MiB transfer heuristic creates1.25MiB chunks, whereas a1MiB transfer creates512KiB chunks. These can select different internal engines. This is a release-source inference, not exact installed-binary evidence. Existing manifests pin the audited source at c5824370effd4609972653f6490f7313f5bc18af under the allreduce-tuning checkpoint-source-audit directory. No environment setting was changed.

Prepared trace.py uses the installed rocprofv3 for HIP runtime, HSA AMD, kernel and memory-copy traces of qualified and peer1 standalone harnesses. Run only after model64991 completes and restoration is verified. It reuses built binaries, checks exact outputs and restores the selected trial. Profiled timings must not be used as speed evidence. Inspect whether5MiB peer transfers produce expected device-to-device copies and whether smaller chunks invoke shader-copy kernels. No profiler, runtime or driver installation is needed.

## Possible follow-up after current model result

The first peer-copy variant deliberately reuses ggml_cuda_ar_wait_for_compute, which records the local source-ready event and also makes the local copy stream wait for it. A pull from the peer needs the peer's source-ready event and local scratch-release event; it does not inherently need the current local source to be ready. Thus the current implementation can delay a peer pull behind unrelated local producer computation. A separate follow-up could record each local source-ready event without the extra local copy-stream wait, retaining peer-source readiness, previous scratch-reader completion and BOTH current copy-completion waits before add/reuse. Do not change the running variant. Require queued correctness and model validation for such an overlap change. This is a source-derived opportunity, not measured removable time.

## Forward whole-model result

Session64991 exited0. All21 paired first-step full logits,128-token outputs and work/draft counts match exactly; all26 within-history comparisons pass. Near-full edit and restore reuse232960 tokens and process28672 on both builds. Independently restored selected original trial PID2581885.

| Context | Control PP | Peer PP | PP gain | Control cached TG | Peer cached TG | TG change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|8192|1127.57|1232.68|+9.32%|67.33|67.00|-0.49%|
|32768|1056.43|1151.75|+9.02%|66.27|64.88|-2.11%|
|65536|909.97|975.86|+7.24%|63.73|63.21|-0.82%|
|131072|740.51|782.02|+5.61%|59.75|59.35|-0.68%|
|196608|589.40|615.23|+4.38%|53.60|53.54|-0.10%|
|261632|489.63|508.43|+3.84%|48.55|48.33|-0.46%|

PP is growing-prefix throughput; cached TG averages two128-token windows. Deep edit/restore PP466.42/466.57 ->484.95/485.10 tok/s. Do not accept the TG differences as noise without confirmation: reverse the process order with the same six-depth/checkpoint gate. If benefit persists, complete24 saved-reference objective tasks and the prepared diagnostic transport trace. Adam requested finishing this lead and necessary checks, then reviewing findings; size-grid model and overlap variants remain deferred.

Runtime scope: both model logs report ROCm NO_VMM=1. Process-local hipDeviceEnablePeerAccess applies to the tested ordinary allocations. This does not qualify VMM allocations, other device counts, CUDA or other datatype contracts; a generic upstream port would need its own allocation/access coverage. The candidate does not set GGML_CUDA_P2P or enable old APEX_V2_DIRECT_P2P.

## Reverse confirmation and pooled model curve

Session30394 exited0 and independently restored the selected trial PID2603119. Both process orders pass:42 paired full first-step logits,128-token outputs and work/draft counts;52 within-history checks; deep edit/restore reuse232960 tokens. Initial reverse launch refused a recently occupied benchmark port before stopping any service; read-only checks found it free before retry.

Equal-work harmonic means across both orders:

| Context | Control PP | Peer PP | PP change | Control cached TG | Peer cached TG | TG change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|8192|1129.64|1235.94|+9.41%|67.14|67.17|+0.05%|
|32768|1055.84|1153.11|+9.21%|66.16|64.93|-1.86%|
|65536|907.69|977.10|+7.65%|63.68|63.21|-0.74%|
|131072|737.07|782.51|+6.17%|59.67|59.36|-0.52%|
|196608|589.30|615.38|+4.43%|53.53|53.57|+0.06%|
|261632|490.00|508.80|+3.84%|48.53|48.32|-0.43%|

The prefill benefit reproduces in both orders, including near262K+3.841%/+3.831%. Cached TG is a tradeoff:32K-2.105%/-1.612%,64K-0.816%/-0.659%,128K-0.679%/-0.361%,near262K about-0.4% in both orders.8K/192K changes reverse sign. Do not call this a generation optimization or claim the repeated slowdowns are all noise. The tested small-kernel implementation is unchanged; the cause of the small TG penalty is not isolated.

The main value is large uncached spans: near262K processing65536 tokens saves about4.9seconds; edited-history28672-token prompt saves about2.1-2.35seconds. Cached512-token-tail plus128-token generation server times remain close; MODEL_SUMMARY.json records sums of prompt_ms and predicted_ms, not client TTFT. No general small-append performance claim. Two process orders establish repeatability of this fixture, not a population-level confidence interval.

## Completed-answer compatibility

Session23771 exited0. All24 tasks at131072/260096 pass, with exact complete messages, full first-step logits and prompt/cache/generation/draft work relative to the saved qualified scheduling/Q6 reference.5157 output tokens; six tool-call finishes and18 normal stops, no length truncations. Tool names/arguments were graded, not executed. Reference artifacts, request bodies/output budgets and grader hashes matched. This is transport compatibility with the original grouped kernel retained, not a new grouped-versus-Core accuracy study. Selected original trial independently restored as PID2614998. Evidence: capability-check/PASSED.json, comparisons.json, peer-mtp2/results.json, runtime.json and RESTORED_INTEGRITY.json.

## Completed transport trace

Session28185 exited0. Both profiled harnesses pass189 full-array/history cases on both ranks (378 combined); profiled timings are excluded from performance conclusions. Independently restored original selected trial PID2617073, healthy with original artifacts, libraries and gates.

The peer trace contains16554 hipMemcpyPeerAsync calls.13970 correlate to __amd_rocclr_copyBuffer shader dispatches;2584 contain hsa_amd_memory_async_copy_on_engine calls with matching device-to-device completion records. For the actual5MiB message, all2456 chunks across307 collective calls use the HSA copy-engine route, with1.25MiB per chunk and no peer-correlated shader dispatch. Smaller chunks use the shader path. This confirms size-dependent routing on this installed runtime; it does not identify a physical SDMA engine number.

Transfer sizes are reconstructed from the hash-pinned harness's deterministic call order and unchanged source chunk heuristic; this CSV omits byte counts. Same-thread timestamp containment joins nested HSA calls to HIP calls, and correlation IDs join HSA calls to copy completion records. The memory-copy CSV reports identical source/destination agent labels for D2D records; those labels are not interpreted as physical ownership. Explicit cross-device source/destination arguments are established by the inspected peer-copy source. No kernel or system setting was changed for profiling. Reproducer: allreduce-peer-copy/analyze_trace.py; evidence: transport-trace/ANALYSIS.json, PEER_ROUTING.json, CSV hashes, logs and restoration receipt.

## Decision and review boundary

RETAIN as a qualified experimental prefill option, not a generation optimization. Prefill gains reproduce across8K-262K, with3.84% at the deepest point. Cached TG costs about1.86% at32K,0.74% at64K,0.52% at128K and0.43% near262K in pooled results;8K/192K are effectively unchanged. These tradeoffs remain in the record. The two-order cached512-token-tail plus128-token generation server-time sums improve only0.2-1.7%; this is not a client TTFT or arbitrary agent-turn claim. Large-prefix growth and substantial history replay benefit most.

The change adds no observed numerical/behavioral regression relative to the qualified grouped/scheduling/Q6/MTP2 reference:42 exact model pairs,52 exact history comparisons and24/24 objective completed tasks with complete-message/logit/work equality. Original grouped-versus-Core accuracy differences remain unchanged and are documented in DISCOVERY_REVIEW_20260930.md.

Keep the peer-off scheduling/Q6 build as the decode-oriented option and the peer-on build as the prefill-oriented option. Both remain isolated; no new configuration was selected for Vivi. Final original grouped MTP1 trial restored and independently verified as PID2617073 on port8083. Production trees/binaries, dependencies, drivers and GPU2/Bonsai were untouched; no usage reset. Size-grid model, alternate event overlap and other new leads are deferred for Adam's review. No GPU experiment remains running.
