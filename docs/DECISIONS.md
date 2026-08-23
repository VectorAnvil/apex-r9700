# Architecture Decisions

## 2026-08-08: Keep Apex as the optimization base, conditionally

**Decision:** Retain Apex for agent backends, grading safeguards, knowledge, reflection, trajectories, and reporting. Add llama.cpp adapters instead of forcing its serving-stack path to fit.

**Reason:** These systems save meaningful work; Apex's launcher, tracer, model grader, and reintegration are Python/vLLM/SGLang oriented.

**Rejected:** Rebuilding everything around AgentKernelArena. It may reach the first POC faster but loses Apex's broader workflow.

## 2026-08-08: Treat compiled llama.cpp as a new lifecycle

**Decision:** Use pinned worktrees, allowlisted patches, controlled builds, immutable correctness harnesses, registered candidates, and Llama Lab comparisons.

**Reason:** llama.cpp requires compilation; installed Python hot-patching is unsafe and inapplicable.

## 2026-08-08: Replace dynamic tracing with ROCm profiling

**Decision:** Add `rocprofv3` capture/normalization and compiled-symbol mapping. Reuse only general Apex result concepts.

**Reason:** Apex AST-patches Python Triton/custom-op/AITER launch sites and cannot establish compiled GGML provenance.

## 2026-08-08: Keep Llama Lab as E2E authority

**Decision:** Apex submits registered baseline/candidate profiles and suites and consumes persisted results. PP, TG, and MTP-effective TG remain distinct measurements.

**Reason:** Llama Lab already owns safe argv construction, capability probing, GPU serialization, activity guards, parsing, provenance, and comparison.

## 2026-08-08: Use a llama.cpp-owned correctness reference

**Decision:** Start with Apex/Magpie `library_test` mode against an immutable GGML test or fixed-input harness.

**Reason:** PyTorch ground truth is inappropriate for Q6_K; Accordo becomes useful after a standalone harness exists.

## 2026-08-08: Add only verified RDNA4 metadata

**Decision:** Introduce R9700/`gfx1201` as RDNA4 and leave unknown properties absent until verified or measured.

**Reason:** Copying stock `gfx950`/CDNA guidance would mislead optimization agents.

## 2026-08-08: Detect hardware without a default substitution

**Decision:** Use `rocm_agent_enumerator` for architecture discovery and `rocm-smi --showproductname --json` for product identity. Detection failure or an unsupported target remains explicit `unknown`; it never becomes `gfx950`.

**Reason:** The live host exposes two `gfx1201` agents plus an integrated `gfx1036`, while stock Apex excluded the relevant rocminfo ISA line and silently supplied MI355X data.

## 2026-08-08: Share target-aware prompt hardware context

**Decision:** Kernel and model prompts consume one verified hardware-context registry. R9700 prompts state RDNA4/wave32/GDDR6/WMMA facts and treat tiling, block size, coalescing, and achieved performance as measurements.

**Reason:** Hard-coded wave64, MFMA, HBM, and MI355X prose remained incorrect even when the compile target string changed.

## 2026-08-08: Keep Phase 1 smoke independent of model infrastructure

**Decision:** Validate gfx1201 with a fixed integer vector-add kernel compiled and run in a temporary directory. Preserve target/correctness/timing evidence but require no model, serving stack, or production build.

**Reason:** This proves the toolchain and hardware foundation with minimal risk. Magpie compatibility remains blocked until a pinned Magpie installation exists.

## 2026-08-08: Use rocprofv3 kernel traces as canonical profiling evidence

**Decision:** Capture ROCm 7.2 `rocprofv3` kernel CSV with full mangled names, preserve the raw artifacts, and derive bounded Apex records by phase, agent, and exact symbol. Use profiler statistics only as a cross-check. Discover gfx1201 capabilities dynamically and leave PMCs disabled for the base pass.

**Reason:** Kernel trace timestamps are stable, correlation-rich evidence for compiled HIP workloads. Static Instinct counter sets are not valid for Radeon gfx1201, and counter passes can perturb or reject a workload.

## 2026-08-08: Isolate PP and TG with numeric llama-bench pairs

**Decision:** Build separate registered processes with `-p 0 -n 0 -pg <prompt>,0` for PP and `-p 0 -n 0 -pg 0,<generation>` for TG. ROCTx remains optional supplementary instrumentation; never use `--kernel-rename` in a source-mapping capture.

**Reason:** This llama.cpp revision parses `-pg` as a numeric pair, not `pp` or `tg` literals. It currently emits no ROCTx ranges, while kernel renaming destroys the raw symbol required for source mapping.

## 2026-08-08: Disable HIP graphs only in profiler diagnostic runs

**Decision:** Set `GGML_CUDA_DISABLE_GRAPHS=1` inside the recorded profiling boundary. Treat llama-bench throughput from that run as diagnostic only, never as a comparable Llama Lab performance result.

**Reason:** ROCprofiler-SDK 1.1.0 segfaulted inside `ggml_cuda_graph_evaluate_and_capture` on the registered build. Disabling graph capture allowed rocprofv3 to observe the same compiled kernel families without modifying or rebuilding llama.cpp, but it changes submission behavior.

## 2026-08-08: Select Q6_K token-generation MVM for Phase 3

**Decision:** Use `mul_mat_vec_q<(ggml_type)14, 1, ...>` as the primary Phase 3 target, with the patched `ggml_ar_hip_kernel<float>` retained as secondary evidence rather than optimized concurrently.

**Reason:** The two Q6_K MVM variants account for 70.903% of summed TG kernel duration across both R9700 agents. The all-reduce kernel accounts for 16.383% and has a material agent imbalance, but it is already the known-good P2P experiment patch and would expand the first optimization loop's risk.

## 2026-08-08: Use Hipfire Kernel Atlas as a Phase 3 methodology reference only

**Decision:** Adapt small post-rocprof concepts from `maxhbr/hipfire` commit `a99b46438a33e1471d7e871bbb32dd77d5e209f6`: immutable task/eval contracts, fresh-baseline variance handling, SHA-256 binary/diff/code-object provenance, evidence-ranked source/dispatch candidates, optional AMDHSA/ISA metadata extraction, and exact-fingerprint attempt history. Do not add Hipfire as a dependency or change the Phase 2 capture adapter.

**Reason:** Kernel Atlas has useful experiment-discipline implementations, but its single-process phase parser, Hipfire-specific operation rules, generic command runner, MD5 provenance, static gfx1201 capability table, and unstable-result history policy do not satisfy Apex's llama.cpp safety and evidence requirements without adaptation.

## 2026-08-08: Keep hipBLASLt conditional on verified dense GEMM evidence

**Decision:** Distinguish custom GGML quantized kernels from vendor dense GEMM in every normalized profile. Require a separate official library-log capture for exact M/N/K, layouts, batching, datatypes, API/backend, and algorithm provenance before considering hipBLASLt. Preserve the maintained `ROCm/rocm-libraries` QuickTune, offline override, measured algorithm selection, Stream-K, and gfx1201 solution paths as conditional Phase 3 candidates only.

**Reason:** The complete Phase 2 PP/TG kernel statistics contain no rocBLAS, hipBLAS, hipBLASLt, Tensile, or equivalent dense GEMM dispatch. PP and TG are dominated by custom GGML Q6_K MMQ/MMV kernels, while rocprofv3's HIP runtime trace contains no BLAS argument payload. hipBLASLt cannot directly optimize that custom quantized hotspot, and no M/N/K or datatype record may be inferred from a library kernel name.

## 2026-08-08: Gate SuperSonic concepts on exact llama.cpp hotspot evidence

**Decision:** Preserve `DeanoC/SuperSonic` commit `b667b645efce909d5a8fee8a227ad17f0db050af` as a Phase 3 concept and experiment-method reference only. Do not copy or port its code. Activate an independently derived Q6_K MLP-down specialization, gate/up plus SwiGLU fusion, residual epilogue, lm-head argmax, gfx12 WMMA adapter, or launch-bound experiment only after rocprofv3 and source evidence establish the corresponding llama.cpp operation, exact shape/types/layouts, phase share, dispatch adjacency, and gfx1201 code-object facts. Require a rollback switch, parity-first contract, variance-aware same-build A/B, and exact-fingerprint attempt history.

**Reason:** Phase 2 proves that generic custom Q6_K MMQ and MMV families dominate PP and TG, but it does not attribute them to SuperSonic's operations or `m=16` shapes. The reviewed kernels belong to SuperSonic's Qwen3.5 full-attention path used by a distinct gfx1100 DFlash workload, its gfx1201 support lacks checked-in ISA/resource evidence, and the pinned repository contains no tracked license file. Its successes, reversions, and parity log are valuable hypothesis and methodology evidence, not transferable performance or implementation evidence.

## 2026-08-08: Keep Fluke/FlyDSL as an evidence-gated prototype path

**Decision:** Preserve `BonsonW/fluke` commit `7460d64f3d638223dd4c33084abb48e448056b54` and Apache-2.0 `ROCm/FlyDSL` commit `2d65dea1380876f4faa68159cadfec1eeaa598ab` as Phase 3 references for gfx1201 WMMA layout, preshuffled weights, register-direct and software-pipelined GEMM, operator fusion, and target-specific AOT deployment. Do not copy Fluke code or artifacts. Permit an isolated FlyDSL prototype only after rocprofv3 establishes an exact corresponding operation and the llama.cpp-owned harness, oracle, inputs, and evaluator are frozen. Require explicit gfx1201 compilation, IR/HSACO/ABI provenance, code-object resource inspection, fail-closed dispatch, rollback, parity, and variance-aware A/B.

**Reason:** Fluke provides a concrete FP8 E4M3 wave32 experiment and embedded-HSACO C-ABI design, but its pinned tree is unlicensed, fixed to different layouts and small model dimensions, and has no committed RDNA4 performance evidence. The current Apex workload is Q6_K: its dominant custom GGML MMQ/MMV kernels do not activate FP8 dense GEMM or Fluke's fusion paths. FlyDSL is nevertheless a strong licensed authoring mechanism for future exact-shape gfx1201 prototypes, and becomes more directly relevant only if a separately registered and profiled FP8 Qwen workload proves a material matching hotspot.

## 2026-08-08: Preserve Zinc K-parallel Q6_K as a gated PP reference

**Decision:** Preserve MIT-licensed `fusion44/zinc` commit `8c6b5988ac1335b0ea156f575f453bc1891b3362` as a Phase 3 reference for Q6_K batched K-parallel matrix-vector work, one-row wave64 cooperation with a cross-subgroup fallback, register-resident column accumulation, and reset-and-replay validation. Do not integrate or port Zinc. Treat PP Q6_K as a proven family-level hotspot because `mul_mat_q<14,128,...>` accounts for 85.011% of summed PP dispatch duration, but require exact llama.cpp operation, M/N/K, types/layouts, dispatch/source, and loaded-code-object attribution before proposing a corresponding port. Keep TG Q6_K MMV as the first Phase 3 target.

**Reason:** Zinc's `dmmv_q6k_batch_kpar` is a concrete gfx1201-oriented design, but it is Vulkan GLSL batched DMMV rather than llama.cpp HIP MMQ. Its `MAX_COLS=40` setting is supported by a Q4_K Qwen3-8B/R9700 sweep and is merely mirrored by Q6_K, so it cannot be transferred without a fresh Q6_K resource and variance sweep. Zinc's validate mode is immediately useful methodology because it resets state, replays a trusted per-token path, compares final logits, and can capture intermediate operation boundaries to distinguish kernel errors from downstream state bugs.

## 2026-08-08: Retain the four-wave Q6_K MMV candidate without promotion

**Decision:** Preserve the isolated RDNA4 Q6_K `ncols_dst=1` `calc_nwarps` change from eight to four as `promising_not_promotion_ready`. Do not promote it from the Phase 3 microbenchmark result.

**Reason:** The candidate passed the GGML CPU oracle and produced three stable model-derived microbenchmark wins, while the long-K case exceeded the frozen variance threshold. Phase 3 had no workload-weighted, normal-condition, dual-R9700 end-to-end result. Its original harness shapes were full GGUF dimensions rather than the registered tensor-parallel slices.

## 2026-08-08: Join Q6_K launch geometry to runtime tensor attribution

**Decision:** Use a bounded, env-gated, normal-graph host-submission capture to map Q6_K tensor names, logical dimensions, fusion state, and planned launches, then join those shapes to the preserved Phase 2 trace by agent, fusion variant, and launch geometry. Treat graph replay counts as unavailable and retain ambiguous joins explicitly.

**Reason:** This establishes fused `ffn_up` at per-device `M=8704,N=1,K=5120` as an exact 29.842% TG dispatch-duration hotspot and exposes the required Phase 4 shape correction. The unfused `M=5120` family remains ambiguous across `attn_output`, `attn_qkv`, `ffn_down`, and `ssm_out`, so no individual share may be invented. The exact fused operation activates SuperSonic's gate/up methodology as a reference, but not a new Phase 4 port or candidate.

## 2026-08-08: Reject unconditional four-wave Q6_K MMV dispatch

**Decision:** Reject `rdna4-q6k-nwarps4` as a global Q6_K `N=1` optimization and skip E2E registration or benchmarking. Record the exact patch fingerprint as a failed global idea so prior-result history demotes it.

**Reason:** All 36 exact-slice correctness rows passed, and several middle unfused shapes improved, but `M=24` regressed stably to about `0.933x` and `M=512` to `0.526x/0.481x` on the two R9700s. Exact fused `ffn_up`, 29.842% of summed TG dispatch duration, was stable at `0.9983x` on both devices and missed the frozen `1.01x` gate. An unconditional workgroup reduction cannot be promoted by averaging these failures with unrelated wins.

## 2026-08-08: Preserve procedure failures without changing the gate

**Decision:** Supersede the single-snapshot idle procedure with an evidence-preserving wait for a passing snapshot at the same threshold. Retain the original finalizer failure and use a separately identified parser recovery only to translate fused dimension representation and exact comma-delimited shapes.

**Reason:** Zero-VRAM telemetry fluctuated above 5% without a KFD owner, making a single snapshot unable to complete the schedule. The parser confused fused token/output dimensions with logical GEMM dimensions and `M=512` with `M=5120`. Neither repair changed measurements, tolerances, stability limits, speedup thresholds, or the resulting rejection.

## 2026-08-08: Reject shape-gated four-wave Q6_K dispatch

**Decision:** Reject `rdna4-q6k-shape-gated-nwarps4` and prohibit E2E registration. Preserve both the incorrect v1 implementation and corrected v2 result in prior-result history.

**Reason:** Corrected v2 proved all 36 dispatch selections and passed all 36 CPU-oracle rows. The three selected tuples retained favorable medians, but five per-device rows failed the frozen stability/regression contract. Device-0 fallback `M=512` measured `0.9878x`, below the `0.99x` floor. Local wins cannot bypass exact fallback and variance gates.

## 2026-08-08: Use an architecture-independent specialization selector

**Decision:** Never place architecture-dependent `calc_nwarps()` output in a host-visible template default for MMV dispatch. Future experimental wave counts must use an architecture-independent template value or boolean while leaving the default kernel's device-side architecture calculation intact.

**Reason:** Phase 5 v1 could compile the fallback reduction for a different wave count than the host launch dimensions, and the first excluded candidate trace timed out. The v2 boolean specialization restored correct 8-wave fallback dispatch on all cases and both GPUs.

## 2026-08-08: Select fused-only 12-wave Q6_K as the next hypothesis

**Decision:** If Phase 6 proceeds, evaluate exactly one 12-wave specialization only for fused Q6_K `8704x1x5120`. Preserve eight waves for all unfused and nonmatching paths. Do not sweep 16 waves in the same phase.

**Reason:** Fused `ffn_up` is an exact 29.842% TG dispatch-duration hotspot. Q6_K has 20 quant blocks at `K=5120`; eight waves advance eight blocks per K-loop iteration, while 12 advance 12 and reduce the loop from three iterations to two. Sixteen waves do not reduce it below two and add more reduction/LDS pressure, so it remains a separate later candidate. This is a hypothesis requiring compiled-resource and timing evidence, not a predicted win.

## 2026-08-08: Require per-candidate resource and efficiency scorecards

**Decision:** For every compiled GPU candidate, preserve wave/workgroup size, VGPR, SGPR, LDS, private memory, scratch, spills, occupancy, and achieved bandwidth per relevant specialization and physical GPU. Classify each value as `static_code_object`, `derived_bound`, `runtime_counter`, or `unavailable`. Reserve "achieved" for exact-dispatch runtime counters and retain raw capability/counter provenance.

**Reason:** The four-wave experiment reduced LDS without changing VGPR/SGPR and still did not improve fused `ffn_up`. Kernel time and static resource reductions alone cannot distinguish useful parallelism from occupancy, bandwidth, or reduction-pressure tradeoffs. gfx1201 counter and occupancy capabilities must be discovered dynamically; missing metrics cannot be filled with Instinct assumptions or tensor-size estimates.

## 2026-08-08: Reject fused-only twelve-wave Q6_K ffn_up specialization

**Decision:** Reject `rdna4-q6k-fused-nwarps12` and prohibit E2E registration or benchmarking. Preserve the exact task, dispatch, correctness, timing, code-object, counter, and attempt evidence as a failed optimization idea. Do not retry twelve waves or test sixteen waves from this result alone.

**Reason:** The exact fused `ffn_up` `8704x1x5120` selector passed all 36 dispatch witnesses and all 36 CPU-oracle rows, but measured 1.0000x on device 0 and 1.0033x on device 1, below the frozen 1.01x hotspot requirement. Device-0 `attn_kv` also had a 3.6975% baseline spread, above the 3.5% stability limit. The shorter K loop did not produce a qualifying win, and sixteen waves would share the two-loop count while increasing reduction work.

## 2026-08-08: Mark Phase 6 exact runtime counters unavailable

**Decision:** Record achieved occupancy and achieved bandwidth as unavailable for the Phase 6 candidate and baseline. Preserve the relative-path procedure failure and all four 120-second timeouts. Do not infer counters from static code-object data, unprofiled timing, or Instinct limits.

**Reason:** The first counter invocation launched no workload because its relative binary path became invalid under the runner working directory. The corrected frozen nonmultiplexed `rocprofv3 --kernel-trace --pmc OccupancyPercent,FETCH_SIZE --mangled-kernels` captures then timed out for baseline/candidate on both GPUs without a joinable result. Dynamic gfx1201 discovery also did not expose the allocation and slot inputs needed for a combined occupancy bound.

## 2026-08-08: Begin Phase 7 with PP MMQ attribution, not a kernel candidate

**Decision:** Freeze a Phase 7 contract for the registered profile 5/suite 2 PP512 Q6_K MMQ workload: first collect three independent normal-graph baseline processes, then attribute the existing Phase 2 PP trace to exact host operations. Do not port, compile, register, or benchmark a kernel candidate in Phase 7.

**Reason:** `mul_mat_q<(ggml_type)14, 128, false>` and `true` account for 85.011% of summed Phase 2 PP dispatch duration, but `128` is the MMQ X tile and the boolean is `need_check`; neither means `N=128` or fusion. The actual registered PP workload is `N=512`. Existing trace/source evidence is family-level and low-confidence, so an exact dual-device tensor/shape/layout/launch join and a fresh ordinary baseline are required before choosing a material operation.

## 2026-08-08: Close Phase 7 with exact PP MMQ attribution

**Decision:** Record Phase 7 as an attribution and baseline result, not a kernel optimization. Preserve the exact dual-device operation shares and resource facts; do not infer a speedup or create a candidate from them alone.

**Reason:** The normal-graph PP512 baseline passed at 0.607% spread. The positional join mapped 496 host operations per device across six repeats with zero mismatch, ambiguity, or unmatched rows, resolving the Q6_K MMQ 85.0106% family into material exact operations. Dynamic resources show 57,856 B LDS, wave32, 256-thread workgroups, 229/230 VGPR, 27/30 SGPR, and zero static private/spills. These facts justify one later, separately contracted `mmq_y=64` versus `128` hypothesis, while holding `mmq_x=128` and eight waves fixed; fusion remains out of scope.

## 2026-08-08: Reject constant-only Q6_K MMQ y64 before build

**Decision:** Reject the global gfx1201 Q6_K `mmq_y=64`, `mmq_x=128`, eight-wave candidate as statically infeasible. Do not repair it by changing to four waves and do not run a build, workload, counter pass, or E2E comparison.

**Reason:** The pinned WMMA writeback code requires `nwarps * tile_C::I == mmq_y`. gfx1201's approved eight-wave Q6_K path has `tile_C::I=16`, so `8 * 16 = 128`, not 64. A 64-row implementation requires a new partition/writeback design and is not a constant-only tile selector experiment.

## 2026-08-08: Gate Phase 10 on exact-shape paired PP MMQ evidence

**Decision:** Permit only an explicitly approved, default-off gfx1201 Q6_K shared-Q8_1 paired dispatcher for the Phase 7/9 exact PP shape `8704x512x5120`. Quantize the common activation once, reuse it for the two existing MMQ launches, and retain the current F32 SwiGLU operation. Preserve normal graph capture, separate gate/up weights, per-device tensor-parallel ownership, and baseline fallback. Require a new MMQ-specific interface; do not repurpose the N=1 MMV fusion path or add a monolithic dual-accumulator kernel in the same phase.

**Reason:** Phase 9 proves exact adjacency and a shared normalized F32 input for two material Q6_K MMQ projections, whose combined PP share is 38.304%, while showing that existing fusion stops at MMV N=1. A paired dispatcher may remove duplicate activation quantization; a monolithic kernel might also remove intermediate traffic, but the existing 229/230-VGPR and 57,856-byte dynamic-LDS footprint makes dual accumulation high risk. Static resource design, exact dispatch/fallback, parity, variance-aware timing, graph replay, and whole-PP gates are necessary before an E2E claim or registration.

## 2026-08-08: Reject shared-Q8_1 gate/up launch-only optimization

**Decision:** Reject the default-off exact-shape gfx1201 Q6_K shared-Q8_1
gate/up dispatcher for registration. Preserve its complete evidence as a failed
idea and do not retry launch/duplicate-quantization-only variants of this chain
without new evidence.

**Reason:** The candidate correctly changed the exact PP512 pair from two
`quantize_mmq_q8_1` launches to one while retaining two Q6_K MMQs; both-device
fallback, CPU/F32 parity, normal HIP graph replay, and tensor-parallel
ownership passed. Yet the isolated combined gate/up result was `0.999896x`
with a `0.6008%` stable spread, and normal-graph whole PP was only `1.003627x`
with a `0.2589%` spread. The candidate misses the minimum exact-operation
speedup despite a valid implementation. The evidence points away from launch
overhead and toward intermediate traffic or a different exact PP hotspot; that
requires a separately profiled and contracted direction.

## 2026-08-08: Close the Q6_K decode idle-lane hypothesis without a candidate

**Decision:** Mark Phase 11 `complete_no_go`. Do not port lighttransport's
Q6_K launch-width change and do not authorize another four-, twelve-, or
sixteen-wave llama.cpp MMVQ experiment from terminal K-loop arithmetic alone.

**Reason:** Both R9700s dynamically report physical wave32 execution. In the
registered llama.cpp mapping, each active 32-lane wave cooperatively processes
one Q6_K block, so the reference kernel's mostly idle scalar threads do not
exist. K=3072, 5120, and 8704 do have final-iteration whole-wave underfill,
but the hottest K=5120 fused `ffn_up` path was already flat when four waves
made its 20 blocks divide exactly, and twelve waves was also flat. The cited
reference throughput gain additionally combined several IQ and Q6 rewrites on
a different MoE workload; it is not isolated evidence for this dense Q6 path.

## 2026-08-08: Select the PP SwiGLU-to-MMQ-D4 ffn_down boundary

**Decision:** Mark Phase 12 source/trace feasibility complete and permit a new
immutable candidate contract only for a default-off exact-shape
SwiGLU-to-`block_q8_1_mmq(D4)` producer feeding Q6_K `ffn_down` at
`M=5120,N=512,K=8704`. Do not implement the RMSNorm/gate/up alternative in
the same experiment.

**Reason:** A new positional join proves 384 adjacent producer, D4 quantizer,
and `ffn_down` MMQ chains per device across six repeats and all 64 layers, with
zero ambiguity. The chain accounts for 19.812% of summed PP time; its F32
intermediate has one consumer and models 35,651,584 avoidable write/read bytes
per layer/device. PP MMQ uses a private 128-value D4 packing, not ordinary
Q8_1, so the later candidate requires fused SwiGLU/packing plus a prequantized
MMQ entrypoint. The larger-share RMSNorm option has two consumers, lower
modeled staging traffic, harder reductions, and Phase 10's flat shared-Q8
partial result. No performance gain is claimed by this selection.

## 2026-08-08: Reject exact SwiGLU-to-D4 ffn_down producer

**Decision:** Reject the default-off PP512 Q6_K SwiGLU-to-MMQ-D4 `ffn_down`
candidate for registration. Preserve the patch and all failed attempts, and
demote the same one-wave direct-packing design in future candidate ranking.

**Reason:** The final candidate selected 128 times per R9700, removed one
ordinary SwiGLU and Q8_1 quantizer per exact chain, passed 18/18 CPU-oracle
rows, fallback, normal graph replay, provenance, and resource gates. Whole PP
measured `1.005222x`, but the predeclared exact-chain gate regressed to
`0.995165x` on GPU 0 and `0.992729x` on GPU 1. Both rows were stable. The
contract requires at least `1.010x` exact speedup and forbids promotion on any
failed gate. gfx1201 occupancy and `FETCH_SIZE` remain unavailable after
bounded rocprofv3 attempts aborted inside HIP initialization on both devices.

## 2026-08-08: Register Q6_K MMA float conversion

**Decision:** Accept and register the one-line Q6_K RDNA4 MMA change that
converts `C.x[l]` to F32 before multiplication by the integer Q6 scale. Keep
the Q2_K, dispatch-threshold, Stream-K, and geometry changes from draft
llama.cpp PR #25940 out of Phase 13.

**Reason:** The immutable both-device gate passed 24/24 CPU-oracle executions,
normal graph replay, tensor-parallel/P2P dispatch equivalence, stability, and
TG regression requirements. PP512 improved from `814.363` to `1040.482 tok/s`
(`1.277663x`) and profiled Q6_K MMQ duration improved `1.617272x` with
identical dispatch counts. ISA changed from pathological integer-scale
multiplication toward explicit int-to-F32 conversion. VGPR allocation rose
from 232 to 256, while SGPR, eight-wave geometry, 57,856 B dynamic LDS,
private memory, scratch, and zero-spill status remained unchanged. An
independent registered-build smoke measured `1039.410 tok/s`.

## 2026-08-08: Re-baseline before paired RDNA4 Y64/W4

**Decision:** Phase 14A must refresh hotspot attribution and Q6_K resources on
the promoted float-cast build before any geometry candidate. Do not test
`mmq_y=64,nwarps=4` against the pre-cast kernel or combine it with Stream-K.

**Reason:** The float conversion changed Q6_K instruction dependencies,
reduced profiled MMQ time by `1.617272x`, and increased runtime VGPR allocation
from 232 to 256. Both the resource picture and relative PP hotspot shares have
therefore moved. If Q6_K MMQ remains the leading actionable cost after the
refresh, paired Y64/W4 is structurally valid because
`4 * tile_C::I(16) == 64`; unlike Phase 8, it must be evaluated as one matched
geometry on top of the registered Phase 13 fix.

## 2026-08-08: Authorize paired RDNA4 Y64/W4 as Phase 15

**Decision:** Complete Phase 14A without a candidate and authorize Phase 15 to
evaluate only the paired gfx1201 Q6_K `mmq_y=64,nwarps=4` geometry on top of
the registered Phase 13 float-cast baseline.

**Reason:** Three normal-graph PP512 processes average `1042.144 tok/s` with
`0.0492%` spread. A fresh exact six-repeat attribution join places Q6_K MMQ at
`77.7567%` of summed no-graph GPU kernel duration; gate/up/down alone total
`51.6601%`. The two-device Q6_K imbalance is only `1.2980%`, and communication
copy kernels are not the exposed leader. Current resources remain 256 runtime
VGPR, 128 runtime SGPR, 57,856 B dynamic LDS, and eight wave32 waves. The
matched geometry satisfies `4 * 16 = 64` and is materially different from the
invalid Phase 8 Y64/eight-wave proposal. Do not test it against pre-cast code
or combine it with Stream-K or another optimization.

## 2026-08-08: Reject paired RDNA4 Y64/W4 and switch Phase 16 to TG

**Decision:** Reject the global RDNA4 MMQ `mmq_y=64,nwarps=4` candidate for the
registered post-cast Q6_K workload. Do not register it or combine it with
Stream-K as a rescue. Phase 16 moves to N=1 Q6_K TG MMVQ K2/K4 pipeline audit;
Stream-K remains a later independent PP backlog item.

**Reason:** The candidate was structurally correct and passed 96/96 oracle
rows, exact Q6-only dispatch, graph replay, and resource capture. It reduced
dynamic LDS from 57,856 B to 38,400 B and halved the workgroup, but VGPR stayed
252/253 static and 256 runtime while SGPR increased. Stable exact results were
about `0.993x` gate/up, `0.958x` ffn_down, and `0.975x` combined on both GPUs.
A user-requested three-pair whole-PP diagnostic was also flat-negative at
`0.998916x` (`1042.838 -> 1041.707 tok/s`). The mechanism worked, but did not
produce throughput on this Q6_K geometry.

## 2026-08-09: Reject Q6_K N=1 K2 scheduling

**Decision:** Reject both the global and fused-only gfx1201 Q6_K N=1 K2 packet
pipelines. Do not proceed to K4, TG128, PP512, or registration. Preserve the
registered Phase 13 cast library unchanged.

**Reason:** Static analysis found a real outer-K scheduling gap, but LLVM
retained future-block overlap only in the fused specialization. The narrowed
fused candidate passed 84/84 correctness rows and kept the unfused machine code
byte-identical, but raised fused VGPR from 35 to 60. Correct full-graph timing
measured `0.982617x` on GPU 0, `0.958899x` on stable GPU 1, and `0.970618x`
combined. Earlier 3-us rows repeated only a trailing ADD and are retained as
invalid harness evidence, not used in the decision.

## 2026-08-09: Gate native dot8 arithmetic statically

**Decision:** Make Phase 17 a register-only source/algebra/ISA feasibility gate
for exact Q6_K x Q8_1 native dot8 arithmetic. Keep it independent from K2/K4
and prohibit integration or GPU benchmarking unless the static gate passes.

**Reason:** Q6_K's two QR6_K contributions have independent scales and Q8_1
factors, while a native dot8 returns one scalar reduction. The four-term nibble
identity is exact per element, but packing eight values can destroy the scale
boundary; padding to four live lanes may also erase dot8's throughput benefit.
Exhaustive signedness/parity proof and generated gfx1201 instruction/resource
evidence are therefore cheaper and more decisive than another speculative
production candidate.

## 2026-08-09: Reject standalone Q6_K x Q8_1 native dot8

**Decision:** Close Phase 17 as `complete_no_go`. Do not integrate or benchmark
the standalone dot8 arithmetic and do not treat native instruction emission as
evidence of a faster kernel.

**Reason:** The corrected construction passed 131,072 exhaustive scalar cases
and emitted native gfx1201 `v_dot8_i32_iu4`. Q8_1 int8 decomposition and the
two independent QR6 scale boundaries nevertheless require four half-utilized
dot8 reductions per four-value term: eight dot8s replace two productive
signed-int8 DP4As. Matched code grew from 372 to 876 bytes, 60 to 158 counted
instructions, one to 16 ALU waits, 2 to 6 VGPR, and 16 to 22 SGPR. This fails
the frozen lane-efficiency and dependency gates before production integration.

## 2026-08-09: Reject fused K2 plus native dot8 stack

**Decision:** Reject the separately frozen Phase 18 combined candidate before
GPU work. Do not rescue it by relaxing the resource gate, adding K4, or using
TG128 as a less-specific performance signal.

**Reason:** The actual fused Q6_K N=1 llama.cpp candidate compiled and retained
K2's partial next-block load overlap. Its unfused symbol stayed byte-identical,
and global loads stayed 30. But eight DP4As became 32 dot8 instructions, fused
ISA grew from 918 to 1,095 instruction lines, and VGPR rose from 60 to 66,
exceeding the predeclared ceiling of 64. The two mechanisms stack structurally;
their arithmetic and live-state costs also stack. No correctness or benchmark
stage was authorized, and the registered Phase 13 build remains unchanged.

## 2026-08-09: Trace real MTP widths before narrow DP4A

**Decision:** Make Phase 19 a candidate-free trace of real Qwen3.6 MTP
verification. Authorize a narrow gfx1201 Q6_K DP4A experiment only if exact
N=2/3/4 verification shapes are observed on both devices and have a material,
demonstrably wasteful current route.

**Reason:** Ordinary N=1 TG remains dominated by Q6_K MMVQ, but K2 and dot8
have now failed with exact source, ISA, resource, and timing evidence. The
distinct MTP hypothesis concerns wasted fixed-width matrix columns at N=2-4,
not another N=1 scheduling or arithmetic rewrite. A semantic shape/dispatch
trace is the cheapest way to determine whether that boundary exists in the
actual workload before building another candidate.

## 2026-08-09: Reject the N=2-4 MTP boundary and select N=5

**Decision:** Close Phase 19 as `complete_no_go_for_n2_to_n4`. Do not build a
Q6_K N=2-4 candidate for the registered `draft_n_max=4` workload. Make Phase
20 a candidate-free source/ISA feasibility audit of the actual N=5 MMVQ path,
including an explicit MTP verification marker design.

**Reason:** A deterministic target-backed MTP request joined all 43,970 Q6_K
dispatches exactly. N=2 and N=3 were absent; N=4 accounted for only 2.6467%
of summed request GPU-kernel duration and failed the frozen 10% combined and
5% per-device materiality gates. N=5 produced 39,390 calls and owned 56.9103%.
The registered gfx1201 route is already a one-wave DP4A MMVQ kernel, not a
fixed-width WMMA/MMQ or dense GEMM path, so forcing narrow DP4A would reproduce
the current mechanism. The HIP backend also lacks an MTP marker; a shape-only
selector would affect ordinary small batches.

## 2026-08-09: Authorize N=5 two-row Q8_1 reuse

**Decision:** Close Phase 20 as
`complete_go_for_separate_phase20b_r2_only`. Authorize one separately frozen
gfx1201 Q6_K N=5 two-row MMVQ experiment behind explicit target-verification
provenance. Do not include R4, K scheduling, dot8, MMQ, Stream-K, or a
shape-only selector.

**Reason:** The registered N=5 symbol already shares five Q6 fields across all
columns, but two output-row blocks reload the same twenty Q8_1 fields. R2
models 30 global loads per row pair instead of 50, while retaining 20 DP4As,
ten reductions, one wave, and zero LDS/barriers. The registered Q6_K two-row
MoE code object independently emits fourteen relevant loads rather than
eighteen, proving LLVM can retain Q8_1 values across rows. The projected N=5
live set is 55-64 VGPR; Phase 20B must reject before GPU work above 64 VGPR,
32 SGPR, or on any LDS/private/spill allocation. Server row provenance, an
all-rows verification decode mode, graph-cache separation, and a backend-
visible tensor flag are required because neither N=5 shape nor MTP context
type safely identifies the hot target verification graph.

## 2026-08-09: Reject Phase 20B N=5 R2 at static resources

**Decision:** Close the separately frozen Phase 20B gfx1201 Q6_K N=5 R2
candidate as `complete_reject_no_promotion`. Do not relax its resource limits,
benchmark it, or register it.

**Reason:** Both isolated variants built and every generic unfused Q6_K N=1-5
machine encoding plus AMDHSA metadata was identical across the compile gate.
The dedicated R2 symbol retained the intended 30 global loads, 20 `v_dot4`,
and zero barriers/LDS/private memory/spills, but required 80 VGPR and 34 SGPR.
Those exceed the frozen ceilings of 64 VGPR and 32 SGPR. Correctness, exact
timing, MTP trace/server, graph, TG/PP, P2P/all-reduce, and counters were not
authorized after that fail-fast result. The registered Phase 13 library stays
`facd1354c4eba6afec9af0b22694e6ca11bf2b2bd976368158f225d6692c4311`.

## 2026-08-09: Keep DFlash/DDTree as a separate major-work branch

**Decision:** Close Phase 21 as `feasible_with_major_work`. Do not implement a
tree candidate yet and do not combine it with Phase 20B R2 or another kernel
experiment. Make the next phase an isolated linear-DFlash compatibility and
instrumentation study on the unchanged Phase 13 target, followed by offline
tree-policy replay.

**Reason:** Registered llama.cpp already supports a linear DFlash block, and a
local 1.7B mostly-BF16 block-16 draft GGUF matches the target's 5120 hidden
width and five requested target layers. Official DDTree and pinned Lucebox
prove best-first tree construction, ancestor-masked one-forward verification,
path walking, and single-device hybrid-state rollback. Apex, however, has no
parent-aware tree operations or commit for its 48 Gated DeltaNet/conv layers,
and Lucebox explicitly disables production tree verification in its multi-GPU
layer-split target. Tensor-split correctness, actual VRAM fit, Huihui-target
acceptance, dispatch shapes, communication, and end-to-end value all remain
unmeasured. Those facts justify the branch, not an immediate implementation.

## 2026-08-09: Validate DFlash asset and DDTree host policy; hold tensor split

**Decision:** Accept the local Qwen3.6 DFlash GGUF as runtime-compatible in the
two-R9700 layer-split lane and accept the pinned official DDTree host policy as
algorithmically valid. Do not call the registered tensor-split path validated,
do not benchmark or integrate a tree, and do not install the official CUDA
dependency stack.

**Reason:** Control and DFlash produced identical raw token IDs for all 192
requested tokens. The DFlash run loaded all layers, used both GPUs, generated
754 proposals, and accepted 137. Official DDTree at `c96427a...` passed an
independent host-only oracle for exhaustive prefix order, budgets through 32,
parent/depth/visibility invariants, deterministic replay, and accepted-path
walking. However, the draft inherits `-sm tensor` and creates an unsupported
`Meta()` context; only layer split with both target-weight devices visible ran.
Real top-32 replay, recurrent tree state, GPU tree verification, and stable
performance evidence remain future gates.

## 2026-08-09: Accept isolated draft layer split only at n_max 12

**Decision:** Keep the target tensor split and accept the isolated DFlash
draft-layer plumbing only with `--spec-draft-n-max 12`. Do not register the
build, use `n_max=15`, or begin GPU DDTree work.

**Reason:** The draft-specific split option correctly placed DFlash layers on
ROCm0/ROCm1, but shared target token/output tensors required the target Meta
device in the draft scheduler. That exposed a precise one-descriptor external-
view shortfall, fixed by increasing documented headroom 16->17. Parser tests,
ordinary tensor control, memory release, and two independent `n_max=12`
requests passed exact 192-token parity. Widths 1, 4, 8, and 15 did not; the
`n_max=15` divergence repeated exactly at token index 159. This is a bounded
compatibility result, not a promotion or performance result.

## 2026-08-09: Advance DDTree to parent-aware correctness only

**Decision:** Accept the Phase 21D real-distribution replay as a GO for a
separately frozen parent-aware Qwen35 tree-correctness phase. Do not call it a
GPU, throughput, or promotion result. Use chain-seeded budget 20 as the
no-linear-regression reference and pure best-first 20 as the higher-depth
comparator.

**Reason:** Two passive recorder processes preserved exact 192-token output and
emitted identical top-32 payloads. Across 50 full cycles, pure best-first 20
raised mean canonical depth from 2.60 to 3.20 and rescued 24 cycles, while
chain-seeded 20 reached 3.14, rescued 22, and never lost depth versus linear.
The fixed `3x4` and `4x3` policies lost overall. Replay uses the serial token
stream and therefore cannot establish alternate-branch attention, KV,
DeltaNet/recurrent state, rollback, or tensor-split correctness. Those remain
hard gates before GPU tree timing.

## 2026-08-09: Stop direct DDTree integration at the recurrent substrate

**Decision:** Close Phase 21E as `complete_no_go` before source edits, builds,
or GPU work. Do not create a mask-only or KV-only tree. Split the next work
into parent-aware Qwen35 recurrent primitives first and DDTree integration
only after primitive-level serial equivalence.

**Reason:** Pinned llama.cpp has bounded linear rollback snapshots but no
parent-index graph input, parent-aware SSM convolution, or parent-aware Gated
DeltaNet operation. A DFS sibling would therefore inherit the preceding DFS
node's recurrent and convolution state rather than its actual parent's state.
Lucebox proves the required monolithic mechanism but explicitly disables its
production layer-split tree path and rejects sibling topology. The 20-node
Phase 21D policy remains promising; the execution substrate is what failed.
