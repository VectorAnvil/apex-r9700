# Apex Phases 04–15 Evidence Ledger

Date: 2026-08-18
Scope: Phases Four through Fifteen, inclusive
Method: historical evidence recovery only; no code changes, builds, GPU reruns, or candidate combinations were performed.

## Reading rules

- Speedup is candidate divided by baseline; values above `1.0x` are faster.
- `NOT MEASURED` is intentionally distinct from zero, failure, or predicted regression.
- “Qwen inference” means a full llama.cpp model workload. Model-derived `test-backend-ops` shapes are not counted as full-model inference.
- “Production-like” means the registered dual-R9700 PP/TG workload, not merely execution on both GPUs.
- The evidence class is the present forensic classification, not the historical decision.
- Exact token parity, semantic/model quality, tool calling, structured output, and agent behavior were not tested for any candidate in this phase range.

## Candidate index

| ID | Phase | Candidate | Historical disposition | Evidence class | Decisive recovered fact |
|---|---:|---|---|---|---|
| C01 | 4 | Global Q6_K MMV four-wave | Rejected | D | Several exact shapes gained 6–11%; no TG/E2E trial. |
| C02 | 5 | Shape-gated four-wave v1 | Superseded as defective | B | Host/device specialization mismatch launched an incompatible reduction and timed out. |
| C03 | 5 | Shape-gated four-wave v2 | Rejected | D | Selected shapes gained 5.6–11.0%; no TG/E2E trial. |
| C04 | 6 | Fused-only 12-wave `ffn_up` | Rejected | F | Exact hotspot was `1.0000x/1.0033x`; E2E not reached. |
| C05 | 6 | Fused-only 16-wave follow-up | Held/not executed | E | Rejected by loop/reduction prediction; performance UNKNOWN. |
| C06 | 8 | Global Q6_K MMQ Y64/W8 | Rejected before build | E | Static writeback invariant failed; performance UNKNOWN. |
| C07 | 9–10 | Shared-Q8_1 gate/up dispatcher | Rejected | C | Whole PP improved `1.003627x`; isolated gate vetoed it. |
| C08 | 9 | Monolithic gate+up+SwiGLU MMQ | Not authorized | E | Resource risk predicted; no build or performance result. |
| C09 | 11 | Lighttransport-style decode lane/launch change | Closed no-go | E | Source mapping contradicted the idle-lane premise; performance UNKNOWN. |
| C10 | 12 | RMSNorm→shared D4 gate/up producer | Not selected | E | Feasible boundary with 38.304% connected share; no implementation or performance result. |
| C11 | 12 | SwiGLU→D4 `ffn_down` producer | Rejected | C | Whole PP improved `1.005222x`; isolated chain vetoed it. |
| C12 | 13 | Tiled MMQ/all-reduce overlap | Administratively displaced | E | Proposal was not implemented or measured; performance UNKNOWN. |
| C13 | 13 | Explicit F32 Q6_K MMA scale conversion | Accepted/banked | G | PP `1.277663x`, Q6 trace `1.617272x`, TG `0.999895x`; later rebaseline confirmed. |
| C14 | 14A–15 | Paired Q6_K MMQ Y64/W4 | Rejected | F | Isolated chain `~0.975x`; whole PP `0.998916x`, smaller than run spread. |
| C15 | 13–15 | Q6_K Stream-K PP path | Backlog/not executed | E | Proposed and explicitly kept separate; performance UNKNOWN. |

## Detailed mechanical ledger

### C01 — Phase 4 global Q6_K MMV four-wave

- Source/identity: `rdna4-q6k-nwarps4`; candidate patch SHA-256 `8003...ad62`; task fingerprint `12f47...8b`; exact full hashes are in the Phase 4 manifest.
- Change/target: globally changed Q6_K `N=1` MMV from eight to four waves for registered per-device model shapes. Baseline was the then-registered eight-wave path; candidate used 128 threads.
- Hypothesis: lower LDS and reduction work would improve underfilled Q6_K decode shapes.
- Gates/thresholds: 36 CPU-oracle rows; stable shapes had to retain at least `0.99x`; fused `ffn_up` had to reach `1.01x`; variance limits were frozen per task.
- Actual isolated/microbenchmark result: D0/D1 speedups: `M24 .9334/.9333`; `M512 .5261/.4812`; `3072x5120 1.0654/1.0692`; `5120x3072 1.0985/1.1128`; `5120x5120 1.0641/1.0828`; `5120x8704 1.0499/1.0479`; `6144x5120 1.0690/1.0810`; fused `8704x5120 .9983/.9983`; output `151936x5120 1.0087/1.0074`. Some baseline rows were unstable as recorded in the gate result.
- PP/TG/production: PP `NOT MEASURED`; TG `NOT MEASURED`; production/E2E `NOT MEASURED`. Real Qwen inference: no. Dual-R9700 production-like workload: no.
- Correctness: 36/36 CPU-oracle comparisons passed, NMSE maximum `0.0005`; this was tolerance-based numerical kernel correctness. No token-parity or semantic/model test.
- Resources: baseline/candidate 256/128 threads; LDS unfused 896/384 B and fused 1792/768 B; VGPR unchanged at 26 unfused and 35 fused; no spills. Runtime occupancy/bandwidth `NOT MEASURED`.
- Static prediction: four waves were expected to reduce underfill/reduction overhead; actual results were shape-dependent.
- Historical result/reason: FAIL/rejected because global dispatch materially regressed `M24` and `M512` and fused `ffn_up` missed `1.01x`. Execution stopped before E2E.
- Behavior coverage: stability was tested only as bounded timing spread; long-running inference, model behavior, tools, structured output, and agents were `NOT MEASURED`.
- Later-stack combination: `NOT MEASURED`; predates Phase 13 and was not tested with the later registered stack.
- Current evidence status/class: **D**. Positive local results and real local regressions coexist; whole TG impact is UNKNOWN.
- Evidence: `results_phase04_q6k_nwarps4_20260808/attempts/confirm-v1-01/gate-result-recovered.json`, `results_phase04_q6k_nwarps4_20260808/evidence/`, `docs/apex-r9700/PHASE_04_HANDOFF.md`, `docs/apex-r9700/DECISIONS.md`.

### C02 — Phase 5 shape-gated four-wave v1

- Source/identity: first implementation of `rdna4-q6k-shape-gated-nwarps4`; exact patch/worktree retained under the Phase 5 result root.
- Change/target: four waves for selected middle Q6_K `N=1` shapes with eight-wave fallback elsewhere.
- Hypothesis/gates: retain C01’s wins without its bad shapes; same correctness, dispatch, stability, regression, and `1.01x` selected-shape gates.
- Actual result: the host-visible template default and device architecture specialization disagreed. A four-wave-compiled fallback reduction could be launched with eight waves; the first excluded candidate trace timed out. No accepted timing, PP, TG, or E2E result exists.
- Correctness type/result: hard computational/dispatch implementation failure. This invalidates v1 measurements, not the shape-gating concept.
- Resources/static prediction: `NOT RELIABLY MEASURED` for the invalid launch. The defect was established from source/dispatch behavior.
- Historical result/reason: v1 superseded by corrected v2. Execution stopped and implementation was repaired.
- Qwen/production/behavior/stability/long run/later-stack: all `NOT MEASURED`.
- Current evidence status/class: **B** for this implementation revision.
- Evidence: `results_phase05_q6k_shape_gated_20260808/`, `docs/apex-r9700/PHASE_05_HANDOFF.md`, `docs/apex-r9700/DECISIONS.md`.

### C03 — Phase 5 shape-gated four-wave v2

- Source/identity: corrected boolean specialization; patch SHA-256 `b052...a16`; task fingerprint `442950...bac`.
- Change/target: four waves only for `3072x1x5120`, `5120x1x3072`, and `6144x1x5120`; unchanged fallback otherwise.
- Hypothesis/gates: preserve selected C01 gains; selected stable rows `>=1.01x`, fallback stable rows `>=0.99x`, correctness/dispatch 36/36, strict variance with no outlier removal.
- Actual isolated/microbenchmark result, D0/D1: selected `3072 1.0558/1.0643`; selected `5120x3072 1.0929/1.1096`; selected `6144 1.0686/1.0807`. Fallback `M24 .9950/.9933`; `M512 .9878/.9969`; `5120x5120 .9974/.9960`; `ffn_down .9984/.9995`; fused `ffn_up .9983/1.0000`; output `1.0001/1.0006`. Five device rows violated stability/regression contracts; raw evidence retains isolated outliers rather than deleting them.
- PP/TG/production: all `NOT MEASURED`; no real Qwen inference or production-like full workload.
- Correctness: 36/36 CPU-oracle and 36/36 dispatch witnesses passed; tolerance-based numerical kernel correctness, not token/semantic correctness.
- Resources: selected/fallback/fused 128/256/256 threads; LDS 384/896/1792 B; VGPR 26/26/35; no spills. Dynamic occupancy/bandwidth `NOT MEASURED`.
- Historical result/reason: FAIL/rejected because D0 fallback `M512` was `0.9878x` and several rows missed stability requirements. E2E was prohibited.
- Behavior/stability: bounded microbenchmark stability tested; tools, structured output, agents, semantic quality, long-running inference `NOT MEASURED`.
- Later-stack combination: `NOT MEASURED`; predates Phase 13.
- Current evidence status/class: **D**. Selected-shape gains were measured; relevant TG effect remains UNKNOWN.
- Evidence: `results_phase05_q6k_shape_gated_20260808/`, `docs/apex-r9700/PHASE_05_HANDOFF.md`, `docs/apex-r9700/DECISIONS.md`.

### C04 — Phase 6 fused-only 12-wave `ffn_up`

- Source/identity: `rdna4-q6k-fused-nwarps12`; task fingerprint `454240...bff`.
- Change/target: 12 waves only for fused Q6_K `8704x1x5120`; baseline eight waves, all other paths fallback.
- Hypothesis: reduce K-loop iterations from three to two for the 20-block fused hotspot.
- Gates: 36/36 correctness and dispatch; exact hotspot `>=1.01x`; fallback `>=0.99x`; spread `<=3.5%`.
- Actual isolated/microbenchmark: fused `ffn_up` `1.0000x` D0 and `1.0033x` D1, stable. D0 fallback `attn_kv` baseline spread was `3.6975%`.
- PP/TG/production/Qwen: all `NOT MEASURED`; stopped before E2E.
- Correctness: 36/36 tolerance-based CPU-oracle rows and 36/36 dispatch witnesses passed.
- Resources: 8/12 waves, 256/384 threads, 1792/2816 B LDS; VGPR/SGPR unchanged 35/42; no spills. Achieved occupancy/bandwidth unavailable after four 120-second profiler timeouts; not inferred.
- Historical result/reason: FAIL/rejected because exact speedup missed `1.01x` and one fallback baseline spread missed the limit.
- Model behavior/tools/structured output/agents/long-run/later-stack: `NOT MEASURED`.
- Current evidence status/class: **F**. Exact operation was effectively flat; whole TG effect is UNKNOWN.
- Evidence: `results_phase06_q6k_fused_nwarps12_20260808/`, `docs/apex-r9700/PHASE_06_HANDOFF.md`, `docs/apex-r9700/DECISIONS.md`.

### C05 — Phase 6 fused-only 16-wave follow-up

- Source/commit: proposal only; no patch, build, or worktree.
- Change/target: 16 waves for fused Q6_K `8704x1x5120`.
- Hypothesis/static gate: like 12 waves, 16 waves yields two K-loop iterations, but predicts more reduction/LDS work; the phase contract prohibited sweeping it.
- Actual isolated/micro/PP/TG/production/correctness/resources/stability/model behavior: all `NOT MEASURED`.
- Historical result/reason: not executed; demoted by a static scheduling argument after C04.
- Current evidence status/class: **E**; performance UNKNOWN.
- Evidence: `docs/apex-r9700/PHASE_06_HANDOFF.md`, `docs/apex-r9700/DECISIONS.md`.

### C06 — Phase 8 global Q6_K MMQ Y64/W8

- Source/identity: constant-only proposal; task fingerprint `0857...b1d7`; no candidate build.
- Change/target: PP Q6_K MMQ `mmq_y=64`, `mmq_x=128`, eight waves; baseline Y128/X128/W8.
- Hypothesis: smaller Y tile might lower the 57,856 B LDS footprint and improve residency.
- Static gate: required invariant `nwarps * tile_C::I == mmq_y`; `8 * 16 = 128`, not 64.
- Actual build/correctness/isolated/micro/PP/TG/production/resources/behavior/stability: all `NOT MEASURED`.
- Historical result/reason: rejected before build as structurally invalid for the existing writeback partition. Four waves was explicitly a separate design, later tested as C14.
- Current evidence status/class: **E**; performance UNKNOWN. The static gate established implementation infeasibility, not a measured slowdown.
- Evidence: `results_phase08_pp_mmq_y64_20260808/`, `docs/apex-r9700/PHASE_08_HANDOFF.md`, `docs/apex-r9700/DECISIONS.md`.

### C07 — Phases 9–10 shared-Q8_1 gate/up dispatcher

- Source/identity: default-off exact-shape candidate patch at `results_phase10_shared_q8_gate_up_20260808/evidence/candidate.patch`; source commit `259f2e...`; task fingerprint `e569...907`.
- Change/target: at `8704x512x5120`, quantize shared F32 activation once, then call the two existing Q6_K gate/up MMQs and existing F32 SwiGLU. Baseline quantized twice.
- Hypothesis: remove one activation quantizer from a pair representing 38.304% of PP trace time.
- Gates/thresholds: correctness, dispatch, graph, TP, and fallback; exact combined pair `>=1.01x`, spread `<=3.5%`; whole PP `>=1.005x`, spread `<=5%`; reject on any failed mandatory gate.
- Actual isolated/micro: exact combined pair `0.999896x`, spread `0.6008%`; operation was effectively flat.
- Actual PP: baseline three-process mean approximately `814.891 tok/s`, candidate approximately `817.798 tok/s`; **`1.003627x` (+0.3627%)**, spread `0.2589%`. TG and production server `NOT MEASURED`.
- Correctness: CPU/F32 tolerance checks passed; dispatch/fallback, normal graph capture/replay, and TP ownership passed. No exact token parity or semantic/model-quality test.
- Resources: unchanged MMQ kernel, 256 threads/eight wave32 waves, 57,856 B dynamic LDS, 229 VGPR, 27 SGPR, no private segment/spills. Occupancy and achieved bandwidth unavailable.
- Historical result/reason: FAIL/rejected because isolated `1.01x` and whole-PP `1.005x` promotion thresholds were missed, despite positive PP.
- Real Qwen/production-like: yes for registered dual-R9700 PP512 benchmark; no production service trial. Stability: three independent PP processes and bounded exact timings; long-running inference, tools, structured output, agents `NOT MEASURED`.
- Later-stack combination: `NOT MEASURED`; predates Phase 13’s large Q6_K change.
- Current evidence status/class: **C**. Positive whole-workload result rejected by an intermediate gate.
- Evidence: `results_phase09_gate_up_fusion_feasibility_20260808/`, `results_phase10_shared_q8_gate_up_20260808/phase10-gate-result.json`, `results_phase10_shared_q8_gate_up_20260808/evidence/resources.json`, `results_phase10_shared_q8_gate_up_20260808/task-v1.json`, `docs/apex-r9700/PHASE_10_HANDOFF.md`.

### C08 — Phase 9 monolithic gate+up+SwiGLU MMQ

- Source/commit: design proposal only; no patch/build/worktree.
- Change/target: one dual-accumulator Q6_K PP MMQ producing gate/up and applying SwiGLU, removing intermediate traffic.
- Hypothesis: go beyond C07’s duplicate-quantizer removal on the exact 38.304% PP pair.
- Static gate: existing kernel was 229/230 VGPR with 57,856 B dynamic LDS; dual accumulation was judged high resource risk and excluded from Phase 10.
- Actual correctness/performance/PP/TG/production/resources/model behavior/stability: `NOT MEASURED`.
- Historical disposition: not authorized in the same phase; performance UNKNOWN.
- Current evidence status/class: **E**.
- Evidence: `results_phase09_gate_up_fusion_feasibility_20260808/`, `docs/apex-r9700/PHASE_09_HANDOFF.md`, `docs/apex-r9700/DECISIONS.md`.

### C09 — Phase 11 lighttransport-style decode lane/launch change

- Source/commit: source/ISA hypothesis audit only; no patch/build/worktree.
- Change/target: port a narrower Q6 decode launch inspired by lighttransport/Zinc to N=1 MMVQ.
- Hypothesis: eliminate allegedly idle scalar lanes/waves.
- Static result/gate: gfx1201 is wave32 and llama.cpp maps one Q6_K block cooperatively to an active wave; the hypothesized within-wave idle lanes were absent. Only terminal whole-wave K underfill remained. Prior C01/C04 timing had already made four and twelve waves flat on the fused hotspot.
- Actual candidate correctness/performance/PP/TG/production/resources/behavior: `NOT MEASURED`; no candidate existed.
- Historical disposition: closed `complete_no_go` based on source mapping and prior, non-equivalent launch tests.
- Current evidence status/class: **E**; performance of an actual port is UNKNOWN.
- Evidence: `results_phase11_q6k_decode_lane_audit_20260808/`, `docs/apex-r9700/PHASE_11_HANDOFF.md`, `docs/apex-r9700/DECISIONS.md`.

### C10 — Phase 12 RMSNorm→shared D4 gate/up producer

- Source/identity: ranked feasibility candidate `rmsnorm_to_shared_q81_gate_up`; no implementation.
- Change/target: fuse RMSNorm production with the backend-private 128-value D4/Q8_1 packing used by two Q6_K MMQs at `8704x512x5120`.
- Hypothesis: eliminate producer materialization plus shared quantization across a 38.304% connected PP pair.
- Gates/static evidence: feasibility hard gates passed; exact adjacency on both devices, six repeats, 64 layers; modeled avoidable F32 traffic `31,457,280 B/layer/device`; two consumers. It ranked behind C11 because it combines a 5120-value row reduction, learned scaling, packing reductions, and retained shared storage. Modeled bytes are not measured performance.
- Actual build/correctness/isolated/PP/TG/production/resources/model behavior/stability: all `NOT MEASURED`.
- Historical disposition: not selected; C07’s partial shared-quantization result was cited, but C10 removes a producer boundary C07 did not remove.
- Current evidence status/class: **E**; performance UNKNOWN.
- Evidence: `results_phase12_producer_q81_feasibility_20260808/boundary-rank.json`, `source-graph.json`, `trace-joins.json`, `WHAT_DIDNT_WORK.md`, `docs/apex-r9700/PHASE_12_HANDOFF.md`.

### C11 — Phase 12 SwiGLU→D4 `ffn_down` producer

- Source/identity: default-off candidate patch under `results_phase12_swiglu_d4_ffn_down_20260808/evidence/`; task fingerprint `6dc70...c04e7`.
- Change/target: fused SwiGLU output directly into private `block_q8_1_mmq` D4 packing and used a prequantized Q6_K `ffn_down` entrypoint at `5120x512x8704`. Baseline materialized F32 then quantized.
- Hypothesis: remove 128 SwiGLU and 128 quantizer dispatches per device and modeled `35,651,584 B/layer/device` materialization traffic.
- Gates/thresholds: correctness/dispatch/graph/TP/resources; exact chain `>=1.010x`, spread `<=2.5%`; fallback `>=0.99x`; whole PP `>=1.005x`, spread `<=2%`; any failed gate vetoed promotion.
- Actual isolated/micro: D0 `1315.917→1322.310 us`, `0.995165x`; D1 `1337.133→1346.927 us`, `0.992729x`; both stable. Fallback approximately `1.0011x` on both.
- Actual PP: `815.208808→819.466109 tok/s`, **`1.005222x` (+0.5222%)**; whole-PP gate passed. TG and production server `NOT MEASURED`.
- Correctness: 18/18 CPU-oracle tolerance rows, 6/6 dispatch, graph replay, TP ownership and fallback passed. No token-parity or semantic/model test.
- Resources: no candidate LDS/private/scratch/spills reported; exact resource gate passed. Achieved occupancy and FETCH_SIZE unavailable after profiler attempts aborted in HIP initialization.
- Historical result/reason: FAIL/rejected solely because the exact-chain speedup gate failed, despite positive qualifying whole PP.
- Real Qwen/production-like: yes for registered dual-R9700 PP512; no production-service behavior. Stability tested over bounded benchmark repeats; long run/tools/structured output/agents `NOT MEASURED`.
- Later-stack combination: `NOT MEASURED`; predates Phase 13.
- Current evidence status/class: **C**.
- Evidence: `results_phase12_swiglu_d4_ffn_down_20260808/final-result.json`, `task-v1.json`, `evidence/`, `docs/apex-r9700/PHASE_12_HANDOFF.md`, `docs/apex-r9700/DECISIONS.md`.

### C12 — Phase 13 tiled MMQ/all-reduce overlap

- Source/commit: phase proposal/backlog; no candidate patch or build recovered.
- Change/target: overlap tiled Q6_K PP MMQ work with tensor-parallel all-reduce/communication.
- Hypothesis: hide communication behind compute.
- Static/trace context: Phase 13’s scheduled communication-overlap feasibility work was displaced when the explicit-F32 defect was discovered. It was not technically disproved.
- Actual correctness/isolated/PP/TG/production/resources/model behavior/stability: all `NOT MEASURED`.
- Historical disposition: administratively superseded as the Phase 13 activity, but not Class H because C13 did not solve the same communication-overlap problem.
- Current evidence status/class: **E**; performance UNKNOWN.
- Evidence: `docs/apex-r9700/PHASE_13_HANDOFF.md`, `docs/apex-r9700/DECISIONS.md`.

### C13 — Phase 13 explicit F32 Q6_K MMA scale conversion

- Source/identity: one-line explicit F32 conversion before multiplying `C.x[l]` by integer Q6 scale; patch SHA-256 `e1c70...`; source base `259f2e...`; upstream PR/commit #25940 lineage.
- Change/target: all RDNA4 Q6_K PP MMQ; baseline allowed pathological integer-scale arithmetic, candidate forced F32.
- Hypothesis: correct the accumulator-scale instruction path without geometry, Stream-K, Q2_K, or threshold changes.
- Gates/thresholds: 24/24 numerical correctness; graph and TP/P2P dispatch equivalence; PP `>=1.05x`; TG floor `>=0.99x`; stability/resources/no spills.
- Actual isolated/micro: exact Q6 trace combined `1.617272x` with identical dispatch counts.
- Actual PP: `814.362714→1040.481514 tok/s`, `1.277663x`. Actual TG128: `36.730846→36.726971 tok/s`, `0.999895x`. Independent registered smoke: `1039.410 tok/s`. Production service behavior beyond these registered workloads `NOT MEASURED` in this phase.
- Correctness: 24/24 CPU-oracle tolerance rows passed, NMSE limit `0.0005`; graph replay and tensor-parallel/P2P dispatch equivalence passed. Numerical kernel correctness only; no semantic/model, token-parity, tool, structured-output, or agent tests.
- Resources: 57,856 B LDS, eight waves; static VGPR 229/230→252/253 and runtime 232→256; SGPR materially unchanged; no private segment, scratch, or spills.
- Historical result/reason: PASS/accepted and registered because all frozen gates passed.
- Stability: bounded PP/TG and independent registered smoke passed; long-running inference `NOT MEASURED`.
- Later confirmation: Phase 14A rebaseline averaged `1042.144 tok/s`; C13 became the baseline for C14.
- Current evidence status/class: **G**.
- Evidence: `results_phase13_q6k_mma_float_cast_20260808/final-result.json`, `evidence/`, `docs/apex-r9700/PHASE_13_HANDOFF.md`, `results_phase14a_post_cast_rebaseline_20260808/`, `docs/apex-r9700/DECISIONS.md`.

### C14 — Phases 14A–15 paired Q6_K MMQ Y64/W4

- Source/identity: `results_phase15_q6k_y64_w4_20260808/evidence/candidate-y64-w4.patch`; task fingerprint `f1203...`; baseline includes C13.
- Change/target: global gfx1201 Q6_K PP MMQ Y128/W8→Y64/W4 with X128 fixed; structurally valid `4*16=64` writeback.
- Hypothesis: reduce dynamic LDS and workgroup size on the post-cast hotspot.
- Gates/thresholds: 96/96 oracle; exact dispatch/graph/resources; exact gate/up/down combined `>=1.02x`; whole PP `>=1.005x`; TG would follow only after exact pass.
- Actual isolated/micro D0/D1: gate/up `.992509/.993241`; down `.958872/.956940`; combined `.975425/.974826`. Trace `0.995263x`.
- Actual PP diagnostic: `1042.837743→1041.707461 tok/s`, `0.998916x`. Candidate spread was approximately `0.5395%`, exceeding the magnitude of the `0.1084%` mean delta; this does not establish a meaningful whole-model regression. TG and production server `NOT MEASURED`.
- Correctness: 96/96 CPU-oracle tolerance rows, exact Q6 dispatch and graph replay passed. No token/semantic/model tests.
- Resources: 57,856→38,400 B dynamic LDS; 256→128 threads; eight→four waves; static VGPR remained 252/253 and runtime 256; small SGPR rise; no spills.
- Historical result/reason: FAIL/rejected because the isolated chain was slower and whole PP did not reach `1.005x`; TG was stopped by the exact gate.
- Real Qwen/production-like: dual-R9700 PP512 yes; TG, server behavior, tools, structured output, agents, long run `NOT MEASURED`.
- Current evidence status/class: **F** for whole workload; local Q6 operations were measured regressions, while PP was effectively flat/inconclusive.
- Evidence: `results_phase15_q6k_y64_w4_20260808/final-result.json`, `resources/resource-summary-static.json`, `task-frozen.json`, `WHAT_DIDNT_WORK.md`, `docs/apex-r9700/PHASE_15_HANDOFF.md`.

### C15 — Phases 13–15 Q6_K Stream-K PP path

- Source/commit: proposal/backlog only; draft PR #25940 contained related ideas but Phase 13 explicitly excluded Stream-K; no isolated Apex candidate patch was tested.
- Change/target: Stream-K scheduling for PP Q6_K MMQ.
- Hypothesis: improve work distribution/utilization for material PP shapes.
- Gates/static prediction: explicitly kept separate from C13 and C14 to preserve attribution; after C14 it remained a later independent PP backlog item.
- Actual build/correctness/resources/isolated/PP/TG/production/model behavior/stability: all `NOT MEASURED`.
- Historical disposition: held, not technically rejected.
- Current evidence status/class: **E** because performance is UNKNOWN; its “held” status is preserved.
- Evidence: `docs/apex-r9700/PHASE_13_HANDOFF.md`, `docs/apex-r9700/PHASE_14A_HANDOFF.md`, `docs/apex-r9700/PHASE_15_HANDOFF.md`, `docs/apex-r9700/DECISIONS.md`.

## Evidence-only experiments (not counted as candidates)

| ID | Phase | Experiment | Recovered result | Classification |
|---|---:|---|---|---|
| X01 | 7 | PP512 baseline and MMQ attribution | Three-process baseline `812.746964, 815.693745, 817.701996 tok/s`, spread `0.6075%`; Q6 MMQ was `85.0106%` of traced dispatch duration. | Control/attribution; not A–H. |
| X02 | 9 | Gate/up fusion feasibility | 64 adjacent pairs/device, exact `8704x512x5120`, 38.304% combined PP share; modeled 71,303,168 B/layer/device F32 traffic. No performance claim. | Feasibility study; feeds C07/C08. |
| X03 | 14A | Post-C13 rebaseline | PP512 mean `1042.144 tok/s`, spread `0.0492%`; Q6 MMQ `77.7567%`, gate/up/down `51.6601%`; 1.298% device imbalance. | Control/positive confirmation of C13; feeds C14. |

## Cross-candidate correctness coverage

No candidate in this range was shown to degrade semantic/model correctness, because none was tested for semantic/model quality. C02 had a hard implementation failure. All other compiled candidates passed their stated numerical tolerance checks. No exact-token-parity evidence was used in these decisions.

| Coverage question | Result across C01–C15 |
|---|---|
| Real model behavior/answer quality | `NOT MEASURED` for every candidate |
| Tool calling | `NOT MEASURED` for every candidate |
| Structured output | `NOT MEASURED` for every candidate |
| Agent behavior/task completion | `NOT MEASURED` for every candidate |
| Long-running inference | `NOT MEASURED` for every candidate |
| Production service soak | `NOT MEASURED` for every candidate |
| Exact token parity | `NOT MEASURED` for every candidate |
| Numerical kernel correctness | Measured for C01–C04, C07, C11, C13, C14; passed except C02’s invalid dispatch revision |

## Evidence limitations

- Several manifests expose full SHA-256 values while phase handoffs abbreviate them. This ledger preserves the abbreviated human-readable identifier and points to the manifest for independent recovery.
- The profiler could not produce reliable achieved occupancy/bandwidth for the affected gfx1201 kernels. Static resource values are retained as static values only.
- Whole PP measurements were benchmark-process results, not long-lived serving traces. “Positive whole PP” does not establish production significance; it does establish that the result was not a measured regression.
