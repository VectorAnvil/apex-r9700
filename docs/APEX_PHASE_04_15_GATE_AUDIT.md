# Apex Phases 04–15 Gate Audit

## Conclusion

The hard numerical-correctness, dispatch-integrity, graph-replay, fallback, and structural writeback gates were useful safety gates. The mandatory isolated-operation speedup gates were not reliable vetoes for whole-model value: they produced two confirmed false negatives, C07 and C11. Minimum whole-PP promotion thresholds also converted small positive measurements into historical “FAIL” labels without establishing regression. Static gates were useful for defining valid implementations, but some were allowed to terminate performance questions that therefore remain UNKNOWN.

## Gate inventory

| Gate | Phases/candidates | What it measured | Original reason | Threshold | Empirical basis recovered | Predictive record | Audit finding |
|---|---|---|---|---|---|---|---|
| CPU-oracle/NMSE | C01, C03, C04, C07, C11, C13, C14 | Kernel output versus trusted CPU/F32 result | Prevent invalid math | Usually all rows pass; NMSE up to `0.0005` where stated | Appropriate computational tolerance; exact derivation not always restated | C13 passed and became a win; all valid compiled candidates passed. C02’s invalid launch was found separately. | Useful safety gate. It did not create a recovered false negative. It is numerical correctness, not semantic/model quality. |
| Dispatch witness/exact selector | C02–C04, C07, C11, C13, C14 | Intended kernel/shape selected; fallback preserved | Prevent measuring wrong code and host/device mismatch | 100% required | Strongly justified | C02 exposed a real hard defect; corrected implementations passed. | Essential. C02 is a true positive. |
| Graph capture/replay | C07, C11, C13, C14 | Candidate works under normal HIP graph execution | Production execution compatibility | Pass required | Operationally justified | All later compiled candidates passed; no false negative found. | Essential compatibility gate. |
| Tensor-parallel/P2P ownership/equivalence | C07, C11, C13 | Both devices retain correct tensor ownership/dispatch | Dual-R9700 correctness | Pass required | Operationally justified | C13 passed and was banked; C07/C11 passed but were vetoed elsewhere. | Useful diagnostic; no false negative attributable to it. |
| No crash/hang/invalid launch | C02 | Basic computational execution | Safety | Pass required | Fundamental | C02 timed out due incompatible launch/reduction specialization. | Essential hard gate. |
| Fallback regression floor | C01, C03, C04, C11 | Nonselected/excluded exact microbench shapes | Ensure specialization does not hurt unrelated paths | Usually `>=0.99x` | Conservative; no whole-workload calibration recovered | C03 was stopped partly by D0 fallback `M512=.9878x`, so TG was never measured. No evidence shows whether that local delta affected TG. | Valid diagnostic, unjustified as an automatic E2E veto when fallback path is workload-light or noise-sensitive. Produced an unresolved stop, not a proven false negative. |
| Selected/hotspot isolated speedup | C01, C03, C04 | Exact N=1 kernel shapes | Require material local gain before E2E cost | `>=1.01x` | Campaign convention; no production-value derivation recovered | C03 had 5.6–11.0% selected wins but stability/fallback vetoed E2E; C04 was flat. | Reasonable triage signal, not a substitute for workload measurement. |
| Exact gate/up pair speedup | C07 | Combined isolated gate/up chain | Require local proof that shared quantization pays | `>=1.01x` | Conservative/arbitrary relative to actual PP | Failed at `.999896x`, while whole PP improved `1.003627x`. | **Confirmed false-negative veto.** It did not predict whole-PP direction. |
| Exact SwiGLU→D4/down chain speedup | C11 | Isolated fused chain | Require local proof before registration | `>=1.01x` | Conservative/arbitrary relative to actual PP | Failed at `.995165x/.992729x`, while whole PP passed at `1.005222x`. | **Confirmed false-negative veto.** The isolated harness and whole graph disagreed in direction. |
| Exact Y64/W4 chain speedup | C14 | Gate/up/down Q6 MMQ microbench | Require geometry win before TG | `>=1.02x` | Conservative; no direct production-value derivation recovered | Correctly indicated local losses (`~.975x`). Whole PP was `.998916x`, effectively flat within spread. | Useful triage here, but it still prevented TG measurement; no proof about TG-specialized applicability. |
| Whole PP minimum promotion | C07, C11, C14 | Registered dual-R9700 PP512 throughput | Require margin over variance/maintenance cost | C07/C11/C14 `>=1.005x` | A conservative campaign floor; not tied to a documented production latency/cost requirement | C07 positive `1.003627x` was labeled fail; C11 `1.005222x` passed; C14 `.998916x` missed. | Threshold distinguished promotion preference, not win/loss. Historical reports should say “positive below promotion floor,” not “performance failure.” |
| PP large-win gate | C13 | Registered PP512 throughput | Demand material result for global arithmetic change | `>=1.05x` | Conservative but candidate exceeded it widely | C13 achieved `1.277663x` and remained fast after rebase. | Predictive positive control, though it does not validate the smaller-gain floor elsewhere. |
| TG regression floor | C13 | Registered TG128 throughput | Protect decode while optimizing PP | `>=0.99x` | Operationally sensible for global change | C13 achieved `.999895x`; later PP rebaseline confirmed value. | Useful cross-workload protection. A workload-gated PP implementation could warrant a different rule. |
| Timing spread/stability | C01, C03, C04, C07, C11, C13, C14 | Repeat variability | Avoid selecting noise/outliers | Phase-specific: 2–5%; examples C04 `3.5%`, C07 exact `3.5%`, C11 exact `2.5%`, PP `2%` | Conservative; exact threshold provenance not empirically demonstrated | Exposed genuine variance and preserved outliers. In C03 it stopped E2E; in C14 the spread correctly makes the `.998916x` PP direction inconclusive. | Useful confidence diagnostic. It should trigger replication or uncertainty labeling, not imply the candidate is slow. |
| Resource/no-spill | Compiled candidates | VGPR, SGPR, LDS, private, scratch, spills | Prevent pathological kernels and explain timing | No spills/private; phase-specific static limits | Safety portion justified; occupancy thresholds often unavailable | C13 increased runtime VGPR 232→256 and still won 27.8% PP. C14 cut LDS but did not improve PP. | Resource counts were not predictive of direction by themselves. Keep as diagnostics unless a hard architectural limit is violated. |
| Achieved occupancy/bandwidth | C04, C07, C11 | Runtime hardware counters | Establish mechanism | Desired but unavailable | Tooling failed/timeouts | No joinable result; static values were correctly not relabeled as achieved. | Missing diagnostic, never a valid veto. Do not repair profiler before evidence warrants it. |
| Static writeback invariant | C06, later C14 | Whether kernel partition is internally valid | Prevent invalid build/execution | `nwarps*tile_C::I==mmq_y` | Direct source invariant | C06 invalid (`8*16!=64`); paired C14 valid and safely ran. | Strong structural gate. It establishes implementation validity, not performance. |
| K-loop/reduction model | C05 | Predicted loop count and reduction cost | Avoid low-value wave sweep | 16 waves offered no loop-count reduction over 12 | Static arithmetic | Candidate never built; performance UNKNOWN. | Appropriate prioritization evidence, not proof of regression. |
| Reference-premise/source-mapping gate | C09 | Whether reference idle-lane mechanism exists locally | Avoid porting an inapplicable technique | Premise must map to llama.cpp/gfx1201 | Direct source/ISA evidence | No port was run. Prior non-equivalent 4/12-wave evidence was flat for one hotspot. | Strong feasibility screen; actual port performance remains UNKNOWN. |
| Feasibility/ranking gate | C10/C11 | Adjacency, consumer count, private format, modeled traffic, complexity | Select one bounded implementation | Hard feasibility pass; relative rank | Trace/source evidence; modeled bytes not measured gain | Selected C11 produced positive PP. C10 was never measured. | Useful scope control. It cannot support a negative claim about C10. |
| Phase isolation/no-combination | C07–C15 | One change at a time | Preserve attribution | Mandatory | Methodologically justified | Enabled clean C13/C14 attribution but left old candidates untested on the later baseline. | Useful for causal evidence; requires a later combination phase so it does not become permanent omission. |
| Stop-on-any-failed-gate | C03, C04, C07, C11, C14 | Composite policy | Minimize GPU time and promotion risk | Any mandatory failure stops later tests | Conservative policy, not empirically calibrated | Prevented TG/E2E for C03/C04/C14; rejected C07/C11 despite positive PP. | The central gate-design error. Safety failures should veto; diagnostic/local performance misses should not overrule already-measured whole-workload gains. |

## False-negative findings

### Confirmed

1. **C07 shared Q8 gate/up:** isolated `0.999896x` and PP `1.003627x`. The local `1.01x` gate and PP `1.005x` promotion floor turned a positive PP measurement into a historical rejection.
2. **C11 SwiGLU→D4/down:** isolated `.995165x/.992729x` and PP `1.005222x`. The whole-PP gate itself passed, but the mandatory exact-chain gate vetoed promotion.

These are false negatives with respect to the question “did registered PP throughput improve?” They are not yet proof of production usefulness, because effect repeatability on the current stack, maintenance cost, serving stability, and model behavior were not tested.

### Unresolved stops

- C03’s selected shapes improved materially, but fallback and spread gates prevented TG. The whole TG direction is UNKNOWN.
- C14’s local chain loss prevented TG. Because it was a global PP geometry candidate and whole PP was effectively flat, the historical rejection is understandable for PP; TG remains NOT MEASURED rather than a regression.
- C05, C06, C08–C10, C12, and C15 lack candidate performance results. Their static evidence ranges from structural invalidity (C06) to prioritization predictions (others). Only C06’s exact implementation is definitively invalid.

## Gate policy learned from the evidence

- Hard computational correctness, valid dispatch, architectural invariants, graph replay, and TP/P2P integrity may remain vetoes.
- Local speed, static resource, and variance gates should control confidence and test order. They should not erase a measured whole-workload result.
- Report measured direction separately from a promotion threshold: `+0.36%, below 0.5% promotion floor` is faithful; `failed performance` is not.
- If the candidate can be workload/shape gated, measure the intended workload before demanding neutrality on unrelated shapes.
- Semantic/model-quality claims require semantic/model tests. No such degradation evidence exists in this phase range.
