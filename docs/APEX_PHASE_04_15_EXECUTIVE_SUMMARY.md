# Apex Phases 04–15 Forensic Audit — Executive Summary

## Plain findings

The campaign contained **15 meaningful candidates or candidate-level sub-experiments**. It also contained three evidence-only control/feasibility studies that are documented in the ledger but excluded from candidate counts.

| Question | Answer |
|---|---:|
| Meaningful candidates | **15** |
| Accepted/banked | **1** |
| Formally rejected/closed no-go | **8** |
| Superseded because the attempted implementation was defective | **1** |
| Held/not executed/not selected | **5** |
| Rejected after a measured meaningful real E2E regression | **0** |
| Hard correctness failures | **1 implementation revision** |
| Positive whole-workload result but rejected by intermediate gates | **2** |
| Promising local result but no E2E trial | **2** |
| Static/predictive performance never measured | **7** |
| Effectively flat/inconclusive | **2** |
| Genuinely worth a minimum rerun now | **3** |

The eight formal historical rejections/no-go closures were C01, C03, C04, C06, C07, C09, C11, and C14. C02 was superseded after a genuine dispatch defect. C05, C08, C10, C12, and C15 were held, excluded, displaced, or not selected without candidate performance. These categories sum to all 15 with C13, the one accepted win.

## The uncomfortable result

**No rejected candidate in this phase range was rejected because a relevant whole-model workload was measured to be meaningfully slower.**

- C14 had a measured local Q6 regression, but its whole PP result was `0.998916x`; the mean delta was smaller than the observed candidate spread, so the honest whole-workload classification is flat/inconclusive rather than measured real regression.
- C01 had severe exact-shape regressions, but full TG was never run.
- C04 was locally flat and never reached TG.
- C06 and C09 were stopped by structural/source reasoning and never had candidate GPU performance.
- C07 and C11 actually improved whole PP.

This does not mean every rejection was wrong. It means the historical evidence does not support describing any of them as a measured whole-model performance loss.

## Confirmed false negatives

Two mandatory isolated-operation gates overruled positive whole-model evidence:

1. **Phase 10 shared Q8 gate/up (C07):** isolated pair `0.999896x`; whole PP **`1.003627x`**. It was rejected because isolated speedup was below `1.01x` and PP was below the `1.005x` promotion floor.
2. **Phase 12 SwiGLU→D4/down (C11):** isolated chain `0.995165x/0.992729x`; whole PP **`1.005222x`**. The PP gate passed, but the isolated `1.01x` gate vetoed registration.

These are proven gate false negatives for PP direction. They are not yet proven production wins: both effects were below 1%, predate Phase 13, and lack current-stack repetition and serving validation.

## Positive control

Phase 13’s explicit F32 conversion (C13) is correctly represented as a banked win:

- Q6 trace: `1.617272x`
- PP512: `814.362714→1040.481514 tok/s`, `1.277663x`
- TG128: `0.999895x`
- Phase 14A rebaseline: `1042.144 tok/s`

It passed numerical, graph, TP/P2P, PP, TG, and resource gates and survived rebasing. Its larger VGPR allocation also proves that resource counts alone did not predict throughput direction.

## Which gates were useful

Useful and justified vetoes:

- hard numerical correctness;
- valid dispatch and selector behavior;
- no crash/hang/NaN/corruption;
- graph capture/replay;
- tensor-parallel/P2P integrity;
- architectural structural invariants such as `nwarps*tile_C::I==mmq_y`.

Useful diagnostics that were overpromoted into vetoes:

- isolated-operation minimum speedup;
- stable fallback floor on workload-unweighted shapes;
- arbitrary whole-PP minimum promotion margins;
- static VGPR/LDS/occupancy/resource predictions;
- stop-on-any-failed-gate policies.

The exact-operation gate was demonstrably non-predictive twice. Static resource reductions also failed to predict C14’s outcome, while C13 won despite increased VGPR allocation.

## Did Apex leave plausible gains on the table?

**Yes, plausibly—but the evidence does not establish their present production value.**

At least C07 and C11 had positive whole-PP results that history mislabeled through gate policy. C03 had material selected-shape gains and was never allowed to answer the actual TG question. Seven other candidates have UNKNOWN performance because they were never run, although several would require substantial new implementation and are not automatically worth pursuing.

The minimum evidence-resolving action is three current-baseline reruns, in order:

1. C11 SwiGLU→D4/down.
2. C07 shared Q8 gate/up.
3. C03 corrected shape-gated four-wave, TG128 only.

No other resurrection work is recommended before those adjudications. The audit does not authorize the reruns.

## Model correctness finding

No phase-four-through-fifteen candidate was shown to degrade semantic/model quality. Those tests were not performed. C02 had a genuine hard computational/dispatch failure. Other compiled candidates passed their defined numerical tolerances. Exact token parity, tool calling, structured output, agent behavior, and long-running inference were `NOT MEASURED` across this candidate set.

## Bottom line

The campaign’s strongest safety gates were sound. Its strongest performance mistake was treating local performance gates and promotion margins as authoritative even after whole-model evidence existed. Two candidates were rejected with measured positive PP; one additional TG candidate was stopped before the workload that mattered. That is enough to justify a small adjudication phase, not a broad revival campaign.
