> Historical research record. Statements about the selected service are as of this report, not current deployment status. See [the V2 index](../README.md). Local raw evidence, private runtime files and logs are not included; local artifact paths are provenance references, not downloadable links.

# Prefill scheduling and Q6 verification combination, 2026-09-30

Status: combined operator, whole-model and completed-task gates passed. Retain as a qualified experimental combination. The worktree starts at cbc3985 and replays three pure kernel commits separately: prefill scheduling e98a42c, exact integer multiplication637ec67, two-row verification86b7942. The original grouped implementation and known-good binaries remain untouched.

The isolated mul24 change already passed its full8K-262K matched-MTP2 model curve, improving cached TG4.08-5.37% with15 exact paired outputs, identical first-step full logits and preserved checkpoints. Rows2 separately passed120 CPU-reference cases and40 exact GPU outputs, improving large two/three-column operations12.68-18.42%; it has no isolated whole-model speed claim. This experiment tests whether both Q6 changes coexist with the qualified prefill scheduling configuration. Their standalone percentages must not be added.

The candidate rebuilds only prefill dispatch, its isolated scheduling template instance and MMVQ. All other frozen objects remain hash-verified. Both Q6 compile gates apply only to MMVQ. The reference is the already qualified20260930-schedule-mtp2-stack/build-stack/bin, not a new baseline rebuild. Both model lanes use schedulingON and MTP2; the measured incremental effect is the Q6 combination. Build receipts retain both original baseline and scheduling-control artifact hashes.

Combined gfx1201 width2 uses54VGPR/3584bytes LDS; width3 uses61VGPR/5376bytes LDS, identical resources to rows2 alone. The combined code contains the intended16/24 mul24 instructions in those two-row kernels. No scratch/spills. One-row resources remain32VGPR/896bytes and ordinary width4 remains47VGPR/0LDS. These are static resources, not measured performance or achieved occupancy.

run_stack_screen.py compares frozen, qualified scheduling control and new candidate for120 CPU cases and40 candidate GPU output hashes. Symmetric control/mul24/rows2/stack/stack/rows2/mul24/control timings on each card produce256 points across representative matrix geometries. It reuses the existing seeded Q6 backend harness. This separates beneficial interaction from merely copying the faster single change; no model test starts before the operator/numerical gate is reviewed.

validate_stack_model.py --execute then runs42 requests: boundary, grow and two repeated128-token windows at8K/32K/64K/128K/192K/261632, plus nearfull history edit and restore. It records actual process argv, gates and mapped libraries for both model lanes after readiness, checks first full logits and outputs, and preserves checkpoint/grid settings. Both lanes enable prefill scheduling, so this comparison does not re-estimate the scheduling-only PP gain. Review the complete curve before accepting the combined configuration.

If the model gate passes, validate_stack_capabilities.py --execute compares24 completed tasks at131072/260096 with the qualified saved MTP2 reference. Requests, output budgets, reference build and grader hashes must match. It covers tools, JSON, retrieval, arithmetic, code and longer deterministic reasoning with thinkingON; no tools are executed. This qualifies the new combination rather than rerunning every standalone kernel through the same capability suite. It is not a general population-level accuracy proof or a new original-grouped-versus-Core evaluation.

Each GPU stage checks the selected Vivi trial is idle, stops only that authorized service, and restores it in finally. Compilation and GPU timing never overlap. Independent restoration verification checks original binary hashes, mapped libraries and health. No system libraries, drivers, production sources/binaries or usage resets change.

## Completed interaction screen

Exec53922 completed120 CPU-reference cases,40 byte-exact candidate GPU outputs and256 operator timings. All targeted large width2/3 matrices benefit from combining the changes, on both cards.

| GPU | Rows | K | Width2 gain vs control | Width3 gain vs control | Width2 gain vs mul24 | Width3 gain vs mul24 | Width2 gain vs rows2 | Width3 gain vs rows2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3072 | 5120 | +19.66% | +26.02% | +14.23% | +16.52% | +4.58% | +8.34% |
| 0 | 5120 | 17408 | +24.94% | +30.86% | +16.15% | +18.73% | +9.65% | +12.55% |
| 0 | 8704 | 5120 | +25.62% | +31.47% | +17.65% | +20.56% | +8.33% | +11.24% |
| 1 | 3072 | 5120 | +20.18% | +24.30% | +13.83% | +14.67% | +6.55% | +7.84% |
| 1 | 5120 | 17408 | +23.59% | +30.09% | +15.25% | +17.90% | +9.76% | +12.80% |
| 1 | 8704 | 5120 | +25.40% | +31.26% | +17.17% | +20.04% | +8.68% | +11.32% |

Small/fallback shapes remain part of the decision: GPU0 m512/n1 loses5.08% against control (control8.79/9.53us, candidate9.65/9.65us) and GPU1 m5120/k17408/n1 loses0.55%. The full-model curve must determine their net effect. Do not present operator gains as generation gains.

The selected trial was restored and independently verified as PID2229251, with11 unchanged baseline artifacts,7 original mapped libraries, MTP1 and gatesOFF. The prepared model gate now additionally requires exact prompt/cache/generation work and draft/accept counts, so unequal work cannot silently masquerade as a kernel gain. No overlapping compilation or GPU experiments are permitted. Evidence: screen/PASSED.json, numerical.json, results.json and RESTORED_INTEGRITY.json.

## Completed whole-model comparison

Exec67368 exited0. All21 paired first-step full-vocabulary logits and128-token outputs match exactly. All26 within-lane grow/repeat/history-restore comparisons also match. Every paired request has identical cache/prompt/generation work and drafted/accepted token counts. Both lanes reuse232960 tokens and process28672 after near-full history edit and restoration. The selected MTP1 trial was independently verified as PID2251643 with11 unchanged artifacts,7 original mapped libraries and experimental gatesOFF.

| Context | Control growing PP | Combined growing PP | PP change | Control cached TG | Combined cached TG | TG change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 1128.85 | 1121.32 | -0.67% | 61.32 | 67.34 | +9.81% |
| 32768 | 1061.75 | 1054.96 | -0.64% | 60.84 | 66.03 | +8.53% |
| 65536 | 910.90 | 910.30 | -0.07% | 58.51 | 63.79 | +9.04% |
| 131072 | 740.25 | 740.55 | +0.04% | 55.30 | 59.82 | +8.18% |
| 196608 | 589.33 | 589.35 | +0.00% | 49.77 | 53.61 | +7.72% |
| 261632 | 490.70 | 490.88 | +0.04% | 45.26 | 48.61 | +7.38% |

Units are tokens/s. TG averages two cached128-token windows. PP is growing-prefix processing, not cold full-context processing at each depth. Both lanes already have prefill scheduling and MTP2; this table measures the increment from the two Q6 changes together. It does not re-estimate the earlier scheduling or MTP-depth gains. The two short-context PP reductions remain in the record; the generation-targeted change leaves deep PP within0.07% in this run. This is one ordered candidate-first comparison, not a population-level significance estimate.

The16K boundary improves9.40% TG. Near-full edited-history TG46.22 ->49.65 (+7.41%); restored-history44.36 ->47.62 (+7.36%). No context-curve reversal appears. Speculative step and acceptance counts match, so the speedup is not a different accepted-token workload. Logged per-step elapsed times include drafting and other overhead; they are not isolated attention or verification kernel times.

Recommendation at this stage: retain the combination for its completed-task gate. It passes model numerical/checkpoint qualification and adds7.38-9.81% generation speed over qualified scheduling/MTP2. Do not call it broadly accuracy-qualified or change the selected serving configuration before the prepared24-task comparison is reviewed. Evidence: model-curve/PASSED.json, comparisons.json, within-lane.json, analysis.json (including context_curve), both runtime.json files and RESTORED_INTEGRITY.json.

## Completed capability gate

Exec94013 exited0. All24 objective tasks pass in both the saved qualified MTP2 reference and the new combined build, at131072 and260096context targets with thinkingON. All24 complete-message and first-step full-vocabulary-logit comparisons are exact. There are6tool-call finishes and18normal stops, no length truncations,5157total output tokens and a largest completion of768tokens. Generated tool calls were graded but not executed. Requests, output budgets, reference artifacts and grader hashes were checked before launch.

This covers tool selection/arguments, missing-record handling, early/middle/late retrieval, JSON joining, ledger reasoning, modular arithmetic, bracket code, longer structured reasoning and a repeated tool request. It is a targeted compatibility screen on the existing task set, not a new original-grouped-versus-Core population study. The complete answers provide stronger evidence than the earlier fixed128-token windows; neither proves all future workloads equivalent.

The selected MTP1 trial was restored and independently verified as PID2266196, with11 unchanged baseline artifacts,7 original mapped libraries and new experimental gatesOFF. The first launch attempt had stopped before service mutation because the previous benchmark port was in TIME_WAIT; inspection found no listening process, and the retry started normally after it expired. No duplicate GPU run was launched.

Recommendation: RETAIN the scheduling/MTP2 plus mul24/rows2 combination as a qualified experimental candidate for controlled use. It adds7.38-9.81% generation speed across the measured curve, preserves deep prefill speed and checkpoints, and matches the completed capability reference on24/24 tasks. The selected serving default is still the original corrected grouped MTP1 trial. Evidence: capability-check/PASSED.json, comparisons.json, stack-mtp2/results.json, runtime.json and RESTORED_INTEGRITY.json.

The next isolated source experiment is20260930-grouped-narrow-query at baselinecbc3985, kernelbec81de. It tests fewer unused query columns while preserving four partials per query; no new measured result yet. The original grouped kernel remains frozen.
