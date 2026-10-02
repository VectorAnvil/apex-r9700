> Historical research record. Statements about the selected service are as of this report, not current deployment status. See [the V2 index](../README.md). Local raw evidence, private runtime files and logs are not included; local artifact paths are provenance references, not downloadable links.

# Scheduling plus MTP2 combined candidate, 2026-09-30

Completed: exec69460 exited0. All20 paired requests passed exact full-first-logit and128-token comparisons. The selected MTP1 trial was restored and its health, eleven baseline artifact hashes and seven mapped library paths verified. This worktree starts at known-good cbc3985 and carries only the exact scheduling kernel diff from20da13e. It does not include the numerical-changing two-partition candidate. Baseline worktree and artifacts stay untouched.

The scheduling-only model screen passed14 exact full-logit/64-token pairs and observed nearfull growing PP473.40 to491.88tok/s. The MTP2-only cache screen passed20 exact pairs, including execution boundaries and deep history restoration, and observed nearfull repeated TG40.49 to45.54tok/s against a saved reference. Those separate screens do not establish a combined gain. MTP2 completed-answer qualification has now passed:24/24 tasks correct in each lane, all24 complete-message and full-first-logit pairs exact; exec73194 exited0 and restored the verified trial before this combined run.

After that qualification passed, prepare_stack_artifacts.py verified source and artifact hashes, then made independent copies of the existing qualified scheduling binaries into build-stack/bin. It neither recompiles nor hardlinks the known-good files. Kernel sources in this worktree must match the source receipt exactly. The artifact copy is separately named and verified at launch.

validate_stack_curve.py --execute runs candidate first, then a fresh selected-baseline control. Both use the same model, dual-GPU split, Q8 KV, context capacity,512 prompt grid, checkpoint coverage, inputs and128-token output windows. Candidate changes only scheduling gateON and MTP depth2; control uses unchanged baseline and MTP1. At8K/32K/64K/128K/192K/near262K each lane runs growth plus two cached repeats. Nearfull history edit and restoration complete the40-request experiment. First-step full logits and outputs are checked; repeated outputs and restored history must match within each path. These are growing-prefix PP and repeated TG measurements, not cold full-context PP at every depth. The order reverses the earlier scheduling-only A/B, but this is still one ordered combined B/A rather than a repeated statistical study.

The existing guard checks idle selected Vivi and expected executable before pausing it, and restores the same selected MTP1 trial in finally. No compilation, other GPU experiments, driver/library changes or system configuration changes may overlap this run. After launch verify actual mapped libraries and gate values, and after completion verify restoration and artifact hashes. Compare the whole context curve, not just the highest gain. Individual-factor attribution stays with earlier experiments; this run measures whether their combination works.

Recommendation: retain as a qualified experimental combination for a controlled Vivi trial. It passes the current numerical/checkpoint gate and improves TG across the measured context curve. No production binary or configuration has been changed.


## Completed whole-context measurements

Both lanes used dualR9700, identical Q6_K model/Q8 KV,512 canonical prompt grid,4096 checkpoint spacing and72 retained checkpoints. Candidate ran first with schedulingON/MTP2; the fresh unchanged grouped control ran second with MTP1. PP is growing-prefix processing, not a cold full-context request after8K. TG is the mean of two cached128-token windows per context.

| Context | MTP1 PP | Combined PP | PP gain | MTP1 TG | Combined TG | TG gain |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1125.33 | 1127.73 | +0.21% | 52.49 | 60.21 | +14.70% |
| 32768 | 1052.55 | 1051.47 | -0.10% | 52.11 | 60.69 | +16.48% |
| 65536 | 901.72 | 910.32 | +0.95% | 50.76 | 56.22 | +10.76% |
| 131072 | 725.15 | 739.94 | +2.04% | 45.51 | 55.24 | +21.37% |
| 196608 | 572.93 | 589.37 | +2.87% | 42.58 | 49.75 | +16.82% |
| 261632 | 472.69 | 490.91 | +3.85% | 38.00 | 44.35 | +16.71% |

All20 paired full-first-step logits and128-token rollouts match exactly (2560 compared output positions, not independent prompts). All12 grow/repeat pairs per lane match, and each restored nearfull history matches its original grow output. Both deep edit and restore reuse232960 tokens and process28672. Edit PP is450.17 vs468.18tok/s; restore450.47 vs468.36tok/s. Checkpoint coverage is preserved.

The fresh nearfull PP gain of3.85% agrees with the earlier scheduling-only observation of3.90%. The128-token TG gain is16.71%; its absolute rates differ from older64-token windows because the token sequence and draft acceptance differ. Do not compare those windows as identical work. This single ordered B/A supports the combination; within-lane repeats do not establish population-level significance. Prior independent MTP2 capability checks passed24/24 complete tasks in both lanes with exact messages/logits. This does not requalify grouped attention against Core on a new task population.

Evidence: curve/PASSED.json, analysis.json, comparisons.json, CANDIDATE_RUNTIME.json, BASELINE_RUNTIME.json and RESTORED_INTEGRITY.json; build-stack/COPIED.json records source/artifact identity. The next isolated hypothesis is local-memory-only tile fences; no local-fence code was present in these measurements.

Usage: Adam confirmed that he applied the authorized reset himself. No further resets are permitted without his explicit new approval.
