# Apex Phases 04–15 Minimum Rerun Plan

## Scope

This is a proposed adjudication matrix, not authorization. It intentionally excludes new kernel designs and profiler repair. The current registered production build—including Direct-P2P, C13, and all other registered Apex fixes—must be the baseline. Candidate changes must be applied one at a time and default-off.

## Three-candidate matrix

| Order | Candidate | Why history is insufficient | Exact unanswered question | Minimum test | Baseline | Workload | Model-quality test? | Establishes a real win | Establishes a real loss |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | C11 SwiGLU→D4/down | Historical whole PP was `+0.5222%`, but it predates C13/current service and was rejected by a contradictory local gate. | Does the recovered producer boundary still improve current PP/latency repeatably with flat TG and stable serving? | Rebuild recovered patch on current source; numerical/dispatch/graph checks; interleaved A/B PP512 with enough independent processes for a confidence interval; TG128 regression check; one bounded server smoke/soak. Do not require the old isolated `1.01x` gate. | Current registered dual-R9700 production build/config/model | PP primary; TG guard | Bounded semantic/tool/structured-output smoke only because execution path changes numerics; exact token parity is not required. | PP confidence interval is positive and the absolute latency/throughput gain is operationally repeatable; no meaningful TG, crash, NaN, graph, or service-stability regression. | PP confidence interval is meaningfully negative, or hard correctness/stability failure, or material TG/service regression. |
| 2 | C07 shared-Q8 gate/up | Historical whole PP was `+0.3627%`, below the old floor; C13 changed the consumer cost balance. | Is any positive effect still separable from noise on the current stack? | Numerical/dispatch/graph checks; interleaved A/B PP512 only, using more repeats than the historical three processes; stop if confidence interval straddles the predeclared operational minimum. TG smoke only if shared code can affect TG. | Same current baseline | PP primary | Only a brief service smoke if PP effect is repeatable; no token-parity veto. | Repeatable PP/latency improvement exceeding a predeclared operational value (for example, measured service capacity/latency worth more than maintenance cost), with no hard failure. | Repeatable meaningful PP regression or hard correctness/stability failure. A confidence interval spanning zero is inconclusive, not loss. |
| 3 | C03 corrected shape-gated four-wave v2 | Exact selected shapes gained 5.6–11.0%, but the actual TG workload was never run. | Do selected dispatch wins improve current TG128 after weighting real model calls, despite fallback exposure? | Port only the corrected boolean selector; verify exact dispatch and numerical tolerance; interleaved TG128 A/B on both GPUs; collect per-shape dispatch counts to attribute exposure. PP is unnecessary unless shared code path changes it. | Same current baseline | TG primary | Short normal generation and service stability smoke; semantic spot check, not exact-token parity. | Repeatable positive TG/latency result with valid dispatch, numerical tolerance, and no crash/service regression. | Repeatable meaningful TG regression or hard correctness/stability failure. |

## Measurement rules

1. Freeze model, quant, context, tensor split, P2P mode, server flags, clocks/thermal policy, and build provenance.
2. Use interleaved baseline/candidate process pairs rather than comparing widely separated runs.
3. Predeclare an operationally meaningful effect from current serving economics or latency goals. Do not automatically reuse `1.005x`, `1.01x`, or `1.02x`.
4. Report mean/median, dispersion, paired deltas, and confidence interval. If the interval spans zero and no operational floor is crossed, classify as inconclusive.
5. Hard correctness, graph, dispatch, crash, NaN/Inf, and service-stability failures remain vetoes.
6. Exact token parity is diagnostic only unless the production contract explicitly requires it. Use deterministic task/semantic outcomes for model-quality adjudication.
7. Do not combine C07 and C11 until each is independently adjudicated on the current baseline.

## Explicitly excluded from the minimum matrix

- C01 global four-wave: C03 is the safer exact refinement.
- C04/C05: flat 12-wave local evidence and lower information value than C03.
- C06: exact Y64/W8 implementation is structurally invalid.
- C08/C10/C12/C15: require new design/implementation or current tracing, not a historical rerun.
- C14: post-C13 PP already measured effectively flat with clear local regression; TG relevance of this global PP geometry was not established.
- Profiler-counter repair: not necessary to answer throughput direction.

## Stop condition

After these three adjudications, stop and review. Do not begin C08, C10, C12, C15, or combined-candidate implementation without separate authorization.
