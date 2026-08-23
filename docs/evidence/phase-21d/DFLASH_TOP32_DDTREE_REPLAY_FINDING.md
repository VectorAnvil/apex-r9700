# DFlash Top-32 and DDTree Replay Finding

## Result

Phase 21D passes its recorder gates and authorizes a separate parent-aware
correctness phase. It does not authorize GPU tree execution, performance
claims, or promotion.

Two independent normal-graph recorder processes matched the accepted Phase
21C `n_max=12` baseline for all 192 token IDs and response content. Their 55
cycle distribution payloads were identical after excluding timing fields.
Fifty cycles contained the full 12-position horizon; five terminal cycles were
shortened by the server's remaining-token budget and excluded from replay.

Every replayed row contained 32 unique, descending tokens with finite F32
log-probabilities normalized against the full vocabulary. Across all 630
recorded canonical positions, 496 (78.73%) appeared in the DFlash top 32.

## Policy Results

The pinned official DDTree best-first construction was replayed host-only
against the exact serial token stream. Rectangular and chain-seeded policies
are explicitly Apex comparators.

| Policy | Nodes | Mean depth | Rescued vs linear | Worse cycles | Nodes / committed token |
|---|---:|---:|---:|---:|---:|
| linear-12 | 12 | 2.60 | 0 | 0 | 3.33 |
| best-first-8 | 8 | 2.68 | 15 | 6 | 2.17 |
| best-first-12 | 12 | 3.02 | 21 | 4 | 2.99 |
| best-first-20 | 20 | 3.20 | 24 | 1 | 4.76 |
| chain-seeded-20 | 20 | 3.14 | 22 | 0 | 4.83 |
| best-first-32 | 32 | 3.36 | 27 | 0 | 7.34 |
| 3x4 | 12 | 2.02 | 5 | 9 | 3.97 |
| 4x3 | 12 | 1.86 | 7 | 12 | 4.20 |

The requested 20-node point is a reasonable acceptance knee: pure best-first
improves mean canonical depth 23.08% over linear-12, while gains above 20
saturate quickly. It is not automatically the fastest choice because node work
per committed token rises substantially. Chain-seeded 20 is the safer
correctness reference because it retains the complete top-1 chain and had no
cycle-level depth regression.

## Whole-System Evidence

Median instrumented stage times in recorder run 1 were:

| Stage | Median |
|---|---:|
| target feature gather | 60.091 ms |
| DFlash encode | 0.307 ms |
| K/V injection | 0.409 ms |
| DFlash block decode | 9.123 ms |
| top-32 export | 13.486 ms |
| target verification | 2.987 ms |
| accept/rollback | 0.438 ms |

The top-32 export is instrumentation overhead and invalidates recorder
throughput as a benchmark. The much larger target-feature gather is an actual
pipeline boundary worth revisiting only after tree correctness exists.

Normal graphs remained enabled and the first recorder run reported 49 graph
reuses. The fresh recorder build's normalized gfx1201 MMVQ and Q6 MMQ ISA were
identical to Phase 21C. Target HIP, Q6 arithmetic, P2P, all-reduce, and
registered Phase 13 assets were unchanged.

## Limits

Replay follows the known serial canonical stream. It does not establish target
logits on alternate branches, ancestor-only attention in llama.cpp, per-node
Qwen35 Gated DeltaNet/conv state, branch-aware KV ownership, rollback, graph
identity, or tensor-split GPU correctness. Those are Phase 21E gates.

The fixed `3x4` and `4x3` layouts are rejected as primary policies. The
`n_max=15` numerical divergence remains unresolved and is not relaxed by this
result.
