# Qwen3.8 Q6_K base vs Heretic vs HuiHui — 2026-08-19

## Outcome

Heretic and HuiHui are performance-equivalent to each other on the promoted dual-R9700 stack. Both are materially faster than the Unsloth base Q6_K for prompt processing, while all three are effectively tied for real MTP-enabled generation at speculative width 2.

The two alternate launchers use the complete production runtime: Direct-P2P, Phase 13, FA-1 rocWMMA, C11+C07+C03c, dual-GPU tensor split, 262,144 context, Q8 KV, and embedded MTP `n_max=2`.

## Models

| Model | SHA-256 | GGUF tensor composition | Loaded model bytes |
|---|---|---|---:|
| Base Unsloth Q6_K | `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` | 456 F32, 361 Q6_K, 49 Q8_0 | 22,873,411,584 |
| Heretic Q6_K | `b2a719fde75244116d7a721d82b10ca0c7aeb5fd586e25490114d31e520f8bed` | 360 F32, 506 Q6_K | 22,420,004,864 |
| HuiHui Q6_K | `a5c159519d7bdba523578977d649dbfc48f2e47277ec77edd77670003b5b2492` | 360 F32, 506 Q6_K | 22,420,004,864 |

All three contain 866 tensors and 27,320,697,856 parameters. Heretic and HuiHui have identical tensor-type counts and are about 454 MB smaller than base. Their additional Q6_K coverage explains why they exercise more of the Apex Q6_K-specialized paths.

## Kernel benchmark

The shallow cells used a balanced `base, Heretic, HuiHui, HuiHui, Heretic, base` schedule with five repetitions per process. Figures below are the median across all ten repetitions per model, which is robust to one HuiHui PP512 and one base PP4096 first-repetition setup stall.

| Cell | Base | Heretic | vs base | HuiHui | vs base |
|---|---:|---:|---:|---:|---:|
| PP512 d0 | 1024.385 t/s | 1092.275 t/s | +6.63% | 1090.950 t/s | +6.50% |
| PP4096 d0 | 987.449 t/s | 1048.565 t/s | +6.19% | 1048.855 t/s | +6.22% |
| PP4096 d65536 | 480.589 t/s | 497.661 t/s | +3.55% | 497.752 t/s | +3.57% |
| TG128 d0 | 36.156 t/s | 36.356 t/s | +0.55% | 36.336 t/s | +0.50% |

The 64K cell is one materialized-depth run per model, so it is a guard/result rather than a high-rep estimate.

## Production server with embedded MTP

The server comparison used two balanced starts per model and ten identical 128-token requests per start. All 60 requests returned exactly 128 committed tokens.

| Model | Mean t/s | Median t/s | MTP accepted/proposed | Acceptance |
|---|---:|---:|---:|---:|
| Base | 60.875 | 60.361 | 1472/2112 | 69.70% |
| Heretic | 60.816 | 59.344 | 1490/2074 | 71.84% |
| HuiHui | 60.703 | 60.078 | 1486/2078 | 71.51% |

The mean spread is only 0.28%, so there is no defensible decode-throughput winner. Heretic and HuiHui have slightly higher MTP agreement, but at width 2 it does not produce a measurable end-to-end advantage over base on this corpus.

## Launcher validation

Both alternate scripts were started independently at the full 262K production context. Both became healthy, created their embedded MTP draft contexts, and stopped cleanly. Logs and PID state are isolated under `/home/adam/.local/shared`.

- `/home/adam/scripts/start-qwen38-heretic.sh`
- `/home/adam/scripts/start-qwen38-huihui.sh`
- Heretic logs: `/home/adam/.local/shared/apex-qwen38-heretic/server.log`
- HuiHui logs: `/home/adam/.local/shared/apex-qwen38-huihui/server.log`

Both scripts use port 8083 by default and require both R9700s, so only one of base, Heretic, or HuiHui should run at a time.

## Recommendation

Heretic and HuiHui remain performance-equivalent, but the subsequent frozen parity gate gives Heretic a production advantage: Heretic passes 10/12 exact ordinary-versus-MTP cases, while HuiHui and the contemporaneous promoted base each pass 8/12. Prefer Heretic if deterministic MTP agreement matters. Heretic is not perfectly parity-safe; `list` and `forced128` still fail.

Either obliterated model is a better PP choice than this particular base Q6_K recipe. Do not interpret the PP gap as an ablation benefit; it comes from the differing GGUF tensor-type recipe.

## Evidence

- Robust analysis: `results_qwen38_q6k_model_comparison_20260819/evidence/analysis.json`
- Raw `llama-bench`: `results_qwen38_q6k_model_comparison_20260819/evidence/llama-bench/`
- Raw server/MTP comparison: `results_qwen38_q6k_model_comparison_20260819/evidence/service-mtp/`
- Exact MTP parity report: `docs/apex-r9700/QWEN38_Q6K_OBLITERATED_MTP_PARITY_20260819.md`
