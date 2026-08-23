# C03 Current-Stack Phase 4 Adjudication

## Outcome

The original three-tuple C03 bundle is rejected, but splitting it identified one very small production-level winner worth retaining:

- Retain `6144x1x5120` as promotion-possible: TG128 `1.0006605712x` mean (+0.0661%) and `1.0006987259x` median (+0.0699%) across six candidate and six baseline processes.
- Keep `5120x1x3072` on the watchlist: `0.9999433562x` mean (−0.0057%) and `1.0004929006x` median (+0.0493%). This is flat/mixed, not a claimed gain.
- Reject `3072x1x5120`: TG128 `0.9848280806x` mean and `0.9865499613x` median.
- Reject all three enabled together: six-sample aggregate baseline 36.029913 t/s versus candidate 35.543486 t/s, approximately `0.9865x`.

Nothing from C03 was promoted into the registered source or production service. The canonical service was restored unchanged and healthy at the end of the run.

## Scope and provenance

- Source commit: `4695f001fece1660d8bb1b3748f50726ddcc100b` (llama.cpp build 10457).
- Starting stack: Direct-P2P + Phase 13 Q6_K F32 cast + FA-1 gfx1201 rocWMMA.
- Isolated worktree: `/home/adam/workspaces/ChatGPT/Apex/results_phase04_15_c03_rerun_20260819/worktrees/current-c03`.
- Evidence root: `/home/adam/workspaces/ChatGPT/Apex/results_phase04_15_c03_rerun_20260819`.
- Model: `/mnt/storage/ai/models/llm/qwen3.8-27b-unsloth/Qwen3.8-27B-Q6_K.gguf`.
- Candidate llama-bench SHA-256: `41f53aa43e62393ca0d267f6266936e2f03866b0875b0a937754215a4df3cbe0`.
- Retained `6144x1x5120` HIP library SHA-256: `44b4c6175078e5a3c1174b5580477e683f62543ce8b776bdbbab019b722b3b82`.

The selector remained restricted to gfx1201/RDNA4, Q6_K, N=1, no IDs, no fusion, and the exact M×K pair under test. Every other call used the existing eight-wave fallback.

## Correctness and dispatch

- Correctness: 18/18 rows passed on the two R9700 GPUs: eight regular Q6_K N=1 slices plus one fused fallback slice per GPU.
- Exact dispatch: 12/12 profiler records passed. Each of the three requested tuples launched four waves on both GPUs; representative `5120x1x5120`, `512x1x5120`, and `5120x1x8704` fallbacks stayed at eight waves.
- Current compiler resources: selected specialization 512 B LDS, 32 VGPR, zero scratch; fallback 1024 B LDS, 32 VGPR, zero scratch.
- Real-model supplemental trace with graphs disabled: each surviving one-tuple TG128 build recorded 4,128 four-wave launches and 82,818 eight-wave launches. This trace is identity evidence only, not performance evidence.
- rocprof with graph capture enabled crashed inside graph evaluation before emitting a trace. Unprofiled runs were stable; exact-shape normal-graph traces had already passed on both GPUs. No profiler repair was attempted.

## Isolated kernel result

The current stack reproduced the historical microkernel wins:

| Tuple | GPU 0 | GPU 1 |
|---|---:|---:|
| `3072x1x5120` | about `1.0632x` | about `1.0665x` |
| `5120x1x3072` | about `1.1048x` | about `1.1131x` |
| `6144x1x5120` | about `1.0658x` | about `1.0767x` |

The representative `5120x1x5120` fallback remained above the 0.99 floor on both GPUs (about `0.9934x` and `0.9923x`). These kernel results authorized model testing but were not treated as promotion gains.

## TG128 adjudication

All runs used the production Q6_K model and settings: tensor split `1/1`, batch 2048, ubatch 512, FA on, Q8 K/V cache, 12 CPU threads, 99 GPU layers, five internal repetitions, and both R9700s.

### Three-tuple bundle

Two opposite-order six-process schedules produced six samples per side overall:

- Baseline mean/median: 36.029913 / 36.0425955 t/s.
- Candidate mean/median: 35.543486 / 35.5641980 t/s.
- Decision: reject the bundle.

### Individual tuples

The first 12-process matrix tested baseline and all three individual tuple builds. A second reverse-order nine-process matrix replicated the two non-rejected candidates.

| Variant | Samples | Mean speedup | Median speedup | Decision |
|---|---:|---:|---:|---|
| `3072x1x5120` | 3 | `0.9848280806x` | `0.9865499613x` | Reject |
| `5120x1x3072` | 6 | `0.9999433562x` | `1.0004929006x` | Flat/mixed watchlist |
| `6144x1x5120` | 6 | `1.0006605712x` | `1.0006987259x` | Retain as small promotion option |

## PP512 guard

Three processes per variant yielded:

| Variant | Mean speedup | Median speedup |
|---|---:|---:|
| `5120x1x3072` | `0.9958757340x` | `0.9951948826x` |
| `6144x1x5120` | `0.9959329174x` | `0.9951664998x` |

Both clear the 0.99 regression floor. Supplemental PP traces recorded only four eight-wave MMVQ launches and zero four-wave launches for either variant, so this cross-build delta cannot be attributed to the selected four-wave kernel executing. It remains a conservative guard caveat for any final combined-stack trial.

## Behavior and production restoration

The aggregate-positive `6144x1x5120` server passed health, ordinary chat, strict JSON schema, and forced tool-call smoke tests with the production context, split, FA, Q8 KV, cache, and embedded MTP settings.

The canonical launcher was then restored. Final state: PID 985486, port 8083, `/health` returned `{"status":"ok"}`. No C03 source, library, or launcher was registered into production.

## Promotion guidance

Do not promote the original C03 bundle. Preserve `6144x1x5120` for the final portfolio experiment, and preserve `5120x1x3072` as a flat watchlist toggle if combination testing is cheap. Any trial with C11 or another candidate must be built and measured as a new combined stack; do not add the independent percentages.
