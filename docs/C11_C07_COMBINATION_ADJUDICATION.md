# C11 + C07 Combination Adjudication

## Outcome

The exact C11 + C07 combination is a repeatable positive PP512 portfolio candidate on the current production stack:

- PP512 aggregate: `1.0063610674x` mean (+0.6361%) and `1.0061525197x` median (+0.6153%), with six samples per state across two opposite schedules.
- Interaction: essentially additive. The measured aggregate mean is `1.0063610674x`; the descriptive product of the independently measured same-binary C11 and C07 effects is `1.0063286188x`.
- TG128 guard: `0.9989841945x` mean and `0.9999780667x` median across six processes per side. The median is flat; the mean reflects two early low/noisy candidate processes. Neither exact 512-token selector can dispatch at TG width.
- Behavior: health, ordinary chat, strict JSON schema, and forced tool-call checks passed with both selectors enabled.

This is the strongest adjudicated PP-only portfolio option from the Phase 4 retained candidates so far. It is eligible for the final promotion decision but was not copied into the registered source, build, launcher, or service.

## Scope and provenance

- Source commit: `4695f001fece1660d8bb1b3748f50726ddcc100b` (llama.cpp build 10457).
- Starting stack: Direct-P2P + Phase 13 Q6_K F32 cast + FA-1 gfx1201 rocWMMA.
- Candidate set: C11 Q6_K SwiGLU-D4/down specialization plus C07 shared Q8_1 gate/up preparation.
- Isolated worktree: `/home/adam/workspaces/ChatGPT/Apex/results_phase04_15_combo_c11_c07_20260819/worktrees/current-c11-c07`.
- Evidence root: `/home/adam/workspaces/ChatGPT/Apex/results_phase04_15_combo_c11_c07_20260819`.
- Model: `/mnt/storage/ai/models/llm/qwen3.8-27b-unsloth/Qwen3.8-27B-Q6_K.gguf`.
- Combined worktree diff SHA-256: `ab3eb4bfa73de329b649bc90104a50782fbae1ba2d0821ee5b7911235dbd5aa0`.
- llama-bench SHA-256: `468ee02f63a444479ebe9f64bd55713c9cb6802024ec8260ce401559641a6fce`.
- llama-server SHA-256: `2d59604e66bd20b5f30828c558c15479db804f047037ef22b8d23ed78b3fc603`.

Both selectors remained default-off. The four performance states used one binary and changed only:

- C11: `GGML_CUDA_Q6K_SWIGLU_D4_FFN_DOWN=1`.
- C07: `GGML_CUDA_Q6K_SHARED_Q8_GATE_UP=1`.

## Correctness and isolated dispatch

- Correctness passed 32/32 cases across both physical GPUs.
- The matrix covered all four selector states and the exact 512-token and 256-token fallback shapes for both standalone graph harnesses.
- Isolated dispatch passed 20/20 traces across both GPUs:
  - C11 exact: one fused SwiGLU-D4 quantizer, no ordinary SwiGLU, two ordinary Q8_1 quantizers, and three Q6_K MMQs.
  - C07 exact: one ordinary Q8_1 quantizer and two Q6_K MMQs.
  - Both 256-token fallbacks returned to their ordinary kernel sequences.
  - Enabling the unrelated selector did not alter either standalone witness.

The synthetic C11 harness does not contain the `MUL_MAT_VEC_FUSION` node recognized by C07. An initial trace expecting C07 to fire in that harness was retained under `evidence/dispatch-attempt1`; the corrected adjudication uses separate unit witnesses and the real model graph for simultaneous composition.

## Real-model simultaneous attribution

A PP512 trace with graphs disabled for profiler stability recorded:

| State | Ordinary Q8_1 quantizers | Fused SwiGLU-D4 quantizers | Ordinary SwiGLU | Q6_K MMQs |
|---|---:|---:|---:|---:|
| Baseline | 1,600 | 0 | 448 | 1,408 |
| C11 only | 1,344 | 256 | 192 | 1,408 |
| C07 only | 1,344 | 0 | 448 | 1,408 |
| C11 + C07 | 1,088 | 256 | 192 | 1,408 |

The combined state preserves all 256 C11 fused launches and removes C07's additional 256 ordinary quantizer launches. MMQ count is unchanged. This proves that both tuplets apply in the same real Qwen PP graph. The profiled runs are attribution evidence only; performance claims use unprofiled, normal-graph runs.

## PP512 result

Each process used the production Qwen3.8 Q6_K model, both R9700s, tensor split `1/1`, batch 2048, ubatch 512, FA on, Q8 K/V cache, five internal repetitions, 12 CPU threads, and 99 GPU layers.

The first balanced schedule measured:

| State | Mean t/s | Median t/s | Mean vs baseline |
|---|---:|---:|---:|
| Baseline | 1017.222543 | 1017.575321 | — |
| C11 | 1021.623478 | 1021.645874 | `1.0043264227x` |
| C07 | 1019.513215 | 1020.105532 | `1.0022518887x` |
| C11 + C07 | 1024.087583 | 1024.355472 | `1.0067488084x` |

The reverse schedule measured combined `1.0059731027x` mean and `1.0065330899x` median. Aggregating both schedules gives six samples per state:

| State | Mean t/s | Median t/s | Mean vs baseline | Median vs baseline |
|---|---:|---:|---:|---:|
| Baseline | 1016.929195 | 1016.815110 | — | — |
| C11 | 1021.240087 | 1020.937021 | `1.0042391273x` | `1.0040537463x` |
| C07 | 1019.045090 | 1018.818303 | `1.0020806712x` | `1.0019700656x` |
| C11 + C07 | 1023.397950 | 1023.071085 | `1.0063610674x` | `1.0061525197x` |

The combined state is +0.2113% over C11 alone and +0.4272% over C07 alone by aggregate mean. The interaction is neutral to slightly positive within measurement noise; neither candidate cancels the other.

## TG128 guard

Across six baseline and six combined processes:

- Baseline mean/median: 36.034293 / 36.041093 t/s.
- Combined mean/median: 35.997689 / 36.040302 t/s.
- Speedup mean/median: `0.9989841945x` / `0.9999780667x`.

The guard is adjudicated flat. The exact PP recognizers require 512 tokens, the 256-token fallback traces proved non-selection, and TG uses a one-token execution width. The median differs by only −0.0022%; the more negative mean is not supported by dispatch or by the center of the distribution.

## Behavior and restoration

The combined server passed health, ordinary chat, strict structured output, and forced tool-call checks with the production-style 262K context, tensor split, FA-1, Q8 KV, cache, and embedded MTP settings.

The unchanged canonical service was restored afterward. Final state: PID 1475208 on port 8083 with `/health` returning `{"status":"ok"}`.

## Promotion guidance

Retain C11 + C07 as the current leading PP portfolio bundle at +0.62–0.64%. Do not replace the independent C11 or C07 ledger entries; those remain valid smaller promotion choices. The next exact combination, if desired, is C11 + C07 + C03c so the small TG-positive specialization is measured rather than arithmetically assumed.
