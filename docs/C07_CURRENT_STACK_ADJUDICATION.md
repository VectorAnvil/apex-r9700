# C07 Current-Stack Adjudication

## Outcome

C07 remains a small positive PP option on the current production stack:

- PP512: `1.0013055939x` mean (+0.1306%) and `1.0015909794x` median (+0.1591%) across six selector-off and six selector-on processes.
- TG128: `0.9997344775x` mean (−0.0266%) and `1.0000277845x` median (+0.0028%) across six processes per side; adjudicated flat.
- Behavior: health, ordinary chat, strict JSON schema, and forced tool-call checks passed.

C07 is retained in the promotion ledger but was not registered or promoted. Its current gain is smaller than the historical +0.3627%, consistent with Phase 13 reducing the relative cost of duplicated Q8 preparation.

## Scope and provenance

- Source commit: `4695f001fece1660d8bb1b3748f50726ddcc100b` (llama.cpp build 10457).
- Starting stack: Direct-P2P + Phase 13 Q6_K F32 cast + FA-1 gfx1201 rocWMMA.
- Isolated worktree: `/home/adam/workspaces/ChatGPT/Apex/results_phase04_15_c07_rerun_20260819/worktrees/current-c07`.
- Evidence root: `/home/adam/workspaces/ChatGPT/Apex/results_phase04_15_c07_rerun_20260819`.
- Model: `/mnt/storage/ai/models/llm/qwen3.8-27b-unsloth/Qwen3.8-27B-Q6_K.gguf`.
- llama-bench SHA-256: `d15e272a80480a6d021e2b4efda08aa454b50fd0e9d7905c2ecfcfcf4cdeba1d`.
- HIP library SHA-256: `0cac012d9ed32349bf6acb8df444a5a653dd4d3ef01e135ddcaaa70cb5b6130c`.

The selector stayed default-off through `GGML_CUDA_Q6K_SHARED_Q8_GATE_UP=1`. It matches only gfx1201 HIP, Q6_K gate/up weights, the shared F32 normalized activation, Qwen layer names, un-swapped SwiGLU, and the exact PP shape `8704x512x5120`. The candidate quantizes the common input once, then reuses it for the two existing independent MMQ launches. It does not fuse the MMQs or alter the F32 SwiGLU.

## Current-source port

The historical core applied structurally, but current MMQ interfaces required a narrow compatibility update:

- Supply the current `mmq_args::y_scale` field as null for Q8_1.
- Use the current `ggml_cuda_mmq_get_J_max` helper to size scratch.
- Remove the obsolete `use_stream_k` aggregate argument; current MMQ dispatch selects its configuration internally.

No algorithm, selector, tensor ownership, output layout, or numerical contract changed.

## Correctness and dispatch

- Correctness passed 8/8 rows: selector off/on, exact 512-token graph and 256-token fallback, on both GPUs.
- Dispatch passed 6/6 records:
  - exact selector off: two Q8_1 quantizers and two Q6_K MMQs;
  - exact selector on: one Q8_1 quantizer and two Q6_K MMQs;
  - selector-on 256-token fallback: two Q8_1 quantizers and two Q6_K MMQs.
- The test exercises the complete gate/up/SwiGLU graph with deterministic inputs under the normal graph-enabled build.
- C07 introduces no new GPU kernel. It reuses the existing quantizer and Phase-13 MMQ kernels, so their kernel resource envelopes remain unchanged.

## Real-model attribution

A supplemental PP512 trace with graphs disabled for profiler stability recorded:

| Variant | Q8_1 quantizer launches | Q6_K MMQ launches |
|---|---:|---:|
| Selector off | 1,600 | 1,412 |
| Selector on | 1,344 | 1,412 |

The exact 256-launch reduction confirms that C07 removes duplicated preparation while preserving matrix execution. This trace is dispatch evidence only; performance claims come from unprofiled normal-graph runs.

## PP512 result

Both schedules used the same candidate binary and changed only the default-off selector. Each process used the production Qwen3.8 Q6_K model, both R9700s, tensor split `1/1`, batch 2048, ubatch 512, FA on, Q8 K/V cache, five internal repetitions, 12 CPU threads, and 99 GPU layers.

The first ordering produced +0.0872% mean and +0.0817% median. The reverse ordering produced +0.1740% mean and +0.1063% median. Aggregated:

- Baseline mean/median: 1018.382753 / 1017.992134 t/s.
- Candidate mean/median: 1019.712348 / 1019.611739 t/s.
- Speedup mean/median: `1.0013055939x` / `1.0015909794x`.

The effect is positive but small and the sample ranges overlap. It should be retained for portfolio testing, not treated as independently sufficient justification for promotion.

## TG128 guard

The first schedule was slightly negative and the reverse schedule slightly positive. Aggregated across six samples per side:

- Baseline mean/median: 36.022684 / 36.027308 t/s.
- Candidate mean/median: 36.013119 / 36.028309 t/s.
- Speedup mean/median: `0.9997344775x` / `1.0000277845x`.

This is flat and well within the regression guard.

## Behavior and restoration

The production-matched C07 server passed health, chat, strict structured output, and forced tool-call checks with full context, tensor split, FA-1, Q8 KV, cache, and embedded MTP settings.

The unchanged canonical service was restored afterward. Final state: PID 1405011 on port 8083 with `/health` returning `{"status":"ok"}`. No C07 files were copied into the registered source, build, launcher, or service state.

## Promotion guidance

Retain C07 as a +0.13–0.16% PP option. Do not add that percentage to C11 or C03c. The next portfolio phase must build the exact desired candidate combination and measure it as a new stack, because C07 and C11 both alter PP feed paths around Phase-13 MMQ and may interact.
