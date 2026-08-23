# C11 + C07 + C03c Combination Adjudication

## Outcome

The exact C11 + C07 + C03c portfolio is retained as the leading mixed-workload promotion bundle:

- C11 + C07 PP512 gain remains the previously adjudicated `1.0063610674x` mean (+0.6361%) and `1.0061525197x` median (+0.6153%) versus the production baseline.
- Adding C03c to that exact bundle changes PP512 by `1.0000796782x` mean (+0.0080%) and `0.9997834539x` median (−0.0217%): flat, with no C03c dispatch in the PP graph.
- TG128, C11 + C07 + C03c versus C11 + C07: `1.0006390732x` mean (+0.0639%) and `1.0003742195x` median (+0.0374%) across twelve samples per state.
- The TG mean closely reproduces C03c's independent +0.0661% result.
- Behavior: health, ordinary chat, strict JSON schema, and forced tool-call checks passed with the full triple enabled.

The triple is eligible for the final promotion decision but was not copied into the registered source, build, launcher, or service.

## Scope and provenance

- Source commit: `4695f001fece1660d8bb1b3748f50726ddcc100b` (llama.cpp build 10457).
- Starting stack: Direct-P2P + Phase 13 Q6_K F32 cast + FA-1 gfx1201 rocWMMA.
- Portfolio base: validated C11 + C07 exact bundle.
- Added candidate: C03c Q6_K `6144x1x5120` gfx1201 four-wave MMVQ specialization.
- Isolated worktree: `/home/adam/workspaces/ChatGPT/Apex/results_phase04_15_combo_c11_c07_c03c_20260819/worktrees/current-c11-c07-c03c`.
- Evidence root: `/home/adam/workspaces/ChatGPT/Apex/results_phase04_15_combo_c11_c07_c03c_20260819`.
- Model: `/mnt/storage/ai/models/llm/qwen3.8-27b-unsloth/Qwen3.8-27B-Q6_K.gguf`.
- Combined worktree diff SHA-256: `d9ac35eee1c1f1058a5ef89cbe8b13aa407ec9fb38dcec5a9180376cdf394648`.
- llama-bench SHA-256: `4e33ccc9c0b4c3b0099d86469076559710335f7993ceca4d43c6522b45729a8b`.
- llama-server SHA-256: `45c34cb4a08a22b422501ff59e930e48f53a1622b2758caf4dc437b53e2f74c0`.
- C11 + C07 control HIP library SHA-256: `998e6ec1034fd785b1566bba84d3a9f3e367730e181b719d5631e8ecc0e2b582`.
- Triple HIP library SHA-256: `be00eb82f0b701b7ce360eeec4d2bc67b7cb0583d65e0013547777b7b0ddd25f`.

C11 and C07 remained controlled by their default-off environment selectors, both enabled throughout portfolio testing. C03c remained the previously adjudicated compile-time exact-shape dispatch with no runtime selector branch added to the hot path.

## Build-control corrections

Two controls were caught and corrected before any benchmark data was accepted:

- Automatic HIP detection initially included `gfx1036` and duplicate `gfx1201` targets. The build was regenerated with explicit `CMAKE_HIP_ARCHITECTURES=gfx1201`, matching the validated candidate builds and returning the HIP library to the expected size class.
- A versioned library file alone did not override the executable's `libggml-hip.so.0` SONAME. Matching `libggml-hip.so.0` links were added to both frozen variant directories, and `ldd` verified that control and candidate resolved to different intended libraries.

No correctness, dispatch, or performance result was collected under either invalid setup.

## Correctness and isolated dispatch

- Correctness passed 24/24 cases across both physical GPUs.
- Both frozen HIP libraries were exercised with C11 and C07 enabled.
- The matrix covered C03c exact and fallback shapes plus the C11 and C07 exact and fallback graph harnesses.
- Dispatch passed 8/8 resource traces:
  - C11 + C07 control, exact `6144x1x5120`: eight waves.
  - Triple, exact `6144x1x5120`: four waves.
  - Both libraries, fallback `5120x1x5120`: eight waves.
  - The result was identical on both physical GPUs.

The selected triple kernel used a `32x4x1` workgroup; the control and fallback used `32x8x1`.

## Real-model attribution

Graphs-disabled traces recorded:

| Workload | Variant | Four-wave MMVQ | Eight-wave MMVQ | Total MMVQ |
|---|---|---:|---:|---:|
| TG128 | C11 + C07 | 0 | 86,946 | 86,946 |
| TG128 | Triple | 4,128 | 82,818 | 86,946 |
| PP512 | C11 + C07 | 0 | 4 | 4 |
| PP512 | Triple | 0 | 4 | 4 |

C03c converts exactly 4,128 real TG launches from eight waves to four without changing total MMVQ count. It does not dispatch in PP512. These are attribution traces only; performance claims use unprofiled normal-graph runs.

## TG128 result

The first two opposite schedules produced mixed signs across six samples per state:

- Mean: `0.9998304440x` (−0.0170%).
- Median: `1.0003530754x` (+0.0353%).

Because the expected effect was only about 0.066%, testing was extended to twelve samples per state rather than adjudicating the mixed six-sample result. Full aggregate:

| Variant | Mean t/s | Median t/s | Standard deviation |
|---|---:|---:|---:|
| C11 + C07 | 36.031213 | 36.053706 | 0.094223 |
| C11 + C07 + C03c | 36.054240 | 36.067198 | 0.039648 |

- Mean speedup: `1.0006390732x` (+0.0639%).
- Median speedup: `1.0003742195x` (+0.0374%).

The mean agrees closely with the independent C03c `1.0006605712x` result. The positive effect survives the exact C11 + C07 combination.

## PP512 guard

Across six processes per state:

- C11 + C07 mean/median: 1022.032497 / 1022.045598 t/s.
- Triple mean/median: 1022.113930 / 1021.824278 t/s.
- Speedup mean/median: `1.0000796782x` / `0.9997834539x`.

This is flat. The real-model trace proves C03c has zero selected PP launches, so the small mixed movement is a library-build guard rather than a causal C03c PP effect. The triple therefore preserves the adjudicated C11 + C07 PP benefit.

## Behavior and restoration

The triple server passed health, ordinary chat, strict structured output, and forced tool-call checks with the production-style 262K context, tensor split, FA-1, Q8 KV, cache, and embedded MTP settings.

The unchanged canonical service was restored afterward. Final state: PID 1554193 on port 8083 with `/health` returning `{"status":"ok"}`.

## Promotion guidance

Retain C11 + C07 + C03c as the leading mixed-workload bundle:

- PP512: preserve approximately +0.62–0.64% from C11 + C07.
- TG128: add approximately +0.04–0.06% from C03c.

Keep C11, C07, C03c, and C11 + C07 as independent smaller promotion choices in the ledger. Workload-specific gains are not summed into a single percentage.
