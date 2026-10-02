> Historical research record. Statements about the selected service are as of this report, not current deployment status. See [the V2 index](../README.md). Local raw evidence, private runtime files and logs are not included; local artifact paths are provenance references, not downloadable links.

# Grouped trial checkpoint coverage

Vivi changed its serialized prompt around 29K-32K while requests contained 35K-45K tokens. The prompt grid kept only checkpoints near the end of the preceding prefill. Those states were after the common prefix, so the recurrent model correctly rejected them and replayed the full prompt. The live log recorded nine full replays; unchanged/append-only smoke requests had reused the cache successfully.

`APEX_V2_GRID_CHECKPOINTS=1` permits intermediate checkpoints at completed canonical microbatch boundaries. It changes checkpoint placement only. It does not change token batch geometry, kernel dispatch, recurrent-state validation, or the grouped kernel. Gate OFF preserves original behavior. The trial uses `--checkpoint-min-step 4096 --ctx-checkpoints 72`; checkpoints are roughly 149.6 MiB each, so the configured maximum is about 10.5 GiB of host checkpoint data before other cache allocations. Retention still follows the existing eviction policy.

The isolated build replaces only `libllama-server-impl.so`. All GPU/model libraries and executables match frozen candidate 8608915. The original worktree and binaries remain untouched.

Validation artifacts are in `../validation` relative to this experiment's top-level source directory. Sixteen requests checked the original failure, rebuilt gate-OFF equivalence, changed history at token 32258, a mutation at checkpoint boundary 29184, exact repeated prompts, and an earlier change in an 8192-token prompt. Cached/cold full first-step logits and 32 generated tokens matched exactly. Both ordinary decode and MTP depth 1 passed at 8K; the 45K cases used MTP depth 1.

For the 45053-token prompt with an edit at 32258, the original processed all 45053 tokens in 42.212 seconds. The correction reused 29184 tokens and processed 15869 in 16.702 seconds, a 60.4% reduction in prompt time. An identical repeat reused 44544 tokens and processed 509 in 0.702 seconds. Initial cold prefill was 42.192 seconds original versus 43.160 seconds corrected; across the sampled cold 45K requests the extra cost was about 1-1.4 seconds. The corresponding seed generation rates were 44.92 and 44.94 tok/s over 32 output tokens. These are targeted single-run timings, not a statistical throughput qualification.

The Sept30 curve completed 75 requests across 8192, 32768, 65536, 131072, 196608, and 261632 tokens. All 37 original/corrected pairs matched full first-step logits and generated tokens exactly; restoring the unedited near-full prompt also reproduced its initial logits and 128 output tokens. Maximum observed retention was 66 checkpoints. Artifacts: `../curve-20260930/PASSED.json` and `TRIAL_RESTORED.json`.

| Context tokens | Original grouped TG | Corrected grouped TG | Growing-prefill rate change |
| --- | --- | --- | --- |
| 8192 | 52.83 | 52.58 | -3.3% |
| 32768 | 52.13 | 52.17 | -2.0% |
| 65536 | 50.75 | 50.75 | -1.7% |
| 131072 | 45.51 | 45.53 | -1.7% |
| 196608 | 42.58 | 42.56 | -1.2% |
| 261632 | 38.61 | 38.53 | -0.8% |

TG is tok/s over 128 output tokens on identical repeated prompts, MTP depth 1, both GPUs. Prefill is growth from the preceding context, except the initial cold 8K request; it is not fresh full-context throughput. This is one ordered A/B, not a statistical equivalence claim.

Editing token 235000 in the 261632-token prompt let the original reuse 196096 tokens and process 65536 in 137.417 seconds. The correction reused 232960 and processed 28672 in 63.566 seconds, reducing prompt time by 53.7%. The running driver's original expectation of zero cached tokens was incorrect for this growing-prefix fixture. The saved artifacts were revalidated with the correct requirement of greater reuse; no numerical gates were relaxed. Appending 1/16/64/256 tokens at near-full context required 0.205/0.292/0.413/0.774 seconds of server prompt time with the correction, versus 0.206/0.301/0.416/0.778 seconds original. These are not streaming TTFT measurements.

Vivi was restored to the corrected trial and health checked after validation. Very early prefix changes, evicted states, and unsupported media paths can still require full replay. Changing the serialized history necessarily requires processing the changed suffix. The cache correction is qualified on these fixtures; it does not broaden the original grouped kernel's capability-parity evidence.

Related upstream leads: [history changes and checkpoint invalidation](https://github.com/ggml-org/llama.cpp/issues/24890), [checkpoint eviction policy](https://github.com/ggml-org/llama.cpp/issues/25023). These are related mechanisms; the controlled local reproduction and exact comparisons establish this correction's tested behavior.
