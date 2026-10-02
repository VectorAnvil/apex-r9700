> Historical research record. Statements about the selected service are as of this report, not current deployment status. See [the V2 index](../README.md). Local raw evidence, private runtime files and logs are not included; local artifact paths are provenance references, not downloadable links.

# Current llama.cpp versus the August 31 Apex release

Audit date: September 27, 2026. This is a source and release comparison, not a newly measured performance comparison. No production process, model, library, or launcher was changed.

Follow-up: the [four-build GPU benchmark](four-build-benchmark.md) now provides measured performance, ordinary/MTP parity checks, and profiler evidence. The source audit's MTP-speed hypothesis did not produce a measurable win on this Apex configuration.

## Conclusion

Upstream has incorporated changes that matter to this machine. The strongest immediate candidates are **R9700 D256 Flash Attention tuning (#28102)** for long-context prompt processing and **separate MTP graph caches (#28549)** for generation overhead. **Qwen GDN normalization (#28068), its matching RMSNorm/scale fusion (#29393), tensor-parallel fixes, and cache-restore fixes** also deserve explicit treatment.

Current upstream does not subsume the Apex release. In particular, its HIP AllReduce uses host-memory staging, while Apex has direct peer-VRAM transfer and a fused boundary. The Phase 13 Q6_K cast and packed canonical MTP implementation are also still separate. Replacing Apex with stock upstream can gain some features while losing others; source changes alone do not establish which complete build is faster.

## Exact versions compared

| Identity | Revision | Meaning |
|---|---|---|
| Apex August 31 | `q6k-fused-boundary-20260831` | Local appliance release, including August 31 additions |
| Its upstream base | `4695f001fece1660d8bb1b3748f50726ddcc100b`, build 10457 | Upstream commit dated **August 17**, not August 31 |
| Latest published nightly observed | **b11211**, `d7fb90e8e2494b2908934d956a3202fd60152ee0` | Published September 27; 754 commits beyond the base |
| Master inspected | `c9064dded732d81f90b34e6b33d4fbd77cbfa058` | September 27; 760 commits beyond the base |
| GitHub's latest named stable release | **v0.5.0**, `d2e54583c7452353eb35d40431281f6ee984332f` | September 23; distinct from the newest nightly |

[Nightly b11211](https://github.com/ggml-org/llama.cpp/releases/tag/b11211), [master pin](https://github.com/ggml-org/llama.cpp/commit/c9064dded732d81f90b34e6b33d4fbd77cbfa058), [stable release](https://github.com/ggml-org/llama.cpp/releases/tag/v0.5.0).

The six commits between the nightly and inspected master concern grammar error handling, Vulkan Adreno argsort, CDNA attention above D256, attention tiles for head sizes 40–112, SYCL FWHT, and OpenCL loading. None is the D256/gfx1201 change discussed below. Every merged PR in the main findings below is already in b11211. These are pinned observations; release labels can advance during the day.

The comparison used downloaded upstream base/head archives and the actual August 31 `worktrees/fused-boundary` source snapshot, not an assumed list of Apex patches. The source snapshot has no independent Git history at that location: running `git log` there resolves to the enclosing Apex documentation repository and is not a llama.cpp source revision.

The previous live audit (local evidence reference) pins the server and loaded HIP library. The comparison manifest (local evidence reference) records source hashes, PR merge identities, and archive hashes; all 760 commit titles (local evidence reference) and the Apex patch inventory (local evidence reference) are retained. Full source snapshots and focused diffs are under ignored `results_raw/upstream-20260927/`.

## Changes that directly matter

### 1. R9700 long-context prefill: #28102, merged September 11

This is the important newer attention change. It enables native AMD WMMA attention for D256, changes GQA grouping and tile configuration, and avoids unconditional stream-K splitting when whole tiles provide adequate utilization. It uses the modern `fattn-mma-f16` implementation, **not the old restored rocWMMA FA-1 implementation**. It therefore does not depend on Apex's disabled `GGML_HIP_ROCWMMA_FATTN` option. [Merged PR #28102](https://github.com/ggml-org/llama.cpp/pull/28102)

The author tested an R9700 with Qwen3.8 27B IQ4_XS:

| Published workload | Before | After | Calculated change |
|---|---:|---:|---:|
| PP512 at 40,000 depth | 426.36 t/s | 639.46 t/s | +50.0% |
| PP512 at 150,000 depth | 164.42 t/s | 399.01 t/s | +142.7% |
| TG128 at 150,000 depth | 19.70 t/s | 19.67 t/s | −0.15% |

These are external measurements with substantial prefill variance, different weights, and no Apex stack. The model benchmark table does not explicitly specify KV datatype. They demonstrate a relevant prefill improvement, not a promised Q6_K/Q8 speedup on Cornelius. [Published measurements](https://github.com/ggml-org/llama.cpp/pull/28102)

**Source comparison:** Apex still excludes D256 from its native AMD WMMA selector; its alternative restored path is compiled inactive in the inspected release. Current upstream admits D256 once `query_rows × effective_GQA > 16`. For the model's GQA6, effective GQA is 2, giving a minimum of nine query rows under the other eligibility conditions. Ordinary decode and MTP2 verification remain below that threshold.

Consequently, this can address long-context input processing but is not, by itself, a solution to the generation slope. The older **#26419 remains open**, even though a different merged change now addresses the D256 dispatch limitation. Do not describe #26419 itself as merged. [Open proposal](https://github.com/ggml-org/llama.cpp/pull/26419)

### 2. MTP graph recapture: #28549, merged September 16

Upstream now gives output-producing and no-output batches separate graph-result storage. Previously they alternated through one storage object and could invalidate one another's captured graph. Apex's `llama-context.cpp/.h` still use the old single-result scheme. [Merged PR #28549](https://github.com/ggml-org/llama.cpp/pull/28549)

The published RTX 5090 / Windows / Qwen3.6-35B-A3B Q4_K_M MTP3 test reports 279.04 to 291.33 predicted t/s, about +4.4%, with unchanged acceptance. That is not an AMD measurement. The fix is in backend-independent context code, and HIP uses the shared graph machinery with HIP graph API mappings, making it directly worth testing here. Tensor-parallel graph keys and Apex's extra paths still need runtime verification.

**Expected effect:** reduce repeated graph capture and launch overhead during MTP. This would primarily reduce per-step overhead; it does not remove the context-length-dependent KV scan. Capture counts plus same-prompt acceptance are the smallest useful diagnostic.

### 3. GDN numerics plus fusion: #28068 and #29393

The September 6 normalization fix changes Qwen recurrent-layer q/k normalization from `x / max(sqrt(sum(x*x)), eps)` to the reference form `x * rsqrt(sum(x*x) + eps)`. The old formula remains in Apex's Qwen model code. This is a **model-correctness change**, not simply a faster implementation of identical arithmetic. Cross-version output hashes and MTP acceptance can change legitimately. [Normalization fix #28068](https://github.com/ggml-org/llama.cpp/pull/28068)

Its implementation introduces RMSNorm followed by scaling. September 25's #29393 fuses that pair, removing up to 96 additional scale launches per microbatch for the 48 GDN layers in Qwen3.8-27B. Published dual-GTX-1080-Ti, Windows, Q4_K_XL tests without CUDA graphs recovered approximately 4–5% of server prefill throughput; ordinary TG improved only about 0.6%. The fusion preserves the corrected formula, not the August 31 formula. [Fusion #29393](https://github.com/ggml-org/llama.cpp/pull/29393)

The fusion is in code compiled for HIP as well, without a NVIDIA-only guard around this path. Its R9700 benefit remains unmeasured. **Evaluate the correction and its fusion together**, and establish a new correctness reference for the corrected model. Preserve ordinary-versus-MTP parity within that reference; do not label every difference from August hashes a regression.

### 4. Multi-GPU correctness and optional fused QKV

- **#27574, August 23:** fixes segmented recurrent-cache split propagation, attention gate placement, and other meta-backend split behavior. The corresponding Apex code still differs from this fix. It is relevant to validating Qwen tensor parallelism, but does not prove the already-tested equal-split appliance is currently incorrect. [PR #27574](https://github.com/ggml-org/llama.cpp/pull/27574)
- **#22780 plus #28965:** optional conversion to fused Q/K/V weights, followed by Qwen full-attention Q/gate split and granularity fixes. The latter was tested with Qwen3.8 27B and Qwen3.6 35B-A3B on four P100s. This is not automatically enabled for an existing GGUF with separate full-attention Q/K/V weights. Fused shapes would also require reconsidering Apex's exact-shape projection kernels. [Conversion support](https://github.com/ggml-org/llama.cpp/pull/22780), [Qwen TP correction](https://github.com/ggml-org/llama.cpp/pull/28965)
- **#29294, September 25:** handles fused QKV with unequal K/V head dimensions. Our D256/D256 model does not meet that unequal-head condition. [PR #29294](https://github.com/ggml-org/llama.cpp/pull/29294)

These changes overlap architecture code used by Apex's boundary fusion, so they need source reconciliation and multi-GPU tests during a rebase.

### 5. Prompt-cache restore can affect perceived response speed

**#27991, August 31** batches contiguous runs while restoring scattered KV cells. Its motivating Qwen agentic workload used unified KV, two slots, and idle-slot caching; the author recorded a roughly 62-second restore before prompt processing. It is a restore-throughput fix, not a decode-kernel speedup. Apex's KV/context files retain the older implementation. Our one-slot deployment is a different case, so instrument cache-load time before assigning it the same penalty. [PR #27991](https://github.com/ggml-org/llama.cpp/pull/27991)

**#27530, September 26** cleans tensor data and deferred writes after a failed KV or recurrent-state restore, including hybrid attention/recurrent cleanup. It deliberately leaves successful restore and normal decode unchanged. This is a robustness addition to include in an upgraded baseline, with no ordinary-throughput gain claimed. [PR #27530](https://github.com/ggml-org/llama.cpp/pull/27530)

## What has not replaced our work

| Apex component | Current upstream finding | Consequence |
|---|---|---|
| Direct peer-VRAM AllReduce | #27825 enables the existing **host-staged** collective for HIP | Related capability, different transfer path; no basis to drop Apex P2P |
| Eight-block and fused reduction/residual/RMSNorm boundary | Apex gates and exact-shape implementation absent | A stock build loses these specializations |
| Phase 13 Q6_K accumulator conversion | Relevant upstream expression still lacks Apex's explicit F32 cast | Patch still distinct; remeasure its effect under the chosen compiler |
| Packed canonical N=2 projection kernels; shared-Q8/emulate8/FFN paths | Apex implementations absent | Stock MTP does not inherit their speed or parity properties |
| Canonical Q8 attention for verification rows 2–3 | Apex override absent | Stock N=3 may select TILE where Apex deliberately uses VEC |
| Natural recurrent split and speculative KV-padding guard | Apex-specific behavior absent | Revalidate their original failure cases before retaining or retiring them |
| Restored rocWMMA FA-1 | Not restored upstream; newer native D256 WMMA exists instead | Compare the two implementations independently |

The source audit found **19 Apex-modified files in the inspected core paths; 14 also changed upstream, and none of the 19 has identical final contents**. This measures file overlap, not the number of merge conflicts or proof that every local hunk remains necessary.

Upstream #27825's mixed RX 9070 + RX 6800 XT / Gemma4-31B Q6_K result was about +15.9% prefill and +2.2% decode over the former generic collective. Those percentages cannot be added to our existing P2P gains. The implementation explicitly stages through pinned host memory. [HIP AllReduce PR](https://github.com/ggml-org/llama.cpp/pull/27825)

## Other changes and applicability limits

| Change | Relevance to our configuration |
|---|---|
| #27870: fix divergent block barrier in F16 MMA attention | Correctness dependency to include when enabling the modern D256 path; not a demonstrated current Apex failure. [PR](https://github.com/ggml-org/llama.cpp/pull/27870) |
| #28079: select compiled KV quant pairs | Useful build control. Current defaults include Q8/Q8, Q4_0/Q4_0, F16/F16 and BF16/BF16. Missing vector instances can fall back to F16 conversion, so record compiled pairs during benchmarking. [PR](https://github.com/ggml-org/llama.cpp/pull/28079) |
| #28198: concurrent streams per device in multi-GPU graphs | Opt-in `GGML_CUDA_GRAPH_OPT=1`; published dense Qwen Q6_K result was neutral/slightly negative, unlike MoE. Different from the MTP graph-cache fix. [PR](https://github.com/ggml-org/llama.cpp/pull/28198) |
| #26079 and later MMVQ/MMQ crossovers | New dense-model thresholds target NVIDIA families; gfx1201 still uses the existing default crossover. They are not an R9700 Q6_K/MTP2 speedup. [PR](https://github.com/ggml-org/llama.cpp/pull/26079) |
| #26705: branchless unpack and prefetch | Unpack targets Q4_K/Q5_K; prefetch is gated to DGX Spark. No corresponding direct Q6_K/gfx1201 gain. [PR](https://github.com/ggml-org/llama.cpp/pull/26705) |
| Qwen4 sparse Flash Attention | Different model family; the examined sparse gather explicitly excludes HIP. Not sparse attention for our `qwen35` GGUF. [PR](https://github.com/ggml-org/llama.cpp/pull/28770) |
| #28576 and #28907: MFMA attention changes | CDNA-specific cases; R9700 uses RDNA WMMA. [FP32 accumulation](https://github.com/ggml-org/llama.cpp/pull/28576), [large heads](https://github.com/ggml-org/llama.cpp/pull/28907) |
| FP8 header/version changes | Do not add an integrated native-FP8 KV path for this model. Q8 integer KV remains distinct from FP8. [PR](https://github.com/ggml-org/llama.cpp/pull/29231) |
| #24669: extended batch API | Adds a compatibility layer but changes context/batch internals; relevant to integration, not a measured speed improvement. [PR](https://github.com/ggml-org/llama.cpp/pull/24669) |

The D256 quantized vector kernel still uses two cooperating threads per KQ dot. Its base-to-head changes are small and do not implement the depth-dependent quantized-KV redesign considered in the discovery report. A newer build is therefore not evidence that the long-context generation problem is solved.

## Recommended comparison builds

1. **A: frozen August 31 release.** Keep the exact executable/library hashes, Q6_K model, Q8 KV, tensor split 1:1, MTP2, batch settings, and all active gates. Its disabled restored FA is part of the measured baseline, not something to silently change.
2. **B: stock b11211 HIP, built locally for gfx1201.** This answers how current upstream performs. Use the same compiler/ROCm installation, model and settings as A; explicitly include Q8/Q8 kernels. Do not copy Apex flags and assume they work in stock code. Start each build with fresh cache state.
3. **C: Apex plus the MTP graph fix alone.** This isolates a plausible generation improvement without changing GDN arithmetic or collective implementation. Count graph captures, measure committed tokens/s, and check acceptance and exact same-build ordinary/MTP parity.
4. **D: isolated modern-attention candidate on Apex.** Bring in #28102 with its required correctness dependencies. Benchmark separately from restoring old FA-1; verify the actual selected kernel. Test prefill at 16K/64K/128K and make sure ordinary/MTP generation does not regress.
5. **Rebase candidate:** combine current correctness fixes, GDN correction plus fusion, and retained Apex components after individual attribution. Validate the new target numerics against a corrected reference, then ordinary/MTP agreement, deep-context continuation, cache restore, and two-GPU stability.

For A/B, report prompt-processing time, time to first token, sustained generation, cache-restore time, and MTP acceptance separately. Use repeated matched requests at short context and approximately 16K/64K/128K occupied depth. A full-upstream A/B identifies the better complete package; C/D determine which upstream changes can improve Apex without discarding its existing gains.

No binary performance ranking was claimed in this source audit. The subsequent [GPU comparison](four-build-benchmark.md) supplies those measurements and their validation limits.
