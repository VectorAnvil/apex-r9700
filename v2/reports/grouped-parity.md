> Historical research record. Statements about the selected service are as of this report, not current deployment status. See [the V2 index](../README.md). Local raw evidence, private runtime files and logs are not included; local artifact paths are provenance references, not downloadable links.

# Grouped attention accuracy investigation

2026-09-29. **Recommendation: CONTINUE TESTING. Original grouped remains frozen and unapproved.** The scoped source/ISA audit, numerical corrections, capability comparisons, budget diagnostics and clean throughput curve are complete. No tested correction is adopted.

The evidence supports a deterministic arithmetic difference that changes some close decisions. It does not establish a race, a broken checkpoint, an RDNA4 arithmetic defect, or a missing FP32 final merge. There is one confirmed additional wrong math answer with thinking disabled, plus a minor supporting factual error in one prose response. Those findings cannot be dismissed as harmless wording. Conversely, different tokens are not automatically accuracy failures: some alternatives improve recorded-token likelihood, and many changed reasoning paths finish correctly.

## What changed numerically

| Stage | Core | Original grouped |
|---|---|---|
| Q.K | Q8 query reconstruction and integer dot products, with half KQ storage for this RDNA4 path | Half Q/K operands, WMMA with FP32 result |
| Softmax state | Floating-point maxima and normalization state | FP32 maxima, exponentials and row sums; P converted to half before P.V |
| P.V | RDNA4 half2 accumulation and packed half arithmetic | Half-result/recurrent-accumulator WMMA and half running-output rescaling |
| KV decomposition | Vector attention decomposition | 48 fixed round-robin partitions, reusing each KV load across six query heads |
| Final combination | Global maximum followed by FP32 combination | Already FP32, fixed reverse-order online maximum/rescaled-sum combination |

Relevant sources are `ggml/src/ggml-cuda/fattn-vec.cuh`, `fattn-mma-f16.cuh` (`flash_attn_ext_f16_process_tile`, native-Q8 `flash_attn_ext_f16`), and `fattn-common.cuh` (`flash_attn_stream_k_fixup_uniform`, `flash_attn_combine_results`). The latest checked upstream uniform merge is text-identical to the frozen implementation. Upstream comparison (local evidence reference).

Core is **not** a full-FP32 attention reference. The compiled width-1/head-256/Q8-Q8 Core object contains 128 static packed F16 FMA instructions, 140 packed F16 multiplies and 64 integer Q8 dot instructions. Grouped changes the arithmetic decomposition; it does not uniquely introduce half accumulation. “Half accumulation” here describes stored/recurrent accumulator and result rounding, not every internal hardware product. Core ISA evidence (local evidence reference).

Grouping reuses KV loads across query heads; it does not average those heads into one attention distribution. Each head retains its own softmax and output. The original grouped scan uses wave32, 221 VGPRs and 34,816 bytes of LDS, with no scratch/spills. The original merge uses 11 VGPRs, global-max and compensated FP32 use 10, and FP64 merge uses 22; none spills. The occupancy API reports one active block per reported multiprocessor. These are resource observations, not measured CU utilization or memory bandwidth. Ordinary grouped decode launches 96 KV workgroups per GPU; the traced Core path launches 288. Grouped's gain includes shared KV loading/unpacking across six heads, not simply more workgroups.

Each partition scans 64-row KV tiles: roughly 11 tiles per partition at 32K and 85–86 near 262K. The final merge still combines 48 results. Therefore growing context length increases the within-partition work, not the number of final merge terms in this configuration. Both decomposition and operand rounding are plausible sources of the observed differences; their relative contribution is not yet isolated.

For partial state `(m_i, d_i, o_i)`, stable combination is:

`M = max(m_i); D = sum(exp(m_i-M)*d_i); O = sum(exp(m_i-M)*o_i)/D`.

The original online recurrence is algebraically equivalent but uses a different floating-point order. Its order is fixed; there is no arrival-order atomic sum. New variants test global-max ascending FP32, compensated FP32, and FP64 final merging of the same float partials. They retain the -20 softmax cutoff, applied relative to the final maximum rather than successive running maxima. This tests ordering, rounding and cutoff placement together. FP64 merging cannot restore information already rounded inside the scan.

The model reports architecture `qwen35` and includes recurrent attention as well as ordinary KV attention. Identical forced tokens do not guarantee identical hidden state after a changed attention operation. Q8 cache writes (`llama-kv-cache.cpp`, `set-rows.cu`, `cpy-utils.cuh::quantize_f32_q8_0_block`) can amplify differences at quantization thresholds. This is a source-supported propagation mechanism, **not a measured attribution** of the present error to KV writes.

## External implementations and applicability

Pinned commits, source receipts and a fuller matrix are in EXTERNAL_LEADS.md (local evidence reference).

| Source | Technique | Local conclusion |
|---|---|---|
| [FlashAttention split-KV](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h) | Stable reduction of partial log-sum-exp values, then weighted partial-output merge | Motivates our common-maximum merge diagnostic; CUDA scheduling is not imported |
| [FlashInfer state](https://github.com/flashinfer-ai/flashinfer/blob/4a6381331a58ca4e900b0455cd55088fb6ed2ecb/include/flashinfer/attention/state.cuh) | FP32 output/max/denominator; common-max rescaling and empty-state guards | Original already uses this family of stable online recurrence |
| [AITER decode](https://github.com/ROCm/aiter/blob/475cf0f607d71010a02288094f20e97254d7d2d2/aiter/ops/triton/_triton_kernels/attention/pa_decode.py), [Composable Kernel combine](https://github.com/ROCm/composable_kernel/blob/2b053c6c5e60efc86cc62b12a9ca490ee2d5ebcf/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline.hpp), [vLLM decode](https://github.com/vllm-project/vllm/blob/af5b4857e1353c01fd6bf41bc3cb9f84dc82dd89/vllm/v1/attention/ops/triton_decode_attention.py) | Separate partition scan and hierarchical/max-shifted reduction of partial states | Relevant reduction references; CDNA/Triton results are not gfx1201 measurements |
| [SGLang decode](https://github.com/sgl-project/sglang/blob/f731e82f09be137cfc5001c732f044dd44740c1d/python/sglang/kernels/ops/attention/decode_attention.py) | Current gfx1250 branch avoids P-to-BF16 conversion, citing reasoning loss | P operand precision can matter separately from accumulator precision; different GPU and BF16 format from our F16 path |
| [Dual-R9700 fork](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference/blob/54520fe72ed8b80e829c382f2361df2f2499e9a8/patches/087-rdna4-flash-decode-bf16-pv.patch) | BF16 P.V operands, FP32 accumulation; another patch raises KV splits | Our operands are already low precision; their different model and needle tests do not qualify our implementation |
| [RDNA4 fork issue 45](https://github.com/stew675/llama-cpp-rdna-boosts/issues/45) | Fixed round-robin partitioning prevents moving contiguous boundaries between decode/verify | Our native-Q8 grouped traversal already uses round-robin; no second rewrite is warranted |
| [FlashInfer fixed split size](https://docs.flashinfer.ai/api/attention.html), [batch invariance analysis](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Stabilize arithmetic decomposition across dispatch configurations | Repeatable execution is different from equality across algorithms; neither proves accuracy |
| [FlashInfer PR 4993](https://github.com/flashinfer-ai/flashinfer/pull/4993) | Open September 29: narrows an NVFP4 split-KV disabling workaround for symmetric head shapes | Supports preserving parallelism, but is a CUDA/NVFP4 performance lead, not an AMD accuracy patch |
| [Softmax stability analysis](https://arxiv.org/abs/1909.03469), [online normalizer](https://arxiv.org/abs/1805.02867) | Analyze shifted/log-sum-exp formulas and online normalization | Algebraically equivalent formulas need not have equal finite-precision error |
| [FoldAttention](https://arxiv.org/abs/2609.33410) | Fixed-reference additive partials rather than repeated rescaling | Larger Hopper-based research lead; only an all-key, range-checked prototype would be appropriate initially |

ROCm, rocWMMA, hipBLASLt, LLVM/AMDGPU, Triton and relevant llama.cpp issues were also checked. The examined hipBLASLt and large-M GEMM reports do not target our changed intrinsic-based decode attention kernel. No compiler or RDNA4 numerical defect was established. Full issue links and limitations are in the external matrix. No external dependency or system library was installed or replaced.

## Original: expanded conditional next-token testing

512 predictions at each of six depths, 3,072 total, using the same model, Q8 KV, both GPUs, fixed 512-token prefill and one-row forced-token decode. Full-vocabulary Core references are retained. The first prediction at each depth comes from common prefill; 8K has grouped dispatch disabled and serves as an exact control.

| Context | Changed choices /512 | Conditional PPL change vs Core | Core / grouped recorded-token top matches |
|---:|---:|---:|---:|
| 8,192 | 0 | 0.0000% | 376 / 376 |
| 32,768 | 6 | -0.0356% | 256 / 256 |
| 65,536 | 2 | -0.0135% | 342 / 343 |
| 131,072 | 9 | -0.0721% | 341 / 342 |
| 196,608 | 10 | +0.1078% | 308 / 309 |
| 261,632 | 5 | -0.0749% | 332 / 331 |

There are 32 changed choices: eight lost recorded-token top matches and ten gains. None overturns a Core choice with probability at least 0.9 among 839 active-grouped predictions meeting that threshold. The highest Core top probability among disagreements is 0.53972. Of 32 changes, 31 have Core top-two logit margin below 0.1, 26 below 0.05, and 11 below 0.01. Some changed words carry meaning; low margin does not make them harmless. Every disagreement, both choices, logits and probabilities (local evidence reference).

No monotonic growth in argmax changes is observed across the active depths: 6, 2, 9, 10, 5. Mean logit RMSE is about 0.0339 at 32K, 0.0426 at 128K and 0.0415 near full. Near-full first/last 64-step RMSE is 0.0426/0.0424; this does not show an accelerating 512-step collapse. However, continuation content differs across depths and windows, so this is not a causal depth-isolation experiment. One WikiText corpus and nested prefixes cannot establish general capability parity. Descriptive block-bootstrap intervals cross zero at every active original depth; they are not an equivalence test.

## Original: capability and deterministic rollout testing

The screen uses exact tool/argument checks, absent-record abstention, dated updates, early/middle/late needles, relational joins, JSON, arithmetic/logic, executable Python cases and longer JSON/code/prose rollouts. It includes 12 fixed held-out GSM8K questions, adapted to numeric-only output, not an official GSM8K score. Dataset index 749 has contradictory labeling identified before testing; its literal answer is reviewed separately. Two end-history repeats check cached-request stability.

Thinking OFF completed 432 requests, 216 per lane:

| Context | Core automatic pass /34 | Grouped automatic pass /34 | Additional grouped loss |
|---:|---:|---:|---:|
| 8,192 | 20 | 20 | 0 |
| 32,768 | 20 | 20 | 0 |
| 65,536 | 21 | 21 | 0 |
| 131,072 | 21 | 20 | 1 |
| 196,608 | 21 | 21 | 0 |
| 260,096 | 21 | 21 | 0 |

Core passes 124/204 automatic checks, grouped 123/204: 80 shared failures, one additional grouped failure, no gains. Dedicated tool requests pass 30/30 per lane, retrieval 24/24 and code 24/24. Most shared failures are thinking-disabled math and strict JSON behavior; some JSON requests cause the same unsolicited tool call in both lanes. Those shared failures are retained, not hidden. No canonical tool call differs between lanes.

**Confirmed loss:** at 128K, `a = 2*(a-5)` gives current age 10 and age **13** three years later. Core returns 13; grouped returns **16**. The first digit/probabilities are identical. At the next digit, Core assigns 0.31508 to `3` and 0.30773 to `6`; grouped assigns 0.30754 to `3` and 0.31109 to `6`. Core's margin is 0.02361 logits. This is a real wrong answer at a close decision. It occurs at the second generated digit after an identical first token/probability record, so a long divergent autoregressive history is not required to trigger it. Both answer correctly at all other tested depths. The context-dependent margin is smallest at this failing depth. Per-depth decision evidence (local evidence reference).

Five prose texts differ. Required IDs and the copper/amber/R-42/node-608 chain remain correct, but grouped at 64K incorrectly says the three separately archived links come from two entries. Both lanes exceed the requested 150–220-word limit in every thinking-OFF prose case. The changed 64K code has an identical executable AST after removing its docstring, and both sets of examples are correct. Full manual review (local evidence reference).

Thinking ON completed another 144 requests:

| Context | Core / grouped automatic pass | Grouped budget loss | Grouped budget gain | Regressions where both finish |
|---:|---:|---|---|---:|
| 131,072 | 33/34 / 33/34 | Newspaper math reaches 768-token cap | Ledger code finishes before 1,024-token cap | 0/32 pairs |
| 260,096 | 33/34 / 33/34 | Same math reaches cap | Same code completes | 0/32 pairs |

Both correctly solve the age problem with thinking enabled. All changed completed reasoning passages were reviewed: no additional completed-answer defect was found. One Core logic passage makes and corrects an intermediate negation mistake; grouped reaches the same correct conclusion more directly. Both prose responses at each depth truncate at 512 tokens without a final answer. Budget losses and gains remain genuine task-completion outcomes; a larger-budget replay cannot erase them. Paired results (local evidence reference), semantic review (local evidence reference).

Across OFF and ON, 54,173 generated positions were recorded across both lanes. Only 19,667 decisions share the same generated prefix through the first divergence; after that, trajectories are not paired next-token tests. No first divergence overturns a Core choice with probability at least 0.9 in those shared-prefix decisions. Repeated tasks and context depths are correlated; these are not independent accuracy samples or a statistical noninferiority result.

All end-history repeats reproduce their lane's content, calls and reasoning. Five targeted cold/cache diagnostics reproduce Core content and recorded probabilities exactly, including the age problem and baseline JSON/math failures. The additional original-grouped cold replay also returns 16 with all recorded probabilities exactly matching its earlier cached failure, while Core cold returns 13. This specific grouped regression is therefore not explained by checkpoint restoration. Cold diagnostic (local evidence reference).

## Correction experiments

The original scan is unchanged. New merge modes are isolated in a separate worktree/build. Mode zero reproduces every recorded 32K cross-distribution metric exactly. A first build failed mandatory dispatch verification because the linker selected an unchanged weak host launcher; its candidate results were rejected. A unique experimental host-launcher name fixed selection in build r2. Actual mode dispatch on both GPUs is verified.

All 270 unchanged-tolerance CPU-reference operator cases pass: nine configurations, five depths, widths 1/2/3, on both GPUs. Timing order is forward and reverse per card, with no concurrent compilation. These are **operator latencies, not complete generation speeds**.

| Variant | Near-262K width-2 latency | Cost vs original | 32K changed choices vs Core | 32K recorded-token top matches /512 | 32K PPL change vs Core |
|---|---:|---:|---:|---:|---:|
| Original 48 partitions | 612.65 us | 0% | 6 | 256 | -0.0356% |
| Rebuilt mode zero | 615.09 us | +0.40% | 6 | 256 | -0.0356% |
| Global-max FP32 | 619.92 us | +1.19% | 6 | 258 | -0.0215% |
| Compensated FP32 | 623.11 us | +1.71% | 7 | 256 | +0.1744% |
| FP64 final merge | 659.96 us | +7.72% | 5 | 254 | +0.0162% |
| 16 partitions | 1,395.34 us | +127.75% | 5 | 256 | +0.1957% |
| 32 partitions | 777.04 us | +26.83% | 6 | 258 | -0.0913% |
| 64 partitions | 801.16 us | +30.77% | 7 | 257 | +0.0488% |
| 96 partitions | 656.83 us | +7.21% | 6 | 257 | -0.0633% |

Original 48 partitions is fastest in the pooled width-2 operator timings at every measured depth. The rebuilt control differs by 0.40% near full, which gives context for the small global-max FP32 overhead. The 96-partition width-2 cost is approximately 40%, 16%, 12%, 10%, 7% at 16K, 64K, 128K, 192K, 262K. Global-max/compensated FP32 cost roughly 8–11% at 16K but 1–2% near full. FP64 merge costs about 87% at 16K and 8% near full. This rules out treating any as a free whole-curve replacement. Complete operator curve (local evidence reference).

At 32K, all merge corrections reduce Core logit RMSE by only about 1%, while quality indicators remain mixed. Compensated summation worsens likelihood despite smaller numerical distance. FP64 has fewer disagreements but loses two recorded-token top matches relative to original. Partition geometry also has no monotonic quality trend. No correction is selected from these results. 32K model results (local evidence reference).

### Near-full correction results

Each candidate uses the same 512 forced continuation tokens at 261,632 context. Original grouped changes five choices, matches 331 recorded tokens and has PPL change -0.0749% versus Core; Core matches 332.

| Variant | Changed choices vs Core | Recorded-token top matches /512 | PPL change vs Core | PPL change vs original |
|---|---:|---:|---:|---:|
| global-f32 | 3 | 332 | +0.0732% | +0.1482% |
| merge-f64 | 2 | 332 | +0.0718% | +0.1468% |
| splits-96 | 5 | 332 | +0.1085% | +0.1835% |

All three gain one recorded-token top match over original but worsen conditional likelihood on this continuation. No candidate changes a Core top choice with probability at least 0.9 among 191 such predictions. The descriptive block intervals all cross zero. This is mixed evidence, not a correction qualified for adoption.

Near-full logit RMSE is 0.04467 for global-max FP32, 0.04062 for FP64 merge, and 0.04130 for 96 partitions, versus original 0.04147. Global-max FP32 therefore does not consistently reduce even raw numerical distance across depths. At its largest element-error position (341), maximum logit difference is 7.22 and RMSE 1.305, yet KL is only 0.0000343 and the top choice is unchanged. Core and candidate assign 99.9188% and 99.9030% respectively to that same top token. Raw logit distance can mislead when probabilities saturate and is not invariant to common offsets. Probability-space and objective task evidence remain necessary.

Deep summaries (local evidence reference); adjacent confidence-analysis files contain both distributions for every disagreement.

### Known-loss and expanded-budget diagnostics

This post-hoc replay uses the captured 128K age failure with thinking OFF, then thinking-ON code/prose/newspaper math with a 4,096-token cap. It is not held-out qualification. Token counts include reasoning.

| Configuration | Cold age answer (correct: 13) | Code tokens | Prose tokens | Newspaper-math tokens (answer: 8) |
|---|---:|---:|---:|---:|
| core | 13 | 1826 | 3130 | 720 |
| original | 16 | 1021 | 3066 | 804 |
| global-f32 | 16 | 1447 | 3133 | 1332 |
| splits-96 | 16 | 1257 | 3066 | 534 |

Neither global-max FP32 nor 96 partitions fixes the known age loss. Original grouped cold also exactly reproduces its prior cached probabilities. Every variant shares the same first prefill probabilities on this case.

All enlarged code and math responses finish correctly. All final prose answers are exactly identical, 188 words, with correct identifiers, three archive links and missing-record abstention. The executable code ASTs are identical after removing docstrings; examples are correct. Core and original grouped reproduce every recorded token/probability from their earlier capped responses before continuing. These are genuine budget extensions, not changed prefixes. The original 768-token math loss resolves at 804 tokens; the prior capped outcome is retained.

Reasoning length is itself variable: global-max FP32 takes 1,332 tokens on the same math problem versus original 804, while 96 partitions takes 534. These selected examples do not establish a general latency ranking. Kernel tok/s and tokens needed to finish a task must both be evaluated. No production speed claim is taken from these instrumented runs.

Results and first differing decisions (local evidence reference), manual review (local evidence reference), Core prefix check (local evidence reference), original prefix check (local evidence reference).

## Clean throughput curve

The historical matched-MTP result is Core 24.68 versus original grouped 40.93 tok/s near 262K, a 65.9% gain. It is not reclassified as a new measurement. Probability-instrumented capability runs above are not production speed benchmarks.

The fresh run uses both GPUs, matched MTP depth 1, context capacity 262,144, checkpoint correction ON, prompt grid 512, and the frozen binary with grouped OFF/ON. Tracing and token-probability collection are OFF. Each point is the median of two 128-token cached-repeat generations after prefix growth. Every repeat reproduces its lane’s first output tokens exactly. Core runs first, then grouped; this is a controlled screen, not a broad statistical performance study.

| Context | Core tok/s | Original grouped tok/s | Gain |
|---:|---:|---:|---:|
| 8,192 | 52.53 | 52.33 | -0.4% |
| 32,768 | 46.89 | 52.08 | +11.1% |
| 65,536 | 41.25 | 50.59 | +22.6% |
| 131,072 | 31.96 | 44.58 | +39.5% |
| 196,608 | 28.04 | 42.62 | +52.0% |
| 261,632 | 23.56 | 38.04 | +61.5% |

The 8K slice has grouped dispatch disabled and differs by -0.38%. Original grouped improves every active point. From 8K to near 262K, Core loses about 55.2% of its throughput; grouped loses about 27.3%. The current near-full result is +61.5%, rather than substituting the historical +65.9% figure. The original kernel is unchanged; the new number describes this fixture/profile/run.

The emitted 128-token sequences match between lanes at every depth except 64K, where the wording differs. Draft acceptance is also identical at 8K, 32K, 128K and 192K. At 64K it is Core 59/68 versus grouped 60/67; near full it is Core 57/69 versus grouped 56/70, while output tokens still match. Thus matching MTP depth does not guarantee matching acceptance. The deep gain is present despite slightly lower grouped acceptance. Paired outputs, repeat timings and acceptance (local evidence reference).

**Prompt processing below is measured growing-prefix extension throughput, not fresh full-prompt PP** (except the initial 8K point). The request reuses the previous prefix and processes the newly added span.

| Destination context | Core extension tok/s | Grouped extension tok/s |
|---:|---:|---:|
| 8,192 | 1170.74 | 1170.67 |
| 32,768 | 1074.90 | 1067.92 |
| 65,536 | 914.01 | 911.20 |
| 131,072 | 732.78 | 731.79 |
| 196,608 | 574.34 | 574.37 |
| 261,632 | 474.23 | 474.39 |

Observed extension-rate differences stay between about -0.65% and +0.03%; this run shows no meaningful prompt-processing gain from grouped decode. Prompt-growth TTFT and full timing records are retained. No production performance claim is taken from the earlier probability-instrumented capability runs.

Speed summary (local evidence reference), deltas (local evidence reference), measurement completion (local evidence reference).

## Ranked next steps and decision

**No measured correction is adopted. Original grouped remains a CONTINUE TESTING candidate.** Its performance merits further work, but the observed objective loss, supporting factual error and budget-sensitive trajectories prevent a practical-parity claim.

| Priority | Strategy | Accuracy benefit versus cost | Minimal next discriminator / expected signal |
|---|---|---|---|
| 1 | Capture and replay real Q/K/V, softmax state and partial outputs against a higher-precision reference | Diagnostic, not a production correction; no steady-state cost when capture is disabled | Capture the known age failure at its first changed decode step at 128K, plus matched steps at 32K and near full context. Separate score/P rounding, within-partition numerator error, and final-merge error. A local reference is needed before claiming which stage dominates |
| 2 | If the scan is implicated, short half-WMMA chunks periodically combined in FP32 | Potentially retains fast matrix operations; benefit and VGPR/LDS cost unknown. Materially different from the rejected full-FP32 P.V kernel | First prototype a bounded chunk cadence on captured inputs. Advance only if it reduces reference error without losing the long-context operator gain, then test independent tasks |
| 3 | Global-max FP32 merge | Measured near-full operator cost about 1.2%; no consistent likelihood benefit, and raw numerical distance worsens at deep context | It also fails the known cold age case. Current results do not warrant adoption. A captured-partial comparison must show a meaningful merge defect before adding more merge variants |
| 4 | Context-dependent geometry, particularly 96 partitions only at deep context | About 7% operator cost near full, larger at shallower depths; observed likelihood is mixed across depth | Same continuation at all depths plus independent objective tasks. A real Pareto improvement must survive both quality measures and whole-model timing; current evidence does not, and 96 partitions still fails the known cold age case |
| 5 | Compensated FP32 or FP64 final merge | No consistent quality benefit; FP64 adds about 8% near-full and 87% short-context operator latency | Keep as diagnostics, reject these implementations as default replacements |
| 6 | Different P/Q/K/V precision or fixed-reference additive attention | Source-supported possibilities, not measured fixes. SGLang's BF16 finding cannot be copied directly into our F16 kernel | Measure actual underflow/rounding on captured tensors first. Any all-key redesign needs finite-range validation; approximate pruning is outside this parity correction |

A fixed-continuation context experiment is still needed to isolate depth causally: reuse identical target tokens and task evidence while changing only the unrelated prefix length. The current corpus slices change continuation content with depth. More partitions also change which tiles share a running maximum, so a geometry response alone cannot identify half-accumulation error.

For objective qualification, preserve the original candidate and compare complete trajectories, reasoning budgets and wall time by task family. Do not use the known age failure to choose a production kernel: it is a post-hoc diagnostic. A gain in code completion cannot establish noninferiority in math, and matching tools in this screen cannot qualify all Vivi tool workflows.


The prior Core-query-quantization reconstruction and full-FP32 P.V experiments were not repeated. Query reconstruction previously reduced numerical divergence but worsened likelihood; full-FP32 P.V cost 58–80%, and removing spills did not recover it. FP64 **final merge only** in this round is a distinct, much smaller diagnostic.

A practical-parity claim requires independent tasks, a declared acceptable degradation margin, paired domain-level evaluation and sufficient coverage. This round's synthetic archive, twelve math probes, repeated depths and one corpus do not provide that. [LongBench v2](https://github.com/THUDM/LongBench), [RULER](https://github.com/NVIDIA/RULER) and [BFCL](https://sky.cs.berkeley.edu/project/berkeley-function-calling-leaderboard/) are relevant next qualification sources; none was run or scored here. Expanded MTP2 capability qualification was not performed in this round; earlier tool checks are historical evidence only.

## Isolation and reproducibility

Core and original grouped use the frozen binary with grouped dispatch OFF/ON respectively. FROZEN_CANDIDATE.json (local evidence reference) identifies 3,732 tracked source files and 11 real binaries/libraries. Original source commit: `8608915dadcbc044a7bacda8089222f49fe0422a`; Core baseline: `bccaecf439ed0a0d4918317afd9996255a1f1e17`.

Kernel experiments are in `../20260928-long-context/grouped-merge-20260929/src`, separate build `build-merge-r2/bin`. Kernel commit `2c959d0` adds merge variants; `6da9c70` fixes experimental launcher linkage. Validation/runtime geometry work is in the separate `grouped-geometry-20260929/src` worktree. No production source, binary, driver, ROCm installation, system library, configuration or Vivi service was modified.

All A/B comparisons keep Core's established Q6_K, packed-attention, recurrent/checkpoint and communication profile fixed. Direct-P2P and fused-boundary gates remain disabled in that profile. This round does not claim tested stacking with a different communication configuration.

All scoped GPU phases completed and restored production. Final restoration was verified at 2026-09-29T09:10:30.575254-05:00: the existing August production server is healthy, returns READY, its slots are idle, the experimental listener is absent, production binary/library hashes match, and the unrelated Bonsai process is unchanged. Final restoration receipt (local evidence reference).

Frozen integrity verification (local evidence reference) confirms all 3,732 source hashes and 11 binary/library hashes match, and the baseline, original, merge and geometry worktrees are clean. No grouped candidate was promoted and no usage reset was performed.
