> Historical research record. Statements about the selected service are as of this report, not current deployment status. See [the V2 index](../README.md). Local raw evidence, private runtime files and logs are not included; local artifact paths are provenance references, not downloadable links.

# August 31 Apex versus upstream b11211

September 27, 2026. Completed: two timing passes per build, focused operator/model checks, and separate GPU traces. The original August service is restored and running.

## Findings

**The modern D256 attention backport gives a repeatable long-input improvement. The isolated MTP graph fix is speed-neutral here. Stock upstream is faster on these deep-context requests but fails the short exact ordinary/MTP parity gate. Expanded 128K checks also expose parity gaps in both August and the attention backport. No candidate has been promoted.**

- **Attention backport:** 128K fresh prefill falls from **275.23 to 180.91 seconds**, a **34.3% time reduction** (about 52% higher input throughput). At 64K the reduction is **23.2%**. Short generation stays essentially unchanged at **61.91 versus 61.81 tokens/s**.
- **Important attention tradeoff:** the 128K cached continuation generates at **28.48 versus 35.79 tokens/s**, a **20.4% decrease**, alongside acceptance falling from **78/96 (81.3%) to 66/121 (54.5%)**. This repeats across both process starts. The generated prefix and subsequent output differ between these builds, so this is a measured serving-workload outcome, not an isolated decode-kernel comparison.
- **MTP graph fix:** **61.88 tokens/s** short and **34.91 tokens/s** after 128K prefill, effectively tied with the original. It preserves all **48/48** first-pass MTP outputs compared with August, including deep requests and continuations, plus **12/12** ordinary outputs.
- **Stock b11211:** **179.61 seconds** for 128K prefill and **39.20 tokens/s** for its first response, versus August's **275.23 seconds and 34.90 tokens/s**. Those are whole-package results with changed output sequences. Stock averages **60.45 tokens/s** short and matches ordinary decoding in only **9/12** parity cases.

The timing matrix completed **12 independent server runs and 396 requests**, with no context truncation. Every build reproduced all **36/36** short/deep/continuation output sequences across its two MTP runs. All eight MTP processes received identical token IDs for each fresh deep prompt.

!Four-build performance comparison (local evidence reference)

Points are two-run means; error bars show the observed two-run range, not confidence intervals. Fresh-prefill and continuation generation use different prompts. The original and MTP-only curves largely overlap.

## Short generation

Means across 30 fixed prompts per process, two independent processes per build. Each request generates 128 tokens. TPS is the server generation metric, excluding prefill.

| Build | Requests | Mean tokens/s | Individual run means | Accepted / drafted |
|---|---:|---:|---|---:|
| August 31 Apex | 60 | 61.812 | 61.737, 61.888 | 4430 / 6298 |
| Stock b11211 | 60 | 60.452 | 60.031, 60.873 | 4422 / 6318 |
| Apex + MTP fix | 60 | 61.882 | 61.834, 61.930 | 4430 / 6298 |
| Apex + attention | 60 | 61.911 | 61.777, 62.045 | 4438 / 6274 |

## Populated-context measurements

Two-run means. Actual fresh input lengths are 16,395 / 65,545 / 131,085 tokens. The cached continuation processes 458 additional tokens after retaining the previous prompt and output. The continuation prefix can differ between builds because their generated output can differ.

| Build | Nominal depth | Fresh prefill (s) | Fresh TTFT (s) | Generation (t/s) | Cached new-input processing (s) | Continuation generation (t/s) |
|---|---:|---:|---:|---:|---:|---:|
| August 31 Apex | 16K | 18.55 | 18.82 | 54.51 | 0.94 | 55.58 |
| August 31 Apex | 64K | 100.53 | 100.54 | 46.23 | 1.53 | 39.43 |
| August 31 Apex | 128K | 275.23 | 275.24 | 34.90 | 2.28 | 35.79 |
| Stock b11211 | 16K | 17.58 | 17.84 | 57.39 | 0.87 | 55.14 |
| Stock b11211 | 64K | 78.24 | 78.25 | 47.05 | 1.21 | 47.30 |
| Stock b11211 | 128K | 179.61 | 179.63 | 39.20 | 1.63 | 40.78 |
| Apex + MTP fix | 16K | 18.59 | 18.86 | 54.67 | 0.94 | 55.70 |
| Apex + MTP fix | 64K | 100.26 | 100.27 | 46.28 | 1.54 | 39.67 |
| Apex + MTP fix | 128K | 275.11 | 275.13 | 34.91 | 2.31 | 35.56 |
| Apex + attention | 16K | 17.11 | 17.37 | 58.17 | 0.86 | 54.51 |
| Apex + attention | 64K | 77.19 | 77.20 | 45.33 | 1.21 | 38.59 |
| Apex + attention | 128K | 180.91 | 180.93 | 35.58 | 1.70 | 28.48 |

## Correctness observations

| Build | Ordinary vs MTP, 12 cases | MTP repeat output equality, 30 short + 6 deep/continuation |
|---|---:|---:|
| August 31 Apex | 12/12 | 36/36 |
| Stock b11211 | 9/12 | 36/36 |
| Apex + MTP fix | 12/12 | 36/36 |
| Apex + attention | 12/12 | 36/36 |

## Interpreting the results

The short-stock mean includes one unusually slow request: performance prompt 14 took 3.042 seconds of generation in the first process versus 1.933 seconds in the repeat, with identical output and acceptance. The approximately 1.109-second extra delay is unexplained and has not been discarded. Stock's full run means are 60.03 and 60.87 tokens/s; August's are 61.74 and 61.89. Avoid treating the pooled short-speed difference as a precise general slowdown for every workload.

Stock's ordinary/MTP mismatches occur in `prose`, `list`, and `forced256`, beginning at zero-based output token indices 17, 89, and 175. These comparisons are **within stock**, so they are separate from upstream's legitimate GDN normalization change relative to August. They establish failure of Apex's exact parity contract, not a semantic quality verdict on all stock output.

The attention candidate matches August ordinary output in **10/12** cases and matches **27/48** first-pass MTP requests overall. It passes its own **12/12** ordinary/MTP suite. Modern native WMMA becomes eligible at nine query rows for this GQA geometry; the inherited `GGML_CUDA_FA1_MIN_NQ=32` governs the older restored path and does not impose that threshold on the new native branch. Thus this backport can change short-prompt arithmetic as well as long-prefill arithmetic.

These measurements do not quantify the effect of turning the older restored rocWMMA implementation back on: that implementation remains compiled OFF. Ordinary Flash Attention was not globally disabled in August; its vector/tile paths were active. The measured attention gain is specifically from adding the newer native D256 route.

## Diagnostics

The corrected operator filter selected **13 cases per GPU**, and **13/13 passed on both R9700s**: the six upstream F16 cases, six matching Q8 cases, and one existing matching F16 case. The backend suite compares against CPU with its `5e-4` maximum NMSE threshold. These focused shapes are not a comprehensive model-quality evaluation.

The focused attention-candidate model test matches ordinary decoding for the first 128K response, but the cached continuation differs at zero-based output token **47**. Both runs reused 131,212 tokens and processed the same 458 new input tokens. The added original-build control also finds a parity gap, on the fresh response rather than the continuation:

| Build | Fresh 131,085-token input | Continuation on the same full input as its MTP reference |
|---|---|---|
| August 31 | **Different from ordinary**, first difference at token 31 | Exact output match |
| Apex + attention | Exact output match | **Different from ordinary**, first difference at token 47 |

Each row uses one additional ordinary process and the saved MTP reference, whose outputs reproduced across two MTP processes. The August continuation explicitly uses the MTP-generated prefix, because its ordinary first response differs. It consequently restores/reuses **131,081** tokens and processes **589**, versus **131,212 cached + 458 processed** for MTP. The attention pair both use **131,212 + 458**. Thus the continuation controls have identical full token inputs within each build, but the August cache-processing boundary differs. This is not an isolated proof of which attention or recurrent operation caused the mismatches.

**Neither August nor the attention candidate passes both expanded deep parity cases.** The old short suite and historical deep prompts do not establish exactness for all fresh and cached requests. The attention candidate's lower continuation acceptance cannot be dismissed as merely benign content variation now that its paired ordinary output also differs. A quality verdict or root-cause attribution would require further work.

A separate HIP API trace ran a warmup and the same three 128-token prompts on August and the MTP-only candidate. Both recorded exactly **560 `hipStreamBeginCapture`, 560 `hipStreamEndCapture`, 540 `hipGraphInstantiate`, 560 `hipGraphExecUpdate`, and 40,588 `hipGraphLaunch` calls**. Thus the isolated fix did not reduce capture activity in this bounded probe, consistent with the timing results. This does not establish that the upstream fix is ineffective on other models, speculation widths, or backends.

The attention candidate's separate kernel trace recorded **272 calls to `flash_attn_ext_f16<256,256,32,2,...>`**, the modern D256 matrix-attention route compiled for RDNA WMMA, and **128 Q8 vector-attention calls**. The restored rocWMMA implementation remains compiled OFF. HIP graphs were disabled only for this kernel-symbol trace; they remained enabled for all primary timing runs and the HIP API comparison.

The profiler finalized its CSV output on SIGTERM, then its chained signal handling left the instrumented process waiting until the runner's 60-second cleanup timeout. The runner cleaned up only its own profiling processes. All traced requests and CSV finalization completed first. Instrumented timings and shutdown behavior are excluded from the performance tables.

## Builds and scope

| Label | Source | Purpose |
|---|---|---|
| August 31 | Original hash-checked fused-boundary release | Fresh measurement of the deployed appliance |
| Stock | llama.cpp b11211, `d7fb90e8e2494b2908934d956a3202fd60152ee0` | Complete upstream package comparison |
| Apex + MTP | August source plus [upstream #28549](https://github.com/ggml-org/llama.cpp/pull/28549) | Isolate graph-result storage change |
| Apex + attention | August source plus [#27870](https://github.com/ggml-org/llama.cpp/pull/27870) and [#28102](https://github.com/ggml-org/llama.cpp/pull/28102) | Isolate modern D256 WMMA attention and its barrier correctness dependency |

The August source is the custom fork based on upstream `4695f001fece1660d8bb1b3748f50726ddcc100b` (build 10457, August 17). The two Apex candidates are separate experiments. Neither includes the other's patch. The original release and source remain unchanged. This is a comparison, not production promotion or a completed rebase.

All new builds use ROCm 7.2.0, Release optimization, HIP graphs, no VMM, no RCCL, and `gfx1201`. The original release used the same ROCm/compiler installation but contains additional GPU architecture targets. Restored rocWMMA FA remains compiled OFF in both isolated Apex candidates. The attention candidate enables the newer native WMMA route through its source changes; it does not reactivate the older FA-1 implementation.

Stock explicitly compiles Q8/Q8, Q4_0/Q4_0, F16/F16 and BF16/BF16 attention pairs. It receives no Apex runtime gates. Apex candidates retain every gate from the August launcher. The original executable's generated version string is not reliable source provenance: it inherited the enclosing repository's build metadata. The source audit, exact binary hashes and patch manifests identify the releases.

## Measurement contract

- Two Radeon AI PRO R9700 GPUs, tensor split 1:1. The separate Bonsai service on GPU2 remains running.
- Same Qwen3.8 27B Heretic Q6_K GGUF and BF16 multimodal projector; Q8_0 K/V; context capacity 262144; one slot; 12 threads; batch 2048; microbatch 512.
- MTP maximum two, minimum zero; greedy sampling, fixed seeds. Independent ordinary-decoding processes check the historical 12-case parity suite.
- Thirty historical short prompts, each generating 128 tokens with EOS ignored. Two independent MTP process starts per build; second pass reverses build order, ending with the original release as a control.
- Real populated prompts near 16K, 64K and 128K, using the same repeated sentence and suffix. These synthetic prompts exercise context cost; they are not a long-context quality evaluation.
- Fresh prefill uses `cache_prompt=false`. Each deep request is followed by a request containing its exact token prefix and generated output plus additional input, using `cache_prompt=true`. Actual processed/cached token counts are retained.
- Streaming records client time to first output and total wall time. Server timings separately record input processing, generation, drafted tokens and accepted tokens. Generation speed excludes prompt processing.
- Compilation finishes before GPU timing. No profiler is attached to the timing matrix. Diagnostic traces are separate and their speeds are excluded.

Raw runner, logs, token sequences and incremental results: `results_raw/benchmark-20260927/`. The runner restores the original hash-checked release in a `finally` block.

## Correctness boundaries

Within-build ordinary/MTP token equality is distinct from cross-version equality. Upstream includes the GDN normalization correction, while both isolated Apex candidates retain the original arithmetic. Stock output changes therefore require interpretation rather than automatic classification as regressions.

The attention backport includes the six upstream D256 F16 correctness cases plus six matching Q8 cases. Its first operator invocation accidentally selected zero cases because the filter omitted decimal formatting in floating-point fields. That invocation is not accepted as a pass; corrected, nonempty operator runs are required and recorded separately.

The timing matrix and added deep checks do not establish full production promotion gates, arbitrary cache eviction/restore behavior, multimodal quality, or a corrected-model quality reference for stock upstream. The ordinary deep checks are limited to the two explicit 128K cases above. Cached-continuation timings are not measurements of an evicted prompt-cache restore.

## Decision and restoration

Keep the original deployment while investigating the expanded deep parity failures. The attention change is a useful input-throughput candidate, but its changed outputs, failed continuation parity, and lower acceptance on that request prevent a blanket speed or promotion claim. The graph fix can be retained during a future rebase for its upstream behavior, but these tests do not justify a speed claim for it on this configuration. A full upstream rebase should preserve/revalidate Apex's parity safeguards and treat the GDN normalization correction plus fusion as a new numerical reference.

The original server, HIP library, and base library hashes match the pre-test release. The original launcher also verified the model and projector SHA-256 hashes successfully; see the verification log (local evidence reference). The restored process uses the original model, projector, launcher arguments, two GPUs, 262144 capacity, Q8 KV and MTP2 on port 8083. The separate Bonsai process remained alive and was not stopped.

A plain background restoration initially passed health checks but was reaped when its tool session ended. Final restoration therefore uses the existing user service manager with a **transient, forking unit**, `Restart=no`, running the original launcher. The unit is collected when that server exits; no persistent startup-unit file was installed. Independent post-run checks verify the daemon is active outside the benchmark session. See restoration verification (local evidence reference).

## Evidence and reproduction

- Evidence index (local evidence reference)
- Timing and parity summary (local evidence reference), all primary measurements (local evidence reference)
- Full short parity tokens (local evidence reference), deep parity and trace diagnostics (local evidence reference)
- Exact run configuration (local evidence reference), toolchain (local evidence reference), binary hashes (local evidence reference)
- MTP backport patch (local evidence reference), attention backport and tests (local evidence reference), source manifest (local evidence reference)
- [Source/release audit](upstream-comparison.md)

The scripts, source trees, binaries, full logs and profiler CSVs remain under `results_raw/benchmark-20260927/`. The evidence directory contains review copies of the scripts and compact proof objects.
