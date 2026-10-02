> Historical research record. Statements about the selected service are as of this report, not current deployment status. See [the V2 index](../README.md). Local raw evidence, private runtime files and logs are not included; local artifact paths are provenance references, not downloadable links.

# Bounded matrix-tile scheduling, 2026-09-30

Decision: keep as a promising experimental prefill change. Operator gains reproduced on both cards; the full-model curve and checkpoint checks preserve exact sampled outputs. No live deployment of this kernel change.

Isolated worktree from cbc3985. APEX_V2_FA_PREFILL_SCHEDULE=1 selects the new head256/GQA6/Q8 ncols2=2 prefill specialization for Q>=32. It inserts __builtin_amdgcn_sched_barrier(0) after each QK accumulator tile and PV output tile. Precision, query residency, K/V staging, loop geometry, arithmetic dependence and shared allocation stay unchanged. The existing grouped decode route is not modified.

This is a compiler scheduling constraint, not a workgroup synchronization primitive. ISA inspection confirms unchanged 48 barrier-signal/wait pairs and unchanged 384 static WMMA instructions. It changes scheduling and wait instructions, so throughput still needs measurement.

The selected kernel's private segment decreases from 424 to 196 bytes per workitem; VGPR spill slots fall from 146 to 50; static scratch loads/stores fall from 365/142 to 172/49. Both kernels still report 256 VGPRs. Static LDS load/store counts match; LDS waits increase while load waits decrease. These are static instruction counts, not measured dynamic execution shares.

## Operator screen

Eight CPU-reference tests passed. Six KV depths with Q512 were tested in ABBA order on each card, 48 points. Positive values below mean attention-operator throughput improvement, not model prompt-processing speed.

| KV tokens | GPU0 | GPU1 | Mean |
|---:|---:|---:|---:|
| 8192 | +5.73% | +6.11% | +5.92% |
| 32768 | +4.51% | +3.83% | +4.17% |
| 65536 | +4.20% | +3.39% | +3.80% |
| 131072 | +3.68% | +3.24% | +3.46% |
| 196608 | +3.25% | +2.61% | +2.93% |
| 262144 | +3.06% | +2.15% | +2.60% |

## Dual-GPU model curve

Matched Qwen3.8-27B-Heretic Q6_K, Q8 KV, tensor split 1,1, context 262144, batch 2048 / ubatch 512, MTP1, original grouped decode and cache coverage enabled. Both lanes processed identical prompt suffixes and generated 64 tokens per request. The candidate's mapped HIP/server libraries and runtime gates were verified in CANDIDATE_RUNTIME.json.

All 14 paired requests had bit-identical full first-step logits, identical 64-token rollouts, identical processed/cache counts and identical MTP draft/accept counts. Both deep edit and unedited-history restoration reused 232960 tokens; each restored output matched the earlier unedited grow output exactly. These are sampled numerical/checkpoint results, not an independent broad capability benchmark.

| Context tokens | Baseline growing PP tok/s | Candidate growing PP tok/s | PP change | Baseline repeat TG tok/s | Candidate repeat TG tok/s |
|---:|---:|---:|---:|---:|---:|
| 8192 | 1125.58 | 1131.67 | +0.54% | 50.87 | 50.96 |
| 32768 | 1040.32 | 1062.04 | +2.09% | 51.01 | 51.43 |
| 65536 | 895.22 | 914.45 | +2.15% | 48.23 | 48.40 |
| 131072 | 724.28 | 742.77 | +2.55% | 45.02 | 45.06 |
| 196608 | 572.20 | 590.72 | +3.24% | 43.50 | 43.45 |
| 261632 | 473.40 | 491.88 | +3.90% | 40.49 | 40.53 |

Growing-prefix PP is not fresh full-context PP except at 8K. At 261632, both processed 65536 new/replayed suffix tokens: 138.437 s baseline versus 133.237 s candidate. Repeating that prompt processed 512 tokens: 1.392 s versus 1.348 s. History edit processed 28672 tokens: 63.520 s versus 61.083 s; restoration processed the same count: 63.527 s versus 61.095 s. Generation stayed essentially unchanged; this kernel change targets prefill.

This is a single ordered A/B at model level. The observed 2-4 percent PP improvement beyond 8K needs reverse-order confirmation before calling the exact model speedup statistically established. The independent operator ABBA screen supports a real kernel gain. Keep the candidate for stacking; do not redefine the selected baseline. The guard restored the existing cache-corrected grouped trial after all 28 requests completed.

## Sources and limits

[Clang AMDGPU builtin reference](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html) documents the scheduler mask. [AMD's CDNA4 GEMM example](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html) demonstrates instruction-order control and unroll tuning. The example uses different hardware and workload; our measurements are the evidence for this candidate. [LLVM discussion](https://discourse.llvm.org/t/prevent-code-motion-in-ir-optimization/87058) clarifies that this intrinsic does not prevent all earlier IR-level code motion.

Evidence in experiment root: build-schedule/BUILT.json, resources/comparison.json, resources/instruction-counts.json, screen/PASSED.json, raw logs and completed model-curve outputs. Baseline/frozen sources and artifacts remain untouched. Model runner restores the selected cache-corrected grouped trial in its finally guard. No host dependency changes.
