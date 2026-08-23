# Phase 3 Handoff

## Outcome

Phase 3 proved an isolated, contract-bound optimization loop for the real
llama.cpp Q6_K token-generation MMV family on gfx1201. One candidate changes
the RDNA4 Q6_K `ncols_dst=1` workgroup from eight wave32 warps to four. It
passed the GGML CPU oracle and produced three stable model-derived
microbenchmark wins, but it remains `promising_not_promotion_ready`.

A final normal-graph attribution capture then corrected an important shape
assumption: the registered two-GPU tensor split submits per-device slices, not
the full GGUF matrix dimensions used by the Phase 3 harness. Phase 4 must retest
the exact runtime slices before any end-to-end promotion decision.

## Candidate Result

| Model-derived harness shape | Baseline median us | Candidate median us | Speedup | Status |
|---|---:|---:|---:|---|
| `M=17408,N=1,K=5120` | 123.240 | 120.980 | 1.0187x | stable microbenchmark win |
| `M=5120,N=1,K=17408` | 114.020 | 103.500 | 1.1016x | unstable; baseline spread 5.148% |
| `M=1024,N=1,K=5120` | 12.180 | 11.770 | 1.0348x | stable microbenchmark win |
| `M=5120,N=1,K=6144` | 39.050 | 36.910 | 1.0580x | stable microbenchmark win |

The instability threshold was frozen at 3.5% before confirmation. Baseline and
candidate passed four non-fused cases plus one fused case with the GGML CPU
backend oracle and maximum NMSE `0.0005`. No workload-weighted aggregate is
reported.

## Runtime Attribution

An env-gated hook in an isolated source clone recorded bounded Q6_K `N=1` host
submissions from the registered TG128 workload with HIP graphs enabled and both
R9700s visible. It recorded tensor names, `ne`/`nb`, device, fusion state, and
the dynamically selected launch. It did not synchronize a GPU, inspect buffers,
or claim graph replay counts. The direct `llama-bench` diagnostic returned 0;
both devices reported `gfx1201`, wave32, and the original `32x8x1` block.

The normal-graph capture produced 3,464 bounded records, split equally by
device and by capture-active/capture-none host submission. These counts are not
runtime frequencies. Joining its unique tensor/shape map to the preserved
Phase 2 graphs-disabled trace by agent, fusion variant, and launch geometry
recovered all 555,106 Q6_K MMV dispatches and the original 70.903% family share.

| Runtime attribution | Per-device shape | Summed TG dispatch duration | Quality |
|---|---|---:|---|
| fused `ffn_up` | `M=8704,N=1,K=5120` | 29.842% | exact |
| `attn_output` / `attn_qkv` / `ffn_down` / `ssm_out` | `M=5120,N=1,K=3072/5120/8704` | 27.983% | ambiguous |
| `attn_gate` | `M=3072,N=1,K=5120` | 4.317% | exact |
| output head | `M=124160,N=1,K=5120` | 3.197% | exact |
| `attn_q` | `M=6144,N=1,K=5120` | 2.704% | exact |
| `attn_k` / `attn_v` | `M=512,N=1,K=5120` | 1.992% | ambiguous |
| `ssm_alpha` / `ssm_beta` | `M=24,N=1,K=5120` | 0.868% | ambiguous |

The trace cannot separate operations that share the same Q6_K template,
fusion state, and launch geometry. In particular, it cannot allocate the
27.983% `M=5120` bucket among its four candidate operations. No vendor dense
GEMM dispatch appeared; this remains custom GGML quantized work.

## Code Object

- Target: `hipv4-amdgcn-amd-amdhsa--gfx1201`, wave32.
- Workgroup: 256 to 128 threads.
- Q6_K false/true LDS: 896/1792 to 384/768 bytes.
- False variant VGPR/SGPR: unchanged at 26/26.
- Fused variant VGPR/SGPR: unchanged at 35/42.
- Private segment and spills: zero in both builds.
- ISA instruction counts: 294 to 284 false; 652 to 620 fused.

These deltas confirm that the intended four-wave specialization compiled. They
do not prove why any timing changed.

## Provenance

- Source commit: `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f` plus the exact registered P2P diff.
- Task fingerprint: `297a07be77dbabde2a93c9429c45ef9c75928a1f9e0d4ec2cfdbd4a0d209c9b0`.
- Task SHA-256: `806d4646a1ac5e96ba133ecb37e78317cd372eaac627883ea1103b2eaade5cab`.
- Candidate patch SHA-256: `bdbe7b5098e860e8d2870fac2cad24020a8c4937f5998de4286646fb9a847d6b`.
- Candidate result SHA-256: `6860ea76a8c5a37306df68c476a2b9afd2f6cba828c52f34e51749ca240caefc`.
- Attribution JSONL SHA-256: `c3a82f3e31cc2bb105403ac745f188048694916b348250a292c5726e7a6cbc15`.
- Phase 2 TG trace SHA-256: `dd4fe80729863c7485267189d618dbbd72195978feb95d2ed20f45a0d53dfa98`.
- Raw Phase 3 root: `results_phase03_q6k_mmv_20260808/` (ignored, approximately 738 MB before the attribution build).
- Portable evidence: `docs/apex-r9700/artifacts/phase-03/`.

The registered source/build, model, services, and known-good P2P patch were not
modified. The original untracked `documents/` directory remains untouched.

## Phase 4 Recommendation

Evaluate only `rdna4-q6k-nwarps4`. First freeze and run exact per-device shape
cases, especially fused `8704x1x5120`, unfused `5120x1x8704`, and the
`124160x1x5120` output head. If correctness or stable timing fails, reject the
candidate. Only after that gate should isolated full baseline/candidate builds
enter normal-graph, two-R9700 Llama Lab PP/TG comparison. Do not begin a second
candidate or port a reference kernel during Phase 4.
