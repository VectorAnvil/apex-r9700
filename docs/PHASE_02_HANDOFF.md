# Phase 2 Handoff

## Outcome

Phase 2 implemented the llama.cpp profiling and source-mapping adapter and produced separate real PP and TG hotspot lists from the registered Huihui Qwen3.6-27B Q6_K workload on both Radeon AI PRO R9700 GPUs. No kernel was optimized, no llama.cpp source/build/model was changed, and Phase 3 did not begin.

## Repository

- Branch: `apex-r9700`
- Phase 1 closeout: `c4db38427308d8665a3861a3779aba4dc4ba2cd0`, tag `phase-01-complete`
- Phase 2 implementation and portable evidence: `2ecd8db029d3516334700818dca4b355d3c98810`
- Phase 2 closeout: tag `phase-02-complete` on the documentation commit containing this handoff
- Upstream remote: `https://github.com/AMD-AGI/Apex.git`
- Hosted `origin`: not configured
- Original untracked `documents/`: preserved and untouched

## Implemented

- `tools/llama_cpp_profile.py`: read-only Llama Lab registry discovery, allowlisted argv construction, Llama Lab protected-activity checks, target-GPU idle check, dynamic profiler/GPU evidence, separate PP/TG launch, raw artifact preservation, normalization, source mapping, and reports.
- `pipeline/rocprofv3_adapter.py`: streaming ROCm 7.2 CSV validation and per-phase/per-agent/exact-symbol aggregation. Raw CSV remains dispatch-level evidence; normalized correlation samples are bounded.
- `pipeline/compiled_hip_mapper.py`: safe `c++filt` argv invocation, lookup-only `.kd` removal, exact `compile_commands.json` hash, configured GGML HIP roots, file/line candidates, and conservative `high`/`medium`/`low`/`none` confidence.
- `pipeline/kernel_bottleneck.py`: backward-compatible rocprofv3 provenance fields and direct ingestion without fabricating Magpie or serving-stack provenance. GGML compiled kernels classify as HIP with origin `ggml`.
- Offline fixtures and focused parser, mapper, runner, integration, and regression tests.

## Registered Input

- Llama Lab database: configured `data/llama-lab.db`, opened read-only
- Profile 5: `HIP TP patched #25197 - Huihui Qwen3.6 27B Q6_K`
- Suite 2: `HIP P2P AllReduce #25197 decode`
- Model 2: `Huihui-Qwen3.6-27B-abliterated`, Q6_K, 22,430,998,848 bytes
- Build 2: registered non-production P2P build, validation `valid`
- llama.cpp: `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`, build 9940
- P2P patch repository: `f6f66106f2a960cec2da5a2f1a0f6d476989815f`; both registered patch hashes are in the evidence manifest
- Build configuration: Release HIP, `AMDGPU_TARGETS=gfx1201`, HIP graphs and MMQ MFMA compiled in, RCCL and ROCWMMA flash attention off
- Workload: PP512, TG128, batch 2048, microbatch 512, tensor split, flash attention on, 5 tool repetitions, 12 threads, 99 GPU layers, `HIP_VISIBLE_DEVICES=0,1`

The adapter discovered every path from the registered profile/build/model/suite. No model, build, device, or user path is hard-coded in implementation code.

## Exact Invocation

```bash
/home/adam/workspaces/ChatGPT/llama-lab/.venv-linux/bin/python \
  tools/llama_cpp_profile.py \
  --llama-lab-root /home/adam/workspaces/ChatGPT/llama-lab \
  --profile-id 5 \
  --suite-id 2 \
  --results-dir /home/adam/workspaces/ChatGPT/Apex/results_phase02_qwen36_p2p_nographs_20260808
```

The generated phase controls were:

```text
PP: -p 0 -n 0 -pg 512,0
TG: -p 0 -n 0 -pg 0,128
```

Both ran under `rocprofv3 --runtime-trace --stats --mangled-kernels --output-config --output-format csv json`. The exact complete argv, cwd, registered identity, and environment are preserved per capture in `profile_evidence.json`.

## Profiler Constraint

The first PP attempt with HIP graphs enabled failed with SIGSEGV in `ggml_cuda_graph_evaluate_and_capture`. It is preserved under ignored `results_phase02_qwen36_p2p_20260808/`. No GPU memory remained allocated after failure.

The successful diagnostic captures used the recorded override:

```text
GGML_CUDA_DISABLE_GRAPHS=1
```

This avoids ROCprofiler-SDK 1.1.0 graph-capture failure without changing the binary or kernel source. The observed llama-bench rates, PP 780.426 tokens/s and TG 22.942 tokens/s, are profiler diagnostics and must not be compared with ordinary Llama Lab results.

## Evidence

- Portable evidence: `docs/apex-r9700/artifacts/phase-02/profile_evidence.json`
- Readable ranked lists: `docs/apex-r9700/artifacts/phase-02/HOTSPOTS.md`
- Raw/normalized host root: `/home/adam/workspaces/ChatGPT/Apex/results_phase02_qwen36_p2p_nographs_20260808`
- Preserved artifacts: 38 files, 5,805,767,316 bytes
- PP capture: `33a989b1-34aa-4cf3-8f11-5535a451eb59`
- PP kernel CSV SHA-256: `8563e42c0964c572530a170c05b13262eccee0483b68d8fd2bea6136e7c913bd`
- TG capture: `3730bdee-5ddf-4ec2-ace5-f15090749177`
- TG kernel CSV SHA-256: `dd4fe80729863c7485267189d618dbbd72195978feb95d2ed20f45a0d53dfa98`
- ROCm 7.2.0; rocprofv3 1.1.0; SDK revision `fc0010cf6a5a972d42b276df946510f30343d493`
- Dynamic capabilities for both gfx1201 agents are stored in `rocprofv3_capabilities.txt`; no PMCs, PC sampling, ATT, or power-state changes were used

## Hotspots

Percentages below sum the independently retained records for both agents. They are percentages of summed GPU dispatch duration, not wall time.

### PP

| Kernel family | Both-agent GPU time | Calls | Source evidence |
|---|---:|---:|---|
| `mul_mat_q<(ggml_type)14, 128, false>` | 81.002% | 4,800 | low; `mmq.cuh:3528`, `mmq.cuh:3542` among candidates |
| `gated_delta_net_cuda<128, false, false>` | 5.494% | 576 | medium; `gated_delta_net.cu` |
| `mul_mat_q<(ggml_type)14, 128, true>` | 4.009% | 1,152 | low; `mmq.cuh` candidates |
| add broadcast kernel | 2.087% | 3,648 | medium; `binbcast.cu` |

### TG

| Kernel family | Both-agent GPU time | Calls | Source evidence |
|---|---:|---:|---|
| `mul_mat_vec_q<(ggml_type)14, 1, false, false>` | 41.061% | 473,058 | low; `mmvq.cu:480`, `mmvq.cu:807` among candidates |
| `mul_mat_vec_q<(ggml_type)14, 1, true, false>` | 29.842% | 82,048 | low; `mmvq.cu` candidates |
| `ggml_ar_hip_kernel<float>` | 16.383% | 164,096 | medium; `allreduce.cu:986`, `allreduce.cu:1164` |
| `quantize_q8_1` | 2.698% | 555,106 | medium; `quantize.cu` |
| `rms_norm_f32<1024, true, false>` | 1.892% | 165,378 | medium; `norm.cu` |

The all-reduce split is asymmetric: Agent 1 accounts for 13.065 percentage points and Agent 2 for 3.318. This is real trace evidence worth later investigation, but it is not the first Phase 3 target.

## Phase 3 Selection

Primary target:

```text
mul_mat_vec_q<(ggml_type)14, 1, false/true, false>
```

Rationale: the Q6_K TG variants account for 70.903% of summed dispatch duration across both R9700s and map to the exact build's `ggml/src/ggml-cuda/mmvq.cu` family. Phase 3 must narrow the template/dispatch locus with an isolated Q6_K correctness and timing harness before editing anything.

Secondary evidence only: `ggml_ar_hip_kernel<float>`. It is material and imbalanced, but it belongs to the known-good dirty P2P patch and would complicate the first scientific kernel loop.

## Validation

Focused and workload regression suite:

```text
85 passed in 0.41s
```

Related Phase 1/prompt/reflector suite:

```text
192 passed, 1 failed
```

The one failure is unchanged from Phase 1: `Qwen/Qwen3.5-27B` has zero experts while `test_dense_models_have_one_expert` expects one. Ruff passed with explicit exclusions for line length, intentional path-bootstrap import order, and the legacy modernization/blind-exception rules in `pipeline/kernel_bottleneck.py`; the unfiltered policy still reports those non-behavioral findings. `git diff --check` passed.

## Risks

- HIP graphs had to be disabled for profiler compatibility. Phase 3 timings need an isolated harness and eventual normal-graph Llama Lab validation; profiler throughput is not a baseline.
- Template mapping remains deliberately low confidence where several source/dispatch sites are plausible. Phase 3 must not treat a candidate line as an exact launch locus without more evidence.
- Runtime tracing created a 5.7 GB TG artifact. Kernel-only tracing is preferable for repeated investigations after the initial evidence capture.
- The patched source tree remains intentionally dirty only in its known-good P2P files. Do not edit or clean it.
- Magpie remains unprovisioned. Phase 3 may add a llama.cpp-owned harness, but must not weaken correctness or substitute PyTorch truth for Q6_K.
- Llama Lab remains an unversioned workspace snapshot.

## Stop

Phase 2 is complete. Do not begin kernel optimization, candidate rebuilding, promotion, or Llama Lab E2E comparison until Phase 3 is explicitly started.
