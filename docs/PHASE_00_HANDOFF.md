# Phase 0 Handoff

## Outcome

Phase 0 completed the feasibility and architecture audit. No gfx1201 implementation, profiling run, model download, llama.cpp build, vLLM launch, or production mutation was performed.

Apex remains the recommended orchestration base, with a firm boundary: reuse agent/learning/grading infrastructure, but add dedicated llama.cpp profiling, compiled-source mapping, and rebuild/promotion adapters. Do not adapt Python hot-patching or AST tracing.

## Files and Commit State

Phase 0 created:

- `docs/apex-r9700/PROJECT_CHARTER.md`
- `docs/apex-r9700/PROJECT_STATE.md`
- `docs/apex-r9700/DECISIONS.md`
- `docs/apex-r9700/PHASE_00_HANDOFF.md`
- `docs/apex-r9700/NEXT_PHASE_PROMPT.md`

No Phase 0 commit exists because `/home/adam/workspaces/ChatGPT/Apex` is not a Git worktree. Phase 1 must create the Apex fork first, then add these documents to it without rewriting their audit conclusions.

## Revisions and Runtime Inspected

| Item | Revision / state |
|---|---|
| AMD-AGI/Apex | `ecfad4d74812b5825a8bb5adf4da5e8a01abf188`, `https://github.com/AMD-AGI/Apex.git`; temporary shallow audit clone because no checkout existed |
| AgentKernelArena | `ea4c0ee0e9c0550f01af1289b03f89ccd784bf7e` |
| Magpie | upstream head `12896a49a731ad72c791b7a23abcef7a0d6c4487`; no local checkout |
| Llama Lab | unversioned workspace snapshot; no reliable commit available |
| stock llama.cpp | `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`, clean, `llama-stock-259f2e2a5`, `build-gfx1201-stock` |
| patched llama.cpp | same detached commit; dirty only `ggml/src/ggml-cuda/allreduce.cu/.cuh`; two isolated P2P build dirs |
| P2P patch repo | `f6f66106f2a960cec2da5a2f1a0f6d476989815f` |
| Host | 2 x R9700 `gfx1201`, 64 CUs; ROCm/HIP `7.2.26015`; `rocprofv3` 1.1.0 |

Builds are Release HIP with `AMDGPU_TARGETS=gfx1201`, HIP graphs/MMQ MFMA enabled, RCCL disabled, and ROCWMMA flash attention disabled. Existing `compile_commands.json` files can drive source mapping. llama.cpp itself correctly gates MFMA to CDNA and WMMA to RDNA3/RDNA4 in `ggml/src/ggml-cuda/common.cuh`.

Llama Lab registers the Huihui Qwen3.6-27B Q6_K model, stock/patched profiles, and PP512/TG128 tensor-split suites. Existing evidence reports TG 33.552 to 36.695 tokens/s (+9.37%) and PP 812.215 to 820.486 tokens/s (+1.02%) for the P2P experiment; Phase 0 did not rerun it.

## Architecture

```text
Llama Lab registered model/build/profile/suite
 -> controlled argv launch -> rocprofv3 adapter
 -> canonical PP/TG kernel measurements
 -> Apex ranking + source mapper
 -> Apex agent + knowledge/reflection/trajectory
 -> isolated llama.cpp candidate build
 -> correctness + microbenchmark
 -> Llama Lab baseline/candidate E2E grading
 -> retain or reject artifacts
```

## Architecture Matrix

| Subsystem | Classification | Reason |
|---|---|---|
| `agents/backends.py` | REUSE UNCHANGED | Agent selection is workload independent. |
| `pipeline/knowledge_base.py` | REUSE UNCHANGED | Durable optimization knowledge is backend neutral. |
| `pipeline/trajectory.py` and leaderboard | REUSE UNCHANGED | Run history/reporting can store adapter results. |
| `pipeline/reflector.py` | EXTEND | Generic reflection works; remove/filter `gfx950` advice. |
| Anti-tamper/cache framework | REUSE UNCHANGED | Benchmark-integrity controls remain valuable. |
| GPU-info MCP | EXTEND | Add truthful R9700 RDNA4 data; remove `gfx950` fallback. |
| Prompt model/config/kernel tables | EXTEND | Add RDNA4 context and avoid CDNA hints. |
| Kernel grader/config generator | EXTEND | Add llama.cpp-owned `library_test` harness mode. |
| `graders/ground_truth.py` | EXTEND | Current discovery scans Python ROCm tests, not GGML. |
| Source-finder MCP | EXTEND | Add llama roots, symbol normalization, build identity, confidence. |
| `pipeline/kernel_bottleneck.py` | ADAPTER REQUIRED | Ranking is reusable; schemas/classifiers assume Magpie/serving stacks. |
| Bottleneck record | EXTEND | Add phase, build, full symbol, source evidence, confidence. |
| ROCm profiler ingestion | ADAPTER REQUIRED | Normalize real rocprof CSV/JSON measurements. |
| Llama Lab E2E integration | ADAPTER REQUIRED | Submit registered IDs and consume results; do not duplicate logic. |
| Llama Lab registry/profile/suite/queue/parser | REUSE UNCHANGED | Safe argv, serialization, parsing, and provenance already exist. |
| `workload_optimizer.py` workload path | REPLACE FOR LLAMA.CPP | Defaults to `gfx950` and Python-visible serving workloads. |
| `pipeline/kernel_tracing/` | REPLACE FOR LLAMA.CPP | Python AST tracing cannot map compiled HIP launches. |
| Reintegration/hot-patching | REPLACE FOR LLAMA.CPP | Installed-module copying must become isolated rebuilding. |
| `graders/model_grader.py` | REPLACE FOR LLAMA.CPP | Accepts vLLM/SGLang and Magpie E2E only. |
| Magpie kernel compare | ADAPTER REQUIRED | Needs llama.cpp-owned correctness harness. |
| Magpie serving Docker E2E | DISABLE FOR OUR BACKEND | Wrong engine and benchmark authority. |
| AITER specs/Triton task paths | DISABLE FOR OUR BACKEND | Do not represent GGML HIP. |
| vLLM TP/cleanup/Docker assumptions | DISABLE FOR OUR BACKEND | Irrelevant and unsafe for this workload. |

## Material Stock Assumptions

- `workload_optimizer.py` defaults compiler/CLI targets to `gfx950` and defines Python module specs, installed shared-object patching, backups, and in-place copies.
- Its trace CLI targets Triton or Python-visible HIP/custom operations.
- `graders/model_grader.py` restricts frameworks to vLLM/SGLang; `graders/score.py` defaults Magpie to vLLM Docker.
- `pipeline/kernel_bottleneck.py` accepts Magpie-shaped data, but classification/provenance patterns are vLLM/SGLang/AITER specific.
- `pipeline/kernel_tracing/runner.py`, `patch_triton.py`, and `patch_wrapper.py` use Python AST/wrapper injection.
- GPU-info MCP defines CDNA `gfx950/gfx942/gfx90a` and RDNA3 `gfx1100`; unknown/error paths fall back to `gfx950`.
- Prompt tables and reflector hints contain CDNA/`gfx950` assumptions.
- Source finder searches ROCm roots and has reusable demangling/`.cpp/.cu/.hip` grep, but no llama.cpp root/build identity.
- Ground-truth discovery scans Python ROCm-tool tests, not GGML harnesses.

## Profiler and Source Mapping

The bottleneck ranker is not fundamentally serving-coupled: an adapter can provide normalized llama.cpp measurements. Its classifiers/provenance do require GGML extensions. The dynamic tracer must be replaced for compiled HIP.

Mapping should retain full mangled/demangled symbols, derive a separate normalized lookup key, use `c++filt` plus the exact build's `compile_commands.json`, search configured `ggml/src/ggml-cuda`, and record candidates/confidence. Known loci include Q6_K conversion in `convert.cu`, `mul_mat_q` in `mmq.cuh/mmq.cu`, Q6_K MVM in `mmvq.cu`, and `ggml_ar_hip_kernel<T>` in patched `allreduce.cu`.

## Magpie, Correctness, and Reintegration

Magpie can plausibly compile/compare `.hip`/`.cu`, but cannot establish llama.cpp correctness alone. Start with `library_test` and an immutable GGML/fixed-input harness. PyTorch is unsuitable for Q6_K; Accordo is later-stage.

Future promotion creates a pinned isolated worktree, applies an allowlisted patch, uses captured CMake settings in a candidate build dir, runs immutable correctness/microbenchmarks, registers the build with Llama Lab, submits the same suite for baseline/candidate, and preserves raw evidence. Never use installed-module hot-patching.

## Smallest Useful POC (Phase 2, Not Implemented)

1. Add an allowlisted profiling wrapper around a registered Llama Lab build/profile.
2. Run separate `llama-bench -pg pp` and `-pg tg` cases under `rocprofv3 --kernel-trace --runtime-trace --stats`; there are no ROCTx markers and the current queue does not emit `-pg`.
3. Validate ROCm 7.2 CSV/JSON schemas from a fresh trace; no profiler artifacts currently exist.
4. Normalize full symbol, calls, total/average duration, phase percentage, build/source identity, command, and raw artifact.
5. Feed records to generalized Apex bottleneck extraction and attach demangled source candidates/confidence.
6. Persist independent top-N PP and TG hotspot lists.

No optimizer, Magpie, model download, vLLM, production mutation, or rebuild is needed for this POC.

## Exact Extension Points

- Apex: `workload_optimizer.py`
- Apex: `pipeline/kernel_bottleneck.py`, `kernel_tracing/`, `knowledge_base.py`, `reflector.py`, `trajectory.py`
- Apex: `agents/backends.py`
- Apex: `graders/kernel_grader.py`, `model_grader.py`, `ground_truth.py`, `config_generator.py`, `score.py`
- Apex: `prompts/models.py`, `configs.py`, `kernel_prompt.py`, `model_prompt.py`
- Apex: `tools/mcps/source_finder/server.py` and GPU-info MCP database/detection
- Llama Lab: `services/build_registry.py`, `command_builder.py`, `benchmark_queue.py`, `benchmark_matrix.py`, `benchmark_parser.py`, `result_analysis.py`, `activity_guard.py`, `process_manager.py`
- llama.cpp: `ggml/src/ggml-cuda`, selected by actual profiler evidence

## Blockers and Risks

- No Apex fork exists. Phase 1 must create/pin it and document upstream sync.
- Llama Lab lacks version-control identity; add reliable provenance before integration.
- No real rocprof artifact exists; schema/symbol quality remains unverified until Phase 2.
- The P2P checkout is dirty rather than a candidate commit; use patch hashes/manifests.
- RDNA4 metadata must be sourced/measured, never invented.
- Template symbols may map to multiple locations; report confidence/evidence.
- Llama Lab queue policy rejects production builds while some design prose permits them; resolve without weakening safe defaults.
- Kernel wins may not move E2E results; Llama Lab remains mandatory.

## Recommended Phase 1 Scope

Create a small Apex fork; detect R9700/`gfx1201` without fallback; add verified RDNA4 metadata/prompt context/compiler propagation/preflight; and add a tiny HIP smoke test proving compile, execution, correctness, timing, and Apex/Magpie grading compatibility. Test detection, target selection, metadata, and prompt output.

Do **not** add llama.cpp profiling, source mapping, rebuilding, Llama Lab invocation, autonomous agents, vLLM, or optimization in Phase 1.

## Validation

Phase 0 was read-only: source/revision/build-cache/Llama Lab database inspection and installed-profiler capability checks. No runtime benchmark or GPU test was authorized or run.

Documentation validation performed:

```bash
find docs/apex-r9700 -maxdepth 1 -type f -printf '%f\n' | sort
grep -nE 'Architecture Matrix|Smallest Useful POC|Recommended Phase 1 Scope|DO NOT BEGIN PHASE 2' docs/apex-r9700/*.md
git rev-parse --show-toplevel  # expected failure: not a Git worktree
```

All five required files exist, all required handoff sections were found, and the generated Markdown contains no non-ASCII bytes.

These documents live in `Apex/docs/apex-r9700` because they govern the Apex adaptation across all three systems. Llama Lab remains the referee, not the project owner.
