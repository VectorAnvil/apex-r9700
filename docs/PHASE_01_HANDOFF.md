# Phase 1 Handoff

## Outcome

Phase 1 created the Apex project repository and implemented the truthful Radeon AI PRO R9700 / `gfx1201` foundation. Work stopped before llama.cpp profiling or Phase 2 code.

## Repository and Commits

- Branch: `apex-r9700`
- Upstream remote: `https://github.com/AMD-AGI/Apex.git`
- Upstream base: `ecfad4d74812b5825a8bb5adf4da5e8a01abf188`
- Phase 0 documentation commit: `cd6401aef98976b16d6131468eef9f6208568c45`
- Phase 1 implementation commit: `7583c68b2925e7f5cee75f4457bbf6b49cb51d14`
- Hosted fork remote: not configured because no fork URL was supplied

The original `documents/` planning folder remains untracked and untouched. To synchronize later, fetch `upstream`, inspect its new commits, and rebase or merge them into `apex-r9700` with focused conflict review. Configure `origin` only when the real hosted fork URL is known.

## Implemented

- Added verified R9700 metadata: RDNA4, 64 CUs, 4096 stream processors, 32 GB GDDR6, 640 GB/s peak bandwidth, 64 MB Infinity Cache, 8 MB L2, wave32 on the validated host, 64 KiB HSA group segment, and max workgroup size 1024.
- Added official AMD/ROCm source URLs beside the static GPU-info metadata.
- Replaced GPU-info's silent `gfx950` fallback with explicit recognized/unknown detection state.
- Detects architectures through `rocm_agent_enumerator` and product names through machine-readable `rocm-smi` output, correctly tolerating the host's integrated `gfx1036` alongside two R9700s.
- Added a shared prompt hardware-context registry so R9700 prompts do not inherit Instinct, CDNA, wave64, HBM, or CDNA instruction guidance.
- Made unknown prompt auto-detection fail with an actionable `--target` diagnostic.
- Removed unconditional gfx950/MI355X advice from reflector text touched by this scope.
- Preserved `gpu_arch` when the kernel grader regenerates a trusted config and threaded the configured target through workload grading calls.
- Added an isolated preflight/smoke runner and a fixed HIP vector-add correctness/timing kernel. Build output exists only under a temporary directory.
- Added focused tests for recognized/unknown GPU detection, prompt context, compiler target propagation, config/grader propagation, and smoke command construction.

## Files Changed

- `tools/mcps/gpu_info/server.py`
- `prompts/hardware.py`
- `prompts/kernel_prompt.py`
- `prompts/model_prompt.py`
- `pipeline/reflector.py`
- `graders/kernel_grader.py`
- `workload_optimizer.py`
- `tools/gfx1201_smoke.py`
- `tests/gfx1201_smoke/vector_add.hip`
- `tests/test_gfx1201_phase1.py`
- `tests/test_prompts.py`
- `tests/test_tools.py`

## Authoritative Hardware Evidence

Official sources:

- `https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html`
- `https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html`

Live host evidence:

- Two `AMD Radeon AI PRO R9700` devices, each `gfx1201` and 64 CUs
- `rocm_agent_enumerator`: `gfx1201`, `gfx1201`, `gfx1036`
- ROCm 7.2.0; HIP `7.2.26015-fc0010cf6a`; AMD clang 22
- HIP compiler accepts `--offload-arch=gfx1201`

No fixed block/tile sizes, memory coalescing width, CDNA MFMA list, occupancy heuristic, achieved bandwidth, or empirical performance prescription was added.

## Validation Results

Focused Phase 1 tests:

```text
8 passed in 0.04s
```

Related prompt/tool/reflector suite:

```text
317 passed, 1 failed
```

The failure predates Phase 1: `Qwen/Qwen3.5-27B` has zero experts while `test_dense_models_have_one_expert` expects one. It was not changed because it is unrelated.

Broader CPU suite, excluding the obsolete GPU grader script and torch-dependent fused-MoE directory:

```text
858 passed, 1663 skipped, 7 failed
```

All seven failures are existing environment/data issues: missing cloned `tools/rocm`, ground-truth discovery expectations, absent live dataset artifacts, and the same Qwen registry mismatch.

Live smoke command:

```bash
python3 tools/gfx1201_smoke.py --arch gfx1201 --device 0
```

Result:

```text
device: AMD Radeon AI PRO R9700
arch: gfx1201
compile target: --offload-arch=gfx1201
elements checked: 1,048,576
mismatches: 0
correct: true
HIP-event iterations: 100
average kernel time: 0.0257818 ms
```

The timing is a smoke measurement, not a benchmark claim.

## Blockers and Known Issues

- Magpie is absent: no executable, Python module, or checkout exists. Real Apex/Magpie HIP grading could not run. The target propagation/config interface is covered by unit tests, but Magpie must be pinned and provisioned before its live smoke gate can pass.
- The `mcp` Python package is not installed in available Apex/system environments. GPU-info logic is unit tested and the equivalent live preflight passed, but the MCP server process was not launched.
- The Apex repository has only `upstream`; configure `origin` when a real fork URL is available.
- Llama Lab remains an unversioned workspace snapshot.
- The planning source files under `documents/` remain untracked by design.

## Phase 2 Objective

Implement only the llama.cpp profiling and source-mapping adapter. Run separate registered Q6_K PP and TG cases under `rocprofv3`, normalize real artifacts into Apex bottleneck records, map full symbols into the exact GGML HIP build/source, and persist independent top-N hotspot lists with evidence/confidence.

Do not optimize a kernel, rebuild a candidate, integrate Llama Lab E2E grading, mutate production, or begin Phase 3.
