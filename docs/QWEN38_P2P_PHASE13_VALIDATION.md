# Qwen3.8-27B on dual R9700: Direct-P2P and Phase 13

Date: 2026-08-14

## Summary

Two independently useful changes define the current Qwen3.8-27B Q6_K
optimization stack on Cornelius:

1. The externally authored HIP Direct-P2P AllReduce patch improves
   tensor-split token generation.
2. The Phase 13 Q6_K MMQ explicit F32 conversion improves prompt processing.

The Direct-P2P patch is not our implementation. It was authored by
`JohnTDI-cpu` and obtained from
`https://github.com/JohnTDI-cpu/llama-hip-p2p-allreduce`. Our work on that
patch was integration, controlled isolation, current-upstream rebasing, and
validation on the dual-R9700 Qwen3.8 workload. Phase 13 has separate origin and
attribution, documented below.

On current llama.cpp build 10434, the complete stack measured:

**1018.723 PP512 / 36.923 TG128**

Against current upstream stock, this is:

- **+16.005% PP512**
- **+9.241% TG128**

The current canonical configuration is:

`Qwen3.8-27B-Q6_K + two-GPU tensor split + HIP Direct-P2P + Phase 13 Q6_K F32 cast`

## HIP Direct-P2P

### Problem

The older HIP tensor-split path used the generic meta-backend butterfly for
AllReduce. For small decode reductions, this host-staged route moved partials
across PCIe more than necessary and imposed a material communication cost.

The experimental HIP Direct-P2P path gives each of the two GPUs peer-visible
staging and performs the small reduction directly across the devices. It is
conservatively gated and falls back to the ordinary path when its requirements
are not met. Large prompt-processing reductions continue to use the fallback
path.

### Isolated current-upstream result

| Build | PP512 | TG128 |
| --- | ---: | ---: |
| Current upstream stock | 878.173 t/s | 33.800 t/s |
| Current + Direct-P2P only | 875.083 t/s | 36.907 t/s |

Direct-P2P alone changed:

- PP512: **-0.352%**, effectively flat to slightly lower
- TG128: **+9.191%**

This isolates Direct-P2P as primarily a decode optimization for this workload.
It does not explain the large prompt-processing result previously associated
with the older research binary.

### P2P provenance

- Original author: `JohnTDI-cpu`
- Upstream patch repository:
  `https://github.com/JohnTDI-cpu/llama-hip-p2p-allreduce`
- Original llama.cpp research base: `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`
- Standalone patch repository: `f6f66106f2a960cec2da5a2f1a0f6d476989815f`
- Rebased llama.cpp commit: `7e4c0a96880dae4fc4268ad441f8a6446bd5460a`
- Rebased llama.cpp build: 10434
- Modified P2P files:
  - `ggml/src/ggml-cuda/allreduce.cu`
  - `ggml/src/ggml-cuda/allreduce.cuh`

Direct-P2P must be credited to `JohnTDI-cpu`; it must not be described as an
Apex-authored patch. The Apex contribution reported here is its reproducible
evaluation, component isolation, and rebase alongside the independently
implemented Phase 13 change.

## Phase 13 Q6_K MMQ F32 conversion

### Finding

Phase 13 adds an explicit F32 conversion at the Q6_K MMA accumulation
operation. On the older llama.cpp source tree the operation was in `mmq.cuh`.
Current upstream moved it to `mmq-vec-dot.cuh`; applying the same conversion at
the corresponding operation preserved and strengthened the prompt-processing
gain.

The current-upstream form changes the accumulation expression from the
implicit arithmetic form:

```cpp
C.x[l] * sc[k01/4] * x_df[i*sram_stride] * dB
```

to an explicit F32 conversion of the MMA result:

```cpp
((float) C.x[l]) * sc[k01/4] * x_df[i*sram_stride] * dB
```

### Incremental result

| Build | PP512 | TG128 |
| --- | ---: | ---: |
| Current + Direct-P2P only | 875.083 t/s | 36.907 t/s |
| Current + Direct-P2P + Phase 13 | 1018.723 t/s | 36.923 t/s |

Adding Phase 13 to the otherwise identical P2P build changed:

- PP512: **+16.414%**
- TG128: **+0.045%**, effectively neutral at this resolution

This isolates Phase 13 as primarily a Q6_K prompt-processing/MMQ
optimization. The result surviving the upstream file and code reorganization
supports the conclusion that it is tied to Q6_K arithmetic and code generation,
not to the old filename or an accidental surrounding layout.

### Origin and attribution

Phase 13 was independently implemented and experimentally validated during
the Apex llama.cpp/RDNA4 optimization campaign. Earlier phases studied
external RDNA4 and Q6_K implementations, including Zinc-informed ideas, as
architectural references. The retained Phase 13 records do not identify the
explicit llama.cpp F32-cast change as a copied patch, cherry-pick, or direct
port from another repository.

The supported attribution is therefore:

> Phase 13 was independently implemented and benchmarked during the Apex
> optimization campaign, informed by broader RDNA4/Q6_K research.

This does not claim invention of F32 conversion or accumulation as a general
concept. The specific finding is that this minimal change in llama.cpp's HIP
Q6_K MMQ path produces a large and reproducible PP512 improvement on the tested
gfx1201 workload.

## Provenance correction

The historical build-9940 result of approximately `931 PP / 36.5 TG` was
initially labeled Direct-P2P. A later source audit showed that its dirty
research tree contained three relevant modifications:

1. the Direct-P2P change in `allreduce.cu`;
2. the Direct-P2P declaration change in `allreduce.cuh`;
3. the Phase 13 Q6_K explicit F32 conversion in `mmq.cuh`.

That binary was a cumulative **Direct-P2P + Phase 13** build. It was not a
P2P-only build. The current-upstream three-lane isolation supersedes any
interpretation that attributes the historical PP gain to P2P alone.

## Current validation

### Environment

- GPUs: 2 x AMD Radeon AI PRO R9700, 32 GB each
- GPU architecture: `gfx1201`
- Backend: llama.cpp HIP
- ROCm/HIP: 7.2.26015
- llama.cpp: `7e4c0a96880dae4fc4268ad441f8a6446bd5460a`, build 10434
- Model: `Qwen3.8-27B-Q6_K.gguf`
- Model SHA-256: `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727`

### Command

```text
HIP_VISIBLE_DEVICES=0,1 llama-bench \
  -m /mnt/storage/ai/models/llm/qwen3.8-27b-unsloth/Qwen3.8-27B-Q6_K.gguf \
  -p 512 -n 128 -b 2048 -ub 512 -sm tensor -fa 1 -r 5
```

Three independent processes were executed per build. Each process aggregated
five internal samples. No samples were trimmed.

### Complete results

| Build | PP512 launch means (t/s) | PP512 mean +/- process SD | TG128 launch means (t/s) | TG128 mean +/- process SD |
| --- | --- | ---: | --- | ---: |
| Current stock | 880.64, 878.06, 875.82 | 878.173 +/- 2.412 | 33.90, 33.77, 33.73 | 33.800 +/- 0.089 |
| Current + Direct-P2P | 875.17, 875.25, 874.83 | 875.083 +/- 0.223 | 36.92, 36.91, 36.89 | 36.907 +/- 0.015 |
| Current + Direct-P2P + Phase 13 | 1021.30, 1017.50, 1017.37 | 1018.723 +/- 2.232 | 37.01, 36.88, 36.88 | 36.923 +/- 0.075 |

Compared with the historical cumulative build-9940 result of
`931.247 PP / 36.487 TG`, the current canonical stack is:

- **+9.394% PP512**
- **+1.197% TG128**

## Claim boundary

The current evidence supports this narrow performance statement:

> On two Radeon AI PRO R9700 GPUs running Qwen3.8-27B Q6_K with llama.cpp/HIP
> tensor split, Direct-P2P increased TG128 from 33.800 to 36.907 t/s. Adding an
> explicit F32 conversion at the Q6_K MMQ accumulation operation increased
> PP512 from 875.083 to 1018.723 t/s while leaving TG128 effectively unchanged.
> The complete stack measured 1018.723 PP512 and 36.923 TG128.

Do not generalize these results to other GPU architectures, CUDA, other ROCm
versions, quantization formats, model families, single-GPU operation, or
arbitrary prompt and batch sizes without independent validation.

## Separate MTP track

MTP is not part of the canonical baseline. Embedded-MTP produced strong
throughput results on Qwen3.8, but deterministic output diverged from ordinary
decode and followed a different EOS path. Those results remain a separate
correctness investigation until exact-token parity passes.

## Registered baseline

Register the following as the current Qwen3.8 Q6_K canonical stack:

`llama.cpp build 10434 + HIP Direct-P2P + Phase 13 Q6_K MMQ explicit F32 conversion`

- PP512: **1018.723 t/s**
- TG128: **36.923 t/s**

Retain current stock and current P2P-only as immutable comparison lanes. Label
the historical build-9940 `931 PP / 36.5 TG` binary as a cumulative P2P plus
Phase 13 research build.
