# Apex R9700

Reproducible llama.cpp patches, runtime configuration, and benchmark evidence for Qwen3.8 Q6_K on two AMD Radeon AI PRO R9700 GPUs (`gfx1201`).

This repository contains the actual promoted source delta. It is based on llama.cpp commit [`4695f001fece1660d8bb1b3748f50726ddcc100b`](https://github.com/ggml-org/llama.cpp/commit/4695f001fece1660d8bb1b3748f50726ddcc100b), build 10457.

## Headline results

| Change | Workload | Baseline | Promoted | Result |
|---|---:|---:|---:|---:|
| Direct P2P | TG128 | 33.800 t/s | 36.907 t/s | +9.19% |
| Phase 13 Q6_K F32 conversion | PP512 over P2P | 875.083 t/s | 1018.723 t/s | +16.41% |
| FA-1 rocWMMA restoration | PP4096 at 127K | 333.911 t/s | 569.339 t/s | +70.51% |
| Packed canonical N=2 rows | MTP decode | 49.8535 t/s | 58.0286 t/s | +16.40% |

These measurements have different workload contracts and must not be multiplied into a synthetic end-to-end gain.

## What is included

- `patches/0001-promoted-heretic-stack.patch`: Direct P2P, Phase 13, FA-1, guarded FFN paths, canonical-row correctness, recurrent natural-split handling, speculative KV guard, and backend tests.
- `patches/0002-packed-n2-mmvq.patch`: the promoted concurrent packed N=2 Q6_K MMVQ optimization.
- `scripts/apply-series.sh`: verifies the exact upstream base and applies the ordered series.
- `scripts/build-rocm.sh`: reproduces the release ROCm/gfx1201 build configuration.
- `scripts/serve-heretic.sh`: portable production-form Qwen3.8 Heretic launcher.
- `configs/production.env.example`: model, projector, binary, hash, and service configuration.
- `evidence/n2-packed/`: compact kernel, parity, isolation, and performance evidence.
- `docs/`: the complete Apex R9700 research record, including negative results.

Model weights, binaries, profiler databases, raw server logs, credentials, and multi-gigabyte result directories are intentionally not included.

## Quick start

```bash
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout 4695f001fece1660d8bb1b3748f50726ddcc100b

./scripts/apply-series.sh ./llama.cpp
./scripts/build-rocm.sh ./llama.cpp ./build

cp configs/production.env.example production.env
# Edit production.env with local model and projector paths.
./scripts/serve-heretic.sh ./production.env ./build
```

The patches are intentionally gated. Generic behavior remains active unless the documented environment variables and validated shapes are selected.

## Production configuration

The promoted lane is:

- two Radeon AI PRO R9700 GPUs;
- ROCm 7.2;
- Qwen3.8-27B Heretic Q6_K with its matching BF16 multimodal projector;
- 262144 context, parallel 1, batch 2048, microbatch 512;
- tensor split `1,1`, Flash Attention enabled, Q8_0 K/V cache;
- embedded MTP with `n_max=2`.

See [docs/CURRENT_PRODUCTION_STACK.md](docs/CURRENT_PRODUCTION_STACK.md) for the exact gates and promotion boundaries.

## Correctness boundary

The promoted correctness claim is width-2 MTP on Heretic Q6_K. The packed path passed 28/28 bit-exact kernel cases, 120/120 expanded prompts, repeated cold-start checks, and long-depth gates. DFlash width 3 remains 10/12 exact and is research-only.

## Attribution

The HIP Direct-P2P implementation was authored by [JohnTDI-cpu](https://github.com/JohnTDI-cpu) in [`llama-hip-p2p-allreduce`](https://github.com/JohnTDI-cpu/llama-hip-p2p-allreduce). This project rebased, isolated, validated, and integrated it; it does not claim authorship of that implementation.

The FA-1 path adapts the earlier llama.cpp rocWMMA Flash Attention design to current source. Source and benchmark provenance are retained in `docs/` and `manifests/`.

## Repository roles

- [`VectorAnvil/apex-r9700`](https://github.com/VectorAnvil/apex-r9700): public, reproducible production patch stack.
- [`VectorAnvil/Apex`](https://github.com/VectorAnvil/Apex): full experimental branch history and Apex profiling-tool development.
- [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp): upstream inference engine.

## Status

The patch series is frozen to the recorded base. Rebasing to newer llama.cpp revisions requires rerunning arithmetic, parity, long-context, tensor-parallel, and performance gates.
