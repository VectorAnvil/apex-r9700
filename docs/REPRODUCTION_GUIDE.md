# R9700 reproduction guide

This guide defines the minimum contract for reproducing an Apex result. Exact commands vary by registered build and model path; copy them from the relevant phase handoff or evidence file instead of inferring flags.

## Environment record

Capture before testing:

```bash
git rev-parse HEAD
rocminfo | grep -m 2 -E 'Name:.*gfx1201'
rocprofv3 --version
sha256sum /path/to/llama-bench /path/to/llama-server
sha256sum /path/to/model.gguf
```

The reference system used ROCm/HIP 7.2.26015, rocprofv3 1.1.0, and two 64-CU Radeon AI PRO R9700 devices. Record the exact build, patch, shared-library hash, model hash, runtime gates, visible devices, and tensor split for every result.

## Benchmark contract

A comparison is valid only when baseline and candidate share:

- executable generation and all unrelated patches;
- model and multimodal projector;
- GPU visibility, tensor split, context, batch, microbatch, graphs, and Flash Attention setting;
- prompt/decode sizes, seed, sampler parameters, and speculative width;
- process lifecycle and warmup policy.

Use multiple fresh processes for long-context stability work. Report every sample, median, minimum, maximum, and spread. Never discard an outlier without documenting the adjudication.

## Shallow performance pass

Use the phase-specific `llama-bench` command and collect at least the registered PP512 and TG128 cells. A typical shape skeleton is:

```bash
HIP_VISIBLE_DEVICES=0,1 /path/to/llama-bench \
  -m /path/to/model.gguf \
  -ngl 99 -sm row --tensor-split 1,1 \
  -fa 1 -b 2048 -ub 512 \
  -p 512 -n 128 -r 5
```

Treat this only as a skeleton. The evidence document owns the authoritative switches and graph mode.

## Long-context Flash Attention matrix

At minimum compare the current native path and FA-1 with fresh processes for:

- PP512 at depth 0, 16K, 64K, and 127K;
- PP4096 at the same depths;
- TG128 at the same depths.

Preserve load logs and kernel traces outside Git. Commit the compact matrix, process spreads, command line, executable and library hashes, and any crash classification. The clean FA-1 adjudication used a 24-process rerun.

## Frozen deterministic parity suite

Run ordinary decoding and MTP from identical initial state with fixed seed and temperature zero. Compare committed token IDs, not rendered text. The registered suite contains 12 cases and includes the historically sensitive `list`, `forced128`, `forced256`, and long-limit cases.

For each case record:

- ordinary and speculative token-ID hashes;
- first divergent token index, if any;
- accepted and drafted token counts;
- elapsed time and generation throughput;
- MTP width and every active experimental gate.

Kernel arithmetic gates require bit-exact FP32 row output against independent canonical N=1 invocations. Epsilon agreement is insufficient for a deterministic parity claim.

## Profiling

Use bounded, allowlisted `rocprofv3` captures and retain kernel names, dispatch geometry, LDS, VGPR, SGPR, private segment, spills, and global-load behavior. Join trace rows losslessly; do not substitute static code-object facts for unavailable runtime counters.

On this ROCm stack, several exact `OccupancyPercent,FETCH_SIZE` captures timed out or failed to produce joinable counter data. This is a profiler limitation, not permission to claim achieved occupancy. Resource metadata plus stable production timings were sufficient to authorize FA-1.

## Evidence policy

Commit:

- Markdown findings and handoffs;
- compact CSV/JSON summaries;
- minimal source patches;
- hashes, commands, provenance, and gate decisions.

Do not commit model weights, binaries, server logs, profiler databases, generated build trees, credentials, or multi-gigabyte raw results. Keep failed and negative results in the ledger so future work does not repeat them.
