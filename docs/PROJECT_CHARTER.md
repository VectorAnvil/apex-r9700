# Apex gfx1201 + llama.cpp + Llama Lab

## Mission

Adapt AMD-AGI/Apex into an autonomous optimization environment for the existing dual-Radeon AI PRO R9700 llama.cpp inference stack. Apex is the research and optimization engine. Llama Lab remains the authoritative end-to-end build and benchmark referee.

Target: 2 x R9700 32 GB (`gfx1201`), Ubuntu 24.04, ROCm 7.2, llama.cpp HIP with direct P2P/AllReduce, Huihui Qwen3.6-27B Q6_K, independently controlled MTP, and separate PP/TG evaluation. Do not replace production llama.cpp with another serving engine.

## Target Lifecycle

```text
real workload -> ROCm profile -> hotspot -> isolated candidate build
-> correctness -> microbenchmark -> isolated full rebuild
-> Llama Lab comparison -> retain or reject
```

A microbenchmark improvement is only a candidate. Validation requires a real-workload improvement beyond noise without correctness, memory, or stability regressions.

## Non-Negotiable Constraints

- Never overwrite production binaries, patch production source, or modify model/GGUF files.
- Keep the known-good P2P patch unchanged unless it is the explicit experiment target.
- Use isolated pinned worktrees/builds and immutable experiment manifests.
- Candidate code cannot alter references, tolerances, workloads, repetitions, timers, expected output, scoring, baselines, or thresholds.
- Preserve raw measurements, variance, build/configuration identity, and patch identity.
- For compiled GPU candidates, preserve a per-specialization resource and
  runtime-efficiency scorecard: wave/workgroup size, VGPR, SGPR, LDS, private
  memory, spills, occupancy, and achieved bandwidth. Label every field as
  `static_code_object`, `derived_bound`, `runtime_counter`, or `unavailable`;
  never substitute Instinct assumptions for dynamically discovered gfx1201
  capabilities.
- Publish an optimization finding under `docs/apex-r9700/evidence/` for every evaluated candidate, including rejected, incorrect, neutral, and unstable experiments; state what did not work so failed ideas are not repeated.
- Do not hard-code users, paths, device IDs, or future control endpoints.
- Model `gfx1201` truthfully as RDNA4; do not clone CDNA/Instinct metadata or advice.
- Keep the Apex fork narrow and friendly to upstream synchronization.

## Responsibilities

- **Apex:** agents, optimization state, kernel grading, knowledge, reflection, trajectories, and reporting.
- **ROCm adapter:** profile capture, canonical PP/TG measurements, and source-map evidence.
- **llama.cpp adapter:** isolated worktrees/builds, correctness harnesses, and candidate promotion.
- **Llama Lab:** build/profile/model registration, serialized E2E benchmarking, parsing, comparison, and persistence.
- **Magpie:** standalone kernel compile/performance comparison where useful, not E2E authority.

## Phased Delivery

Each phase ends with tests, exact revisions, updated state, a handoff, an
optimization finding for every attempted candidate, a standalone next-phase
prompt, and a stop.

1. Phase 0: feasibility and architecture audit.
2. Phase 1: truthful `gfx1201` foundation and smoke test.
3. Phase 2: llama.cpp profiling and source mapping.
4. Phase 3: one valid real-kernel optimization loop.
5. Phase 4: exact runtime-slice gate for the first kernel candidate.
6. Phase 5: exact shape-gated follow-up candidate.
7. Phase 6: exact fused-hotspot kernel candidate.
8. Phase 7: durable repeated-experiment lifecycle.
9. Phase 8: Vivie/Bonsai/Mini-AWS research orchestration.

## Success

An agent can receive a real Qwen Q6_K hotspot, experiment safely on `gfx1201`, independently reject incorrect candidates, and prove through Llama Lab whether the real dual-R9700 workload improves.
