# Phase 13 Artifact Manifest

| File | Purpose |
|---|---|
| `task-frozen.json` | Immutable pre-candidate contract and scope exclusions |
| `correctness.json` | Both-device stock Q6_K CPU-oracle results |
| `pp512.json` | Untrimmed interleaved PP512 process results |
| `tg128.json` | Untrimmed interleaved TG128 process results |
| `final-result.json` | Gate decision, trace/resource/ISA summary, provenance, and failures |

Large kernel/HIP traces, code objects, disassembly, build trees, logs, and
counter-failure records are retained under
`results_phase13_q6k_mma_float_cast_20260808/` and are intentionally not
duplicated in Git.
