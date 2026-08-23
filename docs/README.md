# Apex on Radeon AI PRO R9700 (`gfx1201`)

This directory is the durable index for Apex and llama.cpp optimization work on two Radeon AI PRO R9700 GPUs. It separates promoted production changes, validated but unpromoted candidates, rejected experiments, and raw evidence.

## Current production stack

The promoted Qwen3.8 Heretic Q6_K service uses llama.cpp build 10457 with:

- direct peer-to-peer tensor transfer;
- the Phase 13 Q6_K RDNA4 MMA float-conversion fix;
- the restored rocWMMA Flash Attention path for long-context prefill;
- the C11, C07, and C03c guarded FFN improvements;
- packed canonical-row Q6_K verification for MTP width 2;
- the recurrent natural-split and speculative KV padding guards.

See [CURRENT_PRODUCTION_STACK.md](CURRENT_PRODUCTION_STACK.md) for exact runtime gates, launch arguments, validation boundaries, and rollback guidance.

## Headline measured results

| Change | Workload | Baseline | Candidate | Result | State |
|---|---:|---:|---:|---:|---|
| Direct P2P | TG | 33.800 t/s | 36.907 t/s | +9.19% | promoted |
| Phase 13 cast | PP512 over P2P | 875.083 t/s | 1018.723 t/s | +16.41% | promoted |
| FA-1 rocWMMA restoration | PP4096 at 127K | 333.82 t/s | 543.01 t/s | +62.7% | promoted |
| FA-1 clean stability rerun | PP4096 at 127K | 333.911 t/s | 569.339 t/s | +70.51% | promoted |
| Packed N=2 canonical rows | MTP decode | 49.8535 t/s | 58.0286 t/s | +16.40% | promoted |
| DFlash width 3 | frozen parity suite | ? | 58.5701 t/s | 10/12 exact | research only |

Numbers are tied to their documented workload contracts; they should not be combined into a synthetic end-to-end percentage.

## Navigation

### Production and promotion

- [Current production stack](CURRENT_PRODUCTION_STACK.md)
- [Promotion candidate ledger](APEX_PROMOTION_CANDIDATE_LEDGER.md)
- [C11+C07+C03c production promotion](C11_C07_C03C_PRODUCTION_PROMOTION_20260819.md)
- [P2P and Phase 13 validation](QWEN38_P2P_PHASE13_VALIDATION.md)
- [Flash Attention long-context track](FLASH_ATTN_LONG_CONTEXT_TRACK.md)
- [FA-1 stability adjudication](evidence/fa-s1/FA1_STABILITY_ADJUDICATION.md)
- [FA-1 source transplant patch](evidence/fa-01/FA1_CURRENT_ROCWMMA_TRANSPLANT.patch)

### Correctness and model comparisons

- [Base, Heretic, and HuiHui Q6_K comparison](QWEN38_Q6K_BASE_HERETIC_HUIHUI_COMPARISON_20260819.md)
- [Obliterated-model MTP parity](QWEN38_Q6K_OBLITERATED_MTP_PARITY_20260819.md)
- [DFlash width-3 adjudication](DFLASH2_WIDTH3_ADJUDICATION_20260823.md)

### Forensic audit and rejected work

- [Phase 4?15 executive summary](APEX_PHASE_04_15_EXECUTIVE_SUMMARY.md)
- [Dependency map](APEX_PHASE_04_15_DEPENDENCY_MAP.md)
- [Evidence ledger](APEX_PHASE_04_15_EVIDENCE_LEDGER.md)
- [Gate audit](APEX_PHASE_04_15_GATE_AUDIT.md)
- [Minimum rerun plan](APEX_PHASE_04_15_MINIMUM_RERUN_PLAN.md)
- [Resurrection candidates](APEX_PHASE_04_15_RESURRECTION_CANDIDATES.md)

### Phase records

[PROJECT_STATE.md](PROJECT_STATE.md) is the chronological state ledger. `PHASE_*_HANDOFF.md`, `evidence/`, and `artifacts/` retain the detailed gate decisions and compact proof objects.

## Reproducing results

Start with [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md). Raw profiler databases, model files, binaries, server logs, and multi-gigabyte result directories are intentionally excluded from Git. Compact summaries, hashes, commands, patches, and sufficient validation evidence belong here.

## Scope and attribution

This work validates and adapts upstream projects; it does not claim authorship of third-party patches. In particular, the direct-P2P implementation is attributed to JohnTDI-cpu. Apex-specific work covers rebasing, isolation, gfx1201 validation, integration, and promotion decisions.

All architecture-specific paths are expected to fail closed outside their documented `gfx1201`, datatype, shape, and runtime gates.
