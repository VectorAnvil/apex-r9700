# Phase 21 Handoff

Phase 21 closes the DFlash/DDTree study as `feasible_with_major_work`. It does
not authorize a tree candidate or a performance claim.

The registered llama.cpp source already implements linear `draft-dflash`: it
extracts target features, runs the DFlash encoder and K/V injection, evaluates a
seed-plus-mask block, and currently retains only the top-1 candidate at each
future position. A local mostly-BF16 1.7B DFlash GGUF exists with hidden 5120,
block size 16, five draft layers, and target layers 2/17/32/47/62.

Official DDTree uses one DFlash block distribution to build a best-first tree,
flattens root plus nodes under an ancestor-only mask, calls the target once,
walks the longest target-selected child path, and compacts accepted state. This
is one target model forward, not one complete GPU step or cycle.

The difficult boundary is Qwen3.6 state. The target has 16 full-attention and
48 Gated DeltaNet/conv layers. Every flattened node must use its parent's
recurrent and convolution state; accepted K/V, SSM, conv, and five-layer target
features must then be committed in path order. Current Apex has no tree-parent
ops or tree rollback. Pinned Lucebox supplies a strong single-device design
reference, but its production multi-GPU layer-split target explicitly disables
tree verification. Apex tensor split therefore remains new work.

All proposed budgets are expected to cross from Q6_K MMVQ to MMQ: budget 15 is
root-inclusive `N=16`, budget 22 is `N=23`, while registered MMVQ stops at
`N=8`. Phase 13 should apply to those Q6 MMQs if a future trace confirms the
route. No timing is inferred from PP512.

The next phase is linear DFlash compatibility and instrumentation only. Use the
Phase 13 target unchanged, prove deterministic target output, capture top-K
coverage, stage timing, actual per-device memory, target/draft shapes, graphs,
and P2P/all-reduce. Replay fixed and adaptive tree policies offline before
authorizing parent-aware recurrent graph work.

Evidence:

- `docs/apex-r9700/evidence/phase-21/DFLASH_DDTREE_FEASIBILITY_STUDY.md`
- `docs/apex-r9700/artifacts/phase-21/`
- ignored raw root `results_phase21_dflash_ddtree_feasibility_20260809/`

No candidate was built, no GPU was launched, and the registered Phase 13
library remains `facd1354c4eba6afec9af0b22694e6ca11bf2b2bd976368158f225d6692c4311`.
