# Phase 21D Handoff

Phase 21D sealed real top-32 distributions for the accepted target-tensor,
draft-layer, `n_max=12` configuration. Two recorder processes preserved exact
192-token parity and produced identical semantic payloads.

The 20-node DDTree point is worth a correctness prototype:

- Pure best-first 20: mean canonical depth `2.60 -> 3.20`, 24 rescued cycles,
  one cycle worse than linear.
- Chain-seeded 20: mean depth `3.14`, 22 rescued cycles, no cycle worse than
  linear.
- Best-first 12 is the efficiency reference: mean depth `3.02` with fewer
  modeled nodes per committed token than linear-12.

Phase 21E should be correctness-only. Implement parent-aware tree attention,
KV ownership, and Qwen35 recurrent/DeltaNet state in an isolated build. Use
chain-seeded 20 as the no-linear-regression reference and pure best-first 20 as
the higher-depth comparator. Every node must match a serial ancestor-prefix
oracle, and accepted-path state must remain identical through a continued
suffix before any performance work.

Do not use `n_max=15`, optimize kernels, or register the Phase 21C/21D build.
The ignored raw root is
`results_phase21d_dflash_top32_ddtree_20260809/`.
