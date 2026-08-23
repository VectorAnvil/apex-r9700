# Phase 21D DFlash Distribution and DDTree Replay

Phase 21D added a passive, environment-gated top-32 recorder to an isolated
copy of the accepted Phase 21C target-tensor/draft-layer build. Two independent
normal-graph requests matched the Phase 21C baseline for all 192 token IDs and
produced identical semantic distribution data.

Fifty full 12-position cycles were replayed host-only against the clean pinned
DDTree commit. Best-first 20 improved mean canonical depth from 2.60 to 3.20;
chain-seeded 20 reached 3.14 without a cycle-level regression. Rectangular
`3x4` and `4x3` policies lost overall.

This is a GO only for a separately frozen parent-aware Qwen35 recurrent/KV
correctness phase. It is not a GPU tree, performance result, or promotion.
Raw responses, distributions, stage records, trees, builds, code objects, and
failed paths remain in this ignored result root.
