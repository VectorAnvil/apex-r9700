# Phase 21C Artifacts

Phase 21C implements a default-off draft-specific split-mode override in an
isolated llama.cpp worktree. The target remains tensor split; DFlash uses layer
split and can access the target's shared Meta tensors.

The bounded accepted configuration is `n_max=12`. The full block-15 route is
not accepted because exact output parity fails reproducibly.

The tracked source diff is `draft-split-mode-v4.patch`; the final plumbing and
width contracts are `task-frozen-v4.json` and `task-frozen-v5.json`. Expanded
builds, logs, responses, failed attempts, and memory snapshots remain under ignored
`results_phase21c_dflash_split_mode_20260809/`.
