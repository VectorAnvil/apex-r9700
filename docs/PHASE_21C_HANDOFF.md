# Phase 21C Handoff

Phase 21C establishes a working target-tensor/draft-layer configuration in an
isolated build. Use:

```text
-sm tensor
--spec-type draft-dflash
--spec-draft-device ROCm0,ROCm1
--spec-draft-split-mode layer
--spec-draft-ngl all
--spec-draft-n-max 12
```

The `n_max=12` limit is binding: two independent 192-token requests passed
exact parity, while 1, 4, 8, and 15 failed on this prompt. Do not infer that a
smaller batch is safer, and do not use 15 for evidence until the verification
numerics are understood.

The next task is a passive pre-sampling top-32 recorder for all 12 available
future positions. It should use this same source/build boundary, preserve
target tensor split and DFlash layer split, and refuse incomplete cycles. Replay
the pinned DDTree policies host-only after the recorder artifact is sealed.

No registered assets were changed. The implementation patch is tracked at
`docs/apex-r9700/artifacts/phase-21c/draft-split-mode-v4.patch`.
