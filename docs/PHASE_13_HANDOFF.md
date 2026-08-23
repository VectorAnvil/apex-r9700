# Phase 13 Handoff

Phase 13 accepted and registered the one-line Q6_K MMA float-conversion fix.
The registered Qwen3.6 27B Q6_K workload improved PP512 by `1.277663x` with
TG128 at `0.999895x`. The exact Q6_K kernel trace improved `1.617272x` with
identical dispatch counts, eight-wave geometry, and no spills.

Phase 14A is a short post-promotion re-baseline, not a candidate evaluation.
Use only the newly registered float-cast build and normal PP512 graph settings.
Refresh per-operation attribution, gate/up/down shares, Q6_K MMQ time and
resources/ISA, GPU0/GPU1 balance, and direct-P2P/all-reduce share. Preserve
explicit unavailable status for gfx1201 counters that remain unjoinable.

Only after Phase 14A should Phase 14B be selected. If Q6_K MMQ remains the
dominant actionable PP cost, freeze the paired RDNA4 `mmq_y=64,nwarps=4`
geometry as one matched candidate on top of the float-cast baseline. This is
structurally different from Phase 8's invalid Y64/eight-wave proposal because
`4 * tile_C::I(16) == 64`. Do not test Y64/W4 against the old kernel or mix it
with Stream-K, dispatch-threshold changes, or other arithmetic changes.

If refreshed attribution shows that Q6_K MMQ is no longer the leading
actionable cost, Phase 14B should select the new leader instead. Dense
gfx1201 Stream-K remains a later independent candidate, not the automatic next
experiment.

The prior tiled MMQ/communication-overlap feasibility proposal remains a later
backlog item; it was superseded as Phase 13, not rejected on technical grounds.
