# Phase 5 Portable Evidence

Decision: `reject`.

This directory contains bounded, reviewable evidence for the shape-gated
four-wave Q6_K MMV experiment. `task-v1.json` and `candidate-v1.patch` preserve
the incorrect architecture-dependent template-default attempt. `task.json`
and `candidate.patch` describe the corrected boolean specialization.

The corrected result has 36/36 dispatch-selection records and 36/36 CPU-oracle
rows, but its exact timing gate failed. `e2e.json` therefore records that no
candidate registration or PP/TG run was permitted.

Raw builds, binaries, gfx1201 HSACO, rocprofv3 CSV, activity snapshots,
stdout/stderr, and process-level timing directories remain under the ignored
`results_phase05_q6k_shape_gated_20260808/` root.
