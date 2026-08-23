# Phase 9 Procedure

## Performed

Host-only analysis read the frozen Phase 7 attribution JSONL and exact sequence
summary, then checked pair adjacency, same-layer names, shared RHS, logical
shape, Q8_1 MMQ layout, grid/block, `need_check`, repeat count, and trace call
counts. Source feasibility and pinned source hashes were recorded separately.

## Not Performed

No GPU command, build, source integration, candidate worktree, HSACO capture,
CPU oracle, dispatch witness, timing run, counter pass, occupancy estimate,
achieved bandwidth calculation, or PP/TG E2E run was performed.

## Measurement Boundary

The intermediate upper bound of 71,303,168 bytes per layer/device is derived
from two exact F32 `8704 x 512` projection tensors, their stores, and their
later reads. It is modeled traffic only; it is not a profiler counter, achieved
bandwidth, or proof that every byte is avoidable.
