# Rejected As Infeasible: Q6_K PP MMQ `mmq_y=64`

## Summary

Phase 8 evaluated one minimal hypothesis: change gfx1201 Q6_K PP MMQ from
`mmq_y=128` to `mmq_y=64` while retaining AMD WMMA, `nwarps=8`, and the
existing implementation. It was rejected as **infeasible before build or GPU
execution**.

This is a host-only source/invariant result. No candidate worktree, build,
HSACO, GPU workload, correctness run, timing run, occupancy measurement,
bandwidth measurement, or E2E comparison exists. Each absent datum is
`unavailable`, not zero or neutral.

## Invariant

The gfx1201 AMD WMMA path has the invariant:

```text
nwarps * tile_C::I == mmq_y, with tile_C::I = 16
8 * 16 == 128
```

The implementation uses `static_assert(nwarps * tile_C::I == mmq_y)`. Thus
`mmq_y=64` cannot be a constant-only specialization while retaining eight
waves and the existing tile mapping.

An eight-wave y64 remap would alter row/column ownership, accumulation, and
writeback. It is a new kernel design, not the frozen minimal hypothesis, and
was not silently widened into this phase.

## Evidence Status

| Evidence | Status |
|---|---|
| Source invariant review | pass: hypothesis infeasible |
| Candidate source/worktree/build/HSACO | unavailable |
| Dynamic occupancy limits | unavailable |
| CPU oracle | unavailable |
| Exact dispatch/tile witness | unavailable |
| Same-build local timing | unavailable |
| Achieved occupancy/bandwidth | unavailable |
| Whole PP gate / E2E | unavailable |

Phase 7 remains the target-selection evidence: custom Q6_K MMQ is 85.011% of
dual-device PP dispatch duration. It does not make an infeasible Phase 8
constant change executable.

## What Did Not Work

| Assumption | Outcome | Learning |
|---|---|---|
| `mmq_y=64` with existing AMD WMMA eight-wave mapping | infeasible | The tile product fixes y at 128. |
| Treating y64 as a parameter-only candidate | invalid | The required remap is a different kernel design. |
| `mmq_y=64,nwarps=4` follow-up | not authorized | It changes two coupled dimensions and should not be chased as a rescue variant. |

## Recommendation

Do not pursue y64+nwarps4. The next separately approved candidate should be a
gate+up fusion path, whose attributed combined PP share is 38.304%. A lower-risk
alternate is an `ffn_down` Zinc K-parallel investigation at 18.402%. Each needs
its own frozen source, correctness, exact operation attribution, resource, and
normal-graph PP gate.
