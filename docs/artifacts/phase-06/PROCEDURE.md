# Phase 6 Procedure

The immutable task is `task.json` with fingerprint
`4542409930c19a8897fad3c35b4c41e066e6e8e14b76c4f863d25fcb777debff`.

## Capability discovery

Recorded commands were:

```text
rocminfo
rocm-smi --showproductname --showuniqueid --showmeminfo vram --showuse --json
HIP_VISIBLE_DEVICES=<0|1> rocprofv3 --list-avail
```

Both agents dynamically reported gfx1201, wave32, and a 1024-thread workgroup
limit. The 384-thread candidate was legal. Dynamic input did not expose enough
allocation or slot detail for a combined occupancy bound.

## Counter diagnostic

The frozen counter command prefix was:

```text
rocprofv3 --kernel-trace --pmc OccupancyPercent,FETCH_SIZE --mangled-kernels
```

The first `counter-pass/baseline/d0` invocation used a relative binary path
that became invalid under the runner working directory, so it launched no
workload. That procedure failure is preserved in `attempts.jsonl`; the
corrected capture did not alter the task or timing gate.

The exact nonmultiplexed captures below each timed out at 120 seconds without
a joinable counter result:

```text
baseline device 0: OccupancyPercent,FETCH_SIZE
baseline device 1: OccupancyPercent,FETCH_SIZE
candidate device 0: OccupancyPercent,FETCH_SIZE
candidate device 1: OccupancyPercent,FETCH_SIZE
```

Therefore achieved occupancy and achieved read/total bandwidth are
`unavailable`, not estimates. Exact dispatch/correctness/timing evidence is in
`dispatch-selection.json` and `attempts/p6-fused12-a/`; static resource facts
are in `resource-scorecard.json` and `evidence/code-object.json`.
