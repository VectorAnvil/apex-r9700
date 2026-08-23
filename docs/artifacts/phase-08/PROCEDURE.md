# Phase 8 Procedure Record

This package records the static rejection of the proposed global gfx1201 Q6_K
MMQ `mmq_y=64` experiment. The request fixed `mmq_x=128` and eight waves.

The pinned WMMA writeback implementation requires
`nwarps * tile_C::I == mmq_y`. On gfx1201, the relevant values are eight waves
and a 16-row output tile, which requires `mmq_y=128`. A constant-only
`mmq_y=64` candidate therefore cannot compile with the existing ownership
mapping under the requested launch shape.

No source candidate, build, benchmark, GPU workload, occupancy query, counter
collection, code object, or HSACO exists for this rejected proposal. The P2P
baseline patch is identified by SHA-256
`36edac6b545c1e244ead573a466eb8f581337c86163d0fde91ce6a2fd04e94e9`.

The immutable `static-task.json` and `static-gate-result.json` are included in
this directory. The frozen decision is `reject_static_infeasible`.
