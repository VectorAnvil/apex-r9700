# Phase 8 Handoff

## Outcome

**Rejected as statically infeasible.** The global gfx1201 Q6_K MMQ
`mmq_y=64`, `mmq_x=128`, eight-wave candidate was stopped before candidate
build. Immutable task fingerprint:
`0857b439db45cfe5bc8c549e5354c5ae5cab58ddc4b49a91e97d25125f4bb1d7`.
The static gate decision is `reject_static_infeasible`.

## Static Proof

The pinned Q6_K WMMA writeback invariant is:

```text
nwarps * tile_C::I == mmq_y
```

On gfx1201 the approved values are `8 * 16 = 128`; therefore a constant-only
`mmq_y=64` change cannot compile. Changing to `nwarps=4` is not a repair: it is
explicitly outside the contract and would confound the comparison. A valid
64-row variant requires a new WMMA fragment/writeback or output/K partition
design, which was not authorized.

## Not Run

No candidate build, GPU workload, correctness run, timing run, occupancy or
bandwidth capture, counter pass, E2E comparison, database mutation, or asset
registration occurred. No performance gain is claimed.

## Next Boundary

Any next phase requires approval and a new immutable contract. The evidence-led
possibility is gate+up fusion feasibility from their exact combined 38.304% PP
share. `ffn_down` at 18.402% remains the Zinc-methodology alternate. Do not
integrate Zinc or any reference kernel yet.
