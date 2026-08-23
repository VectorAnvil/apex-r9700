# Phase 8 Static Infeasibility

## Decision

No `mmq_y=64`, `mmq_x=128`, eight-wave Q6_K candidate was built or run. The
pinned gfx1201 WMMA implementation statically requires the existing `mmq_y=128`
shape when `mmq_get_nwarps_device()` is eight.

## Pinned Source Facts

Source commit: `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`.

- `mmq_get_nwarps_host()` returns eight for AMD MFMA and otherwise `256 / warp_size`.
  gfx1201 has a physical wave size of 32, so the host launcher uses eight waves.
- `mmq_get_nwarps_device()` returns eight whenever `AMD_WMMA_AVAILABLE` is
  compiled.
- Q6_K WMMA selects `tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>`.
- `mmq_write_back_mma` contains the hard check
  `static_assert(nwarps*tile_C::I == mmq_y, "nwarps*tile_C::I != mmq_y")`.
  Thus the requested values evaluate to `8 * 16 != 64` and fail compilation.

## Why An Eight-Wave Remap Is Not Minimal

For `mmq_x=128`, the WMMA granularity is 32 and each two-wave group owns a
32-row output region while iterating all output columns. At `mmq_y=128`, four
such groups cover rows 0--127. Reducing only the y tile to 64 leaves the final
two groups with out-of-range row ownership. Keeping them active requires a new
partition of output or K work, plus an explicit reduction or changed fragment
and writeback ownership. That is a new kernel design, not a global Q6_K tile
selector experiment, and it exceeds the approved Phase 8 scope.

Changing the wave count to four would satisfy the old equation, but it is
expressly disallowed by the task and would confound the requested comparison.

## Scope Result

No source change, build, GPU workload, benchmark, correctness run, or
registered asset change occurred. Phase 8 is stopped before candidate build;
the baseline Q6_K MMQ remains `mmq_x=128`, `mmq_y=128`, eight waves.
