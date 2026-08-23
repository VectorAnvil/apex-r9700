# What Didn't Work

- Achieved gfx1201 occupancy and bandwidth remain unavailable. Corrected
  nonmultiplexed Phase 13 attempts using the same runtime failed on both
  devices inside rocprofv3 with `std::out_of_range: unordered_map::at`.
- Rocprof's runtime LDS column was zero. The host recorder supplies the exact
  57,856-byte dynamic allocation; static code-object group memory is zero.
- The diagnostic emitted no `ggml_ar_hip_kernel`. Its 1,280 runtime copy-buffer
  dispatches are separate and are not called all-reduce kernel or wall time.
- Profiled absolute durations were not compared with normal-graph wall time or
  the Phase 7 trace. Only fresh internal shares and ranking are used.
