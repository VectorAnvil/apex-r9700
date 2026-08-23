# Phase 5 Procedure Notes

- Revision 1 used an architecture-dependent integer template default for the
  fallback kernel. The eligible pilot proved a 32x4 launch, but the first
  excluded candidate trace timed out because host launch geometry and the
  gfx1201-compiled fallback reduction could disagree. No timing evaluation was
  accepted. Its task, patch, tools, and traces remain under the raw result root.
- Revision 2 replaced that mechanism with an architecture-independent
  `force_four_waves` boolean. The default branch again derives its wave count
  inside the kernel exactly as upstream does. Thresholds, cases, and schedule
  did not change.
- `dispatch-selection.json` contains 36 rocprofv3 witness records. Profiler
  durations are not evaluation evidence. Candidate launches were 32x4 only
  for the three selected tuples and 32x8 for every excluded tuple, including
  fused `ffn_up`.
- The unprofiled gate used normal graph settings and produced all 36 oracle
  rows plus 54 timing process groups. The v2 result failed the frozen
  per-device stability/regression policy, so E2E registration and PP/TG were
  not permitted.
