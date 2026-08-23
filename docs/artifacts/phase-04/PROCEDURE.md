# Phase 4 Procedure Notes

- `task.json` is the immutable sustained-idle v1 task. The original v0 task and
  its four pre-launch safety aborts remain in the ignored raw result root.
- The v1 idle procedure did not change the 5% activity threshold. It retained
  failed snapshots and waited for a passing snapshot before each process.
- `gate-manifest.json` contains the 36 correctness rows and fixed timing
  schedule. Raw stdout, stderr, and activity snapshots remain ignored.
- The original v1 finalizer failed on fused dimension representation and
  substring matching. `gate-result.json` was produced by a separately retained
  recovery parser using the original samples and frozen thresholds.
- No E2E workload or Llama Lab registration followed because the exact-slice
  gate failed.
