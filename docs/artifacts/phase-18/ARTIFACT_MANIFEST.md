# Phase 18 Artifact Manifest

Raw evidence root:

- `results_phase18_q6k_fused_k2_dot8_20260809/`

```text
8d0da7a673fa58f3d4eff35985f371089025aa56b59c23e95c88b7181d593519  final-result.json
b5d66279a673c5b12232faa329c717a81800479ae210c27274aca49c7d7819ec  README.md
52be3b9744d23fd7eb31c0f95a41a3b5768cb2e9c870a39c4e3ad0de6e4eab26  resource-summary.json
ff7d9c51a2e5122e4c2455e1129cfca319f71ddda05e33aa259980fa79cedd80  static-result.json
24504366fcca672e6b926831562533671b18371de26da38fb61bb168ba0ad801  task-frozen.json
ead05db4071fc49fe758a3165ae0bfff211ed3ec9d406e0631f5331d513c040a  WHAT_DIDNT_WORK.md
```

The raw root retains the isolated worktree/build, registered-state and K2
patches, dot8-on-K2 patch, build logs, object/fatbin/HSACO, full ISA, normalized
fused/unfused slices, and exact binary/source hashes. No GPU output exists
because the immutable static gate failed.
