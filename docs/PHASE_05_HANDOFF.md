# Phase 5 Handoff

## Outcome

Phase 5 rejected `rdna4-q6k-shape-gated-nwarps4`. The corrected candidate used
four waves only for unfused `3072x1x5120`, `5120x1x3072`, and
`6144x1x5120`, with eight-wave fallback everywhere else. All 36 runtime
dispatch records and all 36 CPU-oracle rows passed. The exact timing gate did
not pass, so no candidate was registered and no full-model PP/TG run occurred.

The complete negative finding is
`docs/apex-r9700/evidence/phase-05/Q6K_SHAPE_GATED_FOUR_WAVE_FINDING.md`.

## Implementation History

Revision 1 used an integer template default derived from the architecture
parameter table. Host-side instantiation and gfx1201 device compilation could
resolve it differently, allowing an eight-wave launch against a four-wave
fallback reduction. The eligible pilot produced the expected `32x4` launch,
then excluded `M=24` timed out under rocprofv3. No timing result was accepted.

Revision 2 used an architecture-independent `force_four_waves` boolean. The
default path again computes `nwarps` inside the kernel as upstream does. A fresh
immutable task linked the v1 task and patch without changing cases, thresholds,
or schedule.

## Exact Gate

The v2 task fingerprint is
`442950aead8acf273c90069fb38dd544bf9c533ba1198b94a555e11f391babac`.
The fixed three-round schedule produced six baseline and three candidate
process samples per device/case under normal graph settings.

| Case | Device 0 | Device 1 | Gate evidence |
|---|---:|---:|---|
| `24x1x5120` | 0.9950x | 0.9933x | pass |
| `512x1x5120` | 0.9878x | 0.9969x | device 0 unstable and below floor |
| selected `3072x1x5120` | 1.0558x | 1.0643x | device 1 candidate unstable |
| selected `5120x1x3072` | 1.0929x | 1.1096x | pass |
| `5120x1x5120` | 0.9974x | 0.9960x | pass |
| `5120x1x8704` | 0.9984x | 0.9995x | pass |
| selected `6144x1x5120` | 1.0686x | 1.0807x | device 0 baseline unstable |
| fused `8704x1x5120` | 0.9983x | 1.0000x | device 1 candidate unstable |
| output head `124160x1x5120` | 1.0001x | 1.0006x | device 1 baseline unstable |

The selected medians remained favorable, but the contract required every
selected row to be stable and at least `1.01x`, and every fallback row to be
stable and at least `0.99x`. No outlier was removed and no aggregate was used.

## Dispatch And Code Object

All 36 rocprofv3 witness records passed. The candidate launched `32x4` only
for the three selected tuples and `32x8` for all excluded tuples, including
fused `ffn_up`. Profiler durations were not used for evaluation.

The gfx1201 code object shows:

- unfused four: 128 threads, 384 B LDS, 26 VGPR, 26 SGPR;
- unfused fallback eight: 256 threads, 896 B LDS, 26 VGPR, 26 SGPR;
- fused fallback eight: 256 threads, 1792 B LDS, 35 VGPR, 42 SGPR;
- no private segment or VGPR/SGPR spills.

## Provenance

- V1 task fingerprint: `c94e743838b32917c5b35e469829af9e7691970d415f86a3bd10f476e9704fc9`.
- V2 task SHA-256: `838b60dc3d9b8d2d489b7a7be7b0e95c8aeef93cfbc7416eb869bb7bab98a59f`.
- V2 candidate patch SHA-256: `b052fb327bdcdb3f3ac79e4ea7345af5d7cd7b6539dc31f1f6eb6b4c43ff4a16`.
- Dispatch result SHA-256: `063bff0de9ed5ccb348a11ec333aa47bd786150a447390a4be6c4921b410a47b`.
- Gate result SHA-256: `3eec788372cd5e057fe40cf5b344f51e63eefa53954ae083c7235cff76a8f473`.
- Raw root: `results_phase05_q6k_shape_gated_20260808/`.
- Portable evidence: `docs/apex-r9700/artifacts/phase-05/`.

Registered assets and the user's `AGENTS.md` and `documents/` changes remain
untouched.

## Recommendation

Do not promote or rerun either four-wave candidate. Phase 6 may evaluate one
new fused-only 12-wave specialization for exact Q6_K `ffn_up`
`8704x1x5120`. Preserve eight waves for every other case and keep 16 waves as
a separate later experiment.
