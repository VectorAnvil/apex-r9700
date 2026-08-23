# Phase 4 Handoff

## Outcome

Phase 4 rejected `rdna4-q6k-nwarps4` as a global Q6_K `N=1` MMV
optimization. All 36 required exact-slice correctness rows passed, but the
candidate violated the frozen regression floor on both GPUs and did not improve
the exact high-share fused `ffn_up` case. The E2E gate remained closed, so no
full candidate build was registered and no Llama Lab throughput run occurred.

The template-based optimization record, including the complete "What Did Not
Work" analysis, is
`docs/apex-r9700/evidence/phase-04/Q6K_FOUR_WAVE_FINDING.md`.

## Exact-Slice Gate

The immutable v1 task used nine runtime-attributed per-device slices, both
physical R9700s, normal graph settings, deterministic inputs, and the GGML CPU
oracle at maximum NMSE `0.0005`. Three rounds used the fixed order
`D0-A,D1-A,D0-B,D1-B,D1-A,D0-A`, producing baseline `n=6` and candidate `n=3`
per device and case.

| Case | Device 0 | Device 1 | Decision evidence |
|---|---:|---:|---|
| `ssm`, `24x1x5120` | 0.9334x | 0.9333x | stable regression |
| `attn_k/v`, `512x1x5120` | 0.5261x | 0.4812x | stable regression |
| `attn_gate`, `3072x1x5120` | 1.0654x | 1.0692x | stable win |
| `M5120,K3072` | 1.0985x | 1.1128x | stable win |
| `attn_qkv`, `5120x1x5120` | 1.0641x | 1.0828x | device 0 unstable |
| `ffn_down`, `5120x1x8704` | 1.0499x | 1.0479x | device 0 unstable |
| `attn_q`, `6144x1x5120` | 1.0690x | 1.0810x | stable win |
| fused `ffn_up`, `8704x1x5120` | 0.9983x | 0.9983x | stable, no win |
| output head, `124160x1x5120` | 1.0087x | 1.0074x | device 0 unstable; no win |

The frozen policy required every case to remain stable and at least `0.99x`,
plus fused `ffn_up` at least `1.01x` on both devices. The stable small-shape
regressions independently reject the candidate. Fused `ffn_up`, exactly 29.842%
of the Phase 2 TG dispatch duration, independently fails the required win.

Three device-0 series contained isolated timing outliers: candidate `attn_qkv`,
baseline `ffn_down`, and baseline output head. They remain unstable and cannot
be used as wins. No cross-shape aggregate is reported.

## Procedure History

The v0 task's single-snapshot guard aborted four times before workload launch on
transient card0 readings of 7%, 9%, 6%, and 11% use with zero VRAM and no
`/dev/kfd` owner. The superseding v1 task retained the same 5% threshold but
preserved each snapshot and waited up to 60 seconds for a passing idle sample.
No performance or regression threshold changed.

The frozen v1 finalizer initially failed because fused harness parameters encode
`m=tokens,n=output_rows`, while the contract stores logical GEMM `M,N`; its
substring matcher also confused `M=512` with `M=5120`. A separate recovery
parser translated those representations only. It changed no samples,
correctness rows, thresholds, or decision logic. Both failures and the recovery
are preserved in the attempt history.

## Code Object

Fresh baseline and candidate HSACO were extracted from the Phase 4 builds and
confirmed as `hipv4-amdgcn-amd-amdhsa--gfx1201`, wave32.

- Baseline false/true: 256 threads, LDS 896/1792 bytes.
- Candidate false/true: 128 threads, LDS 384/768 bytes.
- VGPR/SGPR remain 26/26 false and 35/42 true.
- Private segment and VGPR/SGPR spills remain zero.

These facts confirm the intended specialization compiled; they do not explain
the measured regressions or wins.

## Provenance

- Task fingerprint: `12f47b5b9a632cf32e0c3ce95008fc531b1dc33617326f31b863c59b01f9218b`.
- Task SHA-256: `e53ef7aa22c11a6e30cbe68a4fda1bb804498743b5a117093292f5e53e9e2dfd`.
- Gate manifest SHA-256: `26ade33868c84ba87dd97775303f09fce2d5c5de4434d54dc846ab7e76309b8b`.
- Result SHA-256: `05b46d79b826879782be7a9add9905d997dab813f79c0de7c336654dfb3f3e53`.
- Candidate patch SHA-256: `8003dba642a37092a6fb1a9601d13d0c4dd1c7338a2d9c532bbbf4dd9026ad62`.
- Harness patch SHA-256: `1b822039c4835d8e6f2aa123e9324cdbceb79f46bd4578c0fc95805f55e96547`.
- Raw Phase 4 root: `results_phase04_q6k_nwarps4_20260808/`.
- Portable evidence: `docs/apex-r9700/artifacts/phase-04/`.

The registered Llama Lab source/build, model, database, services, and P2P patch
were not modified. The user's `AGENTS.md` change and untracked `documents/`
remain outside the Phase 4 commit.

## Recommendation

Do not promote or retest the global four-wave candidate. Its result history must
demote the same unconditional idea. A later phase may test one new
shape-and-fusion-gated dispatch that preserves eight waves for small and fused
cases while considering four waves only for the stable winning middle unfused
shapes. That is a new candidate requiring a new immutable contract; Phase 4 did
not implement it.
