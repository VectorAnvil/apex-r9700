# Phase 6 Handoff

## Outcome

**Rejected.** Phase 6 evaluated only fused Q6_K `ffn_up`
`8704x1x5120` at twelve waves on two gfx1201 R9700 GPUs. All other dispatches
retain eight-wave behavior. The final finding is
`docs/apex-r9700/evidence/phase-06/Q6K_FUSED_TWELVE_WAVE_FINDING.md`.

## Established Evidence

- Both physical GPUs dynamically report gfx1201, wave32, and 1024 maximum
  workgroup threads; the 384-thread candidate is legal.
- Separate per-GPU `rocprofv3` capability outputs are retained. Availability
  is not evidence that a counter is usable.
- Runtime dispatch selection passed 36/36 witness records; profiler duration
  is not performance data.
- Compiler metadata reports fused eight-wave: 35 VGPR, 42 SGPR, 1792 B LDS;
  fused twelve-wave: 35 VGPR, 42 SGPR, 2816 B LDS. Both show zero private
  segment and zero VGPR/SGPR spills.

## Gate Status

- CPU-oracle: 36/36 passed.
- Unprofiled exact-slice timings: fused `ffn_up` was 1.0000x on device 0 and
  1.0033x on device 1, both stable but below the 1.01x requirement.
- Device-0 `attn_kv` additionally failed the fallback stability gate at 3.70%.
- Fail-closed occupancy bound: unavailable; required gfx1201 allocation and
  slot inputs were not exposed.
- Exact-symbol counter diagnostic: all four 120-second, nonmultiplexed
  `OccupancyPercent+FETCH_SIZE` captures timed out without a join; occupancy
  and read/total bandwidth are unavailable.
- E2E PP/TG: not eligible and not run.
- Final decision: reject. Demote sixteen waves because it shares the two-loop
  count and adds reduction work.

## Provenance And Safety

Task, candidate patch/build, dispatch, gate, code-object, scorecard, and
attempt hashes are in the immutable task/result artifacts. The task fingerprint
is `4542409930c19a8897fad3c35b4c41e066e6e8e14b76c4f863d25fcb777debff`.
Raw evidence lives in
`results_phase06_q6k_fused_nwarps12_20260808/`.

Registered Llama Lab assets, model files, services, known-good P2P diff,
`AGENTS.md`, and user `documents/` remain untouched. E2E registration is
allowed only after the exact local gate passes. Preserve all failures and stop
at Phase 6.
