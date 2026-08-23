# Phase 7 Finding: Exact Q6_K PP MMQ Attribution

## Summary

Phase 7 established an exact operation-level attribution for the graph-enabled,
two-R9700 prompt-processing Q6_K MMQ workload. This is not an optimization
experiment: no candidate kernel, gain claim, registration, or E2E comparison
occurred.

- Baseline PP512 median: `815.694` average tokens/s; spread `0.607%`
  (`812.747`, `815.694`, `817.702`), below the 5% procedure limit.
- Exact join: 0 unmatched and 0 ambiguous trace groups.
- Q6_K MMQ: `4,249,887,805 ns`, `85.011%` of total dual-device PP dispatch
  duration.
- Significant dense BLAS routing was not observed; the hotspot is custom GGML
  `mul_mat_q`, not a hipBLAS/hipBLASLt candidate.

## Attribution Proof

The isolated recorder captured 496 records per device for each capture state
(`none` and `active`). The two ordered sequences were identical after excluding
only sequence number and capture state. Phase 2 contains 2,976 Q6_K trace
calls per device, exactly six repeats of the 496-record sequence. Rows sorted
by numeric dispatch ID were positionally verified against the recorder's
normalized grid (`grid blocks * block`), workgroup, `need_check` bool, and the
Q6_K MMQ mangled symbol token.

This reconciles 4,800 `need_check=false` and 1,152 `need_check=true` calls.
Host records describe submissions, not replay counts.

| PP operation class | Dual-device PP share |
|---|---:|
| `ffn_up` | 19.197% |
| `ffn_gate` | 19.107% |
| `ffn_down` | 18.402% |
| `attn_qkv` | 8.110% |

## Semantics And Resources

The source correction is material: template `128` is MMQ X tile width, not
logical N; the template bool is `need_check`, not fusion. Actual PP logical N
is 512. The recorder emits `GGML_TYPE_Q6_K` as integer 14.

Static Q6_K code-object facts are wave32, 256-thread workgroup, zero static
group segment/private segment/spills, and 229 VGPR/27 SGPR (`need_check=false`)
or 230 VGPR/30 SGPR (`true`). The recorder's 57,856 B LDS is a dynamic dispatch
observation, distinct from static group-segment metadata.

Recorder/build/patch and code-object raw hashes are frozen in
`results_phase07_pp_mmq_attribution_20260808/{task-v1.json,analysis-task.json,evidence/}`;
the baseline attribution patch and Q6_K AMDHSA notes/HSACO are preserved there.

## Procedure Failures Preserved

The v0 baseline runner was rejected by the idle guard three times. Console
values were GPU use `49/80`, `48/79`, and `82/81` percent, each with 0 VRAM
use. The retained activity snapshots do not contain those high samples, so
these are console-only observations and are not evidence of persistent load.

The first v1 baseline attempt stopped before workload launch because the
parent output directory was missing. After that directory was created, the
unchanged hashed runner succeeded. No safety guard was bypassed and no process
was killed.

## Recommendation

Phase 8 should test one minimal gfx1201 Q6_K MMQ hypothesis: `mmq_y=64` versus
128, retaining `nwarps=8` and `mmq_x=128`. First dynamically query occupancy
resources, then require correctness, exact operation attribution, and total-PP
gate evidence. Do not infer occupancy from incomplete limits.

Reserve a gate+up fusion hypothesis (combined 38.304% PP) and a Zinc-informed
`ffn_down` path (18.402%) for later, separately frozen experiments. The Phase
7 evidence authorizes choosing a target; it does not authorize merging ideas
or assuming a speedup.
