# Phase 7 Handoff

## Outcome

Phase 7 completed the graph-enabled registered PP512 baseline and exact Q6_K
MMQ attribution on both gfx1201 GPUs. It selected no kernel candidate and makes
no performance-gain claim.

## Baseline

Three independent normal-graph PP-only processes produced `avg_ts` values
`812.746964`, `815.693745`, and `817.701996`. Their relative spread was
`0.006074623`, passing the frozen 5% limit without outlier removal. No
rocprofv3, `GGML_CUDA_DISABLE_GRAPHS`, database mutation, candidate build, or
registered asset mutation occurred.

## Exact Attribution

The positional join found 496 host operations per device across six repeats,
with zero mismatched, ambiguous, or unmatched records. The Phase 2 PP Q6_K MMQ
family accounts for 85.0106% of summed dispatch duration. Exact dual-device
operation shares are:

| Operation | PP dispatch-duration share |
|---|---:|
| `ffn_up` | 19.197% |
| `ffn_gate` | 19.106% |
| `ffn_down` | 18.402% |
| `attn_qkv` | 8.110% |
| `attn_gate` | 5.036% |

The workload is actual `N=512`; template `128` is MMQ X tile width and the
boolean template argument is `need_check`, not output N or fusion.

## Resource Facts

Dynamic resource evidence records LDS 57,856 bytes, wave32, and 256-thread
workgroups. The two specializations use 229/230 VGPR and 27/30 SGPR; static
private segment and spills are zero. These are evidence for a later hypothesis,
not proof of a bottleneck or an expected speedup.

## Next Boundary

Phase 8 requires separate approval and a new immutable contract. The sole
proposed hypothesis is global gfx1201 Q6_K MMQ `mmq_y=64` versus `128`, while
keeping `mmq_x=128` and eight waves. It must dynamically query occupancy,
preserve correctness and variance gates, verify the exact attributed operation,
and pass a whole-PP gate. Fusion work remains later. Do not reopen the rejected
TG four/twelve-wave ideas.
