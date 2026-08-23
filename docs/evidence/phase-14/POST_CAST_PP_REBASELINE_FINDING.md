# Post-Cast PP Re-Baseline - Phase 15 Y64/W4 Authorized

## Summary

Phase 14A re-profiled the registered Phase 13 Q6_K float-conversion build on
the dual-R9700 Qwen3.6 27B Q6_K PP512 workload. It made no candidate change
and makes no new speedup claim.

Normal-graph PP was `1042.144 tok/s` across three independent processes with
`0.0492%` relative spread. In a separate profiler-induced no-graph capture,
Q6_K MMQ remained the largest actionable boundary at `77.7567%` of summed GPU
kernel duration. Paired RDNA4 `mmq_y=64,nwarps=4` is therefore authorized as
Phase 15, strictly on top of the promoted cast baseline.

## Workload And Method

- GPUs: 2 x AMD Radeon AI PRO R9700, dynamically identified as `gfx1201`
- Model: Huihui-Qwen3.6-27B Q6_K
- Source: llama.cpp `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`
- Registered HIP library SHA-256: `facd1354...c4311`
- Runtime: tensor split, direct P2P patch, flash attention, PP512
- Normal command: `-p 0 -n 0 -pg 512,0 -b 2048 -ub 512 -sm tensor -fa 1 -r 5 -t 12 -ngl 99`

Three normal-graph processes measured wall throughput. A separate host
recorder captured 496 Q6_K logical operations per device in both uncaptured
and captured states. A fresh six-repeat rocprof kernel trace was joined by
device, order, symbol, logical shape, geometry, layout, and `need_check`.
The join had zero unmatched or ambiguous rows.

Absolute profiler durations are not normal-graph wall time. Shares below use
the sum of all GPU dispatch durations in only the fresh no-graph capture.

## Normal-Graph Result

| Process | PP512 tok/s |
|---:|---:|
| 1 | 1041.862 |
| 2 | 1042.195 |
| 3 | 1042.375 |
| **Mean** | **1042.144** |

Relative process spread was `0.0492%`. The HIP runtime witness counted 514
each of stream capture begin/end, graph instantiate, and graph launch.

## Refreshed Attribution

Q6_K MMQ consumed `2,631,943,816 ns`, or `77.7567%` of summed GPU kernel time.

| Operation | Logical M/N/K | Calls | Summed time | Share |
|---|---:|---:|---:|---:|
| ffn_up | 8704 / 512 / 5120 | 768 | 590.523 ms | 17.4461% |
| ffn_gate | 8704 / 512 / 5120 | 768 | 588.459 ms | 17.3851% |
| ffn_down | 5120 / 512 / 8704 | 768 | 569.632 ms | 16.8289% |
| attn_qkv | 5120 / 512 / 5120 | 576 | 252.011 ms | 7.4453% |
| attn_gate | 3072 / 512 / 5120 | 576 | 157.011 ms | 4.6386% |
| ssm_out | 5120 / 512 / 3072 | 576 | 153.797 ms | 4.5437% |

The three FFN projections alone remain `51.6601%`. The next kernel family,
gated-delta-net, was `8.1901%`; Q8_1 MMQ quantization was `2.2171%`.

GPU 0 held `50.6490%` and GPU 1 `49.3510%` of Q6_K time. Absolute imbalance
was `1.2980%`, or `1.0263x` slow/fast. This does not point to tensor-split
imbalance as the leading PP boundary.

## Communication

The no-graph diagnostic recorded 1,280 `__amd_rocclr_copyBuffer` dispatches,
`5.466 ms` total and `0.1615%` of summed dispatch duration. No
`ggml_ar_hip_kernel` appeared in this capture. Copy-buffer and all-reduce
kernel time are reported separately and never summed into a wall-time share.

## Resources And ISA

The active post-cast Q6_K MMQ geometry remains 32 x 8 threads: 256 threads and
eight physical wave32 waves. The exact launch contract allocates 57,856 bytes
of dynamic LDS.

| Symbol | Static VGPR | Static SGPR | Runtime VGPR | Runtime SGPR |
|---|---:|---:|---:|---:|
| `need_check=false` | 252 | 27 | 256 | 128 |
| `need_check=true` | 253 | 30 | 256 | 128 |

Static group segment, private segment, runtime scratch, VGPR spills, and SGPR
spills are all zero. The promoted focused ISA contains 320 `v_cvt_f32_i32`,
93 `v_mul_lo_u32`, 40 scalar `v_mul_f32`, 148 dual F32 multiplies, and 32
integer WMMA instructions across the two active symbols. This retains the
Phase 13 post-cast codegen signature.

## Decision

Proceed to Phase 15 with one matched RDNA4 geometry candidate:

```text
mmq_y = 64
nwarps = 4
4 * tile_C::I(16) == 64
```

The candidate must be applied on top of the registered Phase 13 cast fix. It
must change host and device geometry together and must not include Stream-K,
threshold retuning, arithmetic changes, or the pre-cast kernel. This is not a
retry of Phase 8's invalid Y64/eight-wave proposal.

## What Didn't Work

- Achieved occupancy and bandwidth remain unavailable. Corrected bounded
  rocprofv3 attempts with this same promoted runtime aborted on both gfx1201
  devices in `std::unordered_map::at`. Static allocation was not substituted.
- Rocprof reported zero runtime LDS for the Q6_K dispatch. Dynamic LDS is
  therefore taken from the exact host launch recorder; static LDS remains a
  distinct code-object fact.
- The normal graph workload and profiler trace could not be combined safely.
  Graph wall throughput and no-graph attribution remain separate evidence
  lanes.
- The no-graph run did not witness `ggml_ar_hip_kernel`, so no all-reduce
  kernel duration is invented. Only observed copy-buffer time is reported.

## Evidence

Compact evidence is in `docs/apex-r9700/artifacts/phase-14a/`. Lossless traces,
attribution JSONL, build logs, activity guards, code object, and ISA remain in
`results_phase14a_post_cast_rebaseline_20260808/`.
