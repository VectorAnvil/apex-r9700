# Paired RDNA4 Y64/W4 - 0.9989x PP on 2 x R9700 / gfx1201

## Summary

Phase 15 tested a structurally valid paired RDNA4 MMQ geometry on top of the
registered Phase 13 Q6_K float-cast fix:

```text
baseline:  mmq_x=128, mmq_y=128, nwarps=8
candidate: mmq_x=128, mmq_y=64,  nwarps=4
invariant: 4 * tile_C::I(16) == 64
```

The change achieved its resource goal: Q6_K dynamic LDS fell from 57,856 B to
38,400 B (`-33.6%`) and the workgroup fell from eight to four wave32 waves.
It did not improve performance. Exact gate/up was about `0.993x`, ffn_down was
about `0.958x`, and a user-requested three-pair llama-bench diagnostic measured
`1042.838 -> 1041.707 tok/s` (`0.998916x`, `-0.108%`).

The candidate is rejected and was not registered. Stream-K was not included.

## Hardware And Software

- GPUs: 2 x AMD Radeon AI PRO R9700, dynamically reported as `gfx1201`
- CPU: AMD Ryzen 9 7900X
- ROCm: 7.2.26015; LLVM 22.0.0git
- llama.cpp: `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`
- Registered direct-P2P plus Phase 13 cast patch: `e1c70a...f5690`
- Model: Huihui-Qwen3.6-27B Q6_K
- Build: Release, `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1201`

Baseline and candidate were built independently in isolated worktrees. The
geometry-only patch SHA-256 is `dcf44182...df981`. Registered source, build,
model, and P2P state were not modified.

## Why This Was Tested

The post-cast Phase 14A trace showed Q6_K MMQ still consumed `77.7567%` of
summed PP GPU dispatch duration. Gate, up, and down alone were `51.6601%`.
The old kernel used 57,856 B dynamic LDS, 256 runtime VGPR, and eight waves.

Unlike Phase 8's invalid Y64/eight-wave idea, the paired geometry satisfies the
WMMA writeback contract. The hypothesis was that less LDS and smaller blocks
might outweigh the extra Y workgroups.

## Correctness And Dispatch

Correctness passed 96/96 baseline/candidate rows across both GPUs and three
repetitions. Q6_K covered M=63/64/65/127/128/129 around the tile boundary;
Q4_K M=65 provided cross-quant coverage because the minimal selectors are
global to RDNA4 MMQ. `test-backend-ops` enforced NMSE <= `5e-4` internally.

Both traces contained exactly 1,984 Q6_K MMQ launches, 992 per device, and no
non-Q6 MMQ launch. Baseline workgroups were 32x8; candidate workgroups were
32x4. Candidate Y grids doubled as expected while logical operation coverage
remained unchanged. Both variants retained 514 graph capture/end/instantiate/
launch calls.

## Fresh Resources

| Resource | Cast baseline | Y64/W4 |
|---|---:|---:|
| Dynamic LDS | 57,856 B | 38,400 B |
| Workgroup | 256 threads | 128 threads |
| Physical waves | 8 | 4 |
| Static VGPR, false/true | 252 / 253 | 252 / 253 |
| Runtime VGPR | 256 | 256 |
| Static SGPR, false/true | 27 / 30 | 30 / 31 |
| Runtime SGPR | 128 | 128 |
| Private / scratch / spills | 0 / 0 / 0 | 0 / 0 / 0 |

Fresh HSACO hashes were `acfc7077...edd40` baseline and
`27a5a2db...99405` candidate. Core Q6_K WMMA, int-to-F32 conversion, and F32
multiply counts were unchanged. Candidate instruction length increased by 92
lines for `need_check=false` and 87 for `true`.

The result is scientifically useful: reducing LDS did not lower VGPR pressure.
It slightly increased SGPR requirements and doubled the number of Y workgroups.

## Exact Results

All values are untrimmed means from three interleaved rounds; maximum relative
spread was `1.2075%`, below the `3.5%` limit.

| GPU | Operation | Baseline us | Candidate us | Speedup |
|---:|---|---:|---:|---:|
| 0 | gate/up shape | 864.777 | 871.303 | 0.992509x |
| 0 | ffn_down | 862.310 | 899.297 | 0.958872x |
| 1 | gate/up shape | 872.880 | 878.820 | 0.993241x |
| 1 | ffn_down | 865.823 | 904.783 | 0.956940x |

Combined speedup was `0.975425x` on GPU 0 and `0.974826x` on GPU 1. Both fail
the predefined `1.02x` exact gate. A paired profiler diagnostic was also flat
at `0.995263x` for all Q6_K MMQ dispatches.

## Whole PP Diagnostic

After the hard rejection, a user-requested normal-graph llama-bench diagnostic
was run to test whether stack scheduling reversed the microbench result.

| Variant | Three PP512 results | Mean | Spread |
|---|---|---:|---:|
| Baseline | 1044.257, 1041.636, 1042.620 | 1042.838 | 0.251% |
| Y64/W4 | 1039.435, 1040.631, 1045.056 | 1041.707 | 0.540% |

Whole PP was `0.998916x`, below the `1.005x` gate. TG128 was not run because
the PP-only candidate failed exact and whole-PP gates. Phase 16 instead begins
a separate TG optimization search.

## What Didn't Work

- A 33.6% LDS reduction was insufficient because VGPR allocation remained at
  256 runtime registers and the smaller tile doubled Y workgroups.
- ffn_down regressed by about 4.2-4.5%; gate/up was slightly slower. All rows
  were stable, so there is no variance-based rescue.
- Initial tail correctness selected cases absent from the stock generated test
  list. Identical test-only cases in both worktrees produced the accepted
  96/96 result.
- Initial microbench attempts added cases only to the evaluation list; perf
  mode has a separate list. Identical perf-only cases fixed the harness.
- Several partial trace/PP attempts stopped on the idle guard because the
  previous process had not cooled down. Accepted attempts used 15-25 seconds
  and exclude all partial data.
- Rocprof reported zero runtime LDS. Fresh host launch instrumentation proved
  57,856/38,400 B; static code-object group memory remains separately zero.
- Achieved occupancy and bandwidth remain unavailable because corrected gfx1201
  counter attempts on this runtime abort inside rocprofv3 `unordered_map::at`.
- Stream-K was deliberately not tried as a rescue or bundled change.

## Decision And Next Step

Reject Y64/W4 and demote this global geometry for the registered post-cast
Q6_K workload. Do not register or retest it with minor launch variations absent
new evidence. Preserve Stream-K as a later PP candidate, not Phase 16.

Phase 16 switches to TG and first audits the registered N=1 Q6_K MMVQ K loop
for an actual K2/K4 software-pipeline/codegen gap. No GPU candidate is allowed
unless the static audit finds one.
