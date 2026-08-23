# Q6_K PP SwiGLU-to-D4 ffn_down Experiment

## Summary

**Rejected by the immutable Phase 12 exact-chain gate.** The default-off
gfx1201 candidate directly packed F32 SwiGLU inputs into the private
`block_q8_1_mmq` D4 layout and fed the unchanged Q6_K `ffn_down` MMQ. It was
correct, selected on both R9700s, preserved fallback and graph replay, and
slightly improved whole PP. It nevertheless made the isolated chain slower
on both devices and therefore is not registered.

- Exact chain, GPU 0: `1315.917 -> 1322.310 us`, `0.995165x`.
- Exact chain, GPU 1: `1337.133 -> 1346.927 us`, `0.992729x`.
- Whole PP512: `815.209 -> 819.466 tok/s`, `1.005222x`.
- Correctness: 18/18 CPU-oracle rows passed at the frozen NMSE `0.0005` limit.
- Decision: **rejected; selector remains default-off and unregistered**.

No sample was trimmed and no threshold was relaxed after seeing the result.

## Hardware And Software

- GPUs: 2 x AMD Radeon AI PRO R9700 32 GB, `gfx1201`, wave32.
- ROCm/HIP: `7.2.26015`; rocprofv3 `1.1.0`.
- llama.cpp base: `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`.
- Model: Huihui Qwen3.6 27B Q6_K, SHA-256
  `c03727f9fdfb7c8c02a8d7a0662563a0b0d7a6e192a03834a45f58300408b4ea`.
- Runtime: tensor split over both R9700s with the registered direct-P2P patch.

The candidate was built in an isolated result-local worktree. Registered
llama.cpp source, build products, model data, and the Llama Lab database were
not modified.

## Why This Target

Phase 12 feasibility joined the actual PP512 sequence
`SwiGLU -> D4 quantizer -> Q6_K ffn_down MMQ` across all 64 layers and both
devices. The chain represented 19.812% of summed PP dispatch time. Its
contiguous F32 `[8704,512]` intermediate modeled 35,651,584 avoidable bytes of
write/read staging per layer and device.

The selected operation was logical `M=5120,N=512,K=8704`. PP MMQ consumes a
private 144-byte D4 block containing 128 int8 values and four F32 scales; it
does not consume an ordinary GGML Q8_1 graph tensor.

## Optimization

The opt-in selector was:

```text
GGML_CUDA_Q6K_SWIGLU_D4_FFN_DOWN=1
```

The candidate added one wave32 workgroup per 128-value D4 block. Each lane
computes four baseline SwiGLU values, reduces four independent 32-value scale
groups, rounds to int8, and writes the exact D4 layout. A private MMQ entrypoint
then calls the existing Q6_K MMQ with the prequantized buffer and skips only
its ordinary activation quantizer.

Dispatch is fail-closed on gfx1201, Q6_K/F32 types, the exact PP512 shapes,
contiguity, views, graph topology, tensor names, and MMQ selection. All other
cases use the original SwiGLU and quantizer path.

## Correctness And Dispatch

| Gate | Result | Status |
|---|---:|---|
| Exact, fallback, selector-off CPU oracle | 18/18 across both GPUs | PASS |
| Synthetic dispatch witnesses | 6/6 | PASS |
| Exact candidate dispatch | 1 fused producer, 0 ordinary SwiGLU, 2 quantizers, 3 Q6_K MMQs | PASS |
| Exact baseline/fallback dispatch | 0 fused, 1 ordinary SwiGLU, 3 quantizers, 3 Q6_K MMQs | PASS |
| Registered PP dispatch | 128 fused calls per GPU | PASS |
| Normal graph capture/replay | 24/24 timed rows observed graph warmup | PASS |

The first registered trace selected zero fused calls because the generic
four-node fusion memory check treated the already-consumed gate/up input as
live. The implementation was corrected to validate the two nodes that really
execute at the fusion boundary, SwiGLU and down MMQ. A new trace then observed
exactly 128 fused calls on each GPU and 256 fewer ordinary quantizers.

## Results

### Exact Chain

| GPU | Baseline samples (us) | Candidate samples (us) | Speedup | Spread |
|---:|---|---|---:|---:|
| 0 | 1313.15, 1317.33, 1317.27 | 1320.70, 1323.57, 1322.66 | `0.995165x` | 0.318% / 0.217% |
| 1 | 1330.06, 1335.22, 1346.12 | 1341.85, 1349.32, 1349.61 | `0.992729x` | 1.201% / 0.576% |

Both rows are stable but miss the required `1.010x` speedup. This is the sole
failed promotion gate.

### Fallback

The 256-token nonmatching shape measured `1.001115x` on GPU 0 and `1.001145x`
on GPU 1, passing the frozen `0.99x` no-regression floor.

### Whole PP512

| Variant | Samples (tok/s) | Mean | Relative spread |
|---|---|---:|---:|
| Baseline | 815.442, 813.982, 816.202 | 815.209 | 0.272% |
| Candidate | 818.107, 821.187, 819.105 | 819.466 | 0.376% |

The throughput ratio is `1.005222x`, barely above the required `1.005x` whole-
PP floor. It does not override the failed exact-chain gate.

## GPU Resources

| Metric | Fused producer |
|---|---:|
| Code-object VGPR / SGPR | 18 / 18 |
| Runtime allocated VGPR / SGPR | 24 / 128 |
| LDS / private segment / runtime scratch | 0 / 0 / 0 bytes |
| VGPR / SGPR spills | 0 / 0 |
| Wave size | 32 |
| Workgroup / waves per workgroup | 32 threads / 1 wave |
| Waves per exact dispatch | 34,816 |

The median unprofiled fused-producer duration was about 51.48 us per device.
Logical F32-input plus D4-output bytes imply about 790 GB/s, but that is a
modeled byte rate, not an achieved hardware-bandwidth counter.

Achieved occupancy and `FETCH_SIZE` bandwidth are **unavailable**. Bounded,
nonmultiplexed rocprofv3 attempts on both gfx1201 devices aborted in HIP buffer
initialization with `std::out_of_range: unordered_map::at`, produced no
joinable trace, and were stopped by the 20-second guards. These values are not
reported as zero or inferred from Instinct limits.

## WHAT_DIDNT_WORK

Direct SwiGLU-to-D4 packing removed the intended F32 materialization and one
quantizer launch, but its per-128-value reductions, rounding, and D4 stores
cost more than the original SwiGLU plus quantizer in the isolated chain. The
stable regressions were 0.48% on GPU 0 and 0.73% on GPU 1.

The small whole-PP improvement is real enough to preserve but not strong
enough to waive the contract. It sits only 0.022 percentage points above the
whole-PP threshold and conflicts with the direct operation measurement. Do not
register this patch, present it as an optimization, or retry the same one-wave
D4 producer without a materially different mapping and a new immutable task.

The initial synthetic test also failed to model allocator reuse in the real
tensor-parallel graph. The resulting zero-dispatch trace is retained so future
fusion tests include registered-graph selection before whole-workload timing.

## Provenance

- Task fingerprint:
  `6dc70c86aeb4c05f1bfa6cc341e80450de9971c6f8adba72a66775298a6c04e7`.
- Candidate patch SHA-256:
  `0451d2fe82ddcf4cce337dc13c0b5ec2ad83d8e7e1304dc9c0e38c2000f29116`.
- Final result: `results_phase12_swiglu_d4_ffn_down_20260808/final-result.json`.
- Compact tracked evidence: `docs/apex-r9700/artifacts/phase-12/`.
- Raw builds, traces, logs, HSACO, and failed attempts remain under the ignored
  result root.
