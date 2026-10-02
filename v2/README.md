# APEX V2 experimental research snapshot

Published October 2, 2026; measurements through September 30. This is a separate experimental source and evidence export for dual Radeon AI PRO R9700 / gfx1201. The August release in the repository root remains frozen.

V2 uses pinned llama.cpp b11211, commit `d7fb90e8e2494b2908934d956a3202fd60152ee0`, not whatever upstream master contains today. The model file is Qwen3.8-27B-Heretic-Q6_K; its GGUF architecture is `qwen35`. Measurements use two GPUs, Q8_0 K/V, tensor split 1,1, a 262144-token allocation, batch 2048 and microbatch 512.

The combined grouped/checkpoint/prefill/Q6/peer-copy build was selected for a reversible local application trial on September 30. This export is **not a production promotion or a claim of general accuracy parity with Core**. It does not change a running service.

## Measured combined curve

Control already contains original grouped attention, canonical paired checkpoints, prefill scheduling, Q6 mul24/two-row verification and MTP depth 2. Candidate adds large-message peer-copy AllReduce. Both process orders were run; rates pool equal work across those orders.

| Context tokens | Control PP tok/s | Candidate PP tok/s | PP change | Control TG tok/s | Candidate TG tok/s | TG change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 1129.64 | 1235.94 | +9.41% | 67.14 | 67.17 | +0.05% |
| 32768 | 1055.84 | 1153.11 | +9.21% | 66.16 | 64.93 | -1.86% |
| 65536 | 907.69 | 977.10 | +7.65% | 63.68 | 63.21 | -0.74% |
| 131072 | 737.07 | 782.51 | +6.17% | 59.67 | 59.36 | -0.52% |
| 196608 | 589.30 | 615.38 | +4.43% | 53.53 | 53.57 | +0.06% |
| 261632 | 490.00 | 508.80 | +3.84% | 48.53 | 48.32 | -0.43% |

PP measures growing-prefix processing; only the first 8K point is cold. It is not full cold 262K prompt throughput. TG uses cached 128-token windows and matched speculative work. These are benchmark rates, not guaranteed conversation or tool-workflow rates. Peer copy improves prefill with a small decode tradeoff; it does not provide another generation gain.

Evidence: [aggregate including both orders](evidence/peer-model-summary.json), [peer-copy report](reports/peer-copy.md). Historical original grouped attention achieved 24.68 -> 40.93 tok/s near 262K (about 66%) in a different fixture. Do not add component percentages or compare absolute rates across unmatched fixtures.

## Retained improvements and accuracy limits

- **Grouped attention:** expanded forced-token testing covered 3072 positions, with 32 changed top-token choices, 8 recorded-token losses and 10 gains. No top-token changes occurred among 839 active-grouped Core predictions with probability at least 0.9. These are sample results, not a general confidence guarantee.
- **Capability:** thinking-off testing found a real additional grouped math failure: an age answer changed from 13 to 16 at 128K, reproduced cold. A prose answer counted three entries as two. Many failures were shared with Core. Dedicated tool, retrieval and code suites passed equally in the reported samples. Thinking-on testing found no regression among completed paired answers at 128K and 260096; budget-limited cases remain documented. Recommendation for original grouped: **CONTINUE TESTING**, not automatic promotion. [Expanded evaluation](reports/grouped-parity.md).
- **Checkpoint coverage:** a deep edit's reprocessing fell from 65536 to 28672 tokens, about 137.4 -> 63.6 seconds in its matched experiment. Ordinary repeats retain a 512-token tail. Up to 72 paired target/draft checkpoints use about 10.5 GiB of host RAM, not GPU VRAM. [Cache report](reports/cache-coverage.md).
- **Prefill scheduling:** static spill slots fell from 146 to 50; measured growing-prefix PP improved about 2-4%. [Scheduling report](reports/prefill-scheduling.md).
- **Q6 mul24 plus two-row verification:** TG improved another 7.38-9.81% over the matched scheduling/MTP2 control across the curve, with full first-step logits, work counts and completed tasks matching that already-grouped reference. [Q6 report](reports/q6-stack.md).
- **Peer-copy compatibility:** 42 paired model cases, 52 history comparisons and 24 completed tasks passed, with exact messages, full first-step logits and work counts. The completed tasks generated 5157 tokens without truncation. These checks do not repair original grouped-versus-Core differences. [Task gate](evidence/peer-completed-tasks.json).
- **Transport tracing:** all traced chunks of the observed 5 MiB collectives used the copy engine. Smaller calls often used shader copies. Profiled timing was excluded from performance claims. [Routing summary and limitations](evidence/peer-routing-summary.json).

## Source reproduction

Six patches reconstruct the frozen grouped/checkpoint source and separately committed additions. They exclude experiment orchestration, machine-local launchers, model weights, binaries, raw server logs, private conversations and profiler databases. Local commit IDs identify provenance; they are not promised to resolve on upstream GitHub.

1. Frozen V2 grouped/checkpoint baseline, including retained Core ports and disabled optional older paths.
2. Prefill scheduling, separately dispatched by a runtime gate.
3. Bounded Q6 integer multiplication (`APEX_V2_Q6_MUL24`).
4. Two-row canonical Q6 verification (`APEX_V2_Q6_VERIFY_ROWS2`).
5. Large peer-copy AllReduce (`APEX_V2_AR_PEER_COPY`).
6. Restrict peer copies to the measured BF16-input / FP32-sum contract.

From this repository's root, use a new isolated checkout and build directory:

```bash
git clone https://github.com/ggml-org/llama.cpp.git llama-apex-v2-experiment
git -C llama-apex-v2-experiment checkout d7fb90e8e2494b2908934d956a3202fd60152ee0
python3 v2/scripts/apply-series.py llama-apex-v2-experiment --check
python3 v2/scripts/apply-series.py llama-apex-v2-experiment
bash v2/scripts/build-rocm.sh llama-apex-v2-experiment build-apex-v2-experiment
```

The apply script checks the pinned revision, clean checkout, hashes and complete series applicability before editing sources. [Source manifest](manifests/source.json) records provenance and compile gates. [Trial configuration](configs/trial.json) records arguments and runtime gates with placeholder paths. Set library paths to the isolated build's `bin` directory. The configuration is data, not a service installer.

The measured binary was linked from frozen baseline objects plus separately rebuilt scheduling, Q6 and AllReduce objects. The publication pass verified all six patches apply and 28 reconstructed source files match the recorded components. [Source validation](manifests/source-validation.json).

**A fresh full build from this portable recipe has not been compiled or benchmark-qualified in this publication pass.** Compiler and build differences can affect numerical behavior and speed. [Measured binary hashes](manifests/measured-binaries.json) document the original artifacts, not guaranteed hashes of fresh builds. Recheck numerical behavior, checkpoint history, complete tasks and the full performance curve before replacing a working build.

Old small-message Direct-P2P and fused-boundary gates are off. New large peer copy requires peer access between the selected devices and has only been qualified in the recorded NO_VMM configuration. The recipe uses an existing ROCm 7.2 installation; it makes no driver, firmware, system library or global environment changes.

## Negative results and remaining work

Core-query quantization reduced numerical differences but worsened held-out likelihood. Full FP32 P.V was 58-80% slower at the attention-operator level with mixed accuracy. Higher-precision merge and partition variants did not establish a consistent quality fix. Extra packed-Q8 conversion, smaller grouped tiles and most AllReduce threshold/chunk/ring/block changes did not yield a useful full-model improvement. Native FP8 remains an operator-level lead, not a validated end-to-end KV format.

Remaining leads include a bounded small-message AllReduce test, transfer overlap, prefill register-pressure redesign, checkpoint transfer batching and small cached append latency. None is an established additional gain. Targets for future rewrites are not measured forecasts.

## Research record

- [Upstream comparison](reports/upstream-comparison.md) and [four-build benchmark](reports/four-build-benchmark.md)
- [Retained August/Core ports](reports/core-integration.md)
- [Grouped capability evaluation](reports/grouped-parity.md)
- [Checkpoint coverage](reports/cache-coverage.md)
- [Prefill scheduling](reports/prefill-scheduling.md), [MTP2 interaction](reports/schedule-mtp2.md), [Q6 combination](reports/q6-stack.md)
- [AllReduce alternatives](reports/allreduce-tuning.md) and [peer-copy validation](reports/peer-copy.md)
- [Consolidated September 30 review](reports/discovery-review.md)
- [Evidence export provenance](manifests/evidence-provenance.json)

Reports retain historical serving-state statements. The combined trial was selected after the consolidated review; earlier restoration statements describe the end of those tests. Publication does not assert current live service health. Raw local artifact paths in reports are provenance references; excluded artifacts are not downloadable through this repository.

The source derives from llama.cpp/ggml. The older optional Direct-P2P implementation is attributed to [JohnTDI-cpu](https://github.com/JohnTDI-cpu/llama-hip-p2p-allreduce). APEX changes and validation were developed with AI assistance under human direction. No upstream PR is implied by this export.
