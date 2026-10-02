> Historical research record. Statements about the selected service are as of this report, not current deployment status. See [the V2 index](../README.md). Local raw evidence, private runtime files and logs are not included; local artifact paths are provenance references, not downloadable links.

# APEX V2 discovery review, 2026-09-30

Review scope: finish the current large-message peer-copy AllReduce lead, its required confirmation and compatibility checks, then pause new optimization work for Adam's review. The known-good source and binaries remain unchanged. Separate experimental worktrees retain clean kernel commits. No usage reset or host/dependency change was made.

## What the reference already contains

V2 began from pinned upstream llama.cpp b11211/d7fb90e8 on September27, not a continuously moving upstream checkout. Core retained Phase13 Q6_K MMQ conversion, canonical packed projections, eight-wave-equivalent reduction on four physical waves for selected shapes, shared gate/up Q8 preparation and fused SwiGLU D4 preparation. Packed attention, recurrent natural splitting, the KV boundary and strict-prefix/live-draft checkpoint controls were already in the integration reference. The disabled August rocWMMA attention implementation was not transplanted.

The September27 retained Core ports improved TG29.4%/23.8%/20.6% at64K/128K/192K versus that round's pre-port V2 reference. PP improved18.5%/13.7%/10.8%. These historical component gains are already inside the later references; they are not additional gains to add to this week's grouped/Q6/transport percentages. Old small-message Direct-P2P and fused boundaries were optional and were left off after their history-dependent PP cost. Source: AUGUST_INTEGRATION_20260927.md and subsequent P2P/history reports.

## Measured improvements before the current transport experiment

| Change | Measured benefit | Correctness/checkpoint evidence | Status |
| --- | --- | --- | --- |
| Original grouped attention | Historical near262K24.68 ->40.93 TG, about66%; later128-token curve23.56 ->38.04,61.5% | Expanded3072 positions had32 top-token changes,8 recorded-token losses and10 gains; none among839 active-grouped Core predictions with probability>=0.9. Expanded task results and confirmed losses are below. | User-authorized experimental trial; practical parity with Core is not established. |
| Canonical prompt grid and paired target/draft checkpoints | Deep edit reprocessing65536 ->28672 tokens, about137.4 ->63.6 seconds in its matched test | Normal repeats process a512-token tail. Deep edit/restore reuses232960 tokens; paired draft state preserved. | Retained in selected experimental trial. |
| Prefill scheduling | Growing-prefix PP gains2.09-3.90% beyond8K; near262K473.40 ->491.88 tok/s |14 exact model pairs, checkpoint/history and operator checks. A later combined reverse-order run corroborated about3.85% deep PP. | Qualified experimental candidate. |
| MTP depth2 with scheduling | Compared with grouped MTP1 on128-token windows, near262K38.00 ->44.35 TG;10.76-21.37% gain across tested curve |20 exact paired model cases, history/checkpoints and24/24 completed tasks | Qualified experimental candidate. |
| Q6 exact mul24 plus two-row verification, on scheduling/MTP2 | Additional7.38-9.81% TG across8K-262K; near262K45.26 ->48.61 TG |21 exact full first-logit/128-token pairs,26 history comparisons, identical draft/work counts;24/24 complete tasks and messages exact | Qualified for controlled Vivi testing; not selected automatically. |

These are separate matched experiments. Do not add their percentages or combine absolute rates from different generation windows. In particular, the original40.93 TG result and the later128-token rates are different fixtures. "Exact" in later transport/Q6 tests is relative to the already-grouped reference; it does not erase the original grouped-versus-Core differences or prove arbitrary future task accuracy.

The original grouped investigation went beyond the initial66 tool checks. Thinking OFF:432 requests,216 per lane; Core124/204 automatic passes versus grouped123/204, with80 shared failures and one additional grouped math failure. At128K an age problem changed the correct13 to16 at a low-margin digit decision, reproduced from a cold prompt. A changed64K prose answer also misstated three archive entries as two. Dedicated tools30/30, retrieval24/24 and code24/24 passed per lane. Thinking ON:144 additional requests; both lanes scored33/34 at each tested depth128K/260096, with no regression among32 pairs where both finished at each depth. Grouped lost one math completion to the token budget and gained one code completion at each depth. Both correctly solved the age problem with thinking enabled. These known results remain in the decision; later exact Q6/transport tests do not repair or requalify original grouped attention. Detailed evidence:20260929-grouped-parity/GROUPED_PARITY_REPORT_20260929.md.

The checkpoint pool is host memory: approximately149.6MiB per paired checkpoint, up to72 checkpoints, about10.5GiB. It is not10.5GiB of GPU VRAM. Cache reuse means processing the uncached/rolled-back portion, not avoiding all prompt work.

## Qualified scheduling/Q6 reference curve

Both sides of the original Q6 comparison already enabled scheduling and MTP2. PP below is growing-prefix throughput, not a cold full-length prompt at each depth. TG is the mean of two cached128-token windows.

| Context tokens | PP tok/s | TG tok/s | Q6 TG gain over matched control |
| --- | ---: | ---: | ---: |
|8192|1121.32|67.34|9.81%|
|32768|1054.96|66.03|8.53%|
|65536|910.30|63.79|9.04%|
|131072|740.55|59.82|8.18%|
|196608|589.35|53.61|7.72%|
|261632|490.88|48.61|7.38%|

The Q6 run recorded short-context PP differences of-0.67% and-0.64%; deep PP was essentially unchanged. Do not hide these differences. Source: the scheduling/Q6 report and raw evidence under20260930-schedule-q6-stack.

## AllReduce investigation

The diagnostic saw28272 internal-provider calls and zero fallback:8580 large5MiB prefill calls and19692 small10/20/30KiB calls. Old APEX small-message Direct-P2P remains off. The current large peer-copy experiment is a different path and preserves the original BF16-input/FP32-sum contract.

| Experiment | Result | Decision |
| --- | --- | --- |
| Raise copy threshold to8MiB | Standalone5MiB collective faster, but whole-model PP1.06-3.07% slower; exact outputs/checkpoints | Reject. |
| Force2MiB chunks | Near262K PP+0.053%, TG+0.069%; mixed small-append results | Keep defaults; no convincing workload gain. |
| Ring2/4/8 | No useful gain at observed sizes; ring8 slower at30KiB | Keep ring2. |
| Eliminate empty small-message blocks | Small10/20KiB microbenchmark improvement; full curve essentially unchanged, near262K490.31 ->490.26 PP and48.58 ->48.51 TG | Do not retain. |
| Globally reduce small-kernel blocks |4 blocks improved30KiB about12.7%, but slowed160KiB and near1MiB | Reject global change. |
|4 blocks only up to32KiB | Exact microbenchmark and queued grid-transition checks;30KiB improved, severe larger-message loss avoided | Deferred, no whole-model gain established. |
| Large peer copies |5MiB whole collective about1113-1118 ->607us in both orders; original arithmetic retained | Model and confirmation results are in ALLREDUCE_PEER_COPY_20260930.md. |

The two populated cards are expected to run x8/x8. Read-only inspection separately found upstream32GT/s on one path and16GT/s on the other; both report32GT/s maximum. Cause is unestablished. All comparisons retain the same hardware configuration; no BIOS, link, driver or other host setting was changed.

Stock RCCL is not a precision-equivalent switch: its small path can use unrounded FP32 inputs, and its large path rounds the sum back to BF16. The internal reference rounds inputs to BF16 but sums into FP32. No provider replacement or RCCL protocol setting was applied.

## Other negative or incomplete leads

| Lead | Result/status |
| --- | --- |
| Match Core query quantization | Reduced numerical difference but worsened held-out likelihood; not a correction to retain. |
| Full FP32 P.V accumulation | Mixed accuracy and58-80% slower attention; removing spills did not recover speed. |
| Final-merge precision and partition corrections | Global-max FP32, compensated FP32, FP64 merge and different group counts did not produce a consistent quality improvement. Global-max and96 partitions still failed the known cold age case. No correction adopted. |
| Extra packed-Q8 conversion integration | Small operator gain did not translate into whole-model improvement; not retained. This is distinct from the successful original packed/grouped attention. |
| Narrower grouped query tile |16-column/stage64 variants slower;32-column alternative not a convincing model gain. |
| Generic Q4 KV | Slower in tested path. |
| Native FP8 | Interesting operator/conversion lead, not qualified end-to-end KV format or accuracy. Deferred. |
| Old Direct-P2P/fused-boundary combination | Harmed PP/history in V2 comparisons; off. |
| HIP graphs | Replay observed on both GPUs. Internal AllReduce executes between device graph subgraphs. "Enabled" was not used as proof of replay. |
| Checkpoint transfer batching/pinning | Source-level lead only; no host environment or runtime change applied. |

## Serving state and review boundary

The selected service remains the original corrected grouped MTP1 Vivi trial on port8083, restored after isolated tests. Qualified scheduling/Q6/MTP2 and peer-copy binaries are separate candidates; completion of benchmarks does not silently select a new serving configuration. The final independent restoration receipts attest executable, loaded libraries, gates and health.

Vivi log: <workspace>/apex-v2-experiments/20260929-grouped-vivi-trial/server.log

Known-good source:cbc3985. Peer transport kernel commits:075c19f and644bc89. Scheduling/Q6 pure commits:e98a42c,637ec67,86b7942. Build manifests record exact composition and hashes; the peer build reuses the qualified scheduling/Q6 objects and replaces only AllReduce.

After finishing peer-copy validation, preserve size-based grids, event-overlap changes, FP8 and checkpoint-transfer ideas as deferred leads. Review measured benefit, practical capability evidence and the desired next Vivi trial with Adam before expanding scope.

## Latest peer-copy increment on the qualified stack

Both model process orders completed. Pooled equal-work rates,8K/32K/64K/128K/192K/261632 tokens:

| Context | Control PP | Peer PP | PP gain | Control TG | Peer TG | TG change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|8192|1129.64|1235.94|+9.41%|67.14|67.17|+0.05%|
|32768|1055.84|1153.11|+9.21%|66.16|64.93|-1.86%|
|65536|907.69|977.10|+7.65%|63.68|63.21|-0.74%|
|131072|737.07|782.51|+6.17%|59.67|59.36|-0.52%|
|196608|589.30|615.38|+4.43%|53.53|53.57|+0.06%|
|261632|490.00|508.80|+3.84%|48.53|48.32|-0.43%|

These are experimental benchmark rates, not a claim that the selected Vivi service has been switched.42 paired first-step full logits,128-token rollouts and work/draft counts match exactly, with52 exact history checks. Checkpoints retain232960 tokens after the deep edit/restore. The peer change is a repeatable prefill gain with a small decode tradeoff; it does not flatten the generation curve further. Completed-answer compatibility also passed24/24 at128K/260096: complete messages, full first-step logits and work/draft counts exact;5157 generated tokens and no truncated answers. Transport tracing also passed378 full-array/history checks on both ranks and confirmed copy-engine routing for all traced5MiB chunks. Profiled timings were excluded. All scoped tests are complete; original selected Vivi trial restored and independently verified as PID2617073.

## Evidence index

- [Core ports and August integration](core-integration.md)
- [Original grouped accuracy, actual regressions and speed curve](grouped-parity.md)
- [Cache coverage correction](cache-coverage.md)
- [Prefill scheduling](prefill-scheduling.md)
- [Scheduling and MTP2](schedule-mtp2.md)
- [Scheduling/Q6 combination and completed tasks](q6-stack.md)
- [AllReduce alternatives and negative results](allreduce-tuning.md)
- [Current peer-copy implementation and results](peer-copy.md)
- Two-order raw aggregate (local evidence reference)

## Recommendation for review

Retain scheduling/Q6/MTP2 as the generation-oriented experimental candidate and its peer-copy build as the prefill-oriented candidate. Peer copies add3.84-9.41% PP in the pooled whole curve, with a0.43% near-full cached-TG cost and a1.86%32K cost. They preserve all tested outputs/checkpoints relative to that qualified reference. Original grouped remains an authorized experimental trial with known Core differences; this transport work does not convert that into a general accuracy-equivalence claim.

No serving switch was made. Original corrected grouped MTP1 is healthy on8083 as PID2617073; all scoped GPU tests are finished. Review these measured options before choosing the next Vivi test or another kernel lead. No further optimization was started.
