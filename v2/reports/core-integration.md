> Historical research record. Statements about the selected service are as of this report, not current deployment status. See [the V2 index](../README.md). Local raw evidence, private runtime files and logs are not included; local artifact paths are provenance references, not downloadable links.

# August production integration into V2

Status: complete. Core ports are retained in the V2 development profile, improving deep decode by 20.6-29.4% over the pre-port V2 control. Direct-P2P and fused boundaries remain available as an optional combination because their small extra decode gain comes with a reproducible 32K prefill penalty. August production is restored and verified unchanged.

The control is V2 at d7fb90e8 plus the accepted concurrent Q6_K, canonical packed attention, natural recurrent split, 256-token KV boundary, strict-prefix checkpoint guard and live draft KV reuse. `results/august-integration-20260927/baseline-bin` freezes the exact pre-port binaries. Each candidate build has separate hashes. All serving tests use Q6_K weights, Q8 KV, tensor split 1:1 on gfx1201, and MTP depth 1 unless labelled otherwise.

| August component | V2 disposition / candidate |
|---|---|
| Phase 13 Q6_K MMQ float cast | Ported to current `mmq-vec-dot.cuh`; isolated binary comparison |
| Direct peer-VRAM AllReduce (JohnTDI-cpu, APEX integration) | `APEX_V2_DIRECT_P2P=1`, two gfx1201 GPUs, eight blocks, <=512 KiB; upstream fallback for larger tensors |
| Fused AllReduce + residual + RMSNorm + scale | `APEX_V2_FUSED_BOUNDARY=1`, requires active P2P; exact 5120-wide, 1-8 row shape; preserves upstream delayed-reduction logic |
| Canonical packed projection columns | `APEX_V2_Q6K_PACKED=1`; two/three columns share one launch and one-column reduction geometry. Uses V2's eight-wave arithmetic, not August's shape-specific four-wave arithmetic |
| Shared gate/up Q8 preparation | `APEX_V2_SHARED_Q8=1`; exact production decode and PP512 shapes, preserving two projection outputs |
| Fused SwiGLU to D4 Q8 MMQ input | `APEX_V2_SWIGLU_D4=1`; production PP512 FFN shape only |
| Canonical vector attention / recurrent natural split / KV boundary | Already adapted in accepted V2 controls |
| Old restored rocWMMA FA implementation | Not transplanted: disabled in August binary; modern upstream attention and V2 packed Q8 path retained |
| August model/GDN behavior | Upstream corrected normalization, fusion and cache behavior retained |

Precision control: current upstream internal AllReduce defaults to rounding F32 inputs through BF16. August's P2P implementation kept F32. The port honors `GGML_CUDA_AR_BF16_THRESHOLD`, so transport-only comparisons use identical arithmetic. Separate F32 comparisons set that threshold to zero in both lanes. The port initially retains all release/acquire fences; August skipped the acquire fence. Do not attribute historical August gains to this port without measuring these differences.

The first short-prompt run matched all 12 outputs with P2P but failed the dispatch-evidence check because server verbosity 3 suppressed backend INFO logs. This was an instrumentation failure, not a numerical failure. `smoke-logged` repeats with verbosity 4 and requires nonzero kernel-launch counts.

Reproducer: `python3 apex/scripts/august_integration.py PHASE --execute`. The driver validates production identity, hashes, idle slots and GPU ownership before pausing it, and restores it in `finally`. Historical evidence is preserved; the original August source and release are read-only.

## Integration details

The source worktree is pinned at `f4752bfac3f91cf0b93a4ed60c3bffe9cc2962be`; its AllReduce file hashes identically to the August release's archived source. The old CMake and HIP vendor-header additions exclusively support the disabled restored rocWMMA path, so they need no replacement in the active V2 route.

The four-wave emulation candidate (`APEX_V2_Q6K_EMULATE8=1`) maps eight logical reduction waves onto four physical waves for the August mask-7 row counts (3072, 5120, 8704). It retains two separate partial sums and combines them in the eight-wave order. August's special direct four-wave arithmetic for the 6144-row projection is intentionally not selected: V2 retains its qualified upstream one-row arithmetic there. The packed-column adaptation covers V2's same homogeneous Q6_K multi-query scope.

Current upstream added `mmq_args::ncols_opt`. The initial August helper port omitted the new field, implicitly zeroing it and choosing an unsuitable tile. That initial `projections` run passed output checks but lost prefill speed. The corrected `projections_v2` build initializes both `ncols_max` and `ncols_opt` to `ne1`, as upstream does. The initial build and results remain archived and are excluded from final speed comparisons.

Initial short precision qualification: host/P2P match 12/12 at BF16 wire precision and 12/12 at F32. Changing BF16 to F32 changes 3/12 outputs. This confirms why transport and precision must be tested separately.

## Qualification before deep timing

- Direct-P2P matches all 12 short cases with ordinary decode and MTP1 at the matched V2 precision. P2P-off reproduces the frozen control. The separate matched F32 pair also passes 12/12.
- Phase 13 alone and with P2P pass 12 short cases plus cold 32K.
- The fused boundary passes 12/12 in ordinary decode and MTP1, with nonzero dispatch counters.
- Packed projections, shared Q8, SwiGLU D4, and their combination each pass 12 short cases plus cold and cached 32K. The corrected MMQ ports repeat the checks successfully.
- Four-wave emulation alone, the full MTP1 combination, and the full ordinary combination each pass the same 14 cases.
- The final full combination passes 576 fixed-token positions at query widths 1, 2 and 3: 1,728 rows, zero differing logits versus the one-query reference, and all expected argmax tokens. This tests row-width arithmetic independently of speculative acceptance.

Preliminary component timings show Phase 13 improving 32K prefill by about 19%; packed projections improve 32K MTP1 decode from approximately 35.0 to 46.7 tokens/sec. Shared Q8 and SwiGLU contribute less than 1% individually in the short qualification pass. Some early 32K qualifications showed about 11% lower prefill throughput with P2P, but the clean deep experiment did not reproduce that cost at 64K and above. The separate reverse-order history repeat below confirms that this is a workload-dependent cost of the communication combination.

## Deep experiment design

Four configurations are measured in forward and reverse process order: frozen V2; Phase 13 plus the projection/quantization ports; those ports plus Direct-P2P; and the full set including the fused boundary. All use packed attention and the strict-prefix/live-draft checkpoint pair. GPU compilation is finished before these timings.

Each process starts cold at 64K, then extends the prefix to 128K and 192K. At every depth it generates 128 tokens, repeats 256-token decode twice, and measures a 16-token append after a one-token seed. Comparisons require identical input tokens, output tokens and processed/cached token counts. Every 128-token growth output is also checked against the previously established independently cold ordinary reference. This avoids repeating the earlier cached-history reference mistake.

Decode summaries aggregate the server's generation time across four 256-token samples per configuration/depth. Append TTFT has two samples per configuration/depth. Sample ranges are retained; they are not confidence intervals. The workload is synthetic, text-only, and uses MTP1. An MTP-depth ranking measured before these projection ports should not be assumed to remain optimal afterward.

## Repeated deep results

All values below are generated tokens/sec with the same MTP1 acceptance counts and exact output tokens. The eight processes complete 120 requests. Every growth request and the first 128 tokens of every sustained decode are also checked against independently cold ordinary output (72 checks).

| Context | Frozen V2 control | Core ports | Core + P2P | Full stack | Full vs control |
|---|---:|---:|---:|---:|---:|
| 64K | 31.13 | 40.28 | 40.39 | 41.17 | +32.3% |
| 128K | 24.84 | 30.76 | 30.82 | 31.16 | +25.5% |
| 192K | 22.39 | 27.01 | 27.14 | 27.34 | +22.1% |

Core includes Phase 13, packed Q6_K projections, eight-wave reduction emulation on four physical waves, shared gate/up Q8 preparation and fused SwiGLU D4 preparation. Full adds Direct-P2P and the fused boundary. Direct-P2P alone adds only 0.2-0.5% over core at these depths. The boundary adds a further 0.7-1.9% over P2P. The full stack is 1.2-2.2% faster than core.

| Context | Control prompt tokens/sec | Full prompt tokens/sec | Change | Control append16 TTFT | Full append16 TTFT |
|---|---:|---:|---:|---:|---:|
| 64K | 850.69 | 1005.79 | +18.2% | 365.4 ms | 374.7 ms |
| 128K | 650.42 | 737.62 | +13.4% | 410.8 ms | 414.0 ms |
| 192K | 526.02 | 581.79 | +10.6% | 461.3 ms | 463.2 ms |

The 64K prompt is cold; the later rows process 66,052 new tokens while extending the existing prefix. They are not cold 128K/192K prefill rates. P2P and boundary fusion do not reproduce the earlier 32K prefill penalty in this ladder. Append16 latency has no gain here: full adds about 9 ms at 64K and 2-3 ms at deeper contexts, with only two samples per point. The much larger checkpoint-reuse improvement was already present in every lane.

!Measured throughput and append latency (local evidence reference)

The full stack saves about 8 ms per generated token at each depth. This is consistent with reducing context-independent projection and communication work. The per-token latency increase from 64K to 192K is almost unchanged, so these ports raise the throughput curve but do not resolve its context-dependent slope. Further KV/attention work remains necessary to flatten it.

## Confirmed interaction and retained configuration

The final history repeat runs 12 short prompts, a cold 32K prompt and its cached repeat in core/full/full/core process order. All 56 outputs match the original V2 control. The two full-stack prefill samples are 970.36 and 970.78 tokens/sec; core gives 1095.10 and 1091.67. Thus the communication pair reduces 32K prefill throughput by 11.2% relative to core. Its 32K decode improves from 46.73 to 47.86 tokens/sec (+2.4%). This penalty survives the corrected MMQ helper and repeated process order. It must not be dismissed as the earlier `ncols_opt` porting error.

The evidence establishes a cost for the combined communication configuration on this request history, not its exact kernel-level cause. The same cost does not occur in the 64K-start deep ladder. Earlier P2P-only qualification points toward that component, but localization remains open.

The default development profile (local evidence reference) therefore retains Phase 13 and all core ports, packed attention, natural recurrent splitting, the 256-token KV boundary, and the paired strict-prefix/live-draft checkpoint controls. It pins the qualified upstream BF16 wire threshold. Default decode gains are +29.4%, +23.8% and +20.6% at 64K, 128K and 192K. Its long prompt-processing gains are +18.5%, +13.7% and +10.8%, with essentially unchanged append TTFT.

The optional communication profile (local evidence reference) enables the tested full stack when loaded after the development profile. Both modes are available in the same qualified binary. No original August launcher or binary was changed, and no production switch was made.

The integration answers the stacking question positively for the core ports, and finds a real tradeoff for the communication pair. The next useful work is to localize that 32K history-dependent penalty and remeasure optimal MTP depth with packed projections before assuming the old MTP ranking still applies. The larger remaining long-context target is attention/KV traversal.

## Scope and interpretation

These comparisons measure August-derived ports on the qualified V2 numerical reference. They are not a fresh speed comparison against the August production executable. V2 is pinned to the upstream commit above; this work does not imply a continuously updated upstream checkout.

The combined projection changes preserve the one-query arithmetic that resolved the earlier MTP failures. Packed projection columns and packed attention are separate optimizations: the former is added in this round, while the latter is enabled in every control and candidate. Phase 13 is compiled into the candidate; the other new controls remain opt-in at runtime.

The supported development target remains two gfx1201 GPUs, this Q6_K model, Q8 KV, one slot and MTP1. The teacher-forced width checks exercise numerical consistency at widths 1-3, but do not establish the fastest speculative depth. Multimodal behavior and a broader production workload have not been requalified in this round. August production remains the operational fallback.

## Evidence and restoration

The evidence bundle (local evidence reference) preserves source/build identities, exact input/output tokens, dispatch logs and raw timings. See the deep timing summary (local evidence reference), 32K history summary (local evidence reference), qualification counts (local evidence reference) and independent cold-reference checks (local evidence reference). The current build matches all 11 files in the final frozen build manifest (local evidence reference).

The final restoration receipt (local evidence reference) verifies original August executable/library hashes, active service PID 2489142, a successful READY completion on port 8083, idle slots, no experimental listener on port 18098 and unchanged Bonsai PID 3540264. All 728 previously archived evidence files remain unchanged.
