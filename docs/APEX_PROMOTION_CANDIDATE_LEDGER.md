# Apex R9700 Promotion Candidate Ledger

This ledger retains every measured positive production-promotion option, including small gains. A candidate remains independent until the exact combined stack is rebuilt and benchmarked; gains in this table must not be added or multiplied as a prediction of combined performance.

## Promotion candidates

| ID | Candidate | Workload affected | Best adjudicated production-level result | Guard result | Validation state | Combination state | Promotion state |
|---|---|---|---:|---:|---|---|---|
| N2-PACK-ALL | Packed canonical rows for every selected Q6_K MTP N=2 projection | Heretic Q6_K speculative decode | Two-run candidate mean `58.0286 t/s` versus bracketed control `49.8535 t/s`: `1.1639837x` (+16.40%); isolation replication `1.1631946x` (+16.32%) | Exact at 16K/65K/127K; historical same-harness depth TG gains approximately +13.84%/+10.37%/+8.96%; two production-form and two installed-launcher activations with multimodal smoke passed | 28/28 bit-exact kernel cases across both GPUs; 120/120 candidate/isolation prompts exact; sensitive hashes exact across three cold starts and both live activations | Includes the already-promoted packed FFN path; isolated recurrent core and attention projection contributions measured separately | **Promoted to production 2026-08-22**; live PID `867874`, server `10856b...`, HIP `5f3656...` |
| N2-PACK-CORE | Packed canonical rows for recurrent mixed-QKV, gate, and `ssm_out` | Heretic Q6_K speculative decode | `1.12164598x` (+12.16%): 55.8835 versus 49.8228 t/s | 30/30 exact; full depth/service gates belong to N2-PACK-ALL and must be rerun if this fallback is promoted alone | Both-GPU arithmetic is covered by the packed-all 28/28 gate | Subset of N2-PACK-ALL | Retain as positive fallback; not promoted |
| N2-PACK-ATTN | Packed canonical rows for full-attention Q/K/V/output projections | Heretic Q6_K speculative decode | `1.03218751x` (+3.22%): 51.4265 versus 49.8228 t/s | 30/30 exact; full depth/service gates belong to N2-PACK-ALL and must be rerun if this fallback is promoted alone | Both-GPU arithmetic is covered by the packed-all 28/28 gate | Subset of N2-PACK-ALL | Retain as positive fallback; not promoted |
| C11+C07+C03c | Exact mixed-workload bundle: C11 + C07 PP paths plus C03c `6144x1x5120` four-wave TG specialization | PP512 and TG128 on the current Direct-P2P + Phase 13 + FA-1 stack | Final A/B vs FA-1 production: PP512 `1.0051742377x`, PP4096 d0 `1.0064199872x`, PP4096 d65536 `1.0039268416x`; adds TG `1.0006390732x` mean versus C11+C07 | Eight-process 64K PP guard positive; C03c isolated PP guard flat; FA-1 path retained | 24/24 correctness; 8/8 exact/fallback dispatch; 4,128 of 86,946 TG MMVQs converted; 3/3 full 262K/MTP service restarts passed health/chat/JSON/tool | Exact triple measured and deployed with both runtime gates | **Promoted to production 2026-08-19**; server `45c34c…`, HIP `be00eb…`, live PID verified |
| C11+C07 | Exact bundle: Q6_K SwiGLU-D4/down specialization + shared gate/up Q8_1 preparation | PP512 on the current Direct-P2P + Phase 13 + FA-1 stack | Two-schedule aggregate `1.0063610674x` mean (+0.6361%), `1.0061525197x` median (+0.6153%), six samples/state | TG128 `0.9989841945x` mean, `0.9999780667x` median (flat; selectors do not dispatch at TG width) | 32/32 correctness; 20/20 isolated dispatch; real model 1,600→1,088 ordinary quantizers with 256 C11 fused launches retained and MMQ count fixed; server health/chat/JSON/tool smoke pass | Exact C11+C07+C03c triple measured; C03c adds TG with a flat PP guard | Leading PP-only portfolio bundle; eligible for final decision; not promoted |
| C11 | Q6_K `M=5120, K=5120` N=1 launch-bound specialization | PP512 on the current Direct-P2P + Phase 13 + FA-1 stack | `1.0048166868x` paired median (+0.4817%); reverse-order replication `1.0046569221x` mean (+0.4657%); 10/10 pairs positive | TG128 `0.9992581622x` mean, `0.9999303777x` median (flat) | 18/18 correctness, 6/6 dispatch, 256 real-model fused launches, server health/chat/JSON/tool smoke pass | Exact C11+C07 bundle measured at `1.0063610674x` mean / `1.0061525197x` median | Eligible independently or through C11+C07; not promoted |
| C07 | Shared Q8_1 preparation for the Q6_K gate/up PP pair | PP512 on the current Direct-P2P + Phase 13 + FA-1 stack | Six-sample aggregate `1.0013055939x` mean (+0.1306%), `1.0015909794x` median (+0.1591%) | TG128 six-sample aggregate `0.9997344775x` mean, `1.0000277845x` median (flat) | 8/8 correctness; 6/6 exact/fallback dispatch; real PP trace removed 256 quantizers with MMQ count unchanged; server health/chat/JSON/tool smoke pass | Exact C11+C07 bundle measured at `1.0063610674x` mean / `1.0061525197x` median | Retain independently or through C11+C07; not promoted |
| C03c | Q6_K `6144x1x5120` gfx1201 four-wave specialization | TG128 on the current Direct-P2P + Phase 13 + FA-1 stack | Independent `1.0006605712x` mean (+0.0661%), `1.0006987259x` median (+0.0699%); inside C11+C07 `1.0006390732x` mean (+0.0639%), `1.0003742195x` median (+0.0374%) | Exact triple PP guard `1.0000796782x` mean, `0.9997834539x` median (flat; zero selected launches) | Triple: 24/24 correctness, 8/8 resource dispatch, 4,128 real TG four-wave launches, health/chat/JSON/tool smoke pass | Exact C11+C07+C03c triple measured; positive TG effect survives | Retain independently or through the triple; not promoted |

## Alternate configuration candidates

These candidates trade away a production capability and therefore are not drop-in promotions, but their measured gains are retained for deployment-specific decisions.

| ID | Candidate | Best adjudicated result | Tradeoff | Validation state | Promotion state |
|---|---|---:|---|---|---|
| CFG-49K | Reduce Qwen3.8 service context from 262,144 to 49,152, retaining parallel 1 and all other production flags | Reverse-order aggregate `1.0052591611x` mean (+0.5259%): 61.68078 vs 61.35809 t/s | Maximum context falls from 262K to 49K | Two runs/state, 120/120 valid requests, identical `2225/3142` MTP acceptance and committed tokens | Retain as a short-context service profile only; do not replace long-context production |

## Watchlist

| ID | Candidate | Current production-level result | State |
|---|---|---:|---|
| N3-NATIVE | Native N=4 verification at MTP width 3 on the packed N=2 production baseline | Two-run mean `1.04935729x` (+4.94%): 61.0356 versus 58.1647 t/s, but only 14/30 exact | Nonpromotable correctness headroom. Reopen only if full canonical N=4 execution can retain a net gain. |
| C03b | Q6_K `5120x1x3072` gfx1201 four-wave specialization | Six-sample TG128 `0.9999433562x` mean (−0.0057%), `1.0004929006x` median (+0.0493%); reverse replication alone was positive by both statistics | Flat/mixed. Keep available for later combination testing, but do not claim an independent gain. |

## Adjudicated non-candidates

| ID | Candidate | Result | Decision |
|---|---|---:|---|
| N3-CANON-MMVQ | Bit-exact packed canonical N=4 Q6_K MMVQ at width 3 | `0.99975294x` (-0.025%): 58.1504 versus 58.1647 t/s, with remaining N=4 FA parity only 17/30 | Reject for production. It is already slower than width 2 before canonical N=4 FA cost. |
| N2-PACK-CONTROL | Packed canonical rows for recurrent `ssm_alpha`/`ssm_beta` only | `0.99738062x` (-0.26%): 49.6923 versus 49.8228 t/s | Reject independently; retain only inside the positive packed-all bundle. |
| C03a | Q6_K `3072x1x5120` gfx1201 four-wave specialization | TG128 `0.9848280806x` mean, `0.9865499613x` median | Reject independently. |
| C03-all | All three C03 four-wave tuples enabled together | Six-sample TG128 approximately `0.9865x` mean and `0.9867x` median | Reject bundle; the harmful C03a tuple masks the other two. |

## Adjudication rule

- Preserve every repeatable positive production-level result, even below 1%.
- Record kernel-only gains as evidence, not as promotion gains, until they survive the intended model workload.
- Do not infer combined gain from independent measurements. Measure each desired candidate set as a new stack.
- Keep rejected or flat candidates in their investigation reports; move them into the promotion table only when production-level evidence is positive and correctness/behavior gates pass.
