# DFlash2 width-3 adjudication

Date: 2026-08-23

## Decision

Width 3 is not production-ready. It is the fastest tested DFlash setting that remains above the width-2 reference, but it passes only 10/12 cases in the frozen deterministic parity suite. Production remains on the promoted width-2 embedded-MTP path.

## Width comparison

| Width | Exact cases | Throughput | Acceptance | Decision |
|---:|---:|---:|---:|---|
| 2 | 12/12 | 54.6400 t/s | 69.54% | correctness reference |
| 3 | 10/12 | 58.5701 t/s | 60.79% | investigate |
| 4 | 10/12 | 57.1329 t/s | not recorded | reject for now |
| 7 | 10/12 | 49.6539 t/s | not recorded | reject |

Width 3 fails `list` at token 167 and `forced128` at token 76. A parity fix must preserve its performance advantage over width 2 to be promotion-relevant.

## Rejected corrective prototypes

| Prototype | Exact cases | Throughput | Outcome |
|---|---:|---:|---|
| N=4 canonical MMVQ + FA | 9/12 | 53.4985 t/s | worse correctness and speed |
| revised N=4 canonical MMVQ + FA | 9/12 | 56.1279 t/s | still worse correctness |
| full checkpoint | 9/12 | 23.6662 t/s | unusable throughput |
| canonical GDN | 10/12 | 46.7993 t/s | no parity gain, large loss |
| broad N=4 FA | 9/12 | 46.2608 t/s | worse |
| speculative-only N=4 FA | 8/12 | 45.9080 t/s | worse |

These results rule out indiscriminate canonicalization and broad checkpointing as production fixes. They do not rule out a narrowly located state or arithmetic correction at the actual first divergence.

## First-divergence evidence

At the state-matched position-13 probe, ordinary and speculative execution remained exact through convolution, Q/K, gate, and beta. The first observed difference was inside Gated DeltaNet. Forcing canonical K=1 GDN made the row-6144 FP32 output exact, but did not improve the 12-case suite and reduced throughput to 46.7993 t/s.

Therefore that early local difference is not established as the cause of the late committed-token failures. Correcting every detectable floating-point difference is neither sufficient nor economical.

## Next bounded trace

Trace only the two failing cases:

1. `forced128`, absolute positions 84-87 around the failure that commits at token 76.
2. `list`, immediately before the failure that commits at token 167.

For each position, compare ordinary and width-3 execution from identical saved state, operation by operation, through:

- recurrent convolution input and state;
- mixed Q/K/V, `z`, beta, and recurrent state update;
- `ssm_out` and its layout handoff;
- FFN gate/up/down;
- full-attention Q/K/V and output projection;
- final logits and sampler inputs.

Stop at the first bit-different live value and record whether the divergence survives to logits. Do not start another corrective kernel until this causal trace identifies a bounded target.

## Promotion gate

Width 3 requires 12/12 exact committed-token parity, repeated process stability, and throughput that remains meaningfully above the promoted width-2 path. Until all three pass, it stays default-off and research-only.
