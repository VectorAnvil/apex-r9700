# DFlash Draft-Specific Split Mode Finding

## Result

The target can remain tensor split across both R9700s while DFlash uses layer
split across those same devices. The accepted source remains isolated and is
not registered.

The new `--spec-draft-split-mode layer` option is default-off. Without it, the
draft inherits the target split mode exactly. With it, only the temporary draft
model parameters change; parser tests prove the original target parameters
remain tensor split.

## Required Companion Plumbing

Split mode alone was insufficient because DFlash reuses the target token
embedding and LM-head projection. Under tensor split these are allocated on the
target `Meta()` backend. The DFlash context now adds only missing devices from
`ctx_other`, allowing its scheduler to execute those shared tensors without
changing either model's placement.

This exposed a one-tensor-descriptor shortage in ggml's rotating Meta
external-view container. Raising its documented headroom from 16 to 17 fixed
the exact 368-byte shortfall. A prior change to a different Meta pool had no
effect and was reverted.

## Correctness

The candidate ordinary tensor-split control matched the registered control for
all 192 raw token IDs. DFlash with `n_max=12` also matched all 192 tokens in two
independent server processes.

The width result was non-monotonic:

| `n_max` | Exact | First differing token (zero-based) |
|---:|:---:|---:|
| 1 | no | 159 |
| 4 | no | 79 |
| 8 | no | 159 |
| 12 | yes, twice | none |
| 15 | no, repeatable | 159 |

Consequently Phase 21C accepts only `n_max=12`. The intended block-15 route is
still rejected. This appears to be a target verification batch-numerics
boundary, not runtime randomness, because the divergent outputs repeat exactly.

## Scope And Provenance

The accepted patch changes common argument/plumbing, DFlash shared-device
enumeration, one host metadata headroom constant, and an argument test. It does
not change target kernels, Q6 arithmetic, P2P, all-reduce, graph topology, or
registered assets. Candidate and registered HIP source files are byte-identical.

Loaded candidate VRAM was 44%/43%; both devices returned to 0% after exit. The
single-process throughput values are smoke evidence only. No stable benchmark,
top-32 recorder, DDTree policy replay, GPU tree execution, or promotion ran.
