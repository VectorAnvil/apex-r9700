# FA-1 Vivi Serving Promotion

## Decision

**Promoted and running.** The canonical Qwen3.8 service launcher now pins a
dedicated `llama-server` built from the exact accepted FA-1 worktree. The
production service is healthy at `127.0.0.1:8083` with full 262,144-token
context, two-R9700 tensor split, Q8 KV cache, Flash Attention, Direct-P2P,
Phase-13 Q6_K conversion, and embedded MTP at `n_max=2`.

This is the serving-stack promotion that was deliberately separate from Llama
Lab build registration.

## Pinned Runtime

- llama.cpp base: `4695f001fece1660d8bb1b3748f50726ddcc100b`, build 10457.
- FA-1 source: the accepted current-source rocWMMA worktree from FA-1/FA-S1.
- Server root:
  `results_fa01_current_rocwmma_20260817/builds/current-rocwmma-server`.
- `llama-server` SHA-256:
  `b5c8bb281e0e4abec2a18a73d5350a0a5d9bc31666447058d9592f1275a1df22`.
- `libggml-hip.so` SHA-256:
  `0aa78c06c795544ffbbc59c6f8f350705b189212b56255f731363beda92505e0`.
- Updated launcher SHA-256:
  `dda76acdb93006d19a65384eea5f0783485036c2f01d871a42ab87615238bfe9`.
- Model SHA-256 remains the existing canonical Qwen3.8 pin:
  `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727`.

The prior production server and HIP hashes remain recorded in repository
history and the old build directory remains intact for rollback. No registered
FA-1 benchmark binary was mutated to add the server target; the serving build
uses a separate build directory.

## Gates

The server build completed successfully for `gfx1201` and advertised every
launcher-required option. `ldd` found no missing runtime dependency. The
launcher's fail-closed runtime and model hash verification passed before start.

An isolated port-18083 smoke then loaded the canonical model with the exact
production arguments. It reached healthy status and returned the requested
response exactly. The production launcher subsequently passed its own start,
health, request, and status lifecycle on port 8083.

| Gate | Result |
| --- | --- |
| Full context | `n_ctx_slot = 262144` |
| Model load | PASS |
| Health before request | `{"status":"ok"}` |
| Exact response | `FA1 serving smoke passed` |
| Prompt | 61 tokens, 70.21 t/s on production |
| Generation | 43 tokens, 60.47 t/s on production |
| Embedded MTP | 27/32 drafted tokens accepted |
| Health after request | `{"status":"ok"}` |
| Production process | PID `3624835`, launcher-owned, port 8083 |

The smoke is a serving correctness and integration gate, not a replacement for
the FA-S1 long-context benchmark matrix. FA-S1 remains the performance evidence
for the restored attention path.

## Final State

The canonical service was intentionally left running on the FA-1 server after
the successful production request. The obsolete isolated port-18083 smoke
process was stopped before production start. Vivi can now consume FA-1 through
the existing canonical endpoint.
