# C11+C07+C03c production promotion — 2026-08-19

## Decision

The exact C11+C07+C03c bundle passed final adjudication and was promoted into the canonical Qwen3.8 service. It is layered on the existing Direct-P2P, Phase 13, and FA-1 production stack; none of those paths were removed.

## Final performance gate

Interleaved ABBA/BAAB runs used the production Qwen3.8-27B Q6_K model, dual R9700 tensor split, batch 2048, ubatch 512, Q8 KV, and FA enabled.

| Cell | FA-1 current | Promoted bundle | Mean ratio |
|---|---:|---:|---:|
| PP512 d0 | 1018.5006 t/s | 1023.7706 t/s | 1.0051742x |
| PP4096 d0 | 980.5340 t/s | 986.8290 t/s | 1.0064200x |
| PP4096 d65536 | 478.4616 t/s | 480.3405 t/s | 1.0039268x |

The final 127K replication was curtailed after one current anchor because real depth materialization made the original eight-process schedule disproportionately expensive. The completed eight-process 64K guard was uniformly positive. Existing FA-1 long-context evidence and the fact that this bundle does not alter FA dispatch remain the 127K protection basis.

Prior exact-bundle evidence also remains applicable: 24/24 correctness, 8/8 exact/fallback resource dispatch, and a `1.0006390732x` mean TG128 gain for C03c inside C11+C07.

## Stability and serving gate

Three independent candidate server starts used the full production configuration: 262,144 context, embedded MTP `n_max=2`, Q8 KV, dual-GPU tensor split, Direct-P2P, Phase 13, and FA-1. All three passed health, ordinary chat, strict JSON-schema output, and forced tool calling.

The promoted live request returned `PROMOTED_OK`, with MTP accepting 26/26 draft tokens. The service was healthy after promotion.

## Pinned production identity

- Server SHA-256: `45c34cb4a08a22b422501ff59e930e48f53a1622b2758caf4dc437b53e2f74c0`
- HIP library SHA-256: `be00eb82f0b701b7ce360eeec4d2bc67b7cb0583d65e0013547777b7b0ddd25f`
- Source base commit: `4695f001fece1660d8bb1b3748f50726ddcc100b`
- Required gates: `GGML_CUDA_Q6K_SWIGLU_D4_FFN_DOWN=1` and `GGML_CUDA_Q6K_SHARED_Q8_GATE_UP=1`
- Canonical launcher: `/home/adam/workspaces/ChatGPT/Apex/tools/start-apex-qwen38-canonical.sh`
- User launcher: `/home/adam/scripts/start-apex-qwen38-fa1.sh`
- Logs/state: `/home/adam/.local/shared/apex-qwen38-fa1`

The pre-promotion canonical launcher is retained as `start-apex-qwen38-canonical.sh.pre-c11-c07-c03c-20260819` for immediate rollback.

## Evidence

- Machine-readable final adjudication: `results_phase04_15_combo_c11_c07_c03c_20260819/evidence/final-adjudication.json`
- Raw final A/B runs: `results_phase04_15_combo_c11_c07_c03c_20260819/evidence/final-promotion-matrix/`
- Three-cycle service soak: `results_phase04_15_combo_c11_c07_c03c_20260819/evidence/final-server-soak/`
