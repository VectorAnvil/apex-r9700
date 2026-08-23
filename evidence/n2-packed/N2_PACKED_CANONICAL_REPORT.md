# Qwen3.8 Heretic N=2 packed canonical-row result

Date: 2026-08-22

Status: **PROMOTED and running on port 8083**

## Outcome

Extending the existing packed canonical-row Q6_K MMVQ path from FFN to all selected N=2 projections improves the frozen 30-prompt workload from 49.8535 to 58.0286 t/s, or **+16.40%**. Both candidate runs were 30/30 bit-exact against ordinary decoding and retained identical 70.34% draft acceptance.

The source path remains default off. Production activates it only through the hash-pinned launcher with `GGML_CUDA_Q6K_CANONICAL_PACKED_ALL=1`.

## Arithmetic gate

The first build passed 12/14 shapes on each GPU. Only `6144x5120` failed because the packed dispatch did not retain the promoted C03c four-wave specialization. Passing `force_four_waves` through the packed launch restored the exact N=1 geometry.

Final result:

- GPU 0: 14/14 bit-exact N=2/3 cases;
- GPU 1: 14/14 bit-exact N=2/3 cases;
- total: 28/28.

## Bracketed production workload

| Cell | Mean TG | Exact vs ordinary |
|---|---:|---:|
| Control A | 50.0007 t/s | 30/30 |
| Candidate A | 58.0898 t/s | 30/30 |
| Candidate B | 57.9674 t/s | 30/30 |
| Control B | 49.7062 t/s | 30/30 |
| **Bracketed control** | **49.8535 t/s** | **30/30** |
| **Candidate mean** | **58.0286 t/s (+16.40%)** | **30/30** |

## Projection-family isolation

The independent isolation bracket measured 49.8228 t/s.

| Packed family | Mean TG | Relative | Exact vs ordinary |
|---|---:|---:|---:|
| Recurrent core (`attn_qkv`, `attn_gate`, `ssm_out`) | 55.8835 t/s | +12.16% | 30/30 |
| Recurrent control (`ssm_alpha`, `ssm_beta`) | 49.6923 t/s | -0.26% | 30/30 |
| Full-attention Q/K/V/output projections | 51.4265 t/s | +3.22% | 30/30 |
| All selected projections | 57.9536 t/s | +16.32% | 30/30 |

The all-family result exceeds the sum of the two positive isolated families, so the small control-only loss does not justify excluding control from the winning bundle.

## Stability and depth

All four sensitive prompts were exact and hash-repeatable across three independent candidate cold starts. Ordinary-versus-MTP parity was exact at 16K, 65K, and 127K.

| Depth | Ordinary TG | Candidate TG | Previous promoted MTP TG | Approximate gain |
|---:|---:|---:|---:|---:|
| 16K | 32.69 t/s | 51.34 t/s | 45.10 t/s | +13.84% |
| 65K | 27.60 t/s | 36.48 t/s | 33.05 t/s | +10.37% |
| 127K | 23.10 t/s | 34.89 t/s | 32.02 t/s | +8.96% |

The previous promoted values are historical same-harness results, not a new interleaved depth bracket. They are supporting evidence; the 30-prompt A/B is the primary performance gate.

## Service gate

Two production-form cold starts loaded the Heretic Q6_K model and matching BF16 multimodal projector. Both passed health before and after the frozen sensitive requests, reproduced every promoted content hash, and remained alive. The image smoke answered `The first human moon landing (Apollo 11).`

## Candidate identity

- Source: `worktrees/n2-packed`
- Build: `builds/n2-packed-make`
- `llama-server`: `10856b0b2fcab07c83a7fc0f01f1a0537b130bfdabcf608f97f348f670289393`
- `libggml-hip.so`: `5f36565e00881c19d8f189661527cd70e3d7ecd2e5a40a13ac6d06c69c380091`
- New all-family gate: `GGML_CUDA_Q6K_CANONICAL_PACKED_ALL=1`
- Existing `GGML_CUDA_Q6K_CANONICAL_FFN_PACKED=1` behavior remains unchanged.

## Evidence

- `evidence/KERNEL_GATE.json`
- `N2_PACKED_AB.json`
- `N2_PACKED_ISOLATION.json`
- `evidence/stability/STABILITY.json`
- `evidence/service-soak/SERVICE_SOAK.json`

## Live promotion

The user launcher now delegates to `tools/start-apex-qwen38-heretic-n2-packed.sh`. Two launcher-driven live activations passed health, all four sensitive hashes, and BF16 multimodal inference. The second activation is running as PID `867874` on port 8083 with the expected candidate binary and packed-all environment gate visible in `/proc`.

- User launcher: `/home/adam/scripts/start-qwen38-heretic.sh`
- Logs/state: `/home/adam/.local/shared/apex-qwen38-heretic`
- Rollback wrapper: `/home/adam/scripts/start-qwen38-heretic.sh.pre-n2-packed-20260822`
- Live evidence: `evidence/live-promotion-cycle0.json` and `evidence/live-promotion-cycle1.json`

## Decision

Packed-all is the promoted N=2 production path. Keep recurrent-core and attention-only as separately measured positive fallbacks. Do not promote the control-only variant. N=3 work may resume from this new production baseline.
