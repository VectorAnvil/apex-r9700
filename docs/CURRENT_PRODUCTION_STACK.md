# Current R9700 production stack

Status date: 2026-08-23

The promoted service is Qwen3.8 Heretic Q6_K on two Radeon AI PRO R9700 GPUs (`gfx1201`), using the llama.cpp generation-10457 build. This document records the active layers; it is not a promise that the flags transfer to other architectures, models, quantizations, or batch shapes.

## Promoted implementation layers

1. Direct-P2P tensor transfer, rebased and validated on the two-R9700 topology.
2. Phase 13 explicit F32 conversion of the Q6_K RDNA4 MMA accumulator before scale multiplication.
3. FA-1 restoration of the rocWMMA `<256,16,4,64,float,false>` decomposition plus its combine pass.
4. C11+C07+C03c guarded FFN paths.
5. Packed canonical-row Q6_K MMVQ for N=2 MTP verification.
6. Recurrent natural-split handling and the speculative KV padding guard.

The FA-1 path is frozen. Its stable long-context result was +70.51% for PP4096 at 127K, with TG128 at 127K changing by -0.10%. Do not merge unrelated tile, LDS, or PR #26419 experiments into the production path without a new isolated gate.

## Runtime gates

The promoted launcher exports the following feature gates:

```bash
export HIP_VISIBLE_DEVICES=0,1
export GGML_CUDA_FA1_MIN_NQ=32
export GGML_CUDA_Q6K_SWIGLU_D4_FFN_DOWN=1
export GGML_CUDA_Q6K_SHARED_Q8_GATE_UP=1
export GGML_CUDA_Q6K_CANONICAL_ROWS=1
export GGML_CUDA_Q6K_CANONICAL_STAGE=3
export GGML_CUDA_Q6K_CANONICAL_CONTROL_STAGE=2
export GGML_CUDA_Q6K_CANONICAL_FFN_STAGE=3
export GGML_CUDA_Q6K_CANONICAL_FFN_PACKED=1
export GGML_CUDA_Q6K_CANONICAL_PACKED_ALL=1
export GGML_CUDA_Q6K_CANONICAL_ATTN_STAGE=4
export GGML_CUDA_FA_CANONICAL_VEC=1
export APEX_SPEC_KV_PAD_GUARD=256
export APEX_RECURRENT_NATURAL_SPLIT=1
```

All gates must remain default-off in generic builds and restricted to their validated architecture, quantization, shapes, and speculative widths.

## Server contract

The registered service uses:

- context size 262144;
- parallel slots 1;
- CPU threads 12;
- batch 2048 and microbatch 512;
- 99 GPU layers;
- tensor split `1,1`;
- Flash Attention enabled and automatic fitting disabled;
- Q8_0 K/V cache;
- 20480 MiB RAM cache, prompt cache, and idle slots;
- embedded MTP with `n_max=2`;
- the model's matching multimodal projector.

Host-specific model paths, ports, log paths, and process-management policy remain in the deployment launcher and are not portable source configuration.

## Promotion evidence

- Direct P2P: 33.800 to 36.907 t/s TG.
- Phase 13 over P2P: 875.083 to 1018.723 t/s PP512.
- FA-1 stability matrix: PP512/127K 333.592 to 560.634 t/s; PP4096/127K 333.911 to 569.339 t/s; TG128/127K -0.10%; worst process spread 0.804%.
- Packed N=2 canonical rows: 49.8535 to 58.0286 t/s, +16.40%; 28/28 bit-exact kernel rows; 120/120 prompts in the expanded validation; long-depth gates passed.
- C11+C07+C03c: retained as individually guarded small gains; see the promotion ledger for exact component and combination measurements.

## Correctness boundary

The production claim is width-2 MTP for the promoted Heretic Q6_K model. DFlash width 3 remains 10/12 exact on the frozen suite and is not production-enabled. Width 4 and larger are also unpromoted. Base and HuiHui results are evidence of generality or remaining gaps, not permission to silently change the production model.

## Operational safety

Do not stop or restart a live model merely to run a benchmark. Confirm the GPUs are clear or obtain explicit permission first. Keep benchmark logs outside the repository and record only compact summaries and hashes here.

Rollback is performed by disabling the individual environment gate or selecting the preceding registered build. The older Phase 13 build and its recorded hashes remain intact.

## Related records

- [Promotion candidate ledger](APEX_PROMOTION_CANDIDATE_LEDGER.md)
- [Flash Attention track](FLASH_ATTN_LONG_CONTEXT_TRACK.md)
- [P2P and Phase 13 validation](QWEN38_P2P_PHASE13_VALIDATION.md)
- [C11+C07+C03c promotion](C11_C07_C03C_PRODUCTION_PROMOTION_20260819.md)
- [DFlash width-3 adjudication](DFLASH2_WIDTH3_ADJUDICATION_20260823.md)
