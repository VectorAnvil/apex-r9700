# Phase 10 Handoff

## Outcome

**Rejected, not registered.** The default-off gfx1201 Q6_K PP512 paired
dispatcher was correct but did not meet the performance gate. It retained the
two existing gate/up MMQ projections and current F32 SwiGLU, while quantizing
their common activation once at exact `(M,N,K) = (8704,512,5120)`.

## Evidence

- Both devices showed the intended dispatch change: `quantize_mmq_q8_1` went
  from two launches to one; two Q6_K MMQ launches remained.
- Selector-off and nonmatching fallback witnesses retained the baseline path.
- CPU/F32 parity, normal HIP graph capture/replay, and per-device
  tensor-parallel ownership passed.
- Three interleaved same-build rounds produced a combined exact gate/up result
  of `0.999896x`, with `0.6008%` relative spread.
- Three normal-graph PP512 processes produced `1.003627x` whole-PP speedup,
  with `0.2589%` relative spread.

The procedure was stable; the candidate simply did not create a meaningful
isolated gain. No candidate registration or general enablement is authorized.

## What This Means

Removing one quantization launch is real but insufficient at this shape. Do
not continue tuning launch count or duplicate-quantization-only variants of
this pair without evidence that changes the limiting mechanism.

The next direction is intentionally not chosen here. It must start with fresh
profiling and a new immutable contract. The most plausible classes are a
resource-gated attempt to reduce intermediate traffic, or another exact PP
hotspot; neither is approved by this handoff. `ffn_down` remains a
Zinc-methodology reference only.
