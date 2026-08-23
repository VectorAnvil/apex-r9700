# Phase 11 Handoff

## Outcome

**Complete no-go.** The registered llama.cpp Q6_K decode kernel has no
lighttransport-style idle-lane defect. Each active wave32 cooperatively owns a
Q6_K block. Only terminal whole-wave underfill remains, and the prior four-
and twelve-wave experiments already rejected the direct launch-width response.

Phase 11 changed no llama.cpp source, built no candidate, ran no performance
workload, and registered nothing. Fresh capability discovery confirmed two
64-CU gfx1201 R9700 devices with wave32 execution. Existing matched HSACO/ISA
evidence retained static register, LDS, private-memory, and spill facts;
achieved occupancy and bandwidth remain unavailable.

## Next Boundary

Proceed to Phase 12 as a source/trace feasibility study for eliminating a
producer-to-Q8_1 materialization boundary:

- RMSNorm output directly into the Q8_1 activation consumed by gate/up MMQ.
- SwiGLU output directly into the Q8_1 activation consumed by `ffn_down` MMQ.

First prove the exact graph, tensor, ownership, layout, and dispatch boundaries
on both devices. Do not implement a fused kernel, change quantization
semantics, or reuse Phase 10's rejected duplicate-quantization-only dispatcher
without a new immutable contract. Phase 13 overlap and upstream Q8_1/MMVQ
layout work remain reference-only.
