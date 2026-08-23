# Q6_K Vec-Dot Load/ISA Audit

## Finding

**No-go: the registered gfx1201 compiler output already uses packed Q6_K and
Q8_1 payload loads.** The static audit found no q6k_dot4-style vectorization
gap that authorizes a candidate. No source was changed, no candidate was
built, and no GPU workload was run.

The exact registered unfused `mul_mat_vec_q<GGML_TYPE_Q6_K,1,false,false>`
code object emits one `global_load_b32` for `ql`, one for `qh`, and two for the
two Q8_1 `qs` payload words consumed by each lane. The corresponding source
uses `get_int_b2`/`get_int_b4`; LLVM combines the two 16-bit `ql`/`qh` pieces
into dword VMEM rather than scalarizing them.

| Field | Registered gfx1201 ISA | Wave-level access |
|---|---|---|
| `ql[128]` | one `global_load_b32` | lanes 0-31 cover 32 consecutive dwords exactly once |
| `qh[64]` | one `global_load_b32` | 16 dwords are covered; paired lanes reuse each dword with a different bit shift |
| `scales[16]` | two `global_load_i8` | the requested bytes are 4 bytes apart and each is shared by four lanes |
| Q8_1 `qs[32]` | two `global_load_b32` | eight-lane groups cover each selected payload contiguously with no repeated payload dwords |
| Q8_1 `ds` | two `global_load_b32` | one scale word per selected block, shared by its eight consumer lanes |
| Q6_K `d` | one `global_load_u16` | one super-block scale shared by the wave |

The fused `mul_mat_vec_q<GGML_TYPE_Q6_K,1,true,false>` specialization preserves
the same widths for both distinct Q6_K weight streams. It hoists the Q8_1 `qs`
and `ds` loads and reuses them across both dot products; the second Q6 load
sequence addresses the other fused weight tensor rather than reloading the
same bytes.

The repeated `qh`, scale, and block-scale addresses do not represent compiler
scalarization. They are operands shared by the cooperative lane mapping and
are issued as single wave VMEM instructions, so identical/nearby lane
addresses coalesce into cache-line transactions. Eliminating the repeated
lane values would require masked leader loads plus lane shuffles; changing the
C++ load type alone cannot remove them and would add exchange instructions.

## q6k_dot4 Comparison

The archived lighttransport `q6k_dot4` uses two 4-byte `__builtin_memcpy`
loads for `ql`, one for `qh`, four scalar signed-byte scale loads, and four
`float4` activation loads per `(half,lp)`. Its lane mapping covers consecutive
Q6 dwords. llama.cpp reaches the same packed Q6 payload width with fewer
source-level packed operands per lane because 32 lanes cooperate on a block.

There are two necessary corrections to the comparison:

1. `q6k_dot4` consumes FP32 activations. It has no Q8_1 `qs` or `ds` access
   pattern to copy.
2. It was introduced by commit
   `5ffd54467a7909a460817595e6816953e9e729ea`, after the Phase 11 pinned
   reference commit `60c7de682c4e3df6d833572ad244aa39172dba41`.

The only byte-width Q6 accesses in registered ISA are the two noncontiguous
scale bytes. q6k_dot4 itself performs four scalar byte scale loads, so it does
not expose a missing packed-scale implementation. The Q8_1 quant payload is
already dword-packed and coalesced.

## WHAT_DIDNT_WORK

The proposed opportunity was that the compiler might turn `get_int_b2`,
scales, or Q8_1 access into repeated byte/halfword traffic that an explicit
q6k_dot4-style vector load could fix. The exact code object rejects that for
`ql`, `qh`, and Q8_1 `qs`: all are dword VMEM. Scale bytes remain scalar
because the two values used by a lane are noncontiguous, and the reference is
more scalar there, not less.

The result is `complete_no_go`. A masked-load/shuffle redesign would be a new
cooperative mapping experiment, not a thin explicit-vectorization correction,
and static evidence does not justify paying its register and instruction cost.

## Provenance

- Registered llama.cpp commit: `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`.
- Registered gfx1201 code-object SHA-256: `23c590e31a77f55c0e3f702c740796ebd8261df7c6c0aa220ad94ce3cc6de835`.
- lighttransport archive HEAD: `5c49ba4c37a9c5bcf2a563c74fd4536e8034e47f`.
- q6k_dot4 introduction: `5ffd54467a7909a460817595e6816953e9e729ea`.
- Result-local record: `results_q6k_vecdot_load_audit_20260808/`.
