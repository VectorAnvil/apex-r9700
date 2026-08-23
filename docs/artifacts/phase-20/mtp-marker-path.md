# Fail-Closed MTP Verification Marker

The hot N=5 work belongs to target-context verification, not the separate MTP
draft context. `LLAMA_CONTEXT_TYPE_MTP` and `LLM_GRAPH_TYPE_DECODER_MTP` cannot
identify it.

## Ownership path

1. `server_slot::handle_last_sampled_token()` knows when `spec_draft` is
   nonempty. Extend the server-private token sidecar with an `mtp_verify` row
   bit and mark the sampled-plus-draft rows.
2. `server_batch::get_view(off,n)` derives `all_rows_mtp_verify`. It is true
   only when every row in that decode sub-batch is marked. Mixed prompt,
   ordinary, or multi-slot views fall back.
3. Add an explicit decode mode to a new decode entrypoint; the existing
   `llama_decode()` remains the default wrapper. Do not use context-global
   mutable state.
4. Carry the mode through context processing into `llm_graph_params` and add it
   to `allow_reuse()`. Ordinary and MTP-verification graphs must never share a
   cached or captured topology.
5. During graph construction, set a newly audited backend-visible tensor flag
   on MUL_MAT result nodes created in verification mode. A dedicated flag is
   preferable to a name string, TLS variable, or unreviewed op-param byte.
6. CUDA reads the flag at both `ggml_cuda_mul_mat_vec_q()` and
   `ggml_cuda_op_mul_mat_vec_q()`. Only
   `gfx1201 && Q6_K && N==5 && even_local_M && mtp_verify` may launch the new
   R2 symbol.

The marker must be present before graph capture and included in graph reuse
identity. Tensor split, P2P, all-reduce, quantization, and ordinary graph nodes
remain unchanged.

## Threat model

- Shape-only N=5 is forbidden because it changes ordinary and multi-slot
  batches.
- A context-type check marks the MTP draft graph, not target verification.
- A batch-level true value is unsafe when ordinary and verification rows mix.
- Mutating a captured graph or using a global/TLS flag can replay the wrong
  specialization.
- If the tensor flag is lost during graph transforms or backend splitting, the
  selector must fall back rather than infer provenance.
