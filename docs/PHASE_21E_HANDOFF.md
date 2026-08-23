# Phase 21E Handoff

Phase 21E closed before implementation because the registered llama.cpp lacks
the parent-aware recurrent substrate required for correct Qwen35 sibling-tree
verification. This is a structural no-go for a direct DDTree integration, not
a rejection of the Phase 21D policy result.

The next phase should isolate the substrate from the policy. Port or implement
parent-aware SSM convolution and Gated DeltaNet operations with an explicit
root-inclusive parent map and persistent per-node state capture. Validate
chain, root siblings, early branches, late branches, and mixed-depth trees
against a serial CPU/F32 oracle and serial GPU execution on both R9700s.

Do not attach DFlash policy selection, a 20-node production request, feature
gather optimization, or performance timing to that first substrate task. Once
the primitives pass, a later correctness integration can add tree masks, KV
ownership, accepted-path compaction, graph identity, tensor-split ownership,
and a 64-token continued suffix.

The registered Phase 13 library and all Phase 21C/21D evidence remain
unchanged. Raw Phase 21E evidence is under
`results_phase21e_qwen35_tree_correctness_20260809/`.
