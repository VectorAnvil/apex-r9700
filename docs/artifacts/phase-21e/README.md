# Phase 21E Compact Evidence

Phase 21E stopped at its pre-GPU structural correctness gate. The registered
llama.cpp cannot represent parent-indexed Qwen35 recurrent and convolution
state for sibling nodes, and the Lucebox split reference fails closed at the
same boundary.

- `task-frozen.json`: compact immutable scope and task hash.
- `final-result.json`: static no-go and successor boundary.
- `ARTIFACT_MANIFEST.md`: raw evidence identities.

Full source identities and the detailed audit remain in the ignored raw root
`results_phase21e_qwen35_tree_correctness_20260809/`.
