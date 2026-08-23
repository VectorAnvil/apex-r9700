# Qwen35 Parent-Aware Tree Correctness Finding

## Result

Phase 21E is a static `complete_no_go`. No candidate was built or run.

The Phase 21D distributions still justify a 20-node DDTree experiment, but
the pinned llama.cpp cannot yet execute that tree correctly for Qwen3.6. Its
recurrent memory supports a linear sequence with bounded rollback snapshots;
it has no per-token parent input, parent-aware SSM convolution, or parent-aware
Gated DeltaNet operation.

An ancestor-only attention mask solves only the dense-attention/KV portion.
For each of the model's 48 recurrent layers, a sibling must also inherit the
exact DeltaNet matrix and convolution window of its own parent. Processing DFS
nodes as one ordinary linear batch instead makes a sibling inherit the state
of the preceding DFS node, which is wrong whenever that node is not its
parent.

## Reference Evidence

Lucebox's monolithic Qwen35 target shows the required mechanism:

- a root-inclusive `parent_ids` graph input;
- `ggml_ssm_conv_tree` for ancestor-derived convolution history;
- `ggml_gated_delta_net_tree` for branch-point state reload;
- persistent per-node recurrent intermediates;
- accepted-path SSM restoration and convolution ancestry gathering;
- accepted-path KV and feature compaction.

This validates the architecture, but not our dual-R9700 path. Lucebox's own
layer-split implementation returns `false` from `supports_tree_verify()` and
rejects sibling topology at its pure-chain guard. Its comments explicitly say
sibling visibility and depth positions are not wired for layer-split
execution.

The registered and reference Gated DeltaNet files are also materially
different: 327 versus 690 lines. The SSM convolution files are 206 versus 299
lines. The reference change is a backend substrate, not a thin DFlash/DDTree
adapter.

## Decision

We did not create a mask-only or KV-only tree candidate. It could emit
plausible tokens while carrying incorrect state through the recurrent layers,
so its output and any timing result would be scientifically invalid.

A separate substrate phase is required before DDTree integration. It should
port only the parent-aware recurrent primitives and prove them against CPU and
serial GPU oracles for small chain and sibling parent maps on each gfx1201
device. Only after that passes should Apex add the 20-node tree interface,
accepted-path compaction, tensor-split graph identity, and the full 64-token
continuation gate.

## What Did Not Work

- Existing recurrent rollback snapshots cannot represent simultaneous sibling
  states inside one graph.
- Attention masks and KV ancestry cannot substitute for DeltaNet and
  convolution ancestry.
- Lucebox's monolithic tree code cannot be treated as proven multi-GPU code;
  its production split path fails closed for siblings.
- No build, GPU oracle, graph trace, benchmark, or promotion followed the hard
  structural gate.
