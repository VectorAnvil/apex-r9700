# Apex DFlash / DDTree Feasibility Study

## Executive Decision

**Decision: `feasible_with_major_work`.**

DFlash plus DDTree belongs in a separate Apex research branch. The high-level
algorithm is real, the registered llama.cpp snapshot already contains a linear
`draft-dflash` implementation, and a compatible-shape Qwen3.6 DFlash GGUF is
already present locally. The missing work is not a new Q6 kernel. It is exact
tree-aware state propagation and commit for Qwen3.6's hybrid full-attention,
Gated DeltaNet, and convolution layers under two-GPU tensor split.

No performance gain is claimed. Phase 21 performed source, asset, and static
memory analysis only: no candidate build, GPU launch, or registered mutation.

The recommended first experiment is **linear DFlash instrumentation**, not a
DDTree implementation. It should establish compatibility, deterministic output,
acceptance, top-K coverage, total cycle cost, actual memory headroom, and target
dispatch shapes. Recorded distributions can then evaluate DDTree policies
offline before tree-aware recurrent code is authorized.

## Evidence Status

| Claim | Status | Evidence |
|---|---|---|
| DFlash produces all future-position logits in one draft pass | Proven | Registered `common/speculative.cpp:1111-1175`; official [DFlash paper](https://arxiv.org/abs/2602.06036) |
| The current Apex path preserves only one linear top-1 draft | Proven | `common/speculative.cpp:1163-1184` |
| DDTree builds alternate prefixes without rerunning DFlash | Proven | Pinned `liranringel/ddtree` `ddtree.py:84-166,363-405` |
| Root and all tree nodes enter one target model call | Proven for reference implementations | `ddtree.py:408-417`; pinned Lucebox `qwen35_dflash_target.cpp:421-486` |
| One target forward is the whole speculative cycle | False | Draft feature extraction, draft encode/injection, tree build, state commit, and communication are separate |
| Official DDTree K/V compaction is enough for Qwen3.6 | False | It handles `DynamicCache`; Qwen3.6 also has 48 recurrent layers |
| Lucebox proves the necessary recurrent-tree mechanism | Proven on its single-device target | Parent-aware graph and rollback source exists |
| Lucebox proves our two-GPU route | False | Its production layer-split `supports_tree_verify()` returns `false` |
| The local DFlash draft matches target dimensions | Proven statically | Hidden 5120; target layers 2/17/32/47/62; block size 16 |
| The Huihui target has useful DFlash acceptance | Unknown | Requires measurement; draft was trained for the base Qwen target |
| DDTree improves Apex end-to-end TG | Unknown | No local tree execution exists |

Primary snapshots are pinned in `source-provenance.json`: DDTree
`c96427a`, z-lab DFlash `94e4abc`, and Lucebox `22b4ad1`. External headline
results are reference evidence only. The [Qwen3.6 DFlash model card](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)
says engine support is still evolving; [Lucebox](https://github.com/Luce-Org/lucebox)
documents Qwen3.6 DDTree and HIP support, but its published hardware, target
quantization, topology, and runtime do not match Apex.

## How DDTree Actually Works

### 1. Draft Pass

The local DFlash draft is a five-layer, 1.7B, mostly-BF16 GGUF. Its metadata is:

```text
architecture       dflash
block size         16 (one seed plus up to 15 predictions)
hidden             5120
target layers      2, 17, 32, 47, 62
attention layers   four sliding-window, one full
file size          3,471,497,312 bytes (3.233 GiB)
SHA-256             9153f730...c91c112
```

Registered llama.cpp already supports this boundary. It extracts five target
layer inputs, gathers them into an F32 host buffer, runs `llama_encode()` in
the draft context, injects the resulting draft K/V through another
`llama_decode()`, then evaluates one seed-plus-mask block with non-causal draft
attention. The one block call exposes logits at all requested future positions.

This qualifies the phrase "one draft forward", but not "one GPU operation".
The current feature gather, encoder, K/V injection, and noise-block decode are
distinct work. They must be timed separately.

### 2. Best-First Tree Construction

The official DDTree code takes top-K token log-probabilities independently at
each future position. It begins with the top-1 token at depth 1 and uses a max
heap keyed by prefix log-weight. When a prefix node is popped, it can add:

- the next-ranked sibling at the same depth and parent; and
- the top-1 child at the next depth.

The child score adds the next position's marginal log-probability; the sibling
score replaces the current rank's marginal. Popping continues until the fixed
node budget is exhausted. Complexity is `O(B log B)` after top-K extraction.

This is exact best-first enumeration **for the factorized DFlash surrogate**.
The deeper distribution is not recomputed after choosing an alternate parent.
Lucebox calls out this same limitation in source: descendants beyond depth one
come from a single spine-conditioned block-draft pass. This is a modeling
qualification, not a target-output correctness failure, because the target
still verifies every accepted node.

Lucebox's default `chain_seed=true` is not identical to the paper/reference
policy: it reserves a full top-1 chain before spending the remaining budget.
Both pure best-first and chain-seeded variants must be evaluated explicitly.

### 3. Flat Tree and Ancestor Mask

The tree is flattened into arrays:

```text
token_id[node]
depth[node]
parent[node]
children[parent][token]
visibility[query_node][ancestor_node]
```

Every node receives the position `committed + depth`. Full-attention rows see
the committed prefix and only the node's ancestor chain. Siblings cannot see
one another. Shared prefixes appear once, so a prefix is not redundantly
evaluated for every descendant.

### 4. One Target Forward

The official reference calls the target once with the root plus all flattened
nodes and the ancestor mask. Lucebox builds a fixed-width `N = budget + 1`
graph, uploads positions, mask, and parent IDs, and issues one
`ggml_backend_graph_compute()` for all nodes.

That is the valid meaning of **one target forward**. It does not include:

- DFlash feature extraction, encode, K/V injection, or draft block compute;
- top-K projection and CPU/GPU tree construction;
- input/mask/parent uploads or graph setup;
- accepted-path walk and bonus-token selection;
- attention K/V compaction;
- DeltaNet and convolution state restore;
- target-feature ring compaction;
- tensor-parallel P2P/all-reduce and synchronization.

### 5. Accepted Path and Bonus Token

Target logits produce one target-selected next token for each verified node.
Starting at the root, verification follows the child whose token matches the
target selection. Walking stops at the first target token absent from the
current node's children. Nodes on the walk are committed; the unmatched target
token becomes the bonus token for the next step.

For greedy decoding this preserves the target sequence if state commit is
exact. General sampling requires a separately proven speculative acceptance
and residual-distribution contract; the first Apex tree experiment should be
temperature zero only.

## 3x5 and 5x3

Both proposed rectangles are valid if interpreted as independent chains under
one shared root:

| Policy | Draft nodes | Root-inclusive target N | Maximum drafted depth | Strength | Weakness |
|---|---:|---:|---:|---|---|
| Linear 1x15 | 15 | 16 | 15 | Maximum depth | One early miss wastes the suffix |
| 3 chains x 5 | 15 | 16 | 5 | Three early alternatives | Maximum path is much shorter |
| 5 chains x 3 | 15 | 16 | 3 | More first-token coverage | At most three drafted tokens per path |
| Adaptive budget 15 | 15 | 16 | 1-15 | Spends nodes by path score | Depends on marginal quality |

They are not full branching-factor trees. A complete 3-ary tree through depth
5 has 363 non-root nodes; a complete 5-ary tree through depth 3 has 155. Calling
either of those "15 nodes" would be a budget error.

The fixed rectangles also share only prefixes that are explicitly merged. If
three chains start with three different tokens, only the root is shared.
Adaptive DDTree is the stronger default hypothesis because it can allocate
depth to a high-confidence path and breadth near uncertain positions. The
rectangles remain useful controlled policies for interpreting top-rank rescue.

The maximum accepted path length matters as much as coverage. A 5x3 policy may
rescue more first-token misses yet still lose to a deeper policy because its
target forward always verifies 16 rows while committing at most three draft
nodes plus a bonus.

## Qwen3.6 State and Rollback

The registered target is not an attention-only transformer. GGUF metadata and
source establish:

```text
target blocks                 64 target + 1 MTP block
hidden                        5120
full-attention interval       4
full-attention layers         16
Gated DeltaNet layers         48
convolution kernel            4
SSM state                     128 x 128 x 48 values per recurrent layer
```

### Full-Attention K/V

Tree verification appends K/V for nodes in flat order. After the accepted path
is known, accepted DFS slots must be gathered into contiguous committed slots;
rejected nodes must not remain visible. This is the only state handled by the
official PyTorch DDTree reference.

### Convolution State

Each recurrent node needs the previous three convolution inputs along **its
parent chain**, not the previous three flat/DFS rows. After acceptance, the
last accepted node's three ancestor inputs become the live convolution state.

### Gated DeltaNet State

Every node's DeltaNet update must start from its parent's recurrent state. A
flat sequential scan would let one sibling contaminate the next. The graph must
materialize a per-node intermediate state, use `parent_ids` during computation,
and restore the deepest accepted node's state after verification.

Lucebox implements this concept with tree-aware SSM-conv and Gated-DeltaNet
operations plus per-node captures. Its rollback restores the accepted SSM
intermediate, gathers the accepted node's convolution ancestry, compacts K/V,
and realigns the target-feature ring.

Current Apex has linear recurrent rollback support for Qwen35, but no
`parent_ids`, tree SSM-conv, tree DeltaNet, tree mask, or tree rollback symbols.
Linear snapshot/`seq_cp` can evaluate branches level by level, but that costs a
target forward per depth and does not satisfy the hypothesis.

Most importantly, pinned Lucebox single-device tree verification is enabled,
while its production multi-GPU layer-split target explicitly returns `false`
from `supports_tree_verify()` because sibling visibility and depth positions are
not fully wired. Apex uses tensor split, so Lucebox is a design reference, not
a ready multi-GPU transplant.

### Target Hidden Features

DFlash conditions on five target layers. Features produced for rejected DFS
nodes must be discarded; accepted nodes must be compacted into the committed
sequence in path order. A stale or sibling-derived feature silently poisons the
next draft round even if K/V and recurrent state are correct.

## Apex Integration Map

### Reuse Directly

- DFlash GGUF loading and `src/models/dflash.cpp` graph.
- `draft-dflash` target/draft contexts and block-size handling.
- Target hidden-layer extraction and draft K/V injection.
- Existing sampler/speculative server lifecycle for a linear control.
- Qwen35 full-attention and Gated DeltaNet target graph.
- Linear recurrent rollback substrate.
- Phase 13 Q6_K MMQ float-cast optimization.
- Tensor split, direct P2P/all-reduce, HIP graph, and rocprofv3 evidence tools.

### Adapt

- DDTree heap policy, flat node arrays, child lookup, and accepted-path walk.
- Lucebox ancestor-mask, parent-ID, bonus-token, and state-compaction concepts.
- GPU top-K/argmax to avoid full `vocab x N` host copies.
- Fixed-allocation-width graph reuse with dynamic mask/parent tensors.

### Must Implement

- A tree result type in `common/speculative` carrying token, parent, depth,
  rank/log-probability, and child lookup.
- Top-K export from the DFlash block instead of discarding all but top-1.
- Tree batch positions and ancestor-only attention input.
- Parent-aware Qwen35 convolution and DeltaNet graph operations in GGML/HIP.
- Per-node recurrent intermediate storage and accepted-node restore.
- Accepted-path full-attention K/V and target-feature compaction.
- Tree-mode graph/cache key and replay-safe runtime inputs.
- Tensor-split ownership and synchronization for every tree state tensor.
- Greedy exactness tests and later a formal non-greedy sampler contract.
- Per-cycle instrumentation and memory high-water evidence.

### Do Not Bundle

- Rejected Phase 20B R2.
- K2/K4, native dot8, Y64/W4, or Stream-K.
- Draft quantization/conversion changes.
- Adaptive budget tuning in the first integrated tree candidate.

## R9700 Memory Estimate

The registered target file is 20.890 GiB and the local DFlash file is 3.233
GiB. File size is not runtime VRAM, and tensor split can replicate buffers, so
these are lower bounds only.

Static reference projections for the whole 64-layer target are:

| Allocation | Budget 15 (`N=16`) | Budget 22 (`N=23`) |
|---|---:|---:|
| DFlash weights, file lower bound | 3.233 GiB | 3.233 GiB |
| BF16 4096-token target-feature ring | 200 MiB | 200 MiB |
| F32 live SSM state, 48 layers | 144 MiB | 144 MiB |
| F32 SSM snapshot, 48 layers | 144 MiB | 144 MiB |
| Per-node SSM captures, Q8_0 reference | 0.598 GiB | 0.859 GiB |
| Per-node SSM captures, exact F32 | 2.250 GiB | 3.234 GiB |
| Convolution intermediates | 35.6 MiB | 48.8 MiB |
| Q8_0 full-attention K/V append | 0.53 MiB | 0.76 MiB |
| Full F32 logits before GPU argmax | 15.2 MiB | 21.8 MiB |

Lucebox's Q8_0 recurrent captures are a memory optimization, not automatically
an Apex correctness option. If committing a quantized intermediate changes the
next target token, it violates our deterministic target contract. The first
correct tree implementation should use exact state or prove bit/token parity
over long continuations before allowing quantized captures.

The local Apex DFlash path also gathers `5 * 5120` F32 values per processed
target token: 102,400 bytes copied through a host vector. That transfer and its
synchronization may matter more than the compact tree metadata.

**Fit conclusion:** the static lower bound looks plausible on 2x32 GiB, but fit
is unproven. Phase 21B must record actual free/used/high-water bytes per device,
draft placement, replicated allocations, context length, graph workspace, and
allocator fragmentation. Putting the entire draft on GPU0 may create imbalance;
splitting it may add communication. Neither choice is selected by this study.

## Expected Verification Shapes

For a tree budget `B`, the target sees `N = B + 1` root-inclusive rows. The
proposed sweep maps to:

```text
B:  8  12  15  18  22  28  32
N:  9  13  16  19  23  29  33
```

Registered gfx1201 Q6_K uses MMVQ only through `N <= 8`. Source dispatch
therefore predicts Q6_K MMQ for every proposed tree budget, unlike Phase 19's
dominant N=5 one-wave MMVQ path. This must be verified by trace; it is not a
timing claim.

Expected dense-block Q6_K shapes include:

```text
ffn_gate/up       M=8704,   N=B+1, K=5120
ffn_down          M=5120,   N=B+1, K=8704
linear qkv        M=5120,   N=B+1, K=5120
linear gate       M=3072,   N=B+1, K=5120
linear output     M=5120,   N=B+1, K=3072
lm_head           M=vocab,  N=B+1, K=5120
```

The Phase 13 cast applies to Q6_K MMQ arithmetic if this route is observed.
That makes DDTree architecturally interesting on Apex: it trades repeated N=1
or N=5 target work for fewer, larger MMQ passes. But PP512 throughput cannot be
used to predict N16/N23 cost. Attention masks, recurrent tree ops, LM head,
rollback, and communication are new costs.

Tensor-parallel all-reduce dispatch count per layer may remain similar per
target forward while payload grows with `N`; fewer target cycles may still win.
Phase 19's all-reduce shares (27.47% GPU0, 12.31% GPU1 of summed device kernel
duration) are overlap-sensitive diagnostics, not wall-time fractions.

## Instrumentation Plan

Record one row per speculative cycle with:

- prompt/request hash, seed, sampler settings, context position, and method;
- DFlash block size and full top-K IDs/log-probabilities at all 15 positions;
- tree policy, budget, node IDs, depths, parents, ranks, scores, and visibility;
- target canonical token rank at every position (top-1/2/3/5 coverage);
- first divergence and whether a sibling would rescue it;
- verified and padded node counts, accepted DFS path, bonus token, and committed
  tokens;
- target logical M/N/K, types, route, symbol, device, fusion, and duration;
- draft feature gather, encode, injection, block decode, top-K, tree build,
  target verify, K/V compact, recurrent restore, feature compact, and total
  cycle time;
- graph capture/instantiate/launch/reuse identity;
- per-device memory before load, after target, after draft, peak verify, and
  after rollback;
- direct P2P, all-reduce, copy, and synchronization sequence per device.

The decision metric is:

```text
committed target tokens / complete speculative cycle time
```

Acceptance percentage, accepted length, or target-forward duration alone is
not sufficient.

## Immutable Experiment Sequence

### Phase 21B: Linear DFlash Compatibility and Trace

Use the registered Phase 13 target unchanged and the local DFlash GGUF in an
isolated build/runtime. Run `draft-dflash`, block 15, temperature 0, fixed
prompt/seed. No tree or kernel changes.

Hard gates:

1. Asset metadata, tokenizer/output compatibility, target-layer IDs, hashes,
   and startup route all match.
2. Non-speculative target and linear DFlash produce the exact same 128-token
   output and the same continued suffix after rollback on both GPUs.
3. Record all top-K distributions and stage timings with no semantic join loss.
4. Capture actual two-GPU memory high-water and draft placement.
5. Capture target/draft dispatch, graphs, P2P, all-reduce, and synchronization.
6. Three interleaved normal-graph pairs are stable; report performance without
   a promotion decision.

### Phase 21B2: Offline Policy Replay

Using Phase 21B distributions and the exact target sequence, compare:

- linear 15;
- 3x5 and 5x3;
- pure best-first DDTree budgets 8/12/15/18/22/28/32;
- chain-seeded DDTree with the same budgets.

Compute predicted canonical-path coverage, rescue rate, maximum/mean accepted
depth, verified nodes per committed token, and policy sensitivity. This phase
does not estimate GPU speed and cannot authorize promotion.

GO to tree graph work only if at least one budget materially improves predicted
committed tokens per target forward over linear DFlash and current MTP, with the
gain surviving multiple representative prompts.

### Phase 21C: Tree-State Correctness Primitives

Implement only parent-aware conv/DeltaNet and state-commit primitives behind a
default-off tree mode. Compare flattened-tree results against serial evaluation
of every branch on CPU and each gfx1201 independently. Cover sibling walks,
depth shorter than kernel history, padding, rejected branches, and both exact
F32 and any proposed compressed checkpoint representation.

No end-to-end benchmark proceeds until tensor values, accepted live state, and
continued target tokens match the serial oracle.

### Phase 21D: Fixed Budget-15 Greedy Tree

Integrate one fixed allocation width (`N=16`) and the best Phase 21B2 policy.
Keep the Phase 13 target, Q6 kernels, and tensor split unchanged. Require:

- exact target output and continued-state parity;
- byte/dispatch identity for ordinary, MTP, and linear-DFlash fallbacks;
- 100% semantic join and expected Q6 MMQ routing on both devices;
- replay-safe graph mode/N identity and freshly uploaded mask/parents;
- equal tensor ownership and bounded P2P/all-reduce sequence;
- no OOM, leak, stale tree state, or rejected-node visibility;
- three stable interleaved E2E pairs and at least `1.005x` whole-generation
  throughput over the stronger of current MTP and linear DFlash.

Only an E2E win authorizes a budget sweep. Kernel optimization remains later.

### Phase 21E: Bounded Policy/Budget Sweep

Sweep only the predeclared budgets and policies, use identical prompts and
target output, and choose by committed tokens per total time. Refresh hotspot,
resource, graph, communication, and memory evidence before considering any
MMQ/tree kernel work.

## Risks and Likely Failure Modes

1. **Semantic mismatch:** the DFlash draft was trained for base Qwen3.6, while
   the target is a Huihui abliterated derivative; acceptance may collapse.
2. **Sibling state contamination:** a flat sequential recurrent scan can return
   plausible logits while corrupting future generations.
3. **Compressed rollback drift:** Q8 recurrent captures can change later target
   tokens even if the verification step appears correct.
4. **Tensor-split gap:** neither the official reference nor pinned Lucebox proves
   a production two-GPU tree path.
5. **Communication domination:** larger N raises activation/all-reduce payloads;
   fewer target calls may not compensate.
6. **Graph staleness:** reused captures can retain old parents, masks, or N.
7. **LM-head cost:** `vocab x N` logits and host transfers can erase the target
   batching benefit unless GPU argmax/top-K is used.
8. **Memory imbalance:** a 3.233 GiB draft plus exact tree state on one R9700 may
   constrain context or graph workspace.
9. **Policy illusion:** better top-K coverage can reduce maximum accepted depth
   and lose end-to-end.
10. **Concurrency:** tree verification can help batch size one yet lose when
    multiple server slots mix or scheduler batching changes.

## Recommended First Candidate

The next candidate is **not DDTree**. It is an isolated, recorder-only linear
DFlash A/B using the existing `draft-dflash` source path and local model.

Its job is to answer four questions cheaply:

1. Does this draft preserve the registered target output?
2. What are real top-1/2/3/5 coverage and acceptance on our prompts?
3. What does a complete DFlash cycle cost on dual R9700, including host feature
   gathering and communication?
4. Is there enough measured VRAM headroom for exact tree state?

If those answers are favorable, offline DDTree replay decides whether tree
engineering is justified. This keeps the strongest idea alive without betting
the project on an unmeasured, multi-GPU recurrent runtime rewrite.

## Sources

- [DFlash paper](https://arxiv.org/abs/2602.06036) and [official code](https://github.com/z-lab/dflash)
- [DDTree paper](https://arxiv.org/abs/2604.12989) and [official code](https://github.com/liranringel/ddtree)
- [Qwen3.6 DFlash model](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)
- [Lucebox GGUF metadata](https://huggingface.co/Lucebox/Qwen3.6-27B-DFlash-GGUF/blob/main/README.md)
- [Lucebox source](https://github.com/Luce-Org/lucebox)
- Raw pinned snapshots and hashes: `results_phase21_dflash_ddtree_feasibility_20260809/source-provenance.json`
