# Linear DFlash and DDTree Validation

## Result

The local DFlash GGUF is valid for the Huihui Qwen3.6 target. It loaded and ran
on the two R9700s in layer-split mode, producing exactly the same 192 token IDs
as the non-speculative control. Tokens 1-128 and the continued 129-192 suffix
both matched.

This does **not** yet validate the registered tensor-split geometry. DFlash
inherits the target split mode, and tensor split currently produces an
unsupported draft `Meta()` context. A draft-specific split-mode override is the
next compatibility change.

## Model Evidence

The draft SHA-256 is
`9153f73024e0b617392dfbd65bcae4ce5575392438e9564cc1a1308cac91c112`.
It is a 1.7B mostly-BF16 DFlash GGUF with 58 tensors, block size 16, hidden
width 5120, FFN width 17408, and extraction layers 2/17/32/47/62. The target
has matching hidden and FFN widths.

The accepted run used `-fit off -sm layer`, `ROCm0,ROCm1` for both contexts,
`n_max=15`, temperature zero, and a fixed seed. Startup proved
`draft-dflash`, block size 16, five extraction layers, and all six draft layers
offloaded. Loaded VRAM was 43% on GPU 0 and 42% on GPU 1.

The single request measured 24.165 tok/s for control and 36.692 tok/s for
DFlash. DFlash generated 754 proposals and accepted 137 across 53 calls, an
18.17% proposal acceptance rate and mean accepted length 3.58. These values are
a smoke observation only; no variance or promotion claim is made.

## DDTree Evidence

The official source was already downloaded from
`https://github.com/liranringel/ddtree.git` and remains clean at commit
`c96427a185677bf4133ed865dd1626a5041aef9b`. Git integrity passed.

Because the official module imports PyTorch/CUDA and contains no tests, a
dependency-free oracle ported only `build_ddtree_tree` and
`follow_verified_tree`. It passed budgets 0, 1, 3, 8, 12, 15, 18, 22, 28, and
32; exhaustive prefix-score ordering; deterministic replay; budget limits;
parent-before-child and depth invariants; ancestor-only visibility; and the
accepted-path walk.

This validates the host policy mechanics, not DDTree on Qwen3.6. Real DFlash
top-32 distributions, parent-aware DeltaNet/conv state, flat-tree target
verification, GPU route, memory, and throughput remain untested.

## Failed Attempts

The full failure trail is preserved. Auto-fit's Meta simulation first failed by
368 bytes. With fit disabled, tensor-split drafts still formed an unsupported
Meta context. Layer split with only ROCm0 visible loaded DFlash but could not
access the target output weight on ROCm1. Layer split with both devices visible
was the first valid execution.

The registered Phase 13 HIP library was unchanged and all GPU memory was
released after the run.
