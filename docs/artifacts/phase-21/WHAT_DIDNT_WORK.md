# What Did Not Work

- An attention mask and K/V compaction alone are not sufficient for Qwen3.6;
  48 Gated DeltaNet/conv layers require parent-aware tree state.
- Official DDTree's `DynamicCache` implementation is not a hybrid-recurrent
  correctness proof.
- Pinned Lucebox supports single-device tree verification but explicitly
  disables its production multi-GPU layer-split tree path.
- The external DDTree and R9700 results are not transferable to Apex Q6_K
  tensor split and were not reproduced.
- Static file/memory arithmetic cannot establish per-device VRAM fit.
- No build, GPU run, throughput claim, tree candidate, or promotion occurred.
