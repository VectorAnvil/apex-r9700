# Phase 21B: What Didn't Work

- Registered tensor split is inherited by the DFlash draft and creates an
  unsupported multi-device `Meta()` context. The server aborts before inference.
- Restricting the tensor-split draft to `ROCm0` does not remove the inherited
  tensor aggregator.
- Layer split with a `ROCm0`-only draft loads the model, but cannot access the
  target `output.weight` on `ROCm1`.
- Layer split with both devices visible passes, but it is a compatibility lane,
  not the registered tensor-split production geometry.
- One successful request is not a benchmark. No top-32 recorder, real DDTree
  policy replay, tree GPU execution, or recurrent-state validation ran.
- The official DDTree module imports a CUDA/PyTorch stack and has no tests. Its
  data-only policy was validated through a result-local standard-library oracle.
