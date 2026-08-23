# What Didn't Work

- Draft layer split alone could not execute target token/output tensors on the
  tensor-split `Meta()` backend.
- Adding the shared backend exposed a one-descriptor shortfall in the rotating
  Meta external-view pool.
- Enlarging a different Meta pool did not affect the failure and was reverted.
- The accepted pool fix raises documented external-view headroom 16 to 17.
- Draft widths 1, 4, 8, and 15 failed exact 192-token parity. Width 12 passed
  twice. Smaller verification batches were not monotonically safer.
- No benchmark, distribution recorder, DDTree execution, or promotion ran.
