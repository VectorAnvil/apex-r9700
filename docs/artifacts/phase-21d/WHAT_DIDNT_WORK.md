# What Didn't Work

- The fixed `3x4` and `4x3` rectangles reduced mean canonical depth from the
  linear baseline's 2.60 to 2.02 and 1.86. Branching only at the first position
  spends too much of the budget while imposing shallow maximum depth.
- Pure best-first 20 is not monotonic cycle by cycle: it rescued 24 cycles but
  lost one token of depth in one cycle because it can omit part of the top-1
  chain. Chain-seeded 20 removed that regression at a small mean-depth cost.
- More nodes rapidly lose work efficiency. Best-first 20 used 4.76 nodes per
  committed-token proxy versus 3.33 for linear-12; 32 nodes reached only 3.36
  mean depth while rising to 7.34 nodes per committed token.
- Five terminal cycles had fewer than 12 requested positions because the
  server bounded speculation by remaining output budget. They are preserved
  but excluded from full-horizon replay.
- Recorder throughput is invalid as a performance result: exporting and
  normalizing top-32 added a median 13.49 ms per cycle.
- The replay uses serial canonical tokens. It does not prove that alternate
  branches have correct target logits, KV ancestry, DeltaNet/recurrent state,
  rollback, or graph identity. No GPU DDTree execution or promotion occurred.
