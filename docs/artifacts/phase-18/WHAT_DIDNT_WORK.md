# What Didn't Work

- K2 and native dot8 do stack in the actual fused Q6_K N=1 llama.cpp kernel.
  The candidate retained next-K VMEM between current-primary and current-gate
  arithmetic, so this is not a failure to combine the mechanisms.
- The combined arithmetic still needs four dot8 reductions per independently
  scaled four-value group. In the full fused symbol, eight productive DP4As
  became 32 native `v_dot8_i32_iu4` instructions.
- The dot8 construction did not reduce memory traffic. Both K2 variants emit
  30 fused-symbol global loads; the difference is arithmetic and live state.
- Fused VGPR rose from 60 to 66, exceeding the immutable 64-VGPR ceiling.
  SGPR stayed 42, LDS stayed 1,792 B, and no private memory or spills appeared.
- The fused ISA slice grew from 918 to 1,095 instruction lines. Nibble packing
  and reconstruction sharply increased shifts, masks, ORs, and ALU waits.
- The unfused symbol remained byte-identical, proving the fail-closed selector
  worked and that the negative result belongs to the intended fused path.
- The static gate required rejection before GPU work. No correctness run,
  timing run, TG128 run, or registration was performed, and the registered
  Phase 13 library remains untouched.
