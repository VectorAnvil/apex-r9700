# What Didn't Work

- Native dot8 is packed i4 x i4. Q8_1 activation bytes therefore require low
  unsigned and high signed nibble extraction before any dot8 instruction.
- The two QR6_K four-value groups have independent Q6 scales and Q8_1 factors.
  Packing both into one eight-lane scalar dot loses that boundary, so each
  instruction has only four productive lanes.
- Exact arithmetic needs four signedness-specific dot8 reductions per group,
  eight per helper, versus two productive signed-i8 DP4A operations today.
- LLVM emitted all eight native `v_dot8_i32_iu4` instructions; it did not find
  a combined or cheaper form. Matched code grew from 60 to 158 instructions,
  with ALU waits `1 -> 16`, VGPR `2 -> 6`, and SGPR `16 -> 22`.
- The first frozen task encoded `H=sign4(high2 xor 2)`, which is not
  `high2-2`. Its oracle failure was preserved, then a corrected v2 task was
  frozen before any probe compile.
- The first v2 compile used llama.cpp-only `__vsubss4`/`__vsub4` helper names.
  The corrected standalone probe embedded the same saturating byte operation.
- `hipcc --genco` produced an offload bundle. Direct disassembly failed until
  the gfx1201 HSACO member was explicitly unbundled.
- No llama.cpp integration, GPU launch, benchmark, K2/K4 combination, or
  registration was attempted after the hard static failure.
