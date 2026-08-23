# Phase 17 Handoff

Phase 17 closes the standalone native-dot8 arithmetic idea as a static no-go.
The corrected exact Q6_K x Q8_1 decomposition passed 131,072 scalar oracle
cases and emitted native gfx1201 `v_dot8_i32_iu4`, but required eight dot8s
versus two productive signed-int8 DP4As. Every dot8 used only four of eight
lanes because the two QR6 groups have independent scale and Q8_1 factors.

Matched code grew from 372 to 876 bytes, 60 to 158 instructions, and one to
16 ALU waits. Probe resources rose from 2/16 to 6/22 VGPR/SGPR without spills,
but the frozen lane-efficiency and dependency gates were already decisive.
No llama.cpp integration or GPU work was performed in Phase 17.

The user's subsequent request to see whether K2 could make dot8 useful was
handled as separately frozen Phase 18 rather than altering this result.
