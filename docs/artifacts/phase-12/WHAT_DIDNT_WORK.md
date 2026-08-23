# WHAT_DIDNT_WORK

Phase 12 did not find a trivial producer output-pointer change. PP512 Q6_K
MMQ consumes backend-private `block_q8_1_mmq` D4 storage, not a normal GGML
Q8_1 graph tensor. The current MMQ entrypoint asserts an F32 source, allocates
its own temporary, quantizes, consumes the temporary, and releases it on
return. Either direct producer therefore requires a new exact graph/backend
fusion boundary and a prequantized MMQ entrypoint.

RMSNorm-to-D4 is not selected despite the larger 38.304% connected MMQ share.
It must combine a whole-5120-value row reduction and learned scaling with
128-value quantization reductions, then retain one private buffer across two
MMQs. Phase 10 already showed that merely sharing one quantization, without
removing the F32 producer boundary, is flat (`0.999896x` exact pair and
`1.003627x` whole PP). That partial idea must not be retried under Phase 12.

No performance result was collected in this feasibility phase. Logical F32
write/read bytes and observed producer/quantizer durations are not promised
speedup, achieved bandwidth, or wholly removable time.
