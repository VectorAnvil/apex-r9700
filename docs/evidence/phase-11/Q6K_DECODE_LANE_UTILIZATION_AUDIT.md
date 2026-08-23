# Q6_K Decode Lane-Utilization Audit

## Finding

**No-go: the lighttransport launch-width mechanism does not transfer to the
registered llama.cpp Q6_K decode kernel.** No source was changed, no candidate
was built, and no performance workload was run.

Both R9700s dynamically report `gfx1201`, 64 compute units, and physical
wave32 execution. The registered Q6_K N=1 path launches `block=(32,8,1)`.
With `QI6_K=32` and `VDR=1`, `kbx=tid/(qi/vdr)` assigns one Q6_K block to each
wave, while `kqs=tid%32` gives every lane in an active wave useful work. The
audit therefore found zero systematic idle lanes inside an active physical
wave.

## Terminal Underfill

There is terminal whole-wave underfill when the number of 256-element Q6_K
blocks is not divisible by eight:

| K | Q6_K blocks | K-loop slots used | Slot efficiency | Final inactive waves |
|---:|---:|---:|---:|---:|
| 3072 | 12 | 12 / 16 | 75.0% | 4 |
| 5120 | 20 | 20 / 24 | 83.3% | 4 |
| 6144 | 24 | 24 / 24 | 100.0% | 0 |
| 8704 | 34 | 34 / 40 | 85.0% | 6 |

This is real but is not the reference pathology. It occurs only in the last
K-loop iteration and leaves complete worker waves without another block; it
does not leave most lanes idle within waves that own work.

The exact fused `ffn_up` `8704x1x5120` operation remains material at 29.842%
of summed diagnostic TG dispatch duration. It has 16.7% slot loss by the table
above. Apex already tested the strongest direct response: four waves divide
20 blocks exactly, yet the Phase 4 fused result was approximately `0.9983x`.
Twelve waves also remained flat in Phase 6 (`1.0000x` and `1.0033x` on the two
devices). Another global or shape-gated wave-count experiment is not justified.

## Reference Correction

The pinned lighttransport branch uses one whole Q6_K block per scalar thread:
`for (b=tid; b<nb; b+=nthreads)`. At K=5120, only 20 of 64 threads initially
own blocks. Reducing its launch from 256 to 64 threads attacks genuine scalar
thread idleness and reduction overhead. That mapping is fundamentally
different from llama.cpp's cooperative 32-lane block processing.

The cited `36 -> 45.5 tok/s` result was also not an isolated dense Q6_K result.
It combined IQ2_S, IQ3_S, expert, and Q6_K rewrites on a Qwen3.6-35B-A3B
IQ3_S MoE workload. The dense Q6_K branch separately records a one-wave-per-row
microbenchmark success that regressed full decode from 63.615 to 64.850
ms/token and was removed. Those results reinforce the need for whole-workload
gates.

## Resources

The matched existing code-object evidence records 26 VGPR / 26 SGPR / 896 B
dynamic LDS for unfused Q6_K MMV and 35 VGPR / 42 SGPR / 1,792 B dynamic LDS
for the fused specialization, with zero private bytes and zero spills. These
are static code-object facts, not achieved occupancy.

Dynamic occupancy and achieved bandwidth remain unavailable. ROCm 7.2
dynamically lists `OccupancyPercent` and `FETCH_SIZE` for both gfx1201 devices,
but the prior exact nonmultiplexed captures timed out or produced no joinable
values. This audit does not substitute Instinct assumptions or derived tensor
traffic for those missing counters.

## WHAT_DIDNT_WORK

The proposed analogy was that llama.cpp might leave most useful lanes idle in
the same way as lighttransport's scalar-thread Q6_K kernel, making a narrower
launch a cheap decode win. Source arithmetic, measured launch geometry, and
matched wave32 code-object evidence reject that analogy: all 32 lanes in an
active llama.cpp wave cooperate on the assigned block.

Tail underfill does not rescue the proposal. The hottest K=5120 case can be
made arithmetically exact with four waves, and that already failed to improve
the fused hotspot. Twelve waves also failed. The result is therefore
`complete_no_go`; tail remapping remains demoted history rather than an
authorized candidate.

## Provenance

- Immutable task fingerprint:
  `97b597f7014179d7277dd584321c0813758102add929b93ac2b52edbcb193c82`.
- Registered llama.cpp commit:
  `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`.
- Registered `mmvq.cu` SHA-256:
  `6f74bf36e6439c1907c2381aeda2fa2b24f1faba31232bae0bf5e638e6f3eaf8`.
- lighttransport audit branch commit:
  `60c7de682c4e3df6d833572ad244aa39172dba41`.
- Result-local evidence:
  `results_phase11_q6k_decode_lane_audit_20260808/`.
