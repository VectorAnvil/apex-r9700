# What Didn't Work

- Y64/W4 reduced LDS by 33.6% but left VGPR at 252/253 static and 256 runtime;
  SGPR rose slightly and Y workgroups doubled.
- Exact combined work was about `0.975x`; whole PP was `0.998916x`. Both were
  stable failures, and the candidate was not registered.
- Initial correctness and perf selections were absent from separate stock test
  lists. Identical test-only cases fixed both harness lanes.
- Idle guards rejected partial trace and PP attempts until longer cooldowns
  were used. Partial results were excluded.
- Dynamic gfx1201 occupancy/bandwidth remain unavailable; no static substitute
  was reported.
- Stream-K was not bundled or used as a rescue.
