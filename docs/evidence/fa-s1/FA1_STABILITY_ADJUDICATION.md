# FA-1 Stability Adjudication

## Decision

**Pass. Register FA-1 without further Flash Attention tuning.** The occasional
deep-context slowdown is not coupled to the restored rocWMMA kernel. It occurs
at process startup in both native and candidate builds, survives graph disable,
and disappears after the host/device runtime has been preconditioned. A clean
24-process rerun then held every affected cell below 0.81% spread.

FA-S1 does not change the FA-1 selector, tile, LDS use, `ncols`, staging, or
combine pass. PR #26419 is not mixed into the candidate.

## Frozen Runtime

- Current llama.cpp base: `4695f001fece1660d8bb1b3748f50726ddcc100b`.
- Candidate `llama-bench` SHA-256:
  `0df6eee7e7af03c0e97243493f2ea6dd9062660a046c7bcd299bc8a414a9431a`.
- Candidate `libggml-hip.so` SHA-256:
  `69ce00e9724216cd28789b5fb70c6c1fb87a8aa2cf147427af2ddc20a4a25953`.
- Candidate PP specialization: `<256,16,4,64,float,false>` plus the historical
  combine pass.
- Decode remains on the current native tile specialization.
- Both physical R9700 devices previously passed the final 126/126 correctness
  matrix.

## Collapse Adjudication

Collapsed and healthy processes emitted the same normal ROCm discovery log.
The following bounded probes were then run against TG128 at depth 126,976:

| Probe | Result |
| --- | --- |
| Graphs enabled, three processes | `24.855`, `29.288`, `29.293` t/s |
| Graphs disabled, three processes | `18.798`, `28.777`, `28.772` t/s |
| One process, five repetitions | `29.322` to `29.869` t/s |
| Cold monitored five-repetition process | `29.364` to `29.854` t/s |
| Cold monitored single repetition | `29.370` t/s |
| Cold unmonitored single repetition | `29.263` t/s |

This rules out HIP graphs as the trigger, rules out SMI polling as the trigger,
and finds no within-process degradation. Waiting until both GPU edge
temperatures were at most 36 C also did not reproduce the collapse after the
runtime had been exercised. Healthy timed samples showed approximately 3.0 GHz
core clocks, 1,258 MHz memory clocks, and 100% GPU use.

Both cards were in the ROCm `BOOTUP DEFAULT` power profile. A `COMPUTE` profile
A/B could not be performed because changing it requires privileged interactive
sudo. The narrowest supported characterization is therefore an intermittent
early process/device-state effect after long idle, likely in power-management
or driver initialization. Exact attribution remains unavailable without a
privileged profile A/B or a longer reboot/idle campaign. It is not evidence of
an FA-1 correctness or kernel stability failure.

## Clean Affected-Cell Rerun

The adjudication harness first ran one candidate PP4096/d0 process to
precondition the runtime, then launched three independent processes for each
affected build/workload pair. All 24 processes exited successfully and no
sample was removed.

| Cell | Current native | FA-1 | Delta | Worst spread |
| --- | ---: | ---: | ---: | ---: |
| PP512 / 65k | 494.667 | 719.361 | +45.42% | 0.804% |
| PP512 / 127k | 333.592 | 560.634 | +68.06% | 0.167% |
| PP4096 / 127k | 333.911 | 569.339 | +70.51% | 0.075% |
| TG128 / 127k | 29.337 | 29.307 | -0.10% | 0.440% |

The earlier FA-1 PP4096/127k median of `543.01` t/s was conservative relative
to this clean rerun. The registered candidate now has a reproducible
`333.911 -> 569.339` t/s result at that workload, while deep-context decode is
flat within noise.

Exact samples are preserved in `PRECONDITIONED_AFFECTED_MATRIX.tsv`. Raw logs,
telemetry, probe scripts, and all untrimmed outputs remain under the ignored
`results_fas1_stability_20260817/` result root.

## Promotion Boundary

FA-1 is registered in Llama Lab as read-only build ID 4:
`Apex FA-1 current-source rocWMMA gfx1201`. Registration validation passed for
the exact `llama-bench` and `llama-cli` paths. The database was snapshotted to
`llama-lab-pre-fa1-registration-20260817.db` first.

An independent post-registration PP4096/127k process launched through the exact
registered `llama-bench` path and measured `569.546 t/s`, matching the clean
rerun median of `569.339 t/s`. Both GPUs were idle before launch and the process
exited zero.

The older dirty Phase-13 worktree/build was not overwritten. Registration did
not itself repoint the separately hash-pinned Apex Qwen3.8 service launcher.
That independent server build and end-to-end promotion subsequently passed and
is recorded in `FA1_VIVI_SERVING_PROMOTION.md`.
