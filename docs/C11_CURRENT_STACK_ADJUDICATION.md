# C11 Current-Stack Adjudication

Date: 2026-08-19
Candidate: Phase 12 SwiGLU→D4 `ffn_down` producer
Baseline: llama.cpp build 10457 with Direct-P2P, Phase 13 Q6_K F32 conversion, FA-1 rocWMMA, Qwen3.8-27B Q6_K, dual R9700
Disposition: **confirmed positive PP; flat TG; not promoted**

## Result

C11’s historical positive whole-PP result was real. On the current stack, all
ten interleaved baseline/candidate pairs favored the candidate. The robust
paired-median result is:

`1017-class baseline → 1021-class candidate, 1.0048167x (+0.4817%)`

The reverse-order replication, which contained no collapse, measured:

`1016.475489 → 1021.209137 tok/s, 1.0046569x (+0.4657%)`

The first matrix contained one preserved baseline sample collapse to
`333.372 tok/s` inside an otherwise `1003–1020 tok/s` five-sample process.
No sample was removed. Consequently, the all-untrimmed arithmetic mean reports
an inflated `1.018074x`; it is retained in raw evidence but is not the effect
estimate. Process medians and paired medians consistently place the PP gain at
approximately `+0.4–0.5%`.

TG128 was flat after reverse-order replication:

- combined untrimmed mean: `36.020679 → 35.993958 tok/s`, `0.9992582x`;
- combined median: `36.022940 → 36.020432 tok/s`, `0.9999304x`;
- paired directions: three positive, three negative.

## Correctness and execution proof

- 18/18 `test-backend-ops` numerical-oracle cases passed: selector-off exact,
  selector-on exact, and selector-on nonmatching fallback, three repetitions
  on each physical R9700.
- 6/6 rocprof dispatch cases matched the expected decomposition. The exact
  candidate replaced one ordinary SwiGLU and one Q8_1 quantizer with one fused
  producer while retaining three Q6_K MMQs. Fallback remained unchanged.
- A real Qwen3.8 PP512 trace observed 256
  `quantize_swiglu_mmq_q8_1_d4` launches, proving the model graph selected the
  candidate rather than merely passing a synthetic test.
- Normal dual-GPU graph execution completed throughout PP/TG testing. The
  candidate server log reported HIP graphs reused.

## Server/model-behavior smoke

The isolated candidate server used the production configuration: full 262144
context, tensor split 1,1, Q8 KV, FA-1, and embedded MTP width 2.

- health: passed;
- ordinary chat: passed;
- strict JSON-schema output: passed;
- forced tool call and JSON arguments: passed.

The first smoke used insufficient completion limits for this reasoning model.
Chat and structured-output requests stopped at length while still reasoning;
health and tool calling passed. That harness result is preserved at
`evidence/server-smoke/`. The corrected limits passed at
`evidence/server-smoke-v2/`. The first result is not classified as model or
kernel failure.

## Forensic conclusion

The Phase 12 exact-chain veto was a genuine false negative for whole-model PP.
Historically, the candidate produced `1.005222x`; the current-stack rerun
produced approximately `1.0047–1.0048x`. The direction and magnitude agree
despite the large intervening Phase 13 and FA-1 changes.

This establishes a real, repeatable PP gain—not merely a local prediction.
It does not establish that a sub-half-percent PP improvement warrants carrying
the five-source-file fusion in production. Promotion is a separate decision
and was not performed.

## Evidence

- Raw root: `results_phase04_15_c11_rerun_20260819/`
- Combined analysis: `evidence/ab-analysis.json`
- Correctness: `evidence/correctness/correctness.json`
- Synthetic dispatch: `evidence/dispatch/dispatch.json`
- Real-model dispatch: `evidence/model-dispatch/model-dispatch.json`
- PP matrices: `evidence/pp512/pp512.json`,
  `evidence/pp512-replication/pp512.json`
- TG matrices: `evidence/tg128/tg128.json`,
  `evidence/tg128-replication/tg128.json`
- Corrected server smoke: `evidence/server-smoke-v2/server-smoke.json`

## Production state

The original canonical server was restored after testing and verified healthy
at `127.0.0.1:8083`. PID `905493` executes the expected production binary with
SHA-256 `b5c8bb281e0e4abec2a18a73d5350a0a5d9bc31666447058d9592f1275a1df22`.
