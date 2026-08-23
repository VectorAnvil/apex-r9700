# Qwen3.8 Q6_K obliterated-model MTP parity — 2026-08-19

## Result

Heretic has the strongest deterministic MTP parity of the three Q6_K models on the exact promoted Apex runtime.

| Model | Exact cases | Failed cases | MTP acceptance |
|---|---:|---|---:|
| Base Q6_K, promoted control | 8/12 | `prose`, `math`, `list`, `forced256` | 74.648% |
| Heretic Q6_K | **10/12** | `list`, `forced128` | 75.000% |
| HuiHui Q6_K | 8/12 | `json`, `list`, `forced128`, `forced256` | **76.313%** |

Every ordinary lane and every MTP lane reproduced identically on its second run. The failures are deterministic numerical/state-path differences, not sampling noise.

## Forced-case comparison

| Case | Base promoted | Heretic | HuiHui |
|---|---|---|---|
| `forced64` | pass | pass | pass |
| `forced128` | **pass** | fail | fail |
| `forced256` | fail | **pass** | fail |

Heretic fixes `forced256` relative to both base and HuiHui, but it does not fix `forced128`. HuiHui has the highest aggregate acceptance yet the weakest case set, tied with base at 8/12. Acceptance is therefore not a valid parity proxy.

## Important promoted-stack finding

The historical pre-C11/C07 base Q6_K result was 9/12, failing `list`, `forced128`, and `forced256`. The contemporaneous promoted base is 8/12: it fixes `forced128` but newly fails `prose` and `math` while retaining `list` and `forced256`.

This confirms that the promoted Q6_K execution changes move near-tied deterministic parity boundaries. All model comparisons must use the contemporaneous promoted control; the old 9/12 result is not interchangeable with current production.

## Method

- Frozen 12-case production parity suite
- Ordinary decoding versus embedded MTP `n_max=2`
- Two repeats per lane and case: 48 completions per model
- Temperature 0, seed 1234
- Full production configuration: dual R9700 tensor split, Direct-P2P, Phase 13, FA-1, C11+C07+C03c, batch 2048, ubatch 512, 262K context, Q8 KV
- Exact content, token count, stop flag, and stop type required for parity

## Recommendation

If deterministic MTP agreement is part of the production choice, prefer Heretic over HuiHui. It also retains the same prompt-processing performance tier as HuiHui. This is still not perfect parity: `list` and `forced128` remain deterministic failures, so MTP must not be described as token-stream-equivalent to ordinary decoding.

## Q6_K_L / UD-Q6_K_XL extension

The same frozen gate was subsequently run on HuiHui Q6_K_L and base UD-Q6_K_XL.

| Model | Exact cases | Failed cases | MTP acceptance |
|---|---:|---|---:|
| HuiHui Q6_K_L | 7/12 | `prose`, `list`, `forced128`, `forced256`, `long_input` | 74.642% |
| Base UD-Q6_K_XL | 8/12 | `list`, `reasoning`, `forced128`, `forced256` | 71.338% |

Both ordinary and MTP lanes reproduced exactly on their second runs. Neither higher-fidelity recipe improves parity. HuiHui Q6_K_L is one case worse than HuiHui Q6_K, while base UD-Q6_K_XL ties promoted base/HuiHui Q6_K at 8/12 and retains both forced-case failures.

Tensor composition confirms that these are materially different recipes:

- HuiHui Q6_K_L: 360 F32, 374 Q6_K, 132 Q8_0 tensors.
- Base UD-Q6_K_XL: 360 F32, 96 Q5_K, 131 Q6_K, 279 Q8_0 tensors.

The five-model exact-parity ranking is therefore: Heretic Q6_K at 10/12; promoted base Q6_K, HuiHui Q6_K, and base UD-Q6_K_XL tied at 8/12; HuiHui Q6_K_L at 7/12. Higher nominal quant fidelity is not monotonic with deterministic speculative parity.

## Evidence

- Aggregate machine-readable comparison: `results_qwen38_obliterated_parity_20260819/SUMMARY.json`
- Heretic raw/result: `results_qwen38_obliterated_parity_20260819/heretic/`
- HuiHui raw/result: `results_qwen38_obliterated_parity_20260819/huihui/`
- Promoted base raw/result: `results_qwen38_obliterated_parity_20260819/base-promoted/`
- Combined five-model summary: `results_qwen38_xl_parity_20260819/SUMMARY.json`
- HuiHui Q6_K_L raw/result: `results_qwen38_xl_parity_20260819/huihui-q6-k-l/`
- Base UD-Q6_K_XL raw/result: `results_qwen38_xl_parity_20260819/base-ud-q6-k-xl/`
