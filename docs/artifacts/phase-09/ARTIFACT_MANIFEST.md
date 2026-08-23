# Phase 9 Artifact Manifest

Phase 9 is source/trace feasibility only. No GPU workload, build, candidate
integration, code object generation, correctness result, timing result,
occupancy result, bandwidth result, or E2E result exists. The 71,303,168-byte
intermediate figure is **modeled**, not measured.

| Artifact | SHA-256 |
|---|---|
| `FEASIBILITY.md` | `59cc809b0d605f75e43d278c1308aa7f7b364bafc59e4133ed3f87604704e92b` |
| `FINDINGS_SCHEMA.md` | `d8531580ab9499430a3afd11f8ae9d15637c4dc0ab91962459d0a3c1d1109a72` |
| `feasibility.json` | `b252e975d4b8e6de0f472ee3985510a8e3a7b2f6035358ad504e2661199a1cfd` |
| `feasibility-result.json` | `22bb087b8b1390f7f80772318284232ff7de041baee0e770bbee75bfd4814ab7` |
| `findings.json` | `cae0e70b9b9ead4994bef1e4fdd27c3d0991ee97f85db4b3a7a7144652767839` |
| `pair-proof.json` | `1bd1a49144936203df703870ea7e993b77cfb8a80d9835f7ccd3ec5eadf346a9` |
| `task.json` | `506f6b44cf750cac78d5f3a1827d646b771edb6e7517786b509829a3188b1914` |
| `evidence/source-hashes.txt` | `9671713cfa2ec219b16113f3c94fe66fd2d39d532d42962e60fba9e19aed5594` |
| `tools/finalize_feasibility.py` | `d9e2e1fc776b2f89803abd9a8b80b707843c078ea1bfca4ba9ae91639d91656c` |
| `tools/make_findings.py` | `4cc9971abcf32d57ff70cf7b380501f04fff678d7f709f7eb3d169e6da24f856` |
| `tools/make_pair_proof.py` | `3763eafe60639085c6201b5b777758e333432d0314c8a2538847faf9f1a01936` |
| `tools/make_task.py` | `0019c823b467ef3d93d86fde0de214a16b759ea9a56734921c511a85b13a298a` |

Input and source hashes are additionally frozen in `task.json`,
`pair-proof.json`, `findings.json`, and `evidence/source-hashes.txt`.
