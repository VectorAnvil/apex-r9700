# Phase 20 Artifact Manifest

Raw evidence root:

- `results_phase20_q6k_mtp_n5_feasibility_20260809/`

The compact directory contains the immutable task, identity, source/ISA maps,
shape inventory, design comparison, marker path, GO decision, separate Phase
20B contract, final result, and failed-path record. The raw root additionally
retains registered source copies, full AMDHSA notes, and complete N=5 and Q6_K
R2 precedent disassembly.

Phase 20B subsequently built its plumbing baseline and R2 candidate under the
separate ignored root `results_phase20b_q6k_mtp_n5_r2_20260809/`. The compact
`phase20b-final-result.json` records the hard static rejection: R2 used 80
VGPR and 34 SGPR against 64/32 limits. No GPU workload or registered asset was
modified.

```text
07dd3b633537bfdb100020caab550a81a3cbc1300ec4a6db7eb23241f53ff88e  dependency-reuse.md
f61f5ddfc7c9723c8ef65c6aee9ab0c931fa6416866eec5359fafb612f48b148  design-comparison.json
05a6800098e5d6784a079bfdf1a9399e3fc9d137f0115c8836e0ff87195603b9  final-result.json
f3ecc754b5914aaa64a6a6647724005c2e53cc1e5df18bf82a7ad3425455d5f8  go-no-go.json
2c976e2db858e47b9d65f1632b1172ea5269b4fea45069614446c796124d99b9  identity.json
259832a17bdd010bd9fa45c3893d14c8e1014756a4899ed6287580d6b42c1efa  isa-map.json
843b484e1105135bb9a751c316255657100960465cc31ea23b19b38a1e485ed1  mtp-marker-path.md
9f472bcaf05e68d8e36761136b35a64944e617464330f45f7c75079caeaef68d  phase20b-contract.json
8c647dcdfc49e7eacd5992f76e8257f8f7c773465b8e5fc4a0951f28fd567c33  README.md
29473ee9b00651f71d7d7337f22d11bb5416b009436273171117cec946bd7775  shape-inventory.json
aaef6726d741a095fc100226b92fcfa37c1e34a0727468c122cf6ef42d115808  source-map.json
17691b588b6d76003f0b2266c321b9e6ba9a631c4ea0825d0056cfd3d400f16f  task-frozen.json
0a978714be10d0d8f74e26d7d6cfb0ed25bb8674d94c7f7dfd002de75c8da477  WHAT_DIDNT_WORK.md
```
