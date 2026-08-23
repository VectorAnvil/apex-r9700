# Phase 21B Handoff

Phase 21B validates the DFlash asset and official DDTree host policy, but it
does not authorize tree integration.

The DFlash model passed exact 192-token control parity in a two-R9700
layer-split compatibility lane. The registered tensor-split production lane is
blocked because the draft inherits `-sm tensor` and forms an unsupported
`Meta()` context. The next implementation should add a draft-specific split
mode so the target remains tensor split while DFlash uses layer split with both
devices visible. It must be isolated and must not touch target kernels, P2P, or
the registered build.

The official DDTree snapshot is clean at `c96427a...` and its best-first host
policy passed independent invariants through budget 32. No real top-K replay or
tree GPU path ran. After tensor-mode compatibility, add a passive raw-logit
top-32 recorder for all 15 DFlash positions, then replay pure best-first and
chain-seeded policies offline. Parent-aware recurrent-state work remains gated
on a material, repeatable policy advantage.

Evidence:

- `docs/apex-r9700/evidence/phase-21/LINEAR_DFLASH_DDTREE_VALIDATION.md`
- `docs/apex-r9700/artifacts/phase-21/phase21b-final-result.json`
- ignored raw root `results_phase21b_linear_dflash_validation_20260809/`

The registered library remains
`facd1354c4eba6afec9af0b22694e6ca11bf2b2bd976368158f225d6692c4311`.
