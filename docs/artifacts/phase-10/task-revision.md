# Phase 10 task revision

The original frozen contract is preserved as `task-v1.json`.

Its whole-PP runner used `argparse.REMAINDER` but forwarded the separator token
`--` to `llama-bench`. Attempt `p10-b` therefore exited at argument parsing and
did not produce a model measurement. The revised runner strips only that
separator before launching the otherwise unchanged registered PP command.

Because the runner hash is part of the target fingerprint, `task.json` was
refrozen after this correction. Evidence captured under the original fingerprint
is retained for audit but is not used for the final gate.

The original evaluation plan is likewise preserved as `evaluation-plan-v1.json`;
`evaluation-plan.json` carries the revised task fingerprint used by the final
`p10-v2-a` evidence.
