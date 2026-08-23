# Phase 7 Procedure Record

## Scope

An isolated graph-enabled PP512 baseline and Q6_K MMQ host attribution capture
were run against the frozen registered profile command with an isolated
attribution executable. No profiler was used for the capture, and no candidate
kernel or registered asset was modified.

## Guard History

The v0 baseline procedure was stopped by the idle guard three times. Console
GPU-use pairs were `49/80`, `48/79`, and `82/81` percent; each reported 0 VRAM
use. Saved snapshots do not retain these high readings, so they are recorded
as console-only guard evidence, not as a claim of sustained external activity.

The initial v1 baseline invocation stopped because its parent output
directory did not exist. The directory was then created and the unchanged
hashed runner was invoked successfully. This was a procedure repair, not a
code or workload-contract change.

## Capture Contract

- Normal HIP graph settings; `GGML_CUDA_DISABLE_GRAPHS` unset.
- `GGML_CUDA_MMQ_ATTRIBUTION_FILE` and bounded maximum supplied only to the
  isolated attribution executable.
- Llama Lab protected-activity and target-GPU idle guards before workload.
- Raw stdout, stderr, activity snapshot, manifest, and recorder JSONL retained
  in `attribution-run-v1/`.
- Sequence analysis requires 496 records in each active/none state per device,
  six exact ordered trace repeats, and no unmatched or ambiguous geometry.

## Provenance

The immutable task, baseline result, attribution manifest/JSONL, original and
sequence summaries, builder hashes, patch, AMDHSA notes, HSACO, and fatbin
hashes are frozen by `analysis-task.json` and retained in this result root.
