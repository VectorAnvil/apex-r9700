# Hipfire Kernel Atlas Reference for Phase 3

## Scope

This is a read-only methodology review of `maxhbr/hipfire` at commit
`a99b46438a33e1471d7e871bbb32dd77d5e209f6` (2026-05-17). The primary
sources are:

- [`scripts/kernel_atlas.py`](https://github.com/maxhbr/hipfire/blob/a99b46438a33e1471d7e871bbb32dd77d5e209f6/scripts/kernel_atlas.py)
- [`docs/methodology/kernel-atlas.md`](https://github.com/maxhbr/hipfire/blob/a99b46438a33e1471d7e871bbb32dd77d5e209f6/docs/methodology/kernel-atlas.md)
- [`docs/methodology/kernel-atlas-architecture.md`](https://github.com/maxhbr/hipfire/blob/a99b46438a33e1471d7e871bbb32dd77d5e209f6/docs/methodology/kernel-atlas-architecture.md)
- [`tests/test_kernel_atlas.py`](https://github.com/maxhbr/hipfire/blob/a99b46438a33e1471d7e871bbb32dd77d5e209f6/tests/test_kernel_atlas.py)

No Hipfire code, runtime, model, profiler, or build is integrated into Apex.
Hipfire is MIT licensed, but this review recommends adapting small concepts to
Apex's existing safety and provenance boundaries rather than importing the
tool.

## Conclusion

Kernel Atlas validates the direction of Apex's Phase 2 adapter and offers
useful post-profile patterns for Phase 3. The strongest reusable ideas are:

1. immutable task/eval contracts derived from one hotspot;
2. fresh uninstrumented baselines with raw repeated samples and an explicit
   unstable state;
3. SHA-256 provenance for source, diff, harness, binary, and code objects;
4. optional code-object inspection for compiler resource and ISA evidence;
5. source/dispatch candidates with scores and reasons rather than asserted
   ownership; and
6. append-only attempt history that demotes a previously failed idea only for
   an exact matching target fingerprint.

Apex's separate-process rocprofv3 PP/TG collection is stronger than Atlas's
single-process text-section parsing and must remain authoritative.

## Methodology Mapping

| Topic | Kernel Atlas approach | Apex Phase 3 decision |
|---|---|---|
| Phase-separated prefill/decode | `parse_bench_profile_sections()` parses engine-emitted prefill and decode tables; `ar_rows_from_metrics()` emits separate rows. | Keep the existing separate llama-bench processes (`-pg 512,0` and `-pg 0,128`). Add phase, shape, and run identity to downstream contracts; do not adopt the text parser. |
| Profile to operation | `classify_kernel_op()` assigns family/role/phase from kernel-name rules. | Add a small declarative llama.cpp taxonomy for Q6_K MMV/MMQ, all-reduce, quantization, and normalization. Preserve the raw symbol and add `attribution_confidence`; a name rule is a hint, not proof. |
| Dispatch/source provenance | `collect_dispatch_manifest()` records symbol variants, ranked source files, dispatch references, source hashes, environment controls, scores, and reasons. | Extend the compiled-HIP mapper with candidate score/reason and dispatch-call candidates. Search the pinned llama.cpp GGML HIP source and compile commands. Keep Phase 2's low confidence until a compiled symbol/object relation strengthens it. |
| gfx1201-aware ranking | `source_file_score()` strongly favors `.gfx1201` source stems; a static capability table asserts RDNA4 wave32/WMMA. | Reuse conditional architecture-specific ranking only when the pinned build, detected GPU, and code-object target agree. Never copy the static capability table or infer counter/matrix support from the architecture string alone. |
| HSACO/code-object inspection | `inspect_isa_object()` optionally unbundles an AMDGPU target and runs ROCm LLVM metadata and disassembly tools. | Adapt as an optional argv-only inspector for the isolated candidate build after the actual loaded code object can be located. A missing tool, object, symbol, or field must produce `unavailable`, not zero. |
| Resource/spill extraction | `parse_isa_metadata()` reads AMDHSA notes for VGPR, SGPR, LDS/group segment, private segment, spill counts, kernarg size, workgroup limit, wavefront, and dynamic stack. | Reuse the field set and parser shape with checked-in ROCm 7.2 fixtures. Record raw tool output hash, tool version, metadata version, and exact symbol match quality. These fields inform hypotheses; they do not prove occupancy or bottleneck cause. |
| ISA summary | `parse_disassembly_stats()` counts instructions/opcodes and groups WMMA/MFMA, dot, VMEM, SMEM, LDS, branch, wait, SALU, and VALU. | Adapt the stable count categories, but retain exact opcode counts and disassembly hash. Do not turn instruction mix into an unsupported performance claim. |
| Binary/git/diff provenance | Atlas records a short Git SHA, dirty state, binary MD5, optional model MD5, and MD5 of status plus binary diff. | Use full commits and SHA-256. Include source/build/patch commits, changed-file list, binary diff and hash, compile command/hash, harness hash, binary and code-object hashes, model registry identity, ROCm/compiler/tool versions, GPU identities, argv, allowlisted environment, and input hashes. |
| Refreshed baseline | Profiling variables are stripped from a generated task and `requires_fresh_baseline` blocks comparison until `--refresh-baseline` is run. | Required. Phase 2 profiler/graphs-disabled throughput can never be a baseline. Capture a fresh normal-condition isolated-harness baseline with the same binary, inputs, GPU state checks, and evaluation contract. |
| Variance and instability | Warmups and repeated runs retain raw output; summaries include median, mean, min/max, standard deviation, MAD, and relative spread. Spread over a configurable threshold yields `unstable`. | Reuse the raw samples and summaries. Add confidence/uncertainty reporting and prefer interleaved or A/B/A measurement. An unstable result cannot win, be promoted, or establish a durable failed idea. Establish the threshold from the baseline noise envelope rather than accepting Atlas's generic 20% default. |
| Suggest to task to eval | A suggestion carries a deterministic ID, lever, hotspot, allowed files, rationale, steps, and eval contract. A task freezes target/evidence/constraints; eval runs correctness before timing and writes result plus ledger. | Adapt a versioned `apex.phase3.task.v0` contract. Commands remain argv-only and allowlisted. Freeze one kernel family, allowed files, reference/tolerance hashes, input/harness hashes, build command, benchmark command, baseline identity, and metric. Candidate code must not be able to alter the oracle, timer, or contract. |
| Prior-result history | Atlas scans task/result/ledger files and demotes matching no-win suggestions. Matching uses architecture, workload, phase, shape, quant, and suggestion identity. | Add append-only attempts/results. Match on a stronger fingerprint including exact source/build/toolchain/harness/input/kernel/code-object identity. Demote only stable, correctness-valid no-win results. Preserve compile failures, correctness failures, and unstable runs as evidence without treating them as universal rejection. |

## Small Implementations Worth Adapting

### 1. Experiment contract and ledger

Implement this first, before the first candidate. The task contract should be
immutable once evaluation begins and should contain:

- exact target fingerprint and evidence-selected raw/demangled kernel symbol;
- one allowed kernel family and explicit allowed write files;
- immutable correctness oracle, tolerance, input, and harness hashes;
- argv-only build, correctness, and timing commands;
- clean baseline identity and raw sample artifact references;
- source, binary, code-object, compiler, ROCm, and GPU provenance; and
- suggestion lever, candidate patch SHA-256, status, rejection reason, and
  parent attempt ID.

Evaluation should write `baseline.json`, one immutable `result.json` per
candidate, and an append-only `attempts.jsonl`. Never rewrite an earlier result.

### 2. Source and dispatch evidence enrichment

Extend the Phase 2 mapper rather than replacing it. Each source or dispatch
candidate should carry:

- path and line;
- SHA-256;
- match score and explicit match reasons;
- match mode: exact raw symbol, exact demangled symbol, normalized template,
  compile-command membership, or heuristic text;
- architecture/build-target agreement; and
- confidence that remains independent of rank.

For templated llama.cpp kernels, a high rank is not automatically high
confidence. The current Q6_K `mmvq.cu` mapping stays low confidence until Phase
3 ties the runtime symbol to a compiled object and launch/dispatch path.

### 3. Optional code-object inspector

After the isolated harness or candidate build exposes the relevant object,
adapt the Atlas inspection sequence using fixed argv:

```text
clang-offload-bundler --list --type=o --input=<object>
clang-offload-bundler --unbundle --type=o --targets=<observed-gfx1201-target> ...
llvm-readobj --notes <unbundled-object>
llvm-objdump -d --no-show-raw-insn <unbundled-object>
```

The output manifest should use SHA-256 and record:

- source container and extracted object hashes;
- observed bundle and AMDHSA targets;
- exact metadata/disassembly tool paths and versions;
- raw and demangled symbol join status;
- VGPR/SGPR counts and spill counts;
- LDS/group segment and private segment sizes;
- kernarg size, workgroup limit, wavefront, and dynamic stack;
- instruction/opcode/category counts; and
- bounded raw-output references and explicit tool/parser errors.

Do not recursively scan unrelated build trees. Inspect only registered or
isolated-build artifacts associated with the selected Q6_K target.

## Evaluation Policy for Phase 3

1. Run correctness before every timed candidate series.
2. Require target-GPU idle and Llama Lab protected-activity checks before
   baseline and candidate runs.
3. Use normal graph settings in the isolated harness; profiler-disabled graph
   throughput is diagnostic evidence only.
4. Record warmups but exclude them from headline samples.
5. Retain every measured sample and compute median, mean, standard deviation,
   MAD, min/max, relative spread, and uncertainty.
6. Prefer baseline/candidate interleaving or baseline/candidate/baseline when
   thermal and DPM drift can matter.
7. Mark noisy or drifting runs `unstable`; do not compute a winning claim from
   them.
8. Reject correctness failures before considering performance.
9. Treat ISA/resource changes as explanatory evidence, not correctness or
   speed evidence.

## History Fingerprint

Prior-result demotion should require equality on at least:

```text
gpu_identity + gfx_target + rocm/compiler + source_commit + source_diff_hash
+ build_config_hash + harness_hash + input_hash + phase + shape + quant
+ raw_kernel_symbol + code_object_hash + lever_type
```

A stable valid result below the configured win threshold may demote the same
idea. A different llama.cpp commit, shape, quant type, compiler, code object,
or harness is a new experiment. Unstable runs and correctness/compile failures
remain searchable history but should not automatically suppress a future
correct implementation.

## Explicit Non-Adoptions

- Do not replace rocprofv3 or Apex's separate PP/TG capture boundary.
- Do not add Hipfire as a package, submodule, runtime, or build dependency.
- Do not copy Hipfire's static architecture capability table.
- Do not reuse Hipfire paths, environment names, filename assumptions, or
  operation-name rules for llama.cpp.
- Do not use generic shell command execution from Atlas; keep Apex's argv-only
  allowlist.
- Do not use MD5 for scientific provenance.
- Do not assume a profiler name matched to an ELF symbol proves the unique
  source line or runtime dispatch branch.
- Do not use Atlas's default 20% spread threshold as Apex's acceptance policy.
- Do not let an unstable historical result demote an optimization idea.

## Phase 3 Order

1. Freeze the Q6_K harness, oracle, representative inputs, and target
   fingerprint.
2. Emit the task contract and capture a fresh repeated baseline.
3. Enrich the selected hotspot with scored source/dispatch evidence.
4. Locate and inspect only the relevant code object; preserve unavailable
   fields honestly.
5. Evaluate one candidate with correctness-first, repeated timing, variance,
   and exact patch/binary/code-object provenance.
6. Append the result to history and use it to inform, not predetermine, the
   next suggestion.

This review changes no Phase 2 implementation or evidence.
