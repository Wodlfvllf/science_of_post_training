# Study 1: what holds the base model back?

## Scientific protocol

This is an inference-only study of **one checkpoint lineage**, not a controlled
comparison of training algorithms. No models are trained by this experiment.

### E1 — accuracy, elicitation, and sampled coverage

- Task: exact numerical answers to GSM8K word problems.
- Checkpoints: Olmo 3 7B base → Think-SFT → Think-DPO → Think-RLVR.
- First run: base and RL endpoints; add SFT/DPO only after checking the pilot.
- Conditions: common completion prompt versus tokenizer-native format (completion
  for base, chat for post-trained models). These are fixed baselines, **not** claims
  of equally optimal elicitation. Any additional prompt search belongs on development
  data with equal search budgets and a newly locked final protocol.
- Pilot: 50 questions, 8 independent samples, 512/2048 new-token caps. Inspect length
  stops; these caps are deliberately below Olmo's published 32768-token evaluation
  setting. Extend budgets in a **new plan** before interpreting truncation as inability.
- Primary outcome: mean per-question sampled correctness (pass@1), including
  malformed or truncated unsuccessful outputs in the denominator.
- Secondary: unbiased pass@k estimates, majority-vote accuracy (ties abstain),
  strict formatting, parsing failures, unfinished thinking, solved-problem overlap, actual prompt/generated
  token costs, and paired question-group bootstrap intervals.

Final-answer grading requires an explicit `Final answer:`, a terminal boxed answer,
or a `####` answer; there is no last-number fallback. Numeric equivalence is exact
rational arithmetic, not a model judge. An unclosed thinking region—including an
opening supplied by the native chat template—cannot receive final-answer credit.
These conventions may reject otherwise understandable answers; inspect raw outputs
and parse-failure rates before interpreting differences as mathematical ability.

### E2 — executing repair versus initiating a check

- Fresh, generated two-step integer arithmetic: `(boxes * items) + loose_items`.
- Every source has correct/error draft variants with exactly one declared numerical
  change and the same question/gold answer. No final answer is supplied.
- Cross both draft variants with no cue, a neutral continuation cue, and a generic
  checking cue. Every cue is used on both correct and incorrect drafts; the cue
  does **not** reveal which trials contain errors.
- Compare error-draft recovery, correct-draft retention, and check-minus-neutral
  effects within checkpoint under the same remaining generation budget.
- Automatically grade final answers and explicitly restated multiplication equations.
  A missing equation is **not observed**, not evidence that no checking occurred.
  Matching the propagated wrong answer is descriptive, not proof of its causal path.
- This is an assisted draft-continuation diagnostic. It does not measure autonomous
  detection of naturally occurring mistakes, and does not establish internal mechanisms.

### Splits, uncertainty, and interpretation

Development/pilot subsets use disjoint source pools from GSM8K train; final evaluation
uses GSM8K test. These are evaluation roles, not model-training splits. Published
benchmark contamination cannot be excluded. Synthetic splits occupy disjoint numerical
tuples; they are fresh instances, not a claim of out-of-distribution generalization.
Related draft variants remain together for sharding and resampling.

Report incomplete schedules explicitly. Do not compare whichever samples happened to
finish, or tune prompts on the final test. Saved outputs include incorrect answers,
format failures, length stops, failed attempts, and cancellations. Pass@k uses an oracle
checker and is not deployable answer selection; finite coverage is not a capacity bound.
Intervals resample problem groups, not individual tokens or repeated samples. They do
not incorporate uncertainty across training seeds or all possible benchmarks.

A smaller pass@1 gap after prompting supports an elicitation contribution. A pass@1
gain with similar sampled coverage supports increased reliability on the observed set.
Improved cued repair with a persistent uncued gap is consistent with an initiation
deficit, but format/comprehension differences remain alternatives. Failed recovery does
not prove novel reasoning mechanisms. Do not add probes or steering until a robust
behavioral gap warrants them.

## Sources

- [Olmo 3 Think model card and released lineage](https://huggingface.co/allenai/Olmo-3-7B-Think)
- [Base Models Know How to Reason, Thinking Models Learn When, v4](https://arxiv.org/html/2510.07364v4)
- [Neel Nanda's worked example](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability#Worked_Examples)

The paper's hybrid receives thinking-model assistance; it is not an unaided base.
The experiment here tests related behavioral hypotheses without reproducing that hybrid.

## Remote quick start

Run from the repository root in an activated Python 3.11+ environment on the GPU
machine. Install the **CUDA-compatible PyTorch wheel appropriate for that machine's
driver/GPU first**, then the project extras. Do not copy a macOS environment to Linux.
No CUDA wheel/driver or GPU compatibility is established by the local CPU tests.

```sh
python -m pip install -e '.[posttrain,dev]'
python -m pytest -q
python -m interp posttrain --help
```

The base package suffices for inspection/reports; `posttrain` adds native Olmo 3
Transformers support and GSM8K acquisition. The local tested constraints are not a
Linux CUDA lock. Save the remote resolved environment alongside the code commit
before confirmatory runs. No API keys are needed for these public model/dataset
repositories. HF login/environment secrets, if used, must never enter YAML or git.

### 1. Prepare fixed pilot datasets

```sh
python -m interp posttrain data gsm8k --split pilot --count 50 --seed 2026 --allow-downloads --output data/posttrain/gsm8k-pilot.jsonl
python -m interp posttrain data repair --split pilot --count 50 --seed 2026 --output data/posttrain/repair-pilot.jsonl
python -m interp posttrain preview configs/posttrain/math-pilot.yaml
python -m interp posttrain preview configs/posttrain/repair-pilot.yaml
```

JSONL is immutable: recreating identical data is safe, replacing different bytes is
an error. `count` means source questions; repair emits 100 rows for 50 matched pairs.
Dataset reference solutions are not placed in prompts. Acquisition records the full
dataset revision; pass that same `--revision` when preparing other GSM8K roles.

**Worst-case generated-token bounds**, before prompt prefills and runtime overhead:

| 50-source pilot | One checkpoint | Base + RL | All four checkpoints |
| --- | ---: | ---: | ---: |
| E1 math samples | 1,600 | 3,200 | 6,400 |
| E1 generated tokens | 2,048,000 | 4,096,000 | 8,192,000 |
| E2 repair samples | 4,800 | 9,600 | 19,200 |
| E2 generated tokens | 3,072,000 | 6,144,000 | 12,288,000 |

Actual EOS may reduce work substantially. Start with preflight and `--max-new-trials 4`,
measure throughput, then resume within the session allowance. The library generates
one sample at a time for stable per-trial RNG and modest VRAM, not maximum throughput.

### 2. Freeze plans once, then distribute the same bytes/code

```sh
python -m interp posttrain lock configs/posttrain/math-pilot.yaml --output data/posttrain/math-plan.json --allow-downloads
python -m interp posttrain lock configs/posttrain/repair-pilot.yaml --output data/posttrain/repair-plan.json --allow-downloads
```

`lock` resolves model/tokenizer branches to full commit SHAs and embeds all examples,
science, and a recursive package-source digest. **It does not download weights.**
Every worker should use the same plan and code, not independently resolve `main`.
The plan contains gold labels for local grading, but rendering never uses them.
Changing questions, prompts, precision, code, samples, or budgets requires a new plan
and new result directories. Comments in package source count as code changes.

Data and runs are git-ignored. Generate plans on the GPU host after pushing the code,
or copy locked plans explicitly to the remote machine/shared durable storage. A plan
does not depend on its original YAML/JSONL paths during run or resume.

### 3. Check the actual GPU, then run the two endpoints

```sh
python -m interp posttrain preflight data/posttrain/math-plan.json --checkpoint base --device cuda:0 --allow-downloads
python -m interp posttrain preflight data/posttrain/math-plan.json --checkpoint rl --device cuda:0 --allow-downloads
python -m interp posttrain run data/posttrain/math-plan.json --checkpoint base --device cuda:0 --output-dir runs/posttrain/math-base --allow-downloads --max-new-trials 4
python -m interp posttrain resume runs/posttrain/math-base --device cuda:0 --max-seconds 36000 --shutdown-margin-seconds 120
python -m interp posttrain run data/posttrain/math-plan.json --checkpoint rl --device cuda:0 --output-dir runs/posttrain/math-rl --allow-downloads --max-seconds 36000 --shutdown-margin-seconds 120
```

Use `sft` and `dpo` identically once the endpoint pilot has been inspected. For E2:

```sh
python -m interp posttrain preflight data/posttrain/repair-plan.json --checkpoint base --device cuda:0 --allow-downloads
python -m interp posttrain preflight data/posttrain/repair-plan.json --checkpoint rl --device cuda:0 --allow-downloads
python -m interp posttrain run data/posttrain/repair-plan.json --checkpoint base --device cuda:0 --output-dir runs/posttrain/repair-base --allow-downloads --max-seconds 36000 --shutdown-margin-seconds 120
python -m interp posttrain run data/posttrain/repair-plan.json --checkpoint rl --device cuda:0 --output-dir runs/posttrain/repair-rl --allow-downloads --max-seconds 36000 --shutdown-margin-seconds 120
```

BF16 is explicit in the pilot configs. For older GPUs without supported BF16, edit
the config to FP16 **before locking**, and run preflight; no automatic dtype fallback
occurs. Different actual GPU/library profiles are refused on resume and in pooled
reports. GPU ordinal alone is not scientific identity. A model is loaded on one
selected GPU with no model parallelism or quantization; 7B BF16 weights alone need
roughly 14 GB, plus KV cache, runtime/workspace, and CPU loading memory.

### If preflight fails after downloading/loading

Completed shard downloads do not imply that model validation or numerical checks
passed. The CLI identifies controlled failures such as vocabulary/context mismatches
and labels dependency stages without printing raw HTTP messages or prompt contents.
A numerical rejection reports the actual GPU/library profile, cache finiteness,
compared/mismatched logit counts, maximum absolute difference, maximum tolerance
ratio, argmax agreement, and the **separate** seeded-token replay result.

`max_tolerance_ratio` compares each absolute difference with
`atol + rtol * abs(cached_logit)`; a value above one exceeds the existing tolerance.
NaN/Inf always fail, even when both outputs contain the same infinity. Error maxima
are null for nonfinite outputs, and generation replay is skipped in that case.
Matching argmax tokens do **not** waive the logit check. Do not increase tolerances,
disable the guard, resize embeddings, or silently change dtype/attention just to pass.

After a diagnostics-only code update, ordinary preflight/run/resume still reject an
older source-pinned plan. To inspect the **same archived model/data revisions** using
the cached weights without overwriting that plan or resolving a newer `main`:

```sh
git pull --ff-only
python -m interp posttrain preflight data/posttrain/math-plan.json --checkpoint base --device cuda:0 --diagnostic
```

Substitute the plan/checkpoint/device from the failed invocation. `--diagnostic`
writes no trials, reports the old/current source hashes, and does not approve
execution of a changed-source plan. This command remains cached-only without
`--allow-downloads`. Preserve existing plans/results; actual research after a code
fix requires a newly locked plan and new result directories. Inspect the diagnostic
failure before deciding which numerical or configuration change is warranted.

### Multiple GPUs or independent sessions

Prefer different checkpoints on different GPUs, one command/process per GPU. To split
one checkpoint across two GPUs, keep the same `--num-shards 2` for both workers:

```sh
python -m interp posttrain run data/posttrain/math-plan.json --checkpoint base --device cuda:0 --num-shards 2 --shard-index 0 --output-dir runs/posttrain/math-base-0 --allow-downloads
python -m interp posttrain run data/posttrain/math-plan.json --checkpoint base --device cuda:1 --num-shards 2 --shard-index 1 --output-dir runs/posttrain/math-base-1 --allow-downloads
```

Launch those commands in separate sessions/scheduler jobs. Group hashing assigns both
repair variants to the same worker. Trial IDs/seeds do not depend on partitioning,
but a job's partition count is fixed: do not resume it with a different partition
layout. Analysis rejects duplicate jobs/trials and mixed partition counts. Pass all
shards for a represented checkpoint when requesting a complete report.

### 4. Inspect and report without a GPU

```sh
python -m interp posttrain inspect runs/posttrain/math-base
python -m interp posttrain analyze runs/posttrain/math-base runs/posttrain/math-rl --output runs/posttrain/math-report
python -m interp posttrain analyze runs/posttrain/repair-base runs/posttrain/repair-rl --output runs/posttrain/repair-report
```

Reports include JSON, per-cell/per-question/paired-comparison CSVs, and a readable
Markdown table. Point estimates are proportions, not percentages. Available k values
stop at the planned independent sample count; pass@k uses the finite-sample unbiased
combinatorial estimator. Voting is unique-most-frequent parsed answer, with ties or
no votes abstaining; it is not a requirement of more than half of all samples.

Absent *whole checkpoints* are allowed for the initial endpoint comparison. Missing
trials in any represented checkpoint fail by default. `--allow-incomplete` is an
explicit exploratory option: it excludes entire incomplete groups and reports what
was excluded. Do not treat a selectively completed subset as final evidence.
Reports never overwrite an existing directory. Raw prompts, token IDs, text, exact
effective decoding settings, grades, times, and failed/cancelled attempts remain in
manifest-referenced run shards for manual audit.

### Durability, stopping, and limitations

- Default flush is **every sample**. `--flush-records N` trades disk overhead for up
  to N unpublished attempts lost on a hard kill. Only the manifest is completion.
- Ctrl-C and CLI SIGTERM retain finalized attempts; early generation is cancelled,
  not a successful short answer. Resume retries it from the same isolated seed.
- Time limits are cooperative. A synchronous model load/forward can overrun them.
- Use a **persistent local disk/volume**. Local fsync is not remote backup; ephemeral
  GPU-session files can still disappear. Copy the stopped run directory plus plan
  and code to durable storage. Network-filesystem atomicity has not been verified.
- The old `interp sync/restore` commands accept only the small paired-continuation
  format, **not these research jobs**. No automatic remote upload is implemented.
- Short preflight checks test cache logits and seeded replay, not long-context parity.
- These are fixed prompt baselines. Equal-budget development prompt optimization,
  answer-judge audits, OOD datasets, training, probes, steering, and internal-mechanism
  attribution are **not** silently added or claimed.

### Before final evaluation

Review pilot transcripts for parser failures, truncations, task difficulty, and
ceiling/floor effects. Use `--split development` data for any prompt or budget choice;
then freeze a separate final config pointing at data created with `--split test`.
Keep all prompts searched and their search cost in the research log. Increase sample
counts/budgets before final locking if needed; never extend a finished plan in place.
Run the locked final protocol once without choosing questions by checkpoint outcome.

No pretrained checkpoint, actual GPU, remote account, or research result is validated
merely by this implementation. Local engineering checks are recorded in the root README.