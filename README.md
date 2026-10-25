# Personal interpretability research workbench

The workbench contains the original **paired behavioral experiment → inspect → resume**
loop and a first personal **science-of-post-training** study. It is not a public
framework or a GPU-validated toolkit. The broader roadmap remains in
[RESEARCH_PLAN.md](RESEARCH_PLAN.md).

## First research project: base → SFT → DPO → RL

See [experiments/posttraining/README.md](experiments/posttraining/README.md) for the
experimental protocol, complete remote-GPU commands, cost bounds, and limitations.

- **Math gap:** Olmo 3 7B lineage, shared/native prompts, independent samples and token
  budgets, exact numerical answers, pass@k, voting, paired intervals, and overlap.
- **Error repair:** generated matched correct/error arithmetic drafts crossed with
  uncued/neutral/check prompts; final recovery, retention, and observed equation claims.
- Portable locked plans pin dataset contents, full model/tokenizer SHAs, and recursive
  source identity. One checkpoint/group partition per GPU process; resumable atomic
  local checkpoints with raw outputs, seeds, grades, and actual numerical profiles.
- Model-free JSON/CSV/Markdown reports reject incomplete comparisons by default.
- Explicit preflight checks short cache parity and seeded replay on the selected device.

Configs: [configs/posttrain/math-pilot.yaml](configs/posttrain/math-pilot.yaml),
[configs/posttrain/repair-pilot.yaml](configs/posttrain/repair-pilot.yaml), and the
engineering-only [configs/posttrain/smoke.yaml](configs/posttrain/smoke.yaml).
Use `python -m interp posttrain --help`. The `posttrain` extra installs the optional
study dependencies. No weights or public benchmark data are bundled.

The studies use their own rollout contracts in
[src/interp/posttrain/schemas.py](src/interp/posttrain/schemas.py), while reusing exact HF generation,
RNG scopes, source hashes, safe YAML validation, atomic writes, and manifest types.
They do not force mathematical rollouts into the old two-candidate metric schema.
The old bounded Hub sync format is **not** supported for these larger research jobs.

The sections below describe the original paired-continuation workflow and its
previous validation; the research-study validation is recorded separately below.

## What is implemented

- Strict settings/record validation; safe YAML with duplicate-key rejection.
- Exact text, candidate definitions, dataset lineage, complete pairs, and grouped splits.
- An **entirely offline, random, tiny GPT-2 fixture**, using standard byte-level tokenizers.
- A separate pinned, safe-tensors HF GPT-2 loading path; downloads explicitly opt in.
- Multi-token teacher-forced continuation likelihoods, correct causal shift, explicit BOS,
  and no silent truncation. A candidate that changes the original prompt's token prefix
  is rejected rather than scored against different conditioning tokens.
- Optional bounded greedy/sampled generation, per-pair/replicate seeds, and saved effective
  decoding settings. Generation length stops and interruptions remain distinguishable.
- Stable logical trials, unique execution attempts, immutable checksummed JSONL shards,
  single writers, atomic manifests, and conservative resume compatibility checks.
- Read-only descriptive paired effects, incomplete-pair tables, and optional CPU plots.
- Explicit small private-Hub checkpoint sync/restore, with a pinned acknowledged revision.
- A thin CLI and marimo notebook using the **same** run entry point.

The fixture's predictions are meaningless scientifically. It tests plumbing, not
knowledge, research hypotheses, alignment, or a proposed mechanism. Probe decodability,
causal interventions, activation capture, and pretrained GPU parity are **not implemented yet**.

## Installation groups

[pyproject.toml](pyproject.toml) separates lightweight validation/storage from optional tools:

| Group | Purpose |
| --- | --- |
| Base | Pydantic, safe YAML, OS-backed file locking; inspect artifacts without Torch |
| `models` | PyTorch, Transformers, tokenizers, safetensors |
| `posttrain` | Native Olmo 3 inference, Hub revision resolution, optional GSM8K acquisition |
| `hub` | Explicit authenticated HF persistence |
| `notebooks` | marimo and Plotly displays |
| `dev` | Offline pytest checks and Ruff |

Dependency installation follows implementation. Version ranges are compatibility bounds,
not a claim that every version/device combination was tested. Any generated lock is a
dependency resolution, **not** evidence that CUDA or Blackwell execution works.

[requirements-tested.txt](requirements-tested.txt) records the direct versions actually
exercised locally. It is a constraints file, **not a complete transitive/build lockfile**.
Use the selected environment's interpreter when installing or invoking the package:

```text
python -m pip install -c requirements-tested.txt -e ".[models,hub,notebooks,dev]"
```

Install only the groups needed for the task; model-free inspection needs the base package.
The short `interp` command requires that environment to be activated; `python -m interp`
uses the explicitly selected interpreter instead of a possibly unrelated shell entry point.

## First workflow

The checked-in [configs/smoke.yaml](configs/smoke.yaml) points to small synthetic paired
observations in [tests/fixtures/smoke_examples.jsonl](tests/fixtures/smoke_examples.jsonl).
All YAML-relative paths resolve against the configuration directory, not the shell cwd.

After installing the editable package with the required groups, the CLI offers:

```text
interp validate configs/smoke.yaml
interp run configs/smoke.yaml --max-new-trials 1
interp inspect runs/smoke
interp run configs/smoke.yaml
interp resume runs/smoke
```

`validate` is model-free: it does **not** claim tokenization or architecture capability
was checked. `run` explicitly loads the declared model, saves attempts, and skips existing
verified successes. `resume` can use the archived examples even when the original dataset
file is gone. A complete resumed run executes no new trials, although the shared entry point
currently loads the model to verify its numerical/config/tokenizer profile.

Use [notebooks/smoke.py](notebooks/smoke.py) for the corresponding marimo UI. Submit settings,
then click **Run / resume**. Inspection is a separate read-only form. The click guard consumes
each request before executing; a failed request needs a fresh explicit click. Plot recomputation
cannot launch experiments. [.marimo.toml](.marimo.toml) recommends lazy execution; the guard
does not depend on run-button auto-reset behavior.

**Lazy-mode detail:** initialize the notebook cells first. After submitting a form or clicking
an action widget, marimo marks dependent cells stale rather than immediately evaluating them.
Evaluate the relevant stale cell, or use **Run stale cells**, to apply that submitted request.
Evaluating cells without an actual submitted Run click cannot bypass the model-work guard.
Recreating the imports/settings also recreates an unclicked button, never a fresh guard beside
an old positive click count.

## What gets saved

Each explicitly selected run directory contains an immutable resolved snapshot, sealed
content-addressed JSONL shards, and an atomic manifest. Temporary/sealed orphan files are
never inferred to be completed observations. Only manifest-referenced checksum-verified
successful records satisfy logical trials. Failed and cancelled attempts remain inspectable.

The snapshot preserves exact examples, resolved settings, dataset/source hashes, actual
tokenizer/config fingerprints, relevant library versions, and numerical/hardware settings.
Records preserve exact input tokens/masks, token likelihoods, raw generated tokens/text,
finish reason, isolated seeds, and unique attempt IDs. Softmax likelihood calculations are
explicitly float32; forward parameter precision is recorded separately.

Experimental identity includes scientific settings, exact validated data, and recursive package Python
source. It excludes machine-local paths and operational budgets. A source comment edit is
conservatively treated as a new identity. Changed model/tokenizer/library/numerical profiles
cannot silently resume. Historical artifacts remain inspectable without the old dataset/model.

This is a provenance foundation, not a complete source archive: **commit/archive code before
confirmatory research**. The source fingerprint covers the shared package and recipe
subpackages, including the implemented post-training studies. Changes to this package
intentionally invalidate resume of older runs; historical runs remain inspectable.

Use `run_experiment()` and `resume_experiment()` for scientific execution. `run_trials()` is
the low-level injection seam for the packaged recipe and artificial persistence tests; it
cannot authenticate an arbitrary callback's source/closure. Custom project recipes are not
supported under the packaged recipe's identity yet.

## Budgets and interruptions

`max_new_trials` limits fresh attempts per invocation, not total experimental identity.
`max_seconds` includes preparation/model loading, and `shutdown_margin_seconds` reserves
headroom before new work is stopped. Cancellation is cooperative: checks happen between
trials, candidate forwards, and generation steps. A synchronous forward/download can exceed
the budget; there is no hard process watchdog or automatic session-deadline discovery.

Ordinary errors/interrupts finalize validated buffered attempts. A hard process kill can lose
unpublished buffered work. Interrupted logical trials rerun from their own seeds; live
generation/KV state is not serialized. Local filesystem durability assumes ordinary local
macOS/Linux rename/link/fsync semantics, not an unverified network filesystem.

## Explicit remote checkpoints

Local completion is **not** remote durability. A molab storage connection does not prove
Python authentication, write permission, or automatic synchronization.

`sync` requires an explicitly selected **existing private HF dataset repository**, authenticates
through supported HF secrets/existing login, and publishes one synchronous bounded commit.
No repository is created and no token field/CLI argument is provided. Raw prompts/results are
uploaded only to the selected private destination; do not put credentials in those prompts.

The first checkpoint strategy supports at most **128 files / 16 MiB**. Larger snapshots fail
before upload rather than silently changing into a multi-commit transaction. A parent-commit
guard rejects concurrent remote changes. An acknowledged receipt records the exact revision
and completed logical boundary; subsequent unsynced local progress does not advance it.

An identical manifest avoids another commit only after **all referenced remote evidence** is
verified at that revision. A network/offline cache miss is not treated as proof of a missing
remote file. The client's authorization check verifies read access; the actual commit enforces
write permission. Neither molab's storage panel nor a read check proves permission to upload.

`restore` requires a new nonexistent local directory. All files are fetched at one immutable
revision, checksummed and validated in a staging directory, then published atomically. It
does not overwrite existing runs, restore orphan uploads, or run a model. A later explicit
`resume` checks source/model/numerical compatibility before new trials.

**No real HF authentication/upload/restore or GPU run is verified by writing this code.**
Validate a small private-repo round trip and the actual GPU numerical profile before large runs.

## Post-training validation — 2026-10-25

Verified in the existing **macOS arm64, Python 3.11.14, CPU float32** conda environment;
no dependencies, pretrained weights, or public benchmark data were installed/downloaded
for this implementation phase.

| Check | Observed result |
| --- | --- |
| Full offline pytest suite, including the original workflow | **401 passed** in 7.81 seconds |
| Ruff and strict marimo structure checks | Passed |
| Actual study CLI preview → lock → preflight | Eight planned fixture trials; all prompts fit |
| Actual study CLI partial run → resume → report | Four unique successes per checkpoint, eight total; no failed/cancelled attempts |
| Completed study resume | Zero new attempts; model-free fast path also tested with a forbidden loader |
| Short real tiny-GPT-2 cache parity | Maximum absolute logit difference 1.19e-7; identical seeded generation tokens |
| Native tiny random Olmo3 CPU cache parity | Passed; standard local Transformers implementation, no downloaded weights |
| Reports | Complete JSON, cells/comparisons/problems CSVs, and Markdown report produced |
| Original paired CLI after shared changes | Partial run resumed to four unique completed trials |
| Offline wheel build | Passed; all eleven study-package modules included |

New regressions cover immutable code/model/data locks, group-preserving partitions,
retry seeds, crash/orphan recovery, corruption, concurrent-writer rejection before
model allocation, missing-data denominators, exact pass@k, paired bootstrap/voting,
and numerical answer extraction. Unclosed thinking and contradictory boxed suffixes
cannot earn final-answer credit; checkpoint-specific penalties cannot override the
declared decoding settings. Acquisition tests use fake Hub/dataset boundaries.

The checked-in fixture uses the **same random model** for its two checkpoint labels.
Its output is not a base-versus-post-training experiment or an accuracy result.
Real Olmo tokenizer templates, downloaded checkpoints, GPU kernels/long contexts,
throughput, and remote persistence still require validation on the target machine.
The runbook includes the explicit preflight and resumable pilot commands.

### Remote-failure diagnostics update — 2026-10-25

Post-load failures now expose safe operation labels and numerical measurements rather
than a generic `ValueError`. Cache checks also reject nonfinite logits explicitly;
the existing tolerances, dtype, attention choice, and generation settings are unchanged.
The read-only `preflight --diagnostic` mode can inspect an older source-pinned plan
after an update, reporting both source hashes without weakening run/resume locks.

The expanded offline suite passed **424 tests** in 8.48 seconds, including Olmo3
eager/SDPA cache updates across the sliding window, NaN/Inf/mismatch accounting,
safe CLI errors, and diagnostic-mode provenance. The standalone diagnostic command
also passed against an unchanged archived fixture plan after the source update.
This does **not** reproduce or
resolve a particular remote GPU failure without its new diagnostic output.

## Earlier paired-workflow validation

Verified on **macOS arm64, Python 3.11.14, CPU float32**, using the editor-selected conda
environment before the post-training study was added. The existing workspace-local
virtual environment was left unchanged. Browser checks below are historical, not
newly repeated by the current CLI-only validation.

| Check | Observed result |
| --- | --- |
| Offline pytest suite | **58 passed**; final measured run: 7.41 seconds |
| Ruff lint | Passed |
| Strict marimo structure check | Passed |
| Actual CLI partial run → inspect → finish → resume | Four unique successes; final resume executed zero new trials |
| Browser notebook startup/freeze | No model-run artifacts created before the explicit Run action |
| Browser Run / reactive recomputation | Four trials once; unchanged click reused the saved summary |
| Browser inspection/plot/import-cell reset | Read-only analysis worked; published manifest and four attempts stayed unchanged |
| Actual CLI / browser result parity | Identical logical IDs, seeds, raw tokens, likelihoods, and effective decoding settings |

The tests exercise exact multi-token causal likelihoods, Unicode/leading whitespace, unequal
lengths, left/right padded **real GPT-2 forwards with correct position IDs**, BPE-boundary
rejection, sampled replay, and RNG/module-mode restoration on failure. They also inject
payload-fsync/manifest failures, interruptions, duplicate successes, local corruption, writer
conflicts, stale remote branches, lost commit acknowledgements, damaged remote evidence, and
metadata-inclusive restore-size failures. All remote tests use in-memory/fake client boundaries;
outbound socket connections are blocked in pytest.

Tests are in [tests/unit/test_contracts.py](tests/unit/test_contracts.py),
[tests/unit/test_models.py](tests/unit/test_models.py),
[tests/unit/test_hub_adapter.py](tests/unit/test_hub_adapter.py),
[tests/integration/test_checkpoints.py](tests/integration/test_checkpoints.py), and
[tests/integration/test_workflow.py](tests/integration/test_workflow.py).

```text
python -m pytest -q
python -m ruff check .
python -m marimo check --strict notebooks/smoke.py
```

**Remaining limits:** no pretrained checkpoint was downloaded/executed, no actual Hub account
was contacted, and no GPU or molab filesystem persistence was tested. These checks validate
the offline engineering loop, not research conclusions or cross-hardware numerical parity.

The selected conda environment is shared, not a newly isolated clean environment. Its global
dependency check reports unrelated conflicts: Kaggle lacks `python-slugify`, and `pyvers`
requires `packaging<26` while 26.3 is installed. Those unrelated dependencies were not changed
to make this workbench's tests pass. Use a dedicated compatible environment before confirmatory
research; the direct-version snapshot above does not resolve the entire shared environment.

## Module guide

| File | Responsibility |
| --- | --- |
| [src/interp/schemas.py](src/interp/schemas.py) | Validated scientific/configuration/result contracts |
| [src/interp/data.py](src/interp/data.py) | Exact data, pairing, split lineage, canonical hashes |
| [src/interp/config.py](src/interp/config.py) | Safe YAML, path resolution, model-free preparation |
| [src/interp/runtime.py](src/interp/runtime.py) | Provenance, stable IDs/seeds, scoped RNG/inference |
| [src/interp/models.py](src/interp/models.py) | Offline fixture, pinned HF GPT-2, scoring/generation |
| [src/interp/artifacts.py](src/interp/artifacts.py) | Atomic local publication and full checkpoint verification |
| [src/interp/runner.py](src/interp/runner.py) | Ordinary trial scheduling, attempts, budgets, run/resume |
| [src/interp/hub.py](src/interp/hub.py) | Explicit bounded HF/fake-remote checkpoint boundary |
| [src/interp/metrics.py](src/interp/metrics.py) | Read-only descriptive effects and attempt tables |
| [src/interp/plots.py](src/interp/plots.py) | Optional saved-result figures, no model work |
| [src/interp/cli.py](src/interp/cli.py) | Thin shared-function command dispatcher |
| [src/interp/posttrain/plan.py](src/interp/posttrain/plan.py) | Locked numerical-study inputs, source identity, schedules, partitions |
| [src/interp/posttrain/models.py](src/interp/posttrain/models.py) | Native dense-checkpoint loading and exact study prompts/rollouts |
| [src/interp/posttrain/runner.py](src/interp/posttrain/runner.py) | Explicit single-checkpoint study execution and model-free complete resume |
| [src/interp/posttrain/storage.py](src/interp/posttrain/storage.py) | Study-specific validated atomic rollout checkpoints |
| [src/interp/posttrain/metrics.py](src/interp/posttrain/metrics.py) | Complete-group metrics, paired uncertainty, and model-free reports |

Module introductions explain script use. Classes/functions have docstrings and executable
statements have adjacent explanatory comments, including scientific meaning and failure cases.
No plugin registry, custom tracer, distributed trainer, or SAE package has been added.