"""Exercise real posttrain publication/resume boundaries using artificial results.

Every plan, grade, effective decoding setting, and attempt is ordinarily validated;
the injected callback replaces only model execution. Raw text and short token IDs
are explicit bookkeeping fixtures, not measured model outputs. All writes and
fault injection are confined to fresh pytest temporary directories, with no GPU,
optional model libraries, downloads, or changes to checked-in artifacts.
"""

from __future__ import annotations

import builtins
import importlib.util
from collections import Counter
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest
from filelock import Timeout
from pydantic import ValidationError

from interp.artifacts import ArtifactError
from interp.config import source_digest
from interp.data import canonical_bytes, digest_bytes
from interp.posttrain import runner, storage
from interp.posttrain.plan import job_identity, schedule, study_identity
from interp.posttrain.schemas import (
    Attempt,
    Checkpoint,
    Inference,
    Job,
    Problem,
    Prompt,
    Rollout,
    StudyMetadata,
    StudyPlan,
    StudySpec,
)
from interp.posttrain.scoring import grade_answer
from interp.posttrain.storage import StudyStore, read_snapshot
from interp.runner import TrialExecutionError
from interp.runtime import CancellationRequested
from interp.schemas import CheckpointManifest, GenerationRecord, GenerationSpec, ShardRef

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def no_optional_imports(monkeypatch):
    """Reject model/Hub imports even when another test has already imported the package."""
    original = builtins.__import__
    blocked = {
        "torch", "numpy", "transformers", "tokenizers", "safetensors",
        "huggingface_hub", "datasets", "marimo", "plotly",
    }

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        """Keep inspection and the artificial execution seam strictly dependency-light."""
        package = (globals or {}).get("__package__", "")
        if name.split(".")[0] in blocked or (
            name == "models" and package in {"interp", "interp.posttrain"}
        ):
            raise AssertionError(f"Artifact tests must not import optional/model code: {name}")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def study_metadata(tmp_path):
    """Build valid in-memory science pinned to CURRENT package code, without writing files."""
    problems = tuple(Problem(
        id=f"question-{index}", source_id=f"source-{index}", group_id=f"group-{index}",
        split="pilot", task="math", question=f"What is {index} plus forty-two?",
        answer=str(index + 42), provenance="artificial-storage-fixture; no model executed",
    ) for index in range(2))
    spec = StudySpec(
        name="storage-fixture", hypothesis="Artificial durability tests, not research evidence.",
        dataset=tmp_path / "source-never-created.jsonl",
        checkpoints=(Checkpoint(
            name="base", stage="base", backend="fixture", model_id="offline-tiny-gpt2",
            revision="fixture-v1",
        ),),
        prompts=(Prompt(name="common"),), budgets=(4, 6), samples=2,
        generation=GenerationSpec(
            max_new_tokens=4, do_sample=True, temperature=0.7, top_p=0.9,
            top_k=7, use_cache=True,
        ),
        inference=Inference(dtype="float32", attention="eager", max_context_tokens=32),
    )
    data_sha = digest_bytes(canonical_bytes([row.model_dump(mode="json") for row in problems]))
    code_sha = source_digest()
    plan = StudyPlan(
        study_id=study_identity(spec, data_sha, code_sha), created_at=_NOW,
        source_sha256=code_sha, dataset_sha256=data_sha, spec=spec, problems=problems,
    )
    job = Job(checkpoint="base")
    return StudyMetadata(
        run_id=job_identity(plan, job), created_at=_NOW, plan=plan, job=job,
        model_profile={"kind": "artificial-records", "dtype": "float32", "attention": "eager"},
        runtime_profile={"kind": "artificial-runtime", "device": "cpu", "threads": 1},
    )


def _rollout(metadata, trial, *, cancelled=False):
    """Produce a short, exactly graded result with every locked effective setting preserved."""
    problem = next(row for row in metadata.plan.problems if row.id == trial.problem_id)
    settings = metadata.plan.spec.generation
    effective = {
        "max_new_tokens": trial.budget, "do_sample": settings.do_sample,
        "use_cache": settings.use_cache, "num_beams": 1, "num_return_sequences": 1,
        "repetition_penalty": 1.0, "eos_token_id": 2,
    }
    if settings.do_sample:
        effective.update(temperature=settings.temperature, top_p=settings.top_p, top_k=settings.top_k)
    # Gold is used to manufacture test output, never to render the artificial input prompt.
    text = f"Final answer: {problem.answer}"
    return Rollout(
        rendered_prompt=f"Artificial prompt, not model execution: {problem.question}",
        input_ids=(11, 12),
        generation=GenerationRecord(
            token_ids=(7, 2), text=text, finish_reason="cancelled" if cancelled else "eos",
            effective_config=effective,
        ),
        answer_text=text, grade=grade_answer(text, problem), elapsed_seconds=0.125,
    )


def _attempt(metadata, trial):
    """Create one new successful attempt without bypassing schema validation."""
    return Attempt(
        run_id=metadata.run_id, trial=trial, attempt_id=uuid4(),
        started_at=_NOW, finished_at=_NOW, status="completed", result=_rollout(metadata, trial),
    )


def _executor(metadata, calls):
    """Expose the ordinary runner callback while recording actual scheduled trial objects."""
    def execute(trial, should_stop):
        """Record this invocation and return artificial bounded output, not a model prediction."""
        assert not should_stop()
        calls.append(trial)
        return _rollout(metadata, trial)

    return execute


def _files(directory):
    """Capture filenames and exact bytes, including lock/orphan files, to detect reader writes."""
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def _assert_complete(directory, metadata):
    """Verify completion equals the whole declared schedule with exactly one success per trial."""
    saved = read_snapshot(directory)
    expected = {trial.trial_id: trial for trial in schedule(metadata.plan, metadata.job)}
    successes = [row for row in saved.records if row.status == "completed"]
    assert Counter(row.trial.trial_id for row in successes) == Counter(dict.fromkeys(expected, 1))
    assert set(saved.manifest.completed_trial_ids) == set(expected)
    assert len({row.attempt_id for row in saved.records}) == len(saved.records)
    assert all(row.trial == expected[row.trial.trial_id] for row in saved.records)
    return saved


def _rewrite_published_attempt(directory, attempt):
    """Rechecksum a test-owned single-record shard to isolate semantic validation from hashes.

    This intentionally forges local artifact contents. Leaving the original shard
    as an orphan also verifies that readers follow the manifest, not directory scans.
    """
    saved = read_snapshot(directory)
    assert len(saved.manifest.shards) == 1 and len(saved.records) == 1
    payload = canonical_bytes(attempt.model_dump(mode="json")) + b"\n"
    sha = digest_bytes(payload)
    name = f"records-000000-{sha}.jsonl"
    (directory / name).write_bytes(payload)
    reference = ShardRef(name=name, sha256=sha, size_bytes=len(payload), record_count=1)
    raw = saved.manifest.model_dump(mode="python")
    raw["shards"] = (reference,)
    manifest = CheckpointManifest.model_validate(raw)
    (directory / "manifest.json").write_bytes(canonical_bytes(manifest.model_dump(mode="json")))


def test_partial_run_then_resume_preserves_snapshots_and_has_no_duplicate_successes(
    tmp_path, study_metadata,
):
    """A one-attempt invocation is durable; resume fills only the remaining logical schedule."""
    directory = tmp_path / "run"
    calls = []
    execute = _executor(study_metadata, calls)
    first = runner.run_trials(study_metadata, directory, execute, max_new_trials=1)
    assert first["stop_reason"] == "trial_limit"
    assert first["new_attempts"] == first["completed_trials"] == 1
    partial = read_snapshot(directory)
    sealed = partial.manifest.shards[0]
    sealed_bytes = (directory / sealed.name).read_bytes()
    initial_record = partial.records[0]
    expected = tuple(schedule(study_metadata.plan, study_metadata.job))

    resumed = runner.run_trials(study_metadata, directory, execute)
    assert resumed["stop_reason"] == "complete"
    assert resumed["new_attempts"] == len(expected) - 1
    assert resumed["missing_trials"] == 0
    assert tuple(calls) == expected
    saved = _assert_complete(directory, study_metadata)
    assert saved.records[0] == initial_record
    assert (directory / sealed.name).read_bytes() == sealed_bytes
    # Previously returned snapshots are stable publication boundaries, not live store views.
    assert partial.records == (initial_record,)
    assert partial.manifest.completed_trial_ids == (initial_record.trial.trial_id,)
    with pytest.raises(FrozenInstanceError):
        partial.records = ()
    with pytest.raises(ValidationError, match="frozen"):
        initial_record.status = "failed"

    before = _files(directory)
    unexpected = Mock(side_effect=AssertionError("A complete job must not execute again."))
    repeated = runner.run_trials(study_metadata, directory, unexpected)
    assert repeated["new_attempts"] == 0
    assert runner.inspect_job(directory)["completed_trials"] == len(expected)
    unexpected.assert_not_called()
    assert _files(directory) == before


@pytest.mark.parametrize("interruption", ["failed", "cooperative", "partial", "keyboard"])
def test_failed_and_cancelled_trials_retry_same_seed_with_distinct_attempt_ids(
    tmp_path, study_metadata, interruption,
):
    """Failures and even correctly graded cancelled output never satisfy a logical sample."""
    directory = tmp_path / "run"
    attempted = []
    secret_message = "exception detail must not be persisted"

    def interrupt(trial, should_stop):
        """Inject one ordinary failure or cancellation at the model execution seam only."""
        assert not should_stop()
        attempted.append(trial)
        if interruption == "failed":
            raise RuntimeError(secret_message)
        if interruption == "cooperative":
            raise CancellationRequested(secret_message)
        if interruption == "keyboard":
            raise KeyboardInterrupt(secret_message)
        return _rollout(study_metadata, trial, cancelled=True)

    # Buffered records must also be flushed while unwinding exceptions or interruptions.
    if interruption in {"failed", "keyboard"}:
        expected_error = TrialExecutionError if interruption == "failed" else KeyboardInterrupt
        with pytest.raises(expected_error):
            runner.run_trials(study_metadata, directory, interrupt, flush_records=3)
    else:
        stopped = runner.run_trials(study_metadata, directory, interrupt, flush_records=3)
        assert stopped["stop_reason"] == "cancelled_or_time_budget"
        assert stopped["new_attempts"] == 1 and stopped["completed_trials"] == 0
    saved = read_snapshot(directory)
    assert len(saved.records) == 1 and not saved.manifest.completed_trial_ids
    incomplete = saved.records[0]
    assert incomplete.status == ("failed" if interruption == "failed" else "cancelled")
    assert incomplete.trial == attempted[0]
    if interruption == "partial":
        assert incomplete.result is not None and incomplete.result.grade.correct
        assert incomplete.result.generation.finish_reason == "cancelled"
    else:
        assert incomplete.result is None
        assert incomplete.error_type == {
            "failed": "builtins.RuntimeError", "cooperative": "CancellationRequested",
            "keyboard": "KeyboardInterrupt",
        }[interruption]
    assert all(secret_message.encode() not in content for content in _files(directory).values())
    inspection = runner.inspect_job(directory)
    assert inspection["failed_attempts"] == int(interruption == "failed")
    assert inspection["cancelled_attempts"] == int(interruption != "failed")

    retried = []
    runner.run_trials(study_metadata, directory, _executor(study_metadata, retried))
    completed = _assert_complete(directory, study_metadata)
    matching = [row for row in completed.records if row.trial.trial_id == incomplete.trial.trial_id]
    assert len(matching) == 2
    assert matching[0] == incomplete
    assert matching[1].status == "completed"
    assert matching[1].attempt_id != incomplete.attempt_id
    assert matching[1].trial == incomplete.trial == retried[0]
    assert len(completed.records) == len(retried) + 1


def test_completed_run_inspection_and_resume_never_load_models_or_write(tmp_path, study_metadata):
    """The public resume fast path uses archived inputs and remains strictly model-free."""
    directory = tmp_path / "run"
    runner.run_trials(study_metadata, directory, _executor(study_metadata, []))
    before = _files(directory)
    assert not study_metadata.plan.spec.dataset.exists()
    inspected = runner.inspect_job(directory)
    assert inspected["missing_trials"] == 0
    assert inspected["remote_durability"] == "not_managed"
    resumed = runner.resume_study(directory)
    assert resumed["new_attempts"] == 0 and resumed["stop_reason"] == "complete"

    # Re-execute the runner's module body under the import guard without invalidating shared types.
    probe_spec = importlib.util.spec_from_file_location(
        "interp.posttrain._offline_runner_probe", runner.__file__,
    )
    assert probe_spec is not None and probe_spec.loader is not None
    probe = importlib.util.module_from_spec(probe_spec)
    probe_spec.loader.exec_module(probe)
    assert probe.inspect_job(directory) == inspected
    assert _files(directory) == before


def test_historical_snapshot_inspectable_but_not_executable_after_code_change(
    tmp_path, study_metadata, monkeypatch,
):
    """Current-source enforcement belongs to execution, including completed-run resume."""
    directory = tmp_path / "run"
    runner.run_trials(study_metadata, directory, _executor(study_metadata, []))
    before = _files(directory)
    changed_sha = "0" * 64 if study_metadata.plan.source_sha256 != "0" * 64 else "1" * 64
    monkeypatch.setattr(runner, "source_digest", lambda: changed_sha)
    assert runner.inspect_job(directory)["missing_trials"] == 0
    assert read_snapshot(directory).metadata == study_metadata
    with pytest.raises(ArtifactError, match="source changed"):
        runner.resume_study(directory)
    assert _files(directory) == before


@pytest.mark.parametrize("artifact", ["metadata", "shard"])
def test_tampered_artifact_checksum_refuses_inspection(tmp_path, study_metadata, artifact):
    """Readers must refuse changed bytes instead of repairing or trusting saved completion."""
    directory = tmp_path / "run"
    runner.run_trials(study_metadata, directory, _executor(study_metadata, []), max_new_trials=1)
    saved = read_snapshot(directory)
    name = "metadata.json" if artifact == "metadata" else saved.manifest.shards[0].name
    path = directory / name
    # A whitespace-only edit remains parseable JSON/JSONL but invalidates the exact checksum.
    path.write_bytes(path.read_bytes() + b" ")
    before = _files(directory)
    for read in (read_snapshot, runner.inspect_job):
        with pytest.raises(ArtifactError, match="checksum"):
            read(directory)
    assert _files(directory) == before


@pytest.mark.parametrize("failure_point", ["initial_manifest", "before_seal", "after_seal"])
def test_publication_failure_never_promotes_orphans_and_retry_completes_once(
    tmp_path, study_metadata, monkeypatch, failure_point,
):
    """Fail before a manifest, before sealing, or between seal and manifest acknowledgement.

    The injected fault persists through writer finalization; a one-shot fault could
    otherwise be healed by its final flush and would not simulate an unpublished
    checkpoint. Recovery uses a fresh writer, never its optimistic in-memory set.
    """
    directory = tmp_path / "run"
    if failure_point != "initial_manifest":
        with StudyStore(directory).writer(study_metadata):
            pass
    real_write = storage.atomic_write
    failed_writes = []

    def fail_publication(path, content, *, immutable=False):
        """Keep the selected disk boundary unavailable until the writer has exited."""
        fail = (
            path.name == "manifest.json" if failure_point in {"initial_manifest", "after_seal"}
            else path.name.startswith("records-")
        )
        if fail:
            failed_writes.append(path.name)
            raise OSError("Injected publication failure")
        return real_write(path, content, immutable=immutable)

    attempted = []
    with monkeypatch.context() as fault:
        fault.setattr(storage, "atomic_write", fail_publication)
        with pytest.raises(OSError, match="Injected publication failure"):
            runner.run_trials(
                study_metadata, directory, _executor(study_metadata, attempted), max_new_trials=1,
            )
    assert failed_writes
    orphan_paths = set(directory.glob("records-*.jsonl"))
    assert len(orphan_paths) == int(failure_point == "after_seal")
    before = _files(directory)
    if failure_point == "initial_manifest":
        assert attempted == []
        assert (directory / "metadata.json").is_file()
        assert not (directory / "manifest.json").exists()
        with pytest.raises(ArtifactError, match="artifact"):
            runner.inspect_job(directory)
    else:
        assert len(attempted) == 1
        assert read_snapshot(directory).records == ()
        assert runner.inspect_job(directory)["completed_trials"] == 0
    assert _files(directory) == before

    retried = []
    resumed = runner.run_trials(
        study_metadata, directory, _executor(study_metadata, retried), max_new_trials=1,
    )
    assert resumed["new_attempts"] == resumed["completed_trials"] == 1
    if attempted:
        assert retried[0] == attempted[0]
    published = read_snapshot(directory)
    assert len(published.records) == 1
    assert orphan_paths.isdisjoint(directory / ref.name for ref in published.manifest.shards)
    for path in orphan_paths:
        assert path.read_bytes() == before[path.name]
    runner.run_trials(study_metadata, directory, _executor(study_metadata, retried))
    _assert_complete(directory, study_metadata)
    assert tuple(retried) == tuple(schedule(study_metadata.plan, study_metadata.job))


def test_single_writer_lock_conflict_fails_fast_without_executing(tmp_path, study_metadata):
    """An independent contender receives the zero-timeout OS lock failure, never a callback."""
    directory = tmp_path / "run"
    unexpected = Mock(side_effect=AssertionError("A conflicting writer must not execute."))
    with StudyStore(directory).writer(study_metadata):
        before = _files(directory)
        with pytest.raises(Timeout):
            runner.run_trials(study_metadata, directory, unexpected)
        assert _files(directory) == before
        assert read_snapshot(directory).records == ()
    unexpected.assert_not_called()
    # Once ownership is released, the next ordinary writer can publish successfully.
    runner.run_trials(study_metadata, directory, _executor(study_metadata, []))
    _assert_complete(directory, study_metadata)


@pytest.mark.parametrize("difference", ["runtime", "model", "science"])
def test_incompatible_resume_rejected_before_callback_or_artifact_changes(
    tmp_path, study_metadata, difference,
):
    """Changed actual profiles or genuinely different locked science cannot share a run."""
    directory = tmp_path / "run"
    runner.run_trials(study_metadata, directory, _executor(study_metadata, []), max_new_trials=1)
    raw = study_metadata.model_dump(mode="python")
    if difference == "runtime":
        raw["runtime_profile"]["threads"] = 2
    elif difference == "model":
        raw["model_profile"]["tokenizer_sha256"] = "2" * 64
    else:
        # Recompute both IDs so rejection is compatibility, not a malformed-plan shortcut.
        plan_raw = raw["plan"]
        plan_raw["spec"]["prompts"][0]["instruction"] = "A different scientific prompt."
        spec = StudySpec.model_validate(plan_raw["spec"])
        plan_raw["study_id"] = study_identity(spec, plan_raw["dataset_sha256"], plan_raw["source_sha256"])
        plan = StudyPlan.model_validate(plan_raw)
        raw["plan"] = plan
        raw["run_id"] = job_identity(plan, study_metadata.job)
    incompatible = StudyMetadata.model_validate(raw)
    unexpected = Mock(side_effect=AssertionError("Incompatible resumes must not execute."))
    before = _files(directory)
    with pytest.raises(ArtifactError, match="Resume refused"):
        runner.run_trials(incompatible, directory, unexpected)
    unexpected.assert_not_called()
    assert _files(directory) == before


@pytest.mark.parametrize("boundary", ["append", "read"])
def test_grade_corruption_rejected_even_with_valid_schema_and_checksums(
    tmp_path, study_metadata, boundary,
):
    """Saved grade fields must be reproduced from RAW text, not merely be well-typed."""
    directory = tmp_path / "run"
    trial = next(schedule(study_metadata.plan, study_metadata.job))
    valid = _attempt(study_metadata, trial)
    raw = valid.model_dump(mode="python")
    raw["result"]["grade"]["correct"] = False
    corrupt = Attempt.model_validate(raw)
    if boundary == "append":
        with StudyStore(directory).writer(study_metadata) as store:
            with pytest.raises(ArtifactError, match="grade"):
                store.append(corrupt)
            assert not store.buffer and not store.completed
            store.append(valid)
        assert read_snapshot(directory).records == (valid,)
    else:
        with StudyStore(directory).writer(study_metadata) as store:
            store.append(valid)
        _rewrite_published_attempt(directory, corrupt)
        with pytest.raises(ArtifactError, match="grade"):
            read_snapshot(directory)


@pytest.mark.parametrize("boundary", ["append", "read"])
@pytest.mark.parametrize(("setting", "wrong_value"), [
    ("max_new_tokens", 5), ("do_sample", False), ("use_cache", False),
    ("num_beams", 2), ("num_return_sequences", 2), ("repetition_penalty", 1.1),
    ("temperature", 0.8), ("top_p", 0.8), ("top_k", 50), ("top_k", None),
])
def test_effective_decoding_mismatch_rejected_at_writer_and_reader(
    tmp_path, study_metadata, boundary, setting, wrong_value,
):
    """Hidden defaults, missing sampling controls, and cache changes cannot be accepted."""
    directory = tmp_path / "run"
    trial = next(schedule(study_metadata.plan, study_metadata.job))
    valid = _attempt(study_metadata, trial)
    raw = valid.model_dump(mode="python")
    effective = raw["result"]["generation"]["effective_config"]
    if wrong_value is None:
        effective.pop(setting)
    else:
        effective[setting] = wrong_value
    corrupt = Attempt.model_validate(raw)
    if boundary == "append":
        with StudyStore(directory).writer(study_metadata) as store:
            with pytest.raises(ArtifactError, match="Effective decoding"):
                store.append(corrupt)
            assert not store.buffer and not store.completed
        assert read_snapshot(directory).records == ()
    else:
        with StudyStore(directory).writer(study_metadata) as store:
            store.append(valid)
        _rewrite_published_attempt(directory, corrupt)
        with pytest.raises(ArtifactError, match="Effective decoding"):
            read_snapshot(directory)


@pytest.mark.parametrize(("corruption", "message"), [
    ("seed", "identity/seed"), ("eos", "EOS"),
    ("length", "token count"), ("context", "context budget"),
])
def test_valid_record_shapes_still_require_schedule_and_generation_semantics(
    tmp_path, study_metadata, corruption, message,
):
    """Types alone cannot establish isolated seeds, actual EOS, exact lengths, or prompt room."""
    directory = tmp_path / "run"
    trial = next(schedule(study_metadata.plan, study_metadata.job))
    raw = _attempt(study_metadata, trial).model_dump(mode="python")
    if corruption == "seed":
        raw["trial"]["seed"] = (trial.seed + 1) % 2**32
    elif corruption == "eos":
        raw["result"]["generation"]["token_ids"] = (7, 3)
    elif corruption == "length":
        raw["result"]["generation"]["finish_reason"] = "length"
    else:
        raw["result"]["input_ids"] = (11,) * study_metadata.plan.spec.inference.max_context_tokens
    corrupt = Attempt.model_validate(raw)
    with StudyStore(directory).writer(study_metadata) as store:
        with pytest.raises(ArtifactError, match=message):
            store.append(corrupt)
    assert read_snapshot(directory).records == ()


def test_duplicate_attempt_and_second_logical_success_are_refused(tmp_path, study_metadata):
    """Different retry UUIDs cannot turn a published success into two logical observations."""
    directory = tmp_path / "run"
    trial = next(schedule(study_metadata.plan, study_metadata.job))
    original = _attempt(study_metadata, trial)
    with StudyStore(directory).writer(study_metadata) as store:
        store.append(original)
        with pytest.raises(ArtifactError, match="Duplicate"):
            store.append(original)
        with pytest.raises(ArtifactError, match="Duplicate"):
            store.append(_attempt(study_metadata, trial))
    assert read_snapshot(directory).records == (original,)


def test_construction_and_failed_reads_create_no_artifacts(tmp_path, study_metadata):
    """Remembering a destination or inspecting missing files must never initialize a run."""
    assert list(tmp_path.iterdir()) == []
    directory = tmp_path / "missing-parent" / "run"
    store = StudyStore(directory)
    assert store.directory == directory and store.snapshot is None
    assert not store.buffer and not store.completed
    trial = next(schedule(study_metadata.plan, study_metadata.job))
    with pytest.raises(ArtifactError, match="active writer"):
        store.append(_attempt(study_metadata, trial))
    for read in (read_snapshot, runner.inspect_job):
        with pytest.raises(ArtifactError):
            read(directory)
    assert list(tmp_path.iterdir()) == []

    directory.mkdir(parents=True)
    for read in (read_snapshot, runner.inspect_job):
        with pytest.raises(ArtifactError):
            read(directory)
    assert list(directory.iterdir()) == []
    # A zero-work public execution request may count a schedule but must not allocate artifacts.
    unused = tmp_path / "unused-run"
    summary = runner.run_study(study_metadata.plan, study_metadata.job, unused, max_new_trials=0)
    assert summary["new_attempts"] == 0 and summary["model_loaded"] is False
    assert not unused.exists()