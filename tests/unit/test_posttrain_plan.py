"""Offline contracts for portable study locks and partition-independent schedules.

These tests freeze checked-in smoke data or small generated repair pairs, never
model weights. Hub metadata is replaced by an in-memory module when explicitly
testing ref resolution. Source-change tests edit only a temporary package tree;
the installed research package and checked-in configurations remain untouched.
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError

from interp import config as config_module
from interp.artifacts import ArtifactError
from interp.posttrain import plan as planning
from interp.posttrain import runner
from interp.posttrain.data import load_problems, make_repair_problems, write_problems
from interp.posttrain.schemas import Job, StudyPlan, StudySpec

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "posttrain" / "smoke.yaml"


@pytest.fixture
def no_optional_imports(monkeypatch):
    """Fail on optional imports even if another test already cached the dependency."""
    original = builtins.__import__
    blocked = {
        "torch", "numpy", "transformers", "tokenizers", "safetensors",
        "huggingface_hub", "datasets", "marimo", "plotly",
    }

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        """Permit core dependencies but never let planning reach model execution."""
        package = (globals or {}).get("__package__", "")
        if name.split(".")[0] in blocked or (
            name == "models" and package in {"interp", "interp.posttrain"}
        ):
            raise AssertionError(f"Planning must not import optional/model code: {name}")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


def _raw_spec():
    """Start every mutation with a fresh, validated smoke design and absolute data path."""
    return planning.load_spec(SMOKE).model_dump(mode="json")


def _write_spec(directory, raw=None):
    """Write only a test-owned YAML file, preserving all supplied scientific settings."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "study.yaml"
    path.write_text(yaml.safe_dump(_raw_spec() if raw is None else raw), encoding="utf-8")
    return path


def _hf_checkpoint(**overrides):
    """Describe a fictional remote checkpoint without consulting any actual repository."""
    return {
        "name": "base", "stage": "base", "backend": "huggingface",
        "model_id": "artificial/model", "revision": "a" * 40,
        "tokenizer_id": "artificial/tokenizer", "tokenizer_revision": "b" * 40,
        **overrides,
    }


def test_smoke_preview_and_lock_are_model_free(tmp_path, no_optional_imports):
    """The real smoke design can be counted, locked, and read without optional packages."""
    before = tuple(tmp_path.iterdir())
    report = planning.preview(SMOKE)
    assert report["planned_trials"] == 8
    assert report["max_generated_tokens"] == 32
    assert report["groups"] == report["problems"] == 2
    assert report["weights_downloaded"] is report["tokenization_checked"] is False
    assert tuple(tmp_path.iterdir()) == before

    output = tmp_path / "locked" / "plan.json"
    locked = planning.lock_study(SMOKE, output)
    assert locked.problems == load_problems(locked.spec.dataset)
    assert locked.source_sha256 == config_module.source_digest()
    assert planning.read_plan(output, check_source=True) == locked
    assert list(output.parent.iterdir()) == [output]

    # Execute the planning module body afresh without evicting shared schema modules.
    probe_spec = importlib.util.spec_from_file_location(
        "interp.posttrain._offline_plan_probe", planning.__file__,
    )
    assert probe_spec is not None and probe_spec.loader is not None
    probe = importlib.util.module_from_spec(probe_spec)
    probe_spec.loader.exec_module(probe)
    assert probe.read_plan(output) == locked


def test_science_identity_ignores_config_data_output_paths_and_creation_time(tmp_path, monkeypatch):
    """Relocating identical inputs and locking later must preserve study and trial IDs."""
    locks = []
    for index, day in enumerate((1, 2)):
        directory = tmp_path / f"machine-{index}"
        directory.mkdir()
        data = directory / "questions.jsonl"
        data.write_bytes((ROOT / "tests" / "fixtures" / "posttrain_smoke.jsonl").read_bytes())
        raw = _raw_spec()
        raw["dataset"] = data.name
        config = _write_spec(directory, raw)
        monkeypatch.setattr(planning, "utc_now", lambda day=day: datetime(2026, 1, day, tzinfo=UTC))
        locks.append(planning.lock_study(config, directory / "archive" / "plan.json"))

    left, right = locks
    assert left.created_at != right.created_at
    assert left.spec.dataset != right.spec.dataset
    assert left.dataset_sha256 == right.dataset_sha256
    assert left.study_id == right.study_id
    job = Job(checkpoint="base")
    assert planning.job_identity(left, job) == planning.job_identity(right, job)
    assert tuple(planning.schedule(left, job)) == tuple(planning.schedule(right, job))


@pytest.mark.parametrize("destination", ["file", "directory", "dangling_symlink"])
def test_lock_never_overwrites_existing_destination(tmp_path, monkeypatch, destination):
    """Even empty directories and dangling links fail before config/data/network access."""
    output = tmp_path / "occupied"
    if destination == "file":
        output.write_bytes(b"retain these exact bytes")
    elif destination == "directory":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "absent-target")
    read = Mock(side_effect=AssertionError("Existing outputs must be rejected first."))
    monkeypatch.setattr(planning, "load_spec", read)
    with pytest.raises(ArtifactError, match="already exists"):
        planning.lock_study(tmp_path / "nonexistent.yaml", output, allow_downloads=True)
    read.assert_not_called()
    if destination == "file":
        assert output.read_bytes() == b"retain these exact bytes"
    elif destination == "directory":
        assert list(output.iterdir()) == []
    else:
        assert output.is_symlink() and not output.exists()


@pytest.mark.parametrize(("mutation", "message"), [
    ("unknown", "Extra inputs"),
    ("nested_unknown", "Extra inputs"),
    ("duplicate", "Duplicate YAML"),
    ("nested_duplicate", "Duplicate YAML"),
    ("merge", "merges"),
    ("mapping_cycle", "Cyclic YAML"),
    ("sequence_cycle", "Cyclic YAML"),
])
def test_invalid_yaml_fails_before_dataset_network_or_output_io(
    tmp_path, monkeypatch, mutation, message,
):
    """Reading the YAML itself is necessary; invalid science must stop all subsequent I/O."""
    raw = _raw_spec()
    if mutation == "unknown":
        raw["unknown_science"] = True
    elif mutation == "nested_unknown":
        raw["generation"]["unknown_sampling"] = 3
    text = yaml.safe_dump(raw)
    suffixes = {
        "duplicate": "name: silently-replaced\n",
        "nested_duplicate": "extra: {seed: 1, seed: 2}\n",
        "merge": "extra: {<<: {seed: 1}}\n",
        "mapping_cycle": "extra: &loop {again: *loop}\n",
        "sequence_cycle": "extra: &loop [*loop]\n",
    }
    config = tmp_path / "invalid.yaml"
    config.write_text(text + suffixes.get(mutation, ""), encoding="utf-8")
    unexpected = Mock(side_effect=AssertionError("Invalid YAML must stop before further I/O."))
    monkeypatch.setattr(planning, "load_problems", unexpected)
    monkeypatch.setattr(planning, "_resolve", unexpected)
    monkeypatch.setattr(planning, "atomic_write", unexpected)
    output = tmp_path / "not-created" / "plan.json"
    with pytest.raises(ValueError, match=message):
        planning.lock_study(config, output, allow_downloads=True)
    unexpected.assert_not_called()
    assert not output.parent.exists()


@pytest.mark.parametrize("floating", ["model", "tokenizer"])
def test_cached_only_lock_rejects_floating_refs_without_hub_import(
    tmp_path, no_optional_imports, floating,
):
    """Neither a model branch nor a tokenizer tag can be silently resolved from cache/network."""
    raw = _raw_spec()
    checkpoint = _hf_checkpoint()
    checkpoint["revision" if floating == "model" else "tokenizer_revision"] = "moving-ref"
    raw["checkpoints"] = [checkpoint]
    config = _write_spec(tmp_path, raw)
    output = tmp_path / "not-created" / "plan.json"
    with pytest.raises(ValueError, match="Floating refs"):
        planning.lock_study(config, output)
    assert not output.parent.exists()


@pytest.mark.parametrize("implicit_tokenizer", [False, True])
@pytest.mark.parametrize("allow_downloads", [False, True])
def test_full_shas_need_no_hub_import_even_with_network_permission(
    tmp_path, no_optional_imports, implicit_tokenizer, allow_downloads,
):
    """Full refs are already immutable; opt-in is permission, not a mandatory request."""
    checkpoint = _hf_checkpoint()
    if implicit_tokenizer:
        checkpoint.update(tokenizer_id=None, tokenizer_revision=None)
    raw = _raw_spec()
    raw["checkpoints"] = [checkpoint]
    locked = planning.lock_study(
        _write_spec(tmp_path, raw), tmp_path / "plan.json", allow_downloads=allow_downloads,
    )
    pinned = locked.spec.checkpoints[0]
    assert pinned.revision == "a" * 40
    assert pinned.tokenizer_id == ("artificial/model" if implicit_tokenizer else "artificial/tokenizer")
    assert pinned.tokenizer_revision == ("a" * 40 if implicit_tokenizer else "b" * 40)


@pytest.mark.parametrize("same_repository", [False, True])
def test_metadata_resolver_pins_model_and_tokenizer_refs_independently(
    tmp_path, monkeypatch, same_repository,
):
    """Cache keys include both repository and ref; distinct refs must not share a pin."""
    model_repo = "artificial/model"
    tokenizer_repo = model_repo if same_repository else "artificial/tokenizer"
    resolutions = {(model_repo, "model-branch"): "a" * 40,
                   (tokenizer_repo, "tokenizer-tag"): "b" * 40}

    def model_info(repo, *, revision):
        """Return only fictional commit metadata, with no real Hub client or weight API."""
        return SimpleNamespace(sha=resolutions[repo, revision])

    api = Mock()
    api.model_info.side_effect = model_info
    fake_hub = ModuleType("huggingface_hub")
    factory = Mock(return_value=api)
    monkeypatch.setattr(fake_hub, "HfApi", factory, raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    raw = _raw_spec()
    raw["checkpoints"] = [_hf_checkpoint(
        revision="model-branch", tokenizer_id=tokenizer_repo, tokenizer_revision="tokenizer-tag",
    )]
    locked = planning.lock_study(
        _write_spec(tmp_path, raw), tmp_path / "plan.json", allow_downloads=True,
    )
    pinned = locked.spec.checkpoints[0]
    assert (pinned.revision, pinned.tokenizer_revision) == ("a" * 40, "b" * 40)
    assert pinned.tokenizer_id == tokenizer_repo
    assert api.model_info.call_count == 2
    assert {(call.args[0], call.kwargs["revision"]) for call in api.model_info.call_args_list} == set(resolutions)


def test_read_plan_survives_deleted_config_and_source_dataset(tmp_path, no_optional_imports):
    """The archived validated questions, labels, and schedule are sufficient for inspection."""
    data = tmp_path / "questions.jsonl"
    data.write_bytes((ROOT / "tests" / "fixtures" / "posttrain_smoke.jsonl").read_bytes())
    raw = _raw_spec()
    raw["dataset"] = data.name
    config = _write_spec(tmp_path, raw)
    output = tmp_path / "plan.json"
    locked = planning.lock_study(config, output)
    expected = tuple(planning.schedule(locked, Job(checkpoint="base")))
    config.unlink()
    data.unlink()
    before = output.read_bytes()
    restored = planning.read_plan(output, check_source=True)
    assert restored == locked
    assert tuple(planning.schedule(restored, Job(checkpoint="base"))) == expected
    assert output.read_bytes() == before
    assert list(tmp_path.iterdir()) == [output]


def test_recursive_source_change_refuses_execution_not_inspection(
    tmp_path, monkeypatch, no_optional_imports,
):
    """A subpackage edit changes the code pin without touching real workspace sources."""
    package = tmp_path / "artificial-package"
    nested = package / "posttrain" / "nested"
    nested.mkdir(parents=True)
    anchor = package / "config.py"
    anchor.write_text("# artificial package anchor\n", encoding="utf-8")
    recipe = nested / "recipe.py"
    recipe.write_text("# recipe version one\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "__file__", str(anchor))
    before = config_module.source_digest()
    output = tmp_path / "plan.json"
    locked = planning.lock_study(SMOKE, output)
    assert locked.source_sha256 == before
    # Non-Python artifacts must not enter a code-only digest.
    (nested / "notes.txt").write_text("not executable source", encoding="utf-8")
    assert config_module.source_digest() == before
    recipe.write_text("# recipe version two\n", encoding="utf-8")
    changed = config_module.source_digest()
    assert changed != before
    assert planning.read_plan(output) == locked
    with pytest.raises(ArtifactError, match="code changed"):
        planning.read_plan(output, check_source=True)
    # Both accepted run_study inputs must fail before model import or run-directory creation.
    for plan_input in (locked, output):
        with pytest.raises(ArtifactError, match="[Ss]ource changed|code changed"):
            runner.run_study(plan_input, Job(checkpoint="base"), tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()
    recipe.rename(nested / "renamed.py")
    assert config_module.source_digest() != changed


@pytest.fixture
def repair_plan(tmp_path):
    """Lock multiple groups and crossed conditions to expose partition/seed mistakes."""
    data = tmp_path / "repair.jsonl"
    write_problems(data, make_repair_problems(count=5, seed=17))
    raw = _raw_spec()
    raw.update(dataset=str(data), budgets=[4, 8], samples=3)
    raw["prompts"] = [{"name": "control", "cue": "neutral"}, {"name": "check", "cue": "check"}]
    return planning.lock_study(_write_spec(tmp_path, raw), tmp_path / "plan.json")


@pytest.mark.parametrize("num_shards", [1, 2, 3, 7])
def test_schedule_is_exact_disjoint_partition_with_stable_ids_and_seeds(repair_plan, num_shards):
    """Changing worker count changes job IDs, never trial content or group membership."""
    problems = {problem.id: problem for problem in repair_plan.problems}
    for checkpoint in repair_plan.spec.checkpoints:
        whole_job = Job(checkpoint=checkpoint.name)
        whole = tuple(planning.schedule(repair_plan, whole_job))
        assert whole == tuple(planning.schedule(repair_plan, whole_job))
        assert len(whole) == 10 * 2 * 2 * 3
        expected = {trial.trial_id: trial for trial in whole}
        assert len(expected) == len(whole)
        observed = {}
        group_workers = defaultdict(set)
        job_ids = set()
        for index in range(num_shards):
            job = Job(checkpoint=checkpoint.name, shard_index=index, num_shards=num_shards)
            job_ids.add(planning.job_identity(repair_plan, job))
            partition = tuple(planning.schedule(repair_plan, job))
            assert partition == tuple(trial for trial in whole if planning.belongs(problems[trial.problem_id], job))
            for trial in partition:
                assert trial.trial_id not in observed
                assert trial == expected[trial.trial_id]
                observed[trial.trial_id] = trial
                group_workers[problems[trial.problem_id].group_id].add(index)
        assert observed == expected
        assert len(job_ids) == num_shards
        if num_shards > 1:
            assert planning.job_identity(repair_plan, whole_job) not in job_ids
        assert len(group_workers) == 5
        assert all(len(workers) == 1 for workers in group_workers.values())


def test_matched_variants_conditions_and_checkpoints_share_seed_but_samples_do_not(repair_plan):
    """Matched randomness is source/sample-specific across the full factorial design."""
    problems = {problem.id: problem for problem in repair_plan.problems}
    seeds = defaultdict(set)
    variants = defaultdict(set)
    for checkpoint in repair_plan.spec.checkpoints:
        for trial in planning.schedule(repair_plan, Job(checkpoint=checkpoint.name)):
            problem = problems[trial.problem_id]
            key = (problem.source_id, trial.sample)
            seeds[key].add(trial.seed)
            variants[key].add(problem.prefix)
    assert all(len(values) == 1 for values in seeds.values())
    assert all(values == {"correct", "error"} for values in variants.values())
    for source in {problem.source_id for problem in repair_plan.problems}:
        assert len({next(iter(seeds[source, sample])) for sample in range(3)}) == 3


def test_unknown_checkpoint_rejected_by_job_identity_and_schedule(repair_plan):
    """A well-formed label must still resolve to an explicitly locked checkpoint."""
    job = Job(checkpoint="not-in-plan")
    with pytest.raises(ValueError, match="absent"):
        planning.job_identity(repair_plan, job)
    with pytest.raises(ValueError, match="absent"):
        tuple(planning.schedule(repair_plan, job))


@pytest.mark.parametrize("partition", [
    {"num_shards": 0}, {"num_shards": 1025}, {"num_shards": True},
    {"shard_index": -1}, {"shard_index": True}, {"shard_index": 1.5},
    {"shard_index": 2, "num_shards": 2},
])
def test_job_rejects_invalid_partition_contract(partition):
    """Invalid ordinals and coerced counts cannot define overlapping worker schedules."""
    with pytest.raises(ValidationError):
        Job(checkpoint="base", **partition)


@pytest.mark.parametrize("change", ["budget", "prompt"])
def test_scientific_budget_or_prompt_change_gets_new_study_and_trial_ids(tmp_path, change):
    """A token budget is science, unlike an invocation's limit on new attempts."""
    original = planning.lock_study(SMOKE, tmp_path / "original.json")
    raw = _raw_spec()
    if change == "budget":
        raw["budgets"] = [6]
        raw["generation"]["max_new_tokens"] = 6
    else:
        raw["prompts"][0]["instruction"] = "Check the arithmetic, then declare the final answer."
    changed = planning.lock_study(_write_spec(tmp_path, raw), tmp_path / "changed.json")
    assert changed.study_id != original.study_id
    job = Job(checkpoint="base")
    assert {row.trial_id for row in planning.schedule(original, job)}.isdisjoint(
        row.trial_id for row in planning.schedule(changed, job)
    )


@pytest.mark.parametrize(("change", "message"), [
    ("greedy_repeats", "greedy"), ("no_prompt_room", "leave room"),
    ("duplicate_checkpoints", "unique"), ("duplicate_prompts", "unique"),
    ("duplicate_budgets", "unique"), ("disabled_generation", "Enable generation"),
    ("mismatched_first_budget", "first budget"),
])
def test_schema_rejects_inconsistent_study_design(change, message):
    """Reject fake replications, duplicate cells, and budgets incompatible with execution."""
    raw = _raw_spec()
    if change == "greedy_repeats":
        raw["generation"].update(do_sample=False, temperature=1.0, top_p=1.0, top_k=0)
    elif change == "no_prompt_room":
        raw["inference"]["max_context_tokens"] = max(raw["budgets"])
    elif change.startswith("duplicate_"):
        field = change.removeprefix("duplicate_")
        raw[field].append(raw[field][0])
    elif change == "disabled_generation":
        raw["generation"]["enabled"] = False
    else:
        raw["generation"]["max_new_tokens"] += 1
    with pytest.raises(ValidationError, match=message):
        StudySpec.model_validate(raw)


@pytest.mark.parametrize("refs", [
    {"revision": "main"}, {"revision": "a" * 39}, {"revision": "A" * 40},
    {"tokenizer_id": None}, {"tokenizer_revision": None},
    {"tokenizer_revision": "main"}, {"tokenizer_revision": "b" * 39},
])
def test_locked_plan_schema_rejects_missing_or_nonimmutable_refs(tmp_path, refs):
    """A valid configurable checkpoint is not necessarily a valid locked checkpoint."""
    locked = planning.lock_study(SMOKE, tmp_path / "plan.json")
    raw = locked.model_dump(mode="json")
    raw["spec"]["checkpoints"] = [_hf_checkpoint(**refs)]
    with pytest.raises(ValidationError, match="Lock full model and tokenizer revisions"):
        StudyPlan.model_validate(raw)