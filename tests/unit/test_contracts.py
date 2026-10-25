# Exercise input identity, leakage safeguards, and import safety without numerical model work.
"""Offline schema/configuration/UI-gate tests for the first research slice."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from interp.cli import main
from interp.config import compute_run_id, load_config, prepare_run, with_execution
from interp.data import dataset_digest, validate_examples
from interp.runner import ExplicitRunGate, RunSummary
from interp.runtime import trial_id, trial_seed
from interp.schemas import ExecutionSpec, GenerationSpec, ModelSpec


def test_identity_ignores_paths_budgets_and_row_order(prepared, tmp_path):
    """Operational changes/reordering must not redefine the scientific observations."""
    # Change only invocation budgets and the artifact destination.
    other = with_execution(prepared, output_dir=tmp_path / "different", max_new_trials=1, shard_records=10)
    # Preserve logical identity across operational choices.
    assert other.run_id == prepared.run_id
    # Preserve semantic dataset identity across input row order.
    assert dataset_digest(reversed(prepared.examples)) == prepared.dataset_sha256
    # Verify exact prompt whitespace is nevertheless a scientific content change.
    raw = prepared.examples[0].model_dump(mode="python")
    # Introduce one real trailing space rather than formatting JSON differently.
    raw["text"] += " "
    # Revalidate the modified observation without applying cross-pair checks here.
    changed = type(prepared.examples[0]).model_validate(raw)
    # Check that exact textual content is included in semantic identity.
    assert dataset_digest((changed, *prepared.examples[1:])) != prepared.dataset_sha256


def test_pairing_and_lineage_fail_before_model_work(prepared):
    """Reject missing variants, duplicate observations, and source/group leakage."""
    # Reject an incomplete final pair rather than silently dropping it.
    with pytest.raises(ValueError, match="exactly one"):
        # Validate the complete dataset minus one required condition.
        validate_examples(prepared.examples[:-1], prepared.spec.conditions)
    # Reject duplicate logical observations even when they have identical text.
    with pytest.raises(ValueError, match="Duplicate"):
        # Attempt to ingest one source row twice.
        validate_examples((*prepared.examples, prepared.examples[0]), prepared.spec.conditions)
    # Change the split of one derived variant to simulate evaluation leakage.
    raw = prepared.examples[0].model_dump(mode="python")
    # Deliberately move a correlated source into a held-out split.
    raw["split"] = "test"
    # Preserve schema validity while changing the leakage-sensitive lineage assignment.
    changed = type(prepared.examples[0]).model_validate(raw)
    # Reject the cross-split related observation.
    with pytest.raises(ValueError, match="crosses"):
        # Check leakage across all lineage axes, not just example IDs.
        validate_examples((changed, *prepared.examples[1:]), prepared.spec.conditions)


def test_yaml_duplicate_and_unsafe_keys_are_rejected(tmp_path):
    """Safe configuration parsing must not overwrite duplicate values or construct Python objects."""
    # Use an isolated temporary configuration, not the research documents.
    path = tmp_path / "invalid.yaml"
    # Supply a duplicate setting before schema construction can choose the last value.
    path.write_text("name: first\nname: second\n", encoding="utf-8")
    # Expect duplicate-key validation rather than a later missing-field error.
    with pytest.raises(ValueError, match="Duplicate YAML"):
        # Parse only local settings; no environment/model work is needed.
        load_config(path)
    # Exercise an unsafe root tag that SafeLoader must never execute.
    path.write_text("!!python/object/apply:builtins.eval ['1 + 1']", encoding="utf-8")
    # The supported contract refuses the non-mapping root before construction.
    with pytest.raises(ValueError, match="mapping"):
        # Do not instantiate or execute the tagged Python callable.
        load_config(path)


@pytest.mark.parametrize("revision", ["main", "v1", "a" * 7])
def test_pretrained_revisions_must_be_immutable(revision):
    """Moving branches/tags and abbreviated commits cannot define reproducible weights."""
    # Refuse ambiguous checkpoint identities without any pretrained-model request.
    with pytest.raises(ValidationError, match="40-character"):
        # Validate schema only, using a legitimate example model repository name.
        ModelSpec(backend="huggingface", model_id="openai-community/gpt2", revision=revision)


def test_schema_rejects_ineffective_nonfinite_and_unknown_settings():
    """Scientific typos and ignored/numerically undefined settings must fail clearly."""
    # Reject a temperature that would be ignored by greedy decoding.
    with pytest.raises(ValidationError, match="do_sample"):
        # Keep the intended strategy explicit instead of logging a warning and ignoring it.
        GenerationSpec(temperature=0.8)
    # Reject non-finite sampling settings before generating any tokens.
    with pytest.raises(ValidationError):
        # Supply a nonstandard JSON/scientific numeric value deliberately.
        GenerationSpec(do_sample=True, temperature=float("nan"))
    # Reject an unknown setting rather than silently accepting a typo.
    with pytest.raises(ValidationError, match="Extra inputs"):
        # Misspell a generation field to exercise extra='forbid'.
        GenerationSpec(max_new_token=2)
    # Reject a budget with no time available for actual work.
    with pytest.raises(ValidationError, match="smaller"):
        # Give the finalization margin the whole invocation budget deliberately.
        ExecutionSpec(max_seconds=5.0, shutdown_margin_seconds=5.0)


def test_trial_ids_and_paired_seeds_are_order_independent(prepared):
    """Stable IDs distinguish replicates; matched conditions deliberately share sampling seeds."""
    # Select the two matched condition variants of the first pair.
    first, second = prepared.examples[:2]
    # Preserve paired randomness despite distinct prompt/condition identities.
    assert trial_seed(123, first.pair_id, 0) == trial_seed(123, second.pair_id, 0)
    # Give repeated observations distinct deterministic seed assignments.
    assert trial_seed(123, first.pair_id, 0) != trial_seed(123, first.pair_id, 1)
    # Keep the two logical conditions distinguishable even with matched randomness.
    assert trial_id(prepared.run_id, first.id, 0) != trial_id(prepared.run_id, second.id, 0)
    # Keep retries of the same logical trial stable across execution attempts.
    assert trial_id(prepared.run_id, first.id, 0) == trial_id(prepared.run_id, first.id, 0)


def test_explicit_click_is_consumed_once_and_failure_needs_new_click():
    """Lazy/reactive recomputation must not reexecute a previous model-loading action."""
    # Create a model-free notebook action guard.
    gate = ExplicitRunGate()
    # Record action invocations without loading a numerical library.
    calls = []
    # Prepare one small synthetic shared-runner summary.
    summary = RunSummary("0" * 64, "unused", 1, 1, 0, 0, 1, "complete")

    def action():
        """Count one explicit request and return only its summary."""
        # Record the action boundary for duplicate-execution assertions.
        calls.append(1)
        # Return a small summary, not a retained model/cache.
        return summary

    # Perform a genuinely clicked ticket for the first time.
    assert gate.perform(("frozen", 1), action) == summary
    # Recomputing the same cell reuses the previous completed summary.
    assert gate.perform(("frozen", 1), action) == summary
    # Assert no duplicate model action was invoked.
    assert calls == [1]

    def failed_action():
        """Simulate a model error after a ticket has already been consumed."""
        # Count the failing action too; it must not retry under display recomputation.
        calls.append(2)
        # Simulate an arbitrary numerical failure.
        raise RuntimeError("Injected action failure.")

    # A new deliberate click can attempt execution once.
    with pytest.raises(RuntimeError):
        # Fail after the gate marks this exact request consumed.
        gate.perform(("frozen", 2), failed_action)
    # An unchanged ticket must not implicitly retry the failed operation.
    with pytest.raises(ValueError, match="consumed"):
        # Test reactive recomputation after failure.
        gate.perform(("frozen", 2), failed_action)
    # Preserve exactly one success and one failed action invocation.
    assert calls == [1, 2]


def test_cli_validation_is_model_free(prepared, monkeypatch, capsys, tmp_path):
    """CLI validation reports data identity but never claims token/model validation."""
    # Provide prepared local inputs without invoking an execution backend.
    monkeypatch.setattr("interp.cli.prepare_run", lambda path: prepared)
    # Capture the temporary directory state before this read-only command.
    before = set(tmp_path.iterdir())
    # Execute only the validation subcommand through the actual CLI dispatcher.
    assert main(["validate", "unused-config.yaml"]) == 0
    # Read the successful machine-readable preview.
    report = json.loads(capsys.readouterr().out)
    # Do not misrepresent model-free validation as checked tokenization.
    assert report["tokenization_checked"] is False
    # Preserve the full paired observation count.
    assert report["planned_trials"] == 4
    # Verify no result directory, lock, or checkpoint was created.
    assert set(tmp_path.iterdir()) == before


def test_core_imports_have_no_model_network_or_artifact_side_effects(tmp_path):
    """Import all core entry points in a fresh process without optional heavy modules."""
    # Use a new interpreter so earlier numerical tests cannot contaminate import-state checks.
    script = """
from pathlib import Path
import sys
import interp, interp.config, interp.models, interp.runner, interp.hub, interp.plots, interp.cli
assert not any(name in sys.modules for name in ('torch', 'transformers', 'huggingface_hub', 'plotly', 'marimo'))
assert not list(Path.cwd().iterdir())
print('safe')
"""
    # Avoid incidental bytecode writes while preserving the installed editable package path.
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    # Run only the bounded import-safety script with no shell or model command.
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=environment, check=True, text=True, capture_output=True)
    # Assert the new process completed its explicit safety checks.
    assert result.stdout.strip() == "safe"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1: value\n", "keys must be strings"),
        ("settings: &settings {nested: *settings}\n", "Cyclic YAML"),
        ("defaults: &defaults {name: first}\n<<: *defaults\n", "YAML merges"),
        ("model:\n  device: cpu\n  device: cuda\n", "Duplicate YAML"),
    ],
)
def test_ambiguous_yaml_is_rejected_before_schema_construction(tmp_path, text, message):
    """Nested duplicates, non-string keys, merges, and recursive aliases fail deterministically."""
    # Materialize only one disposable malformed settings file for this specific case.
    config = tmp_path / "ambiguous.yaml"
    # Preserve exact YAML syntax so parsing tests real node/alias semantics.
    config.write_text(text, encoding="utf-8")
    # Reject ambiguity before choosing values or loading a model/backend.
    with pytest.raises(ValueError, match=message):
        # Exercise the same explicit safe configuration parser used by both frontends.
        load_config(config)


def test_paths_resolve_against_yaml_and_scientific_changes_alter_identity(prepared, tmp_path, monkeypatch):
    """Changing the shell cwd cannot change data/output resolution; science still changes identity."""
    # Import only ordinary safe YAML serialization for disposable test configuration.
    import yaml

    # Create an isolated configuration/data directory separate from the caller's cwd.
    directory = tmp_path / "inputs"
    # Materialize test fixtures, not any user-managed research dataset.
    directory.mkdir()
    # Save exact validated observations in ordinary JSONL with stable semantics.
    (directory / "data.jsonl").write_text("".join(example.model_dump_json() + "\n" for example in prepared.examples), encoding="utf-8")
    # Convert the reference specification to safe primitive YAML input.
    raw = prepared.spec.model_dump(mode="json")
    # Make both operational/local input locations deliberately relative to the YAML file.
    raw["dataset"]["path"] = "data.jsonl"
    # Do not resolve this destination against the later shell cwd.
    raw["execution"]["output_dir"] = "results"
    # Save a valid local config using only safe primitive values.
    config = directory / "run.yaml"
    # This is disposable test input, not a terminal edit of a workspace file.
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    # Simulate launching the same CLI/notebook from another directory.
    monkeypatch.chdir(tmp_path)
    # Model-free preparation must consistently resolve against the config anchor.
    loaded = prepare_run(config)
    # Preserve exact input resolution across caller cwd changes.
    assert loaded.spec.dataset.path == directory / "data.jsonl"
    # Preserve exact operational result resolution across caller cwd changes.
    assert loaded.spec.execution.output_dir == directory / "results"
    # Machine-local paths alone do not alter the experiment's scientific identity.
    assert loaded.run_id == prepared.run_id
    # Change a predeclared scientific sampling/replication choice deliberately.
    raw = loaded.spec.model_dump(mode="python")
    # Additional repeats change the observation schedule, not just the invocation budget.
    raw["replicates"] = 2
    # Revalidate changed scientific settings independently of operational overrides.
    changed = type(loaded.spec).model_validate(raw)
    # Bind the changed planned schedule into a new scientific run identity.
    assert compute_run_id(changed, loaded.dataset_sha256, loaded.source_sha256) != loaded.run_id
    # Model-free configuration validation must not create its declared output directory.
    assert not loaded.spec.execution.output_dir.exists()


def test_predeclared_dataset_digest_mismatch_blocks_execution(prepared, tmp_path):
    """A changed dataset cannot pass a pinned expected content check under the original claim."""
    # Use established safe YAML primitives for this local test input.
    import yaml

    # Reuse actual checked-in data but deliberately pin a different semantic hash.
    raw = prepared.spec.model_dump(mode="json")
    # This digest differs from the real engineering observations without exposing raw data.
    raw["dataset"]["expected_sha256"] = "0" * 64
    # Choose a disposable configuration location for the failed validation attempt.
    config = tmp_path / "pinned.yaml"
    # Keep absolute validated dataset paths so only the expected content differs.
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    # Reject mismatched predeclaration before models, trials, files, or network work.
    with pytest.raises(ValueError, match="expected_sha256"):
        # Run only the shared model-free preparation path.
        prepare_run(config)


@pytest.mark.parametrize("change", ["unknown-target", "duplicate-id", "duplicate-text"])
def test_candidate_contracts_reject_meaningless_comparisons(prepared, change):
    """Targets must name one of two genuinely distinct candidate labels and exact texts."""
    # Convert one legitimate example to validation primitives for a controlled schema violation.
    raw = prepared.examples[0].model_dump(mode="python")
    # Exercise each candidate contract violation without loading/tokenizing a model.
    if change == "unknown-target":
        # Select a target absent from the exact predeclared candidate set.
        raw["target_candidate"] = "absent"
    # Duplicate labels invalidate downstream likelihood-margin identity.
    elif change == "duplicate-id":
        # Keep different strings but give both candidates the same logical label.
        raw["candidates"][1]["id"] = raw["candidates"][0]["id"]
    # Duplicate texts would manufacture a meaningless equal-choice comparison.
    else:
        # Keep distinct labels but make their exact scientific strings identical.
        raw["candidates"][1]["text"] = raw["candidates"][0]["text"]
    # Reject each invalid comparison as a schema error, not a later metric surprise.
    with pytest.raises(ValidationError):
        # Exercise the actual observation contract used during JSONL ingestion.
        type(prepared.examples[0]).model_validate(raw)
