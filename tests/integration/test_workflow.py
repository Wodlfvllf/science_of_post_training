# Exercise the real tiny model through shared entry points without internet or GPU access.
"""Offline end-to-end CLI/library/notebook safety and dataset-independent resume tests."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import yaml

from interp.artifacts import read_run
from interp.cli import main
from interp.config import prepare_run, with_execution
from interp.runner import resume_experiment, run_experiment

# Keep integration notebook/config locations independent of shell cwd.
ROOT = Path(__file__).resolve().parents[2]


def test_real_offline_cli_and_shared_library_replay(bundle, tmp_path, capsys):
    """The CLI and notebook entry function must produce the same deterministic observations."""
    # Prepare the checked-in engineering dataset with a temporary artifact destination.
    prepared = with_execution(prepare_run(ROOT / "configs" / "smoke.yaml"), output_dir=tmp_path / "library")
    # Execute the exact shared function called by the notebook's explicit Run action.
    library_summary = run_experiment(prepared)
    # Execute the same specification through the thin CLI dispatcher in a separate directory.
    exit_code = main(["run", str(ROOT / "configs" / "smoke.yaml"), "--output-dir", str(tmp_path / "cli")])
    # Verify the actual CLI operation succeeded, not merely argument parsing.
    assert exit_code == 0
    # Read the real machine-readable completion report.
    cli_summary = json.loads(capsys.readouterr().out)
    # Preserve identical science and complete schedule across frontends/locations.
    assert library_summary.run_id == cli_summary["run_id"] and cli_summary["completed_trials"] == 4
    # Inspect exact saved numerical outputs from both execution frontends.
    library_records = read_run(tmp_path / "library").records
    # Load only verified CLI artifacts for comparison.
    cli_records = read_run(tmp_path / "cli").records
    # Compare logical identities, isolated seeds, exact tokens, and full validated score/config results.
    assert [(record.trial_id, record.seed, record.result) for record in library_records] == [(record.trial_id, record.seed, record.result) for record in cli_records]


def test_real_resume_uses_archived_data_after_original_file_disappears(bundle, tmp_path):
    """Clean-machine resume must not depend on original JSONL/config dataset paths."""
    # Validate a canonical reference fixture before copying it into a temporary test dataset.
    reference = prepare_run(ROOT / "configs" / "smoke.yaml")
    # Create a test-local original dataset that can safely be removed later.
    dataset = tmp_path / "original.jsonl"
    # Preserve exact validated examples in ordinary JSONL text.
    dataset.write_text("".join(example.model_dump_json() + "\n" for example in reference.examples), encoding="utf-8")
    # Build a test-local YAML specification pointing only at temporary paths.
    raw = reference.spec.model_dump(mode="json")
    # Override only the dataset's local path, not its identity or content.
    raw["dataset"]["path"] = str(dataset)
    # Isolate all actual results inside this pytest temporary directory.
    raw["execution"]["output_dir"] = str(tmp_path / "restored-run")
    # Persist the test's temporary YAML using standard safe primitive serialization.
    config = tmp_path / "run.yaml"
    # This file is test setup, not a change to any user's checked-in configuration.
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    # Start one actual model trial while preserving the complete scientific schedule.
    limited = with_execution(prepare_run(config), max_new_trials=1)
    # Save a partial real numerical checkpoint through the shared entry point.
    assert run_experiment(limited).completed_trials == 1
    # Remove only the disposable test-local original dataset.
    dataset.unlink()
    # Remove the invocation cap but preserve the archived model/settings/numerical profile.
    execution = with_execution(limited, max_new_trials=None).spec.execution
    # Resume actual model computation solely from the verified archived example snapshot.
    completed = resume_experiment(execution.output_dir, execution=execution)
    # Execute only three missing logical observations from their own stable seeds.
    assert (completed.completed_trials, completed.new_attempts) == (4, 3)


def test_notebook_import_does_not_load_or_run_models(bundle, monkeypatch, tmp_path):
    """Defining the marimo app cannot execute a model/experiment/upload action."""
    # Build an import specification for the actual checked-in thin notebook file.
    specification = importlib.util.spec_from_file_location("research_smoke_notebook", ROOT / "notebooks" / "smoke.py")
    # Create a separate module rather than executing the notebook as __main__.
    module = importlib.util.module_from_spec(specification)

    def forbidden(*args, **kwargs):
        """Make any implicit model action fail this import-safety test immediately."""
        # A notebook's import/startup definition must not launch expensive work.
        raise AssertionError("Notebook import must not run or load models.")

    # Ensure an accidental model load would be observable despite earlier numerical fixtures.
    monkeypatch.setattr("interp.models.load_model", forbidden)
    # Ensure accidental experiment invocation would be observable too.
    monkeypatch.setattr("interp.runner.run_experiment", forbidden)
    # Check a directory where importing the UI must not create result artifacts.
    monkeypatch.chdir(tmp_path)
    # Define all marimo cells without launching the frontend or clicking a Run action.
    specification.loader.exec_module(module)
    # Confirm app construction succeeded without depending on a live model object.
    assert module.app is not None
    # Preserve the empty result-working directory during notebook definition.
    assert not list(tmp_path.iterdir())


def test_cli_errors_do_not_print_dependency_secret_messages(monkeypatch, capsys):
    """Arbitrary HTTP/model exception messages must not leak through CLI error reporting."""

    def forbidden(*args, **kwargs):
        """Simulate a dependency error with an arbitrary secret-bearing message."""
        # This fake value must stay out of stdout/stderr even though the command fails.
        raise RuntimeError("hf_not_a_real_secret_DO_NOT_LOG")

    # Inject a controlled dependency failure at a selected CLI dispatch boundary.
    monkeypatch.setattr("interp.cli.prepare_run", forbidden)
    # Report a machine-detectable failure without dumping a traceback or arbitrary message.
    assert main(["validate", "unused-config.yaml"]) == 1
    # Capture only what the actual CLI exposed to the user.
    captured = capsys.readouterr()
    # Preserve the useful safe error category.
    assert "RuntimeError" in captured.err
    # Do not echo arbitrary exception contents to either output stream.
    assert "DO_NOT_LOG" not in captured.out + captured.err


def test_notebook_guard_recreation_always_resets_button_and_submission():
    """Recomputing one cell cannot reset consumed state while retaining a positive click counter."""
    # Parse the real notebook definition without evaluating any reactive cells.
    source = ast.parse((ROOT / "notebooks" / "smoke.py").read_text(encoding="utf-8"))
    # Track which top-level decorated function owns each action-state assignment.
    owners = {}
    # Inspect definitions, not implementation strings or generated frontend code.
    for cell in source.body:
        # Only actual Python cell functions can own the local action-state variables.
        if isinstance(cell, ast.FunctionDef):
            # Inspect each cell's straight-line local state construction.
            for statement in cell.body:
                # Limit the check to ordinary assignments rather than dependency parameter names.
                if isinstance(statement, ast.Assign):
                    # Capture the actual lexical owner of each assigned state variable.
                    for target in statement.targets:
                        # Ignore assignments to attributes/subscripts that are not cell definitions.
                        if isinstance(target, ast.Name):
                            # Keep object identity so two similarly named cell functions cannot collide.
                            owners[target.id] = cell
    # A recreated guard must be inseparable from a fresh submission and a zero-click button.
    assert owners["run_gate"] is owners["run_button"] is owners["submission_id"]