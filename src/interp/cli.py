# Command-line entry point: validate/inspect are read-only; run/sync/restore are explicit actions.
"""A thin CLI over the same configuration, runner, and persistence functions as the notebook.

Use ``interp --help`` for commands. No credentials are accepted as command arguments
or experiment fields. Exceptions are reported conservatively without raw dependency
tracebacks/messages that could expose authorization tokens or prompt contents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .artifacts import ArtifactError, read_run
from .config import prepare_run, with_execution
from .metrics import paired_effects
from .posttrain.errors import StudyError
from .runner import TrialExecutionError, resume_experiment, run_experiment, summarize_run
from .schemas import ExecutionSpec


def _parser() -> argparse.ArgumentParser:
    """Build command syntax without loading a model, opening a run, or contacting storage."""
    # Describe the explicit execution boundary in the top-level command help.
    parser = argparse.ArgumentParser(prog="interp", description="A small, explicit research workflow.")
    # Require the caller to choose an operation rather than executing a default experiment.
    commands = parser.add_subparsers(dest="command", required=True)
    # Keep the research recipes separate from the original paired-continuation contract.
    from .posttrain.cli import add_commands

    add_commands(commands)
    # Provide a model-free configuration/data inspection operation.
    validate = commands.add_parser("validate", help="Validate config/paired data; no model or writes.")
    # Require one local YAML file for the preview.
    validate.add_argument("config", type=Path)
    # Provide an explicit shared-runner execution operation.
    run = commands.add_parser("run", help="Explicitly load and run/resume a configured experiment.")
    # Require one validated configuration rather than guessing the selected research project.
    run.add_argument("config", type=Path)
    # Allow only a deliberate opt-in to model/tokenizer network downloads.
    run.add_argument("--allow-downloads", action="store_true", help="Allow pinned HF model/tokenizer downloads.")
    # Allow invocation budgets without changing scientific identity.
    _budget_arguments(run, output=True)
    # Provide dataset-path-independent continuation from a restored/saved snapshot.
    resume = commands.add_parser("resume", help="Resume verified saved examples without the original data file.")
    # Require the explicit existing checkpoint directory.
    resume.add_argument("run_dir", type=Path)
    # Keep restored pretrained models cached-only unless explicitly allowed to download.
    resume.add_argument("--allow-downloads", action="store_true")
    # Allow new invocation budgets but always keep the explicitly selected run directory.
    _budget_arguments(resume, output=False)
    # Expose CPU-only inspection independent of optional Torch/model libraries.
    inspect = commands.add_parser("inspect", help="Verify results and report paired/incomplete comparisons.")
    # Require one explicit checkpoint to inspect.
    inspect.add_argument("run_dir", type=Path)
    # Expose a separate synchronous upload action, never a run-time implicit side effect.
    upload = commands.add_parser("sync", help="Publish a small checkpoint to an existing private HF dataset repo.")
    # Require the checkpoint whose exact bytes will be published.
    upload.add_argument("run_dir", type=Path)
    # Require the user to select the results destination explicitly.
    upload.add_argument("--repo", required=True)
    # Expose revision-pinned, checksum-verified restoration as another deliberate action.
    download = commands.add_parser("restore", help="Restore a verified remote checkpoint into a new directory.")
    # Require immutable experimental identity, not a user-readable name alone.
    download.add_argument("run_id")
    # Require the private results destination independently of model repositories.
    download.add_argument("--repo", required=True)
    # Refuse overwriting by requiring an explicit new local destination.
    download.add_argument("--output-dir", type=Path, required=True)
    # Allow the exact previously acknowledged commit instead of a moving branch.
    download.add_argument("--revision")
    # Return only parser definitions; no operation has executed yet.
    return parser


def _budget_arguments(parser: argparse.ArgumentParser, *, output: bool) -> None:
    """Add operational invocation controls, never scientific settings, to a subcommand."""
    # Allow a finite number of fresh attempts for incremental engineering/research runs.
    parser.add_argument("--max-new-trials", type=int)
    # Allow a cooperative wall-time limit including explicit model loading.
    parser.add_argument("--max-seconds", type=float)
    # Leave a declared finalization/upload margin instead of scheduling to a hard reset.
    parser.add_argument("--shutdown-margin-seconds", type=float)
    # Only config-driven runs may choose a new explicit output destination.
    if output:
        # Keep archive-based resume anchored to its original selected directory.
        parser.add_argument("--output-dir", type=Path)


def _overrides(arguments: argparse.Namespace) -> dict[str, Any]:
    """Collect explicitly supplied operational overrides without changing absent defaults."""
    # Keep only declared ExecutionSpec field names used by these commands.
    fields = ("max_new_trials", "max_seconds", "shutdown_margin_seconds", "output_dir")
    # Ignore unspecified values rather than wiping the validated YAML defaults with None.
    return {field: getattr(arguments, field) for field in fields if getattr(arguments, field, None) is not None}


def _dispatch(arguments: argparse.Namespace) -> dict[str, Any]:
    """Call one shared library operation selected explicitly by the CLI parser."""
    # Research subcommands own their explicit dataset/model/result boundaries.
    if arguments.command == "posttrain":
        from .posttrain.cli import dispatch

        return dispatch(arguments)
    # Keep model-free validation separate from expensive execution.
    if arguments.command == "validate":
        # Validate all settings, pairing, and split lineage without model/tokenizer loading.
        prepared = prepare_run(arguments.config)
        # Report resolved inputs and semantic identity without writing any artifact.
        return {"run_id": prepared.run_id, "dataset_sha256": prepared.dataset_sha256, "examples": len(prepared.examples), "planned_trials": len(prepared.examples) * prepared.spec.replicates, "model_backend": prepared.spec.model.backend, "output_dir": str(prepared.spec.execution.output_dir), "tokenization_checked": False}
    # Execute precisely the same entry point used by the thin research notebook.
    if arguments.command == "run":
        # Apply validated operational overrides to the frozen scientific preparation.
        prepared = with_execution(prepare_run(arguments.config), **_overrides(arguments))
        # Explicitly start model work, keeping downloads opt-in and uploads separate.
        return run_experiment(prepared, allow_downloads=arguments.allow_downloads).to_dict()
    # Continue from verified saved examples even after their original paths disappear.
    if arguments.command == "resume":
        # Resolve the explicit current checkpoint location, not an old archived machine path.
        directory = arguments.run_dir.expanduser().resolve()
        # Obtain original operational defaults from a model-free checksum-verified snapshot.
        raw = read_run(directory).metadata.spec.execution.model_dump(mode="python")
        # Apply only new explicitly supplied invocation controls.
        raw.update(_overrides(arguments))
        # Force the selected existing run destination for archive-based resume.
        raw["output_dir"] = directory
        # Validate budget overrides before any pretrained model is loaded.
        execution = ExecutionSpec.model_validate(raw)
        # Resume through the same ordinary trial runner and numerical-compatibility checks.
        return resume_experiment(directory, execution=execution, allow_downloads=arguments.allow_downloads).to_dict()
    # Inspect verified saved likelihoods without importing optional model dependencies.
    if arguments.command == "inspect":
        # Resolve the explicit artifact location for read-only inspection.
        directory = arguments.run_dir.expanduser().resolve()
        # Report all planned comparisons, including pairs lacking completed evidence.
        return {"summary": summarize_run(directory).to_dict(), "paired_effects": paired_effects(read_run(directory))}
    # Import network adapters only for an explicitly chosen sync/restore action.
    from .hub import HuggingFaceRemote, restore, sync

    # Validate the exact selected private results destination without reading a token.
    remote = HuggingFaceRemote(arguments.repo)
    # Explicitly publish one bounded atomic checkpoint when sync was selected.
    if arguments.command == "sync":
        # Await publication acknowledgement before reporting any remote completion.
        receipt = sync(arguments.run_dir.expanduser().resolve(), remote)
    # The only remaining parser-defined action is pinned restore into a new directory.
    else:
        # Verify every referenced artifact before exposing the restored run directory.
        receipt = restore(arguments.output_dir.expanduser().resolve(), remote, arguments.run_id, revision=arguments.revision)
    # Return destination/revision/completion metadata, never authentication credentials.
    return receipt.model_dump(mode="json")


def main(argv: list[str] | None = None) -> int:
    """Parse an explicit command, execute it once, and return a process exit code."""
    # Parse syntax/help before optional models or remote clients can be imported.
    arguments = _parser().parse_args(argv)
    # Keep successful JSON output distinct from controlled safe error reporting.
    try:
        # Dispatch only the explicitly selected operation.
        result = _dispatch(arguments)
    # Preserve ordinary user interruption without showing dependency tracebacks/secrets.
    except KeyboardInterrupt:
        # Explain that local finalization ran but remote upload was not automatic.
        print("Interrupted; inspect the last local checkpoint before resuming.", file=sys.stderr)
        # Use the conventional shell interrupt exit code.
        return 130
    # Controlled artifact/runner errors contain intentionally safe, actionable messages.
    except (ArtifactError, TrialExecutionError, StudyError) as error:
        # Show only workbench-controlled errors, not nested model/HTTP exception messages.
        print(str(error), file=sys.stderr)
        # Report an unsuccessful operation without falsely acknowledging completion.
        return 1
    # Handle validation/missing dependencies/network errors without printing secret-bearing values.
    except Exception as error:
        # Preserve the error category while suppressing arbitrary input/HTTP tracebacks.
        print(f"{type(error).__name__}: operation failed. Check config/data/dependencies; inspect saved attempts if a run started.", file=sys.stderr)
        # Make failure machine-detectable for notebook subprocesses and remote shells.
        return 1
    # Emit complete finite JSON suitable for notebook or shell inspection.
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    # Report success only after the selected shared operation actually returns.
    return 0
