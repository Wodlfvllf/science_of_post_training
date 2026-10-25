# Schedule ordinary trial functions here; the notebook and CLI share this same runner.
# Resume is at completed trial boundaries, never inside a live model generation.
"""Stable paired trial scheduling, safe attempts, cooperative budgets, and local resume."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .artifacts import ArtifactError, RunStore, read_run
from .config import PreparedRun, prepare_run, source_digest
from .runtime import CancellationRequested, runtime_profile, trial_id, trial_seed, utc_now
from .schemas import Example, ExecutionSpec, RunMetadata, SyncReceipt, TrialRecord, TrialResult

# Keep the recipe seam an ordinary function, not a registry or inheritance hierarchy.
TrialFunction = Callable[[Example, int, int, Callable[[], bool]], TrialResult]


class TrialExecutionError(RuntimeError):
    """A failed attempt was saved; the runner stopped instead of hiding the failure."""


@dataclass(frozen=True)
class RunSummary:
    """A small result summary distinguishing local completion from remote acknowledgement."""

    # Identify the science independently of its filesystem location.
    run_id: str
    # Record the directory containing the verified local checkpoint.
    run_dir: str
    # Preserve the complete planned observation count, not just this invocation's budget.
    planned_trials: int
    # Count only validated published local successes.
    completed_trials: int
    # Count the last acknowledged remote boundary, which may lag local progress.
    remotely_acknowledged_trials: int
    # Preserve failed attempts for inspection instead of silently excluding them.
    failed_attempts: int
    # Show how much actual new execution this invocation performed.
    new_attempts: int
    # Explain a partial invocation without claiming the whole experiment completed.
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible summary data for the CLI and thin notebook."""
        # Convert this frozen record without changing experiment or artifact state.
        return asdict(self)


class ExplicitRunGate:
    """Consume a notebook click ticket once, even under lazy/reactive recomputation.

    The gate intentionally keeps small in-memory bookkeeping, not a model/cache.
    A failed click is also consumed; retry needs a new explicit click. Recreating
    the notebook/kernel resets this guard, while the durable runner still prevents
    duplicate logical successes.
    """

    def __init__(self) -> None:
        """Create an empty action guard without executing work or touching files."""
        # Remember results by frozen-submission ID and monotonic button-click count.
        self._requests: dict[tuple[str, int], RunSummary | None] = {}

    def perform(self, ticket: tuple[str, int], action: Callable[[], RunSummary]) -> RunSummary:
        """Execute a fresh explicit click once or return its already completed summary."""
        # Reject an unclicked or invalid UI request before invoking any model action.
        if ticket[1] < 1:
            # Require an actual frontend click, not initial widget construction.
            raise ValueError("An explicit Run click is required.")
        # Return/deny a previously consumed request rather than re-running an expensive action.
        if ticket in self._requests:
            # Retrieve the previous result, or None if that action failed/interrupted.
            previous = self._requests[ticket]
            # A failed action cannot be retried implicitly by a display/plot recomputation.
            if previous is None:
                # Require a fresh explicit ticket for intentional retry.
                raise ValueError("This Run click was consumed; click Run again to retry.")
            # Reuse only the small summary, never a hidden live model object.
            return previous
        # Consume the ticket before model loading, writes, or any possible interruption.
        self._requests[ticket] = None
        # Execute the caller's shared run entry point exactly once for this ticket.
        result = action()
        # Cache only a successful summary for safe reactive redisplay.
        self._requests[ticket] = result
        # Return the shared runner result to the thin notebook.
        return result


def summarize_run(directory: Path, *, new_attempts: int = 0, stop_reason: str = "inspection") -> RunSummary:
    """Inspect a verified checkpoint without loading a model, running trials, or uploading."""
    # Verify published artifacts and derive successes from actual records.
    snapshot = read_run(directory)
    # Default to no remote acknowledgement; a local write is not an upload.
    acknowledged = 0
    # Read only an explicit acknowledgement receipt, never query storage during inspection.
    receipt_path = directory / "sync-receipt.json"
    # Preserve a previous remote boundary if the local receipt is available.
    if receipt_path.exists():
        # Refuse linked receipts rather than following arbitrary paths from a result viewer.
        if receipt_path.is_symlink():
            # Keep read-only provenance inspection inside the explicit run directory.
            raise ArtifactError("A sync receipt must not be a symlink.")
        # Validate all receipt fields without capturing any authentication information.
        receipt = SyncReceipt.model_validate_json(receipt_path.read_bytes())
        # Check acknowledgement identity and subset membership against actual local successes.
        if receipt.run_id != snapshot.metadata.run_id or not set(receipt.completed_trial_ids).issubset(snapshot.manifest.completed_trial_ids) or len(set(receipt.completed_trial_ids)) != len(receipt.completed_trial_ids):
            # Do not label mismatched or fictional receipt entries as durable work.
            raise ArtifactError("The remote receipt does not match this checkpoint.")
        # Report exactly the last acknowledged boundary, not later unsynced local progress.
        acknowledged = len(receipt.completed_trial_ids)
    # Return compact CPU-readable bookkeeping from verified observations.
    return RunSummary(run_id=snapshot.metadata.run_id, run_dir=str(directory), planned_trials=len(snapshot.metadata.examples) * snapshot.metadata.spec.replicates, completed_trials=len(snapshot.manifest.completed_trial_ids), remotely_acknowledged_trials=acknowledged, failed_attempts=sum(record.status == "failed" for record in snapshot.records), new_attempts=new_attempts, stop_reason=stop_reason)


def run_trials(
    metadata: RunMetadata,
    trial_function: TrialFunction,
    *,
    execution: ExecutionSpec | None = None,
    should_cancel: Callable[[], bool] | None = None,
    started_monotonic: float | None = None,
) -> RunSummary:
    """Low-level scheduler for the source-bound recipe and deterministic storage tests.

    Scientific callers should use ``run_experiment`` or ``resume_experiment``.
    This injection seam cannot fingerprint an arbitrary callback or its closure;
    custom project recipes are unsupported until their source enters run identity.

    Args:
        metadata: Immutable science, dataset, source, model, and numerical identity.
        trial_function: Internal ordinary function taking example, replicate, isolated
            seed, and a cooperative stop predicate. It implements the source-bound
            packaged recipe, or explicit artificial outputs in persistence tests.
        execution: Optional new invocation budgets/output location, not new science.
        should_cancel: Optional caller-owned cancellation predicate.
        started_monotonic: Earlier start time when the budget includes model loading.

    Returns:
        A verified local completion summary; this function never uploads.

    Raises:
        TrialExecutionError: A model/recipe failure was persisted and execution stopped.
        KeyboardInterrupt: An interrupted attempt was persisted before re-raising.
        ArtifactError: Corrupt/incompatible artifacts or invalid trial outputs.
        filelock.Timeout: Another process already owns the run writer.
    """
    # Use explicit invocation settings or the original operational configuration.
    settings = execution or metadata.spec.execution
    # Include earlier preparation/model-loading time when the shared entry point supplies it.
    started = time.monotonic() if started_monotonic is None else started_monotonic
    # Reserve the configured finalization/upload margin before scheduling new model work.
    deadline = started + settings.max_seconds - settings.shutdown_margin_seconds if settings.max_seconds is not None else None
    # Track fresh attempts independently of already completed logical observations.
    new_attempts = 0
    # Assume completion only if no budget/cancellation prevents the full missing schedule.
    stop_reason = "complete"
    # Latch the first observed stop so a transient caller flag cannot accidentally restart work.
    requested_reason: str | None = None

    def stop_requested() -> bool:
        """Check the caller's cancel flag and cooperative invocation deadline."""
        # Preserve the actual first stop cause without querying a stateful predicate twice.
        nonlocal requested_reason
        # Once cancellation/deadline is observed, keep this invocation stopped.
        if requested_reason is not None:
            # Do not launch another forward merely because a transient flag later cleared.
            return True
        # Give an explicitly observed user cancellation a distinct safe category.
        if should_cancel is not None and should_cancel():
            # Record the actual observation before returning to the model recipe.
            requested_reason = "cancelled"
        # Only report time-budget expiry if an actual configured deadline elapsed.
        elif deadline is not None and time.monotonic() >= deadline:
            # Preserve the first reached deadline for final summary bookkeeping.
            requested_reason = "time_budget"
        # Return a latched boolean without signals, threads, writes, or notebook mutations.
        return requested_reason is not None

    # Create a side-effect-free writer object before entering its exclusive context.
    store = RunStore(settings.output_dir, shard_records=settings.shard_records, max_shard_bytes=settings.max_shard_bytes)
    # Verify/init the resolved run under a process-safe single-writer boundary.
    with store.writer(metadata):
        # Begin from verified published successes only; failed trials remain eligible to retry.
        completed = set(store.snapshot.manifest.completed_trial_ids)
        # Preserve stable dataset order, independent of prior interrupted attempts.
        for example in metadata.examples:
            # Give every declared repeat an independent logical trial identity.
            for replicate in range(metadata.spec.replicates):
                # Compute the same logical identity on every machine and resume invocation.
                logical_id = trial_id(metadata.run_id, example.id, replicate)
                # Never reexecute a published success just because this invocation has a new budget.
                if logical_id in completed:
                    # Continue to the next missing scheduled observation.
                    continue
                # Respect cancellation/time budgets before starting another forward pass.
                if stop_requested():
                    # Reuse the already observed cause rather than querying cancellation again.
                    stop_reason = requested_reason or "cancelled"
                    # Stop this replicate loop without manufacturing a completed attempt.
                    break
                # Limit fresh execution attempts without changing the experiment's total schedule.
                if settings.max_new_trials is not None and new_attempts >= settings.max_new_trials:
                    # Report a partial invocation instead of a completed experiment.
                    stop_reason = "trial_limit"
                    # Stop this replicate loop and finalize already complete work.
                    break
                # Derive paired randomness independently of execution order or retries.
                seed = trial_seed(metadata.spec.seed, example.pair_id, replicate)
                # Assign a fresh attempt UUID, distinct from the stable logical observation.
                attempt = uuid4()
                # Record the start time before calling the ordinary model/recipe function.
                started_at = utc_now()
                # Default to complete only after a valid trial function actually returns.
                status = "completed"
                # Preserve a result only if execution returns an auditable observation.
                result = None
                # Record safe exception categories rather than possibly secret-bearing messages.
                error_type = None
                # Keep model/recipe failures distinct from subsequent artifact publication failures.
                try:
                    # Execute the shared ordinary recipe with an explicit cooperative stop predicate.
                    result = trial_function(example, replicate, seed, stop_requested)
                    # An early generation stop is incomplete even if candidate scores are available.
                    if result.generation is not None and result.generation.finish_reason == "cancelled":
                        # Ensure resume reruns this logical observation from its isolated seed.
                        status = "cancelled"
                # Preserve interrupted execution at the trial boundary before propagating the interrupt.
                except KeyboardInterrupt:
                    # Save an explicit incomplete attempt, not fictitious partially recovered model state.
                    store.append(_attempt_record(metadata, example, replicate, seed, logical_id, attempt, started_at, "cancelled", None, "KeyboardInterrupt"))
                    # Let the writer's finalization scope seal the attempt before caller interruption handling.
                    raise
                # Treat cooperative early cancellation as incomplete, not a model failure.
                except CancellationRequested:
                    # Keep the missing logical trial eligible for later retry.
                    status = "cancelled"
                    # Preserve a safe execution category without logging a raw exception string.
                    error_type = "CancellationRequested"
                # Stop after a real execution error rather than silently selecting surviving trials.
                except Exception as error:
                    # Persist the attempt before surfacing failure to the user.
                    store.append(_attempt_record(metadata, example, replicate, seed, logical_id, attempt, started_at, "failed", None, f"{type(error).__module__}.{type(error).__name__}"))
                    # Do not keep using a possibly compromised model after OOM/other execution failures.
                    raise TrialExecutionError("Trial failed; its safe error category is saved in the run.") from error
                # Validate and append complete or cooperatively interrupted returned outputs.
                store.append(_attempt_record(metadata, example, replicate, seed, logical_id, attempt, started_at, status, result, error_type))
                # Count actual attempted execution, not skipped existing successes.
                new_attempts += 1
                # Only validated full outputs satisfy a logical observation.
                if status == "completed":
                    # Include buffered successes in this invocation's duplicate-prevention boundary.
                    completed.add(logical_id)
                # Stop the invocation after an incomplete trial rather than repeatedly attempting it.
                else:
                    # Internal recipe cancellation without an expired deadline is not a time limit.
                    stop_reason = requested_reason or "cancelled"
                    # Leave this observation for the next explicit resume invocation.
                    break
            # Propagate a partial invocation boundary out of the outer dataset loop.
            if stop_reason != "complete":
                # Finalize already completed attempts through the writer context's exit.
                break
    # Derive final counts after buffered attempts have been published to the manifest.
    return summarize_run(settings.output_dir, new_attempts=new_attempts, stop_reason=stop_reason)


def _attempt_record(metadata: RunMetadata, example: Example, replicate: int, seed: int, logical_id: str, attempt: Any, started_at: Any, status: str, result: TrialResult | None, error_type: str | None) -> TrialRecord:
    """Attach immutable dataset lineage and runner-owned bookkeeping to one attempt."""
    # Validate bookkeeping and result status before the record enters the artifact store.
    return TrialRecord(run_id=metadata.run_id, trial_id=logical_id, attempt_id=attempt, example_id=example.id, pair_id=example.pair_id, group_id=example.group_id, source_id=example.source_id, condition=example.condition, split=example.split, replicate=replicate, seed=seed, status=status, started_at=started_at, finished_at=utc_now(), input_text=example.text, result=result, error_type=error_type)


def run_experiment(
    prepared: PreparedRun | Path,
    *,
    allow_downloads: bool = False,
    should_cancel: Callable[[], bool] | None = None,
) -> RunSummary:
    """Shared explicit CLI/notebook entry point: prepare, load once, and run/resume.

    The offline default performs no downloads. Real HF loading remains cached-only
    unless ``allow_downloads`` is explicitly enabled. Uploads are always separate.
    """
    # Start the cooperative wall-time budget before data preparation or model loading.
    started = time.monotonic()
    # Reuse an already inspected preparation result or validate the requested config now.
    inputs = prepare_run(prepared) if isinstance(prepared, Path) else prepared
    # Refuse changed package code after a notebook froze its inputs.
    if source_digest() != inputs.source_sha256:
        # Require a fresh explicit preparation instead of falsely labeling changed calculations.
        raise ArtifactError("Prepared package source changed; freeze the settings again.")
    # Import the explicit execution layer only inside the run action.
    from .models import behavioral_trial, load_model

    # Load exactly one declared model/tokenizer for this explicit invocation.
    bundle = load_model(inputs.spec.model, allow_downloads=allow_downloads)
    # Freeze actual library/hardware/config/tokenizer provenance before scheduling observations.
    metadata = RunMetadata(run_id=inputs.run_id, created_at=utc_now(), spec=inputs.spec, dataset_sha256=inputs.dataset_sha256, source_sha256=inputs.source_sha256, runtime_profile=runtime_profile(bundle.device), model_profile=bundle.profile, examples=inputs.examples)

    def trial(example: Example, replicate: int, seed: int, stop: Callable[[], bool]) -> TrialResult:
        """Adapt the reference behavioral recipe to the ordinary runner function seam."""
        # Let the runner own replicate identity while the recipe consumes its isolated seed.
        return behavioral_trial(bundle, example, inputs.spec, seed, stop)

    # Invoke the same checkpointed runner used by future small project recipes.
    return run_trials(metadata, trial, should_cancel=should_cancel, started_monotonic=started)


def resume_experiment(
    directory: Path,
    *,
    execution: ExecutionSpec | None = None,
    allow_downloads: bool = False,
    should_cancel: Callable[[], bool] | None = None,
) -> RunSummary:
    """Resume from saved examples/provenance without requiring the original dataset file.

    Current package source, loaded model/tokenizer, and numerical profile must match
    the saved run. An artifact can still be inspected/restored after a code change,
    but silently mixing calculations from that changed source is not allowed.
    """
    # Include restore inspection and explicit model loading in the invocation's budget.
    started = time.monotonic()
    # Read only checksum-verified published science and observations.
    saved = read_run(directory)
    # Reject source changes before loading any expensive model.
    if source_digest() != saved.metadata.source_sha256:
        # Require a deliberate new experiment identity for changed scientific calculations.
        raise ArtifactError("Resume refused: executing package source changed.")
    # Import optional execution code only when the caller explicitly requests resume.
    from .models import behavioral_trial, load_model

    # Recover the pinned model/tokenizer identity from the archived specification.
    bundle = load_model(saved.metadata.spec.model, allow_downloads=allow_downloads)
    # Compare actual model/config and numerical stack rather than trusting the old machine name.
    if bundle.profile != saved.metadata.model_profile or runtime_profile(bundle.device) != saved.metadata.runtime_profile:
        # Prevent an apparent replication from silently mixing numerical profiles.
        raise ArtifactError("Resume refused: loaded model or numerical profile changed.")
    # Reuse original settings unless the caller declares new operational invocation budgets.
    raw = (execution or saved.metadata.spec.execution).model_dump(mode="python")
    # Always write into the explicitly restored/current directory, not an old saved machine path.
    raw["output_dir"] = directory
    # Revalidate the operational override before any new attempt is started.
    settings = ExecutionSpec.model_validate(raw)

    def trial(example: Example, replicate: int, seed: int, stop: Callable[[], bool]) -> TrialResult:
        """Use the archived dataset snapshot and the same declared behavioral recipe."""
        # Do not reread an original dataset path that may not exist on this machine.
        return behavioral_trial(bundle, example, saved.metadata.spec, seed, stop)

    # Schedule only missing logical successes against the immutable restored metadata.
    return run_trials(saved.metadata, trial, execution=settings, should_cancel=should_cancel, started_monotonic=started)
