"""Explicit remote-GPU run/resume entry points with per-sample durable boundaries.

Use one process per checkpoint/partition, with separate output directories. This
module is also callable from marimo; imports and inspection never start model work.
SIGTERM handling belongs to the CLI's scoped signal guard, not a persistent global
handler. An interrupted sample reruns from its own seed; KV state is not serialized.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from filelock import FileLock

from ..artifacts import ArtifactError
from ..config import source_digest
from ..runner import TrialExecutionError
from ..runtime import CancellationRequested, runtime_profile, utc_now
from .plan import checkpoint_for, job_identity, read_plan, schedule, validate_plan
from .schemas import Attempt, Job, Rollout, StudyMetadata, StudyPlan, Trial
from .storage import StudyStore, read_snapshot


def inspect_job(directory: Path) -> dict[str, Any]:
    """Inspect completion only, with no model imports, writes, or remote operations."""
    saved = read_snapshot(directory)
    planned = sum(1 for _ in schedule(saved.metadata.plan, saved.metadata.job))
    return {
        "run_id": saved.metadata.run_id, "study_id": saved.metadata.plan.study_id,
        "job": saved.metadata.job.model_dump(mode="json"), "planned_trials": planned,
        "completed_trials": len(saved.manifest.completed_trial_ids),
        "failed_attempts": sum(row.status == "failed" for row in saved.records),
        "cancelled_attempts": sum(row.status == "cancelled" for row in saved.records),
        "missing_trials": planned - len(saved.manifest.completed_trial_ids),
        "run_dir": str(directory), "remote_durability": "not_managed",
    }


def run_trials(
    metadata: StudyMetadata, directory: Path,
    execute: Callable[[Trial, Callable[[], bool]], Rollout], *,
    max_new_trials: int | None = None, should_stop: Callable[[], bool] | None = None,
    flush_records: int = 1,
    _writer_lock: FileLock | None = None,
) -> dict[str, Any]:
    """Schedule an internal source-bound recipe, persisting failures rather than skipping.

    The injection seam is for this packaged recipe and artificial offline tests,
    not an untracked custom callback. Only published full successes are skipped.
    An ordinary exception saves its safe category and stops the process; an OOM
    does not silently retry with changed precision/batching or drop a hard problem.
    """
    if max_new_trials is not None and (type(max_new_trials) is not int or max_new_trials < 0):
        raise ValueError("max_new_trials must be a nonnegative integer.")
    store = StudyStore(directory, flush_records=flush_records)
    count = 0
    reason = "complete"
    with store.writer(metadata, held_lock=_writer_lock):
        for trial in schedule(metadata.plan, metadata.job):
            if trial.trial_id in store.completed:
                continue
            if max_new_trials is not None and count >= max_new_trials:
                reason = "trial_limit"
                break
            if should_stop is not None and should_stop():
                reason = "cancelled_or_time_budget"
                break
            started_at = utc_now()
            attempt_id = uuid4()
            result = None
            attempt_status = "completed"
            error_type = None
            caught: BaseException | None = None
            try:
                result = execute(trial, should_stop or (lambda: False))
                if result.generation.finish_reason == "cancelled":
                    attempt_status = "cancelled"
            except (KeyboardInterrupt, CancellationRequested) as error:
                attempt_status, error_type = "cancelled", type(error).__name__
                caught = error
            except Exception as error:
                attempt_status = "failed"
                error_type = f"{type(error).__module__}.{type(error).__name__}"
                caught = error
            # Artifact failures are outside the model exception handler and remain visible.
            store.append(Attempt(
                run_id=metadata.run_id, trial=trial, attempt_id=attempt_id,
                started_at=started_at, finished_at=utc_now(), status=attempt_status,
                result=result, error_type=error_type,
            ))
            count += 1
            if isinstance(caught, KeyboardInterrupt):
                raise caught
            if attempt_status == "failed":
                raise TrialExecutionError("Study trial failed; its safe category was saved.") from caught
            if attempt_status == "cancelled":
                reason = "cancelled_or_time_budget"
                break
    return {**inspect_job(directory), "new_attempts": count, "stop_reason": reason}


def run_study(
    plan: StudyPlan | Path, job: Job, directory: Path, *, device: str = "cuda:0",
    allow_downloads: bool = False, max_new_trials: int | None = None,
    max_seconds: float | None = None, shutdown_margin_seconds: float = 30.0,
    flush_records: int = 1, should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Load once, validate all prompts, then resume one independently writable GPU job.

    Wall-time includes preparation/loading; cancellation is cooperative between
    decode steps. A synchronous load/forward may exceed it. Complete runs return
    without loading the model. Resume requires identical code and actual numerical
    profiles; moving an incomplete job to different hardware requires a new run.
    """
    started = time.monotonic()
    if not math.isfinite(shutdown_margin_seconds) or shutdown_margin_seconds < 0 or (
        max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= shutdown_margin_seconds)
    ):
        raise ValueError("Time limit must exceed the nonnegative shutdown margin.")
    if max_new_trials is not None and (type(max_new_trials) is not int or max_new_trials < 0):
        raise ValueError("max_new_trials must be nonnegative.")
    if type(flush_records) is not int or not 1 <= flush_records <= 100:
        raise ValueError("flush_records must be between 1 and 100.")
    plan = read_plan(plan, check_source=True) if isinstance(plan, Path) else plan
    validate_plan(plan)
    if source_digest() != plan.source_sha256:
        raise ArtifactError("Study source changed; restore original code or lock a new plan.")
    checkpoint = checkpoint_for(plan, job)
    planned = sum(1 for _ in schedule(plan, job))
    if not planned:
        raise ValueError("This shard has no problem groups; choose fewer partitions.")
    run_id = job_identity(plan, job)
    if (directory / "manifest.json").exists():
        previous = read_snapshot(directory)
        if previous.metadata.run_id != run_id:
            raise ArtifactError("Output directory belongs to a different job/study.")
        if len(previous.manifest.completed_trial_ids) == planned:
            return {**inspect_job(directory), "new_attempts": 0, "stop_reason": "complete"}
    if max_new_trials == 0:
        return {"run_id": run_id, "planned_trials": planned, "new_attempts": 0,
                "stop_reason": "trial_limit", "model_loaded": False}
    # Latch cancellation so a transient callback cannot restart a timed-out invocation.
    stopped = False

    def stop() -> bool:
        """Stop future decode steps after cancellation or the cooperative deadline."""
        nonlocal stopped
        stopped = stopped or bool(should_cancel and should_cancel()) or (
            max_seconds is not None
            and time.monotonic() - started >= max_seconds - shutdown_margin_seconds
        )
        return stopped

    if stop():
        return {"run_id": run_id, "planned_trials": planned, "new_attempts": 0,
                "stop_reason": "cancelled_or_time_budget", "model_loaded": False}
    from .models import execute_trial, load_checkpoint, preflight_inputs

    if directory.is_symlink():
        raise ArtifactError("Study run directory must not be a symlink.")
    directory.mkdir(parents=True, exist_ok=True)
    # Reject duplicate invocations before they allocate another multi-gigabyte model.
    with FileLock(str(directory / ".writer.lock"), timeout=0) as writer_lock:
        bundle = load_checkpoint(checkpoint, plan.spec.inference, device_name=device, allow_downloads=allow_downloads)
        try:
            preflight_inputs(bundle, checkpoint, plan)
            profile = runtime_profile(bundle.device)
            # Local device ordinals may differ under CUDA_VISIBLE_DEVICES; hardware does not.
            profile["device"] = bundle.device.type
            metadata = StudyMetadata(
                run_id=run_id, created_at=utc_now(), plan=plan, job=job,
                model_profile=bundle.profile, runtime_profile=profile,
            )
            problems = {row.id: row for row in plan.problems}
            prompts = {row.name: row for row in plan.spec.prompts}

            def execute(trial: Trial, cancellation: Callable[[], bool]) -> Rollout:
                """Bind the packaged numerical recipe to its immutable declared inputs."""
                return execute_trial(
                    bundle, checkpoint, problems[trial.problem_id], prompts[trial.prompt],
                    plan, trial, cancellation,
                )

            return run_trials(
                metadata, directory, execute, max_new_trials=max_new_trials,
                should_stop=stop, flush_records=flush_records, _writer_lock=writer_lock,
            )
        finally:
            # Clear the closure's model reference without touching other processes or GPUs.
            bundle = None


def resume_study(directory: Path, **options: Any) -> dict[str, Any]:
    """Resume using only archived questions and settings, independent of original files."""
    metadata = read_snapshot(directory).metadata
    return run_study(metadata.plan, metadata.job, directory, **options)