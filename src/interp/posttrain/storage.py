"""Single-writer study checkpoints reusing the workbench's atomic publication layer.

Only checksummed, manifest-referenced JSONL attempts count. Each GPU partition has
its own directory/lock. Sealed orphan files never advance completion; a hard kill
can lose only the bounded unpublished buffer. This format is separate from the
older two-candidate schema and is not compatible with its bounded Hub sync command.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock

from ..artifacts import ArtifactError, atomic_write
from ..data import canonical_bytes, digest_bytes
from ..runtime import utc_now
from ..schemas import CheckpointManifest, ShardRef
from .plan import job_identity, schedule, validate_plan
from .schemas import Attempt, StudyMetadata, Trial
from .scoring import grade_answer


@dataclass(frozen=True)
class Snapshot:
    """One fully verified publication boundary, suitable for CPU-only inspection."""

    metadata: StudyMetadata
    manifest: CheckpointManifest
    records: tuple[Attempt, ...]


def _bytes(path: Path, maximum: int) -> bytes:
    """Bound reads and reject symlink/special-file artifact references."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ArtifactError(f"Missing, linked, or oversized study artifact: {path.name}")
    return path.read_bytes()


def validate_attempt(attempt: Attempt, metadata: StudyMetadata, trials: dict[str, Trial]) -> None:
    """Recheck schedule, seed, budget, answer grade, and complete-output semantics."""
    if attempt.run_id != metadata.run_id or trials.get(attempt.trial.trial_id) != attempt.trial:
        raise ArtifactError("Attempt identity/seed does not match this job's schedule.")
    if attempt.result is None:
        return
    result = attempt.result
    generated = result.generation
    settings = metadata.plan.spec.generation
    effective = generated.effective_config
    # Reject hidden decode settings, including accidental top-k=50 library defaults.
    expected = {
        "max_new_tokens": attempt.trial.budget, "do_sample": settings.do_sample,
        "use_cache": settings.use_cache, "num_beams": 1, "num_return_sequences": 1,
        "repetition_penalty": 1.0,
    }
    if settings.do_sample:
        expected.update(temperature=settings.temperature, top_p=settings.top_p, top_k=settings.top_k)
    if any(effective.get(key) != value for key, value in expected.items()):
        raise ArtifactError("Effective decoding disagrees with the locked study.")
    size = len(generated.token_ids)
    if size > attempt.trial.budget or (generated.finish_reason == "length" and size != attempt.trial.budget):
        raise ArtifactError("Generated token count disagrees with its budget/finish reason.")
    eos = effective.get("eos_token_id")
    eos_ids = eos if isinstance(eos, list) else [eos]
    if generated.finish_reason == "eos" and (not size or generated.token_ids[-1] not in eos_ids):
        raise ArtifactError("EOS finish requires an actual declared EOS token.")
    if len(result.input_ids) + attempt.trial.budget > metadata.plan.spec.inference.max_context_tokens:
        raise ArtifactError("Saved input exceeds the locked context budget.")
    if not math.isfinite(result.elapsed_seconds):
        raise ArtifactError("Elapsed time must be finite.")
    # Answer correctness is reproducible without models or arbitrary saved labels.
    problem = next(row for row in metadata.plan.problems if row.id == attempt.trial.problem_id)
    if result.initial_thinking != result.rendered_prompt.rstrip().endswith("<think>"):
        raise ArtifactError("Saved initial thinking state disagrees with the rendered prompt.")
    if result.grade != grade_answer(generated.text, problem, initial_thinking=result.initial_thinking):
        raise ArtifactError("Saved numerical grade disagrees with the raw generated text.")


def read_snapshot(directory: Path) -> Snapshot:
    """Verify all published artifacts; ignore directory contents absent from the manifest.

    No code-equality check is applied here so historical results remain readable.
    Execution performs the separate source/model/numerical compatibility checks.
    """
    manifest = CheckpointManifest.model_validate_json(_bytes(directory / "manifest.json", 64 * 1024**2))
    content = _bytes(directory / "metadata.json", 64 * 1024**2)
    if digest_bytes(content) != manifest.metadata_sha256:
        raise ArtifactError("Study metadata failed checksum verification.")
    metadata = StudyMetadata.model_validate_json(content)
    validate_plan(metadata.plan)
    if manifest.run_id != metadata.run_id or job_identity(metadata.plan, metadata.job) != metadata.run_id:
        raise ArtifactError("Study job identity is inconsistent.")
    trials = {trial.trial_id: trial for trial in schedule(metadata.plan, metadata.job)}
    records: list[Attempt] = []
    attempts: set[str] = set()
    completed: set[str] = set()
    for index, ref in enumerate(manifest.shards):
        if ref.name != f"records-{index:06d}-{ref.sha256}.jsonl":
            raise ArtifactError("Study shards have inconsistent indices or names.")
        payload = _bytes(directory / ref.name, 16 * 1024**2)
        if len(payload) != ref.size_bytes or digest_bytes(payload) != ref.sha256:
            raise ArtifactError("Study shard failed checksum verification.")
        lines = payload.splitlines()
        if not payload.endswith(b"\n") or len(lines) != ref.record_count or any(not line for line in lines):
            raise ArtifactError("Study shard has incomplete JSONL/counts.")
        for line in lines:
            attempt = Attempt.model_validate_json(line)
            validate_attempt(attempt, metadata, trials)
            if str(attempt.attempt_id) in attempts:
                raise ArtifactError("Duplicate study attempt ID.")
            attempts.add(str(attempt.attempt_id))
            if attempt.status == "completed":
                if attempt.trial.trial_id in completed:
                    raise ArtifactError("Duplicate successful study trial.")
                completed.add(attempt.trial.trial_id)
            records.append(attempt)
    if len(set(manifest.completed_trial_ids)) != len(manifest.completed_trial_ids) or set(manifest.completed_trial_ids) != completed:
        raise ArtifactError("Study manifest completion disagrees with verified records.")
    return Snapshot(metadata, manifest, tuple(records))


class StudyStore:
    """A small project-specific writer; atomic bytes/manifest types are shared code.

    The writer holds the OS lock for its entire context. Default publication after
    each sample prioritizes durability over throughput. Increase ``flush_records``
    only if losing that many unpublished samples on hard kill is acceptable.
    """

    def __init__(self, directory: Path, *, flush_records: int = 1) -> None:
        """Remember explicit destination and a bounded publication interval."""
        if type(flush_records) is not int or not 1 <= flush_records <= 100:
            raise ValueError("flush_records must be an integer from 1 to 100.")
        self.directory = directory
        self.flush_records = flush_records
        self.snapshot: Snapshot | None = None
        self.buffer: list[Attempt] = []
        self._active = False
        self._trials: dict[str, Trial] = {}
        self._attempts: set[str] = set()
        self.completed: set[str] = set()

    @contextmanager
    def writer(self, metadata: StudyMetadata, *, held_lock: FileLock | None = None) -> Iterator[StudyStore]:
        """Initialize/resume exclusively and flush complete attempts on ordinary exit."""
        if self._active or self.directory.is_symlink():
            raise ArtifactError("Study writer is active or its directory is linked.")
        validate_plan(metadata.plan)
        if job_identity(metadata.plan, metadata.job) != metadata.run_id:
            raise ArtifactError("Invalid study job identity.")
        self.directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.directory / ".writer.lock"
        # The GPU entry point holds this same lock before allocating weights. Tests
        # and low-level callers let this store own its ordinary exclusive context.
        if held_lock is not None and (
            not held_lock.is_locked or Path(held_lock.lock_file).resolve() != lock_path.resolve()
        ):
            raise ArtifactError("Externally held writer lock does not own this directory.")
        lock_context = nullcontext() if held_lock is not None else FileLock(str(lock_path), timeout=0)
        with lock_context:
            path = self.directory / "metadata.json"
            if path.exists():
                existing = StudyMetadata.model_validate_json(_bytes(path, 64 * 1024**2))
                if (
                    existing.run_id != metadata.run_id
                    or existing.model_profile != metadata.model_profile
                    or existing.runtime_profile != metadata.runtime_profile
                ):
                    raise ArtifactError("Resume refused: study/model/numerical profile changed.")
            else:
                if (self.directory / "manifest.json").exists() or any(self.directory.glob("records-*.jsonl")):
                    raise ArtifactError("Study artifacts exist without their metadata.")
                content = canonical_bytes(metadata.model_dump(mode="json"))
                if len(content) > 64 * 1024**2:
                    raise ArtifactError("Study metadata exceeds 64 MiB.")
                atomic_write(path, content, immutable=True)
            if not (self.directory / "manifest.json").exists():
                manifest = CheckpointManifest(
                    run_id=metadata.run_id, metadata_sha256=digest_bytes(_bytes(path, 64 * 1024**2)),
                    updated_at=utc_now(),
                )
                atomic_write(self.directory / "manifest.json", canonical_bytes(manifest.model_dump(mode="json")))
            self.snapshot = read_snapshot(self.directory)
            self._trials = {row.trial_id: row for row in schedule(metadata.plan, metadata.job)}
            self._attempts = {str(row.attempt_id) for row in self.snapshot.records}
            self.completed = set(self.snapshot.manifest.completed_trial_ids)
            self._active = True
            try:
                yield self
            finally:
                try:
                    self.flush()
                finally:
                    self._active = False

    def append(self, attempt: Attempt) -> None:
        """Validate a complete attempt before it enters the bounded unpublished buffer."""
        if not self._active or self.snapshot is None:
            raise ArtifactError("Study append requires an active writer.")
        validate_attempt(attempt, self.snapshot.metadata, self._trials)
        if str(attempt.attempt_id) in self._attempts or (
            attempt.status == "completed" and attempt.trial.trial_id in self.completed
        ):
            raise ArtifactError("Duplicate attempt or logical success.")
        line_size = len(canonical_bytes(attempt.model_dump(mode="json"))) + 1
        if line_size > 16 * 1024**2:
            raise ArtifactError("Study attempt exceeds 16 MiB.")
        previous_size = sum(len(canonical_bytes(row.model_dump(mode="json"))) + 1 for row in self.buffer)
        if self.buffer and previous_size + line_size > 16 * 1024**2:
            self.flush()
        self.buffer.append(attempt)
        self._attempts.add(str(attempt.attempt_id))
        if attempt.status == "completed":
            self.completed.add(attempt.trial.trial_id)
        if len(self.buffer) >= self.flush_records:
            self.flush()

    def flush(self) -> None:
        """Seal content first, atomically acknowledge it second; never scan orphan files."""
        if not self.buffer:
            return
        if not self._active or self.snapshot is None:
            raise ArtifactError("Study flush requires an active writer.")
        payload = b"".join(canonical_bytes(row.model_dump(mode="json")) + b"\n" for row in self.buffer)
        sha = digest_bytes(payload)
        name = f"records-{len(self.snapshot.manifest.shards):06d}-{sha}.jsonl"
        atomic_write(self.directory / name, payload, immutable=True)
        reference = ShardRef(name=name, sha256=sha, size_bytes=len(payload), record_count=len(self.buffer))
        manifest = CheckpointManifest(
            run_id=self.snapshot.metadata.run_id,
            metadata_sha256=self.snapshot.manifest.metadata_sha256,
            shards=(*self.snapshot.manifest.shards, reference),
            completed_trial_ids=tuple(sorted(self.completed)), updated_at=utc_now(),
        )
        atomic_write(self.directory / "manifest.json", canonical_bytes(manifest.model_dump(mode="json")))
        self.snapshot = Snapshot(self.snapshot.metadata, manifest, (*self.snapshot.records, *self.buffer))
        self.buffer.clear()