# Save sealed result shards here; inspection functions are strictly read-only.
# A file becomes completed work only when an atomically published manifest names it.
"""Single-writer local checkpoints with checksums and conservative resume validation.

The OS-backed file lock is released on process death. Temporary files and sealed
orphans are ignored by readers, never promoted into completion by directory scanning.
This first release retains small runs in memory and is not a large activation store.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock

from .config import compute_run_id
from .data import canonical_bytes, dataset_digest, digest_bytes, validate_examples
from .runtime import trial_id, trial_seed, utc_now
from .schemas import CheckpointManifest, RunMetadata, ShardRef, TrialRecord


class ArtifactError(ValueError):
    """A checkpoint is corrupt, incompatible, incomplete, or violates record invariants."""


@dataclass(frozen=True)
class RunSnapshot:
    """A verified immutable view of one published local checkpoint."""

    # Preserve the validated resolved scientific inputs and execution profile.
    metadata: RunMetadata
    # Preserve the exact publication boundary that the reader verified.
    manifest: CheckpointManifest
    # Include failed/interrupted attempts as well as successful observations.
    records: tuple[TrialRecord, ...]


def sync_directory(path: Path) -> None:
    """Flush directory entries after an atomic publication on macOS/Linux filesystems."""
    # Open the directory itself rather than an artifact file.
    descriptor = os.open(path, os.O_RDONLY)
    # Guarantee descriptor cleanup even if the filesystem reports a durability error.
    try:
        # Request persistence of rename/link directory entries.
        os.fsync(descriptor)
    # Leave no open file descriptors after this bounded persistence action.
    finally:
        # Release the directory handle.
        os.close(descriptor)


def atomic_write(path: Path, content: bytes, *, immutable: bool = False) -> None:
    """Write and fsync a same-directory temporary file, then publish it atomically.

    Args:
        path: Destination under an already-created run/staging directory.
        content: Final bytes, not a mutable stream being produced concurrently.
        immutable: Link without overwrite; an existing byte-identical sealed file
            is safe to reuse after an interrupted manifest publication.

    Raises:
        ArtifactError: An immutable name already contains different bytes/a symlink.
        OSError: The write or filesystem durability operation fails. It is not hidden.
    """
    # Reject symlink destinations so publication cannot escape the declared directory.
    if path.is_symlink():
        # Do not follow user-created links for result writes.
        raise ArtifactError("Artifact destinations must not be symlinks.")
    # Create the temporary file on the same filesystem as its final destination.
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    # Wrap the actual temporary location for reliable cleanup.
    temporary = Path(temporary_name)
    # Ensure temporary files never count as sealed records, even after a failure.
    try:
        # Transfer ownership of the descriptor to a bounded binary file context.
        with os.fdopen(descriptor, "wb") as handle:
            # Write the complete final artifact bytes.
            handle.write(content)
            # Flush buffered userspace writes before requesting durable storage.
            handle.flush()
            # Request persistence of the complete file content.
            os.fsync(handle.fileno())
        # Never overwrite a sealed content-addressed artifact.
        if immutable:
            # Publish with an atomic no-overwrite hard link on supported local filesystems.
            try:
                # An existing file causes an error rather than being replaced.
                os.link(temporary, path)
            # A retry may encounter a previously sealed orphan with the same content.
            except FileExistsError as error:
                # Reuse only a regular byte-identical file at that exact content name.
                if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                    # Reject conflicting/corrupt content rather than replacing evidence.
                    raise ArtifactError("Immutable artifact name contains different content.") from error
        # Only a checkpoint pointer/receipt is allowed to replace an earlier version.
        else:
            # Atomically switch readers from the old complete bytes to the new complete bytes.
            os.replace(temporary, path)
        # Persist the new directory entry after publishing the complete file.
        sync_directory(path.parent)
    # Clean up on success, exception, or ordinary notebook interruption.
    finally:
        # Unpublished files cannot be mistaken for finalized JSONL records.
        temporary.unlink(missing_ok=True)


def _regular_bytes(path: Path, *, max_bytes: int | None = None) -> bytes:
    """Read a declared regular artifact while rejecting symlinks and oversized payloads."""
    # Reject unavailable files and links before reading their contents.
    if path.is_symlink() or not path.is_file():
        # Report a missing/corrupt reference without echoing artifact contents.
        raise ArtifactError(f"Missing or non-regular artifact: {path.name}")
    # Bound payload allocation when a schema declares a small artifact format.
    if max_bytes is not None and path.stat().st_size > max_bytes:
        # Refuse unbounded or unexpectedly large checkpoint payloads.
        raise ArtifactError(f"Artifact exceeds the supported size: {path.name}")
    # Return exact bytes for checksum and schema verification.
    return path.read_bytes()


def validate_metadata(metadata: RunMetadata) -> None:
    """Recompute dataset/scientific identity from a saved snapshot, without a model."""
    # Apply the same pairing/leakage requirements to restored dataset snapshots.
    validate_examples(metadata.examples, metadata.spec.conditions)
    # Verify dataset content independently of any original machine-local path.
    if dataset_digest(metadata.examples) != metadata.dataset_sha256:
        # Refuse metadata whose stored data and declared identity disagree.
        raise ArtifactError("The metadata dataset digest does not match its examples.")
    # Recompute identity using the archived source digest, not the reader's current code.
    if compute_run_id(metadata.spec, metadata.dataset_sha256, metadata.source_sha256) != metadata.run_id:
        # Make historical artifacts inspectable while still rejecting altered science.
        raise ArtifactError("The metadata scientific identity is inconsistent.")


def validate_record(record: TrialRecord, metadata: RunMetadata) -> None:
    """Verify attempt identity, lineage, candidate definitions, and the declared metric."""
    # Find the original immutable observation in the archived dataset snapshot.
    examples = {example.id: example for example in metadata.examples}
    # Reject records that reference another run or a nonexistent observation.
    if record.run_id != metadata.run_id or record.example_id not in examples:
        # Prevent unrelated results from being accepted into this checkpoint.
        raise ArtifactError("A trial references a different run or unknown example.")
    # Retrieve the exact source observation used to define this logical trial.
    example = examples[record.example_id]
    # Verify the full lineage/text snapshot, not just a matching example label.
    if (record.pair_id, record.group_id, record.source_id, record.condition, record.split, record.input_text) != (
        example.pair_id, example.group_id, example.source_id, example.condition, example.split, example.text
    ):
        # Reject modified condition labels or prompts disguised as the original trial.
        raise ArtifactError("Trial lineage or prompt disagrees with the saved example.")
    # Validate logical identity and isolated randomness against the declared schedule.
    if record.replicate >= metadata.spec.replicates or record.trial_id != trial_id(metadata.run_id, example.id, record.replicate) or record.seed != trial_seed(metadata.spec.seed, example.pair_id, record.replicate):
        # Do not count an arbitrary attempt as a declared replication.
        raise ArtifactError("Trial identity, replicate, or seed disagrees with the schedule.")
    # Failed/interrupted attempts may have no model output at all.
    if record.result is None:
        # Record-level schema validation already forbids output-free success.
        return
    # Compare both result labels/texts to the exact predeclared candidate definitions.
    if tuple((score.candidate_id, score.text) for score in record.result.candidate_scores) != tuple((candidate.id, candidate.text) for candidate in example.candidates):
        # Reject ranked/reordered or edited candidate definitions in persisted results.
        raise ArtifactError("Scored candidates disagree with the saved dataset.")
    # Require unpadded actual prompt attention masks in this reference recipe.
    if any(value != 1 for value in record.result.attention_mask):
        # Do not silently reinterpret physical and logical sequence coordinates.
        raise ArtifactError("This trial recipe requires unpadded prompt masks.")
    # Index the two saved candidate likelihoods by their declared labels.
    scores = {score.candidate_id: score.sum_logprob for score in record.result.candidate_scores}
    # Locate the alternative without assuming a particular label name.
    other = next(candidate.id for candidate in example.candidates if candidate.id != example.target_candidate)
    # Verify the primary raw sequence-likelihood margin independently.
    expected = scores[example.target_candidate] - scores[other]
    # Allow only negligible reduction/serialization round-off in the saved margin.
    if not math.isclose(record.result.target_logprob_margin, expected, rel_tol=1e-6, abs_tol=1e-6):
        # Reject a metric that does not describe the actual retained token scores.
        raise ArtifactError("Saved target_logprob_margin disagrees with candidate scores.")
    # Retrieve the optional saved generation to verify its declared settings.
    generated = record.result.generation
    # Successful runs must follow the specified scoring-only or generation-enabled recipe.
    if record.status == "completed" and (generated is not None) != metadata.spec.generation.enabled:
        # Do not count a skipped requested generation as complete.
        raise ArtifactError("Trial generation presence disagrees with the specification.")
    # Validate generated length and a few critical effective settings when output exists.
    if generated is not None:
        # Retrieve the scientific generation budget rather than an invocation time budget.
        settings = metadata.spec.generation
        # Check the reference generation settings actually used by the model.
        if generated.effective_config.get("max_new_tokens") != settings.max_new_tokens or generated.effective_config.get("do_sample") != settings.do_sample or generated.effective_config.get("use_cache") != settings.use_cache:
            # Reject silently inherited decoding/cache behavior.
            raise ArtifactError("Effective generation settings disagree with the specification.")
        # Refuse a suffix longer than the complete predeclared generation budget.
        if len(generated.token_ids) > settings.max_new_tokens or (generated.finish_reason == "length" and len(generated.token_ids) != settings.max_new_tokens):
            # Preserve the exact meaning of a length-terminated generation.
            raise ArtifactError("Generation length and finish_reason are inconsistent.")


def read_run(directory: Path) -> RunSnapshot:
    """Read and verify a published run; never load a model, write, or contact the Hub.

    Checks include metadata identity, complete file hashes/sizes, JSONL record counts,
    unique attempt IDs, unique logical successes, and exact manifest completion.
    Unreferenced sealed or temporary files are ignored as unpublished work.
    """
    # Read only the explicit manifest pointer, bounded to the initial small-run format.
    manifest_bytes = _regular_bytes(directory / "manifest.json", max_bytes=16 * 1024 * 1024)
    # Validate schema and allowed reference filename syntax before filesystem access.
    manifest = CheckpointManifest.model_validate_json(manifest_bytes)
    # Read the resolved snapshot separately so its checksum can be independently checked.
    metadata_bytes = _regular_bytes(directory / "metadata.json", max_bytes=16 * 1024 * 1024)
    # Verify that the published manifest refers to exactly these metadata bytes.
    if digest_bytes(metadata_bytes) != manifest.metadata_sha256:
        # Reject a changed configuration or truncated dataset snapshot.
        raise ArtifactError("Metadata checksum does not match the manifest.")
    # Validate all fields in the restored scientific snapshot.
    metadata = RunMetadata.model_validate_json(metadata_bytes)
    # Recompute scientific and dataset identity without reading the original input file.
    validate_metadata(metadata)
    # Require consistent run labels at both publication layers.
    if manifest.run_id != metadata.run_id:
        # Refuse a manifest referring to a different resolved experiment.
        raise ArtifactError("Manifest and metadata run identities differ.")
    # Collect every published attempt, including errors and cooperative interruptions.
    records: list[TrialRecord] = []
    # Track immutable artifact names to reject duplicate references.
    names: set[str] = set()
    # Track unique execution attempts independently of logical completion.
    attempts: set[str] = set()
    # Derive actual completed logical trials from verified successful records.
    completed: set[str] = set()
    # Verify every referenced file rather than trusting directory listings.
    for index, reference in enumerate(manifest.shards):
        # Require contiguous writer indices and content hashes embedded in their names.
        if reference.name in names or reference.name != f"records-{index:06d}-{reference.sha256}.jsonl":
            # Make duplicate/reordered/shadowed shard references a clear corruption error.
            raise ArtifactError("Manifest shard names or indices are inconsistent.")
        # Record this now-unique declared artifact name.
        names.add(reference.name)
        # Read only a regular, bounded referenced shard.
        content = _regular_bytes(directory / reference.name, max_bytes=16 * 1024 * 1024)
        # Check both size and cryptographic content before parsing observations.
        if len(content) != reference.size_bytes or digest_bytes(content) != reference.sha256:
            # Reject partial writes and altered likelihood records.
            raise ArtifactError("A referenced shard failed size/checksum verification.")
        # Parse only complete lines from a newline-terminated finalized shard.
        lines = content.splitlines()
        # Verify the exact persisted attempt count and finalization convention.
        if not content.endswith(b"\n") or len(lines) != reference.record_count or any(not line for line in lines):
            # Never turn partial or formatting-corrupt writes into completed observations.
            raise ArtifactError("A referenced shard has invalid JSONL finalization/counts.")
        # Validate each attempt against the archived scientific schedule.
        for line in lines:
            # Apply schema-level token, finite-score, timestamp, and status requirements.
            record = TrialRecord.model_validate_json(line)
            # Independently check dataset/metric identities before accepting the attempt.
            validate_record(record, metadata)
            # UUID uniqueness distinguishes legitimate retries from repeated artifact ingestion.
            if str(record.attempt_id) in attempts:
                # Refuse a duplicated execution attempt.
                raise ArtifactError("A checkpoint contains duplicate attempt IDs.")
            # Mark this execution attempt as uniquely retained.
            attempts.add(str(record.attempt_id))
            # Only complete validated observations advance logical completion.
            if record.status == "completed":
                # Detect two successes for one logical trial instead of selecting one arbitrarily.
                if record.trial_id in completed:
                    # Protect paired aggregation from duplicate observations.
                    raise ArtifactError("A checkpoint contains duplicate logical successes.")
                # Mark the verified logical observation completed.
                completed.add(record.trial_id)
            # Preserve failed and interrupted attempts for inspection too.
            records.append(record)
    # Compare the independently derived successes to the exact declared completion boundary.
    if len(set(manifest.completed_trial_ids)) != len(manifest.completed_trial_ids) or set(manifest.completed_trial_ids) != completed:
        # Reject missing, duplicated, or fictitious manifest completions.
        raise ArtifactError("Manifest completion disagrees with verified successful records.")
    # Return a read-only immutable checkpoint view.
    return RunSnapshot(metadata, manifest, tuple(records))


class RunStore:
    """A bounded single-writer result buffer with explicit checkpoint publication.

    Construction has no filesystem side effects. ``writer(metadata)`` acquires the
    OS-backed lock, validates/inititializes the run, and flushes on ordinary exit,
    failure, and KeyboardInterrupt. A hard process kill can lose only unpublished
    buffered work; the last acknowledged manifest remains independently verifiable.
    """

    def __init__(self, directory: Path, *, shard_records: int = 8, max_shard_bytes: int = 4 * 1024 * 1024) -> None:
        """Remember operational limits without creating files or opening a checkpoint."""
        # Keep run-directory selection explicit and local to this writer object.
        self.directory = directory
        # Bound each immutable shard by its number of retained attempts.
        self.shard_records = shard_records
        # Bound the maximum serialized payload for this writer's small-run format.
        self.max_shard_bytes = max_shard_bytes
        # Mark the writer inactive until its exclusive context is entered.
        self._active = False
        # Keep unpublished records separate from the checkpoint's durable boundary.
        self._buffer: list[TrialRecord] = []
        # Avoid loading or writing artifacts as a construction side effect.
        self._snapshot: RunSnapshot | None = None

    @property
    def snapshot(self) -> RunSnapshot:
        """Return the last locally published checkpoint, not unpublished buffered work."""
        # Require explicit writer initialization before exposing a checkpoint.
        if self._snapshot is None:
            # Avoid pretending an uninitialized store is a completed run.
            raise ArtifactError("Enter the writer context before accessing its snapshot.")
        # Return the verified publication boundary that this writer last acknowledged.
        return self._snapshot

    @contextmanager
    def writer(self, metadata: RunMetadata) -> Iterator[RunStore]:
        """Acquire one process-safe writer lock, initialize/verify, and finalize on exit.

        Incompatible run identities or numerical/model profiles are refused. A
        concurrent writer raises filelock.Timeout immediately instead of blocking.
        """
        # Refuse reentrant use of this mutable buffer in a reactive notebook or recipe.
        if self._active:
            # Keep a single explicit execution scope for each writer object.
            raise ArtifactError("This writer is already active.")
        # Validate operational limits even for callers not using ExecutionSpec.
        if not 1 <= self.shard_records <= 1000 or not 1024 <= self.max_shard_bytes <= 16 * 1024 * 1024:
            # Reject unbounded or impossible artifact settings before writing anything.
            raise ArtifactError("Writer shard limits are outside the supported range.")
        # Reject linked run directories rather than writing into an unexpected target.
        if self.directory.is_symlink():
            # Keep artifact publication under the explicitly selected directory.
            raise ArtifactError("Run directories must not be symlinks.")
        # Create the requested local directory only after explicit writer entry.
        self.directory.mkdir(parents=True, exist_ok=True)
        # Use an established OS-backed lock; no manual stale-lock deletion is needed.
        with FileLock(str(self.directory / ".writer.lock"), timeout=0):
            # Pin this invocation's resolved scientific snapshot and provenance.
            self._initialize(metadata)
            # Mark the exclusive write boundary as active.
            self._active = True
            # Always finalize complete buffered attempts before releasing this writer.
            try:
                # Let the runner append ordinary validated trial results.
                yield self
            # Publish buffered attempts even after errors or an ordinary interrupt.
            finally:
                # Guarantee state cleanup even if final publication itself fails.
                try:
                    # Make complete buffered attempts recoverable at the manifest boundary.
                    self.flush()
                # Ensure this writer object cannot remain permanently active after an exception.
                finally:
                    # Release the local active-state guard before the OS lock exits.
                    self._active = False

    def _initialize(self, metadata: RunMetadata) -> None:
        """Create or verify a resolved snapshot and its first empty manifest under lock."""
        # Independently validate the supplied metadata identity and grouped dataset.
        validate_metadata(metadata)
        # Name the immutable experiment snapshot explicitly.
        metadata_path = self.directory / "metadata.json"
        # Reuse historical snapshot bytes rather than changing creation time or old paths.
        if metadata_path.exists():
            # Read/validate only an ordinary existing metadata file.
            existing = RunMetadata.model_validate_json(_regular_bytes(metadata_path, max_bytes=16 * 1024 * 1024))
            # Verify historical scientific identity before comparing execution compatibility.
            validate_metadata(existing)
            # Refuse changed science, loaded tokenizer/config, or numerical execution stacks.
            if existing.run_id != metadata.run_id or existing.runtime_profile != metadata.runtime_profile or existing.model_profile != metadata.model_profile:
                # Require a new run or an explicitly compatible restoration instead of mixing results.
                raise ArtifactError("Resume refused: scientific, model, or numerical profile changed.")
            # Preserve the original resolved snapshot and its immutable checksum.
            metadata = existing
        # Initialize a genuinely new run without overwriting any scientific snapshot.
        else:
            # Refuse detached result files lacking their essential provenance snapshot.
            if (self.directory / "manifest.json").exists() or any(self.directory.glob("records-*.jsonl")):
                # Prevent an ambiguous artifact folder from becoming a different run.
                raise ArtifactError("Existing result artifacts have no metadata snapshot.")
            # Serialize the complete resolved specification and dataset snapshot once.
            metadata_bytes = canonical_bytes(metadata.model_dump(mode="json"))
            # Keep the first release's metadata format explicitly bounded.
            if len(metadata_bytes) > 16 * 1024 * 1024:
                # Require a streaming dataset-snapshot extension before a much larger run.
                raise ArtifactError("Metadata exceeds the supported 16 MiB snapshot size.")
            # Publish the immutable provenance snapshot safely.
            atomic_write(metadata_path, metadata_bytes, immutable=True)
        # Recover an interruption between metadata publication and the first manifest.
        if not (self.directory / "manifest.json").exists():
            # Publish an empty boundary; no orphan file is automatically counted completed.
            manifest = CheckpointManifest(run_id=metadata.run_id, metadata_sha256=digest_bytes(metadata_path.read_bytes()), updated_at=utc_now())
            # Atomically expose the initial local checkpoint.
            atomic_write(self.directory / "manifest.json", canonical_bytes(manifest.model_dump(mode="json")))
        # Verify all referenced artifacts before this writer may schedule/resume work.
        self._snapshot = read_run(self.directory)

    def append(self, record: TrialRecord) -> None:
        """Validate/buffer one attempt and seal a shard when a configured bound is reached."""
        # Require an explicit exclusive write context.
        if not self._active:
            # Keep imports/display-only functions incapable of accidental writes.
            raise ArtifactError("append() requires an active writer context.")
        # Validate trial/dataset/metric invariants before buffering any result.
        validate_record(record, self.snapshot.metadata)
        # Include buffered attempts in duplicate checks before publication.
        retained = (*self.snapshot.records, *self._buffer)
        # Reject duplicate attempts and any second completed result for a logical trial.
        if any(previous.attempt_id == record.attempt_id or (previous.status == record.status == "completed" and previous.trial_id == record.trial_id) for previous in retained):
            # Do not select the latest retry arbitrarily for scientific analysis.
            raise ArtifactError("Duplicate attempt or completed logical trial.")
        # Measure the exact serialized line length including its JSONL terminator.
        size = len(canonical_bytes(record.model_dump(mode="json"))) + 1
        # Reject an oversized individual record before it enters an unpublishable buffer.
        if size > self.max_shard_bytes:
            # Make necessary artifact-policy extensions explicit.
            raise ArtifactError("One trial record exceeds max_shard_bytes.")
        # Measure the existing bounded buffer using the same deterministic encoding.
        buffered_size = sum(len(canonical_bytes(item.model_dump(mode="json"))) + 1 for item in self._buffer)
        # Seal the previous buffer first if this record would exceed the byte limit.
        if self._buffer and buffered_size + size > self.max_shard_bytes:
            # Keep every emitted immutable shard within its predeclared operational limit.
            self.flush()
        # Add the complete validated attempt to unpublished local work.
        self._buffer.append(record)
        # Publish promptly when the configured attempt-count boundary is reached.
        if len(self._buffer) >= self.shard_records:
            # Seal bytes/checksum before advancing the manifest completion boundary.
            self.flush()

    def flush(self) -> None:
        """Seal buffered JSONL, then atomically publish a manifest naming those exact bytes."""
        # Do nothing when there is no unpublished work to finalize.
        if not self._buffer:
            # Avoid unnecessary manifest updates and empty shards.
            return
        # Require exclusive write ownership for publication as well as appends.
        if not self._active:
            # Never mutate a checkpoint outside its explicit writer boundary.
            raise ArtifactError("flush() requires an active writer context.")
        # Serialize immutable complete attempts with a final newline on every line.
        content = b"".join(canonical_bytes(record.model_dump(mode="json")) + b"\n" for record in self._buffer)
        # Compute a content identity before naming or publishing the shard.
        content_sha = digest_bytes(content)
        # Assign the next manifest-relative index without scanning unpublished orphan files.
        name = f"records-{len(self.snapshot.manifest.shards):06d}-{content_sha}.jsonl"
        # Publish finalized record bytes before publishing any completion reference.
        atomic_write(self.directory / name, content, immutable=True)
        # Describe exactly the immutable file that was successfully sealed.
        reference = ShardRef(name=name, sha256=content_sha, size_bytes=len(content), record_count=len(self._buffer))
        # Combine prior published successes with newly completed buffered attempts.
        completed = set(self.snapshot.manifest.completed_trial_ids) | {record.trial_id for record in self._buffer if record.status == "completed"}
        # Build the next complete checkpoint pointer from the previous verified history.
        manifest = CheckpointManifest(run_id=self.snapshot.metadata.run_id, metadata_sha256=self.snapshot.manifest.metadata_sha256, shards=(*self.snapshot.manifest.shards, reference), completed_trial_ids=tuple(sorted(completed)), updated_at=utc_now())
        # Atomically acknowledge the sealed shard and its logical completions locally.
        atomic_write(self.directory / "manifest.json", canonical_bytes(manifest.model_dump(mode="json")))
        # Advance the in-memory boundary only after manifest publication succeeds.
        self._snapshot = RunSnapshot(self.snapshot.metadata, manifest, (*self.snapshot.records, *self._buffer))
        # Release buffered work only once it belongs to the acknowledged checkpoint.
        self._buffer.clear()