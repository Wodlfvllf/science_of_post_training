# Inject persistence failures offline to test publication boundaries, not a real HF account.
"""Single-writer, local recovery, atomic fake-remote sync, and clean-directory restore tests."""

from __future__ import annotations

import os
import stat
import time
from uuid import uuid4

import pytest
from filelock import FileLock, Timeout

from interp import artifacts, hub
from interp.artifacts import ArtifactError, RunStore, read_run
from interp.config import compute_run_id
from interp.hub import restore, sync
from interp.metrics import paired_effects
from interp.runner import TrialExecutionError, run_trials, summarize_run
from interp.schemas import ExecutionSpec, GenerationRecord, RunMetadata


class FakeRemote:
    """In-memory atomic commits with independent before/after-publication fault injection."""

    def __init__(self):
        """Create a fake existing private repository without network/client dependencies."""
        # Use a syntactically valid non-production destination label.
        self.repo_id = "offline/checkpoint-tests"
        # Give the initial empty repository an immutable-looking test revision.
        self.current = "0" * 40
        # Preserve historical snapshots so restores can target older acknowledged commits.
        self.snapshots = {self.current: {}}
        # Count actual publication operations independently of receipt writes.
        self.commits = 0
        # Count access checks to ensure oversized uploads fail before contacting storage.
        self.head_calls = 0
        # Inject a failure before any manifest/data publication.
        self.fail_before = False
        # Inject acknowledgement loss after a complete remote commit.
        self.fail_after = False

    def head(self, *, write=False):
        """Resolve one stable snapshot, preserving read/write intent for the seam."""
        # Count the explicit storage-access boundary.
        self.head_calls += 1
        # Return only the current full immutable fake revision.
        return self.current

    def read(self, path, *, revision):
        """Read exact pinned bytes; do not automatically follow a later commit."""
        # Return a missing-file sentinel only when this specific snapshot lacks the path.
        return self.snapshots[revision].get(path)

    def commit(self, files, *, parent):
        """Publish all supplied files atomically, with an optimistic parent guard."""
        # Simulate the actual Hub's protection against concurrent remote writers.
        if parent != self.current:
            # Refuse clobbering a branch that advanced after head resolution.
            raise RuntimeError("Injected parent-commit mismatch.")
        # Simulate upload/commit failure before publication.
        if self.fail_before:
            # Leave all remote manifests and acknowledged history unchanged.
            raise RuntimeError("Injected failure before commit.")
        # Count one atomic publication, not one operation per uploaded file.
        self.commits += 1
        # Assign a deterministic distinct full revision for each committed snapshot.
        revision = f"{self.commits:040x}"
        # Publish every file together while preserving older historical snapshots.
        self.snapshots[revision] = {**self.snapshots[self.current], **files}
        # Advance the current branch only after the entire snapshot exists.
        self.current = revision
        # Simulate a lost response after the server has actually committed the checkpoint.
        if self.fail_after:
            # Do not roll back a publication whose acknowledgement was merely lost.
            raise RuntimeError("Injected acknowledgement loss after commit.")
        # Return the actual acknowledged publication revision.
        return revision


def test_incremental_resume_has_unique_successes_and_visible_incomplete_pairs(metadata, fake_trial):
    """Invocation budgets cannot duplicate logical successes or hide incomplete comparisons."""
    # Cap this invocation at one fresh attempt without changing run identity.
    limited = ExecutionSpec.model_validate({**metadata.spec.execution.model_dump(mode="python"), "max_new_trials": 1})
    # Start the ordinary shared runner with artificial scoring-only observations.
    first = run_trials(metadata, fake_trial, execution=limited)
    # Report a partial invocation rather than full experimental completion.
    assert (first.completed_trials, first.new_attempts, first.stop_reason) == (1, 1, "trial_limit")
    # Read-only analysis must retain both planned pairs, including missing conditions.
    rows = paired_effects(read_run(limited.output_dir))
    # Do not silently exclude incomplete source examples.
    assert len(rows) == 2 and all(row["status"] == "incomplete" for row in rows)
    # Complete all remaining declared observations using exactly the same metadata identity.
    second = run_trials(metadata, fake_trial)
    # Count existing success plus only three new logical observations.
    assert (second.completed_trials, second.new_attempts, second.stop_reason) == (4, 3, "complete")
    # A further explicit invocation must execute no already-completed trial function.
    third = run_trials(metadata, fake_trial)
    # Preserve exactly four successful records, not four per invocation.
    assert third.new_attempts == 0 and len(read_run(limited.output_dir).records) == 4
    # A local manifest must not be misrepresented as a remotely acknowledged checkpoint.
    assert third.remotely_acknowledged_trials == 0


def test_errors_and_interrupts_are_retained_then_missing_trials_retry(metadata, fake_trial):
    """A failed/interrupted attempt is evidence, not a logical completed result."""

    def fail(*args):
        """Simulate a model error containing sensitive text that must not enter records."""
        # Check that only the exception class, not this arbitrary message, is persisted.
        raise RuntimeError("hf_not_a_real_secret_DO_NOT_LOG")

    # Fail fast while still finalizing the attempt through the writer context.
    with pytest.raises(TrialExecutionError):
        # Execute an ordinary failing trial function through the real checkpointed runner.
        run_trials(metadata, fail)
    # Verify the unsuccessful attempt is saved but does not advance completion.
    failed = read_run(metadata.spec.execution.output_dir)
    # Preserve explicit failure status and safe category.
    assert failed.records[0].status == "failed" and failed.records[0].error_type == "builtins.RuntimeError"
    # Do not count the failure as a successful observation.
    assert not failed.manifest.completed_trial_ids
    # Do not copy secret-bearing exception text into any published artifact.
    assert "DO_NOT_LOG" not in b"".join(path.read_bytes() for path in metadata.spec.execution.output_dir.glob("*.jsonl")).decode()

    def interrupt(*args):
        """Simulate a user/kernel interruption at the model-call boundary."""
        # Exercise the runner's explicit interruption-finalization branch.
        raise KeyboardInterrupt

    # Retain the interrupted attempt before propagating the ordinary user interrupt.
    with pytest.raises(KeyboardInterrupt):
        # No successful output can be fabricated for an interrupted function.
        run_trials(metadata, interrupt)
    # Later explicit execution retries missing observations from their stable seeds.
    completed = run_trials(metadata, fake_trial)
    # Count four successful logical observations despite two earlier incomplete attempts.
    assert completed.completed_trials == 4 and completed.failed_attempts == 1
    # Retain both failures/interruptions and all successes for inspection.
    assert len(read_run(metadata.spec.execution.output_dir).records) == 6


def test_manifest_failure_leaves_sealed_orphan_uncompleted(metadata, fake_trial, monkeypatch):
    """A sealed shard without published manifest acknowledgement must be rerun, not inferred complete."""
    # Establish one acknowledged local logical success first.
    limited = ExecutionSpec.model_validate({**metadata.spec.execution.model_dump(mode="python"), "max_new_trials": 1})
    # Publish the first observation with the real atomic writer.
    run_trials(metadata, fake_trial, execution=limited)
    # Preserve the actual writer for all non-manifest artifact operations.
    real_write = artifacts.atomic_write

    def fail_manifest(path, content, **kwargs):
        """Fail after shard sealing but before switching the checkpoint pointer."""
        # Let immutable data publication succeed while withholding its completion pointer.
        if path.name == "manifest.json":
            # Simulate a persistent filesystem failure rather than a retryable temporary glitch.
            raise OSError("Injected manifest publication failure.")
        # Use the real durability path for metadata/shard files.
        return real_write(path, content, **kwargs)

    # Keep injection local to this explicit test scope.
    with monkeypatch.context() as scoped:
        # Replace only the manifest publication operation, not record construction.
        scoped.setattr(artifacts, "atomic_write", fail_manifest)
        # Expect the persistence error to surface rather than be marked successful.
        with pytest.raises(OSError):
            # Attempt the next logical observation, whose shard will remain an orphan.
            run_trials(metadata, fake_trial, execution=limited)
    # Verify readers still see only the previous acknowledged completion boundary.
    assert len(read_run(limited.output_dir).manifest.completed_trial_ids) == 1
    # Verify a sealed but unreferenced artifact really exists in this fault scenario.
    assert len(list(limited.output_dir.glob("records-*.jsonl"))) == 2
    # Explicit resume reruns the orphaned logical observation and finishes the rest.
    recovered = run_trials(metadata, fake_trial)
    # Never count the old orphan as an extra logical success.
    assert recovered.completed_trials == 4 and len(read_run(limited.output_dir).records) == 4


def test_corrupt_shards_and_numeric_profile_changes_are_refused(metadata, fake_trial):
    """Resume must fail on changed numerical stacks or checksum-invalid observations."""
    # Publish a complete valid small checkpoint first.
    run_trials(metadata, fake_trial)
    # Preserve scientific identity but change actual execution compatibility deliberately.
    raw = metadata.model_dump(mode="python")
    # Simulate different library/hardware numerical provenance.
    raw["runtime_profile"] = {"kind": "different-execution-profile"}
    # Validate this alternate metadata before passing it to the writer.
    changed = RunMetadata.model_validate(raw)
    # Refuse a silent mix of otherwise identically named logical observations.
    with pytest.raises(ArtifactError, match="profile changed"):
        # Attempt to resume into the existing unchanged checkpoint directory.
        run_trials(changed, fake_trial)
    # Identify an actual immutable record file referenced by the published manifest.
    snapshot = read_run(metadata.spec.execution.output_dir)
    # Deliberately corrupt a temporary-test artifact, never a user research output.
    (metadata.spec.execution.output_dir / snapshot.manifest.shards[0].name).write_bytes(b"corrupt\n")
    # Read-only inspection must reject corruption before reporting any paired effects.
    with pytest.raises(ArtifactError, match="checksum"):
        # Verify hash/size before JSONL parsing or logical completion.
        read_run(metadata.spec.execution.output_dir)


def test_single_writer_and_cancelled_invocation_do_not_execute_trials(metadata, fake_trial):
    """Concurrent writers and pre-start cancellation must not launch duplicate model work."""
    # Create the directory as an explicit test setup operation for the held writer lock.
    directory = metadata.spec.execution.output_dir
    # Materialize only this temporary test run location.
    directory.mkdir()
    # Hold the same OS-backed lock that a real runner owns.
    with FileLock(str(directory / ".writer.lock"), timeout=0):
        # Another writer must fail immediately rather than racing completion.
        with pytest.raises(Timeout):
            # Attempt ordinary execution while a distinct FileLock owns the run.
            run_trials(metadata, fake_trial)
    # Request a cooperative stop before any actual model function is called.
    summary = run_trials(metadata, fake_trial, should_cancel=lambda: True)
    # Preserve a valid empty checkpoint without a fictitious success.
    assert (summary.completed_trials, summary.new_attempts, summary.stop_reason) == (0, 0, "cancelled")


def test_remote_failure_before_commit_has_no_durable_completion(metadata, fake_trial):
    """A failed upload cannot advance a receipt or make any local result remotely durable."""
    # Publish local observations, which alone are not a remote checkpoint.
    run_trials(metadata, fake_trial)
    # Construct a deterministic offline remote store.
    remote = FakeRemote()
    # Inject failure before the atomic remote publication.
    remote.fail_before = True
    # Surface storage errors rather than hiding them behind background retries.
    with pytest.raises(RuntimeError, match="before commit"):
        # Request one explicit synchronous checkpoint publication.
        sync(metadata.spec.execution.output_dir, remote)
    # Preserve the original empty remote history.
    assert remote.commits == 0
    # Do not acknowledge any remotely completed observations on local-only progress.
    assert summarize_run(metadata.spec.execution.output_dir).remotely_acknowledged_trials == 0


def test_acknowledgement_loss_restores_verified_successes_without_original_data(metadata, fake_trial, tmp_path):
    """A clean directory can recover an atomically published checkpoint even without a local receipt."""
    # Publish only one logical observation to leave a real missing schedule after restore.
    limited = ExecutionSpec.model_validate({**metadata.spec.execution.model_dump(mode="python"), "max_new_trials": 1})
    # Establish one local completion boundary.
    run_trials(metadata, fake_trial, execution=limited)
    # Create a fake repository that can lose its commit acknowledgement.
    remote = FakeRemote()
    # Commit succeeds but the client sees an error before writing its receipt.
    remote.fail_after = True
    # Expose acknowledgement uncertainty instead of reporting false durable success locally.
    with pytest.raises(RuntimeError, match="acknowledgement loss"):
        # Run the same explicit synchronous sync function used by the real adapter.
        sync(limited.output_dir, remote)
    # Confirm the remote checkpoint actually exists despite the lost local acknowledgement.
    assert remote.commits == 1 and not (limited.output_dir / "sync-receipt.json").exists()
    # Restore into a clean, previously nonexistent local result directory.
    destination = tmp_path / "clean-machine"
    # Pin recovery to the commit whose response was lost.
    receipt = restore(destination, remote, metadata.run_id, revision=remote.current)
    # Count only the one verified published observation, not the remaining missing schedule.
    assert len(receipt.completed_trial_ids) == 1
    # Reuse the archived dataset snapshot independently of its original local paths.
    restored = read_run(destination)
    # Choose new local operational settings while preserving restored scientific identity.
    execution = ExecutionSpec.model_validate({**metadata.spec.execution.model_dump(mode="python"), "output_dir": destination})
    # Complete only missing observations through the same ordinary runner.
    summary = run_trials(restored.metadata, fake_trial, execution=execution)
    # Preserve unique logical successes and keep later local work outside the old durable boundary.
    assert (summary.completed_trials, summary.new_attempts, summary.remotely_acknowledged_trials) == (4, 3, 1)
    # Retain exactly one successful observation per logical trial.
    assert len(read_run(destination).records) == 4


def test_sync_is_idempotent_and_refuses_an_older_local_branch(metadata, fake_trial, tmp_path):
    """Repeated acknowledgements need no new commit; stale local history cannot overwrite remote progress."""
    # Establish a partial local run and an explicit fake-remote checkpoint.
    limited = ExecutionSpec.model_validate({**metadata.spec.execution.model_dump(mode="python"), "max_new_trials": 1})
    # Save one paired-condition observation.
    run_trials(metadata, fake_trial, execution=limited)
    # Initialize deterministic offline publication history.
    remote = FakeRemote()
    # Obtain the first actual acknowledged remote revision.
    first = sync(limited.output_dir, remote)
    # Repeating the same snapshot publication must not create duplicate remote commits.
    assert sync(limited.output_dir, remote).revision == first.revision and remote.commits == 1
    # Preserve an older independent local copy at the acknowledged first revision.
    stale = tmp_path / "stale-copy"
    # Restore only the prior published boundary, not later moving remote content.
    restore(stale, remote, metadata.run_id, revision=first.revision)
    # Finish the original local run and advance the remote checkpoint deliberately.
    run_trials(metadata, fake_trial)
    # Publish the complete later history.
    latest = sync(limited.output_dir, remote)
    # An older local branch must not silently replace the current remote manifest.
    with pytest.raises(ArtifactError, match="ahead"):
        # Attempt explicit publication from a known stale restored snapshot.
        sync(stale, remote)
    # Preserve the newer acknowledged remote revision and completion boundary.
    assert remote.current == latest.revision and remote.commits == 2


def test_corrupt_remote_restore_never_publishes_destination(metadata, fake_trial, tmp_path):
    """Missing/checksum-invalid referenced bytes must leave the final restore folder absent."""
    # Publish a complete valid local and fake-remote checkpoint.
    run_trials(metadata, fake_trial)
    # Construct the offline atomic store and obtain a real acknowledgement.
    remote = FakeRemote()
    # Publish all declared evidence and the manifest together.
    receipt = sync(metadata.spec.execution.output_dir, remote)
    # Select an actual referenced artifact to corrupt in the test-only remote snapshot.
    reference = read_run(metadata.spec.execution.output_dir).manifest.shards[0]
    # Modify fake bytes without updating their declared immutable digest.
    remote.snapshots[receipt.revision][f"{receipt.prefix}/{reference.name}"] = b"corrupt\n"
    # Restore into a new nonexistent destination so partial visibility is observable.
    destination = tmp_path / "must-not-publish"
    # Surface checksum mismatch instead of exposing a half-valid run directory.
    with pytest.raises(ArtifactError, match="checksum"):
        # Fetch only the referenced pinned checkpoint evidence.
        restore(destination, remote, metadata.run_id, revision=receipt.revision)
    # Keep corrupt/incomplete downloaded artifacts out of the published destination.
    assert not destination.exists()


def test_oversized_sync_fails_before_remote_access(metadata, fake_trial, monkeypatch):
    """Do not silently switch a bounded atomic snapshot into a multi-commit upload."""
    # Create a legitimate local publication boundary first.
    run_trials(metadata, fake_trial)
    # Use a fake remote whose access checks are observable.
    remote = FakeRemote()
    # Lower the payload limit for this test without allocating huge fixture artifacts.
    monkeypatch.setattr(hub, "MAX_CHECKPOINT_BYTES", 1)
    # Refuse oversized snapshots before authentication or any storage publication.
    with pytest.raises(ValueError, match="atomic sync limit"):
        # Request an explicitly too-large atomic checkpoint under the injected limit.
        sync(metadata.spec.execution.output_dir, remote)
    # Ensure the remote client was never contacted.
    assert remote.head_calls == 0 and remote.commits == 0


@pytest.mark.parametrize("remove", [True, False])
def test_identical_remote_pointer_cannot_acknowledge_missing_or_changed_evidence(metadata, fake_trial, remove):
    """An unchanged manifest cannot certify evidence deleted/edited by a later commit."""
    # Produce real verified local records using the deterministic ordinary test recipe.
    run_trials(metadata, fake_trial)
    # Acknowledge an initial valid publication before simulating a later remote edit.
    remote = FakeRemote()
    # Save the valid receipt so a failed later verification must not replace it.
    original = sync(metadata.spec.execution.output_dir, remote)
    # Select one declared result shard, not an unrelated file in the repository.
    reference = read_run(metadata.spec.execution.output_dir).manifest.shards[0]
    # Create a new immutable-looking revision while retaining the unchanged manifest pointer.
    modified = dict(remote.snapshots[original.revision])
    # Identify the exact reference whose remote evidence is being damaged.
    path = f"{original.prefix}/{reference.name}"
    # Cover missing and corrupt evidence independently under the same verification boundary.
    if remove:
        # Simulate a later remote commit deleting an essential shard.
        del modified[path]
    # Exercise checksum-different bytes as a distinct unchanged-pointer failure case.
    else:
        # Simulate a later remote commit modifying an essential shard.
        modified[path] = b"corrupt\n"
    # Advance the fake branch without changing the old immutable snapshot.
    remote.current = "f" * 40
    # Publish the controlled damaged snapshot at only this new revision.
    remote.snapshots[remote.current] = modified
    # Refuse a fresh durability receipt based only on the identical manifest bytes.
    with pytest.raises(ArtifactError, match="evidence"):
        # Repeated sync must inspect all referenced bytes at the pinned new parent.
        sync(metadata.spec.execution.output_dir, remote)
    # Keep the previous valid acknowledgement instead of certifying the damaged revision.
    assert (metadata.spec.execution.output_dir / "sync-receipt.json").read_text().find(original.revision) >= 0
    # Do not manufacture an extra commit or silently repair uncertain remote history.
    assert remote.commits == 1


def test_restore_counts_metadata_in_combined_size_limit(metadata, fake_trial, tmp_path, monkeypatch):
    """Restore must include archived examples/metadata in the same bound used by upload."""
    # Generate and publish a legitimate small offline checkpoint.
    run_trials(metadata, fake_trial)
    # Use the fake remote without any authenticated storage connection.
    remote = FakeRemote()
    # Obtain one pinned, fully published checkpoint.
    receipt = sync(metadata.spec.execution.output_dir, remote)
    # Set a bound just below the complete payload, but above pointer plus declared shards.
    total = sum(len(value) for key, value in remote.snapshots[receipt.revision].items() if key.startswith(receipt.prefix + "/"))
    # Exclude exactly one byte of metadata-inclusive payload from the supported test budget.
    monkeypatch.setattr(hub, "MAX_CHECKPOINT_BYTES", total - 1)
    # A corrupt or oversized restore must not expose a partly verified destination.
    destination = tmp_path / "oversized-restore"
    # Enforce actual cumulative size even when all manifest-declared shard sizes fit.
    with pytest.raises(ArtifactError, match="small-run"):
        # Use the exact acknowledged revision for every remote read.
        restore(destination, remote, metadata.run_id, revision=receipt.revision)
    # Preserve the nonexistent final destination on size rejection.
    assert not destination.exists()


def test_duplicate_successes_are_rejected_before_publication(metadata, fake_trial):
    """A writer cannot append the same successful logical observation a second time."""
    # Produce a complete publication that the writer can verify independently.
    run_trials(metadata, fake_trial)
    # Retain one actual successful record and its already-published logical identity.
    record = read_run(metadata.spec.execution.output_dir).records[0]
    # Acquire the real single-writer boundary before attempting invalid appends.
    with RunStore(metadata.spec.execution.output_dir).writer(metadata) as store:
        # Reject duplicated successful evidence instead of choosing first/last arbitrarily.
        with pytest.raises(ArtifactError, match="Duplicate"):
            # Reappending the identical attempt must not advance the manifest.
            store.append(record)
        # A new UUID must not bypass uniqueness of the already-completed logical observation.
        raw = record.model_dump(mode="python")
        # Simulate a separate retry that claims the same successful scientific observation.
        raw["attempt_id"] = uuid4()
        # Reject duplicate logical success independently of execution-attempt uniqueness.
        with pytest.raises(ArtifactError, match="Duplicate"):
            # Revalidate changed bookkeeping before testing the store-level identity rule.
            store.append(type(record).model_validate(raw))
    # Preserve the four distinct completed observations without extra attempts/shards.
    assert len(read_run(metadata.spec.execution.output_dir).records) == 4


def test_file_sync_failure_during_sealing_preserves_last_manifest(metadata, fake_trial, monkeypatch):
    """An interrupted temporary-file fsync cannot publish truncated evidence or completion."""
    # Save one earlier valid completion boundary using a one-attempt invocation budget.
    limited = ExecutionSpec.model_validate({**metadata.spec.execution.model_dump(mode="python"), "max_new_trials": 1})
    # Publish the baseline checkpoint with real file flush/fsync/link/manifest operations.
    run_trials(metadata, fake_trial, execution=limited)
    # Preserve the actual directory-sync implementation for unaffected operations.
    real_fsync = os.fsync

    def fail_regular_file(descriptor):
        """Fail after writing/flushing temporary payload bytes but before immutable publication."""
        # Distinguish a regular temporary data file from directory-entry durability requests.
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            # Leave this partially finalized payload unpublished at its temporary name.
            raise OSError("Controlled payload fsync failure.")
        # Continue normal directory durability behavior whenever it is reached.
        return real_fsync(descriptor)

    # Isolate this actual filesystem fault to one attempted invocation.
    with monkeypatch.context() as scoped:
        # Affect only filesystem finalization, not trial scoring/status construction.
        scoped.setattr(artifacts.os, "fsync", fail_regular_file)
        # Surface the failed durability operation rather than advancing the manifest.
        with pytest.raises(OSError, match="fsync"):
            # Attempt one next logical trial under the real local writer.
            run_trials(metadata, fake_trial, execution=limited)
    # Read only the last valid pointer after the fault, not temporary partial data.
    snapshot = read_run(limited.output_dir)
    # Preserve exactly the earlier successful observation.
    assert len(snapshot.records) == 1 and len(snapshot.manifest.completed_trial_ids) == 1
    # Clean unpublished temporary payloads on ordinary exceptions.
    assert not list(limited.output_dir.glob(".pending-*"))
    # No immutable result file is linked before its complete payload fsync succeeds.
    assert len(list(limited.output_dir.glob("records-*.jsonl"))) == 1
    # A later explicit invocation reruns only missing observations under normal filesystem semantics.
    assert run_trials(metadata, fake_trial).new_attempts == 3


def test_expired_budget_schedules_no_model_work(metadata, fake_trial):
    """Preparation/loading time counts toward the deadline before the next trial starts."""
    # Declare enough work time for schema validity but start outside the entire budget.
    settings = ExecutionSpec.model_validate({**metadata.spec.execution.model_dump(mode="python"), "max_seconds": 10.0, "shutdown_margin_seconds": 2.0})
    # Simulate a slow preceding explicit load by providing its earlier monotonic start.
    started = time.monotonic() - 100.0
    # Enter the scheduler after preparation/loading consumed its allowed time.
    summary = run_trials(metadata, fake_trial, execution=settings, started_monotonic=started)
    # Do not manufacture an execution attempt or successful observation after expiry.
    assert (summary.stop_reason, summary.new_attempts, summary.completed_trials) == ("time_budget", 0, 0)
    # Preserve a valid empty checkpoint rather than an unverified completed result.
    assert not read_run(settings.output_dir).records


def test_transient_cancel_is_latched_without_second_predicate_query(metadata, fake_trial):
    """A short-lived cancellation flag must stay cancelled and retain its actual cause."""
    # Count caller-owned cancellation checks explicitly.
    checks = []

    def cancel_once():
        """Expose a transient flag to catch unsafe repeated summary-time evaluation."""
        # Record the actual check without relying on wall-clock timing.
        checks.append(1)
        # Request cancellation only on the first observation of this flag.
        return len(checks) == 1

    # A cancelled invocation must not start even one ordinary trial function.
    summary = run_trials(metadata, fake_trial, should_cancel=cancel_once)
    # Preserve the observed cancellation rather than misreporting an absent time limit.
    assert summary.stop_reason == "cancelled" and summary.new_attempts == 0
    # Reuse the latched stop cause rather than querying the now-cleared external flag.
    assert checks == [1]


def test_partial_generation_is_saved_but_never_satisfies_trial(metadata, fake_trial):
    """Candidate scores with cancelled generated output remain an incomplete attempt."""
    # Enable generation in this artificial-output metadata instead of changing the recipe silently.
    raw = metadata.model_dump(mode="python")
    # Declare that every successful logical observation now requires generated output too.
    raw["spec"]["generation"]["enabled"] = True
    # Revalidate the changed scientific specification before computing its identity.
    spec = type(metadata.spec).model_validate(raw["spec"])
    # Bind the changed generation requirement into a distinct scientific run identity.
    raw["run_id"] = compute_run_id(spec, metadata.dataset_sha256, metadata.source_sha256)
    # Preserve the explicitly artificial model/runtime labels in this storage-only test.
    generated_metadata = RunMetadata.model_validate(raw)
    # Keep only the effective fields this fixture claims to implement, never pretend real HF inference.
    config = {"max_new_tokens": 8, "do_sample": False, "use_cache": False}

    def partial(example, replicate, seed, stop):
        """Return correct artificial scores with a deliberately interrupted output suffix."""
        # Reuse the independently schema-valid scoring-only fixture observations.
        result = fake_trial(example, replicate, seed, stop).model_dump(mode="python")
        # Preserve an explicit short partial suffix and its interruption category.
        result["generation"] = GenerationRecord(token_ids=(1,), text="<bos>", finish_reason="cancelled", effective_config=config)
        # Revalidate the complete returned result before runner status handling.
        return type(fake_trial(example, replicate, seed, stop)).model_validate(result)

    # Execute exactly one partial ordinary trial under the actual checkpointed scheduler.
    summary = run_trials(generated_metadata, partial)
    # Preserve partial evidence but not logical completion or an invented expired budget.
    assert (summary.completed_trials, summary.new_attempts, summary.stop_reason) == (0, 1, "cancelled")
    # Inspect the actual retained interrupted output independently of summary bookkeeping.
    snapshot = read_run(spec.execution.output_dir)
    # Retain exact partial output while keeping it outside successful paired comparisons.
    assert snapshot.records[0].status == "cancelled" and snapshot.records[0].result.generation.token_ids == (1,)
    # Keep every planned comparison explicitly incomplete after cancellation.
    assert all(row["status"] == "incomplete" for row in paired_effects(snapshot))
