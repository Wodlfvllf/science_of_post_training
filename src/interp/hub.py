# Explicitly sync or restore small verified checkpoints; never upload during imports.
# Authentication comes from supported HF configuration/secrets, never from experiment YAML.
"""Bounded atomic Hub checkpoints, pinned restore, and a fakeable remote boundary.

This first implementation publishes at most 128 files/16 MiB in one synchronous
``create_commit`` with an optimistic parent-commit guard. It deliberately rejects
larger snapshots instead of pretending a multi-commit upload_folder is transactional.
Only an existing private dataset repository is supported; no repo is auto-created.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from filelock import FileLock

from .artifacts import ArtifactError, RunStore, atomic_write, read_run, sync_directory
from .data import canonical_bytes, digest_bytes
from .runtime import utc_now
from .schemas import CheckpointManifest, SyncReceipt

# Keep one commit comfortably below Hub API limits and small enough to verify in memory.
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
# Bound operation counts independently of payload size.
MAX_CHECKPOINT_FILES = 128


class RemoteCheckpoint(Protocol):
    """Minimal storage seam for real HF commits and deterministic offline failure tests."""

    # Preserve an explicit destination without storing an authentication token in records.
    repo_id: str

    def head(self, *, write: bool = False) -> str:
        """Resolve a full commit and authenticate; a commit must enforce actual write permission."""
        # Protocol implementations provide the actual storage behavior.
        ...

    def read(self, path: str, *, revision: str) -> bytes | None:
        """Read exact pinned bytes; return None only for a genuinely missing file."""
        # Protocol implementations must not substitute a moving revision.
        ...

    def commit(self, files: Mapping[str, bytes], *, parent: str) -> str:
        """Atomically publish all supplied files only if the branch still matches parent."""
        # Protocol implementations must return only after publication acknowledgement.
        ...


class HuggingFaceRemote:
    """Explicit adapter to an existing private HF dataset repo using existing auth.

    Construction does not import the client, authenticate, create a repo, or transfer
    data. All network access is inside the explicit methods below.
    """

    def __init__(self, repo_id: str) -> None:
        """Validate destination syntax without contacting or changing the Hub."""
        # Require a clear namespace/repository destination rather than a URL or local path.
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id) is None:
            # Reject ambiguous storage destinations before any authentication call.
            raise ValueError("Use an existing private HF dataset repo as namespace/repository.")
        # Remember only public destination metadata, never a token value.
        self.repo_id = repo_id

    def head(self, *, write: bool = False) -> str:
        """Check authenticated private read access and resolve one immutable revision.

        ``write`` expresses caller intent. The supported Hub API has no write flag
        on ``auth_check``; only the actual explicit commit verifies write permission.
        Reading an already-published checkpoint does not claim permission to upload.
        """
        # Import remote-storage dependencies only during an explicit network operation.
        from huggingface_hub import HfApi

        # Reuse supported HF environment/existing authentication without persisting a token.
        api = HfApi()
        # Check Python read authorization; create_commit separately enforces write permission.
        api.auth_check(repo_id=self.repo_id, repo_type="dataset")
        # Resolve a single revision and check actual destination visibility.
        information = api.repo_info(repo_id=self.repo_id, repo_type="dataset", revision="main")
        # Refuse accidental publication of raw prompts to a public results repository.
        if not information.private:
            # Require the destination's privacy to be fixed explicitly outside this workbench.
            raise ValueError("Checkpoint sync/restore requires an existing private dataset repository.")
        # Validate the server's full immutable commit identifier.
        return _revision(information.sha)

    def read(self, path: str, *, revision: str) -> bytes | None:
        """Download pinned content and copy bytes without modifying the HF cache."""
        # Import the established cache/download API only for this explicit read.
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

        # Require immutable revision syntax even when called independently of restore().
        pinned = _revision(revision)
        # Distinguish a missing remote artifact from authentication/network failures.
        try:
            # Download exactly the requested file version, not a moving main branch.
            cached = hf_hub_download(repo_id=self.repo_id, repo_type="dataset", filename=path, revision=pinned)
        # A cache miss after offline/network failure is not proof that the remote file is absent.
        except LocalEntryNotFoundError:
            # Preserve unavailable-storage uncertainty instead of treating it as a new checkpoint.
            raise
        # Only a genuinely absent remote file may be treated as an empty prior checkpoint.
        except EntryNotFoundError:
            # Do not hide permission, timeout, or unavailable-repository errors.
            return None
        # Read immutable cached bytes without editing the library-owned cache file.
        return Path(cached).read_bytes()

    def commit(self, files: Mapping[str, bytes], *, parent: str) -> str:
        """Publish one synchronous bounded commit with a concurrent-update guard."""
        # Import commit operations only during an explicit upload action.
        from huggingface_hub import CommitOperationAdd, HfApi

        # Enforce the workbench's safe small-checkpoint contract before uploading anything.
        _bounded(files)
        # Create fresh operation objects; the HF API mutates them during upload.
        operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=content) for path, content in files.items()]
        # Await one actual commit; no background future or auto-retrying scheduler is used.
        information = HfApi().create_commit(repo_id=self.repo_id, repo_type="dataset", revision="main", parent_commit=_revision(parent), operations=operations, commit_message="Publish verified interpretability checkpoint", run_as_future=False)
        # Acknowledge durability only with the actual committed revision returned by the API.
        return _revision(information.oid)


def _revision(value: str) -> str:
    """Require a full remote git SHA, never a mutable tag, branch, or abbreviated ID."""
    # Validate the immutable restore/commit convention used by this release.
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        # Reject moving or ambiguous remote checkpoint references.
        raise ValueError("A remote checkpoint revision must be a full 40-character commit SHA.")
    # Return the unchanged immutable revision.
    return value


def _bounded(files: Mapping[str, bytes]) -> None:
    """Reject checkpoints too large for the explicitly atomic initial upload strategy."""
    # Bound both the number of atomic operations and total in-memory/upload payload.
    if len(files) > MAX_CHECKPOINT_FILES or sum(len(content) for content in files.values()) > MAX_CHECKPOINT_BYTES:
        # Require a separately designed multi-stage persistence extension for large artifacts.
        raise ValueError("Checkpoint exceeds the initial 128-file/16-MiB atomic sync limit.")


def sync(directory: Path, remote: RemoteCheckpoint) -> SyncReceipt:
    """Verify and synchronously publish one bounded local checkpoint, then acknowledge it.

    A remote history ahead of/diverged from local history is refused. An upload error
    leaves the previous receipt untouched. If acknowledgement is lost after commit,
    restore or repeated sync can recover the published manifest without duplicate trials.
    """
    # Inspect and verify local artifacts before acquiring the explicit upload/writer boundary.
    initial = read_run(directory)
    # Prevent local result mutation while preparing and publishing this exact snapshot.
    store = RunStore(directory)
    # Revalidate the latest snapshot under the same single-writer lock used by execution.
    with store.writer(initial.metadata):
        # Use the freshly verified current local checkpoint, not the earlier unlocked view.
        snapshot = store.snapshot
        # Isolate all results under their immutable science identity within the private repo.
        prefix = f"runs/{snapshot.metadata.run_id}"
        # Include only declared provenance, published shards, and the manifest pointer.
        names = ["metadata.json", *(reference.name for reference in snapshot.manifest.shards), "manifest.json"]
        # Freeze exact bounded file contents so local edits cannot race a mutable upload stream.
        files = {f"{prefix}/{name}": (directory / name).read_bytes() for name in names}
        # Check payload limits before any authentication/upload network action.
        _bounded(files)
        # Authenticate and pin the parent once; the actual commit enforces write permission.
        parent = _revision(remote.head(write=True))
        # Read previous remote completion only at that pinned parent revision.
        previous_bytes = remote.read(f"{prefix}/manifest.json", revision=parent)
        # Refuse overwriting a diverged/ahead logical run history.
        if previous_bytes is not None:
            # Validate the remote publication boundary's schema and allowed references.
            previous = CheckpointManifest.model_validate_json(previous_bytes)
            # Require the same immutable provenance snapshot and an ancestral shard history.
            if previous.run_id != snapshot.metadata.run_id or previous.metadata_sha256 != snapshot.manifest.metadata_sha256 or previous.shards != snapshot.manifest.shards[: len(previous.shards)] or not set(previous.completed_trial_ids).issubset(snapshot.manifest.completed_trial_ids):
                # Ask for an explicit restore/reconciliation instead of discarding remote evidence.
                raise ArtifactError("Remote history differs or is ahead; restore it before syncing.")
        # Retrieve the exact manifest bytes included in this bounded atomic snapshot.
        manifest_bytes = files[f"{prefix}/manifest.json"]
        # Avoid duplicate commits after an acknowledged or acknowledgement-lost identical upload.
        if previous_bytes == manifest_bytes:
            # An unchanged pointer is not enough: verify every referenced byte at that revision.
            for path, content in files.items():
                # The manifest was already fetched and compared at this exact pinned parent.
                if path == f"{prefix}/manifest.json":
                    # Avoid one redundant remote read while still verifying all declared evidence.
                    continue
                # Refuse a receipt when a later commit removed or changed referenced evidence.
                if remote.read(path, revision=parent) != content:
                    # Do not infer remote durability merely from an unchanged manifest filename.
                    raise ArtifactError("Remote checkpoint evidence is missing or changed; reconcile it before syncing.")
            # All referenced bytes are verified at the existing immutable publication revision.
            committed = parent
        # Publish all immutable data and the pointer together in one synchronous commit.
        else:
            # A parent mismatch raises rather than silently clobbering a concurrent writer.
            committed = _revision(remote.commit(files, parent=parent))
        # Create a receipt only after the server acknowledged an actual publication revision.
        receipt = SyncReceipt(repo_id=remote.repo_id, prefix=prefix, revision=committed, run_id=snapshot.metadata.run_id, manifest_sha256=digest_bytes(manifest_bytes), completed_trial_ids=snapshot.manifest.completed_trial_ids, acknowledged_at=utc_now())
        # Record the separate remote boundary atomically; do not change scientific metadata.
        atomic_write(directory / "sync-receipt.json", canonical_bytes(receipt.model_dump(mode="json")))
    # Return the immutable acknowledged revision for controlled shutdown or pinned restore.
    return receipt


def restore(directory: Path, remote: RemoteCheckpoint, run_id: str, *, revision: str | None = None) -> SyncReceipt:
    """Restore into a new directory, verifying all pinned bytes before publishing it.

    Original local dataset/config paths are not needed: metadata contains validated
    example snapshots. Existing directories are never overwritten or merged. On any
    corrupt/missing file or network failure the final destination remains unpublished.
    """
    # Require content-identity syntax before using a run ID in any remote path.
    if re.fullmatch(r"[0-9a-f]{64}", run_id) is None:
        # Prevent arbitrary path traversal through a checkpoint namespace.
        raise ValueError("run_id must be a complete SHA-256 content identity.")
    # Refuse existing destinations, including empty directories and symlinks.
    if directory.exists() or directory.is_symlink():
        # Preserve all existing local evidence rather than merging or replacing it.
        raise ArtifactError("Restore requires a new, nonexistent destination directory.")
    # Create the explicitly requested parent only inside the restore operation.
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Serialize competing restore calls without creating the final run directory.
    with FileLock(str(directory.parent / f".restore-{directory.name}.lock"), timeout=0):
        # Recheck after acquiring the lock in case another restore just finished.
        if directory.exists() or directory.is_symlink():
            # Never replace a concurrently restored result folder.
            raise ArtifactError("Restore destination already exists.")
        # Verify private read access, then choose the explicit or current full revision once.
        head = remote.head(write=False)
        # Pin every required artifact to exactly the same immutable commit.
        pinned = _revision(revision or head)
        # Scope artifact paths by immutable experimental identity.
        prefix = f"runs/{run_id}"
        # Read the remote completion pointer at the selected fixed revision.
        manifest_bytes = remote.read(f"{prefix}/manifest.json", revision=pinned)
        # Missing publication is not a recoverable completed checkpoint.
        if manifest_bytes is None:
            # Do not infer completion by scanning remote orphan files.
            raise ArtifactError("No published checkpoint exists at that run/revision.")
        # Bound the pointer before allocating its schema or requesting referenced payloads.
        if len(manifest_bytes) > MAX_CHECKPOINT_BYTES:
            # Keep malformed/untrusted manifests within the same small-checkpoint contract.
            raise ArtifactError("Remote checkpoint exceeds the supported small-run format.")
        # Validate filename syntax and format before accessing any referenced artifact.
        manifest = CheckpointManifest.model_validate_json(manifest_bytes)
        # Require the pointer to identify exactly the requested run.
        if manifest.run_id != run_id:
            # Reject a remote path whose contents identify different science.
            raise ArtifactError("Remote manifest run identity differs from the requested run.")
        # Bound restored file counts/sizes before requesting all artifact payloads.
        if len(manifest.shards) + 2 > MAX_CHECKPOINT_FILES or sum(reference.size_bytes for reference in manifest.shards) + len(manifest_bytes) > MAX_CHECKPOINT_BYTES:
            # Match the explicitly small snapshot strategy used by sync().
            raise ArtifactError("Remote checkpoint exceeds the supported small-run format.")
        # Keep all restored bytes private until checksum/schema verification succeeds.
        with tempfile.TemporaryDirectory(prefix=".restore-pending-", dir=directory.parent) as staging_name:
            # Use a same-filesystem directory so final publication is one atomic rename.
            staging = Path(staging_name)
            # Include metadata and actual payload lengths, not only manifest-declared shard sizes.
            restored_bytes = len(manifest_bytes)
            # Fetch only immutable metadata and the shards actually referenced by the manifest.
            for name in ["metadata.json", *(reference.name for reference in manifest.shards)]:
                # Read each artifact from the same fixed remote commit.
                content = remote.read(f"{prefix}/{name}", revision=pinned)
                # Refuse a pointer whose declared evidence is absent.
                if content is None:
                    # Leave the final destination unpublished on incomplete/corrupt restore.
                    raise ArtifactError("A required remote artifact is missing.")
                # Account for the actual bytes returned by the pinned storage adapter.
                restored_bytes += len(content)
                # Refuse oversized metadata, misleading shard sizes, or an oversized combined snapshot.
                if restored_bytes > MAX_CHECKPOINT_BYTES:
                    # Match upload bounds before writing any more unpublished payload bytes.
                    raise ArtifactError("Remote checkpoint exceeds the supported small-run format.")
                # Write complete immutable bytes only inside the unpublished staging folder.
                atomic_write(staging / name, content, immutable=True)
            # Write the exact pinned pointer after its declared data is locally available.
            atomic_write(staging / "manifest.json", manifest_bytes, immutable=True)
            # Apply full local hash, schema, logical trial, and completion verification.
            read_run(staging)
            # Preserve a local acknowledgement of this exact verified remote publication.
            receipt = SyncReceipt(repo_id=remote.repo_id, prefix=prefix, revision=pinned, run_id=run_id, manifest_sha256=digest_bytes(manifest_bytes), completed_trial_ids=manifest.completed_trial_ids, acknowledged_at=utc_now())
            # Publish the receipt within the same directory-level restore boundary.
            atomic_write(staging / "sync-receipt.json", canonical_bytes(receipt.model_dump(mode="json")), immutable=True)
            # Atomically expose only a fully verified restored run, never a partial download.
            os.replace(staging, directory)
            # Request persistence of the restored directory entry.
            sync_directory(directory.parent)
    # Return the pinned acknowledged boundary; model execution remains a separate action.
    return receipt