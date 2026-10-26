"""Periodic verified ZIP snapshots using an already authenticated fsspec filesystem.

Run in the notebook process, not the GPU environment. Only standard-library imports
are needed here. No credentials are copied and no package source is modified.
Each archive captures manifests first, then their immutable referenced shards;
concurrent generation can continue. Archives are independent, not incremental.
Keep the notebook kernel alive; a stopped kernel cannot continue uploading.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def sha256(content):
    """Hash exact artifact bytes, matching the local manifest format."""
    return hashlib.sha256(content).hexdigest()


def read_regular(path):
    """Refuse symlinks rather than copying unrelated files or credentials."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Not a regular artifact: {path.name}")
    return path.read_bytes()


def capture(root, destination):
    """Write a self-contained ZIP of published results plus recovery material.

    Manifests are atomically replaced locally. Once captured, their referenced
    metadata/shards never change. Orphans, locks, caches, weights and secrets are
    excluded. Reports/logs are best-effort point-in-time copies, not run state.
    """
    root = Path(root).resolve()
    boundaries = []
    for path in sorted((root / "runs/posttrain").glob("*/manifest.json")):
        if path.parent.is_symlink():
            raise ValueError("Linked run directory refused")
        payload = read_regular(path)
        boundaries.append((path, payload, json.loads(payload)))
    if not boundaries:
        raise ValueError("No published posttraining runs found yet")

    completed = {}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, payload, manifest in boundaries:
            metadata_path = path.parent / "metadata.json"
            metadata = read_regular(metadata_path)
            if sha256(metadata) != manifest["metadata_sha256"]:
                raise ValueError("Metadata checksum mismatch")
            archive.writestr(str(metadata_path.relative_to(root)), metadata)
            for index, ref in enumerate(manifest["shards"]):
                expected = f"records-{index:06d}-{ref['sha256']}.jsonl"
                if ref["name"] != expected:
                    raise ValueError("Invalid shard name/index")
                shard = path.parent / expected
                content = read_regular(shard)
                if len(content) != ref["size_bytes"] or sha256(content) != ref["sha256"]:
                    raise ValueError("Shard checksum mismatch")
                archive.writestr(str(shard.relative_to(root)), content)
            archive.writestr(str(path.relative_to(root)), payload)
            completed[path.parent.name] = len(manifest["completed_trial_ids"])

        patterns = (
            "data/posttrain/*.json", "data/posttrain/*.jsonl",
            "configs/**/*.yaml", "src/interp/**/*.py", "scripts/*.sh", "scripts/*.py",
            "pyproject.toml", "requirements-tested.txt", "README.md",
            "runs/posttrain/*report*/*.json", "runs/posttrain/*report*/*.csv",
            "runs/posttrain/*report*/*.md", "runs/posttrain/logs/environment-*.txt",
        )
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if any(parent.is_symlink() for parent in (path, *path.parents)):
                    raise ValueError("Linked backup input refused")
                archive.writestr(str(path.relative_to(root)), read_regular(path))
        archive.writestr("BACKUP.json", json.dumps({
            "captured_at": datetime.now(UTC).isoformat(),
            "completed_trials": completed,
            "scope": "Published trials only; excludes in-flight generation, caches, secrets and weights",
        }, indent=2))
    return completed


def stream_digest(handle):
    """Verify an uploaded archive without loading the whole file into memory."""
    result = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        result.update(chunk)
    return result.hexdigest()


class DriveBackup:
    """One daemon upload worker; errors remain visible through ``status``.

    ``remote_dir`` must be an existing writable folder in the filesystem's own
    path syntax. Use the same authenticated filesystem object as the Drive UI.
    No automatic remote deletion occurs. Full snapshots consume space over time.
    """

    def __init__(self, fs, root, remote_dir, interval=600):
        if interval < 60:
            raise ValueError("Use an interval of at least 60 seconds")
        self.fs, self.root, self.remote_dir = fs, Path(root), remote_dir.rstrip("/")
        self.interval = interval
        self.stop_event = threading.Event()
        self.status = {"state": "not_started"}
        self.thread = None

    def start(self):
        """Start once and return immediately so the notebook remains usable."""
        if self.thread is not None:
            raise RuntimeError("This worker has already been started")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        """Stop after any current upload finishes; never kill the GPU process."""
        self.stop_event.set()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self.once()
            except Exception as error:
                # Do not expose OAuth tokens or raw HTTP response details.
                self.status = {**self.status, "state": "error", "error_type": type(error).__name__}
            if self.stop_event.wait(self.interval):
                break

    def once(self):
        """Upload a unique archive, read it back, then publish its checksum receipt."""
        self.status = {**self.status, "state": "uploading"}
        name = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12] + ".zip"
        remote = self.remote_dir + "/" + name
        with tempfile.TemporaryDirectory(prefix="posttrain-backup-") as temporary:
            local = Path(temporary) / name
            counts = capture(self.root, local)
            with local.open("rb") as handle:
                expected = stream_digest(handle)
            self.fs.put_file(str(local), remote)
            with self.fs.open(remote, "rb") as handle:
                if stream_digest(handle) != expected:
                    raise ValueError("Remote archive checksum mismatch")
            # A ZIP without this receipt is not an acknowledged successful backup.
            with self.fs.open(remote + ".sha256", "wb") as handle:
                handle.write(f"{expected}  {name}\n".encode())
        self.status = {"state": "verified", "archive": remote,
                       "verified_at": datetime.now(UTC).isoformat(),
                       "completed_trials": counts, "sha256": expected}