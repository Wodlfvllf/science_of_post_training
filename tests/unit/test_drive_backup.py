"""Offline checks for consistent live-run archives and remote acknowledgment."""

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "drive_backup", Path(__file__).resolve().parents[2] / "scripts/backup_posttraining.py",
)
backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup)


@pytest.fixture
def saved(tmp_path):
    """A tiny checksummed publication boundary; no model dependency."""
    directory = tmp_path / "runs/posttrain/math-fp32-base"
    directory.mkdir(parents=True)
    metadata = b'{"plan": {}}'
    content = b'{"trial": "one"}\n'
    digest = hashlib.sha256(content).hexdigest()
    name = f"records-000000-{digest}.jsonl"
    (directory / "metadata.json").write_bytes(metadata)
    (directory / name).write_bytes(content)
    manifest = {
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "shards": [{"name": name, "sha256": digest, "size_bytes": len(content)}],
        "completed_trial_ids": ["one"],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "records-orphan.jsonl").write_text("not published")
    return tmp_path, directory, manifest


def test_capture_retains_boundary_during_publication(saved, monkeypatch):
    root, directory, manifest = saved
    original = backup.read_regular

    def advancing_read(path):
        if path.name == "metadata.json":
            # A new publication happens after capture of the original manifest.
            (directory / "manifest.json").write_text('{"new_boundary": true}')
        return original(path)

    monkeypatch.setattr(backup, "read_regular", advancing_read)
    archive = root / "snapshot.zip"
    assert backup.capture(root, archive) == {"math-fp32-base": 1}
    with zipfile.ZipFile(archive) as saved_zip:
        prefix = "runs/posttrain/math-fp32-base/"
        assert json.loads(saved_zip.read(prefix + "manifest.json")) == manifest
        assert prefix + "records-orphan.jsonl" not in saved_zip.namelist()


def test_capture_rejects_corrupt_shard(saved):
    root, directory, manifest = saved
    (directory / manifest["shards"][0]["name"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        backup.capture(root, root / "snapshot.zip")


@pytest.mark.parametrize("corrupt", [False, True])
def test_upload_receipt_requires_remote_verification(saved, corrupt):
    root, _, _ = saved

    class MemoryFS:
        def __init__(self):
            self.files = {}

        def put_file(self, local, remote):
            self.files[remote] = b"corrupt" if corrupt else Path(local).read_bytes()

        def open(self, path, mode):
            if mode == "rb":
                return io.BytesIO(self.files[path])
            files = self.files

            class Writer(io.BytesIO):
                def close(self):
                    files[path] = self.getvalue()
                    super().close()

            return Writer()

    fs = MemoryFS()
    worker = backup.DriveBackup(fs, root, "backups")
    if corrupt:
        with pytest.raises(ValueError, match="checksum"):
            worker.once()
        assert not any(name.endswith(".sha256") for name in fs.files)
    else:
        worker.once()
        assert worker.status["state"] == "verified"
        assert worker.status["archive"] + ".sha256" in fs.files