"""Offline VLM boundary and fsspec acknowledgment tests; no model or GPU calls."""

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "vlm_backup_tests", Path(__file__).resolve().parents[2] / "scripts/backup_vlm.py")
backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def seal(path, value):
    payload = backup.canonical(value)
    write(path, payload)
    return {"path": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload)}


@pytest.fixture
def saved(tmp_path):
    # Use the stdlib-only geometry builder, never generate_dataset/Pillow/render.
    spec = importlib.util.spec_from_file_location(
        "vlm_backup_fixture_data", Path(__file__).resolve().parents[2] / "src/vlm_pilot/data.py")
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    family = data._family(1, "engineering", 0)
    for image in family["images"]:
        payload = b"original codec bytes: " + image["image_id"].encode()
        image["sha256"] = hashlib.sha256(payload).hexdigest()
        write(tmp_path / "data/vlm-pilot" / image["relative_path"], payload)
    dataset = {"schema_version": 1, "name": "reading-or-routing-v1", "seed": 1,
               "image_size": 448, "families": [family],
               "split_counts": dict(zip(backup._SPLITS, (1, 0, 0, 0), strict=True))}
    seal(tmp_path / "data/vlm-pilot/dataset.json", dataset)
    run = tmp_path / "runs/vlm-pilot/mpo-baseline"
    metadata = seal(run / "metadata.json", {"schema_version": 1, "run_id": "run-1",
                                          "plan": {"dataset": dataset}})
    capture = b"selected safetensors bytes"
    artifact = {"path": "captures/trial-1.safetensors", "size_bytes": len(capture),
                "sha256": hashlib.sha256(capture).hexdigest()}
    write(run / artifact["path"], capture)
    row = {"run_id": "run-1", "trial_id": "trial-1", "attempt_id": "attempt-1",
           "status": "completed", "artifacts": [artifact]}
    payload = backup.canonical(row)
    sha = hashlib.sha256(payload).hexdigest()
    ref = {"path": f"records/trial-1-{sha}.json", "trial_id": "trial-1",
           "sha256": sha, "size_bytes": len(payload)}
    write(run / ref["path"], payload)
    manifest = {"schema_version": 1, "run_id": "run-1", "metadata_sha256": metadata["sha256"],
                "records": [ref], "completed": 1, "updated_at": "boundary-1"}
    seal(run / "manifest.json", manifest)
    for path in ("records/orphan.json", "captures/orphan.safetensors", "weights/model.safetensors"):
        write(run / path, b"not published")
    for path in ("src/vlm_pilot/io.py", "scripts/vlm_notebook.py", "configs/vlm/pilot.yaml",
                 "notebooks/vlm_pilot.py", "pyproject.toml", "plan-loader-v2.json",
                 "runs/vlm-pilot/plan-gpu-uuid.json", "runs/vlm-pilot/protocol.json",
                 "runs/vlm-pilot/report-test/REPORT.md", "runs/vlm-pilot/report-test/cells.csv"):
        write(tmp_path / path, b"{}")
    seal(tmp_path / "runs/vlm-pilot/report-test/report.json", {"cells": []})
    for path in (".git/config", ".env", "runs/posttrain/unrelated.json",
                 "runs/vlm-pilot/cache/tokenizer.json", "runs/vlm-pilot/environment.json",
                 "runs/vlm-pilot/hf-cache/config.json", "runs/vlm-pilot/config.json",
                 "runs/vlm-notebook/jobs/job/request.json"):
        write(tmp_path / path, b"SECRET_DO_NOT_COPY")
    write(tmp_path / "runs/vlm-notebook/jobs/job/stdout.log", b"optional log")
    return tmp_path, run, manifest, row


class MemoryFS:
    def __init__(self, corrupt=None):
        self.files, self.directories, self.created, self.puts = {}, set(), [], []
        self.corrupt = corrupt

    def exists(self, path):
        return path in self.directories

    def makedirs(self, path, exist_ok=False):
        assert exist_ok and path == backup.REMOTE_DIR
        self.directories.add(path)
        self.created.append(path)

    def put_file(self, local, remote):
        self.puts.append(remote)
        payload = Path(local).read_bytes()
        self.files[remote] = b"x" + payload[1:] if self.corrupt == "zip" else payload

    def open(self, path, mode):
        if mode == "rb":
            payload = self.files[path]
            if self.corrupt == "receipt" and path.endswith(".sha256"):
                payload += b"corruption"
            return io.BytesIO(payload)
        files = self.files

        class Writer(io.BytesIO):
            def close(self):
                files[path] = self.getvalue()
                super().close()

        return Writer()

    def rm(self, *args, **kwargs):
        pytest.fail("Remote deletion is forbidden")


def test_exact_snapshot_and_manifest_boundary(saved, monkeypatch):
    root, run, manifest, _ = saved
    original = backup._read

    def advance(path, *args):
        if path.name == "metadata.json":
            seal(run / "manifest.json", {"new_boundary": True})
        return original(path, *args)

    monkeypatch.setattr(backup, "_read", advance)
    result = backup.capture(root, root / "snapshot.zip")
    with zipfile.ZipFile(root / "snapshot.zip") as archive:
        assert json.loads(archive.read("runs/vlm-pilot/mpo-baseline/manifest.json")) == manifest
        inventory = json.loads(archive.read("BACKUP.json"))
        assert result["files"] == len(archive.namelist()) == len(set(archive.namelist()))
        assert inventory["completed_trials"] == {"runs/vlm-pilot/mpo-baseline": 1}
        for entry in inventory["entries"]:
            content = archive.read(entry["path"])
            assert len(content) == entry["size_bytes"]
            assert hashlib.sha256(content).hexdigest() == entry["sha256"]
            assert b"SECRET_DO_NOT_COPY" not in content
        assert not any("orphan" in n or "/weights/" in n or "/cache/" in n
                       or n.endswith("stdout.log") for n in archive.namelist())
        assert "src/vlm_pilot/io.py" in archive.namelist()
        assert "runs/vlm-pilot/report-test/report.json" in archive.namelist()
    assert result["sha256"] == hashlib.sha256((root / "snapshot.zip").read_bytes()).hexdigest()


@pytest.mark.parametrize("target", ["metadata", "record", "capture", "image"])
def test_corrupt_referenced_bytes_fail(saved, target):
    root, run, manifest, row = saved
    paths = {"metadata": run / "metadata.json", "record": run / manifest["records"][0]["path"],
             "capture": run / row["artifacts"][0]["path"],
             "image": next((root / "data/vlm-pilot/images").glob("*.png"))}
    paths[target].write_bytes(b"{}")
    with pytest.raises(ValueError):
        backup.capture(root, root / "bad.zip")


@pytest.mark.parametrize("mutation", ["traversal", "duplicate", "size", "completed", "trial"])
def test_invalid_manifest_references(saved, mutation):
    root, run, manifest, _ = saved
    if mutation == "traversal":
        manifest["records"][0]["path"] = "../outside.json"
    elif mutation == "duplicate":
        manifest["records"] *= 2
    elif mutation == "size":
        manifest["records"][0]["size_bytes"] += 1
    elif mutation == "trial":
        manifest["records"][0]["trial_id"] = "other"
    else:
        manifest["completed"] = 2
    seal(run / "manifest.json", manifest)
    with pytest.raises(ValueError):
        backup.capture(root, root / "bad.zip")


@pytest.mark.parametrize("mutation", ["capture-traversal", "capture-nested", "capture-duplicate",
                                    "capture-missing", "capture-size", "run", "status", "artifacts"])
def test_resealed_rows_and_capture_reference_errors(saved, mutation):
    root, run, manifest, row = saved
    artifact = row["artifacts"][0]
    if mutation == "capture-traversal":
        artifact["path"] = "../private.safetensors"
    elif mutation == "capture-nested":
        artifact["path"] = "captures/subdir/private.safetensors"
    elif mutation == "capture-duplicate":
        row["artifacts"] *= 2
    elif mutation == "capture-missing":
        (run / artifact["path"]).unlink()
    elif mutation == "capture-size":
        artifact["size_bytes"] = True
    elif mutation == "artifacts":
        row["artifacts"] = {}
    else:
        row[mutation if mutation == "status" else "run_id"] = "invalid"
    payload = backup.canonical(row)
    sha = hashlib.sha256(payload).hexdigest()
    ref = manifest["records"][0]
    ref.update(path=f"records/trial-1-{sha}.json", sha256=sha, size_bytes=len(payload))
    write(run / ref["path"], payload)
    seal(run / "manifest.json", manifest)
    with pytest.raises(ValueError):
        backup.capture(root, root / "bad.zip")


@pytest.mark.parametrize("mutation", ["path", "duplicate", "schema", "seed", "case"])
def test_dataset_schema_and_paths(saved, mutation):
    root, _, _, _ = saved
    path = root / "data/vlm-pilot/dataset.json"
    dataset = json.loads(path.read_bytes())
    images = dataset["families"][0]["images"]
    if mutation == "path":
        images[0]["relative_path"] = "../private.png"
    elif mutation == "duplicate":
        images[1] = images[0]
    elif mutation == "schema":
        dataset["schema_version"] = True
    elif mutation == "seed":
        dataset["seed"] = True
    else:
        dataset["families"][0]["cases"][0]["image_id"] = "unreferenced-image"
    seal(path, dataset)
    with pytest.raises(ValueError):
        backup.capture(root, root / "bad.zip")


@pytest.mark.parametrize("which", ["root", "parent", "record", "run", "source"])
def test_symlinks_rejected_without_resolving(saved, which):
    root, run, manifest, _ = saved
    if which in ("root", "parent"):
        link = root.parent / (root.name + "-link")
        link.symlink_to(root, target_is_directory=True)
        selected = link if which == "root" else link / "child"
        with pytest.raises(ValueError, match="symlink"):
            backup.capture(selected, root / "bad.zip")
        return
    path = {"record": run / manifest["records"][0]["path"],
            "run": run, "source": root / "src/vlm_pilot/io.py"}[which]
    moved = path.with_name(path.name + "-original")
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=which == "run")
    with pytest.raises(ValueError, match="symlink"):
        backup.capture(root, root / "bad.zip")


@pytest.mark.parametrize("limit,value", [("MAX_TOTAL_BYTES", 100), ("MAX_MEMBERS", 3),
                                        ("MAX_FILE_BYTES", 100)])
def test_limits_fail_closed(saved, monkeypatch, limit, value):
    root, _, _, _ = saved
    monkeypatch.setattr(backup, limit, value)
    with pytest.raises(ValueError):
        backup.capture(root, root / "bad.zip")


def test_logs_opt_in_bounded_and_no_requests(saved, monkeypatch):
    root, _, _, _ = saved
    monkeypatch.setattr(backup, "MAX_STDOUT_BYTES", 4)
    backup.capture(root, root / "logs.zip", include_logs=True)
    with zipfile.ZipFile(root / "logs.zip") as archive:
        assert archive.read("runs/vlm-notebook/jobs/job/stdout.log") == b"opti"
        assert not any(n.endswith("request.json") for n in archive.namelist())


@pytest.mark.parametrize("corrupt", [None, "zip", "receipt"])
def test_remote_acknowledgment_and_sanitized_errors(saved, corrupt):
    root, _, _, _ = saved
    fs = MemoryFS(corrupt)
    worker = backup.VLMDriveBackup(fs, root, backup.REMOTE_DIR)
    assert worker.thread is None and worker.status == {"state": "not_started"}
    if corrupt:
        with pytest.raises(RuntimeError, match="ValueError"):
            worker.once()
        assert worker.status == {"state": "error", "error_type": "ValueError"}
        if corrupt == "zip":
            assert not any(n.endswith(".sha256") for n in fs.files)
    else:
        result = worker.once()
        assert result["state"] == "verified" and result["files"] > 5
        name = result["archive"].split("/")[-1]
        assert fs.files[result["archive"] + ".sha256"] == f"{result['sha256']}  {name}\n".encode()
        result["completed_trials"].clear()
        assert worker.status["completed_trials"]
        worker.once()
        assert len(set(fs.puts)) == 2 and fs.created == [backup.REMOTE_DIR]


def test_provider_secrets_are_not_exposed(saved, monkeypatch):
    root, _, _, _ = saved
    fs = MemoryFS()

    def fail(local, remote):
        raise OSError("SECRET_SIGNED_URL_OR_OAUTH_RESPONSE")

    monkeypatch.setattr(fs, "put_file", fail)
    worker = backup.VLMDriveBackup(fs, root, backup.REMOTE_DIR)
    with pytest.raises(RuntimeError) as caught:
        worker.once()
    assert "SECRET" not in str(caught.value) and caught.value.__suppress_context__
    assert worker.status == {"state": "error", "error_type": "OSError"}


def test_no_overlap_start_once_and_stop_after_upload(saved, monkeypatch):
    root, _, _, _ = saved
    worker = backup.VLMDriveBackup(MemoryFS(), root, backup.REMOTE_DIR)
    with worker._upload_lock, pytest.raises(RuntimeError, match="active"):
        worker.once()
    calls = []

    def put(local, remote):
        calls.append(remote)
        worker.stop()
        MemoryFS.put_file(worker.fs, local, remote)

    monkeypatch.setattr(worker.fs, "put_file", put)
    worker.start()
    worker.thread.join(timeout=10)
    assert not worker.thread.is_alive() and worker.status["state"] == "verified"
    assert len(calls) == 1 and worker.stop_event.is_set()
    with pytest.raises(RuntimeError, match="already"):
        worker.start()
    with pytest.raises(ValueError, match="designated"):
        backup.VLMDriveBackup(worker.fs, root, "science-of-post-training-backups")