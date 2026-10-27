"""Offline recovery regressions: real snapshots/OS locks, mocked installs and GPU runtime.

All archives, destinations and environments are disposable fixtures. No external
downloads, package installs, archived code imports or model allocation are allowed.
"""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from filelock import FileLock, Timeout

from interp.posttrain.plan import job_identity, lock_study, schedule
from interp.posttrain.schemas import Attempt, Job, StudyMetadata
from interp.posttrain.storage import StudyStore, read_snapshot
from interp.runtime import utc_now

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("recovery", ROOT / "scripts/recover_posttraining.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def receipt(path):
    path.with_name(path.name + ".sha256").write_text(f"{recovery.file_digest(path)}  {path.name}\n")


@pytest.fixture
def saved(tmp_path):
    """Create an actual zero-success published job and its source/plan backup."""
    source = tmp_path / "source"
    source.mkdir()
    plan = lock_study(ROOT / "configs/posttrain/smoke.yaml",
                      source / "data/posttrain/math-fp32-plan.json")
    job = Job(checkpoint="base")
    metadata = StudyMetadata(
        run_id=job_identity(plan, job), created_at=utc_now(), plan=plan, job=job,
        model_profile={"kind": "offline-recovery-fixture"},
        runtime_profile={"python": "3.13.11", "threads": 20, "interop_threads": 20},
    )
    directory = source / "runs/posttrain/math-fp32-base"
    with StudyStore(directory).writer(metadata) as store:
        store.append(Attempt(
            run_id=metadata.run_id, trial=next(schedule(plan, job)), attempt_id=uuid4(),
            started_at=utc_now(), finished_at=utc_now(), status="failed",
            error_type="ArtificialOfflineFailure",
        ))
    shutil.copytree(ROOT / "src/interp", source / "src/interp", ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc",
    ))
    (source / "BACKUP.json").write_text(json.dumps({
        "completed_trials": {"math-fp32-base": 0}, "captured_at": "fixture-only",
    }))
    # Executable recovery material is deliberately included but must never be extracted.
    (source / "scripts").mkdir()
    (source / "scripts/evil.py").write_text("raise RuntimeError('ARCHIVED CODE EXECUTED')")
    (directory / "records-orphan.jsonl").write_text("unpublished, must not restore")
    (source / "runs/posttrain/logs").mkdir()
    (source / "runs/posttrain/logs/environment-test.txt").write_text("not live state")
    archive = tmp_path / "snapshot.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.name != ".writer.lock":
                output.write(path, path.relative_to(source).as_posix())
    receipt(archive)
    return archive, source


def stage(saved, tmp_path):
    directory = tmp_path / "stage"
    directory.mkdir()
    return directory, recovery.stage_archive(saved[0], directory)


def test_real_restore_no_code_no_torch_idempotent(saved, tmp_path, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "datasets"}:
            raise AssertionError("Verification imported a heavyweight library")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    staged, info = stage(saved, tmp_path)
    assert info["archived_source_sha256"] == recovery.source_hash(ROOT)
    checked = recovery.validate_snapshots(staged, ROOT)
    assert checked["runs"]["math-fp32-base"]["missing"] > 0
    destination = tmp_path / "destination"
    recovery.restore(staged, destination, info["files"])
    recovery.restore(staged, destination, info["files"])
    snapshot = read_snapshot(destination / "runs/posttrain/math-fp32-base")
    assert len(snapshot.records) == 1 and snapshot.records[0].status == "failed"
    assert not (destination / "scripts").exists()
    assert not (destination / "src").exists()
    assert not (destination / "BACKUP.json").exists()
    assert not (destination / "runs/posttrain/logs").exists()
    assert not list(destination.rglob("*orphan*"))
    for name in info["files"]:
        assert (destination / name).read_bytes() == (saved[1] / name).read_bytes()


@pytest.mark.parametrize("damage", ["missing", "bad-hash", "wrong-name", "multiple", "uppercase"])
def test_receipt_strict(saved, tmp_path, damage):
    archive, _ = saved
    path = archive.with_name(archive.name + ".sha256")
    if damage == "missing":
        path.unlink()
    else:
        text = path.read_text()
        path.write_text({
            "bad-hash": "0" * 64 + f"  {archive.name}\n",
            "wrong-name": text.replace(archive.name, "other.zip"),
            "multiple": text + text,
            "uppercase": text.upper(),
        }[damage])
    with pytest.raises(ValueError, match="receipt|filename"):
        stage(saved, tmp_path)
    assert not list((tmp_path / "stage").iterdir())


@pytest.mark.parametrize("name", [
    "../escape", "/absolute", "src/../../escape", "C:/drive", "data\\escape",
    "data//posttrain/file.json", "./file", "data/./file", "data/\x01file",
])
def test_unsafe_paths_even_ignored_code(saved, tmp_path, name):
    with zipfile.ZipFile(saved[0], "a") as output:
        output.writestr(name, "bad")
    receipt(saved[0])
    with pytest.raises(ValueError, match="Unsafe"):
        stage(saved, tmp_path)


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFSOCK])
def test_special_zip_members(saved, tmp_path, kind):
    item = zipfile.ZipInfo("ignored-special")
    item.create_system = 3
    item.external_attr = (kind | 0o644) << 16
    with zipfile.ZipFile(saved[0], "a") as output:
        output.writestr(item, "target")
    receipt(saved[0])
    with pytest.raises(ValueError, match="special"):
        stage(saved, tmp_path)


@pytest.mark.parametrize("collision", ["exact", "directory", "parent"])
def test_duplicate_collisions(saved, tmp_path, collision):
    with zipfile.ZipFile(saved[0], "a") as output:
        if collision == "exact":
            with pytest.warns(UserWarning, match="Duplicate"):
                output.writestr("BACKUP.json", "{}")
        elif collision == "directory":
            output.writestr("scripts/evil.py/", "")
        else:
            output.writestr("scripts/evil.py/child", "bad")
    receipt(saved[0])
    with pytest.raises(ValueError, match="duplicate|collision"):
        stage(saved, tmp_path)


@pytest.mark.parametrize("cap", ["MAX_MEMBER", "MAX_TOTAL", "MAX_ENTRIES", "MAX_ARCHIVE"])
def test_bounded_archive(saved, tmp_path, monkeypatch, cap):
    monkeypatch.setattr(recovery, cap, 1)
    with pytest.raises(ValueError, match="bound|size|entries|regular"):
        stage(saved, tmp_path)


def test_compression_bomb(tmp_path):
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("ignored.txt", b"0" * 2_000_000)
    with zipfile.ZipFile(archive) as source, pytest.raises(ValueError, match="ratio"):
        recovery.safe_members(source)


def test_existing_conflict_prevents_all_writes(saved, tmp_path):
    staged, info = stage(saved, tmp_path)
    root = tmp_path / "destination"
    conflict = root / "runs/posttrain/math-fp32-base/manifest.json"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("newer live output")
    with pytest.raises(ValueError, match="Existing content differs"):
        recovery.restore(staged, root, info["files"])
    assert conflict.read_text() == "newer live output"
    assert not (root / "data").exists()


def test_linked_destinations_rejected(saved, tmp_path):
    staged, info = stage(saved, tmp_path)
    root = tmp_path / "destination"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked"):
        recovery.restore(staged, root, info["files"])
    assert not list(outside.iterdir())


def test_partial_publication_rerun(saved, tmp_path, monkeypatch):
    staged, info = stage(saved, tmp_path)
    root = tmp_path / "destination"
    link = os.link

    def interrupted(source, target):
        if target.name == "manifest.json":
            raise OSError("simulated reset")
        link(source, target)

    monkeypatch.setattr(os, "link", interrupted)
    with pytest.raises(OSError, match="reset"):
        recovery.restore(staged, root, info["files"])
    assert not list(root.rglob("manifest.json"))
    assert list(root.rglob("metadata.json"))
    monkeypatch.setattr(os, "link", link)
    recovery.restore(staged, root, info["files"])
    read_snapshot(root / "runs/posttrain/math-fp32-base")


def test_filelock_compatibility_both_directions(tmp_path):
    relative = "runs/posttrain/job/.writer.lock"
    with recovery.os_lock(tmp_path, relative):
        with pytest.raises(Timeout), FileLock(str(tmp_path / relative), timeout=0):
            pytest.fail("FileLock bypassed recovery flock")
    with FileLock(str(tmp_path / relative), timeout=0):
        with pytest.raises(ValueError, match="Active"), recovery.os_lock(tmp_path, relative):
            pytest.fail("Recovery bypassed FileLock")


@pytest.mark.parametrize("relative", ["runs/posttrain/.pipeline.lock", "runs/posttrain/job/.writer.lock"])
def test_active_writer_blocks_before_staging_or_install(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(recovery.platform, "system", lambda: "Linux")
    setup = Mock(side_effect=AssertionError("must not install"))
    staging = Mock(side_effect=AssertionError("must not stage"))
    monkeypatch.setattr(recovery, "setup_environment", setup)
    monkeypatch.setattr(recovery, "stage_archive", staging)
    args = argparse.Namespace(archive=tmp_path / "missing.zip", check_only=False, run=False)
    with recovery.os_lock(tmp_path, relative):
        with pytest.raises(ValueError, match="Active"):
            recovery.run_recovery(args, tmp_path)
    setup.assert_not_called()
    staging.assert_not_called()


def test_python_mismatch_before_any_environment_write(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "python_info", lambda _: ("3.11.14", "prefix", "base"))
    subprocess_run = Mock(side_effect=AssertionError("must not install or create venv"))
    monkeypatch.setattr(recovery.subprocess, "run", subprocess_run)
    with pytest.raises(ValueError, match="Archived Python"):
        recovery.setup_environment(tmp_path, [{"python": "3.13.11"}])
    assert not (tmp_path / ".venv").exists()


def test_existing_satisfied_venv_not_reinstalled(tmp_path, monkeypatch):
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").touch()
    (venv / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    monkeypatch.setattr(recovery, "python_info", lambda _: ("3.13.11", str(venv), "base"))
    installed = {**recovery.PINS, "interp-research": "0.1.0"}
    run = Mock(return_value=SimpleNamespace(stdout=json.dumps([
        installed, str(tmp_path / "src/interp/__init__.py"),
    ])))
    monkeypatch.setattr(recovery.subprocess, "run", run)
    assert recovery.setup_environment(tmp_path, [{"python": "3.13.11"}]) == venv / "bin/python"
    assert all("install" not in call.args[0] and "venv" not in call.args[0]
               for call in run.call_args_list)


def test_new_venv_installs_pins_only_inside_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "python_info", lambda _: ("3.13.11", "prefix", "base"))
    run = Mock(return_value=SimpleNamespace(stdout="null"))
    monkeypatch.setattr(recovery.subprocess, "run", run)
    python = recovery.setup_environment(tmp_path, [{"python": "3.13.11"}])
    commands = [call.args[0] for call in run.call_args_list]
    installs = [command for command in commands if "install" in command]
    assert len(installs) == 1
    assert installs[0][0] == str(python) == str(tmp_path / ".venv/bin/python")
    assert "https://pypi.org/simple" in installs[0]
    assert str(tmp_path) + "[posttrain,dev,notebooks]" in installs[0]
    assert all(f"{name}=={version}" in installs[0] for name, version in recovery.PINS.items())


def test_runtime_diff_never_adjusts_threads(monkeypatch):
    current = {"python": "3.13.11", "threads": 4, "interop_threads": 4, "device": "cuda:0"}
    fake_torch = SimpleNamespace(
        device=lambda _: SimpleNamespace(type="cuda"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("interp.runtime.runtime_profile", lambda _: dict(current))
    old = {**current, "device": "cuda", "threads": 20, "interop_threads": 20}
    result = recovery.validate_runtime({"runs": {
        "rl": {"missing": 1014, "runtime_profile": old},
        "base": {"missing": 0, "runtime_profile": old},
    }}, "cuda:0")
    assert set(result["runtime_differences"]) == {"rl"}
    assert result["runtime_differences"]["rl"]["threads"] == {"archived": 20, "current": 4}
    assert current["threads"] == 4


def test_shell_from_other_directory():
    result = subprocess.run(["bash", str(ROOT / "scripts/recover_posttraining.sh"), "--help"],
                            cwd="/tmp", env={**os.environ, "RECOVERY_PYTHON": sys.executable},
                            capture_output=True, text=True, check=True)
    assert "--check-only" in result.stdout and "--run" in result.stdout


@pytest.mark.parametrize("mode", ["check", "prepare", "run", "mismatch"])
def test_orchestration_locks_restore_and_handoff(saved, tmp_path, monkeypatch, mode):
    """Only environment/GPU/subprocess seams are mocked; stage and restore are real."""
    root = tmp_path / "clone"
    shutil.copytree(ROOT / "src/interp", root / "src/interp")
    monkeypatch.setattr(recovery.platform, "system", lambda: "Linux")
    args = argparse.Namespace(archive=saved[0], check_only=mode == "check", run=mode in {
        "run", "mismatch",
    })

    def setup(*_):
        for name in (".pipeline.lock", "math-fp32-base/.writer.lock"):
            with pytest.raises(Timeout), FileLock(str(root / "runs/posttrain" / name), timeout=0):
                pytest.fail("Setup did not hold locks")
        return root / ".venv/bin/python"

    setup_mock = Mock(side_effect=setup)
    monkeypatch.setattr(recovery, "setup_environment", setup_mock)

    def validate(python, trusted_root, directory, runtime=False, pipeline=False):
        result = recovery.validate_snapshots(directory, ROOT)
        if runtime:
            result.update(current_runtime={"threads": 4}, runtime_differences=(
                {"math-fp32-base": {"threads": {"archived": 20, "current": 4}}}
                if mode == "mismatch" else {}
            ))
        return result

    monkeypatch.setattr(recovery, "child_validation", validate)

    def launch(command, *, cwd, check, env):
        assert command == ["bash", "scripts/run_posttraining.sh"]
        assert cwd == root and env["PYTHON"] == str(root / ".venv/bin/python")
        # Both layers must be released, not inherited in a form that deadlocks the old runner.
        for name in (".pipeline.lock", "math-fp32-base/.writer.lock"):
            with FileLock(str(root / "runs/posttrain" / name), timeout=0):
                pass

    launched = Mock(side_effect=launch)
    monkeypatch.setattr(recovery.subprocess, "run", launched)
    if mode == "mismatch":
        with pytest.raises(ValueError, match="Runtime mismatch"):
            recovery.run_recovery(args, root)
    else:
        recovery.run_recovery(args, root)
    if mode == "check":
        setup_mock.assert_not_called()
        assert not (root / "data").exists()
        assert not list(root.rglob("metadata.json"))
    else:
        read_snapshot(root / "runs/posttrain/math-fp32-base")
        audit = next((root / "runs/posttrain/logs").glob("recovery-*.json"))
        assert json.loads(audit.read_text())["archived_source_sha256"] == recovery.EXPECTED_SOURCE
    assert launched.call_count == (1 if mode == "run" else 0)


def rewrite(saved):
    """Rebuild a fixture ZIP after modifying its inert source artifacts."""
    archive, source = saved
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.name != ".writer.lock":
                output.write(path, path.relative_to(source).as_posix())
    receipt(archive)


@pytest.mark.parametrize("artifact", ["metadata.json", "shard", "source", "shard-reference"])
def test_archive_internal_integrity(saved, tmp_path, artifact):
    directory = saved[1] / "runs/posttrain/math-fp32-base"
    if artifact == "metadata.json":
        (directory / artifact).write_text("{}")
    elif artifact == "shard":
        next(directory.glob("records-000000-*.jsonl")).write_text("corrupt\n")
    elif artifact == "source":
        (saved[1] / "src/interp/__init__.py").write_text("# changed source\n")
    else:
        path = directory / "manifest.json"
        content = json.loads(path.read_text())
        content["shards"][0]["name"] = "../../outside"
        path.write_text(json.dumps(content))
    rewrite(saved)
    with pytest.raises(ValueError, match="checksum|source|reference"):
        stage(saved, tmp_path)


def test_valid_hashes_do_not_replace_full_snapshot_validation(saved, tmp_path):
    path = saved[1] / "runs/posttrain/math-fp32-base/manifest.json"
    content = json.loads(path.read_text())
    content["completed_trial_ids"] = ["a" * 64]
    path.write_text(json.dumps(content))
    (saved[1] / "BACKUP.json").write_text(json.dumps({
        "completed_trials": {"math-fp32-base": 1},
    }))
    rewrite(saved)
    staged, _ = stage(saved, tmp_path)
    with pytest.raises(ValueError, match="completion"):
        recovery.validate_snapshots(staged, ROOT)


def test_late_destination_creation_is_never_overwritten(saved, tmp_path, monkeypatch):
    staged, info = stage(saved, tmp_path)
    root = tmp_path / "destination"
    link = os.link

    def racing_link(source, target):
        target.write_text("concurrent output")
        link(source, target)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        recovery.restore(staged, root, info["files"])
    existing = [path for path in root.rglob("*") if path.is_file()]
    assert len(existing) == 1 and existing[0].read_text() == "concurrent output"