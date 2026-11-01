"""Offline checks for explicit marimo actions; never install dependencies or run GPU jobs."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[2] / "scripts" / "vlm_notebook.py"
SPEC = importlib.util.spec_from_file_location("vlm_notebook_test_controls", HELPER)
controls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls)


@pytest.fixture
def controller(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_vlm_pilot.sh").write_text("# offline test only")
    (scripts / "vlm_notebook.py").write_text("# offline test only")
    python = tmp_path / ".venv-vlm" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("# offline fake interpreter")
    plan = tmp_path / "runs" / "vlm-smoke" / "plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text("{}")
    return controls.NotebookPilot(tmp_path)


def settings(**extra):
    return {"action": "run", "mode": "smoke", "checkpoint": "fixture-mpo",
            "phase": "baseline", "split": "discovery", "device": "cpu",
            "confirmed": True, "max_new_trials": 12, **extra}


def test_consumed_click_is_once_only_even_after_failure():
    gate, calls = controls.ClickGate(), []
    assert gate.perform("ticket", lambda: calls.append(1) or "result") == "result"
    assert gate.perform("ticket", lambda: calls.append(2)) == "result"
    assert calls == [1]
    with pytest.raises(ValueError):
        gate.perform("failure", lambda: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(RuntimeError, match="already failed"):
        gate.perform("failure", lambda: calls.append(3))
    assert calls == [1]


def test_command_preview_is_shell_free_bounded_and_resume_path_stable(controller):
    first, env, output = controller.commands(settings())
    second, _, resumed = controller.commands(settings(max_new_trials=24))
    assert first[0][:2] == ["bash", str(controller.launcher)]
    assert "--max-new-trials" in first[0] and "--allow-downloads" not in first[0]
    assert output == resumed and first != second
    assert env["VLM_ENV_DIR"].endswith(".venv-vlm") and env["OMP_NUM_THREADS"] == "1"
    assert not controller.jobs.exists()


def test_no_implicit_research_or_secret_install_index(controller):
    with pytest.raises(ValueError, match="Confirm"):
        controller.launch(settings(confirmed=False))
    with pytest.raises(ValueError, match="public"):
        controller.commands(settings(action="setup", torch_index="https://token@example.invalid"))
    with pytest.raises(ValueError, match="explicit"):
        controller.commands(settings(phase="routing"))
    with pytest.raises(ValueError, match="adjacent"):
        controller.commands(settings(phase="weights", sites="0:0"))
    with pytest.raises(ValueError, match="checkout|symlinks"):
        controller.commands(settings(output="/tmp/outside-run"))


def test_prepare_reuses_files_and_does_not_launch(controller):
    commands, _, plan = controller.commands(settings(action="prepare"))
    assert "data" in commands[0] and "preview" in commands[1]
    assert "lock" not in [token for command in commands for token in command]
    assert commands[-1][0] == str(controller.python)
    assert plan.endswith("plan.json") and not controller.jobs.exists()


def test_analysis_and_component_audit_are_explicit_cli_jobs(controller):
    runs = "runs/vlm-smoke/mpo-baseline\nruns/vlm-smoke/gspo-baseline"
    commands, _, output = controller.commands(settings(action="analyze", analysis_runs=runs,
                                                       allow_incomplete=True))
    assert commands[0][2] == "analyze" and "--allow-incomplete" in commands[0]
    assert "--allow-downloads" not in commands[0] and output.startswith(str(controller.repo))
    commands, _, _ = controller.commands(settings(action="audit-weights", source_checkpoint="fixture-gspo"))
    assert commands[0][2] == "audit-weights" and "--after" in commands[0]
    assert not controller.jobs.exists()


def test_worker_publishes_real_completion_and_stops_after_failure(controller, monkeypatch):
    job = controller.jobs / ("a" * 32)
    request = {"commands": [["first"], ["second"], ["third"]], "repo": str(controller.repo),
               "environment": {}, "action": "test", "output": None}
    controls._write(job / "request.json", request)
    calls = []

    def execute(argv, **kwargs):
        calls.append(argv)
        return type("Result", (), {"returncode": 7 if argv == ["second"] else 0})()

    monkeypatch.setattr(controls.subprocess, "run", execute)
    assert controls.worker(job) == 7
    assert calls == [["first"], ["second"]]
    status = json.loads((job / "status.json").read_text())
    assert status["state"] == "failed" and status["exit_code"] == 7


def test_monitor_never_restarts_dead_job(controller, monkeypatch):
    job = controller.jobs / ("b" * 32)
    controls._write(job / "request.json", {"action": "run", "output": "run-dir"})
    controls._write(job / "launch.json", {"pid": 999999, "created_at": "2026-10-31"})
    controls._write(job / "status.json", {"state": "running"})
    (job / "stdout.log").write_text("verified log")
    monkeypatch.setattr(controls, "_alive", lambda pid: False)
    row = controller.monitor()[0]
    assert row["state"] == "interrupted_or_unacknowledged"
    assert controller.log_tail(row["job_id"]) == "verified log"
    assert json.loads((job / "status.json").read_text())["state"] == "running"


def test_driver_exception_after_success_cannot_acknowledge_completion(controller, monkeypatch):
    job = controller.jobs / ("c" * 32)
    controls._write(job / "request.json", {"commands": [["first"], ["fails"]],
                    "repo": str(controller.repo), "environment": {}, "action": "test", "output": None})

    def execute(argv, **kwargs):
        if argv == ["fails"]:
            raise OSError("Controlled fixture exception")
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(controls.subprocess, "run", execute)
    assert controls.worker(job) == 1
    assert controls._read(job / "status.json")["state"] == "failed"


def test_actual_subprocess_driver_records_completion_without_models(controller):
    job = controller.jobs / ("d" * 32)
    controls._write(job / "request.json", {
        "commands": [[sys.executable, "-c", "print('offline driver verified')"]],
        "repo": str(controller.repo), "environment": {}, "action": "test", "output": None,
    })
    result = controls.subprocess.run([sys.executable, str(HELPER), "--worker", str(job)],
                                    capture_output=True, text=True, check=False)
    assert result.returncode == 0 and "offline driver verified" in result.stdout
    status = controls._read(job / "status.json")
    assert status["state"] == "completed" and status["exit_code"] == 0
    assert not controls._lock_held(job / ".active.lock")


def test_surviving_child_lock_prevents_relaunch_after_driver_crash(controller, monkeypatch):
    job = controller.jobs / ("e" * 32)
    controls._write(job / "request.json", {"action": "run", "output": "run-dir"})
    controls._write(job / "launch.json", {"pid": 999999, "created_at": "2026-10-31"})
    controls._write(job / "status.json", {"state": "running"})
    monkeypatch.setattr(controls, "_alive", lambda pid: False)
    with (job / ".active.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        row = controller.monitor()[0]
        assert row["state"] == "running" and row["active_lock"] and not row["driver_alive"]
        with pytest.raises(RuntimeError, match="still active"):
            controller.launch(settings())
    assert controller.monitor()[0]["state"] == "interrupted_or_unacknowledged"


def test_notebook_import_does_not_import_pilot_runtime_or_launch(monkeypatch):
    before = set(sys.modules)
    notebook = HELPER.parent.parent / "notebooks" / "vlm_pilot.py"
    spec = importlib.util.spec_from_file_location("vlm_notebook_import_test", notebook)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "app")
    assert "vlm_pilot.runtime" not in set(sys.modules) - before


def test_isolated_setup_includes_notebook_dependencies_for_smoke():
    import tomllib

    root = HELPER.parent.parent
    script = (root / "scripts" / "run_vlm_pilot.sh").read_text()
    assert "pip install -e '.[vlm,notebooks,dev]'" in script
    with (root / "pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["optional-dependencies"]["notebooks"]
    assert any(item.startswith("marimo") for item in dependencies)