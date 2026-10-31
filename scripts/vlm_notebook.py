"""Stdlib-only marimo controls; no Torch/model imports or scientific implementation.

Opening a notebook is read-only. An explicitly consumed click launches the same
CLI in a detached subprocess with durable logs and a completion receipt. No
thread, timer, reactive refresh or kernel restart implicitly starts/retries work.
This helper deliberately lives outside both science-source fingerprint trees.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path


def _now():
    return datetime.now(UTC).isoformat()


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError("Missing, linked or oversized notebook job receipt.")
    return json.loads(path.read_text())


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    fd, name = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _alive(pid):
    if type(pid) is not int or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Fail closed: do not start a competing worker.
    return True


def _lock_held(path):
    """Read-only liveness for a worker or surviving command, including after a crash."""
    if not path.exists():
        return False
    if path.is_symlink():
        raise ValueError("Notebook lifetime locks cannot be symlinks.")
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


class ClickGate:
    """Consume a ticket before action, including failed actions; no reactive retries."""

    def __init__(self):
        self.seen = {}

    def perform(self, ticket, action):
        if ticket in self.seen:
            result, error = self.seen[ticket]
            if error is not None:
                raise RuntimeError("This click already failed; submit a fresh action to retry.")
            return result
        self.seen[ticket] = (None, "consumed")
        result = action()
        self.seen[ticket] = (result, None)
        return result


class NotebookPilot:
    """Launch one explicit notebook-owned action; inspection uses no GPU/model code."""

    def __init__(self, repo: Path):
        self.repo = repo.expanduser().resolve()
        if not (self.repo / "scripts" / "run_vlm_pilot.sh").is_file():
            raise ValueError("Open this notebook inside the science_of_post_training checkout.")
        self.jobs = self.repo / "runs" / "vlm-notebook" / "jobs"
        self.helper = self.repo / "scripts" / "vlm_notebook.py"
        self.launcher = self.repo / "scripts" / "run_vlm_pilot.sh"
        self.python = self.repo / ".venv-vlm" / "bin" / "python"

    def _path(self, value):
        path = Path(value).expanduser()
        path = path if path.is_absolute() else self.repo / path
        if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Notebook artifact paths cannot traverse symlinks or parent components.")
        path = path.absolute()
        if not path.is_relative_to(self.repo):
            raise ValueError("Notebook artifacts must remain inside the checkout.")
        return path

    def commands(self, settings):
        """Construct shell-free CLI arguments; neither construct nor preview executes."""
        mode, action = settings["mode"], settings["action"]
        if mode not in ("smoke", "pilot"):
            raise ValueError("Choose smoke or pilot.")
        base = ["bash", str(self.launcher)]
        env = {"PYTHON_BOOTSTRAP": sys.executable, "VLM_ENV_DIR": str(self.python.parent.parent)}
        index = settings.get("torch_index", "")
        if index not in ("", "https://download.pytorch.org/whl/cu130",
                         "https://download.pytorch.org/whl/cu128"):
            raise ValueError("Only the listed public PyTorch indexes are accepted; no secret URLs.")
        if index:
            env["TORCH_INDEX_URL"] = index
        if mode == "smoke":
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if action == "setup":
            return [[*base, "setup"]], env, None
        if not self.python.is_file():
            raise ValueError("Run Setup first; VLM dependencies are not installed in the notebook kernel.")
        plan = self._path(settings.get("plan") or f"runs/vlm-{mode}/plan.json")
        config = self.repo / "configs" / "vlm" / ("smoke.yaml" if mode == "smoke" else "pilot.yaml")
        data = self.repo / "data" / f"vlm-{mode}"
        if action == "prepare":
            commands = []
            if not (data / "dataset.json").exists():
                command = [*base, "data", str(data), "--contact-sheet",
                           str(self.repo / "runs" / f"vlm-{mode}-scenes.png")]
                if mode == "smoke":
                    command += ["--engineering", "1", "--discovery", "1",
                                "--pilot-validation", "0", "--confirmation", "0"]
                commands.append(command)
            commands.append([*base, "preview", str(config)])
            if not plan.exists():
                commands.append([*base, "lock", str(config), "--output", str(plan)])
            else:
                commands.append([str(self.python), "-c", "from pathlib import Path; "
                    "from vlm_pilot.plan import read_plan; import sys; "
                    "p=read_plan(Path(sys.argv[1]),check_source=True); print(p.plan_id)", str(plan)])
            return commands, env, str(plan)
        if action == "smoke":
            return [[*base, "smoke"]], env, None
        if action == "analyze":
            runs = [self._path(value.strip()) for value in settings.get("analysis_runs", "").splitlines()
                    if value.strip()]
            if not runs:
                raise ValueError("Supply saved trial-run directories, one per line, for paired analysis.")
            output = self._path(settings.get("output") or
                                f"runs/vlm-{mode}/report-{uuid.uuid4().hex}")
            args = [*base, "analyze", *(str(path) for path in runs), "--output", str(output)]
            if settings.get("allow_incomplete", False):
                args.append("--allow-incomplete")
            return [args], env, str(output)
        if not plan.is_file():
            raise ValueError("Prepare data/plan before running checkpoint jobs.")
        if action == "baseline-all":
            if mode != "pilot" or not settings.get("allow_downloads", False):
                raise ValueError("All-checkpoint baseline needs pilot mode and explicit model-download opt-in.")
            env.update(DEVICE=settings["device"], VLM_PLAN=str(plan))
            return [[*base, "baseline"]], env, "runs/vlm-pilot/*-baseline-discovery"
        if action == "freeze":
            output = self._path(settings.get("output") or f"runs/vlm-{mode}/protocol.json")
            args = [*base, "freeze", str(plan), "--note", settings.get("note", ""), "--output", str(output)]
            for size in ("fixture", "2b", "8b"):
                sites = settings.get("sites_" + size, "").strip()
                if sites:
                    args += ["--sites-" + size, sites]
            if not any("--sites-" in arg for arg in args) or not settings.get("note", "").strip():
                raise ValueError("Freeze actual discovery sites and a rationale; example sites are not findings.")
            args += ["--projections", settings.get("projections", "q_proj,k_proj")]
            if settings.get("include_norms", False):
                args.append("--include-norms")
            return [args], env, str(output)
        checkpoint = settings["checkpoint"]
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", checkpoint) is None:
            raise ValueError("Invalid checkpoint name.")
        if action == "audit-weights":
            after = settings.get("source_checkpoint", "").strip()
            if not after:
                raise ValueError("Supply the second checkpoint for the actual vision/projector weight audit.")
            output = self._path(settings.get("output") or
                                f"runs/vlm-{mode}/components-{uuid.uuid4().hex}.json")
            args = [*base, "audit-weights", str(plan), "--before", checkpoint, "--after", after,
                    "--output", str(output)]
            if settings.get("allow_downloads", False):
                args.append("--allow-downloads")
            return [args], env, str(output)
        if action not in ("preflight", "run"):
            raise ValueError("Unknown notebook action.")
        args = [*base, action, str(plan), "--checkpoint", checkpoint, "--device", settings["device"]]
        if settings.get("allow_downloads", False):
            args.append("--allow-downloads")
        if settings.get("sites", "").strip():
            args += ["--sites", settings["sites"].strip()]
        if action == "preflight":
            output = self._path(settings.get("output") or f"runs/vlm-{mode}/p0-{checkpoint}-{uuid.uuid4().hex}.json")
            return [[*args, "--output", str(output)]], env, str(output)
        phase, split = settings["phase"], settings["split"]
        if phase not in ("baseline", "routing", "weights") or split not in (
            "engineering", "discovery", "pilot_validation", "confirmation"
        ):
            raise ValueError("Invalid phase or split.")
        if phase != "baseline" and not settings.get("sites", "").strip():
            raise ValueError("Routing/weight runs need explicit discovery-selected layer:KV-group sites.")
        args += ["--phase", phase, "--split", split,
                 "--projections", settings.get("projections", "q_proj,k_proj")]
        source = settings.get("source_checkpoint", "").strip()
        if phase == "weights":
            if not source:
                raise ValueError("Choose the exact adjacent weight-source checkpoint.")
            args += ["--source-checkpoint", source]
        protocol = settings.get("protocol", "").strip()
        if protocol:
            args += ["--protocol", str(self._path(protocol))]
        for key, flag in (("include_norms", "--include-norms"), ("generate", "--generate"),
                          ("save_captures", "--save-captures")):
            if settings.get(key, False):
                args.append(flag)
        profile = settings.get("prompt_profile", "common")
        args += ["--prompt-profile", profile]
        operation = {k: settings.get(k) for k in ("phase", "split", "checkpoint", "sites",
            "source_checkpoint", "projections", "include_norms", "generate", "save_captures", "prompt_profile")}
        tag = hashlib.sha256(json.dumps(operation, sort_keys=True).encode()).hexdigest()[:10]
        output = self._path(settings.get("output") or f"runs/vlm-{mode}/{checkpoint}-{phase}-{split}-{tag}")
        budget = settings.get("max_new_trials", 12)
        if budget is not None:
            if type(budget) is not int or budget < 0:
                raise ValueError("Budget must be a nonnegative integer, or unchecked for a full run.")
            args += ["--max-new-trials", str(budget)]
        return [[*args, "--output", str(output)]], env, str(output)

    def launch(self, settings):
        if settings.get("confirmed") is not True:
            raise ValueError("Confirm this explicit action; no work is launched automatically.")
        commands, environment, output = self.commands(settings)
        self.jobs.mkdir(parents=True, exist_ok=True)
        with (self.jobs / ".launch.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if any(row["active_lock"] or row["state"] in ("starting", "running")
                   for row in self.monitor(limit=None)):
                raise RuntimeError("A notebook-owned action is still active; refresh its status first.")
            job_id = uuid.uuid4().hex
            job = self.jobs / job_id
            job.mkdir()
            request = {"job_id": job_id, "repo": str(self.repo), "commands": commands,
                       "environment": environment, "output": output, "action": settings["action"]}
            _write(job / "request.json", request)
            created = _now()
            _write(job / "launch.json", {"pid": None, "created_at": created})
            with (job / ".active.lock").open("a") as lifetime, (job / "stdout.log").open("wb") as log:
                fcntl.flock(lifetime.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                process = subprocess.Popen([sys.executable, "-u", str(self.helper), "--worker", str(job),
                    "--lock-fd", str(lifetime.fileno())], cwd=self.repo, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lifetime.fileno(),))
                _write(job / "launch.json", {"pid": process.pid, "created_at": created})
        return {"job_id": job_id, "pid": process.pid, "state": "launched", "output": output,
                "log": str(job / "stdout.log"), "automatic_retry": False}

    def monitor(self, limit=10):
        """Read only receipts/process liveness; never import models, write or retry."""
        if not self.jobs.exists():
            return []
        rows = []
        for job in self.jobs.iterdir():
            if not job.is_dir() or job.is_symlink() or not (job / "launch.json").is_file():
                continue
            launched, request = _read(job / "launch.json"), _read(job / "request.json")
            status = _read(job / "status.json") if (job / "status.json").exists() else {"state": "starting"}
            alive = _alive(launched["pid"])
            lifetime_exists = (job / ".active.lock").exists()
            active = _lock_held(job / ".active.lock")
            state = status["state"]
            if state in ("starting", "running") and not (active if lifetime_exists else alive):
                state = "interrupted_or_unacknowledged"
            rows.append({"job_id": job.name, "action": request["action"], "state": state,
                         "pid": launched["pid"], "created_at": launched["created_at"],
                         "exit_code": status.get("exit_code"), "output": request["output"],
                         "log": str(job / "stdout.log"), "active_lock": active,
                         "driver_alive": alive})
        ordered = sorted(rows, key=lambda row: row["created_at"], reverse=True)
        return ordered[:limit] if limit is not None else ordered

    def log_tail(self, job_id, maximum=16000):
        if re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
            raise ValueError("Invalid job receipt identifier.")
        path = self.jobs / job_id / "stdout.log"
        if path.is_symlink() or not path.is_file():
            return "No log published yet."
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - maximum))
            return handle.read(maximum).decode("utf-8", errors="replace")

    def inspect_saved(self, output):
        if not self.python.is_file():
            raise ValueError("Setup is required for model-free saved-result inspection.")
        completed = subprocess.run([str(self.python), "-m", "vlm_pilot", "inspect", str(self._path(output))],
                                  cwd=self.repo, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "Saved-result verification failed.")
        return json.loads(completed.stdout)


def worker(job: Path, lock_fd=None):
    """Detached driver publishes the actual CLI exit code, not a launch-only success."""
    request = _read(job / "request.json")
    lifetime = None
    if lock_fd is None:
        lifetime = (job / ".active.lock").open("a")
        fcntl.flock(lifetime.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd = lifetime.fileno()
    environment = os.environ.copy()
    environment.update(request["environment"])
    status = {"state": "running", "pid": os.getpid(), "started_at": _now(), "exit_code": None}
    _write(job / "status.json", status)
    exit_code = 1
    try:
        for index, argv in enumerate(request["commands"]):
            _write(job / "status.json", status | {"command_index": index})
            print(f"Executing step {index + 1}/{len(request['commands'])}", flush=True)
            result = subprocess.run(argv, cwd=request["repo"], env=environment,
                                    stdin=subprocess.DEVNULL, check=False, pass_fds=(lock_fd,))
            exit_code = result.returncode
            if exit_code:
                break
    except BaseException as error:
        exit_code = 1
        print(f"Notebook action failed ({type(error).__name__}); inspect published pilot records.", flush=True)
    finally:
        _write(job / "status.json", status | {"state": "completed" if exit_code == 0 else "failed",
               "exit_code": exit_code, "finished_at": _now()})
        if lifetime is not None:
            lifetime.close()
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) not in (3, 5) or sys.argv[1] != "--worker":
        raise SystemExit("Open notebooks/vlm_pilot.py in marimo; this is its subprocess helper.")
    inherited_fd = int(sys.argv[4]) if len(sys.argv) == 5 and sys.argv[3] == "--lock-fd" else None
    raise SystemExit(worker(Path(sys.argv[2]), inherited_fd))