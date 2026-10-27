"""Recover published posttraining artifacts, never executable archive material.

Public entry point: recover_posttraining.sh ZIP [--check-only | --run]. Bootstrap,
ZIP verification, locks and copying use only the standard library. Separate trusted
interpreter processes validate project schemas and (only after restore) the GPU
runtime. Nothing imports archive code or changes source, configs or saved metadata.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from uuid import uuid4

EXPECTED_SOURCE = "f9f24e86e8e979e4b59056c6f8cd286a197722cf1aeb9b8b0a2ae0c0e67a827f"
PINS = {
    "torch": "2.11.0", "transformers": "4.57.6", "tokenizers": "0.22.2",
    "safetensors": "0.8.0", "huggingface-hub": "0.36.2", "datasets": "4.8.5",
    "fsspec": "2026.2.0", "pydantic": "2.13.4", "PyYAML": "6.0.3", "filelock": "3.31.0",
}
MAX_MEMBER = 64 * 1024**2
MAX_TOTAL = 2 * 1024**3
MAX_ARCHIVE = 512 * 1024**2
MAX_ENTRIES = 100_000
NAME = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
TARGETS = {f"{study}-fp32-{checkpoint}" for study in ("math", "repair")
           for checkpoint in ("base", "rl", "sft", "dpo")}


def digest(payload):
    """Hash exact artifact bytes, not parsed/reformatted JSON."""
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    """Match interp.data.canonical_bytes without importing the project."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular(path, maximum=MAX_MEMBER):
    """Refuse links/devices/FIFOs before reading any untrusted local input."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise ValueError(f"Not a bounded regular file: {path}")


def checked_path(root, relative):
    """Reject linked or non-directory ancestors inside the owned checkout."""
    path = root
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(p in (".", "..") for p in parts):
        raise ValueError("Invalid destination path")
    for index, part in enumerate(parts):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Linked destination refused: {path}")
        if path.exists() and index < len(parts) - 1 and not path.is_dir():
            raise ValueError(f"Non-directory destination ancestor: {path}")
    return path


@contextmanager
def os_lock(root, relative):
    """Use BSD flock, the same OS lock as Linux flock and Unix filelock.FileLock.

    Never delete lock files: replacing their inode could admit a second writer.
    The only writes before lock acquisition are the required lock parents/files.
    """
    path = checked_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Lock is not a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Active pipeline/writer; stop it before recovery: {relative}") from error
        yield
    finally:
        os.close(fd)


def source_hash(root):
    entries = []
    for path in sorted((root / "src/interp").rglob("*.py")):
        checked_path(root, path.relative_to(root))
        regular(path)
        entries.append((path.relative_to(root / "src/interp").as_posix(), file_digest(path)))
    return digest(canonical(entries))


def safe_members(archive):
    """Check EVERY member, including ignored code; bound CPU/disk amplification."""
    items = archive.infolist()
    if len(items) > MAX_ENTRIES:
        raise ValueError("Too many ZIP entries")
    members, total = {}, 0
    for item in items:
        name = item.filename
        normalized = name.rstrip("/")
        parts = normalized.split("/")
        kind = stat.S_IFMT(item.external_attr >> 16)
        if (item.orig_filename != name or not name or "\\" in name
                or any(p in ("", ".", "..") or ":" in p
                       or any(ord(c) < 32 for c in p) for p in parts)
                or normalized in members or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
                or (kind == stat.S_IFDIR and not item.is_dir())
                or (kind == stat.S_IFREG and item.is_dir()) or item.flag_bits & 1
                or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise ValueError(f"Unsafe/duplicate/special ZIP entry: {name!r}")
        if (item.file_size > MAX_MEMBER or item.file_size > max(1, item.compress_size) * 1000
                or (item.is_dir() and item.file_size)):
            raise ValueError("ZIP member exceeds size/ratio bounds")
        total += item.file_size
        members[normalized] = item
    if total > MAX_TOTAL:
        raise ValueError("ZIP exceeds total size bound")
    for name in members:
        for parent in Path(name).parents:
            if parent.as_posix() in members and not members[parent.as_posix()].is_dir():
                raise ValueError("ZIP file/directory collision")
    return members


def stage_archive(path, destination):
    """Verify the adjacent receipt and extract only data + published run artifacts.

    The already-open ZIP is hashed then read on that same handle. Source files are
    hashed as inert bytes for provenance, but are never extracted. CRC verification
    streams ignored members too. Schema/grade validation follows in read_snapshot.
    """
    receipt = path.with_name(path.name + ".sha256")
    if not receipt.exists():
        raise ValueError("Missing adjacent ZIP.sha256 receipt; download it from the notebook first")
    regular(receipt, 4096)
    match = re.fullmatch(rb"([0-9a-f]{64})  ([^/\\\r\n]+\.zip)\n?", receipt.read_bytes())
    if not match or match[2] != path.name.encode():
        raise ValueError("Malformed receipt or mismatched ZIP filename")
    regular(path, MAX_ARCHIVE)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != match[1].decode():
            raise ValueError("ZIP SHA256 receipt mismatch")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            members = safe_members(archive)
            for item in members.values():
                with archive.open(item) as content:
                    while content.read(1024 * 1024):
                        pass  # Bounded streaming checks the CRC, even for ignored files.

            def read(name):
                if name not in members or members[name].is_dir():
                    raise ValueError(f"Missing regular archive artifact: {name}")
                return archive.read(members[name])

            source = digest(canonical([
                (name.removeprefix("src/interp/"), digest(read(name)))
                for name in sorted(members)
                if name.startswith("src/interp/") and name.endswith(".py")
            ]))
            backup = json.loads(read("BACKUP.json"))
            selected = {name for name, item in members.items() if not item.is_dir()
                        and re.fullmatch(rf"data/posttrain/{NAME}\.(json|jsonl)", name)}
            runs = {}
            for name in sorted(members):
                if not re.fullmatch(rf"runs/posttrain/{NAME}/manifest\.json", name):
                    continue
                parent = name.rsplit("/", 1)[0]
                manifest = json.loads(read(name))
                metadata_bytes = read(parent + "/metadata.json")
                metadata = json.loads(metadata_bytes)
                if (digest(metadata_bytes) != manifest["metadata_sha256"]
                        or metadata["run_id"] != manifest["run_id"]
                        or metadata["plan"]["source_sha256"] != source):
                    raise ValueError("Archive metadata/source/identity checksum mismatch")
                selected.update((name, parent + "/metadata.json"))
                for index, ref in enumerate(manifest["shards"]):
                    if (not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])
                            or ref["name"] != f"records-{index:06d}-{ref['sha256']}.jsonl"):
                        raise ValueError("Invalid manifest shard reference")
                    shard_name = parent + "/" + ref["name"]
                    payload = read(shard_name)
                    if len(payload) != ref["size_bytes"] or digest(payload) != ref["sha256"]:
                        raise ValueError("Archive shard checksum/size mismatch")
                    selected.add(shard_name)
                run = parent.rsplit("/", 1)[1]
                if backup["completed_trials"].get(run) != len(manifest["completed_trial_ids"]):
                    raise ValueError("Backup completion boundary mismatch")
                runs[run] = metadata
            if not runs or set(runs) != set(backup["completed_trials"]):
                raise ValueError("Missing manifests or inconsistent backup run list")
            # Exclusive creation in a fresh private directory; no extractall or code writes.
            for name in sorted(selected):
                target = checked_path(destination, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    output.write(read(name))
    return {"archive_sha256": actual, "archived_source_sha256": source,
            "captured_at": backup.get("captured_at"), "runs": runs, "files": sorted(selected)}


def check_conflicts(stage, root, names):
    """Preflight ALL destinations before publishing anything; never roll back live data."""
    for name in names:
        target = checked_path(root, name)
        if target.exists():
            regular(target)
            if file_digest(target) != file_digest(stage / name):
                raise ValueError(f"Existing content differs; nothing overwritten: {name}. "
                                 "Keep the live run; use a separate clean checkout for this backup.")


def restore(stage, root, names):
    """Publish immutable files first, manifests last, using atomic no-clobber links.

    Interrupted restores may leave harmless temp files or unpublished immutable
    artifacts. Rerunning accepts identical content and completes publication.
    """
    check_conflicts(stage, root, names)
    for name in sorted(names, key=lambda n: (n.endswith("/manifest.json"), n)):
        target = checked_path(root, name)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".recovery-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as output, (stage / name).open("rb") as source:
                shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
            os.link(temporary, target)  # Atomic and refuses even a late-arriving destination.
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            os.unlink(temporary)


def python_info(python):
    """Probe only standard-library facts before any large dependency installation."""
    result = subprocess.run([str(python), "-I", "-c",
                             "import json,sys; print(json.dumps([sys.version.split()[0], "
                             "sys.prefix, sys.base_prefix]))"],
                            check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def check_python(version, profiles):
    if tuple(map(int, version.split(".")[:2])) < (3, 11):
        raise ValueError("Python >=3.11 required")
    expected = {profile["python"] for profile in profiles}
    if expected and expected != {version}:
        raise ValueError(f"Archived Python {sorted(expected)} != available {version}; "
                         "no dependencies installed. Use a matching Python (RECOVERY_PYTHON), "
                         "or --check-only for offline inspection.")


def setup_environment(root, profiles):
    """Preserve a working isolated .venv; never use or repair global site packages."""
    venv = checked_path(root, ".venv")
    python = venv / "bin/python"
    if venv.exists():
        # A venv's interpreter is normally a symlink; its prefix must still be ours.
        if not venv.is_dir() or not (venv / "pyvenv.cfg").is_file() or not python.exists():
            raise ValueError("Existing .venv is incomplete; preserve/inspect it manually, not deleted")
        version, prefix, base = python_info(python)
        if Path(prefix).resolve() != venv or prefix == base:
            raise ValueError("Existing .venv interpreter is not isolated to this checkout")
        if re.search(r"include-system-site-packages\s*=\s*true", (venv / "pyvenv.cfg").read_text(), re.I):
            raise ValueError("Existing .venv exposes global packages; use an isolated recovery venv")
    else:
        version, _, _ = python_info(sys.executable)
    check_python(version, profiles)
    for profile in profiles:
        for name, expected in profile.get("packages", {}).items():
            if name in PINS and PINS[name] != expected:
                raise ValueError(f"Archived dependency differs from recovery pins: {name}")
    if not venv.exists():
        subprocess.run([sys.executable, "-I", "-m", "venv", str(venv)], check=True)
    # Skip a satisfied install entirely on reruns. Metadata inspection does not import Torch.
    probe = """import importlib.metadata as m, json
names = json.loads(__import__('sys').argv[1])
try:
    versions = {n: m.version(n) for n in names}
    import interp
    print(json.dumps([versions, interp.__file__]))
except (m.PackageNotFoundError, ImportError):
    print('null')
"""
    names = [*PINS, "interp-research", "pytest", "ruff", "marimo", "plotly"]
    result = subprocess.run([str(python), "-I", "-c", probe, json.dumps(names)],
                            check=True, text=True, capture_output=True)
    installed = json.loads(result.stdout)
    satisfied = (installed and all(installed[0].get(n) == v for n, v in PINS.items())
                 and installed[0]["interp-research"] == "0.1.0"
                 and Path(installed[1]).resolve() == root / "src/interp/__init__.py")
    if not satisfied:
        subprocess.run([str(python), "-I", "-m", "pip", "--isolated", "install",
                        "--index-url", "https://pypi.org/simple", "-e",
                        str(root) + "[posttrain,dev,notebooks]",
                        *(f"{name}=={version}" for name, version in PINS.items())], check=True)
    subprocess.run([str(python), "-I", "-m", "pip", "--isolated", "check"], check=True)
    return python


def validate_snapshots(directory, root, pipeline=False):
    """Use the installed trusted project's full schema/grade validator, without Torch."""
    import interp
    from interp.config import source_digest
    from interp.posttrain.plan import job_identity, read_plan, schedule
    from interp.posttrain.schemas import Job
    from interp.posttrain.storage import read_snapshot

    if Path(interp.__file__).resolve() != root / "src/interp/__init__.py":
        raise ValueError("Installed interp is not this checkout; install the trusted editable project")
    plans = [read_plan(path) for path in (directory / "data/posttrain").glob("*plan.json")]
    pipeline_plans = {}
    if pipeline:
        # The old orchestrator must not relock lost science or select a different plan.
        for study in ("math", "repair"):
            plan_path = directory / f"data/posttrain/{study}-fp32-plan.json"
            if not plan_path.exists():
                if list((directory / "runs/posttrain").glob(f"{study}-fp32-*/metadata.json")):
                    raise ValueError(f"Restore the original standalone {study} plan")
                continue  # A never-started study is created by the existing pipeline, not recovery.
            pipeline_plans[study] = read_plan(plan_path, check_source=True)
            for checkpoint in ("base", "rl", "sft", "dpo"):
                job_identity(pipeline_plans[study], Job(checkpoint=checkpoint))
    summaries = {}
    for path in sorted((directory / "runs/posttrain").glob("*/manifest.json")):
        snapshot = read_snapshot(path.parent)
        metadata = snapshot.metadata
        if metadata.plan.source_sha256 != source_digest():
            raise ValueError("Installed source differs from saved plan")
        if metadata.plan not in plans:
            raise ValueError(f"Restore the original standalone plan for {path.parent.name}")
        if pipeline and path.parent.name in TARGETS:
            study, _, checkpoint = path.parent.name.split("-")
            expected_job = Job(checkpoint=checkpoint)
            if metadata.run_id != job_identity(pipeline_plans[study], expected_job):
                raise ValueError("Pipeline destination does not match its standalone plan/job")
        planned = sum(1 for _ in schedule(metadata.plan, metadata.job))
        completed = len(snapshot.manifest.completed_trial_ids)
        summaries[path.parent.name] = {
            "completed": completed, "planned": planned, "missing": planned - completed,
            "runtime_profile": metadata.runtime_profile,
        }
    if not summaries:
        raise ValueError("No validated snapshots")
    return {"installed_source_sha256": source_digest(), "runs": summaries}


def validate_runtime(summary, device):
    """Compare the exact runtime before model allocation; never set thread/backend flags."""
    import torch

    from interp.runtime import runtime_profile

    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("A working CUDA device is required for preparation/resume")
    current = runtime_profile(selected)
    current["device"] = selected.type  # Same normalization as posttrain.runner.
    differences = {}
    for name, run in summary["runs"].items():
        if not run["missing"]:
            continue
        saved = run["runtime_profile"]
        diff = {key: {"archived": saved.get(key), "current": current.get(key)}
                for key in sorted(saved.keys() | current.keys()) if saved.get(key) != current.get(key)}
        if diff:
            differences[name] = diff
    return {"current_runtime": current, "runtime_differences": differences}


def child_validation(python, root, directory, runtime=False, pipeline=False):
    """Run installed validators in the selected venv, not the bootstrap/global environment."""
    command = [str(python), "-I", str(Path(__file__).resolve()), "--validate",
               str(directory), "--root", str(root)]
    if runtime:
        command += ["--runtime", "--device", os.environ.get("DEVICE", "cuda:0")]
    if pipeline:
        command += ["--pipeline"]
    result = subprocess.run(command, cwd=root, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def run_recovery(args, root):
    """Hold pipeline AND all known individual writer locks throughout setup/publication."""
    if not args.check_only and platform.system() != "Linux":
        raise ValueError("Preparation/resume requires Linux; use --check-only for offline audit")
    if sys.version_info < (3, 11):  # noqa: UP036 -- bootstrap may precede project installation
        raise ValueError("Bootstrap Python >=3.11 required")
    archive = Path(os.path.abspath(args.archive))  # Do not resolve away a ZIP symlink.
    with ExitStack() as locks:
        locks.enter_context(os_lock(root, "runs/posttrain/.pipeline.lock"))
        locked = set()

        def lock_runs(names):
            for name in sorted(set(names) - locked):
                locks.enter_context(os_lock(root, f"runs/posttrain/{name}/.writer.lock"))
                locked.add(name)

        # Include independent jobs, including writers that have not published a manifest yet.
        existing = [p.name for p in (root / "runs/posttrain").iterdir()
                    if p.name != "logs" and (p.is_dir() or p.is_symlink())]
        lock_runs(existing)
        if not args.check_only:
            lock_runs(TARGETS)
        with tempfile.TemporaryDirectory(prefix="posttrain-recovery-") as temporary:
            stage = Path(temporary)
            info = stage_archive(archive, stage)
            lock_runs(info["runs"])
            local_source = source_hash(root)
            if local_source != info["archived_source_sha256"] or local_source != EXPECTED_SOURCE:
                raise ValueError(f"Source mismatch: checkout={local_source}, "
                                 f"archive={info['archived_source_sha256']}, expected={EXPECTED_SOURCE}")
            print(f"Verified ZIP {info['archive_sha256']}; checkout/archive source {local_source}",
                  flush=True)
            check_conflicts(stage, root, info["files"])
            if args.check_only:
                python = root / ".venv/bin/python"
                if not python.exists():
                    python = Path(sys.executable)
            else:
                profiles = [m["runtime_profile"] for m in info["runs"].values()]
                print("Host warning: 32 GiB RAM may be insufficient for 7.3B FP32 loading; "
                      "96 GB VRAM does not guarantee host capacity. No thread settings changed.",
                      flush=True)
                python = setup_environment(root, profiles)
            validated = child_validation(python, root, stage, pipeline=args.run)
            print(json.dumps(validated, indent=2), flush=True)
            if args.check_only:
                print("CHECK ONLY: no installs/restoration/GPU checks; lock files may have been created.")
                return
            restore(stage, root, info["files"])
            # Validate live outputs as well, including additional runs not present in the ZIP.
            validated = child_validation(python, root, root, runtime=True, pipeline=args.run)
            report = {k: info[k] for k in ("archive_sha256", "archived_source_sha256", "captured_at")}
            report.update(local_source_sha256=local_source, **validated)
            if source_hash(root) != local_source:
                raise ValueError("Checkout source changed during recovery; do not resume")
            logs = checked_path(root, "runs/posttrain/logs")
            logs.mkdir(parents=True, exist_ok=True)
            audit = logs / f"recovery-{uuid4().hex}.json"
            with audit.open("x") as output:
                json.dump(report, output, indent=2)
            print(json.dumps(report, indent=2), flush=True)
            print(f"Validation receipt: {audit}", flush=True)
            if validated["runtime_differences"]:
                raise ValueError("Runtime mismatch: restored artifacts preserved; resume blocked. "
                                 "Do not edit metadata or force threads to bypass this check.")
        # Close individual locks and outer pipeline lock before the old script reacquires it.
        # A competing pipeline can win this handoff; its normal nonblocking lock then fails safely.
    if args.run:
        print("Validated; handing off to the existing pipeline (downloads/model preflights enabled).",
              flush=True)
        subprocess.run(["bash", "scripts/run_posttraining.sh"], cwd=root, check=True,
                       env={**os.environ, "PYTHON": str(python)})
    else:
        print("Prepared and runtime-validated; no experiment launched. Rerun with --run to continue.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, help="ZIP with adjacent ZIP.sha256 receipt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Offline audit; no installs or restore")
    mode.add_argument("--run", action="store_true", help="Prepare, validate, then run existing pipeline")
    parser.add_argument("--validate", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--runtime", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pipeline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.validate:
            result = validate_snapshots(args.validate, args.root, pipeline=args.pipeline)
            if args.runtime:
                result.update(validate_runtime(result, args.device))
            print(json.dumps(result))
        else:
            if not args.archive:
                parser.error("provide a ZIP path (and download its adjacent .sha256 receipt)")
            run_recovery(args, Path(__file__).resolve().parents[1])
    except subprocess.CalledProcessError as error:
        # Local validator failures are useful, but never dump installer/network credentials.
        if "--validate" in error.cmd and error.stderr:
            print(error.stderr.strip(), file=sys.stderr)
        print("Recovery stopped: subprocess failed; saved data/venv preserved. Fix and rerun.",
              file=sys.stderr)
        return 1
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"Recovery stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())