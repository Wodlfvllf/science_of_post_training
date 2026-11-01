"""Bounded, independent VLM snapshots using an already authenticated fsspec FS.

Importing starts nothing and imports no GPU, Pillow, fsspec or science package.
Only runs/vlm-pilot publications and explicit recovery inputs are eligible. All
manifests are read first; immutable references must validate against those exact
boundaries. Reports are best effort, not authoritative state. Logs require an
explicit opt-in because even sanitized CLI output can contain sensitive URLs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

REMOTE_DIR = "science-of-post-training-backups/vlm-posttraining"
MAX_TOTAL_BYTES, MAX_FILE_BYTES, MAX_MEMBERS = 2 * 1024**3, 64 * 1024**2, 30000
MAX_STDOUT_BYTES = 8 * 1024**2
_EXCLUDED = {"records", "captures", "weights", "cache", "caches", "modelcache",
             "models", "env", "envs", "environment", "secrets", "credentials"}
_SPLITS = ("engineering", "discovery", "pilot_validation", "confirmation")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pairs(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("Duplicate JSON keys")
    return value


def _json(payload, *, sealed=False):
    value = json.loads(payload, object_pairs_hook=_pairs)
    encoded = canonical(value)  # Reject NaN, Infinity and overflowing floats.
    if sealed and payload != encoded:
        raise ValueError("Noncanonical sealed JSON")
    return value


def _safe(path):
    path = Path(path).expanduser().absolute()  # Do not resolve away unsafe links.
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Unsafe symlink or traversal")
    return path


def _relative(name):
    if (not isinstance(name, str) or not name or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("Unsafe relative path")
    return name


def _read(path, maximum=MAX_FILE_BYTES):
    path = _safe(path)
    if not path.is_file() or not 0 < path.stat().st_size <= min(maximum, MAX_FILE_BYTES):
        raise ValueError("Missing, empty or oversized file")
    with path.open("rb") as stream:
        payload = stream.read(min(maximum, MAX_FILE_BYTES) + 1)
    if not 0 < len(payload) <= min(maximum, MAX_FILE_BYTES):
        raise ValueError("File size limit exceeded")
    return payload


def stream_digest(stream, size=None):
    """Hash incrementally; a remote stream must also have the exact expected size."""
    digest, total = hashlib.sha256(), 0
    while chunk := stream.read(1024**2):
        total += len(chunk)
        if size is not None and total > size:
            raise ValueError("Remote archive size mismatch")
        digest.update(chunk)
    if size is not None and total != size:
        raise ValueError("Remote archive size mismatch")
    return digest.hexdigest()


def _walk(directory):
    """Walk only the fixed child scope, pruning caches and all hidden directories."""
    directory = _safe(directory)
    for base, dirs, names in os.walk(directory, followlinks=False):
        for name in dirs:
            _safe(Path(base) / name)
        dirs[:] = sorted(n for n in dirs if not n.startswith(".")
                         and n.lower() not in _EXCLUDED and "cache" not in n.lower())
        for name in sorted(names):
            if not name.startswith("."):
                yield _safe(Path(base) / name)


def _dataset(payload):
    """Validate the v1 storage schema and references, without rendering any pixels.

    This is a backup integrity check, not a new scientific scene certification.
    Original PNG bytes (including Pillow codec differences) are preserved.
    """
    data = _json(payload)
    if (type(data) is not dict or set(data) != {"schema_version", "name", "seed",
            "image_size", "split_counts", "families"} or type(data["schema_version"]) is not int
            or data["schema_version"] != 1 or data["name"] != "reading-or-routing-v1"):
        raise ValueError("Invalid dataset schema")
    counts = data["split_counts"]
    if (type(counts) is not dict or set(counts) != set(_SPLITS)
            or any(type(n) is not int or not 0 <= n <= 1024 for n in counts.values())
            or not 1 <= sum(counts.values()) <= 1024
            or type(data["seed"]) is not int or not 0 <= data["seed"] < 2**63
            or type(data["image_size"]) is not int or not 128 <= data["image_size"] <= 1024
            or type(data["families"]) is not list
            or len(data["families"]) != sum(counts.values())):
        raise ValueError("Invalid dataset settings")
    images, seen = [], set()
    splits = [s for s in _SPLITS for _ in range(counts[s])]
    for index, (family, split) in enumerate(zip(data["families"], splits, strict=True)):
        if (type(family) is not dict or set(family) != {"family_id", "split", "index",
                "anchor", "nodes", "palettes", "regions", "program", "images", "cases"}
                or type(family["index"]) is not int or family["index"] != index
                or family["split"] != split or type(family["anchor"]) is not int
                or family["anchor"] != index % 3
                or not isinstance(family["family_id"], str)
                or not re.fullmatch(r"f[0-9]{6}-[0-9a-f]{16}", family["family_id"])
                or type(family["images"]) is not list or len(family["images"]) != 4
                or type(family["cases"]) is not list or len(family["cases"]) != 12
                or type(family["nodes"]) is not list or len(family["nodes"]) != 3
                or type(family["palettes"]) is not list or len(family["palettes"]) != 2
                or type(family["regions"]) is not dict or type(family["program"]) is not dict):
            raise ValueError("Invalid dataset family")
        shapes, colors = ("triangle", "circle", "square"), ("red", "blue", "green", "yellow", "purple", "orange")
        for node in family["nodes"]:
            if (type(node) is not dict or set(node) != {"shape", "x", "y"}
                    or node["shape"] not in shapes
                    or any(type(node[k]) not in (int, float) for k in ("x", "y"))):
                raise ValueError("Invalid dataset node schema")
        palettes = family["palettes"]
        if (any(type(p) is not list or len(p) != 3 for p in palettes)
            or any(c not in colors for p in palettes for c in p)
                or sorted(c for p in palettes for c in p) != sorted(colors)
                or set(family["regions"]) != {"nodes", "chips", "edges", "arrowheads"}
                or set(family["program"]) != {"renderer", "seed", "supersample"}
                or family["program"]["renderer"] != data["name"]
                or type(family["program"]["seed"]) is not int
                or not 0 <= family["program"]["seed"] < 2**63
                or type(family["program"]["supersample"]) is not int
                or family["program"]["supersample"] != 4):
            raise ValueError("Invalid dataset scene schema")
        case_ids = set()
        for case in family["cases"]:
            if (type(case) is not dict or set(case) != {"case_id", "family_id", "split",
                    "task", "question", "gold", "candidates", "image_id", "edge", "direction",
                    "palette", "anchor", "target"} or case["family_id"] != family["family_id"]
                    or case["split"] != split or not isinstance(case["case_id"], str)
                    or case["case_id"] in case_ids or not isinstance(case["question"], str)
                    or not case["question"] or type(case["candidates"]) is not list
                    or case["gold"] not in case["candidates"]
                    or case["task"] not in ("binding", "color_readout", "edge_readout", "text_binding")):
                raise ValueError("Invalid dataset case schema")
            for key, allowed in (("edge", (-1, 1)), ("direction", (-1, 1)),
                                 ("palette", (0, 1)), ("anchor", (0, 1, 2)), ("target", (0, 1, 2))):
                if type(case[key]) is not int or case[key] not in allowed:
                    raise ValueError("Invalid dataset case factor")
            image_id = (None if case["task"] == "text_binding" else
                        f"{family['family_id']}-e{'p' if case['edge'] == 1 else 'm'}-p{case['palette']}")
            if (case["image_id"] != image_id or case["candidates"] !=
                    list(shapes if case["task"] == "edge_readout" else colors)):
                raise ValueError("Invalid dataset case image/candidates")
            case_ids.add(case["case_id"])
        for item, (edge, palette) in zip(family["images"],
                ((1, 0), (1, 1), (-1, 0), (-1, 1)), strict=True):
            image_id = f"{family['family_id']}-e{'p' if edge == 1 else 'm'}-p{palette}"
            if (type(item) is not dict
                    or set(item) != {"image_id", "relative_path", "edge", "palette", "sha256"}
                    or item["image_id"] != image_id or item["relative_path"] != f"images/{image_id}.png"
                    or type(item["edge"]) is not int or item["edge"] != edge
                    or type(item["palette"]) is not int or item["palette"] != palette
                    or item["relative_path"] in seen or not isinstance(item["sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                raise ValueError("Invalid or duplicate dataset image path")
            seen.add(item["relative_path"])
            images.append(item)
    return data, images


def capture(root, destination, *, include_logs=False):
    """Create an exclusive ZIP; return counts, file count, ZIP size and SHA256.

    BACKUP.json lists exact-byte hashes/sizes of every other ZIP member, run
    names, captured time and completed counts. Limits include BACKUP.json itself.
    Any invalid published reference aborts; orphans are never inferred or copied.
    No writer lock, source fingerprint recomputation or model operation occurs.
    """
    root, destination = _safe(root), _safe(destination)
    if type(include_logs) is not bool:
        raise ValueError("Log inclusion requires an explicit boolean")
    paths = list(_walk(root / "runs/vlm-pilot"))
    boundaries, boundary_bytes = [], 0
    for path in paths:
        if path.name == "manifest.json":
            payload = _read(path, 32 * 1024**2)
            manifest = _json(payload)
            if (type(manifest) is not dict or type(manifest.get("schema_version")) is not int
                    or manifest["schema_version"] != 1
                    or not isinstance(manifest.get("run_id"), str)
                    or not isinstance(manifest.get("metadata_sha256"), str)
                    or type(manifest.get("records")) is not list
                    or type(manifest.get("completed")) is not int):
                raise ValueError("Invalid manifest schema")
            boundaries.append((path, payload, manifest))
            boundary_bytes += len(payload)
            if len(boundaries) >= MAX_MEMBERS or boundary_bytes > MAX_TOTAL_BYTES:
                raise ValueError("Manifest snapshot limit exceeded")
    if not boundaries:
        raise ValueError("No published VLM runs")
    entries, completed, total = {}, {}, 0
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        def add(path, payload):
            nonlocal total
            name = _relative(_safe(path).relative_to(root).as_posix())
            if name in entries:
                raise ValueError("Duplicate archive path")
            if (len(payload) > MAX_FILE_BYTES or total + len(payload) > MAX_TOTAL_BYTES
                    or len(entries) + 2 > MAX_MEMBERS):
                raise ValueError("Archive limit exceeded")
            entries[name] = {"path": name, "sha256": hashlib.sha256(payload).hexdigest(),
                             "size_bytes": len(payload)}
            archive.writestr(name, payload)
            total += len(payload)

        def referenced(directory, ref, maximum):
            if type(ref) is not dict or not {"path", "sha256", "size_bytes"} <= ref.keys():
                raise ValueError("Invalid reference schema")
            name = _relative(ref["path"])
            if (type(ref["size_bytes"]) is not int or not 0 < ref["size_bytes"] <= maximum
                    or not isinstance(ref["sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])):
                raise ValueError("Invalid reference size/checksum")
            payload = _read(directory / name, maximum)
            if (len(payload) != ref["size_bytes"]
                    or hashlib.sha256(payload).hexdigest() != ref["sha256"]):
                raise ValueError("Reference checksum/size mismatch")
            add(directory / name, payload)
            return payload

        dataset_path = root / "data/vlm-pilot/dataset.json"
        dataset_payload = _read(dataset_path, 16 * 1024**2)
        dataset, images = _dataset(dataset_payload)
        add(dataset_path, dataset_payload)
        for image in images:
            payload = _read(dataset_path.parent / image["relative_path"], 8 * 1024**2)
            if hashlib.sha256(payload).hexdigest() != image["sha256"]:
                raise ValueError("Dataset image checksum mismatch")
            add(dataset_path.parent / image["relative_path"], payload)
        for path, payload, manifest in boundaries:
            metadata_payload = _read(path.parent / "metadata.json", 32 * 1024**2)
            metadata = _json(metadata_payload, sealed=True)
            if (type(metadata) is not dict or type(metadata.get("schema_version")) is not int
                    or metadata["schema_version"] != 1
                    or hashlib.sha256(metadata_payload).hexdigest() != manifest["metadata_sha256"]
                    or metadata.get("run_id") != manifest["run_id"]):
                raise ValueError("Metadata checksum/run mismatch")
            plan = metadata.get("plan", {})
            if type(plan) is not dict:
                raise ValueError("Invalid embedded plan")
            if ("dataset" in plan and canonical(plan["dataset"]) != canonical(dataset)
                    or "dataset_sha256" in plan and plan["dataset_sha256"] !=
                    hashlib.sha256(canonical(dataset)).hexdigest()):
                raise ValueError("Archived dataset disagrees with run plan")
            add(path.parent / "metadata.json", metadata_payload)
            trials, attempts = set(), set()
            for ref in manifest["records"]:
                if (type(ref) is not dict
                        or set(ref) != {"trial_id", "path", "sha256", "size_bytes"}
                        or not isinstance(ref["trial_id"], str)
                        or not re.fullmatch(r"[A-Za-z0-9_-]+", ref["trial_id"])
                        or ref["path"] != f"records/{ref['trial_id']}-{ref['sha256']}.json"):
                    raise ValueError("Invalid record reference path")
                row = _json(referenced(path.parent, ref, 4 * 1024**2), sealed=True)
                if (type(row) is not dict or row.get("run_id") != metadata["run_id"]
                        or row.get("trial_id") != ref["trial_id"]
                        or not isinstance(row.get("attempt_id"), str) or not row["attempt_id"]
                        or row["attempt_id"] in attempts or row.get("status") not in
                        ("completed", "failed", "cancelled")):
                    raise ValueError("Invalid trial/run/status or duplicate attempt")
                attempts.add(row["attempt_id"])
                if row["status"] == "completed":
                    if row["trial_id"] in trials:
                        raise ValueError("Duplicate completed trial")
                    trials.add(row["trial_id"])
                if type(row.get("artifacts", [])) is not list:
                    raise ValueError("Invalid capture list")
                for artifact in row.get("artifacts", []):
                    if type(artifact) is not dict or set(artifact) != {"path", "sha256", "size_bytes"}:
                        raise ValueError("Invalid capture schema")
                    name = _relative(artifact["path"])
                    if (not name.startswith("captures/") or len(name.split("/")) != 2
                            or not name.endswith(".safetensors")):
                        raise ValueError("Unsafe capture reference")
                    referenced(path.parent, artifact, MAX_FILE_BYTES)
            if type(manifest["completed"]) is not int or len(trials) != manifest["completed"]:
                raise ValueError("Manifest completion mismatch")
            completed[path.parent.relative_to(root).as_posix()] = len(trials)
            add(path, payload)

        patterns = ("src/vlm_pilot/*.py", "configs/vlm/*.yaml", "notebooks/vlm_pilot.py",
                    "scripts/backup_vlm.py", "scripts/vlm_notebook.py", "scripts/run_vlm_pilot.sh",
                    "scripts/vlm_projector_diagnostic.py", "scripts/vlm_research_followup.py",
                    "scripts/vlm_handoff_diagnostic.py",
                    "scripts/vlm_earlier_routing_diagnostic.py", "scripts/vlm_joint_handoff_diagnostic.py",
                    "scripts/vlm_text_state_diagnostic.py",
                    "runs/vlm-pilot/report-*/metadata.json",
                    "experiments/vlm_posttraining/*.md", "README.md", "RESEARCH_PLAN.md",
                    "pyproject.toml", "requirements-tested.txt", "plan-loader-v2.json", "plan-gpu-uuid.json")
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if not path.name.startswith("."):
                    add(path, _read(path))
        for path in paths:
            if (path.name in ("manifest.json", "metadata.json", "request.json")
                or re.search(r"(?i)secret|credential|environment|token|password|cache", path.name)
                    or path.suffix.lower() not in (".json", ".csv", ".md")):
                continue
            if (path.suffix.lower() == ".json" and not path.stem.lower().startswith(
                ("plan", "protocol", "preflight", "p0-", "components", "numerical",
                 "numerics", "report", "integrity", "summary"))
                and not any(p.lower().startswith(("report", "analysis"))
                    for p in path.relative_to(root / "runs/vlm-pilot").parts[:-1])):
                continue  # Not a broad JSON dump of the job tree or model assets.
            try:
                payload = _read(path)
                if path.suffix.lower() == ".json":
                    _json(payload)
            except (OSError, ValueError):
                _safe(path)  # Missing/mid-write reports may be skipped, unsafe links may not.
                continue  # Mutable reports can disappear or be mid-publication.
            add(path, payload)
        if include_logs:
            for path in _walk(root / "runs/vlm-notebook/jobs"):
                if path.name not in ("stdout.log", "launch.json", "status.json"):
                    continue  # Never request.json or the inherited environment.
                try:
                    if path.name == "stdout.log":
                        with path.open("rb") as stream:
                            payload = stream.read(min(MAX_STDOUT_BYTES, MAX_FILE_BYTES))
                    else:
                        payload = _read(path)
                        _json(payload)
                except (OSError, ValueError):
                    _safe(path)
                    continue
                add(path, payload)
        summary = {"schema_version": 1, "captured_at": datetime.now(UTC).isoformat(),
                   "run_names": sorted(completed), "completed_trials": completed,
                   "files": len(entries) + 1, "payload_size_bytes": total,
                   "include_logs": include_logs, "entries": list(entries.values())}
        payload = canonical(summary)
        if len(payload) > MAX_FILE_BYTES or total + len(payload) > MAX_TOTAL_BYTES:
            raise ValueError("Backup inventory limit exceeded")
        archive.writestr("BACKUP.json", payload)
    with destination.open("rb") as stream:
        return {**summary, "size_bytes": destination.stat().st_size,
                "sha256": stream_digest(stream)}


class VLMDriveBackup:
    """One daemon per instance; full snapshots accumulate, with no remote deletion.

    start() starts once, immediately; interval begins after each upload finishes.
    once() rejects overlap. stop() only signals this helper, never GPU processes.
    status returns a locked deep copy. Errors expose their type, never raw text.
    """

    def __init__(self, fs, root, remote_dir, interval=300, *, include_logs=False):
        if remote_dir != REMOTE_DIR:
            raise ValueError("Only the designated VLM backup child directory is allowed")
        if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
            raise ValueError("Backup interval must be finite and positive")
        if type(include_logs) is not bool:
            raise ValueError("Log inclusion requires an explicit boolean")
        self.fs, self.root, self.remote_dir = fs, _safe(root), remote_dir
        self.interval, self.include_logs = interval, include_logs
        self.stop_event, self.thread = threading.Event(), None
        self._state_lock, self._upload_lock = threading.Lock(), threading.Lock()
        self._status = {"state": "not_started"}

    @property
    def status(self):
        with self._state_lock:
            return copy.deepcopy(self._status)

    def _set(self, value):
        with self._state_lock:
            self._status = value

    def start(self):
        with self._state_lock:
            if self.thread is not None:
                raise RuntimeError("This backup worker has already been started")
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self.once()
            except RuntimeError:
                pass  # once() already published a sanitized error, or another upload owns the lock.
            if self.stop_event.wait(self.interval):
                break

    def once(self):
        """Upload, hash remote bytes, then write AND read back the checksum receipt."""
        if not self._upload_lock.acquire(blocking=False):
            raise RuntimeError("A VLM backup upload is already active")
        try:
            self._set({"state": "uploading"})
            name = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex + ".zip"
            remote = self.remote_dir + "/" + name
            # Normalize only OS-owned scratch space (macOS /var can be a symlink),
            # never the supplied research root, destination or artifact paths.
            scratch = Path(tempfile.gettempdir()).resolve()
            with tempfile.TemporaryDirectory(prefix="vlm-backup-", dir=_safe(scratch)) as temporary:
                local = Path(temporary) / name
                snapshot = capture(self.root, local, include_logs=self.include_logs)
                expected, size = snapshot["sha256"], snapshot["size_bytes"]
                if not self.fs.exists(self.remote_dir):
                    self.fs.makedirs(self.remote_dir, exist_ok=True)
                self.fs.put_file(str(local), remote)
                with self.fs.open(remote, "rb") as stream:
                    if stream_digest(stream, size) != expected:
                        raise ValueError("Remote archive checksum mismatch")
                receipt = f"{expected}  {name}\n".encode("ascii")
                with self.fs.open(remote + ".sha256", "wb") as stream:
                    stream.write(receipt)
                with self.fs.open(remote + ".sha256", "rb") as stream:
                    if stream.read(len(receipt) + 1) != receipt:
                        raise ValueError("Remote receipt mismatch")
            self._set({"state": "verified", "archive": remote, "sha256": expected,
                       "verified_at": datetime.now(UTC).isoformat(),
                       "completed_trials": snapshot["completed_trials"],
                       "files": snapshot["files"], "size_bytes": size})
            return self.status
        except Exception as error:
            kind = type(error).__name__
            self._set({"state": "error", "error_type": kind})
            raise RuntimeError(f"VLM backup failed ({kind})") from None
        finally:
            self._upload_lock.release()