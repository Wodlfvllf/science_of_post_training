"""Atomic, checksummed pilot records; independent of active text-study identity."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .schemas import PilotError


def now() -> str:
    return datetime.now(UTC).isoformat()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def identity(value) -> str:
    return digest(canonical(value))


def safe_path(path: Path) -> Path:
    path = path.expanduser().absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise PilotError("Pilot artifact paths must not traverse symlinks or parent components.")
    return path


def _pairs(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise PilotError("Duplicate JSON keys in pilot artifact.")
    return result


def read_json(path: Path, maximum: int = 32 * 1024**2):
    path = safe_path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise PilotError("Missing, empty or oversized pilot JSON artifact.")
    with path.open("rb") as handle:
        content = handle.read(maximum + 1)
    if len(content) > maximum:
        raise PilotError("Pilot JSON artifact exceeds its size limit.")
    value = json.loads(content, object_pairs_hook=_pairs)
    canonical(value)  # Reject nonfinite constants/overflowed JSON floats.
    return value


def atomic_bytes(path: Path, payload: bytes, *, immutable: bool = False) -> None:
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value, *, immutable: bool = False) -> None:
    atomic_bytes(path, canonical(value), immutable=immutable)


class ResultStore:
    """Caller holds .writer.lock before model loading and for the entire invocation.

    Each finished/failed attempt is sealed before an atomic manifest update. Only
    manifest-published successes advance completion; orphans are never inferred.
    Selected captures are optional and must be checksummed references in a record.
    """

    def __init__(self, root: Path):
        self.root = safe_path(root)
        self.manifest = None
        self.metadata = None
        self.records: list[dict] = []
        self.completed: set[str] = set()

    def read(self) -> ResultStore:
        self.manifest = read_json(self.root / "manifest.json")
        self.metadata = read_json(self.root / "metadata.json")
        if self.manifest["metadata_sha256"] != identity(self.metadata):
            raise PilotError("Pilot metadata checksum mismatch.")
        if self.manifest["run_id"] != self.metadata["run_id"]:
            raise PilotError("Pilot run identity mismatch.")
        self.records, self.completed = [], set()
        seen = set()
        for ref in self.manifest["records"]:
            expected_name = f"records/{ref['trial_id']}-{ref['sha256']}.json"
            if ref["path"] != expected_name:
                raise PilotError("Invalid pilot record reference.")
            path = self.root / expected_name
            row = read_json(path, 4 * 1024**2)
            if identity(row) != ref["sha256"] or path.stat().st_size != ref["size_bytes"]:
                raise PilotError("Pilot record checksum mismatch.")
            if row["trial_id"] != ref["trial_id"] or row["run_id"] != self.metadata["run_id"]:
                raise PilotError("Pilot record belongs to a different trial/run.")
            if row["attempt_id"] in seen:
                raise PilotError("Duplicate pilot attempt.")
            seen.add(row["attempt_id"])
            for artifact in row.get("artifacts", []):
                if not artifact["path"].startswith("captures/") or "/" in artifact["path"][9:]:
                    raise PilotError("Unsafe capture reference.")
                file = safe_path(self.root / artifact["path"])
                if (type(artifact["size_bytes"]) is not int or not 0 < artifact["size_bytes"] <= 64 * 1024**2
                    or not file.is_file() or file.stat().st_size != artifact["size_bytes"]):
                    raise PilotError("Missing pilot capture.")
                if digest(file.read_bytes()) != artifact["sha256"]:
                    raise PilotError("Pilot capture checksum mismatch.")
            if row["status"] == "completed":
                if row["trial_id"] in self.completed:
                    raise PilotError("Duplicate successful pilot trial.")
                self.completed.add(row["trial_id"])
            elif row["status"] not in ("failed", "cancelled"):
                raise PilotError("Invalid pilot attempt status.")
            self.records.append(row)
        if len(self.completed) != self.manifest["completed"]:
            raise PilotError("Pilot manifest completion mismatch.")
        return self

    def initialize(self, metadata: dict) -> None:
        path = self.root / "metadata.json"
        if path.exists():
            self.read()
            if self.metadata != metadata:
                raise PilotError("Resume refused: plan/job/model/runtime/protocol changed.")
            return
        if (self.root / "manifest.json").exists() or any(self.root.glob("records/*.json")):
            raise PilotError("Pilot records exist without metadata; refuse overwrite.")
        atomic_json(path, metadata, immutable=True)
        self.metadata = metadata
        self.manifest = {"schema_version": 1, "run_id": metadata["run_id"],
                         "metadata_sha256": identity(metadata), "records": [],
                         "completed": 0, "updated_at": now()}
        atomic_json(self.root / "manifest.json", self.manifest)

    def append(self, row: dict) -> None:
        if self.manifest is None or self.metadata is None:
            raise PilotError("Result store is not initialized.")
        if row["run_id"] != self.metadata["run_id"] or row["trial_id"] in self.completed:
            raise PilotError("Duplicate or incompatible pilot trial.")
        payload = canonical(row)
        if len(payload) > 4 * 1024**2:
            raise PilotError("Pilot trial exceeds its 4 MiB record size.")
        sha = digest(payload)
        name = f"records/{row['trial_id']}-{sha}.json"
        atomic_bytes(self.root / name, payload, immutable=True)
        ref = {"trial_id": row["trial_id"], "path": name, "sha256": sha,
               "size_bytes": len(payload)}
        if row["status"] == "completed":
            self.completed.add(row["trial_id"])
        manifest = dict(self.manifest, records=[*self.manifest["records"], ref],
                        completed=len(self.completed), updated_at=now())
        atomic_json(self.root / "manifest.json", manifest)
        self.manifest = manifest
        self.records.append(row)