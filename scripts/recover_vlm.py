"""Offline, stdlib-only, no-clobber recovery of independent VLM ZIP backups.

Public API: inspect_archive(archive_path, receipt_path) returns BACKUP.json's
inventory; stage_archive(archive_path, receipt_path, staging) returns verification
receipt details; publish_artifacts(staging, checkout) returns artifact_files,
new, skipped, and manifests counts; restore_archive combines stage and publish
and returns {"verification": receipt_details, "publication": counts}.

Nothing runs on import. Archived Python, shell, notebook and configuration bytes
are retained ONLY in a new staging directory, never imported or executed. Only
data/vlm-pilot, runs/vlm-pilot and safe root plan*.json files can be published.
Dataset indexes and run manifests are published last. This checks byte integrity
and path safety, not scientific schemas, authenticity or confidentiality: a
checksum receipt is not a signature, and allowed files may contain private text.

The checkout's runs/vlm-recovery/.restore.lock serializes cooperating restores
ONLY. GPU workers and all other writers MUST be made idle manually before use;
this helper neither stops them nor acquires their locks. Per-file publication is
atomic and exclusive, not a transaction over the entire checkout. An I/O failure
can leave complete non-manifest files; an identical retry safely skips them.
There is no network, OAuth, credential handling, deletion of remote state, model
import, dependency installation, CLI, or execution/resumption of research work.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import struct
import zipfile
import zlib
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

MAX_ARCHIVE_BYTES = 128 * 1024**2
MAX_TOTAL_BYTES = 2 * 1024**3
MAX_MEMBER_BYTES = 64 * 1024**2
MAX_MEMBERS = 30_000
MAX_INVENTORY_BYTES = 16 * 1024**2
CHUNK_BYTES = 1024**2
VERIFICATION_NAME = ".vlm-recovery-receipt.json"
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,254}\Z")
_PLAN = re.compile(r"plan[A-Za-z0-9_-]{0,120}\.json\Z")
_PRIVATE = re.compile(r"secret|credential|password|token|oauth|environment|cache", re.I)
_FORBIDDEN = {
    "git", "env", "envs", "venv", "venvs", "environment", "weights", "weight",
    "models", "model", "modelcache", "cache", "caches", "node_modules",
}
_SCRIPTS = {
    "backup_vlm.py", "vlm_notebook.py", "run_vlm_pilot.sh",
    "vlm_projector_diagnostic.py", "vlm_research_followup.py",
    "vlm_handoff_diagnostic.py", "vlm_earlier_routing_diagnostic.py",
    "vlm_joint_handoff_diagnostic.py", "vlm_text_state_diagnostic.py",
    "vlm_generated_decomposition.py",
    "vlm_decomposition_replication.py", "vlm_cue_context.py",
    "analyze_vlm_decomposition_replication.py", "analyze_vlm_cue_context.py", "recover_vlm.py",
}
_INVENTORY_KEYS = {
    "schema_version", "captured_at", "run_names", "completed_trials", "files",
    "payload_size_bytes", "include_logs", "entries",
}


def _canonical(value):
    """Finite, deterministic UTF-8 JSON, matching backup_vlm's sealed format."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _sealed_json(payload):
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_pairs)
        if _canonical(value) != payload:
            raise ValueError("Noncanonical JSON")
        return value
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError("Invalid bounded finite JSON") from error


def _absolute(path):
    # Never resolve(): that would erase evidence of a supplied symlink.
    path = Path(path).expanduser()
    if ".." in path.parts:
        raise ValueError("Filesystem traversal is forbidden")
    return Path(os.path.abspath(path))


def _open_directory(path, *, create=False):
    """Open every directory component without following links, including parents."""
    path = _absolute(path)
    fd = os.open(path.anchor, _DIR_FLAGS)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass  # The no-follow directory open below still checks its type.
            child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except OSError as error:
        os.close(fd)
        raise ValueError("Unsafe or unavailable directory component") from error


@contextmanager
def _regular_stream(path):
    path = _absolute(path)
    parent = _open_directory(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    except OSError as error:
        raise ValueError("Unsafe or missing regular file") from error
    finally:
        os.close(parent)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Expected a regular file")
        yield stream


def _hash_stream(stream, maximum, expected_size=None, output=None):
    """Always read through EOF; zipfile's CRC check and our size/hash checks run."""
    digest, size, crc = hashlib.sha256(), 0, 0
    while chunk := stream.read(min(CHUNK_BYTES, maximum - size + 1)):
        size += len(chunk)
        if size > maximum or (expected_size is not None and size > expected_size):
            raise ValueError("Stream size limit or declared size exceeded")
        digest.update(chunk)
        crc = zlib.crc32(chunk, crc)
        if output is not None:
            output.write(chunk)
    if expected_size is not None and size != expected_size:
        raise ValueError("Stream size mismatch")
    return {"sha256": digest.hexdigest(), "size_bytes": size}, crc & 0xFFFFFFFF


def _hash_file(path, maximum, expected=None):
    with _regular_stream(path) as stream:
        actual, _ = _hash_stream(stream, maximum,
                                 None if expected is None else expected["size_bytes"])
    if expected is not None and actual != _digest_fields(expected):
        raise ValueError("File checksum or size mismatch")
    return actual


def _digest_fields(entry):
    return {"sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}


def _valid_digest(entry, maximum):
    return (type(entry) is dict and type(entry.get("size_bytes")) is int
            and 0 <= entry["size_bytes"] <= maximum
            and isinstance(entry.get("sha256"), str)
            and _HEX.fullmatch(entry["sha256"]) is not None)


def _relative(name):
    if not isinstance(name, str) or not name or len(name) > 1024:
        raise ValueError("Invalid archive path")
    parts = name.split("/")
    if (len(parts) > 64 or any(not _PART.fullmatch(part) or part.endswith(".")
                             or part.lower() in _FORBIDDEN or _PRIVATE.search(part)
                             for part in parts)):
        raise ValueError("Unsafe, hidden, private or non-normal archive path")
    return parts


def _allowed(name):
    """Closed archive allowlist; safetensors are selected captures, never weights."""
    parts = _relative(name)
    if name == "BACKUP.json":
        return True
    if len(parts) == 1:
        return (name in {"README.md", "RESEARCH_PLAN.md", "pyproject.toml",
                         "requirements.txt", "requirements-tested.txt"}
                or _PLAN.fullmatch(name) is not None)
    if parts[:2] == ["src", "vlm_pilot"]:
        return len(parts) == 3 and name.endswith(".py")
    if parts[0] == "notebooks":
        return len(parts) == 2 and parts[1] in {"vlm_pilot.py", "vlm_pilot.ipynb"}
    if parts[:2] == ["configs", "vlm"]:
        return len(parts) == 3 and (Path(name).suffix in {".yaml", ".yml"}
            or parts[-1] in {"c2-development.json", "cue-context-development.json"})
    if parts[0] == "scripts":
        return len(parts) == 2 and parts[1] in _SCRIPTS
    if parts[:2] == ["experiments", "vlm_posttraining"]:
        return len(parts) == 3 and name.endswith(".md")
    if parts[:2] == ["data", "vlm-pilot"]:
        return (name == "data/vlm-pilot/dataset.json"
                or len(parts) == 4 and parts[2] == "images" and name.endswith(".png"))
    if parts[:2] == ["runs", "vlm-pilot"] and len(parts) >= 3:
        if (parts[-1] in {"request.json", "launch.json", "status.json"}
                or Path(parts[-1]).stem.lower() in _FORBIDDEN):
            return False
        if "captures" in parts:
            return (len(parts) >= 5 and parts[-2] == "captures"
                    and parts.count("captures") == 1 and name.endswith(".safetensors"))
        return Path(name).suffix in {".json", ".csv", ".md", ".png"}
    return False


def _check_names(names):
    """Reject duplicates, file/parent collisions, and case-insensitive aliases."""
    seen, spellings, files = set(), {}, set(names)
    for name in names:
        if name in seen or not _allowed(name):
            raise ValueError("Duplicate or disallowed archive member")
        seen.add(name)
        parts = name.split("/")
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            old = spellings.setdefault(prefix.casefold(), prefix)
            if old != prefix or (index < len(parts) and prefix in files):
                raise ValueError("Archive file/directory or normalization collision")


def _inventory(payload, members=None):
    if len(payload) > MAX_INVENTORY_BYTES:
        raise ValueError("Inventory size limit exceeded")
    value = _sealed_json(payload)
    if (type(value) is not dict or set(value) != _INVENTORY_KEYS
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or not isinstance(value["captured_at"], str)
            or not 1 <= len(value["captured_at"]) <= 128
            or type(value["include_logs"]) is not bool
            or type(value["entries"]) is not list
            or not 1 <= len(value["entries"]) < MAX_MEMBERS
            or type(value["files"]) is not int
            or value["files"] != len(value["entries"]) + 1
            or type(value["payload_size_bytes"]) is not int
            or type(value["run_names"]) is not list
            or type(value["completed_trials"]) is not dict):
        raise ValueError("Invalid BACKUP inventory schema")
    entries, total = {}, 0
    for entry in value["entries"]:
        if (not _valid_digest(entry, MAX_MEMBER_BYTES)
                or set(entry) != {"path", "sha256", "size_bytes"}
                or not isinstance(entry["path"], str) or entry["path"] == "BACKUP.json"
                or entry["path"] in entries):
            raise ValueError("Invalid or duplicate inventory entry")
        entries[entry["path"]] = entry
        total += entry["size_bytes"]
    _check_names(["BACKUP.json", *entries])
    if (total != value["payload_size_bytes"] or total + len(payload) > MAX_TOTAL_BYTES
            or (members is not None and set(entries) != set(members) - {"BACKUP.json"})):
        raise ValueError("Inventory membership or total size mismatch")
    runs = value["run_names"]
    if any(not isinstance(run, str) for run in runs) or len(set(runs)) != len(runs):
        raise ValueError("Invalid inventory run names")
    for run in runs:
        parts = _relative(run)
        if (parts[:2] != ["runs", "vlm-pilot"] or len(parts) < 3
                or run + "/manifest.json" not in entries):
            raise ValueError("Inventory run has no archived manifest")
    if (set(value["completed_trials"]) != set(runs)
            or any(type(count) is not int or count < 0
                   for count in value["completed_trials"].values())):
        raise ValueError("Invalid inventory completion counts")
    if members is not None:
        for name, entry in entries.items():
            if entry["size_bytes"] != members[name].file_size:
                raise ValueError("Inventory member size mismatch")
    return value


def _bounded_directory(stream):
    """Count central records BEFORE ZipFile allocates one object per member.

    Single-disk classic ZIP and bounded ZIP64 end records are supported. Refuse
    trailing data and inconsistent offsets rather than guessing how to normalize
    them. All offsets are bounded by the already capped physical file.
    """
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    tail_size = min(size, 65_535 + 22)
    stream.seek(size - tail_size)
    tail = stream.read(tail_size)
    index = tail.rfind(b"PK\x05\x06")
    if index < 0 or index + 22 > len(tail):
        raise ValueError("Missing ZIP end record")
    fields = struct.unpack_from("<4s4H2IH", tail, index)
    _, disk, directory_disk, disk_count, count, directory_size, offset, comment_size = fields
    if (disk or directory_disk or disk_count != count
            or index + 22 + comment_size != len(tail)):
        raise ValueError("Split, trailing or inconsistent ZIP end record")
    end = size - tail_size + index
    if end >= 20:
        stream.seek(end - 20)
        locator = stream.read(20)
        if locator[:4] == b"PK\x06\x07":
            _, zip64_disk, zip64_offset, disks = struct.unpack("<4sIQI", locator)
            if zip64_disk or disks != 1 or not 0 <= zip64_offset <= end - 20 - 56:
                raise ValueError("Invalid ZIP64 locator")
            stream.seek(zip64_offset)
            record = stream.read(56)
            (signature, length, _made, _needed, record_disk, record_directory_disk,
             record_disk_count, record_count, record_size, record_offset) = struct.unpack(
                 "<4sQ2H2I4Q", record)
            if (signature != b"PK\x06\x06" or length < 44
                    or zip64_offset + 12 + length != end - 20
                    or record_disk or record_directory_disk or record_disk_count != record_count
                    or count not in (0xFFFF, record_count)
                    or directory_size not in (0xFFFFFFFF, record_size)
                    or offset not in (0xFFFFFFFF, record_offset)):
                raise ValueError("Inconsistent ZIP64 end record")
            count, directory_size, offset, end = (
                record_count, record_size, record_offset, zip64_offset)
    if not 1 < count <= MAX_MEMBERS or offset + directory_size != end:
        raise ValueError("ZIP member limit or central directory bounds exceeded")
    position, actual_count = offset, 0
    while position < end:
        if actual_count >= MAX_MEMBERS or position + 46 > end:
            raise ValueError("ZIP central directory member limit exceeded")
        stream.seek(position)
        header = stream.read(46)
        if header[:4] != b"PK\x01\x02":
            raise ValueError("Invalid central directory record")
        name_size, extra_size, member_comment = struct.unpack_from("<3H", header, 28)
        if not 1 <= name_size <= 1024 or struct.unpack_from("<H", header, 34)[0]:
            raise ValueError("Unsafe ZIP name length or split member")
        position += 46 + name_size + extra_size + member_comment
        if position > end:
            raise ValueError("Truncated central directory record")
        actual_count += 1
    if actual_count != count:
        raise ValueError("ZIP declared member count mismatch")
    stream.seek(0)


def _zip_members(archive, stream):
    infos = archive.infolist()
    if not 1 < len(infos) <= MAX_MEMBERS:
        raise ValueError("ZIP member count limit exceeded")
    _check_names([info.filename for info in infos])
    total, compressed, members = 0, 0, {}
    for info in infos:
        mode = info.external_attr >> 16
        if (info.orig_filename != info.filename or info.is_dir()
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                or info.external_attr & 0x18 or info.flag_bits & 0x2041
                or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                or not 0 <= info.file_size <= MAX_MEMBER_BYTES
                or not 0 <= info.compress_size <= MAX_ARCHIVE_BYTES):
            raise ValueError("Unsafe ZIP attributes, method or member size")
        # Do not silently trust central flags when a local header claims encryption
        # or a different method. ZipFile otherwise uses mainly the central values.
        if info.header_offset < 0:
            raise ValueError("Invalid local header offset")
        stream.seek(info.header_offset)
        header = stream.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid local file header")
        flags, method = struct.unpack_from("<HH", header, 6)
        if flags != info.flag_bits or method != info.compress_type:
            raise ValueError("Inconsistent ZIP flags or compression method")
        total += info.file_size
        compressed += info.compress_size
        members[info.filename] = info
    if (total > MAX_TOTAL_BYTES or compressed > MAX_ARCHIVE_BYTES
            or "BACKUP.json" not in members
            or members["BACKUP.json"].file_size > MAX_INVENTORY_BYTES):
        raise ValueError("ZIP total or inventory size limit exceeded")
    return members


def _read_member(archive, info, expected=None, output=None):
    with archive.open(info, "r") as stream:
        actual, crc = _hash_stream(stream, MAX_MEMBER_BYTES, info.file_size, output)
    if crc != info.CRC or expected is not None and actual != _digest_fields(expected):
        raise ValueError("ZIP member CRC, checksum or size mismatch")
    return actual


@contextmanager
def _verified_archive(archive_path, receipt_path):
    archive_path, receipt_path = _absolute(archive_path), _absolute(receipt_path)
    name = archive_path.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.zip", name):
        raise ValueError("Unsafe ZIP basename")
    with _regular_stream(receipt_path) as receipt:
        payload = receipt.read(513)
    pattern = rb"([0-9a-f]{64})  " + re.escape(name.encode("ascii")) + rb"\n"
    match = re.fullmatch(pattern, payload)
    if match is None:
        raise ValueError("Receipt must be lowercase SHA256, two spaces, ZIP basename, newline")
    with _regular_stream(archive_path) as stream:
        if not 0 < os.fstat(stream.fileno()).st_size <= MAX_ARCHIVE_BYTES:
            raise ValueError("Compressed archive size limit exceeded")
        actual, _ = _hash_stream(stream, MAX_ARCHIVE_BYTES)
        if actual["sha256"] != match[1].decode("ascii"):
            raise ValueError("Full ZIP checksum mismatch")
        _bounded_directory(stream)
        try:
            with zipfile.ZipFile(stream, "r") as archive:
                members = _zip_members(archive, stream)
                # The inventory is bounded before decoding; reading to EOF verifies its CRC.
                info = members["BACKUP.json"]
                with archive.open(info) as source:
                    inventory_bytes = source.read(MAX_INVENTORY_BYTES + 1)
                    if len(inventory_bytes) != info.file_size or source.read(1):
                        raise ValueError("Inventory stream size mismatch")
                if zlib.crc32(inventory_bytes) & 0xFFFFFFFF != info.CRC:
                    raise ValueError("Inventory CRC mismatch")
                inventory = _inventory(inventory_bytes, members)
                entries = {entry["path"]: entry for entry in inventory["entries"]}
                for member_name, member in members.items():
                    _read_member(archive, member, entries.get(member_name))
                stream.seek(0)
                if _hash_stream(stream, MAX_ARCHIVE_BYTES)[0] != actual:
                    raise ValueError("ZIP changed during inspection")
                details = {
                    "schema_version": 1,
                    "archive": {"name": name, **actual},
                    "inventory": {"sha256": hashlib.sha256(inventory_bytes).hexdigest(),
                                  "size_bytes": len(inventory_bytes)},
                    "files": len(members),
                }
                yield archive, inventory, details
                # A stage cannot be sealed if its source changed during extraction.
                stream.seek(0)
                if _hash_stream(stream, MAX_ARCHIVE_BYTES)[0] != actual:
                    raise ValueError("ZIP changed during extraction")
        except (zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError,
                zlib.error) as error:
            raise ValueError("Invalid or corrupt ZIP stream") from error


def inspect_archive(archive_path, receipt_path):
    """Read-only full ZIP/receipt/member verification; return the exact inventory."""
    with _verified_archive(archive_path, receipt_path) as (_, inventory, _details):
        return inventory


def _exclusive_bytes(path, payload):
    parent = _open_directory(path.parent, create=True)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(parent)
    finally:
        os.close(parent)


def stage_archive(archive_path, receipt_path, staging):
    """Verify everything before creating a NEW stage; extract exclusive safe files.

    Return and seal verification details inside staging only. A failed stage is
    deliberately not reusable; choose a new staging path rather than overwriting.
    """
    staging = _absolute(staging)
    parent = _open_directory(staging.parent)
    try:
        if os.path.lexists(staging):
            raise ValueError("Staging must not already exist")
        with _verified_archive(archive_path, receipt_path) as (archive, inventory, details):
            # Only after all streams validate may even the stage directory be created.
            os.mkdir(staging.name, mode=0o700, dir_fd=parent)
            os.fsync(parent)
            entries = {entry["path"]: entry for entry in inventory["entries"]}
            for info in archive.infolist():
                destination = staging / info.filename
                directory = _open_directory(destination.parent, create=True)
                try:
                    fd = os.open(destination.name,
                                 os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
                    with os.fdopen(fd, "wb") as output:
                        _read_member(archive, info, entries.get(info.filename), output)
                        output.flush()
                        os.fsync(output.fileno())
                    os.fsync(directory)
                finally:
                    os.close(directory)
        # _verified_archive's final whole-ZIP check occurs BEFORE sealing the stage.
        _exclusive_bytes(staging / VERIFICATION_NAME, _canonical(details))
        return details
    finally:
        os.close(parent)


def _bounded_file(path, maximum):
    with _regular_stream(path) as stream:
        if os.fstat(stream.fileno()).st_size > maximum:
            raise ValueError("File read limit exceeded")
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("File read limit exceeded")
    return payload


def _staged_names(staging, expected):
    """Enumerate fd-relative; extra files/dirs, links and special files all fail."""
    files, directories = set(), set()
    allowed_dirs = {parent.as_posix() for name in expected for parent in Path(name).parents
                    if parent != Path(".")}

    def visit(fd, prefix):
        with os.scandir(fd) as children:
            for child in children:
                name = prefix + child.name
                mode = child.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    if name not in allowed_dirs:
                        raise ValueError("Unexpected staged directory")
                    directories.add(name)
                    child_fd = os.open(child.name, _DIR_FLAGS, dir_fd=fd)
                    try:
                        visit(child_fd, name + "/")
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(mode) and name in expected:
                    files.add(name)
                else:
                    raise ValueError("Unexpected, symlink or special staged file")

    fd = _open_directory(staging)
    try:
        visit(fd, "")
    finally:
        os.close(fd)
    if files != set(expected) or directories != allowed_dirs:
        raise ValueError("Missing staged inventory member or directory")


def _verify_stage(staging):
    receipt = _sealed_json(_bounded_file(staging / VERIFICATION_NAME, 16 * 1024))
    if (type(receipt) is not dict
            or set(receipt) != {"schema_version", "archive", "inventory", "files"}
            or type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1
            or not _valid_digest(receipt["archive"], MAX_ARCHIVE_BYTES)
            or set(receipt["archive"]) != {"name", "sha256", "size_bytes"}
            or not isinstance(receipt["archive"]["name"], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.zip",
                                receipt["archive"]["name"])
            or not _valid_digest(receipt["inventory"], MAX_INVENTORY_BYTES)
            or set(receipt["inventory"]) != {"sha256", "size_bytes"}
            or type(receipt["files"]) is not int):
        raise ValueError("Invalid staging verification receipt")
    payload = _bounded_file(staging / "BACKUP.json", MAX_INVENTORY_BYTES)
    if ({"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
            != receipt["inventory"]):
        raise ValueError("Staged inventory changed")
    inventory = _inventory(payload)
    if receipt["files"] != inventory["files"]:
        raise ValueError("Staging receipt file count mismatch")
    entries = {entry["path"]: entry for entry in inventory["entries"]}
    # The helper's own verification file is not an archived inventory entry.
    _staged_names(staging, {*entries, "BACKUP.json", VERIFICATION_NAME})
    for name, entry in entries.items():
        _hash_file(staging / name, MAX_MEMBER_BYTES, entry)
    return entries


def _publishable(name):
    return (name.startswith(("data/vlm-pilot/", "runs/vlm-pilot/"))
            or "/" not in name and _PLAN.fullmatch(name) is not None)


def _manifest(name):
    return name == "data/vlm-pilot/dataset.json" or Path(name).name == "manifest.json"


def _destination_exists(path, entry):
    """Read-only preflight of every existing component, including missing targets."""
    path = _absolute(path)
    parent = os.open(path.anchor, _DIR_FLAGS)
    try:
        for part in path.parts[1:-1]:
            try:
                child = os.open(part, _DIR_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                return False
            except OSError as error:
                raise ValueError("Unsafe destination directory component") from error
            os.close(parent)
            parent = child
        try:
            mode = os.stat(path.name, dir_fd=parent, follow_symlinks=False).st_mode
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(mode):
            raise ValueError("Destination must be a non-symlink regular file")
    finally:
        os.close(parent)
    _hash_file(path, MAX_MEMBER_BYTES, entry)
    return True


def _copy_file(source, destination, entry):
    """Chunked/fsynced sibling temp -> atomic exclusive link -> remove temp only."""
    directory = _open_directory(destination.parent, create=True)
    temporary = ".vlm-restore-" + uuid4().hex + ".tmp"
    created = False
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        created = True
        with os.fdopen(fd, "wb") as output, _regular_stream(source) as stream:
            actual, _ = _hash_stream(stream, MAX_MEMBER_BYTES, entry["size_bytes"], output)
            if actual != _digest_fields(entry):
                raise ValueError("Staged bytes changed during publication")
            output.flush()
            os.fsync(output.fileno())
        # link() fails if any target already exists, even an identical file or link.
        os.link(temporary, destination.name, src_dir_fd=directory, dst_dir_fd=directory,
                follow_symlinks=False)
        os.fsync(directory)
    finally:
        try:
            if created:
                os.unlink(temporary, dir_fd=directory)
                os.fsync(directory)
        finally:
            os.close(directory)
    _hash_file(destination, MAX_MEMBER_BYTES, entry)


@contextmanager
def _restore_lock(checkout):
    directory = _open_directory(checkout / "runs/vlm-recovery", create=True)
    fd = None
    try:
        fd = os.open(".restore.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                     0o600, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_nlink != 1:
            raise ValueError("Unsafe restore lock file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another cooperating restore holds the checkout lock") from error
        yield
    finally:
        if fd is not None:
            # Closing releases flock. Never unlink the persistent lock inode.
            os.close(fd)
        os.close(directory)


def publish_artifacts(staging, checkout):
    """Revalidate ALL staged bytes, preflight ALL conflicts, then publish no-clobber.

    Return {artifact_files, new, skipped, manifests}; counts exclude staged source,
    BACKUP.json, the stage receipt, and the persistent lock. manifests counts both
    dataset.json and run manifest.json indexes (including identical skipped ones).
    Source/config/notebook/script/docs/dependency files are NEVER publish targets.
    A preflight failure creates no artifact files, only possibly the lock directory.
    """
    staging, checkout = _absolute(staging), _absolute(checkout)
    if staging == checkout or staging in checkout.parents:
        raise ValueError("Checkout must not be the staging directory or its descendant")
    fd = _open_directory(checkout)  # Require an existing checkout, never invent one.
    os.close(fd)
    with _restore_lock(checkout):
        entries = _verify_stage(staging)
        targets = {name: entry for name, entry in entries.items() if _publishable(name)}
        existing = {name: _destination_exists(checkout / name, entry)
                    for name, entry in targets.items()}
        # All non-manifest dependencies precede ALL manifests, irrespective of ZIP order.
        ordered = sorted(targets, key=lambda name: (_manifest(name), name))
        for manifest_phase in (False, True):
            for name in ordered:
                if _manifest(name) == manifest_phase and not existing[name]:
                    _copy_file(staging / name, checkout / name, targets[name])
            # Validate skipped dependencies too BEFORE publishing any manifest references.
            for name in ordered:
                if _manifest(name) == manifest_phase:
                    _hash_file(checkout / name, MAX_MEMBER_BYTES, targets[name])
        # A final all-file recheck also covers changes during the manifest phase.
        for name in ordered:
            _hash_file(checkout / name, MAX_MEMBER_BYTES, targets[name])
        return {"artifact_files": len(targets), "new": sum(not x for x in existing.values()),
                "skipped": sum(existing.values()),
                "manifests": sum(_manifest(name) for name in targets)}


def restore_archive(archive, receipt, staging, checkout):
    """Explicit NEW staging followed by locked, artifact-only no-clobber publication."""
    verification = stage_archive(archive, receipt, staging)
    publication = publish_artifacts(staging, checkout)
    return {"verification": verification, "publication": publication}