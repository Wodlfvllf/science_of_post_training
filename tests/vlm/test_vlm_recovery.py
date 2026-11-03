"""Toy-byte offline recovery tests; stdlib only, no science/model/backup imports.

These tests build realistic sealed BACKUP inventories but never execute any
archived code, decode an image, import a model fixture, or contact a service.
"""

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import stat
import struct
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "vlm_recovery_under_test", Path(__file__).resolve().parents[2] / "scripts/recover_vlm.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def reference(path, payload):
    return {"path": path, "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def toy_members():
    image = b"original PNG codec bytes, deliberately not decoded"
    capture = b"selected activation bytes, not model weights"
    record = canonical({"schema_version": 1, "run_id": "toy-run", "trial_id": "trial-1",
                        "attempt_id": "attempt-1", "status": "completed",
                        "artifacts": [reference("captures/trial-1.safetensors", capture)]})
    record_ref = reference("records/trial-1-" + hashlib.sha256(record).hexdigest() + ".json",
                           record)
    record_ref["trial_id"] = "trial-1"
    dataset = canonical({"schema_version": 1, "name": "reading-or-routing-v1",
                         "families": [{"images": [reference("images/scene.png", image)]}]})
    metadata = canonical({"schema_version": 1, "run_id": "toy-run",
                          "plan": {"dataset": json.loads(dataset)}})
    manifest = canonical({"schema_version": 1, "run_id": "toy-run", "completed": 1,
                          "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
                          "records": [record_ref], "updated_at": "2026-11-02T00:00:00Z"})
    run = "runs/vlm-pilot/toy/"
    # Deliberately put manifests first: publication order must not follow ZIP order.
    return {
        run + "manifest.json": manifest,
        "data/vlm-pilot/dataset.json": dataset,
        run + "metadata.json": metadata,
        run + record_ref["path"]: record,
        run + "captures/trial-1.safetensors": capture,
        "data/vlm-pilot/images/scene.png": image,
        "runs/vlm-pilot/report-toy/metadata.json": canonical({"report": "toy"}),
        "runs/vlm-pilot/report-toy/report.json": canonical({"cells": []}),
        "runs/vlm-pilot/report-toy/cells.csv": b"cell,value\nbinding,1\n",
        "runs/vlm-pilot/report-toy/REPORT.md": b"Toy report\n",
        "runs/vlm-pilot/plan-local.json": canonical({"schema_version": 1}),
        "plan-loader-v2.json": canonical({"locked": True}),
        "plan-gpu-uuid.json": canonical({"previous": True}),
        "plan-safe_extra-3.json": canonical({"historical": True}),
        "src/vlm_pilot/__init__.py": b"raise RuntimeError('NO PACKAGE IMPORTS')\n",
        "src/vlm_pilot/__main__.py": b"raise RuntimeError('NO MODULE EXECUTION')\n",
        "src/vlm_pilot/io.py": b"raise RuntimeError('ARCHIVED CODE MUST NOT RUN')\n",
        "src/vlm_pilot/weights.py": b"# Loader source text, not model weights\n",
        "notebooks/vlm_pilot.py": b"raise RuntimeError('NO NOTEBOOK EXECUTION')\n",
        "scripts/backup_vlm.py": b"raise RuntimeError('NO ARCHIVED IMPORTS')\n",
        "configs/vlm/pilot.yaml": b"archived: true\n",
        "configs/vlm/c2-development.json": b"{}",
        "configs/vlm/cue-context-development.json": b"{}",
        "scripts/vlm_decomposition_replication.py": b"# inert archived source\n",
        "scripts/vlm_cue_context.py": b"# inert archived source\n",
        "scripts/analyze_vlm_cue_context.py": b"# inert archived source\n",
        "experiments/vlm_posttraining/PAPER_DRAFT.md": b"Historical draft\n",
        "README.md": b"Historical README\n",
        "RESEARCH_PLAN.md": b"Historical plan\n",
        "pyproject.toml": b"# Historical dependency metadata, never installed\n",
        "requirements-tested.txt": b"not-installed==0\n",
    }


def inventory_for(members):
    return {"schema_version": 1, "captured_at": "2026-11-02T00:00:00+00:00",
            "run_names": ["runs/vlm-pilot/toy"],
            "completed_trials": {"runs/vlm-pilot/toy": 1},
            "files": len(members) + 1,
            "payload_size_bytes": sum(map(len, members.values())),
            "include_logs": False,
            "entries": [reference(name, payload) for name, payload in members.items()]}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vlm-recovery-tests-")
        self.addCleanup(self.temporary.cleanup)
        # macOS's OS-owned /var alias is normalized ONLY for our temporary test root.
        self.root = Path(self.temporary.name).resolve()
        self.archive = self.root / "snapshot.zip"
        self.receipt = self.root / "snapshot.zip.sha256"
        self.staging = self.root / "staging"
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.members = toy_members()

    def seal_receipt(self):
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.receipt.write_bytes(f"{digest}  {self.archive.name}\n".encode("ascii"))
        return digest

    def build(self, *, members=None, inventory=None, inventory_bytes=None,
              extras=(), method=zipfile.ZIP_STORED):
        members = self.members if members is None else members
        value = inventory_for(members) if inventory is None else inventory
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.archive, "w", compression=method) as archive:
                for name, payload in members.items():
                    archive.writestr(name, payload)
                for info, payload in extras:
                    archive.writestr(info, payload)
                archive.writestr("BACKUP.json", canonical(value) if inventory_bytes is None
                                 else inventory_bytes)
        self.seal_receipt()
        return value

    def stage(self):
        return recovery.stage_archive(self.archive, self.receipt, self.staging)

    def inspect_fails(self):
        with self.assertRaises((ValueError, OSError)):
            recovery.inspect_archive(self.archive, self.receipt)

    def assert_no_publication(self):
        files = [p.relative_to(self.checkout).as_posix()
                 for p in self.checkout.rglob("*") if p.is_file()]
        self.assertEqual(files, ["runs/vlm-recovery/.restore.lock"])

    def test_exact_inventory_inspection_is_read_only(self):
        expected = self.build(method=zipfile.ZIP_DEFLATED)
        before = set(self.root.rglob("*"))
        self.assertEqual(recovery.inspect_archive(self.archive, self.receipt), expected)
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assertFalse(self.staging.exists())

    def test_stage_all_members_and_verification_receipt(self):
        inventory = self.build()
        details = self.stage()
        self.assertEqual(details["archive"]["sha256"], self.seal_receipt())
        self.assertEqual(details["archive"]["name"], "snapshot.zip")
        self.assertEqual(details["archive"]["size_bytes"], self.archive.stat().st_size)
        self.assertEqual(details["files"], len(self.members) + 1)
        for name, payload in self.members.items():
            self.assertEqual((self.staging / name).read_bytes(), payload)
        self.assertEqual(json.loads((self.staging / "BACKUP.json").read_bytes()), inventory)
        self.assertEqual(json.loads((self.staging / recovery.VERIFICATION_NAME).read_bytes()),
                         details)
        self.assertNotIn(recovery.VERIFICATION_NAME,
                         {entry["path"] for entry in inventory["entries"]})

    def test_explicit_restore_api_counts_and_no_source_overwrite(self):
        self.build()
        preserved = {name: b"CURRENT CHECKOUT BYTES" for name in self.members
                     if not recovery._publishable(name)}
        for name, payload in preserved.items():
            put(self.checkout / name, payload)
        result = recovery.restore_archive(self.archive, self.receipt,
                                          self.staging, self.checkout)
        targets = {name for name in self.members if recovery._publishable(name)}
        self.assertEqual(set(result), {"verification", "publication"})
        self.assertEqual(result["publication"], {
            "artifact_files": len(targets), "new": len(targets), "skipped": 0, "manifests": 2})
        for name in targets:
            self.assertEqual((self.checkout / name).read_bytes(), self.members[name])
        for name, payload in preserved.items():
            self.assertEqual((self.checkout / name).read_bytes(), payload)
        self.assertFalse((self.checkout / "BACKUP.json").exists())
        self.assertFalse((self.checkout / recovery.VERIFICATION_NAME).exists())

    def test_publication_never_creates_archived_source_or_dependencies(self):
        self.build()
        self.stage()
        recovery.publish_artifacts(self.staging, self.checkout)
        for name in self.members:
            if not recovery._publishable(name):
                self.assertFalse((self.checkout / name).exists(), name)

    def test_identical_existing_files_skip_and_retry_is_idempotent(self):
        self.build()
        self.stage()
        name = "data/vlm-pilot/images/scene.png"
        put(self.checkout / name, self.members[name])
        first = recovery.publish_artifacts(self.staging, self.checkout)
        self.assertEqual(first["skipped"], 1)
        with mock.patch.object(recovery, "_copy_file", side_effect=AssertionError("no copying")):
            second = recovery.publish_artifacts(self.staging, self.checkout)
        self.assertEqual(second["new"], 0)
        self.assertEqual(second["skipped"], second["artifact_files"])

    def test_all_conflicts_preflight_before_any_artifact_copy(self):
        self.build()
        self.stage()
        # A late manifest conflict must block even missing image/record publications.
        name = "runs/vlm-pilot/toy/manifest.json"
        put(self.checkout / name, b"different previous boundary")
        with mock.patch.object(recovery, "_copy_file") as copying:
            with self.assertRaises(ValueError):
                recovery.publish_artifacts(self.staging, self.checkout)
            copying.assert_not_called()
        self.assertEqual((self.checkout / name).read_bytes(), b"different previous boundary")
        self.assertFalse((self.checkout / "data").exists())
        self.assertFalse((self.checkout / "plan-loader-v2.json").exists())

    def test_directory_conflicts_are_also_all_preflighted(self):
        self.build()
        self.stage()
        put(self.checkout / "runs/vlm-pilot/toy/records", b"not a directory")
        with mock.patch.object(recovery, "_copy_file") as copying:
            with self.assertRaises(ValueError):
                recovery.publish_artifacts(self.staging, self.checkout)
            copying.assert_not_called()
        self.assertFalse((self.checkout / "data").exists())

    def test_manifests_last_after_metadata_records_captures_and_images(self):
        self.build()
        self.stage()
        order, original = [], recovery._copy_file

        def copying(source, destination, entry):
            name = destination.relative_to(self.checkout).as_posix()
            if recovery._manifest(name):
                for dependency in self.members:
                    if recovery._publishable(dependency) and not recovery._manifest(dependency):
                        self.assertTrue((self.checkout / dependency).is_file(), dependency)
            order.append(name)
            return original(source, destination, entry)

        with mock.patch.object(recovery, "_copy_file", side_effect=copying):
            recovery.publish_artifacts(self.staging, self.checkout)
        self.assertTrue(all(recovery._manifest(name) for name in order[-2:]))
        self.assertFalse(any(recovery._manifest(name) for name in order[:-2]))

    def test_receipt_strict_format_and_exact_basename(self):
        self.build()
        original = self.receipt.read_bytes()
        variants = [original.replace(b"  ", b" "), original[:-1], original + b"\n",
                    original.replace(b"\n", b"\r\n"), original.upper(),
                    original.replace(b"snapshot.zip", b"other.zip"),
                    original.replace(b"snapshot.zip", b"../snapshot.zip"),
                    b" " + original, original + b"other receipt", b"x" * 1024,
                    b"0" * 64 + b"  snapshot.zip\n"]
        for payload in variants:
            with self.subTest(payload=payload):
                self.receipt.write_bytes(payload)
                self.inspect_fails()

    def test_full_archive_corruption_rejected_before_stage_creation(self):
        self.build()
        payload = bytearray(self.archive.read_bytes())
        payload[len(payload) // 2] ^= 1
        self.archive.write_bytes(payload)
        with self.assertRaises(ValueError):
            self.stage()
        self.assertFalse(self.staging.exists())

    def test_crc_corruption_even_with_resealed_full_zip(self):
        self.build()
        with zipfile.ZipFile(self.archive) as archive:
            info = archive.getinfo("data/vlm-pilot/images/scene.png")
        payload = bytearray(self.archive.read_bytes())
        name_length, extra_length = struct.unpack_from("<HH", payload, info.header_offset + 26)
        offset = info.header_offset + 30 + name_length + extra_length
        payload[offset] ^= 1
        self.archive.write_bytes(payload)
        self.seal_receipt()
        self.inspect_fails()
        with self.assertRaises(ValueError):
            self.stage()
        self.assertFalse(self.staging.exists())

    def test_inventory_mismatch_cases(self):
        original = inventory_for(self.members)
        for kind in ("sha", "size", "bool-size", "negative-size", "missing", "extra",
                     "duplicate", "files", "total", "schema", "self", "run", "counts"):
            with self.subTest(kind=kind):
                value = copy.deepcopy(original)
                if kind == "sha":
                    value["entries"][0]["sha256"] = "0" * 64
                elif kind == "size":
                    value["entries"][0]["size_bytes"] += 1
                    value["payload_size_bytes"] += 1
                elif kind == "bool-size":
                    value["entries"][0]["size_bytes"] = True
                elif kind == "negative-size":
                    value["entries"][0]["size_bytes"] = -1
                elif kind == "missing":
                    value["entries"].pop()
                    value["files"] -= 1
                    value["payload_size_bytes"] = sum(e["size_bytes"] for e in value["entries"])
                elif kind == "extra":
                    value["entries"].append(reference("plan-missing.json", b"{}"))
                    value["files"] += 1
                    value["payload_size_bytes"] += 2
                elif kind == "duplicate":
                    value["entries"].append(value["entries"][0])
                    value["files"] += 1
                    value["payload_size_bytes"] += value["entries"][0]["size_bytes"]
                elif kind == "files":
                    value["files"] = True
                elif kind == "total":
                    value["payload_size_bytes"] += 1
                elif kind == "schema":
                    value["schema_version"] = True
                elif kind == "self":
                    value["entries"][0]["path"] = "BACKUP.json"
                elif kind == "run":
                    value["run_names"] = ["../toy"]
                else:
                    value["completed_trials"]["runs/vlm-pilot/toy"] = True
                self.build(inventory=value)
                self.inspect_fails()

    def test_inventory_json_must_be_canonical_finite_and_unique(self):
        value = inventory_for(self.members)
        sealed = canonical(value)
        variants = [b" " + sealed, sealed + b"\n", json.dumps(value, indent=2).encode(),
                    b'{"schema_version":1,' + sealed[1:],
                    sealed.replace(b'"sha256":', b'"size_bytes":0,"sha256":', 1),
                    sealed.replace(b'"include_logs":false', b'"include_logs":NaN'),
                    sealed.replace(b'"include_logs":false', b'"include_logs":Infinity'),
                    sealed.replace(b'"include_logs":false', b'"include_logs":1e9999'),
                    b"\xef\xbb\xbf" + sealed, b"\xff", b"[]"]
        for payload in variants:
            with self.subTest(payload=payload[:80]):
                self.build(inventory_bytes=payload)
                self.inspect_fails()

    def test_unsafe_and_out_of_scope_archive_names(self):
        names = ["../escape.json", "/absolute.json", "C:/absolute.json",
                 "data\\vlm-pilot\\dataset.json", "data//vlm-pilot/dataset.json",
                 "data/./vlm-pilot/dataset.json", "data/vlm-pilot/../evil.json",
                 ".git/config", ".env", "runs/vlm-pilot/.hidden/report.json",
                 "runs/vlm-pilot/env/report.json", "runs/vlm-pilot/models/config.json",
                 "runs/vlm-pilot/weights/model.safetensors",
                 "runs/vlm-pilot/model/config.json", "runs/vlm-pilot/cache/report.json",
                 "runs/vlm-pilot/env.json", "runs/vlm-pilot/weights.json",
                 "runs/vlm-pilot/report-secrets.json", "runs/vlm-pilot/request.json",
                 "runs/posttrain/manifest.json", "runs/vlm-pilot/evil.py",
                 "data/vlm-pilot/images/scene.pkl", "src/interp/io.py",
                 "scripts/arbitrary.sh", "plan-credentials.json", "plan.bad.json",
                 "runs/vlm-pilot/odd./report.json", "runs/vlm-pilot/odd /report.json",
                 "runs/vlm-pilot/\u00e9/report.json", "runs/vlm-pilot/empty/",
                 "runs/vlm-pilot/hi\x00there.json"]
        for name in names:
            with self.subTest(name=name):
                members = {**self.members, name: b"unsafe"}
                self.build(members=members)
                self.inspect_fails()

    def test_duplicate_names_and_case_normalization_collisions(self):
        self.build(extras=[("README.md", b"duplicate")])
        self.inspect_fails()
        members = {**self.members, "runs/vlm-pilot/TOY/summary.json": b"{}"}
        self.build(members=members)
        self.inspect_fails()

    def test_explicit_zip_directories_are_rejected(self):
        self.build(extras=[("data/vlm-pilot/images/", b"")])
        self.inspect_fails()

    def test_symlink_and_other_nonregular_zip_attributes(self):
        for mode in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFCHR,
                     stat.S_IFBLK, stat.S_IFDIR):
            with self.subTest(mode=mode):
                info = zipfile.ZipInfo("runs/vlm-pilot/report-special.json")
                info.create_system = 3
                info.external_attr = (mode | 0o600) << 16
                self.build(extras=[(info, b"../../outside")])
                self.inspect_fails()
        info = zipfile.ZipInfo("runs/vlm-pilot/report-dos.json")
        info.external_attr = 0x10
        self.build(extras=[(info, b"{}")])
        self.inspect_fails()

    def test_encrypted_flag_or_unsupported_method_rejected(self):
        for field in ("encrypted", "strong-encrypted", "method", "local-only-encrypted",
                      "local-only-method"):
            with self.subTest(field=field):
                self.build()
                payload = bytearray(self.archive.read_bytes())
                local, central = payload.index(b"PK\x03\x04"), payload.index(b"PK\x01\x02")
                if field in ("method", "local-only-method"):
                    struct.pack_into("<H", payload, local + 8, 99)
                    if field == "method":
                        struct.pack_into("<H", payload, central + 10, 99)
                else:
                    flag = 0x40 if field == "strong-encrypted" else 1
                    struct.pack_into("<H", payload, local + 6, flag)
                    if field != "local-only-encrypted":
                        struct.pack_into("<H", payload, central + 8, flag)
                self.archive.write_bytes(payload)
                self.seal_receipt()
                self.inspect_fails()

    def test_real_bzip2_method_is_not_accepted(self):
        self.build(method=zipfile.ZIP_BZIP2)
        self.inspect_fails()

    def test_limits_without_allocating_large_payloads(self):
        self.build()
        limits = {"MAX_ARCHIVE_BYTES": self.archive.stat().st_size - 1,
                  "MAX_TOTAL_BYTES": 1,
                  "MAX_MEMBER_BYTES": 1,
                  "MAX_MEMBERS": len(self.members),
                  "MAX_INVENTORY_BYTES": 1}
        for name, limit in limits.items():
            with self.subTest(limit=name), mock.patch.object(recovery, name, limit):
                self.inspect_fails()
        with mock.patch.object(recovery, "MAX_ARCHIVE_BYTES", 10**9):
            payload = bytearray(self.archive.read_bytes())
            central = payload.index(b"PK\x01\x02")
            struct.pack_into("<I", payload, central + 24, 64 * 1024**2 + 1)
            self.archive.write_bytes(payload)
            self.seal_receipt()
            self.inspect_fails()

    def test_member_count_is_bounded_before_zipfile_parsing(self):
        self.build()
        payload = bytearray(self.archive.read_bytes())
        footer = payload.rindex(b"PK\x05\x06")
        struct.pack_into("<HH", payload, footer + 8, 30_001, 30_001)
        self.archive.write_bytes(payload)
        self.seal_receipt()
        with mock.patch.object(recovery.zipfile, "ZipFile",
                               side_effect=AssertionError("must not parse oversized directory")):
            self.inspect_fails()

    def test_forged_central_count_or_bounds_rejected(self):
        for field in ("count", "offset", "trailing"):
            with self.subTest(field=field):
                self.build()
                payload = bytearray(self.archive.read_bytes())
                footer = payload.rindex(b"PK\x05\x06")
                if field == "count":
                    struct.pack_into("<HH", payload, footer + 8, 2, 2)
                elif field == "offset":
                    struct.pack_into("<I", payload, footer + 16, 0xFFFF0000)
                else:
                    payload.extend(b"unrecognized trailing bytes")
                self.archive.write_bytes(payload)
                self.seal_receipt()
                self.inspect_fails()

    def test_bounded_small_zip64_end_records_are_supported(self):
        expected = self.build()
        payload = self.archive.read_bytes()
        footer = payload.rindex(b"PK\x05\x06")
        end = bytearray(payload[footer:])
        count = struct.unpack_from("<H", end, 10)[0]
        directory_size, directory_offset = struct.unpack_from("<II", end, 12)
        record = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0,
                             count, count, directory_size, directory_offset)
        locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, footer, 1)
        struct.pack_into("<HHII", end, 8, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
        self.archive.write_bytes(payload[:footer] + record + locator + end)
        self.seal_receipt()
        self.assertEqual(recovery.inspect_archive(self.archive, self.receipt), expected)

    def test_force_zip64_local_header_is_supported(self):
        expected = inventory_for(self.members)
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in self.members.items():
                with archive.open(name, "w", force_zip64=True) as stream:
                    stream.write(payload)
            archive.writestr("BACKUP.json", canonical(expected))
        self.seal_receipt()
        self.assertEqual(recovery.inspect_archive(self.archive, self.receipt), expected)

    def test_staging_must_be_new_even_when_empty(self):
        self.build()
        self.staging.mkdir()
        with self.assertRaises(ValueError):
            self.stage()
        self.assertEqual(list(self.staging.iterdir()), [])

    def test_staging_parent_symlink_and_archive_or_receipt_symlinks(self):
        self.build()
        real = self.root / "real"
        real.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(real, target_is_directory=True)
        with self.assertRaises(ValueError):
            recovery.stage_archive(self.archive, self.receipt, alias / "stage")
        self.assertFalse((real / "stage").exists())
        for path in (self.archive, self.receipt):
            original = path.with_name("original-" + path.name)
            path.rename(original)
            path.symlink_to(original)
            try:
                self.inspect_fails()
            finally:
                path.unlink()
                original.rename(path)
        with self.assertRaises(ValueError):
            recovery.inspect_archive(alias / self.archive.name, self.receipt)

    def test_stage_publish_revalidates_all_bytes_including_unpublished_source(self):
        for name in ("data/vlm-pilot/images/scene.png", "src/vlm_pilot/io.py",
                     "runs/vlm-pilot/toy/manifest.json"):
            with self.subTest(name=name):
                self.build()
                stage = self.root / ("stage-" + hashlib.sha256(name.encode()).hexdigest()[:8])
                recovery.stage_archive(self.archive, self.receipt, stage)
                target = stage / name
                original = target.read_bytes()
                target.write_bytes(b"!" + original[1:])  # Same size, different SHA.
                with mock.patch.object(recovery, "_copy_file") as copying:
                    with self.assertRaises(ValueError):
                        recovery.publish_artifacts(stage, self.checkout)
                    copying.assert_not_called()
        self.assert_no_publication()

    def test_stage_missing_extra_symlink_and_inventory_tampering(self):
        for kind in ("missing", "extra", "empty-directory", "symlink", "inventory", "receipt"):
            with self.subTest(kind=kind):
                self.build()
                stage = self.root / ("stage-" + kind)
                recovery.stage_archive(self.archive, self.receipt, stage)
                image = stage / "data/vlm-pilot/images/scene.png"
                if kind == "missing":
                    image.unlink()
                elif kind == "extra":
                    put(stage / "runs/vlm-pilot/extra.json", b"{}")
                elif kind == "empty-directory":
                    (stage / "extra-directory").mkdir()
                elif kind == "symlink":
                    image.unlink()
                    image.symlink_to(self.receipt)
                elif kind == "inventory":
                    (stage / "BACKUP.json").write_bytes(b"{}")
                else:
                    (stage / recovery.VERIFICATION_NAME).write_bytes(b"{}")
                with self.assertRaises(ValueError):
                    recovery.publish_artifacts(stage, self.checkout)
        self.assert_no_publication()

    def test_symlink_any_destination_component_is_a_preflight_failure(self):
        self.build()
        self.stage()
        outside = self.root / "outside"
        outside.mkdir()
        alias_checkout = self.root / "checkout-link"
        alias_checkout.symlink_to(self.checkout, target_is_directory=True)
        with self.assertRaises(ValueError):
            recovery.publish_artifacts(self.staging, alias_checkout)
        (self.checkout / "data").symlink_to(outside, target_is_directory=True)
        with mock.patch.object(recovery, "_copy_file") as copying:
            with self.assertRaises(ValueError):
                recovery.publish_artifacts(self.staging, self.checkout)
            copying.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])
        (self.checkout / "data").unlink()
        root_plan = self.checkout / "plan-loader-v2.json"
        root_plan.symlink_to(self.staging / "plan-loader-v2.json")
        with self.assertRaises(ValueError):
            recovery.publish_artifacts(self.staging, self.checkout)
        self.assertTrue(root_plan.is_symlink())

    def test_staged_directory_symlink_is_rejected(self):
        self.build()
        self.stage()
        source = self.staging / "data/vlm-pilot/images"
        moved = self.root / "moved-images"
        source.rename(moved)
        source.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(ValueError):
            recovery.publish_artifacts(self.staging, self.checkout)
        self.assert_no_publication()

    def test_restore_lock_blocks_competing_restores_and_is_not_removed(self):
        self.build()
        self.stage()
        lock = self.checkout / "runs/vlm-recovery/.restore.lock"
        lock.parent.mkdir(parents=True)
        with lock.open("w+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(RuntimeError):
                recovery.publish_artifacts(self.staging, self.checkout)
            self.assertFalse((self.checkout / "data").exists())
        recovery.publish_artifacts(self.staging, self.checkout)
        self.assertTrue(lock.is_file())

    def test_symlink_lock_fails_without_modifying_target(self):
        self.build()
        self.stage()
        outside = self.root / "outside-lock"
        outside.write_bytes(b"untouched")
        lock = self.checkout / "runs/vlm-recovery/.restore.lock"
        lock.parent.mkdir(parents=True)
        lock.symlink_to(outside)
        with self.assertRaises((ValueError, OSError)):
            recovery.publish_artifacts(self.staging, self.checkout)
        self.assertEqual(outside.read_bytes(), b"untouched")

    def test_atomic_copy_does_not_replace_a_racing_destination(self):
        source = self.root / "source"
        destination = self.checkout / "result.json"
        source.write_bytes(b"complete result")
        entry = reference("result.json", source.read_bytes())
        original_link = os.link

        def race(*args, **kwargs):
            destination.write_bytes(b"other writer")
            return original_link(*args, **kwargs)

        with mock.patch.object(recovery.os, "link", side_effect=race):
            with self.assertRaises(FileExistsError):
                recovery._copy_file(source, destination, entry)
        self.assertEqual(destination.read_bytes(), b"other writer")
        self.assertEqual([p.name for p in self.checkout.iterdir()], ["result.json"])

    def test_interrupted_atomic_copy_leaves_no_partial_destination(self):
        source = self.root / "source"
        destination = self.checkout / "result.json"
        source.write_bytes(b"complete result")
        entry = reference("result.json", source.read_bytes())
        original = recovery._hash_stream

        def interrupted(stream, maximum, expected_size=None, output=None):
            if output is not None:
                output.write(b"partial")
                raise OSError("simulated disk failure")
            return original(stream, maximum, expected_size, output)

        with mock.patch.object(recovery, "_hash_stream", side_effect=interrupted):
            with self.assertRaises(OSError):
                recovery._copy_file(source, destination, entry)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.checkout.iterdir()), [])
        recovery._copy_file(source, destination, entry)
        self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_final_publication_hash_recheck_detects_changed_skipped_file(self):
        self.build()
        self.stage()
        image = "data/vlm-pilot/images/scene.png"
        put(self.checkout / image, self.members[image])
        original, changed = recovery._copy_file, False

        def copying(*args):
            nonlocal changed
            original(*args)
            if not changed:
                (self.checkout / image).write_bytes(b"changed by noncooperating writer")
                changed = True

        with mock.patch.object(recovery, "_copy_file", side_effect=copying):
            with self.assertRaises(ValueError):
                recovery.publish_artifacts(self.staging, self.checkout)
        self.assertFalse((self.checkout / "data/vlm-pilot/dataset.json").exists())
        self.assertFalse((self.checkout / "runs/vlm-pilot/toy/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()