"""CPU-only UUID/occupancy regression tests; no CUDA initialization or model loads."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from uuid import UUID

import pytest

from vlm_pilot import runtime
from vlm_pilot.schemas import PilotError

TEXT = "12345678-90ab-cdef-1234-567890abcdef"
IDENTIFIER = "GPU-" + TEXT


class CUuuid:
    """Mirror PyTorch's pybind _CUuuid: bytes property returns uint8 vector."""

    def __init__(self, raw):
        self.bytes = list(raw)

    def __str__(self):
        return str(UUID(bytes=bytes(self.bytes)))


class TextOnlyUuid:
    def __str__(self):
        return TEXT


@pytest.mark.parametrize("value", [
    TEXT, TEXT.upper(), TEXT.replace("-", ""), IDENTIFIER, IDENTIFIER.encode(),
    UUID(TEXT), UUID(TEXT).bytes, bytearray(UUID(TEXT).bytes), memoryview(UUID(TEXT).bytes),
    CUuuid(UUID(TEXT).bytes), TextOnlyUuid(),
])
def test_uuid_representations_use_same_actual_identifier(value):
    assert runtime._normalize_gpu_uuid(value) == IDENTIFIER


def test_modern_mig_text_keeps_exact_prefix():
    assert runtime._normalize_gpu_uuid("MIG-" + TEXT) == "MIG-" + TEXT


@pytest.mark.parametrize("value", [
    None, "", "0", "GPU-1234", "GPU-0", "cuda:0", object(), b"invalid", "0" * 32,
    "GPU-" + "0" * 32, "MIG-GPU-" + TEXT + "/1/0", "GPU-" + TEXT + ";echo unsafe",
    SimpleNamespace(bytes=[0] * 16), SimpleNamespace(bytes=[1] * 15),
    SimpleNamespace(bytes=[1] * 15 + [256]), SimpleNamespace(bytes=[1] * 15 + [True]),
    SimpleNamespace(bytes=b"short"), SimpleNamespace(bytes="not-a-byte-vector"),
])
def test_invalid_uuid_never_falls_back_to_gpu_ordinal(value):
    with pytest.raises(PilotError, match="No GPU ordinal fallback"):
        runtime._normalize_gpu_uuid(value)


def test_logical_device_selection_uses_its_properties_not_visible_ordinal(monkeypatch):
    selected = []

    def properties(device):
        selected.append(device)
        return SimpleNamespace(uuid=CUuuid(UUID(TEXT).bytes))

    monkeypatch.setattr(runtime.torch.cuda, "get_device_properties", properties)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,3")
    assert runtime.gpu_identifier("cuda:1") == IDENTIFIER
    assert selected == ["cuda:1"]


def _query(monkeypatch, *, stdout="", returncode=0):
    calls = []
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(runtime.torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(uuid=CUuuid(UUID(TEXT).bytes)))
    monkeypatch.setattr(runtime.os, "getpid", lambda: 123)

    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=stdout, returncode=returncode)

    monkeypatch.setattr(runtime.subprocess, "run", execute)
    return calls


@pytest.mark.parametrize("stdout", ["", "\n", "123\n", " 123 \n"])
def test_idle_or_own_gpu_context_is_verified_with_uuid(monkeypatch, stdout):
    calls = _query(monkeypatch, stdout=stdout)
    assert runtime.verify_idle_gpu("cuda:0") == IDENTIFIER
    argv, options = calls[0]
    assert argv[:3] == ["/usr/bin/nvidia-smi", "-i", IDENTIFIER]
    assert options["timeout"] == 15 and not options["check"]


@pytest.mark.parametrize("stdout", ["456\n", "123\n456\n"])
def test_other_worker_remains_protected(monkeypatch, stdout):
    _query(monkeypatch, stdout=stdout)
    with pytest.raises(PilotError, match="another compute process"):
        runtime.verify_idle_gpu("cuda:0")


@pytest.mark.parametrize("stdout", ["N/A", "not supported", "0", "-1", "123,456"])
def test_unverified_process_output_is_not_idle(monkeypatch, stdout):
    _query(monkeypatch, stdout=stdout)
    with pytest.raises(PilotError, match="unrecognized output"):
        runtime.verify_idle_gpu("cuda:0")


def test_missing_smi_nonzero_exit_and_timeout_fail_closed(monkeypatch):
    _query(monkeypatch, returncode=1)
    with pytest.raises(PilotError, match="occupancy"):
        runtime.verify_idle_gpu("cuda:0")
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    with pytest.raises(PilotError, match="missing"):
        runtime.verify_idle_gpu("cuda:0")
    _query(monkeypatch)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 15)

    monkeypatch.setattr(runtime.subprocess, "run", timeout)
    with pytest.raises(PilotError, match="occupancy"):
        runtime.verify_idle_gpu("cuda:0")