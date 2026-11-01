"""Offline real safetensors layout tests: no CUDA, checkpoint models or HF requests."""

from __future__ import annotations

import json
import traceback
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from safetensors.torch import save_file

from vlm_pilot.audit import audit_components
from vlm_pilot.io import digest
from vlm_pilot.runtime import _weight_files
from vlm_pilot.schemas import PilotError
from vlm_pilot.weights import SafetensorSource

REVISION = "a" * 40
PROJECTION = "language_model.model.layers.0.self_attn.q_proj.weight"


def checkpoint_files(tmp_path, monkeypatch, *, sharded=False, allow_downloads=False):
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    tensors = {PROJECTION: tensor, "vision_model.weight": torch.ones(4),
               "mlp1.weight": torch.full((2, 2), 2.)}
    filename = "model-00001-of-00001.safetensors" if sharded else "model.safetensors"
    file = tmp_path / filename
    save_file(tensors, str(file))
    index = tmp_path / "model.safetensors.index.json"
    if sharded:
        index.write_text(json.dumps({"weight_map": dict.fromkeys(tensors, filename),
                                     "metadata": {"total_size": 80}}))
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["repo_id"] == "fixture/checkpoint"
        assert kwargs["revision"] == REVISION
        assert kwargs["repo_type"] == "model"
        assert kwargs["local_files_only"] is not allow_downloads
        requested = tmp_path / kwargs["filename"]
        if not requested.exists():
            error = EntryNotFoundError if allow_downloads else LocalEntryNotFoundError
            raise error("Expected pinned fixture file absence")
        return str(requested)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return tensors, file, index, calls


@pytest.mark.parametrize("allow_downloads", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_layouts_are_lazy_pinned_and_selected_tensor_only(tmp_path, monkeypatch, sharded, allow_downloads):
    tensors, file, _, calls = checkpoint_files(tmp_path, monkeypatch, sharded=sharded,
                                              allow_downloads=allow_downloads)
    source = SafetensorSource("fixture/checkpoint", REVISION, allow_downloads)
    assert calls == [] and source.layout is None

    def forbidden(*args, **kwargs):
        raise AssertionError("Layout inspection must not load a pickle or model")

    monkeypatch.setattr(torch, "load", forbidden)
    mapping = source._load_index()
    assert mapping == dict.fromkeys(tensors, file.name)
    assert source.layout == ("sharded" if sharded else "single_file")
    result = source.read_tensor(PROJECTION)
    assert torch.equal(result, tensors[PROJECTION]) and result.device.type == "cpu"
    result.zero_()
    assert torch.equal(source.read_tensor(PROJECTION), tensors[PROJECTION])
    assert sum(c["filename"] == "model.safetensors.index.json" for c in calls) == 1


@pytest.mark.parametrize("sharded", [False, True])
def test_fingerprint_covers_actual_file_in_both_layouts(tmp_path, monkeypatch, sharded):
    _, file, _, _ = checkpoint_files(tmp_path, monkeypatch, sharded=sharded)
    checkpoint = SimpleNamespace(model_id="fixture/checkpoint", revision=REVISION)
    report = _weight_files(checkpoint, False)
    assert report == [{"name": file.name, "size_bytes": file.stat().st_size,
                       "sha256": digest(file.read_bytes()),
                       "layout": "sharded" if sharded else "single_file"}]


def test_component_audit_accepts_single_file_checkpoints(tmp_path, monkeypatch):
    checkpoint_files(tmp_path, monkeypatch)
    checkpoint = SimpleNamespace(model_id="fixture/checkpoint", revision=REVISION,
                                 family="2b", backend="internvl", model_dump=lambda **kw: {
                                     "model_id": "fixture/checkpoint", "revision": REVISION})
    report = audit_components(checkpoint, checkpoint)
    assert report["bitexact_equal"] and len(report["tensors"]) == 2
    assert not report["fixture_only"]


def test_corrupt_existing_index_does_not_fallback_to_single_file(tmp_path, monkeypatch):
    _, _, index, calls = checkpoint_files(tmp_path, monkeypatch)
    index.write_text('{"weight_map": {"unsafe": "../escape.safetensors"}}')
    with pytest.raises(ValueError, match="unsafe"):
        SafetensorSource("fixture/checkpoint", REVISION)._load_index()
    assert [c["filename"] for c in calls] == ["model.safetensors.index.json"]


@pytest.mark.parametrize("failure", [ConnectionError, PermissionError, TimeoutError])
def test_transport_failures_never_infer_single_file_layout(monkeypatch, failure):
    calls = []

    def failed(**kwargs):
        calls.append(kwargs["filename"])
        raise failure("secret-containing dependency exception must not be displayed")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", failed)
    with pytest.raises(PilotError, match="model.safetensors.index.json") as error:
        SafetensorSource("fixture/checkpoint", REVISION, True)._load_index()
    assert "secret-containing" not in str(error.value)
    assert "secret-containing" not in "".join(traceback.format_exception(error.value))
    assert error.value.__cause__ is None and error.value.__suppress_context__
    assert calls == ["model.safetensors.index.json"]


def test_online_local_cache_miss_is_not_remote_absence(tmp_path, monkeypatch):
    _, _, _, calls = checkpoint_files(tmp_path, monkeypatch, allow_downloads=True)

    def failed_head(**kwargs):
        calls.append(kwargs)
        raise LocalEntryNotFoundError("Online request failed and index is not cached")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", failed_head)
    with pytest.raises(PilotError, match="cannot infer checkpoint"):
        SafetensorSource("fixture/checkpoint", REVISION, True)._load_index()
    assert len(calls) == 1


def test_missing_both_files_has_safe_asset_error(tmp_path, monkeypatch):
    _, file, _, calls = checkpoint_files(tmp_path, monkeypatch)
    file.unlink()
    with pytest.raises(PilotError, match="asset unavailable: model.safetensors"):
        SafetensorSource("fixture/checkpoint", REVISION)._load_index()
    assert [c["filename"] for c in calls] == ["model.safetensors.index.json", "model.safetensors"]


@pytest.mark.parametrize("cap", ["MAX_INDEX_BYTES", "MAX_INDEX_ENTRIES", "MAX_SHARD_BYTES", "MAX_MODEL_BYTES"])
def test_single_file_header_and_file_limits(tmp_path, monkeypatch, cap):
    checkpoint_files(tmp_path, monkeypatch)
    monkeypatch.setattr(SafetensorSource, cap, 1)
    with pytest.raises(ValueError, match="cap|header"):
        SafetensorSource("fixture/checkpoint", REVISION)._load_index()


def test_single_file_tensor_limits_and_header_corruption(tmp_path, monkeypatch):
    _, file, _, _ = checkpoint_files(tmp_path, monkeypatch)
    source = SafetensorSource("fixture/checkpoint", REVISION)
    monkeypatch.setattr(source, "MAX_TENSOR_BYTES", 8)
    with pytest.raises(ValueError, match="size cap"):
        source.read_tensor(PROJECTION)
    file.write_bytes(b"bad")
    with pytest.raises(ValueError, match="header") as error:
        SafetensorSource("fixture/checkpoint", REVISION)._load_index()
    assert error.value.__context__ is None