"""Offline CPU projector contracts; tests are supplied, not executed by the agent.

Tiny LayerNorm/Linear/GELU/Linear models and local safetensors fixtures exercise
only diagnostic mechanics. Hub calls are intercepted, never real downloads.
These are not GPU/device-failure qualification or behavioral research results.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/vlm_projector_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("projector_diagnostic_tests", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
NAMES = diagnostic.PROJECTOR_NAMES
REVISION = "a" * 40


class TinyModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 6), nn.GELU(), nn.Linear(6, 3))
        self.untouched = nn.Linear(3, 3)
        self.to(device="cpu", dtype=dtype)

    def forward(self, values):
        return self.untouched(self.mlp1(values))


def tiny(dtype=torch.float32):
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        return TinyModel(dtype).eval()


@pytest.fixture
def model():
    return tiny()


def snapshot(model):
    return {name: tensor.detach().clone() for name, tensor in model.named_parameters()}


def mapping(model, changed=True):
    native = snapshot(model)
    result = {name: native[name] for name in NAMES}
    if changed:
        for tensor in result.values():
            tensor.add_(torch.linspace(.03125, .25, tensor.numel()).reshape(tensor.shape))
    return result


def assert_snapshot(model, expected):
    for name, parameter in model.named_parameters():
        assert diagnostic._bitexact(parameter, expected[name]), name


def assert_released(swap):
    assert swap._parts == [] and swap._module is None
    assert swap.model is None and swap.mapping is None
    assert not swap.summary["active"]
    assert swap not in list(diagnostic._ACTIVE_PROJECTORS.values())
    json.dumps(swap.summary, allow_nan=False)


def test_import_is_lazy_stdlib_only_and_has_no_automatic_execution(monkeypatch):
    original = builtins.__import__
    blocked = {"torch", "numpy", "safetensors", "huggingface_hub", "vlm_pilot", "transformers"}

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise AssertionError(f"import must be lazy: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("projector_lazy_import", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__all__ == ["ProjectorSwap", "load_projector"]
    assert not hasattr(module, "main") and not module._ACTIVE_PROJECTORS
    # Construction also remains dependency-free, even with a not-yet-valid model.
    swap = module.ProjectorSwap(object(), dict.fromkeys(module.PROJECTOR_NAMES))
    assert not swap._entered


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_endpoint_identity_cpu_owned_backups_and_exact_restoration(dtype):
    model = tiny(dtype)
    original, source = snapshot(model), mapping(model)
    rng = torch.random.get_rng_state().clone()
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    grad_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    forwards = {name: module.forward.__func__ for name, module in model.named_modules()}
    hooks = {name: (dict(module._forward_hooks), dict(module._forward_pre_hooks))
             for name, module in model.named_modules()}
    swap = diagnostic.ProjectorSwap(model, source)
    with swap as active:
        assert active is swap and swap.summary["active"]
        for part in swap._parts:
            assert part.original.device.type == "cpu" and not part.original.requires_grad
            assert part.original.data_ptr() != part.parameter.data_ptr()
            assert diagnostic._bitexact(part.parameter, source[part.name])
        for name in original.keys() - set(NAMES):
            assert diagnostic._bitexact(dict(model.named_parameters())[name], original[name])
        # No retained replacement or backup aliases the caller's mapping.
        applied = {part.name: part.replacement.clone() for part in swap._parts}
        for value in source.values():
            value.zero_()
        for part in swap._parts:
            assert diagnostic._bitexact(part.parameter, applied[part.name])
            assert diagnostic._bitexact(part.original, original[part.name])
    assert_snapshot(model, original)
    assert identities == {name: id(p) for name, p in model.named_parameters()}
    assert grad_flags == {name: p.requires_grad for name, p in model.named_parameters()}
    assert forwards == {name: module.forward.__func__ for name, module in model.named_modules()}
    assert hooks == {name: (dict(module._forward_hooks), dict(module._forward_pre_hooks))
                     for name, module in model.named_modules()}
    assert all(not module.training for module in model.modules())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert swap.summary["restored_bitexact"] and swap.summary["changedtensors"] == 6
    assert_released(swap)


def test_endpoint_is_direct_copy_not_delta_arithmetic(model):
    with torch.no_grad():
        model.mlp1[0].weight.fill_(1e20)
    original, source = snapshot(model), mapping(model, changed=False)
    source["mlp1.0.weight"].fill_(1.0)
    # Float32 subtraction/addition would erase this source endpoint.
    assert not torch.equal(original["mlp1.0.weight"] +
                           (source["mlp1.0.weight"] - original["mlp1.0.weight"]),
                           source["mlp1.0.weight"])
    with diagnostic.ProjectorSwap(model, source):
        assert diagnostic._bitexact(model.mlp1[0].weight, source["mlp1.0.weight"])
    assert_snapshot(model, original)


@pytest.mark.parametrize("sham", [False, True])
def test_self_swap_has_bitexact_parameters_output_and_zero_norms(model, sham):
    source, original = mapping(model, changed=False), snapshot(model)
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    with torch.no_grad():
        before = model(values)
        with diagnostic.ProjectorSwap(model, source, sham=sham) as swap:
            assert diagnostic._bitexact(model(values), before)
            assert_snapshot(model, original)
    assert swap.summary["changed_parameter_count"] == 0
    assert swap.summary["changedtensors"] == 0
    assert swap.summary["delta_frobnorm"] == swap.summary["source_delta_frobnorm"] == 0
    assert swap.summary["hashes"]["native"] == swap.summary["hashes"]["applied"]


def test_sham_is_local_rng_delta_permutation_not_weight_permutation(model):
    source, original = mapping(model), snapshot(model)
    torch_rng, python_rng = torch.random.get_rng_state().clone(), random.getstate()
    generator = torch.Generator(device="cpu").manual_seed(123)
    expected = {}
    for name in NAMES:
        a, b = original[name].double(), source[name].double()
        permutation = torch.randperm(a.numel(), generator=generator)
        expected[name] = (a + (b - a).reshape(-1)[permutation].reshape(a.shape)).to(original[name].dtype)
    checksums = []
    for _ in range(2):
        with diagnostic.ProjectorSwap(model, source, sham=True, seed=123) as swap:
            for name, parameter in model.named_parameters():
                if name in expected:
                    assert diagnostic._bitexact(parameter, expected[name])
            checksums.append(swap.summary["hashes"]["applied"])
            assert swap.summary["intended_delta_frobnorm"] == pytest.approx(swap.summary["source_delta_frobnorm"])
            assert swap.summary["applied_delta_norm_rounding_difference"] >= 0
            for report in swap.summary["tensors"]:
                assert report["intended_delta_frobnorm"] == pytest.approx(report["source_delta_frobnorm"])
                assert report["applied_delta_rounding_rms"] >= 0
        assert_snapshot(model, original)
    with diagnostic.ProjectorSwap(model, source, sham=True, seed=124) as other:
        assert other.summary["hashes"]["applied"] != checksums[0]
    assert checksums[0] == checksums[1]
    assert torch.equal(torch_rng, torch.random.get_rng_state()) and python_rng == random.getstate()


def test_signed_zero_is_hashed_copied_and_restored_by_bytes(model):
    with torch.no_grad():
        model.mlp1[0].bias.zero_()
    original, source = snapshot(model), mapping(model, changed=False)
    source["mlp1.0.bias"].fill_(-0.0)
    assert torch.equal(source["mlp1.0.bias"], original["mlp1.0.bias"])
    assert not diagnostic._bitexact(source["mlp1.0.bias"], original["mlp1.0.bias"])
    with diagnostic.ProjectorSwap(model, source) as swap:
        assert torch.signbit(model.mlp1[0].bias).all()
        assert "mlp1.0.bias" in swap.summary["changed_tensors"]
    assert_snapshot(model, original)


def test_signed_zero_self_sham_is_bitexact(model):
    with torch.no_grad():
        model.mlp1[0].bias.fill_(-0.0)
    original = snapshot(model)
    with diagnostic.ProjectorSwap(model, mapping(model, changed=False), sham=True) as swap:
        assert_snapshot(model, original)
        assert swap.summary["changedtensors"] == 0
    assert_snapshot(model, original)


def test_mapping_snapshot_is_taken_at_entry_and_norm_summary_is_correct(model):
    original, source = snapshot(model), mapping(model, changed=False)
    swap = diagnostic.ProjectorSwap(model, source)
    source[NAMES[-1]].add_(.125)  # Construction is lazy; entry snapshots current values.
    expected_delta = torch.linalg.vector_norm((source[NAMES[-1]] - original[NAMES[-1]]).double()).item()
    with swap:
        assert diagnostic._bitexact(model.mlp1[3].bias, source[NAMES[-1]])
        assert swap.summary["delta_frobnorm"] == pytest.approx(expected_delta)
        assert swap.summary["changed_parameter_count"] == 3
        assert swap.summary["changed_tensors"] == [NAMES[-1]]
        assert len(swap.summary["hashes"]["source"]) == 64
        for report in swap.summary["tensors"]:
            tensor = original[report["name"]].double()
            assert report["before_frobnorm"] == pytest.approx(torch.linalg.vector_norm(tensor).item())
    assert_snapshot(model, original)
    assert_released(swap)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_body_exception_is_not_suppressed_and_restores(model, failure):
    original = snapshot(model)
    swap = diagnostic.ProjectorSwap(model, mapping(model))
    with pytest.raises(failure, match="body failure"):
        with swap:
            raise failure("body failure")
    assert_snapshot(model, original)
    assert swap.summary["restored_bitexact"]
    assert_released(swap)


@pytest.mark.parametrize("fail_copy", [1, 3, 6])
def test_partial_entry_copy_failure_restores_and_releases(model, monkeypatch, fail_copy):
    original, source = snapshot(model), mapping(model)
    real_copy, calls = diagnostic._copy_parameter, []

    def broken(parameter, replacement):
        calls.append(id(parameter))
        assert not torch.is_grad_enabled()
        if len(calls) == fail_copy:
            parameter.reshape(-1)[0].copy_(replacement.reshape(-1)[0])
            raise RuntimeError("injected partially written copy")
        real_copy(parameter, replacement)

    monkeypatch.setattr(diagnostic, "_copy_parameter", broken)
    swap = diagnostic.ProjectorSwap(model, source)
    with pytest.raises(RuntimeError, match="partially written"):
        with swap:
            pytest.fail("body must not run after entry failure")
    assert_snapshot(model, original)
    assert swap.summary["restored_bitexact"] and calls
    assert_released(swap)
    with diagnostic.ProjectorSwap(model, source):
        pass  # The registry is not left occupied by a failed entry.
    assert_snapshot(model, original)


def test_silent_inexact_copy_is_detected_and_restored(model, monkeypatch):
    original, real_copy = snapshot(model), diagnostic._copy_parameter
    calls = 0

    def inexact(parameter, replacement):
        nonlocal calls
        calls += 1
        real_copy(parameter, replacement)
        if calls == 2:
            parameter.reshape(-1)[0].add_(1)

    monkeypatch.setattr(diagnostic, "_copy_parameter", inexact)
    swap = diagnostic.ProjectorSwap(model, mapping(model))
    with pytest.raises(RuntimeError, match="endpoint copy was not bit-exact"):
        swap.__enter__()
    assert_snapshot(model, original)
    assert_released(swap)


def test_transient_restore_failure_retries(model, monkeypatch):
    original, real_copy = snapshot(model), diagnostic._copy_parameter
    calls = 0

    def transient(parameter, replacement):
        nonlocal calls
        calls += 1
        if calls == 7:
            parameter.reshape(-1)[0].fill_(99)
            raise RuntimeError("transient restore failure")
        real_copy(parameter, replacement)

    monkeypatch.setattr(diagnostic, "_copy_parameter", transient)
    with diagnostic.ProjectorSwap(model, mapping(model)) as swap:
        pass
    assert calls == 13 and swap.summary["restored_bitexact"]
    assert_snapshot(model, original)
    assert_released(swap)


def test_permanent_restore_failure_attempts_peers_and_releases(model, monkeypatch):
    original, real_copy = snapshot(model), diagnostic._copy_parameter
    target = model.mlp1[3].bias
    restoring, attempted = False, []

    def broken(parameter, replacement):
        if restoring:
            attempted.append(id(parameter))
            if parameter is target:
                raise RuntimeError("permanent device failure")
        real_copy(parameter, replacement)

    monkeypatch.setattr(diagnostic, "_copy_parameter", broken)
    swap = diagnostic.ProjectorSwap(model, mapping(model))
    with pytest.raises(RuntimeError, match="do not reuse this model"):
        with swap:
            restoring = True
    assert set(attempted) == {id(p) for p in model.mlp1.parameters()}
    for name, parameter in model.named_parameters():
        if parameter is not target:
            assert diagnostic._bitexact(parameter, original[name])
    assert swap.summary["restored_bitexact"] is False
    assert swap.summary["restore_failed_names"] == ["mlp1.3.bias"]
    assert_released(swap)


def test_nested_alias_projector_rejected_without_releasing_outer(model):
    original, source = snapshot(model), mapping(model)
    alias = nn.Module()
    alias.mlp1 = model.mlp1
    alias.eval()
    with diagnostic.ProjectorSwap(model, source) as outer:
        active = snapshot(model)
        inner = diagnostic.ProjectorSwap(alias, source)
        with pytest.raises(RuntimeError, match="nested/active"):
            inner.__enter__()
        assert diagnostic._ACTIVE_PROJECTORS[model.mlp1] is outer
        assert_snapshot(model, active)
        assert_released(inner)
        with diagnostic.ProjectorSwap(tiny(), source):
            pass  # Independent native projectors may be active simultaneously.
    assert_snapshot(model, original)
    with pytest.raises(RuntimeError, match="single-use"):
        outer.__enter__()


@pytest.mark.parametrize("kind", ["missing", "extra", "short", "not_mapping"])
def test_exact_native_mapping_names(model, kind):
    source = mapping(model)
    if kind == "missing":
        del source[NAMES[-1]]
    elif kind == "extra":
        source["language_model.weight"] = torch.ones(1)
    elif kind == "short":
        source = {name.removeprefix("mlp1."): tensor for name, tensor in source.items()}
    else:
        source = list(source.items())
    with pytest.raises(ValueError, match="exactly the six native"):
        diagnostic.ProjectorSwap(model, source)


@pytest.mark.parametrize("sham,seed", [(1, 0), (False, True), (False, -1),
                                       (False, 2**63), (False, 1.5)])
def test_strict_sham_seed(model, sham, seed):
    with pytest.raises(ValueError, match="sham must be boolean"):
        diagnostic.ProjectorSwap(model, mapping(model), sham=sham, seed=seed)


@pytest.mark.parametrize("where", ["source", "native"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_preflight_is_atomic_and_releases(model, where, bad):
    source = mapping(model)
    with torch.no_grad():
        tensor = source[NAMES[-1]] if where == "source" else model.mlp1[3].bias
        tensor[0] = bad
    original = snapshot(model)
    swap = diagnostic.ProjectorSwap(model, source)
    with pytest.raises(ValueError, match="nonfinite"):
        swap.__enter__()
    assert_snapshot(model, original)  # Raw-byte equality also handles the preexisting NaN.
    assert_released(swap)


@pytest.mark.parametrize("kind", ["shape", "integer", "meta", "complex", "converted_inf"])
def test_source_validation_and_conversion_overflow_precede_mutation(model, kind):
    original, source = snapshot(model), mapping(model)
    if kind == "shape":
        source[NAMES[-1]] = torch.zeros(4)
    elif kind == "integer":
        source[NAMES[-1]] = source[NAMES[-1]].long()
    elif kind == "meta":
        source[NAMES[-1]] = source[NAMES[-1]].to("meta")
    elif kind == "complex":
        source[NAMES[-1]] = source[NAMES[-1]].to(torch.complex64)
    else:
        source[NAMES[-1]] = torch.full((3,), 1e300, dtype=torch.float64)
    swap = diagnostic.ProjectorSwap(model, source)
    with pytest.raises(ValueError):
        swap.__enter__()
    assert_snapshot(model, original)
    assert_released(swap)


@pytest.mark.parametrize("kind", ["training", "child_training", "relu", "extra", "no_bias", "buffer", "shared"])
def test_exact_eval_native_architecture(model, kind):
    if kind == "training":
        model.train()
    elif kind == "child_training":
        model.mlp1[2].train()
    elif kind == "relu":
        model.mlp1[2] = nn.ReLU().eval()
    elif kind == "extra":
        model.mlp1.append(nn.Identity().eval())
    elif kind == "no_bias":
        model.mlp1[3].register_parameter("bias", None)
    elif kind == "buffer":
        model.mlp1.register_buffer("extra", torch.zeros(1))
    else:
        model.mlp1[0].bias = model.mlp1[0].weight
    source = mapping(tiny())
    original = snapshot(model)
    swap = diagnostic.ProjectorSwap(model, source)
    with pytest.raises(ValueError):
        swap.__enter__()
    assert_snapshot(model, original)
    assert_released(swap)


@pytest.mark.parametrize("cap", ["MAX_SOURCE_BYTES", "MAX_NATIVE_BYTES"])
def test_aggregate_swap_bounds_before_backups_or_copies(model, monkeypatch, cap):
    original, source = snapshot(model), mapping(model)
    largest = max(t.numel() * t.element_size() for t in source.values())
    monkeypatch.setattr(diagnostic, cap, largest)  # Each tensor fits, their sum does not.

    def forbidden(*args, **kwargs):
        raise AssertionError("bounds must precede snapshots/copies")

    monkeypatch.setattr(diagnostic, "_owned_cpu", forbidden)
    monkeypatch.setattr(diagnostic, "_copy_parameter", forbidden)
    swap = diagnostic.ProjectorSwap(model, source)
    with pytest.raises(ValueError, match="size cap"):
        swap.__enter__()
    assert_snapshot(model, original)
    assert_released(swap)


def test_default_payload_bounds_and_boundary_acceptance(model, monkeypatch):
    assert diagnostic.MAX_SOURCE_BYTES == 256 * 1024**2
    assert diagnostic.MAX_NATIVE_BYTES == 512 * 1024**2
    source = mapping(model)
    size = sum(t.numel() * t.element_size() for t in source.values())
    monkeypatch.setattr(diagnostic, "MAX_SOURCE_BYTES", size)
    monkeypatch.setattr(diagnostic, "MAX_NATIVE_BYTES", size)
    with diagnostic.ProjectorSwap(model, source) as swap:
        assert swap.summary["source_bytes"] == swap.summary["native_bytes"] == size


def checkpoint_files(tmp_path, monkeypatch, *, sharded=False, tensors=None, allow_downloads=False):
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    from safetensors.torch import save_file

    tensors = mapping(tiny()) if tensors is None else tensors
    # Extra native tensors must never be read by get_tensor.
    saved = tensors | {"vision_model.weight": torch.ones(2), "language_model.weight": torch.ones(2)}
    groups = ([list(saved)[:3], list(saved)[3:]] if sharded else [list(saved)])
    files, weight_map = [], {}
    for number, names in enumerate(groups, 1):
        filename = f"model-{number:05d}-of-00002.safetensors" if sharded else "model.safetensors"
        path = tmp_path / filename
        save_file({name: saved[name].contiguous() for name in names}, str(path))
        files.append(path)
        weight_map.update(dict.fromkeys(names, filename))
    index = tmp_path / "model.safetensors.index.json"
    if sharded:
        index.write_text(json.dumps({"weight_map": weight_map}))
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["repo_id"] == "fixture/projector" and kwargs["revision"] == REVISION
        assert kwargs["local_files_only"] is not allow_downloads
        assert kwargs["repo_type"] == "model"
        path = tmp_path / kwargs["filename"]
        if not path.exists():
            error = EntryNotFoundError if allow_downloads else LocalEntryNotFoundError
            raise error("fixture-only asset absence")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    checkpoint = SimpleNamespace(model_id="fixture/projector", revision=REVISION, backend="internvl")
    return checkpoint, tensors, files, index, calls


def track_safe_open(monkeypatch, *, override_shape=None):
    import safetensors

    real_open, tensor_reads, shape_reads = safetensors.safe_open, [], []

    class Slice:
        def __init__(self, value, name):
            self.value, self.name = value, name

        def get_shape(self):
            shape_reads.append(self.name)
            return (override_shape(self.name, self.value.get_shape()) if override_shape
                    else self.value.get_shape())

    class Handle:
        def __init__(self, *args, **kwargs):
            self.handle = real_open(*args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_slice(self, name):
            return Slice(self.handle.get_slice(name), name)

        def get_tensor(self, name):
            assert set(shape_reads) == set(NAMES), "all six headers must precede ANY get_tensor"
            assert name in NAMES, "must not materialize vision/decoder tensors"
            tensor_reads.append(name)
            return self.handle.get_tensor(name)

    monkeypatch.setattr(safetensors, "safe_open", Handle)
    return tensor_reads, shape_reads


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("allow_downloads", [False, True])
def test_loader_selected_cpu_owned_full_pin_checksums_and_layout(tmp_path, monkeypatch, sharded, allow_downloads):
    from vlm_pilot.weights import SafetensorSource

    checkpoint, expected, files, index, calls = checkpoint_files(
        tmp_path, monkeypatch, sharded=sharded, allow_downloads=allow_downloads)
    reads, _ = track_safe_open(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("no model/pickle or native attention read_tensor shortcut")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(SafetensorSource, "read_tensor", forbidden)
    tensors, metadata = diagnostic.load_projector(checkpoint, allow_downloads=allow_downloads)
    assert set(reads) == set(NAMES) and len(reads) == 6
    assert tuple(tensors) == NAMES
    assert metadata["model_id"] == metadata["repo_id"] == "fixture/projector"
    assert metadata["revision"] == REVISION
    assert metadata["allow_downloads"] is allow_downloads
    assert metadata["layout"] == ("sharded" if sharded else "single_file")
    assert metadata["source_bytes"] == sum(t.numel() * t.element_size() for t in expected.values())
    assert metadata["header_bound_bytes"] == sum(t.numel() * 8 for t in expected.values())
    records = {record["name"]: record for record in metadata["files"]}
    for file in files:
        assert records[file.name]["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
        assert records[file.name]["size_bytes"] == file.stat().st_size
    if sharded:
        assert metadata["index"]["sha256"] == hashlib.sha256(index.read_bytes()).hexdigest()
    else:
        assert metadata["index"] is None
    for name, tensor in tensors.items():
        assert tensor.device.type == "cpu" and not tensor.requires_grad
        assert diagnostic._bitexact(tensor, expected[name])
        tensor.zero_()
    loaded, _ = diagnostic.load_projector(vars(checkpoint), allow_downloads=allow_downloads)
    for name in NAMES:
        assert diagnostic._bitexact(loaded[name], expected[name])
    assert calls and all(call["revision"] == REVISION for call in calls)
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("kind", ["missing", "short", "extra"])
def test_loader_rejects_wrong_projector_names_before_tensor_reads(tmp_path, monkeypatch, kind):
    tensors = mapping(tiny())
    if kind == "missing":
        del tensors[NAMES[-1]]
    elif kind == "short":
        tensors = {name.removeprefix("mlp1."): t for name, t in tensors.items()}
    else:
        tensors["mlp1.2.weight"] = torch.ones(1)
    checkpoint_files_result = checkpoint_files(tmp_path, monkeypatch, tensors=tensors, sharded=True)
    reads, _ = track_safe_open(monkeypatch)
    with pytest.raises(ValueError, match="exactly the six native"):
        diagnostic.load_projector(checkpoint_files_result[0])
    assert not reads


@pytest.mark.parametrize("kind", ["aggregate", "huge", "rank", "bool", "zero", "disconnected"])
def test_loader_header_bounds_and_shapes_before_any_get_tensor(tmp_path, monkeypatch, kind):
    checkpoint, tensors, _, _, _ = checkpoint_files(tmp_path, monkeypatch, sharded=True)
    if kind == "aggregate":
        monkeypatch.setattr(diagnostic, "MAX_HEADER_BYTES", max(t.numel() * 8 for t in tensors.values()))

    def override(name, shape):
        if kind == "huge":
            return {"mlp1.0.weight": [100000], "mlp1.0.bias": [100000],
                    "mlp1.1.weight": [100000, 100000], "mlp1.1.bias": [100000],
                    "mlp1.3.weight": [100000, 100000], "mlp1.3.bias": [100000]}[name]
        if name == NAMES[-1]:
            return {"rank": [1, 3], "bool": [True], "zero": [0],
                    "disconnected": [4]}.get(kind, shape)
        return shape

    reads, _ = track_safe_open(monkeypatch, override_shape=override)
    with pytest.raises(ValueError, match="shape|size cap"):
        diagnostic.load_projector(checkpoint)
    assert not reads


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_loader_rejects_nonfinite_payload(tmp_path, monkeypatch, bad):
    tensors = mapping(tiny())
    tensors[NAMES[-1]][0] = bad
    checkpoint, _, _, _, _ = checkpoint_files(tmp_path, monkeypatch, tensors=tensors)
    with pytest.raises(ValueError, match="nonfinite"):
        diagnostic.load_projector(checkpoint)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_loader_preserves_source_dtype_and_owns_storage(tmp_path, monkeypatch, dtype):
    source = {name: tensor.to(dtype) for name, tensor in mapping(tiny()).items()}
    checkpoint, _, _, _, _ = checkpoint_files(tmp_path, monkeypatch, tensors=source)
    loaded, metadata = diagnostic.load_projector(checkpoint)
    for name, tensor in loaded.items():
        assert tensor.dtype == dtype and diagnostic._bitexact(tensor, source[name])
        assert tensor.data_ptr() != source[name].data_ptr()
    assert all(row["dtype"] == str(dtype) for row in metadata["tensors"])


def test_loader_rejects_integer_payload(tmp_path, monkeypatch):
    source = mapping(tiny())
    source[NAMES[-1]] = source[NAMES[-1]].long()
    checkpoint, _, _, _, _ = checkpoint_files(tmp_path, monkeypatch, tensors=source)
    with pytest.raises(ValueError, match="dense real floating CPU"):
        diagnostic.load_projector(checkpoint)


def test_loader_file_size_cap_precedes_tensor_read(tmp_path, monkeypatch):
    from vlm_pilot.weights import SafetensorSource

    checkpoint, _, _, _, _ = checkpoint_files(tmp_path, monkeypatch, sharded=True)
    monkeypatch.setattr(SafetensorSource, "MAX_SHARD_BYTES", 1)
    reads, _ = track_safe_open(monkeypatch)
    with pytest.raises(ValueError, match="size cap"):
        diagnostic.load_projector(checkpoint)
    assert not reads


def test_loader_rejects_index_mutation_after_layout_resolution(tmp_path, monkeypatch):
    from vlm_pilot.weights import SafetensorSource

    checkpoint, _, _, index, _ = checkpoint_files(tmp_path, monkeypatch, sharded=True)
    real_load = SafetensorSource._load_index

    def changed(source):
        weights = real_load(source)
        document = json.loads(index.read_text())
        document["weight_map"][NAMES[0]] = "different.safetensors"
        index.write_text(json.dumps(document))
        return weights

    monkeypatch.setattr(SafetensorSource, "_load_index", changed)
    reads, _ = track_safe_open(monkeypatch)
    with pytest.raises(ValueError, match="index changed after layout"):
        diagnostic.load_projector(checkpoint)
    assert not reads


def test_loader_does_not_fallback_on_corrupt_index(tmp_path, monkeypatch):
    checkpoint, _, _, index, calls = checkpoint_files(tmp_path, monkeypatch, sharded=True)
    index.write_text('{"weight_map":{"mlp1.0.weight":"../escape.safetensors"}}')
    with pytest.raises(ValueError, match="unsafe"):
        diagnostic.load_projector(checkpoint)
    assert [call["filename"] for call in calls] == ["model.safetensors.index.json"]


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40, None])
def test_loader_requires_full_pin_without_asset_lookup(monkeypatch, revision):
    def forbidden(**kwargs):
        raise AssertionError("invalid pin must precede asset lookup")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)
    with pytest.raises(ValueError, match="revision"):
        diagnostic.load_projector({"model_id": "fixture/projector", "revision": revision})


@pytest.mark.parametrize("allow_downloads", [None, 0, 1, "yes"])
def test_loader_download_consent_must_be_boolean(allow_downloads):
    with pytest.raises(ValueError, match="explicit boolean"):
        diagnostic.load_projector({}, allow_downloads=allow_downloads)