"""Explicit, bounded InternVL ``mlp1`` endpoint diagnostics, not a study runner.

Importing (or executing this file) starts nothing and imports only the stdlib.
Call ``load_projector(checkpoint)`` for an owned CPU mapping and provenance,
then use ``with ProjectorSwap(eval_model, mapping) as swap`` and inspect
``swap.summary``. No model construction, training, CLI, saving, or automatic
evaluation is provided. The caller supplies inference inputs and no-grad scope.

Only the native six mlp1.{0,1,3}.{weight,bias} names are accepted, with an exact
Sequential(LayerNorm, Linear, GELU, Linear) projector. Source mappings must be
CPU tensors; their snapshot is taken on entry, not at construction. Ordinary
swaps copy the source converted to each destination dtype, without delta
subtract/add roundoff. Shams permute that endpoint-minus-original delta within
each tensor using a local CPU generator; destination rounding is reported.

Aggregate source storage is capped at 256 MiB and native storage at 512 MiB.
Loader headers additionally use a conservative eight bytes per element BEFORE
any get_tensor. These are tensor-payload caps, not process-RSS guarantees: CPU
backups, endpoints, byte comparisons, and a sham's int64 permutation add memory.
Numerical diagnostics use bounded chunks. Referenced checkpoint files can be
much larger; their FULL checksums require explicit streaming disk reads.

This depends on private SafetensorSource._load_index/_download interfaces and
preserves their same-revision, cached-only default and fail-closed layout rules.
No decoder/vision weights are materialized. A caller must not concurrently
mutate the model/mapping/cache or change parameter devices/dtypes/identities.
The active-projector guard rejects overlapping contexts, not concurrent forward
calls. A permanently broken device/copy cannot be repaired in software: every
restore is attempted, resources are released, and failure is raised explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

__all__ = ["ProjectorSwap", "load_projector"]

PROJECTOR_NAMES = tuple(f"mlp1.{layer}.{field}" for layer in (0, 1, 3)
                        for field in ("weight", "bias"))
MAX_SOURCE_BYTES = 256 * 1024**2
MAX_NATIVE_BYTES = 512 * 1024**2
MAX_HEADER_BYTES = 512 * 1024**2
_CHUNK_ELEMENTS = 65536
_LOCK = threading.RLock()
_ACTIVE_PROJECTORS: WeakKeyDictionary = WeakKeyDictionary()


def _exact_names(mapping: Mapping) -> None:
    if not isinstance(mapping, Mapping) or set(mapping) != set(PROJECTOR_NAMES):
        raise ValueError("projector mapping must have exactly the six native mlp1 names")


def _shape(shape: Any, name: str) -> tuple[int, ...]:
    rank = 2 if name in ("mlp1.1.weight", "mlp1.3.weight") else 1
    if (not isinstance(shape, (list, tuple)) or len(shape) != rank
            or any(type(size) is not int or size < 1 for size in shape)):
        raise ValueError(f"invalid projector shape: {name}")
    return tuple(shape)


def _connected_shapes(shapes: Mapping) -> None:
    for name in PROJECTOR_NAMES:
        _shape(shapes[name], name)
    a, b = shapes["mlp1.1.weight"], shapes["mlp1.3.weight"]
    if (shapes["mlp1.0.weight"] != shapes["mlp1.0.bias"]
            or shapes["mlp1.0.weight"] != (a[1],)
            or shapes["mlp1.1.bias"] != (a[0],)
            or b[1] != a[0] or shapes["mlp1.3.bias"] != (b[0],)):
        raise ValueError("projector LayerNorm/Linear shapes are not connected")


def _tensor(tensor: Any, name: str, *, cpu: bool = False) -> None:
    import torch

    if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
            or tensor.device.type == "meta" or tensor.is_quantized
            or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or (cpu and tensor.device.type != "cpu")):
        raise ValueError(f"{name} must be a dense real floating {'CPU ' if cpu else ''}tensor")
    _shape(tuple(tensor.shape), name)


def _finite(tensor: Any, description: str) -> None:
    import torch

    # Flattening is a view for owned snapshots; noncontiguous caller inputs may copy.
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), _CHUNK_ELEMENTS):
        if not bool(torch.isfinite(flat[start:start + _CHUNK_ELEMENTS]).all().item()):
            raise ValueError(f"nonfinite {description}")


def _owned_cpu(tensor: Any, dtype: Any = None) -> Any:
    # copy=True prevents aliasing even for a contiguous CPU source or native model.
    return tensor.detach().to(device="cpu", dtype=dtype or tensor.dtype,
                              copy=True).contiguous()


def _checksum(tensor: Any) -> str:
    import torch

    cpu = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256(f"{cpu.dtype}:{tuple(cpu.shape)}:".encode("ascii"))
    # Byte views support BF16 without converting the floating dtype or allocating
    # a Python integer per checkpoint byte. NumPy is a dependency of the pilot.
    digest.update(memoryview(cpu.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def _bitexact(parameter: Any, expected: Any) -> bool:
    import torch

    actual = parameter.detach().to("cpu").contiguous()
    return (actual.dtype == expected.dtype and actual.shape == expected.shape
            and torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8)))


def _norm(chunk: Any) -> float:
    scale = float(chunk.abs().max().item())
    result = 0.0 if scale == 0 else scale * float((chunk / scale).square().sum().sqrt().item())
    if not math.isfinite(result):
        raise ValueError("nonfinite projector norm diagnostic")
    return result


def _relative(delta: float, native: float) -> float | None:
    result = (0.0 if delta == 0 else None) if native == 0 else delta / native
    if result is not None and not math.isfinite(result):
        raise ValueError("nonfinite relative projector norm")
    return result


def _copy_parameter(parameter: Any, tensor: Any) -> None:
    """One in-place copy, kept separate to exercise partial device-copy failures."""
    parameter.copy_(tensor)


def _replacement(original: Any, endpoint: Any, sham: bool, generator: Any) -> tuple[Any, Any]:
    import torch

    if not sham:
        return endpoint, None  # Exact converted endpoint, never subtract/add.
    permutation = torch.randperm(original.numel(), generator=generator, device="cpu")
    if _bitexact(endpoint, original):
        # A self sham must also preserve negative zero, not add a positive-zero delta.
        return endpoint, permutation
    result = torch.empty_like(original)
    before, source, after = original.reshape(-1), endpoint.reshape(-1), result.reshape(-1)
    for start in range(0, before.numel(), _CHUNK_ELEMENTS):
        stop = min(start + _CHUNK_ELEMENTS, before.numel())
        indices = permutation[start:stop]
        delta = source.index_select(0, indices).double() - before.index_select(0, indices).double()
        _finite(delta, "sham delta")
        after[start:stop].copy_((before[start:stop].double() + delta).to(original.dtype))
    _finite(result, "rounded sham endpoint")
    return result, permutation


def _report(name: str, original: Any, source: Any, endpoint: Any,
            replacement: Any, permutation: Any) -> dict[str, Any]:
    import torch

    before, target, after = (t.reshape(-1) for t in (original, endpoint, replacement))
    norms = dict.fromkeys(("before", "after", "source_delta", "intended_delta", "delta", "rounding"), 0.0)
    changed = 0
    for start in range(0, before.numel(), _CHUNK_ELEMENTS):
        stop = min(start + _CHUNK_ELEMENTS, before.numel())
        a, b, c = (t[start:stop].double() for t in (before, target, after))
        source_delta, applied = b - a, c - a
        if permutation is None:
            intended = source_delta
        else:
            indices = permutation[start:stop]
            intended = target.index_select(0, indices).double() - before.index_select(0, indices).double()
        chunks = (a, c, source_delta, intended, applied, applied - intended)
        for key, chunk in zip(norms, chunks, strict=True):
            _finite(chunk, "projector diagnostic delta")
            norms[key] = math.hypot(norms[key], _norm(chunk))
        changed += int(torch.count_nonzero(applied).item())
    if not all(math.isfinite(value) for value in norms.values()):
        raise ValueError("nonfinite aggregate projector tensor norm")
    return {
        "name": name, "shape": list(original.shape), "parameter_count": original.numel(),
        "source_dtype": str(source.dtype), "destination_dtype": str(original.dtype),
        "native_checksum": _checksum(original), "source_checksum": _checksum(source),
        "endpoint_checksum": _checksum(endpoint), "applied_checksum": _checksum(replacement),
        "changed_parameter_count": changed,
        **{f"{key}_frobnorm": value for key, value in norms.items() if key != "rounding"},
        "normalized_delta_norm": _relative(norms["delta"], norms["before"]),
        "applied_delta_rounding_rms": norms["rounding"] / math.sqrt(original.numel()),
    }


class _Part:
    def __init__(self, name: str, parameter: Any, original: Any) -> None:
        self.name, self.parameter, self.original = name, parameter, original
        self.replacement: Any = None
        self.applied = False


class ProjectorSwap:
    """Single-use eval-only context; all validation/staging precedes any mutation.

    Returns itself with a JSON-compatible ``summary`` (no retained tensors).
    Exact byte restoration includes signed zero and body/partial-entry exceptions.
    A failed restoration retries once, attempts every other touched parameter,
    releases the registry/backups, and raises rather than reporting success.
    Model parameters, requires_grad, hooks, forward methods and eval flags are
    never replaced or set. This is not an autograd/training transaction.
    """

    def __init__(self, model: Any, mapping: Mapping, sham: bool = False, seed: int = 0) -> None:
        _exact_names(mapping)
        if type(sham) is not bool or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("sham must be boolean and seed a nonnegative signed-64-bit integer")
        self.model, self.mapping, self.sham, self.seed = model, mapping, sham, seed
        self.summary: dict[str, Any] = {
            "sham": sham, "seed": seed, "active": False, "restored_bitexact": None,
            "source_limit_bytes": MAX_SOURCE_BYTES, "native_limit_bytes": MAX_NATIVE_BYTES,
            "endpoint_definition": "source converted to each destination dtype",
        }
        self._module: Any = None
        self._parts: list[_Part] = []
        self._entered = False

    def _resolve(self) -> dict[str, Any]:
        from torch import nn

        module = getattr(self.model, "mlp1", None)
        if not isinstance(self.model, nn.Module) or type(module) is not nn.Sequential:
            raise ValueError("model.mlp1 must be the native Sequential projector")
        if (self.model.training or any(child.training for child in module.modules())):
            raise ValueError("ProjectorSwap requires an eval model and eval projector")
        if (tuple(module._modules) != ("0", "1", "2", "3")
                or tuple(type(child) for child in module.children()) != (nn.LayerNorm, nn.Linear, nn.GELU, nn.Linear)
                or any(list(child.children()) for child in module.children())
                or list(module.named_buffers())):
            raise ValueError("projector must be exactly LayerNorm/Linear/GELU/Linear without buffers")
        parameters = {f"mlp1.{name}": value for name, value in
                      module.named_parameters(remove_duplicate=False)}
        _exact_names(parameters)
        storages = set()
        for name, parameter in parameters.items():
            if not isinstance(parameter, nn.Parameter):
                raise ValueError("projector weights must be native Parameters")
            _tensor(parameter, name)
            storage = (str(parameter.device), parameter.untyped_storage().data_ptr())
            if storage in storages:
                raise ValueError("projector parameters must not share storage")
            storages.add(storage)
        shapes = {name: tuple(parameter.shape) for name, parameter in parameters.items()}
        _connected_shapes(shapes)
        first, last = module[1], module[3]
        if (tuple(module[0].normalized_shape) != shapes["mlp1.0.weight"]
                or (first.out_features, first.in_features) != shapes["mlp1.1.weight"]
                or (last.out_features, last.in_features) != shapes["mlp1.3.weight"]):
            raise ValueError("projector module dimensions disagree with native parameters")
        if any(parameter.device.type not in ("cpu", "cuda") for parameter in parameters.values()):
            raise ValueError("only native CPU/CUDA projectors are supported")
        self._module = module
        return parameters

    def __enter__(self) -> ProjectorSwap:
        if self._entered:
            raise RuntimeError("ProjectorSwap contexts are single-use")
        self._entered = True
        try:
            import torch

            parameters = self._resolve()
            with _LOCK:
                if self._module in _ACTIVE_PROJECTORS:
                    raise RuntimeError("nested/active projector swap is forbidden")
                _ACTIVE_PROJECTORS[self._module] = self
            _exact_names(self.mapping)
            source_bytes = native_bytes = 0
            for name in PROJECTOR_NAMES:
                source, parameter = self.mapping[name], parameters[name]
                _tensor(source, name, cpu=True)
                if source.shape != parameter.shape:
                    raise ValueError(f"source/native projector shape mismatch: {name}")
                source_bytes += source.numel() * source.element_size()
                native_bytes += parameter.numel() * parameter.element_size()
            if source_bytes > MAX_SOURCE_BYTES or native_bytes > MAX_NATIVE_BYTES:
                raise ValueError("aggregate source/native projector size cap exceeded")
            # Bounds and ALL native/source finite checks precede owned backups or copies.
            for name in PROJECTOR_NAMES:
                _finite(parameters[name], f"native {name}")
                _finite(self.mapping[name], f"source {name}")
            self._parts = [_Part(name, parameters[name], _owned_cpu(parameters[name]))
                           for name in PROJECTOR_NAMES]
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            reports = []
            for part in self._parts:
                source = _owned_cpu(self.mapping[part.name])
                endpoint = _owned_cpu(source, part.original.dtype)
                _finite(endpoint, f"converted source {part.name}")
                part.replacement, permutation = _replacement(part.original, endpoint, self.sham, generator)
                reports.append(_report(part.name, part.original, source, endpoint, part.replacement, permutation))
                del source, endpoint, permutation
            aggregate = {f"{key}_frobnorm": math.hypot(*(r[f"{key}_frobnorm"] for r in reports))
                         for key in ("before", "after", "source_delta", "intended_delta", "delta")}
            if not all(math.isfinite(value) for value in aggregate.values()):
                raise ValueError("nonfinite aggregate projector norm")
            changed = [r["name"] for r in reports if r["native_checksum"] != r["applied_checksum"]]
            hashes = {key: hashlib.sha256(json.dumps(
                [(r["name"], r[f"{key}_checksum"]) for r in reports],
                separators=(",", ":")).encode("ascii")).hexdigest()
                for key in ("native", "source", "endpoint", "applied")}
            self.summary.update({
                "tensors": reports, "selected_names": list(PROJECTOR_NAMES),
                "source_bytes": source_bytes, "native_bytes": native_bytes,
                "paramcount": sum(part.original.numel() for part in self._parts),
                "changedtensors": len(changed), "changed_tensors": changed,
                "changed_parameter_count": sum(r["changed_parameter_count"] for r in reports),
                "hashes": hashes, **aggregate,
                "normalized_delta_norm": _relative(aggregate["delta_frobnorm"], aggregate["before_frobnorm"]),
                "intended_delta_norm_difference": abs(aggregate["intended_delta_frobnorm"] - aggregate["source_delta_frobnorm"]),
                "applied_delta_norm_rounding_difference": abs(aggregate["delta_frobnorm"] - aggregate["intended_delta_frobnorm"]),
            })
            with torch.no_grad():
                for part in self._parts:
                    part.applied = True  # Include a destination modified by a failing copy.
                    _copy_parameter(part.parameter, part.replacement)
                    if not _bitexact(part.parameter, part.replacement):
                        raise RuntimeError(f"projector endpoint copy was not bit-exact: {part.name}")
            self.summary["active"] = True
            return self
        except BaseException:
            self._restore()
            raise

    def _restore(self) -> None:
        errors = []
        try:
            import torch

            with torch.no_grad():
                for part in reversed(self._parts):
                    if not part.applied:
                        continue
                    # A transient restore copy gets one retry; never stop restoring peers.
                    for attempt in range(2):
                        try:
                            _copy_parameter(part.parameter, part.original)
                            if not _bitexact(part.parameter, part.original):
                                raise RuntimeError("projector restoration was not bit-exact")
                            part.applied = False
                            break
                        except BaseException as error:
                            if attempt == 1:
                                errors.append((part.name, error))
            self.summary["restored_bitexact"] = not errors
            self.summary["restore_failed_names"] = [name for name, _ in errors]
        finally:
            self.summary["active"] = False
            with _LOCK:
                if self._module is not None and _ACTIVE_PROJECTORS.get(self._module) is self:
                    del _ACTIVE_PROJECTORS[self._module]
            self._parts.clear()
            self._module = self.model = self.mapping = None
        if errors:
            raise RuntimeError("failed to restore projector parameters; do not reuse this model") from errors[0][1]

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self._restore()
        return False


def _stamp(path: Path, cap: int) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    if not path.is_file() or not 0 < stat.st_size <= cap:
        raise ValueError("checkpoint file is empty, missing, or exceeds its size cap")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _file_record(path: Path, name: str, cap: int) -> tuple[dict, tuple]:
    before = _stamp(path, cap)
    digest, total = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024**2):
            total += len(chunk)
            if total > before[2]:
                raise ValueError("checkpoint file grew during checksum")
            digest.update(chunk)
    if total != before[2] or _stamp(path, cap) != before:
        raise ValueError("checkpoint file changed during checksum")
    return {"name": name, "size_bytes": total, "sha256": digest.hexdigest()}, before


def load_projector(checkpoint: Any, allow_downloads: bool = False) -> tuple[dict, dict]:
    """Load only six finite, CPU-owned projector tensors at a FULL HF commit.

    ``checkpoint`` is a Checkpoint-like object or mapping with ``model_id`` and
    ``revision``; an optional backend must be ``internvl``. Returns ``(mapping,
    metadata)`` with full repository/revision, actual layout, full referenced
    weight-file SHA256s, an index checksum when sharded, and tensor hashes/dtypes.
    No model/tokenizer/remote Python code is constructed or executed. Shapes of
    ALL six tensors and aggregate conservative bytes are checked before ANY
    get_tensor; no mlp1 read is delegated to read_tensor (it rejects that prefix).
    """
    if type(allow_downloads) is not bool:
        raise ValueError("allow_downloads must be an explicit boolean")

    def field(name: str, default: Any = None) -> Any:
        return checkpoint.get(name, default) if isinstance(checkpoint, Mapping) else getattr(checkpoint, name, default)

    if field("backend", "internvl") != "internvl":
        raise ValueError("load_projector requires a pinned native InternVL checkpoint")
    from safetensors import safe_open

    from vlm_pilot.weights import SafetensorSource

    source = SafetensorSource(field("model_id"), field("revision"), allow_downloads)
    weights = source._load_index()
    selected = {name: filename for name, filename in weights.items() if name.startswith("mlp1.")}
    _exact_names(selected)
    groups = {filename: [name for name in PROJECTOR_NAMES if selected[name] == filename]
              for filename in sorted(set(selected.values()))}
    paths, stamps, shapes, files = {}, {}, {}, []
    for filename, names in groups.items():
        path = source._download(filename)
        source._file_size(path, source.MAX_SHARD_BYTES, "projector checkpoint file")
        paths[filename] = path
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            actual = {name for name in handle.keys() if name.startswith("mlp1.")}
            if actual != set(names):
                raise ValueError("projector names disagree with their declared checkpoint file")
            for name in names:
                shapes[name] = _shape(handle.get_slice(name).get_shape(), name)
    _connected_shapes(shapes)
    header_bound = sum(math.prod(shape) * 8 for shape in shapes.values())
    if header_bound > MAX_HEADER_BYTES:
        raise ValueError("aggregate projector header exceeds source size cap before get_tensor")
    # Full file hashes are provenance, not merely hashes of selected tensor payloads.
    for filename, path in paths.items():
        record, stamps[filename] = _file_record(path, filename, source.MAX_SHARD_BYTES)
        files.append(record | {"layout": source.layout})
    index = None
    if source.layout == "sharded":
        index_name = "model.safetensors.index.json"
        index_path = source._download(index_name)
        index, index_stamp = _file_record(index_path, index_name, source.MAX_INDEX_BYTES)
        with index_path.open("rb") as handle:
            raw = handle.read(source.MAX_INDEX_BYTES + 1)
        if len(raw) > source.MAX_INDEX_BYTES or _stamp(index_path, source.MAX_INDEX_BYTES) != index_stamp:
            raise ValueError("checkpoint index changed before projector read")
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=source._unique_object,
                              parse_constant=source._reject_constant)
        if (not isinstance(document, dict) or document.get("weight_map") != weights
                or hashlib.sha256(raw).hexdigest() != index["sha256"]):
            raise ValueError("checkpoint index changed after layout resolution")
    tensors, reports, actual_bytes = {}, [], 0
    for filename, names in groups.items():
        path = paths[filename]
        if _stamp(path, source.MAX_SHARD_BYTES) != stamps[filename]:
            raise ValueError("checkpoint file changed before projector read")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in names:
                if _shape(handle.get_slice(name).get_shape(), name) != shapes[name]:
                    raise ValueError("projector header changed before get_tensor")
                tensor = handle.get_tensor(name)
                _tensor(tensor, name, cpu=True)
                if tuple(tensor.shape) != shapes[name]:
                    raise ValueError("projector payload/header shape mismatch")
                _finite(tensor, f"loaded source {name}")
                actual_bytes += tensor.numel() * tensor.element_size()
                if actual_bytes > MAX_SOURCE_BYTES:
                    raise ValueError("aggregate loaded projector exceeds source size cap")
                tensors[name] = _owned_cpu(tensor)
                reports.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                "size_bytes": tensor.numel() * tensor.element_size(),
                                "sha256": _checksum(tensors[name])})
        if _stamp(path, source.MAX_SHARD_BYTES) != stamps[filename]:
            raise ValueError("checkpoint file changed during projector read")
    if index is not None and _stamp(index_path, source.MAX_INDEX_BYTES) != index_stamp:
        raise ValueError("checkpoint index changed during projector read")
    metadata = {"model_id": source.repo_id, "repo_id": source.repo_id, "revision": source.revision,
                "allow_downloads": source.allow_downloads, "layout": source.layout,
                "files": files, "index": index, "tensors": sorted(reports, key=lambda r: r["name"]),
                "source_bytes": actual_bytes, "header_bound_bytes": header_bound}
    return {name: tensors[name] for name in PROJECTOR_NAMES}, metadata