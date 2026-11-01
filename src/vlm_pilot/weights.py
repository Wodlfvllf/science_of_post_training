"""Bounded, reversible Qwen3 KV-group weight rollback/replay.

Sources expose CPU tensors under native InternVL checkpoint names, always
language_model.model.layers.{layer}.self_attn.{projection_or_norm}.weight.
No checkpoint/model is instantiated by a source. HF/safetensors imports and
index reads are lazy, commit-pinned, and cached-only unless explicitly enabled.

A swap touches all Q rows sharing one KV head, that group's K/V rows, or its
o_proj columns. q_norm/k_norm weights are shared across the ENTIRE layer;
include_norms=True explicitly broadens the intervention beyond one KV group.
Only selected original slices are backed up; Parameter identities, biases,
other groups, and forward methods are unchanged. Nothing saves a hybrid model.

Shams permute the *source-minus-native delta*, separately inside each selected
slice, with a local CPU generator. The intended delta's Frobenius norm is
preserved; destination-dtype addition can round, and that error is reported.
Full swaps copy the source endpoint directly, avoiding subtract/add roundoff.
All runtime use is eval-only and needs independent real-device preflight.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from weakref import WeakKeyDictionary

import torch
from torch import Tensor, nn

from .attention import Capture, Site, _finite, _index, _replay_qk, _resolve_attention, _rms
from .schemas import PilotError

__all__ = [
    "WeightSource", "MappingTensorSource", "SafetensorSource", "WeightSwap",
    "identical_input_replay",
]

_NATIVE_NAME = re.compile(
    r"language_model\.model\.layers\.[0-9]+\.self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\.weight\Z"
)
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _native_name(name: str) -> None:
    if not isinstance(name, str) or _NATIVE_NAME.fullmatch(name) is None:
        raise ValueError("weight names must use the full native language_model.model.layers prefix")


def _checksum(tensor: Tensor) -> str:
    """Hash dtype, shape, and raw CPU bytes, including BF16 without NumPy casting."""
    cpu = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256(f"{cpu.dtype}:{tuple(cpu.shape)}:".encode("ascii"))
    # A byte view avoids a second full bytes allocation and supports BF16 tensors.
    digest.update(memoryview(cpu.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def _norm(tensor: Tensor) -> float:
    cpu = tensor.detach().to("cpu", torch.float64)
    scale = float(cpu.abs().max().item())
    if scale == 0:
        return 0.0
    result = scale * float((cpu / scale).square().sum().sqrt().item())
    if not math.isfinite(result):
        raise ValueError("nonfinite Frobenius norm diagnostic")
    return result


def _relative_norm(delta: float, native: float) -> float | None:
    # A nonzero delta over a zero native norm has no defined relative magnitude.
    if native == 0:
        return 0.0 if delta == 0 else None
    result = delta / native
    if not math.isfinite(result):
        raise ValueError("nonfinite relative weight norm diagnostic")
    return result


class WeightSource(Protocol):
    """Read one full, finite CPU weight tensor; no model construction or name fallback."""

    def read_tensor(self, name: str) -> Tensor: ...


class MappingTensorSource:
    """Immutable snapshots of a caller's mapping, suitable for offline counterparts.

    Supply full native-prefixed keys, e.g. prepend 'language_model.' to a
    Qwen3ForCausalLM's named_parameters keys. Each read returns a fresh clone.
    Extra mapping keys are allowed, but requested keys never fall back to a
    shorter name. GPU input tensors are copied to CPU at construction.
    """

    def __init__(self, tensors: Mapping[str, Tensor]) -> None:
        if not isinstance(tensors, Mapping):
            raise ValueError("MappingTensorSource requires a tensor mapping")
        self._tensors: dict[str, Tensor] = {}
        for name, tensor in tensors.items():
            if not isinstance(name, str) or not name:
                raise ValueError("mapping tensor names must be nonempty strings")
            _finite(tensor, f"source {name}")
            self._tensors[name] = tensor.detach().to("cpu").contiguous().clone()

    def read_tensor(self, name: str) -> Tensor:
        _native_name(name)
        # Indexing raises KeyError on a missing native name; there is no prefix fallback.
        result = self._tensors[name].clone()
        _finite(result, f"source {name}")
        return result


class SafetensorSource:
    """Lazy single-file or indexed HF reads at a full 40-hex commit revision.

    Only the requested full tensor is materialized with safe_open. Limits are
    enforced on index bytes, shard bytes, index entries, total declared bytes,
    and header shape BEFORE get_tensor. HF cache symlinks are accepted, but
    index-provided shard names must be safe basenames, never paths/URLs.
    Single-file maps come from the actual safetensors header, not a guessed list.
    Remote absence and local-only cache misses are distinct; transport/auth errors
    or corrupt indices never trigger a different checkpoint-layout fallback.
    """

    MAX_INDEX_BYTES = 16 * 1024**2
    MAX_SHARD_BYTES = 16 * 1024**3
    MAX_TENSOR_BYTES = 1024**3
    MAX_MODEL_BYTES = 256 * 1024**3
    MAX_INDEX_ENTRIES = 100_000
    _SHARD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors\Z")

    def __init__(self, repo_id: str, revision: str, allow_downloads: bool = False) -> None:
        repo_pattern = r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
        if not isinstance(repo_id, str) or re.fullmatch(repo_pattern, repo_id) is None:
            raise ValueError("repo_id must be an owner/model HF repository identifier")
        if ".." in repo_id or not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("revision must be a pinned lowercase 40-hex HF commit")
        if type(allow_downloads) is not bool:
            raise ValueError("allow_downloads must be an explicit boolean")
        self._repo_id, self._revision, self._allow_downloads = repo_id, revision, allow_downloads
        self._weight_map: dict[str, str] | None = None
        self._layout: str | None = None

    @property
    def repo_id(self) -> str:
        return self._repo_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def allow_downloads(self) -> bool:
        return self._allow_downloads

    @property
    def layout(self) -> str | None:
        return self._layout

    def _download(self, filename: str, *, probe: bool = False) -> Path:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            return Path(hf_hub_download(
                repo_id=self.repo_id, filename=filename, revision=self.revision,
                repo_type="model", local_files_only=not self.allow_downloads,
            ))
        except EntryNotFoundError as error:
            if probe:
                raise
            raise PilotError(f"Pinned checkpoint asset unavailable: {filename} "
                             f"({type(error).__name__}). Check the exact revision/cache; "
                             "downloads are opt-in. No moving-revision fallback.") from None
        except Exception as error:
            raise PilotError(f"Pinned checkpoint asset lookup failed: {filename} "
                             f"({type(error).__name__}). No checkpoint-layout fallback "
                             "on a transport or authentication failure.") from None

    @staticmethod
    def _file_size(path: Path, cap: int, description: str) -> int:
        if not path.is_file():
            raise ValueError(f"{description} is not a regular cached file")
        size = path.stat().st_size
        if size < 1 or size > cap:
            raise ValueError(f"{description} exceeds its size cap or is empty")
        return size

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate checkpoint index key: {key}")
            result[key] = value
        return result

    @staticmethod
    def _reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite checkpoint index number: {value}")

    def _single_file_map(self) -> dict[str, str]:
        """Read only a bounded header; never load all tensors or a pickle model."""
        from safetensors import safe_open

        name = "model.safetensors"
        path = self._download(name)
        size = self._file_size(path, min(self.MAX_SHARD_BYTES, self.MAX_MODEL_BYTES), name)
        with path.open("rb") as handle:
            prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("single safetensors file has an incomplete header prefix")
        header_bytes = int.from_bytes(prefix, "little")
        if not 2 <= header_bytes <= self.MAX_INDEX_BYTES or header_bytes > size - 8:
            raise ValueError("single safetensors header is invalid or exceeds its size cap")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            names = handle.keys()
            if not 1 <= len(names) <= self.MAX_INDEX_ENTRIES:
                raise ValueError("single safetensors header has no tensors or exceeds its entry cap")
            if any(not isinstance(key, str) or not key or len(key) > 512 for key in names):
                raise ValueError("single safetensors header has an invalid tensor name")
            mapping = dict.fromkeys(names, name)
        self._layout, self._weight_map = "single_file", mapping
        return mapping

    def _load_index(self) -> dict[str, str]:
        """Resolve the pinned layout; absence of an index is valid for 1B/2B."""
        if self._weight_map is not None:
            return self._weight_map
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

        single_file = False
        try:
            path = self._download("model.safetensors.index.json", probe=True)
        except LocalEntryNotFoundError:
            if self.allow_downloads:
                # A failed online HEAD with no cached file is not proof of remote absence.
                raise PilotError("Shard-index network/cache lookup failed; cannot infer checkpoint "
                                 "layout from this local cache miss. Retry when the Hub is reachable.") from None
            # Offline: accept only an actually cached, complete file at the SAME revision.
            single_file = True
        except EntryNotFoundError:
            # Confirmed remote file-not-found: the checkpoint may legitimately be unsharded.
            single_file = True
        if single_file:
            # Leave the exception context before opening a header; a corrupt local file
            # must not chain an underlying signed-URL/request error into a notebook traceback.
            return self._single_file_map()
        self._file_size(path, self.MAX_INDEX_BYTES, "safetensors index")
        # A bounded read also protects against file growth between stat and read.
        with path.open("rb") as handle:
            raw = handle.read(self.MAX_INDEX_BYTES + 1)
        if len(raw) > self.MAX_INDEX_BYTES:
            raise ValueError("safetensors index exceeds its size cap")
        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=self._unique_object,
            parse_constant=self._reject_constant,
        )
        if not isinstance(document, dict):
            raise ValueError("safetensors index must be a JSON object")
        weights = document.get("weight_map")
        if not isinstance(weights, dict) or not 1 <= len(weights) <= self.MAX_INDEX_ENTRIES:
            raise ValueError("safetensors index weight_map is empty/invalid/too large")
        for name, filename in weights.items():
            if not isinstance(name, str) or not name or len(name) > 512:
                raise ValueError("invalid safetensors index tensor name")
            if (
                not isinstance(filename, str) or ".." in filename
                or len(filename) > 200 or self._SHARD_NAME.fullmatch(filename) is None
            ):
                raise ValueError("unsafe safetensors shard filename; only basenames are allowed")
        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("safetensors index metadata must be an object")
        total = metadata.get("total_size")
        if total is not None and (type(total) is not int or not 0 < total <= self.MAX_MODEL_BYTES):
            raise ValueError("safetensors index total_size is invalid or exceeds its size cap")
        self._weight_map = dict(weights)
        self._layout = "sharded"
        return self._weight_map

    def read_tensor(self, name: str) -> Tensor:
        _native_name(name)
        filename = self._load_index()[name]
        path = self._download(filename)
        self._file_size(path, self.MAX_SHARD_BYTES, "safetensors shard")
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if name not in handle.keys():
                raise ValueError("indexed tensor is absent from its declared shard")
            shape = handle.get_slice(name).get_shape()
            if (
                not isinstance(shape, (list, tuple)) or not 1 <= len(shape) <= 8
                or any(type(size) is not int or size < 1 for size in shape)
            ):
                raise ValueError("invalid safetensors tensor header shape")
            # Conservative 8-byte/element cap covers every supported real weight dtype.
            if math.prod(shape) * 8 > self.MAX_TENSOR_BYTES:
                raise ValueError("selected safetensors tensor exceeds its size cap")
            result = handle.get_tensor(name)
            _finite(result, f"source {name}")
            if result.device.type != "cpu" or result.element_size() > 8:
                raise ValueError("source tensor must use a supported CPU floating dtype")
            result = result.detach().contiguous().clone()
        return result


@dataclass
class _Part:
    parameter: nn.Parameter
    selection: tuple[slice, ...]
    original: Tensor
    replacement: Tensor
    report: dict[str, Any]
    applied: bool = False


_SWAP_LOCK = threading.RLock()
_ACTIVE_SWAPS: WeakKeyDictionary = WeakKeyDictionary()


class WeightSwap:
    """Single-use, exception-safe selected-slice swap; overlapping layer swaps reject.

    Source tensors must have the full native parameter's exact shape. Only
    selected slices are retained, and all reads are staged before any mutation.
    Nested swaps on the same self_attn are rejected even for different groups:
    otherwise shared norms and the meaning of 'native delta' become ambiguous.
    RoutingSession may be used inside a swap to capture its native routed state.
    """

    def __init__(
        self, language_model: nn.Module, site: Site, source: WeightSource, *,
        projections: tuple[str, ...] = ("q_proj", "k_proj"), include_norms: bool = False,
        sham: bool = False, seed: int = 0,
    ) -> None:
        if not isinstance(site, Site) or not callable(getattr(source, "read_tensor", None)):
            raise ValueError("WeightSwap requires a Site and a WeightSource")
        if not isinstance(projections, tuple) or any(name not in _PROJECTIONS for name in projections):
            raise ValueError("projections must be a tuple of q_proj/k_proj/v_proj/o_proj names")
        if len(set(projections)) != len(projections) or (not projections and not include_norms):
            raise ValueError("select unique projections or explicitly include shared norms")
        if type(include_norms) is not bool or type(sham) is not bool:
            raise ValueError("include_norms/sham must be booleans")
        _index(seed, "seed")
        if seed >= 2**63:
            raise ValueError("seed must fit a signed 64-bit integer")
        self.language_model, self.site, self.source = language_model, site, source
        self.projections, self.include_norms, self.sham, self.seed = projections, include_norms, sham, seed
        self.summary: dict[str, Any] = {
            "site": {"layer": site.layer, "kv_group": site.kv_group},
            "sham": sham, "seed": seed, "active": False, "restored_bitexact": None,
            "source": {
                "kind": type(source).__name__, "repo_id": getattr(source, "repo_id", None),
                "revision": getattr(source, "revision", None),
            },
        }
        self._module: nn.Module | None = None
        self._parts: list[_Part] = []
        self._entered = False

    def __enter__(self) -> WeightSwap:
        if self._entered:
            raise RuntimeError("WeightSwap contexts are single-use")
        self._entered = True
        try:
            module, layout = _resolve_attention(self.language_model, self.site)
            self._module = module
            with _SWAP_LOCK:
                if module in _ACTIVE_SWAPS:
                    raise RuntimeError("nested/overlapping WeightSwap on one self_attn is forbidden")
                _ACTIVE_SWAPS[module] = self
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            names = self.projections + (("q_norm", "k_norm") if self.include_norms else ())
            for name in names:
                parameter = getattr(module, name).weight
                if not isinstance(parameter, nn.Parameter):
                    raise ValueError("selected weights must be native Parameters")
                _finite(parameter, f"native {name}.weight")
                if name == "q_proj":
                    selection = (layout.q_rows, slice(None))
                    axis, start, stop, scope = 0, layout.q_rows.start, layout.q_rows.stop, "all_query_heads_in_kv_group"
                elif name in ("k_proj", "v_proj"):
                    start = self.site.kv_group * layout.head_dim
                    stop = start + layout.head_dim
                    selection = (slice(start, stop), slice(None))
                    axis, scope = 0, "one_shared_kv_head"
                elif name == "o_proj":
                    selection = (slice(None), layout.q_rows)
                    axis, start, stop, scope = 1, layout.q_rows.start, layout.q_rows.stop, "all_query_heads_in_kv_group"
                else:
                    selection = (slice(None),)
                    axis, start, stop, scope = 0, 0, layout.head_dim, "layerwide_shared_head_dim_norm"
                native_name = f"language_model.model.layers.{self.site.layer}.self_attn.{name}.weight"
                full_source = self.source.read_tensor(native_name)
                _finite(full_source, f"source {native_name}")
                if full_source.device.type != "cpu" or full_source.shape != parameter.shape:
                    raise ValueError(f"source {native_name} must be a full CPU tensor of shape {tuple(parameter.shape)}")
                original = parameter[selection].detach().clone()
                _finite(original, f"native selected {native_name}")
                endpoint = full_source[selection].detach().to("cpu", original.dtype).contiguous().clone()
                _finite(endpoint, f"converted source {native_name}")
                before = original.to("cpu", torch.float64)
                source_delta = endpoint.to(torch.float64) - before
                _finite(source_delta, "source weight delta")
                if self.sham:
                    permutation = torch.randperm(source_delta.numel(), generator=generator, device="cpu")
                    intended_delta = source_delta.flatten().index_select(0, permutation).reshape(source_delta.shape)
                    replacement = (before + intended_delta).to(original.dtype)
                else:
                    intended_delta = source_delta
                    replacement = endpoint
                _finite(replacement, "replacement weight slice")
                applied_delta = replacement.to(torch.float64) - before
                changed = int(torch.count_nonzero(applied_delta).item())
                report = {
                    "name": native_name, "shape": list(parameter.shape),
                    "slice": {"axis": axis, "start": start, "stop": stop, "scope": scope},
                    "selected_shape": list(original.shape), "parameter_count": original.numel(),
                    "changed_parameter_count": changed, "before_frobnorm": _norm(before),
                    "after_frobnorm": _norm(replacement), "source_delta_frobnorm": _norm(source_delta),
                    "intended_delta_frobnorm": _norm(intended_delta), "delta_frobnorm": _norm(applied_delta),
                    "normalized_delta_norm": _relative_norm(_norm(applied_delta), _norm(before)),
                    "source_checksum": _checksum(full_source), "source_slice_checksum": _checksum(endpoint),
                    "native_slice_checksum": _checksum(original), "applied_slice_checksum": _checksum(replacement),
                    "source_dtype": str(full_source.dtype), "destination_dtype": str(parameter.dtype),
                    "applied_delta_rounding_rms": _rms(applied_delta - intended_delta),
                }
                self._parts.append(_Part(parameter, selection, original, replacement, report))
                # No full checkpoint tensor is retained by the swap/backups.
                del full_source
            with torch.no_grad():
                for part in self._parts:
                    part.applied = True  # Restore even if a device copy raises partway through.
                    part.parameter[part.selection].copy_(part.replacement)
                    if not torch.equal(part.parameter[part.selection].detach().cpu(), part.replacement):
                        raise RuntimeError("selected source endpoint copy was not bit-exact")
            reports = [part.report for part in self._parts]
            before_norm = math.hypot(*(report["before_frobnorm"] for report in reports))
            after_norm = math.hypot(*(report["after_frobnorm"] for report in reports))
            delta_norm = math.hypot(*(report["delta_frobnorm"] for report in reports))
            source_delta_norm = math.hypot(*(report["source_delta_frobnorm"] for report in reports))
            intended_norm = math.hypot(*(report["intended_delta_frobnorm"] for report in reports))
            if not all(math.isfinite(value) for value in (before_norm, after_norm, delta_norm, source_delta_norm, intended_norm)):
                raise ValueError("nonfinite aggregate weight norm diagnostic")
            changed_names = [report["name"] for report in reports if report["changed_parameter_count"]]
            self.summary.update({
                "active": True, "selected_names": [report["name"] for report in reports],
                "slices": reports, "paramcount": sum(report["parameter_count"] for report in reports),
                "changedtensors": len(changed_names), "changed_tensors": changed_names,
                "changed_parameter_count": sum(report["changed_parameter_count"] for report in reports),
                "before_frobnorm": before_norm, "after_frobnorm": after_norm,
                "delta_frobnorm": delta_norm, "source_delta_frobnorm": source_delta_norm,
                "intended_delta_frobnorm": intended_norm,
                "normalized_delta_norm": _relative_norm(delta_norm, before_norm),
                "intended_delta_norm_difference": abs(intended_norm - source_delta_norm),
                "applied_delta_norm_rounding_difference": abs(delta_norm - intended_norm),
                "shared_norm_scope": "layerwide_all_query_heads_and_all_kv_heads" if self.include_norms else "none",
                "biases_swapped": False, "norm_scope": "selected_slices_only",
            })
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self) -> None:
        errors = []
        try:
            with torch.no_grad():
                for part in reversed(self._parts):
                    if part.applied:
                        try:
                            part.parameter[part.selection].copy_(part.original)
                            if not torch.equal(part.parameter[part.selection], part.original):
                                raise RuntimeError("weight restoration was not bit-exact")
                            part.applied = False
                        except BaseException as error:
                            errors.append(error)
            self.summary["restored_bitexact"] = not errors
            self.summary["active"] = False
        finally:
            with _SWAP_LOCK:
                if self._module is not None and _ACTIVE_SWAPS.get(self._module) is self:
                    del _ACTIVE_SWAPS[self._module]
            self._parts.clear()
        if errors:
            raise RuntimeError("failed to restore one or more selected weight slices") from errors[0]

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self._restore()
        return False


def identical_input_replay(
    language_model: nn.Module, site: Site, source: WeightSource, capture: Capture, *,
    projections: tuple[str, ...] = ("q_proj", "k_proj"), include_norms: bool = False,
    sham: bool = False, seed: int = 0,
) -> dict[str, Any]:
    """Audit Q/K and conditional routing on exactly the same captured hidden input.

    This calls projections/norms only, not a model/attention forward. Native and
    swapped passes use identical hidden rows and actual captured RoPE/masks.
    The swap restores selected weights before return or on any failure. Report
    native-versus-capture reconstruction error separately: reduced-input GEMMs
    and conditional-only softmax need not be bit-exact to the original prefill.
    Results are engineering mediator measurements, not behavioral predictions.
    """
    if not isinstance(capture, Capture) or capture.site != site:
        raise ValueError("same-input replay requires a Capture from the exact selected Site")
    module, layout = _resolve_attention(language_model, site)
    with torch.no_grad():
        native_q, native_k, native_conditional = _replay_qk(module, layout, capture)
        reconstruction = {
            "q_rms": _rms(native_q.cpu() - capture.q),
            "k_rms": _rms(native_k.cpu() - capture.k),
            "conditional_l1": float((native_conditional.cpu() - capture.conditional).abs().sum(-1).mean().item()),
        }
        swap = WeightSwap(
            language_model, site, source, projections=projections,
            include_norms=include_norms, sham=sham, seed=seed,
        )
        with swap:
            source_q, source_k, source_conditional = _replay_qk(module, layout, capture)
            per_head = (source_conditional - native_conditional).abs().sum(-1).flatten()
            report = {
                "kind": "same_captured_hidden_input_weight_replay", "site": capture.metadata["site"],
                "row": capture.row, "image_columns": list(capture.image_columns),
                "sequence_length": capture.sequence_length,
                "q_rms_delta": _rms(source_q - native_q), "k_rms_delta": _rms(source_k - native_k),
                "conditional_routing_l1": float(per_head.mean().item()),
                "conditional_routing_l1_per_query_head": per_head.cpu().tolist(),
                "native_capture_reconstruction": reconstruction,
            }
        report["weight_swap"] = dict(swap.summary)
        return report