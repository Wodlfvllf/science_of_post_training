"""Scoped conditional-image routing for native, uncached Qwen3 attention.

The interface was checked against Transformers 4.57.6's Qwen3Attention:
q_norm/k_norm precede RoPE, consecutive query heads share one KV head, and
eager attention returns (projected_output, dense_attention_weights). This is
an independently implemented row intervention, NOT a replacement HF forward.
The native forward always runs; only an image-context difference is projected
through selected o_proj columns, without its bias, and added at one row.

All routing modes preserve the recipient's image mass, outside-image weights,
and image values, except the explicitly named value interventions. Captures
describe the native state BEFORE intervention and own detached CPU storage.
Prompt/token-prefix alignment remains the caller's additional responsibility.
Use eval(), eager attention, and use_cache=False. Numerical preflight is still
required on each real dtype/device; these helpers make no GPU/parity claim.

HF 4.57.6's model-output recorder wraps forward *inside* the module call and
records before forward hooks. Top-level output_attentions/output_hidden_states
recording is therefore rejected, rather than exposing stale attention or
leaving HF recorder wrappers installed on an exception. A direct self_attn
call can return dense weights: its selected image row is updated on a clone.
"""

from __future__ import annotations

import inspect
import math
import threading
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["Site", "Capture", "RouteAction", "RoutingSession"]


def _index(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative, zero-based integer")


def _columns(columns: tuple[int, ...], row: int, length: int | None = None) -> None:
    if not isinstance(columns, tuple) or not columns:
        raise ValueError("image_columns must be a nonempty tuple")
    for column in columns:
        _index(column, "image column")
    if tuple(sorted(set(columns))) != columns:
        raise ValueError("image_columns must be unique and strictly increasing")
    if columns[-1] > row or (length is not None and columns[-1] >= length):
        raise ValueError("image columns must be within the chosen row's causal support")


def _finite(tensor: Tensor, name: str) -> None:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        raise ValueError(f"{name} must be a real floating-point tensor")
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise ValueError(f"{name} must be a materialized, strided tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")


def _cpu_clone(tensor: Tensor) -> Tensor:
    return tensor.detach().to(device="cpu").contiguous().clone()


def _work_dtype(tensor: Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _rms(tensor: Tensor) -> float:
    # Small row diagnostics are accumulated on CPU in scaled FP64, avoiding overflow.
    cpu = tensor.detach().to("cpu", torch.float64)
    scale = float(cpu.abs().max().item())
    if scale == 0:
        return 0.0
    result = scale * float((cpu / scale).square().mean().sqrt().item())
    if not math.isfinite(result):
        raise ValueError("nonfinite RMS diagnostic")
    return result


@dataclass(frozen=True)
class Site:
    """A zero-based layer and an ENTIRE shared KV group, not an independent K head."""

    layer: int
    kv_group: int

    def __post_init__(self) -> None:
        _index(self.layer, "layer")
        _index(self.kv_group, "kv_group")


@dataclass
class Capture:
    """Bounded, CPU-owned native row state; Q/K are after normalization and RoPE.

    Shapes: q=[1,G,1,D], k/v=[1,1,J,D], conditional=[1,G,1,J],
    mass=[1,G,1,1]. G is all query heads sharing the selected KV head.
    Optional hidden inputs and actual rotary coordinates support same-input
    weight replay without storing the whole hidden-state sequence. mask_row
    stores only selected query heads' additive mask row, including text support.
    """

    q: Tensor
    k: Tensor
    v: Tensor
    conditional: Tensor
    mass: Tensor
    row: int
    image_columns: tuple[int, ...]
    sequence_length: int
    site: Site
    position_ids: Tensor | None = None
    cos_row: Tensor | None = None
    sin_row: Tensor | None = None
    cos_image: Tensor | None = None
    sin_image: Tensor | None = None
    input_rows: Tensor | None = None
    input_image: Tensor | None = None
    mask_row: Tensor | None = None

    def __post_init__(self) -> None:
        # Clone even CPU tensors, so neither the forward nor a source can alias a capture.
        for name in (
            "q", "k", "v", "conditional", "mass", "position_ids", "cos_row", "sin_row",
            "cos_image", "sin_image", "input_rows", "input_image", "mask_row",
        ):
            tensor = getattr(self, name)
            if tensor is not None:
                if not isinstance(tensor, Tensor):
                    raise ValueError(f"capture {name} must be a tensor")
                setattr(self, name, _cpu_clone(tensor))
        self.validate()

    def validate(self) -> None:
        """Recheck public mutable tensors when reused as a donor or replay input."""
        _index(self.row, "row")
        _index(self.sequence_length, "sequence_length")
        if self.sequence_length <= self.row or not isinstance(self.site, Site):
            raise ValueError("capture row/site is invalid")
        _columns(self.image_columns, self.row, self.sequence_length)
        for name in ("q", "k", "v", "conditional", "mass"):
            tensor = getattr(self, name)
            _finite(tensor, f"capture {name}")
            if tensor.device.type != "cpu":
                raise ValueError("captures must own CPU tensors")
        if self.q.ndim != 4 or self.q.shape[0] != 1 or self.q.shape[2] != 1:
            raise ValueError("capture q must have shape [1,group_queries,1,head_dim]")
        groups, dim, count = self.q.shape[1], self.q.shape[3], len(self.image_columns)
        if groups < 1 or dim < 2 or dim % 2:
            raise ValueError("capture query count/head_dim is invalid for Qwen3 RoPE")
        if self.k.shape != (1, 1, count, dim) or self.v.shape != self.k.shape:
            raise ValueError("capture k/v must have shape [1,1,image_columns,head_dim]")
        if self.conditional.shape != (1, groups, 1, count):
            raise ValueError("capture conditional shape does not match q/image_columns")
        if self.mass.shape != (1, groups, 1, 1):
            raise ValueError("capture mass shape does not match q")
        if bool((self.conditional < 0).any()) or not torch.allclose(
            self.conditional.float().sum(-1),
            torch.ones_like(self.conditional.float().sum(-1)),
            atol=1e-5, rtol=1e-5,
        ):
            raise ValueError("capture conditional must be a normalized nonnegative distribution")
        if bool((self.mass <= 0).any()) or bool((self.mass > 1.00001).any()):
            raise ValueError("capture image mass must be positive and at most one")
        for name, shape in (
            ("cos_row", (1, 1, dim)), ("sin_row", (1, 1, dim)),
            ("cos_image", (1, count, dim)), ("sin_image", (1, count, dim)),
        ):
            tensor = getattr(self, name)
            if tensor is not None:
                _finite(tensor, f"capture {name}")
                if tensor.shape != shape or tensor.device.type != "cpu":
                    raise ValueError(f"capture {name} has incompatible coordinates")
        if self.position_ids is not None and (
            self.position_ids.shape != (1, self.sequence_length)
            or self.position_ids.dtype not in (torch.int32, torch.int64)
            or self.position_ids.device.type != "cpu"
        ):
            raise ValueError("capture position_ids must be CPU integers [1,sequence_length]")
        for name, length in (("input_rows", 1), ("input_image", count)):
            tensor = getattr(self, name)
            if tensor is not None:
                _finite(tensor, f"capture {name}")
                if tensor.ndim != 3 or tensor.shape[:2] != (1, length):
                    raise ValueError(f"capture {name} has invalid hidden-input shape")
                if tensor.shape[2] < 1 or tensor.device.type != "cpu":
                    raise ValueError(f"capture {name} must own nonempty CPU hidden inputs")
        if self.input_rows is not None and self.input_image is not None:
            if self.input_rows.shape[2] != self.input_image.shape[2]:
                raise ValueError("capture hidden-input widths differ")
        if self.mask_row is not None:
            if self.mask_row.shape != (1, groups, 1, self.sequence_length):
                raise ValueError("capture mask_row has invalid shape")
            if not self.mask_row.is_floating_point() or self.mask_row.device.type != "cpu":
                raise ValueError("capture mask_row must be a CPU additive mask")
            if bool(torch.isnan(self.mask_row).any() | torch.isposinf(self.mask_row).any()):
                raise ValueError("capture mask_row contains NaN/+Inf")

    @property
    def metadata(self) -> dict[str, Any]:
        """Small JSON-friendly alignment identifiers; rotary tensors remain separate."""
        return {
            "row": self.row, "image_columns": list(self.image_columns),
            "sequence_length": self.sequence_length,
            "site": {"layer": self.site.layer, "kv_group": self.site.kv_group},
            "group_queries": self.q.shape[1], "head_dim": self.q.shape[3],
        }


@dataclass
class RouteAction:
    """Explicit image-only operation; donor mass is never substituted for recipient mass."""

    mode: str
    query: Capture | None = None
    key: Capture | None = None
    value: Capture | None = None
    distribution: Capture | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        required = {
            "q": ("query",), "k": ("key",), "qk": ("query", "key"),
            "v": ("value",), "distribution": ("distribution",),
            "route_permute": (), "value_permute": (), "uniform": (),
        }
        if not isinstance(self.mode, str) or self.mode not in required:
            raise ValueError(f"unsupported route mode: {self.mode!r}")
        _index(self.seed, "seed")
        if self.seed >= 2**63:
            raise ValueError("seed must fit a signed 64-bit integer")
        for name in ("query", "key", "value", "distribution"):
            donor = getattr(self, name)
            if name in required[self.mode]:
                if not isinstance(donor, Capture):
                    raise ValueError(f"{self.mode} requires a {name} Capture")
            elif donor is not None:
                raise ValueError(f"{self.mode} does not use a {name} donor")


@dataclass(frozen=True)
class _Layout:
    query_heads: int
    kv_heads: int
    group_queries: int
    head_dim: int
    hidden_size: int
    head_start: int

    @property
    def heads(self) -> slice:
        return slice(self.head_start, self.head_start + self.group_queries)

    @property
    def q_rows(self) -> slice:
        return slice(
            self.head_start * self.head_dim,
            (self.head_start + self.group_queries) * self.head_dim,
        )


def _resolve_attention(language_model: nn.Module, site: Site) -> tuple[nn.Module, _Layout]:
    """Resolve only the verified native Qwen3 layout; imports are runtime-only."""
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3ForCausalLM,
        Qwen3RMSNorm,
    )

    if type(language_model) is not Qwen3ForCausalLM or not isinstance(site, Site):
        raise ValueError("expected the native language_model Qwen3ForCausalLM and a Site")
    config = language_model.config
    heads, kv, dim, hidden = (
        config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.hidden_size,
    )
    if any(type(value) is not int or value < 1 for value in (heads, kv, dim, hidden)):
        raise ValueError("invalid Qwen3 attention dimensions")
    if heads % kv or dim % 2 or dim < 2:
        raise ValueError("Qwen3 requires integral GQA groups and an even head_dim")
    if len(language_model.model.layers) != config.num_hidden_layers:
        raise ValueError("Qwen3 layer count differs from its config")
    if not 0 <= site.layer < len(language_model.model.layers) or site.kv_group >= kv:
        raise ValueError("Site layer/kv_group is outside the native Qwen3 architecture")
    module = language_model.model.layers[site.layer].self_attn
    if type(module) is not Qwen3Attention or module.layer_idx != site.layer:
        raise ValueError("selected self_attn is not the verified native Qwen3Attention")
    if language_model.training or module.training:
        raise ValueError("attention interventions/weight swaps require eval(), not training")
    if config._attn_implementation != "eager" or module.config._attn_implementation != "eager":
        raise ValueError("only native eager Qwen3 attention is supported")
    if module.attention_dropout != 0:
        raise ValueError("attention_dropout must be zero")
    if module.head_dim != dim or module.num_key_value_groups != heads // kv:
        raise ValueError("native Qwen3 head/GQA layout differs from its config")
    if not math.isclose(float(module.scaling), dim**-0.5, rel_tol=1e-12):
        raise ValueError("unexpected Qwen3 attention scaling")
    shapes = {
        "q_proj": (heads * dim, hidden), "k_proj": (kv * dim, hidden),
        "v_proj": (kv * dim, hidden), "o_proj": (hidden, heads * dim),
    }
    reference = module.q_proj.weight
    if reference.device.type == "meta":
        raise ValueError("meta/offloaded attention parameters are unsupported")
    if reference.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("only unquantized FP16/BF16/FP32/FP64 native attention is supported")
    if torch.is_autocast_enabled(reference.device.type):
        raise ValueError("ambient autocast is unsupported; load/use an explicit model dtype")
    for name, shape in shapes.items():
        projection = getattr(module, name)
        if type(projection) is not nn.Linear or projection.weight.shape != shape:
            raise ValueError(f"{name} is not an unquantized native Linear of shape {shape}")
        if (projection.out_features, projection.in_features) != shape:
            raise ValueError(f"{name} Linear dimensions differ from its native weight shape")
        if not projection.weight.is_floating_point() or projection.weight.layout != torch.strided:
            raise ValueError(f"{name} must have materialized floating-point weights")
        if projection.weight.device.type == "meta":
            raise ValueError("meta/offloaded attention parameters are unsupported")
        if projection.weight.device != reference.device or projection.weight.dtype != reference.dtype:
            raise ValueError("selected attention projections must share device and dtype")
        if (projection.bias is not None) != bool(config.attention_bias):
            raise ValueError(f"{name} bias presence differs from the native config")
        if projection.bias is not None:
            _finite(projection.bias, f"{name}.bias")
            if projection.bias.shape != (shape[0],):
                raise ValueError(f"{name} bias shape is incompatible")
            if projection.bias.device != reference.device or projection.bias.dtype != reference.dtype:
                raise ValueError(f"{name} bias must share the projection dtype/device")
    for name in ("q_norm", "k_norm"):
        norm = getattr(module, name)
        if type(norm) is not Qwen3RMSNorm or norm.weight.shape != (dim,):
            raise ValueError(f"{name} is not the native head-dimension RMSNorm")
        _finite(norm.weight, f"{name}.weight")
        if norm.weight.device != reference.device or norm.weight.dtype != reference.dtype:
            raise ValueError("attention norms must share the projection device/dtype")
        epsilon = float(norm.variance_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0 or epsilon != config.rms_norm_eps:
            raise ValueError(f"{name} RMSNorm epsilon differs from the native config")
    return module, _Layout(heads, kv, heads // kv, dim, hidden, site.kv_group * (heads // kv))


def _arguments(module: nn.Module, args: tuple, kwargs: dict) -> dict[str, Any]:
    try:
        signature = inspect.signature(module.forward)
        bound = dict(signature.bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError) as error:
        raise ValueError("cannot bind the native Qwen3 forward interface") from error
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            bound.update(bound.pop(name, {}))
    return bound


def _reject_cache(module: nn.Module, inputs: dict[str, Any]) -> None:
    if any(inputs.get(name) is not None for name in ("past_key_values", "past_key_value")):
        raise ValueError("past/cache objects are forbidden, including empty DynamicCache")
    use_cache = inputs.get("use_cache")
    if use_cache is None:
        use_cache = getattr(module.config, "use_cache", True)
    if use_cache is not False:
        raise ValueError("use_cache=False is required (including config defaults)")


def _row_mask(
    module: nn.Module, layout: _Layout, mask: Tensor | None, hidden: Tensor, row: int,
    image_columns: tuple[int, ...],
) -> Tensor:
    length = hidden.shape[1]
    if mask is None:
        # Eager Qwen3 does not create a causal mask inside self_attn itself.
        if row != length - 1 or module.sliding_window is not None:
            raise ValueError("an actual causal/sliding mask is required for this row")
        selected = hidden.new_zeros((1, layout.group_queries, 1, length))
    else:
        if not isinstance(mask, Tensor) or mask.ndim != 4 or not mask.is_floating_point():
            raise ValueError("expected the native four-dimensional additive attention mask")
        if mask.device != hidden.device or mask.shape[0] != 1:
            raise ValueError("attention mask batch/device is incompatible")
        if mask.shape[1] not in (1, layout.query_heads):
            raise ValueError("attention mask head dimension is incompatible")
        if mask.shape[2] not in (1, length) or mask.shape[3] < length:
            raise ValueError("attention mask query/key dimensions are incompatible")
        mask_row = 0 if mask.shape[2] == 1 else row
        selected = mask[:, :, mask_row : mask_row + 1, :length]
        if selected.shape[1] == 1:
            selected = selected.expand(1, layout.group_queries, 1, length)
        else:
            selected = selected[:, layout.heads, :, :]
        if bool(torch.isnan(selected).any() | torch.isposinf(selected).any()):
            raise ValueError("attention mask contains NaN/+Inf")
    blocked = torch.isneginf(selected) | (selected <= torch.finfo(selected.dtype).min)
    if row + 1 < length and not bool(blocked[..., row + 1 :].all()):
        raise ValueError("attention mask permits future keys at the chosen row")
    indices = torch.tensor(image_columns, device=hidden.device, dtype=torch.long)
    if bool(blocked.index_select(-1, indices).any()):
        raise ValueError("an image column is masked or outside sliding/causal support")
    return selected


def _read_inputs(
    module: nn.Module, layout: _Layout, args: tuple, kwargs: dict, row: int,
    image_columns: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    inputs = _arguments(module, args, kwargs)
    _reject_cache(module, inputs)
    if module.training or module.config._attn_implementation != "eager":
        raise ValueError("native attention must remain eval/eager throughout the session")
    if module.attention_dropout != 0:
        raise ValueError("attention_dropout must remain zero")
    hidden = inputs.get("hidden_states")
    _finite(hidden, "hidden_states")
    if torch.is_autocast_enabled(hidden.device.type):
        raise ValueError("ambient autocast is unsupported during routing")
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != layout.hidden_size:
        raise ValueError("only batch-one native hidden_states [1,T,hidden_size] are supported")
    if hidden.device != module.q_proj.weight.device or hidden.dtype != module.q_proj.weight.dtype:
        raise ValueError("hidden_states must share the projection dtype/device (no autocast)")
    if row >= hidden.shape[1]:
        raise ValueError("chosen row is outside the current sequence")
    _columns(image_columns, row, hidden.shape[1])
    embeddings = inputs.get("position_embeddings")
    if not isinstance(embeddings, (tuple, list)) or len(embeddings) != 2:
        raise ValueError("native position_embeddings=(cos,sin) are required")
    cos, sin = embeddings
    for name, coordinates in (("cos", cos), ("sin", sin)):
        _finite(coordinates, name)
        if coordinates.shape != (1, hidden.shape[1], layout.head_dim):
            raise ValueError("rotary coordinates must be [1,T,head_dim]")
        if coordinates.device != hidden.device or coordinates.dtype != hidden.dtype:
            raise ValueError("rotary coordinates must share hidden-state device/dtype")
    mask = _row_mask(module, layout, inputs.get("attention_mask"), hidden, row, image_columns)
    positions = inputs.get("position_ids")
    if positions is not None and (
        not isinstance(positions, Tensor) or positions.shape != (1, hidden.shape[1])
        or positions.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("position_ids must be integers [1,T]")
    return hidden, cos, sin, mask, positions


def _linear(module: nn.Linear, hidden: Tensor) -> Tensor:
    # Read parameters, without invoking projection hooks a second time.
    return F.linear(hidden, module.weight.detach(), module.bias)


def _rope(states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    first, second = states.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return states * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


def _conditional(q: Tensor, k: Tensor, mask: Tensor, scaling: float) -> Tensor:
    logits = torch.matmul(q, k.transpose(-1, -2)) * scaling + mask
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    _finite(probabilities, "conditional probabilities")
    return probabilities


def _aligned(donor: Capture, recipient: Capture) -> None:
    donor.validate()
    if (
        donor.site != recipient.site or donor.row != recipient.row
        or donor.sequence_length != recipient.sequence_length
        or donor.image_columns != recipient.image_columns
        or donor.q.shape != recipient.q.shape or donor.k.shape != recipient.k.shape
    ):
        raise ValueError("donor site/head/dim/columns/row/sequence alignment differs")
    # Equal token indices alone do not imply equal RoPE scaling or coordinates.
    for name in ("cos_row", "sin_row", "cos_image", "sin_image"):
        left, right = getattr(donor, name), getattr(recipient, name)
        if left is None or right is None or not torch.equal(left, right):
            raise ValueError(f"donor rotary alignment differs or is missing: {name}")
    for name in ("position_ids", "mask_row"):
        left, right = getattr(donor, name), getattr(recipient, name)
        if (left is None) != (right is None):
            raise ValueError(f"donor {name} alignment metadata is missing")
        if left is not None and not torch.equal(left, right):
            raise ValueError(f"donor {name} alignment differs")


def _donor_tensor(donor: Capture, name: str, reference: Tensor) -> Tensor:
    result = getattr(donor, name).to(device=reference.device, dtype=reference.dtype)
    _finite(result, f"converted donor {name}")
    return result


def _image_weights(conditional: Tensor, mass: Tensor) -> Tensor:
    conditional = conditional.to(mass.dtype)
    conditional = conditional / conditional.sum(-1, keepdim=True)
    weights = conditional * mass
    # Correct only summation roundoff, not the donor's substantive distribution.
    largest = weights.argmax(-1, keepdim=True)
    for _ in range(2):
        weights = weights.scatter_add(-1, largest, mass - weights.sum(-1, keepdim=True))
    _finite(weights, "mass-preserving image weights")
    if bool((weights < 0).any()):
        raise ValueError("mass correction produced a negative image weight")
    return weights


_ROUTING_LOCK = threading.RLock()
_ACTIVE_ROUTING: WeakKeyDictionary = WeakKeyDictionary()


class RoutingSession:
    """Single-use context installing scoped validation/output hooks on selected layers.

    Pre-hooks only reject unsupported inputs before native cache mutation; output
    hooks implement the intervention. No input, parameter, forward method, or
    global HF function is replaced. A failed hook immediately removes all handles.
    Multiple KV groups in one layer are captured before any of their row edits.
    """

    def __init__(
        self, language_model: nn.Module, sites: tuple[Site, ...], *, row: int,
        image_columns: tuple[int, ...], actions: dict[Site, RouteAction] | None = None,
    ) -> None:
        _index(row, "row")
        _columns(image_columns, row)
        if not isinstance(sites, tuple) or not sites or any(not isinstance(s, Site) for s in sites):
            raise ValueError("sites must be a nonempty tuple of Site objects")
        if len(set(sites)) != len(sites):
            raise ValueError("duplicate routing sites")
        self.language_model, self.sites = language_model, sites
        self.row, self.image_columns = row, image_columns
        self.actions = dict(actions or {})
        if set(self.actions) - set(sites):
            raise ValueError("actions include sites not selected by this session")
        if any(action is not None and not isinstance(action, RouteAction) for action in self.actions.values()):
            raise ValueError("actions must contain RouteAction objects or capture-only None")
        self.captures: dict[Site, Capture] = {}
        self.diagnostics: dict[Site, dict[str, Any]] = {}
        self._handles: list[Any] = []
        self._modules: dict[nn.Module, list[tuple[Site, _Layout]]] = {}
        self._entered = False

    def __enter__(self) -> RoutingSession:
        if self._entered:
            raise RuntimeError("RoutingSession contexts are single-use")
        self._entered = True
        try:
            for site in self.sites:
                module, layout = _resolve_attention(self.language_model, site)
                self._modules.setdefault(module, []).append((site, layout))
            with _ROUTING_LOCK:
                if any(module in _ACTIVE_ROUTING for module in self._modules):
                    raise RuntimeError("nested/overlapping routing on one self_attn is forbidden")
                for module in self._modules:
                    _ACTIVE_ROUTING[module] = self
            for model in (self.language_model, self.language_model.model):
                self._handles.append(model.register_forward_pre_hook(
                    self._model_guard, with_kwargs=True, prepend=True,
                ))
            for module, entries in self._modules.items():
                self._handles.append(module.register_forward_pre_hook(
                    self._pre_hook(tuple(entries)), with_kwargs=True, prepend=True,
                ))
                self._handles.append(module.register_forward_hook(
                    self._output_hook(tuple(entries)), with_kwargs=True,
                ))
        except BaseException:
            self._close()
            raise
        return self

    def _close(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        with _ROUTING_LOCK:
            for module in self._modules:
                if _ACTIVE_ROUTING.get(module) is self:
                    del _ACTIVE_ROUTING[module]

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self._close()
        return False

    def _model_guard(self, module: nn.Module, args: tuple, kwargs: dict) -> None:
        try:
            inputs = _arguments(module, args, kwargs)
            _reject_cache(module, inputs)
            if module.training or module.config._attn_implementation != "eager":
                raise ValueError("model must remain eval/eager during routing")
            for name in ("output_attentions", "output_hidden_states"):
                requested = inputs.get(name)
                if requested is None:
                    requested = getattr(module.config, name, False)
                if requested:
                    raise ValueError(f"{name} recording precedes hooks in HF 4.57.6; disable it")
            for name in ("input_ids", "inputs_embeds"):
                tensor = inputs.get(name)
                if tensor is not None and (tensor.ndim < 2 or tensor.shape[0] != 1):
                    raise ValueError("routing supports batch size one only")
        except BaseException:
            self._close()
            raise

    def _pre_hook(self, entries: tuple[tuple[Site, _Layout], ...]):
        def validate(module: nn.Module, args: tuple, kwargs: dict) -> None:
            try:
                for _, layout in entries:
                    _read_inputs(module, layout, args, kwargs, self.row, self.image_columns)
            except BaseException:
                self._close()
                raise
        return validate

    def _capture(
        self, module: nn.Module, site: Site, layout: _Layout, args: tuple, kwargs: dict,
        native_weights: Tensor,
    ) -> tuple[Capture, Tensor, Tensor, Tensor, Tensor, Tensor, float]:
        hidden, cos, sin, mask, positions = _read_inputs(
            module, layout, args, kwargs, self.row, self.image_columns,
        )
        length, dim = hidden.shape[1], layout.head_dim
        indices = torch.tensor(self.image_columns, device=hidden.device, dtype=torch.long)
        # Keep native projection widths/GEMM inputs; only selected heads form a score row.
        raw_q = _linear(module.q_proj, hidden).view(1, length, layout.query_heads, dim)
        q = module.q_norm(raw_q[:, self.row : self.row + 1, layout.heads, :]).transpose(1, 2)
        q = _rope(q, cos[:, self.row : self.row + 1], sin[:, self.row : self.row + 1])
        raw_k = _linear(module.k_proj, hidden).view(1, length, layout.kv_heads, dim)
        k = module.k_norm(raw_k[:, :, site.kv_group : site.kv_group + 1]).transpose(1, 2)
        k = _rope(k, cos, sin)
        raw_v = _linear(module.v_proj, hidden).view(1, length, layout.kv_heads, dim)
        v = raw_v[:, :, site.kv_group : site.kv_group + 1].transpose(1, 2)
        for name, tensor in (("native q", q), ("native k", k), ("native image v", v.index_select(2, indices))):
            _finite(tensor, name)
        if native_weights.shape != (1, layout.query_heads, length, length):
            raise ValueError("native eager attention weights have an unexpected shape")
        row_weights = native_weights[:, layout.heads, self.row : self.row + 1, :].detach()
        _finite(row_weights, "native row weights")
        # Reconstruct all causal keys for diagnostics, but use the native row as the anchor.
        reconstructed = _conditional(q, k, mask, module.scaling).to(q.dtype)
        reconstruction_error = float((reconstructed.float() - row_weights.float()).abs().max().item())
        image_weights = row_weights.index_select(-1, indices).to(_work_dtype(row_weights))
        mass = image_weights.sum(-1, keepdim=True)
        if bool((mass <= 0).any()):
            raise ValueError("image mass is zero/underflowed; conditional routing is undefined")
        conditional = image_weights / mass
        image_k, image_v = k.index_select(2, indices), v.index_select(2, indices)
        capture = Capture(
            q=q, k=image_k, v=image_v, conditional=conditional, mass=mass,
            row=self.row, image_columns=self.image_columns, sequence_length=length, site=site,
            position_ids=positions, cos_row=cos[:, self.row : self.row + 1],
            sin_row=sin[:, self.row : self.row + 1], cos_image=cos.index_select(1, indices),
            sin_image=sin.index_select(1, indices), input_rows=hidden[:, self.row : self.row + 1],
            input_image=hidden.index_select(1, indices), mask_row=mask,
        )
        return capture, q, image_k, image_v, image_weights, mask.index_select(-1, indices), reconstruction_error

    def _intervene(
        self, action: RouteAction | None, capture: Capture, q: Tensor, k: Tensor, v: Tensor,
        old_weights: Tensor, image_mask: Tensor, scaling: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        native_conditional = capture.conditional.to(old_weights.device, old_weights.dtype)
        new_conditional, new_v = native_conditional, v
        if action is not None:
            action.__post_init__()
            for donor in (action.query, action.key, action.value, action.distribution):
                if donor is not None:
                    _aligned(donor, capture)
            if action.mode in ("q", "k", "qk"):
                new_q = _donor_tensor(action.query, "q", q) if action.query is not None else q
                new_k = _donor_tensor(action.key, "k", k) if action.key is not None else k
                # A self clamp must not re-softmax/renormalize native probabilities.
                if not (torch.equal(new_q, q) and torch.equal(new_k, k)):
                    new_conditional = _conditional(new_q, new_k, image_mask, scaling)
            elif action.mode == "distribution":
                new_conditional = _donor_tensor(action.distribution, "conditional", native_conditional)
            elif action.mode == "v":
                new_v = _donor_tensor(action.value, "v", v)
            elif action.mode == "uniform":
                new_conditional = torch.full_like(native_conditional, 1 / len(self.image_columns))
            else:
                generator = torch.Generator(device="cpu").manual_seed(action.seed)
                permutation = torch.randperm(
                    len(self.image_columns), generator=generator, device="cpu",
                ).to(v.device)
                if action.mode == "route_permute":
                    new_conditional = native_conditional.index_select(-1, permutation)
                else:
                    new_v = v.index_select(2, permutation)
        if torch.equal(new_conditional.to(native_conditional.dtype), native_conditional):
            new_weights = old_weights
        else:
            new_weights = _image_weights(new_conditional, capture.mass.to(old_weights.device, old_weights.dtype))
        # The difference contains no text context and no o_proj bias.
        work_v = v.to(old_weights.dtype)
        if torch.equal(new_v, v):
            context_delta = torch.matmul(new_weights - old_weights, work_v)
        else:
            context_delta = torch.matmul(old_weights, new_v.to(old_weights.dtype) - work_v)
        _finite(context_delta, "image context delta")
        return new_weights, context_delta, new_conditional.to(old_weights.dtype)

    def _output_hook(self, entries: tuple[tuple[Site, _Layout], ...]):
        def edit(module: nn.Module, args: tuple, kwargs: dict, output: Any):
            try:
                with torch.no_grad():
                    if not isinstance(output, tuple) or len(output) != 2:
                        raise ValueError("native Qwen3 self_attn must return (output, weights)")
                    native_output, native_weights = output
                    _finite(native_output, "native attention output")
                    hidden = _arguments(module, args, kwargs).get("hidden_states")
                    if native_output.shape != hidden.shape:
                        raise ValueError("native attention output must have shape [1,T,hidden_size]")
                    if not isinstance(native_weights, Tensor):
                        raise ValueError("native eager Qwen3 must return dense attention weights")
                    states = []
                    for site, layout in entries:
                        state = self._capture(module, site, layout, args, kwargs, native_weights)
                        self.captures[site] = state[0]
                        states.append((site, layout, state))
                    updated_weights, total_delta = native_weights, None
                    for site, layout, state in states:
                        capture, q, k, v, old_weights, image_mask, error = state
                        action = self.actions.get(site)
                        new_weights, context_delta, conditional = self._intervene(
                            action, capture, q, k, v, old_weights, image_mask, module.scaling,
                        )
                        selected_weight = module.o_proj.weight[:, layout.q_rows].detach()
                        _finite(selected_weight, "selected o_proj columns")
                        flat_delta = context_delta.transpose(1, 2).reshape(1, 1, -1)
                        if bool(torch.count_nonzero(flat_delta)):
                            delta = F.linear(flat_delta.to(selected_weight.dtype), selected_weight, bias=None)
                            _finite(delta, "projected row delta")
                            total_delta = delta if total_delta is None else total_delta + delta
                        else:
                            delta = native_output.new_zeros((1, 1, native_output.shape[-1]))
                        returned_image = new_weights.to(native_weights.dtype)
                        old_returned = old_weights.to(native_weights.dtype)
                        row_updated = not torch.equal(returned_image, old_returned)
                        if row_updated:
                            if updated_weights is native_weights:
                                updated_weights = native_weights.clone()
                            indices = torch.tensor(self.image_columns, device=native_weights.device)
                            updated_weights[:, layout.heads, self.row : self.row + 1].index_copy_(
                                -1, indices, returned_image,
                            )
                        before, after = old_weights.sum(-1), new_weights.sum(-1)
                        mass_diff = float((after - before).abs().max().item())
                        outside = torch.ones(capture.sequence_length, device=native_weights.device, dtype=torch.bool)
                        outside[list(self.image_columns)] = False
                        text_diff = (
                            updated_weights[:, layout.heads, self.row : self.row + 1, outside].float()
                            - native_weights[:, layout.heads, self.row : self.row + 1, outside].float()
                        )
                        text_maxdiff = float(text_diff.abs().max().item()) if text_diff.numel() else 0.0
                        self.diagnostics[site] = {
                            "mode": "capture" if action is None else action.mode,
                            "query_heads": list(range(layout.head_start, layout.head_start + layout.group_queries)),
                            "kv_group": site.kv_group, "captured_before_intervention": True,
                            "mass_before": before.flatten().tolist(), "mass_after": after.flatten().tolist(),
                            "mass_before_after_maxdiff": mass_diff, "mass_after_maxdiff": mass_diff,
                            "text_weight_maxdiff": text_maxdiff, "donor_finite": True,
                            "row_output_delta_rms": _rms(delta),
                            "conditional_l1": float((conditional - capture.conditional.to(conditional.device)).abs().sum(-1).mean().item()),
                            "reconstruction_weight_maxdiff": error,
                            "returned_attention_mass_maxdiff": float((returned_image.to(old_weights.dtype).sum(-1) - before).abs().max().item()),
                            "attention_row_updated": row_updated,
                            "context_accumulation_dtype": str(old_weights.dtype),
                        }
                    if total_delta is None or not bool(torch.count_nonzero(total_delta)):
                        return output if updated_weights is native_weights else (native_output, updated_weights)
                    modified = native_output.clone()
                    modified[:, self.row : self.row + 1] += total_delta
                    _finite(modified, "modified attention output")
                    return modified, updated_weights
            except BaseException:
                self._close()
                raise
        return edit


def _replay_qk(module: nn.Module, layout: _Layout, capture: Capture) -> tuple[Tensor, Tensor, Tensor]:
    """Project exactly the stored hidden rows/images, using stored actual RoPE/mask.

    Single-row GEMMs can round differently from a full native prefill. Callers
    must report that reconstruction error separately from a same-input delta.
    """
    capture.validate()
    if capture.input_rows is None or capture.input_image is None or capture.mask_row is None:
        raise ValueError("same-input replay requires captured hidden inputs and mask_row")
    if any(getattr(capture, name) is None for name in ("cos_row", "sin_row", "cos_image", "sin_image")):
        raise ValueError("same-input replay requires actual captured rotary coordinates")
    if capture.q.shape != (1, layout.group_queries, 1, layout.head_dim):
        raise ValueError("same-input replay query/head layout differs")
    if capture.input_rows.shape[-1] != layout.hidden_size:
        raise ValueError("same-input replay hidden width differs")
    reference = module.q_proj.weight
    row = capture.input_rows.to(reference.device, reference.dtype)
    image = capture.input_image.to(reference.device, reference.dtype)
    raw_q = _linear(module.q_proj, row).view(1, 1, layout.query_heads, layout.head_dim)
    q = module.q_norm(raw_q[:, :, layout.heads]).transpose(1, 2)
    raw_k = _linear(module.k_proj, image).view(1, len(capture.image_columns), layout.kv_heads, layout.head_dim)
    group = capture.site.kv_group
    k = module.k_norm(raw_k[:, :, group : group + 1]).transpose(1, 2)
    q = _rope(q, capture.cos_row.to(reference.device, reference.dtype), capture.sin_row.to(reference.device, reference.dtype))
    k = _rope(k, capture.cos_image.to(reference.device, reference.dtype), capture.sin_image.to(reference.device, reference.dtype))
    _finite(q, "same-input q")
    _finite(k, "same-input k")
    indices = torch.tensor(capture.image_columns, device=reference.device)
    mask = capture.mask_row.to(reference.device).index_select(-1, indices)
    return q, k, _conditional(q, k, mask, module.scaling)