"""Explicit streaming component-weight equality audit; no second model allocation."""

from __future__ import annotations

import math

from .io import digest, identity
from .schemas import PilotError


def audit_components(before, after, *, allow_downloads=False,
                     components=("vision_model", "mlp1")) -> dict:
    """Check actual saved tensors rather than inferring equality from a freeze recipe."""
    if before.family != after.family or before.backend != after.backend:
        raise PilotError("Component audits require matched architecture checkpoints.")
    if not components or any(c not in ("vision_model", "mlp1") for c in components):
        raise PilotError("Select vision_model and/or mlp1; full LM auditing is not implicit.")
    import torch

    from .runtime import FixtureModel
    from .weights import SafetensorSource

    if before.backend == "fixture":
        a = FixtureModel(before).eval()
        b = FixtureModel(after).eval()
        left = {name: tensor for name, tensor in a.state_dict().items()
                if any(name.startswith(c + ".") for c in components)}
        right = {name: tensor for name, tensor in b.state_dict().items()
                 if name in left}

        def reader(side, name):
            return (left if side == 0 else right)[name]

        names = sorted(left)
    else:
        from safetensors import safe_open

        sources = [SafetensorSource(c.model_id, c.revision, allow_downloads) for c in (before, after)]
        maps = [source._load_index() for source in sources]
        keys = [{name for name in mapping if any(name.startswith(c + ".") for c in components)}
                for mapping in maps]
        if not keys[0] or keys[0] != keys[1]:
            raise PilotError("Saved vision/projector tensor names disagree or are absent.")
        names = sorted(keys[0])

        def reader(side, name):
            source = sources[side]
            file = source._download(maps[side][name])
            source._file_size(file, source.MAX_SHARD_BYTES, "audit shard")
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                shape = handle.get_slice(name).get_shape()
                if not shape or math.prod(shape) * 8 > source.MAX_TENSOR_BYTES:
                    raise PilotError("Audit tensor exceeds bounded memory; no unbounded full-model read.")
                return handle.get_tensor(name).detach()
    report = []
    for name in names:
        left_tensor, right_tensor = reader(0, name), reader(1, name)
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            raise PilotError("Saved component tensor dtype/shape changed.")
        if left_tensor.is_floating_point() and (not torch.isfinite(left_tensor).all()
                or not torch.isfinite(right_tensor).all()):
            raise PilotError("Nonfinite component weights.")
        changed, maximum = 0, 0.
        # Only this tensor and a bounded numerical comparison slice are live.
        a_flat, b_flat = left_tensor.reshape(-1), right_tensor.reshape(-1)
        for start in range(0, a_flat.numel(), 262144):
            a_slice, b_slice = a_flat[start:start+262144], b_flat[start:start+262144]
            changed += int((a_slice != b_slice).sum().item())
            maximum = max(maximum, float((a_slice.double() - b_slice.double()).abs().max().item()))
        header = {"name": name, "shape": list(left_tensor.shape), "dtype": str(left_tensor.dtype)}
        report.append(header | {"parameters": left_tensor.numel(), "changed": changed,
            "max_abs": maximum,
            "before_sha256": digest(left_tensor.contiguous().view(torch.uint8).numpy().tobytes()),
            "after_sha256": digest(right_tensor.contiguous().view(torch.uint8).numpy().tobytes())})
    return {"before": before.model_dump(mode="json"), "after": after.model_dump(mode="json"),
            "components": list(components), "tensors": report,
            "bitexact_equal": bool(report) and all(row["changed"] == 0 for row in report),
            "audit_sha256": identity(report), "fixture_only": before.backend == "fixture",
            "interpretation": "Actual weight equality only; no optimizer-level causal claim."}