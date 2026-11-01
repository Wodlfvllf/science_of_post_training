"""Engineering-only counterhypothesis: selector information in TEXT residuals.

Independent zero-based post-decoder-block OUTPUT edits, not input edits, Q/K
image routing, training, stage attribution, confirmation or an author claim.
Import/argument parsing are stdlib-only and inert. Explicit dispatch uses ONE
pinned cached native Engine, common system/open assistant, answer_prefix=''.
Only recipient lower/Title color candidates are scored. Donor gold/candidates
are metadata: donors are captured only by Engine.forward, without answer tails.

ResidualSession also supports dense tiny nn.Module fixtures; the runtime adapter
restricts actual execution to native Qwen3. Captures own one finite CPU [1,1,H]
vector <=1 MiB. Only the selected text row of a cloned block output changes;
parameters, norms, native forwards and other rows remain untouched. Hooks and
no-grad state are scoped, including errors. Hidden-state/attention recorders
are unsupported; Engine explicitly disables both. No setup CUDA contexts or
early memory queries. Bounds cover payloads, not peak eager-attention memory.

Exact full-ID/per-word helper alignment precedes independently mapped word-end
rows. Traversal (before task words) is an always-run structural negative control:
same-image shape/color prefixes and native residual bytes must be identical.
Capture/self logits and recipient alias scores pass absolute 1e-4/argmax gates
before trials. Optional --audit alone enables six unrestricted patched greedy
32-token audits: first family, +1, finalassistant, layer 24 (must be supplied).
Format-invalid delivery and raw shape mentions are not semantic-zero findings.
No downloads, overwrite, resume, correctness filtering or implicit audit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from weakref import WeakKeyDictionary

CONDITIONS = ("native", "self", "shape_same", "shape_disjoint", "color_sham", "wrong_anchor_shape")
ROWS = ("task_noun", "finalassistant", "traversal")
MAX_VECTOR_BYTES, MAX_CAPTURE_BYTES, MAX_PREPARED = 1 << 20, 512 << 20, 10
SELF_ATOL, AUDIT_BUDGET = 1e-4, 32
_ACTIVE, _GUARD, _STARTED = WeakKeyDictionary(), threading.RLock(), False
__all__ = ["ResidualSession", "ResidualEngine", "FamilyCache", "row_map", "align_rows",
           "parse_layers", "parse_rows", "site_preflight", "prepare_family", "family_results",
           "patched_greedy_audit", "summarize", "main"]


def _source_hash(path):
    payload = Path(path).read_bytes()
    if not payload.strip():
        raise ValueError("Empty source cannot establish provenance.")
    return hashlib.sha256(payload).hexdigest()


def _load_helper():
    """Lazy exact adjacent-file import, not a sys.path/package-name lookup."""
    path = Path(__file__).resolve().with_name("vlm_handoff_diagnostic.py")
    checksum = _source_hash(path)
    spec = importlib.util.spec_from_file_location("_vlm_text_state_handoff", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load exact handoff helper.")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if _source_hash(path) != checksum:
        raise ValueError("Handoff helper changed during import.")
    return helper, checksum


def _vector(tensor, hidden):
    import torch

    if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
            or tensor.device.type != "cpu" or not tensor.is_floating_point()
            or tuple(tensor.shape) != (1, 1, hidden)
            or tensor.numel() * tensor.element_size() > MAX_VECTOR_BYTES
            or not bool(torch.isfinite(tensor).all())):
        raise ValueError("Donor must be finite dense CPU [1,1,native H] <=1 MiB.")
    return tensor.detach().contiguous().clone()


def vector_metadata(vector):
    """Hash RAW bytes (also BF16), not a rounded/converted numerical summary."""
    import torch

    vector = _vector(vector, vector.shape[-1])
    return {"shape": list(vector.shape), "dtype": str(vector.dtype), "device": "cpu",
            "bytes": vector.numel() * vector.element_size(), "finite": True,
            "requires_grad": False, "owned_clone": True,
            "sha256": hashlib.sha256(bytes(vector.view(torch.uint8).flatten().tolist())).hexdigest()}


def exact_vector(left, right):
    import torch

    lm, rm = vector_metadata(left), vector_metadata(right)
    delta = float((left.double() - right.double()).abs().max())
    result = {"passed": torch.equal(left, right) and lm == rm, "bit_exact": torch.equal(left, right),
              "atol": SELF_ATOL, "max_abs": delta, "left": lm, "right": rm}
    if not result["passed"] or delta > SELF_ATOL:
        raise ValueError("Native/self or causal traversal residual bytes differ.")
    return result


class ResidualSession:
    """Scoped layer OUTPUT capture/substitution at one zero-based text row.

    Generic fixture resolution is only language_model.model.layers; hidden_size
    and num_hidden_layers must match. Runtime Qwen3 checks live in ResidualEngine.
    Entering uses no_grad, never inference-mode/RNG/device setters. .capture is
    always the native PRE-EDIT CPU vector; .diagnostic records the actual delta.
    A session supports one block call; Engine.score uses a fresh session for
    each native forward, including append-only recipient candidate tails.
    """

    def __init__(self, language_model, layer, row, donor=None):
        import torch

        config = language_model.config
        layers = language_model.model.layers
        hidden = getattr(config, "hidden_size", None)
        if (not isinstance(language_model, torch.nn.Module) or type(layer) is not int
                or type(row) is not int or row < 0 or not 0 <= layer < len(layers)
                or type(getattr(config, "num_hidden_layers", None)) is not int
                or config.num_hidden_layers != len(layers)
                or type(hidden) is not int or hidden <= 0):
            raise ValueError("Native dense model/layer/row/hidden-size required.")
        self.language_model, self.module = language_model, layers[layer]
        self.layer, self.row, self.hidden = layer, row, hidden
        self.donor = None if donor is None else _vector(donor, hidden)
        self.capture, self.diagnostic = None, None
        self._handles, self._grad, self._entered = [], None, False

    def _eval_check(self):
        if any(m.training for m in self.language_model.modules()):
            raise ValueError("Residual sessions require every module in eval mode.")
        config = self.language_model.config
        if any(getattr(config, k, False) for k in ("output_hidden_states", "output_attentions", "use_cache")):
            raise ValueError("Hidden-state/attention recorders and KV cache are unsupported.")

    def _recording_guard(self, module, args, kwargs):
        self._eval_check()
        if any(kwargs.get(k, False) for k in ("output_hidden_states", "output_attentions", "use_cache")):
            raise ValueError("Hidden-state/attention recorders and KV cache are unsupported.")

    def _hook(self, module, args, output):
        import torch

        try:
            self._eval_check()
            value = output[0] if isinstance(output, tuple) and output else output
            if (self.capture is not None or not isinstance(value, torch.Tensor)
                    or value.layout != torch.strided or value.device.type not in ("cpu", "cuda")
                    or not value.is_floating_point() or value.ndim != 3
                    or value.shape[0] != 1 or value.shape[1] <= self.row
                    or value.shape[2] != self.hidden or value.requires_grad
                    or self.hidden * value.element_size() > MAX_VECTOR_BYTES):
                raise ValueError("Expected one dense no-grad batch-1 native postblock [1,S,H] output.")
            self.capture = _vector(value[:, self.row:self.row + 1].detach().to("cpu"), self.hidden)
            converted = (self.capture if self.donor is None else
                         self.donor.to(device=value.device, dtype=value.dtype))
            if not bool(torch.isfinite(converted).all()):
                raise ValueError("Converted donor is nonfinite.")
            returned = converted.detach().to("cpu").clone()
            delta = returned.double() - self.capture.double()
            scale = float(delta.abs().max())
            rms = scale * float((delta / scale).square().mean().sqrt()) if scale else 0.0
            if not math.isfinite(rms):
                raise ValueError("Nonfinite residual delta RMS.")
            self.diagnostic = {"layer": self.layer, "row": self.row, "scope": "postblock_output_text_row",
                               "sequence_length": value.shape[1], "native_before_edit": True,
                               "native": vector_metadata(self.capture), "substituted": self.donor is not None,
                               "converted_vector": vector_metadata(returned), "output_delta_rms": rms,
                               "embedding_delta_rms": 0.0, "parameters_or_norms_modified": False}
            if self.donor is None:
                return output
            edited = value.clone()
            edited[:, self.row:self.row + 1] = converted
            if not isinstance(output, tuple):
                return edited
            if type(output) is tuple:
                return (edited, *output[1:])
            if hasattr(output, "_fields"):
                return output._replace(**{output._fields[0]: edited})
            return type(output)((edited, *output[1:]))
        except BaseException:
            self._remove_hooks()
            raise

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        import torch

        with _GUARD:
            if self._entered or self.module in _ACTIVE:
                raise ValueError("Nested/concurrent residual session at the same module.")
            self._eval_check()
            _ACTIVE[self.module] = self
            self._entered = True
        try:
            self._grad = torch.no_grad()
            self._grad.__enter__()
            self._handles.append(self.language_model.register_forward_pre_hook(
                self._recording_guard, with_kwargs=True))
            self._handles.append(self.module.register_forward_hook(self._hook))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        self._remove_hooks()
        with _GUARD:
            if _ACTIVE.get(self.module) is self:
                del _ACTIVE[self.module]
            self._entered = False
        if self._grad is not None:
            self._grad.__exit__(*exc)
            self._grad = None


def parse_layers(value):
    parts = value.split(",") if isinstance(value, str) else []
    if not 1 <= len(parts) <= 4 or any(re.fullmatch(r"[0-9]+", p.strip()) is None for p in parts):
        raise argparse.ArgumentTypeError("Supply 1..4 unique zero-based layers, e.g. 8,16,24,32.")
    result = tuple(int(p.strip()) for p in parts)
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Supply 1..4 unique layers.")
    return result


def parse_rows(value):
    result = tuple(p.strip() for p in value.split(",")) if isinstance(value, str) else ()
    if (not {"task_noun", "finalassistant"} <= set(result)
            or len(set(result)) != len(result) or any(p not in ROWS for p in result)):
        raise argparse.ArgumentTypeError("Supply task_noun,finalassistant, optionally traversal; no duplicates.")
    return result


def _device(value):
    if re.fullmatch(r"cpu|cuda:[0-9]+", value) is None:
        raise argparse.ArgumentTypeError("Choose cpu or explicit cuda:N.")
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True, type=str if name == "checkpoint" else Path)
    parser.add_argument("--families", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--layers", required=True, type=parse_layers)
    parser.add_argument("--rows", required=True, type=parse_rows)
    parser.add_argument("--device", type=_device, default="cuda:0")
    parser.add_argument("--audit", action="store_true", default=False)
    return parser


def row_map(engine, prepared):
    """FIRST destination color/shape noun END; traversal word END; final row.

    Unique regex matches are required; the later 'one color/shape name' is NOT
    the task_noun row. Every source is retokenized at its own actual word ends.
    """
    ids, question = prepared.input_ids, prepared.case["question"]
    if (not isinstance(ids, tuple) or not ids or any(type(i) is not int or i < 0 for i in ids)
            or prepared.row != len(ids) - 1 or prepared.embeddings.shape[:2] != (1, len(ids))
            or len(prepared.embeddings.shape) != 3 or engine.settings.answer_prefix
            or prepared.prompt != engine._prompt(question, True, "common")
            or tuple(engine.tokenizer.encode(prepared.prompt, add_special_tokens=True)) != ids
            or prepared.prompt.count(question) != 1):
        raise ValueError("Original full native open-assistant IDs/embeddings required; no padding.")
    columns = tuple(i for i, token in enumerate(ids) if token == engine.img_token)
    if len(columns) != 256 or columns != prepared.image_columns:
        raise ValueError("Actual full image columns differ.")
    patterns = {"task_noun": r"\bdestination (color|shape)\b",
                "traversal": r"\bfollow one arrow (forward|backward)\b"}
    ends = {"finalassistant": len(prepared.prompt)}
    for name, pattern in patterns.items():
        found = list(re.finditer(pattern, question))
        if len(found) != 1:
            raise ValueError("Unique semantic word-end required.")
        match = found[0]
        expected = (prepared.case["answer_kind"] if name == "task_noun" else
                    "forward" if prepared.case["direction"] == 1 else "backward")
        if match[1] != expected:
            raise ValueError("Task/traversal word disagrees with case metadata.")
        ends[name] = prepared.prompt.index(question) + match.end(1)
    result = {}
    for name in ROWS:
        end = ends[name]
        prefix = tuple(engine.tokenizer.encode(prepared.prompt[:end], add_special_tokens=True))
        row = len(prefix) - 1
        if not prefix or prefix != ids[:len(prefix)] or not columns[-1] < row < len(ids):
            raise ValueError("Word-end encoding is not an exact actual text prefix.")
        result[name] = {"row": row, "character_end": end, "prefix_token_ids": list(prefix),
                        "exact_prefix": True}
    if len({v["row"] for v in result.values()}) != len(ROWS):
        raise ValueError("Semantic rows must be distinct.")
    return result


def align_rows(helper, engine, recipient, donor):
    alignment = helper.require_alignment(engine, recipient, donor)  # FULL originals FIRST.
    left, right = row_map(engine, recipient), row_map(engine, donor)
    if any(left[name]["row"] != right[name]["row"] for name in ROWS):
        raise ValueError("Donor's own word-end rows differ; no compensating padding.")
    return {"full_alignment": alignment, "recipient_rows": left, "donor_rows": right,
            "all_rows_same_positions": True}


class ResidualEngine:
    """Composition adapter: native Engine.score dispatches through scoped forwards.

    No instance/class/global forward is replaced. Each Engine.forward (including
    candidate tails) gets its own fully removed hook; original Prepared.row stays
    fixed. Native capture bytes must remain identical across appended score tails.
    """

    def __init__(self, engine, layer, donor=None):
        config = engine.language_model.config
        if (type(engine.language_model).__name__ != "Qwen3ForCausalLM"
                or config._attn_implementation != "eager"
                or any(getattr(config, k, False) for k in ("output_hidden_states", "output_attentions", "use_cache"))):
            raise ValueError("Native eager uncached Qwen3 with recorders disabled required.")
        self.engine, self.layer, self.donor = engine, layer, donor
        self.sessions = []

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def forward(self, prepared, **kwargs):
        if prepared.row in prepared.image_columns or not 0 <= prepared.row < len(prepared.input_ids):
            raise ValueError("Fixed original TEXT row required, not an image/candidate row.")
        with ResidualSession(self.language_model, self.layer, prepared.row, self.donor) as session:
            result = self.engine.forward(prepared, **kwargs)
        if session.capture is None:
            raise ValueError("Native Engine.forward omitted the selected decoder block.")
        if self.sessions:
            exact_vector(self.sessions[0].capture, session.capture)
        self.sessions.append(session)
        return result

    def score(self, prepared):
        return type(self.engine).score(self, prepared)


class FamilyCache:
    """At most ten full prompts and 512 MiB native single-vector captures/family."""

    def __init__(self, helper, engine, root, family, layers):
        if (family.get("split") != "engineering" or not isinstance(layers, tuple)
            or not 1 <= len(layers) <= 4 or len(set(layers)) != len(layers)
            or any(type(layer) is not int or layer < 0 for layer in layers)):
            raise ValueError("Only engineering families and 1..4 explicit unique layers.")
        self.helper, self.engine, self.root, self.family, self.layers = helper, engine, root, family, layers
        self.prepared, self.captures, self.bytes = {}, {}, 0

    def prepare(self, case):
        key = case["case_id"]
        if key not in self.prepared:
            if len(self.prepared) >= MAX_PREPARED:
                raise ValueError("Prepared family cache exceeded ten prompts.")
            self.prepared[key] = self.engine.prepare(self.root, self.family, case, profile="common")
        elif self.prepared[key].case != case:
            raise ValueError("Case ID reused with changed data.")
        return self.prepared[key]

    def capture(self, prepared, layer, row):
        if layer not in self.layers or self.prepared.get(prepared.case["case_id"]) is not prepared:
            raise ValueError("Capture requires selected layer and exact full cached source.")
        key = prepared.case["case_id"], layer, row
        if key not in self.captures:
            adapter = ResidualEngine(self.engine, layer)
            adapter.forward(replace(prepared, row=row))  # NEVER score/generate a donor.
            session = adapter.sessions[0]
            size = session.capture.numel() * session.capture.element_size()
            if self.bytes + size > MAX_CAPTURE_BYTES:
                raise ValueError("Family native captures exceed 512 MiB.")
            self.bytes += size
            self.captures[key] = session.capture, session.diagnostic
        return self.captures[key]

    def clear(self):
        self.prepared.clear()
        self.captures.clear()
        self.bytes = 0
        self.engine._features.clear()
        self.engine._family = None


def _logit_check(reference, actual):
    import torch

    if reference.shape != actual.shape or not bool(torch.isfinite(actual).all()):
        raise ValueError("Missing/nonfinite self logits.")
    delta = float((reference.double() - actual.double()).abs().max())
    same = torch.equal(reference.argmax(-1), actual.argmax(-1))
    if not bool(torch.isfinite(reference).all()) or delta > SELF_ATOL or not same:
        raise ValueError("Absolute capture/self logit parity failed.")
    return {"passed": True, "atol": SELF_ATOL, "rtol": 0, "max_abs": delta, "argmax_match": same}


def _score(cache, view, layer, donor, base):
    adapter = ResidualEngine(cache.engine, layer, donor)
    result, _ = adapter.score(view)
    if not adapter.sessions:
        raise ValueError("Score did not execute a native forward.")
    for session in adapter.sessions:
        exact_vector(base, session.capture)
    return result, [s.diagnostic for s in adapter.sessions]


def site_preflight(cache, recipient, layer, row, classify):
    """Unhooked/capture/self logits AND alias scores, before donor trials."""
    view = replace(recipient, row=row)
    base, _ = cache.capture(recipient, layer, row)
    reference_logits, _, _ = cache.engine.forward(view)
    reference, _ = cache.engine.score(view)
    checks = {}
    candidates = [c for c in recipient.case["candidates"] if c == c.lower()]
    for name, donor in (("capture", None), ("self", base)):
        adapter = ResidualEngine(cache.engine, layer, donor)
        logits, _, _ = adapter.forward(view)
        exact = exact_vector(base, adapter.sessions[0].capture)
        scored, diagnostics = _score(cache, view, layer, donor, base)
        checks[name] = {"logits": _logit_check(reference_logits, logits), "vector": exact,
                        "scores": cache.helper.self_score_check(reference, scored, classify, candidates),
                        "score": scored, "residual": diagnostics}
    return {"passed": True, "layer": layer, "row": row, "checks": checks,
            "native_unhooked_score": reference}


def prepare_family(cache, structural, progress=None):
    """Full alignment + same-image pretask causality for BOTH directions first."""
    bundles = {}
    names = {"self": "self_q", "shape_same": "shape_q", "shape_disjoint": "shape_q_disjoint",
             "color_sham": "color_sham", "wrong_anchor_shape": "wrong_anchor_shape_q"}
    if set(cache.family["palettes"][0]) & set(cache.family["palettes"][1]):
        raise ValueError("Disjoint palette controls require disjoint colors.")
    for direction in (1, -1):
        color, _ = cache.helper.paired_cases(cache.family, 1, direction, 0)
        probe = dict(color, candidates=color["candidates"] + [c.title() for c in color["candidates"]])
        recipient = cache.prepare(probe)
        cases = cache.helper.donor_cases(cache.family, color)
        cases["self_q"] = probe
        donors = {}
        for name, old in names.items():
            donor = cache.prepare(cases[old])
            donors[name] = donor, align_rows(cache.helper, cache.engine, recipient, donor)
        mapping = donors["self"][1]["recipient_rows"]
        shape, alignment = donors["shape_same"]
        r = mapping["traversal"]["row"]
        if (shape.case["image_id"] != recipient.case["image_id"]
                or alignment["donor_rows"]["traversal"]["prefix_token_ids"] != mapping["traversal"]["prefix_token_ids"]):
            raise ValueError("Structural control changed image or pretask prefix.")
        for layer in cache.layers:
            if progress:
                progress("structural_capture", family=cache.family["family_id"], direction=direction, layer=layer)
            left, _ = cache.capture(recipient, layer, r)
            right, _ = cache.capture(shape, layer, r)
            structural.append({"family_id": cache.family["family_id"], "direction": direction,
                               "layer": layer, "row": r, "same_image": True, "same_prefix": True,
                               "check": exact_vector(left, right), "passed": True})
        bundles[direction] = recipient, donors, mapping
    return bundles


def patched_greedy_audit(engine, view, layer, donor, shapes, progress=None):
    """Full-vocabulary native greedy; hooks removed after EVERY forward."""
    import torch

    eos = engine.tokenizer.eos_token_id
    module = importlib.import_module(type(engine.model).__module__)
    separator = module.get_conv_template(engine.model.template).sep.strip()
    if (type(eos) is not int or eos < 0 or layer != 24 or view.row != len(view.input_ids) - 1
            or len(view.input_ids) + AUDIT_BUDGET > engine.settings.max_context_tokens
            or engine.tokenizer.convert_tokens_to_ids(separator) != eos
            or engine.tokenizer.encode(separator, add_special_tokens=False) != [eos]):
        raise ValueError("Layer24/finalassistant/context/native EOS audit guard failed.")
    current, tokens, steps, native = view, [], [], None
    for index in range(AUDIT_BUDGET):
        if current.row != view.row or current.input_ids != view.input_ids:
            raise ValueError("Audit moved original recipient row/IDs.")
        adapter = ResidualEngine(engine, layer, donor)
        logits, _, _ = adapter.forward(current)
        captured = adapter.sessions[0].capture
        if native is not None:
            exact_vector(native, captured)
        native = captured
        token = int(logits[0, -1].argmax())
        tokens.append(token)
        steps.append({"step": index + 1, "token_id": token, "residual": adapter.sessions[0].diagnostic})
        if progress:
            progress("patched_greedy_audit", generated=len(tokens), token_budget=AUDIT_BUDGET)
        if token == eos or index == AUDIT_BUDGET - 1:
            break
        with torch.no_grad():
            embedding = engine.language_model.get_input_embeddings()(torch.tensor([[token]], device=engine.device))
            current = replace(current, embeddings=torch.cat((current.embeddings, embedding), 1))
    text = engine.tokenizer.decode(tokens, skip_special_tokens=True)
    raw = engine.tokenizer.decode(tokens, skip_special_tokens=False)
    answer = text.strip()
    ended = tokens[-1] == eos
    candidates = [c for c in view.case["candidates"] if c == c.lower()]
    valid = ended and answer in candidates
    return {"status": "completed", "token_ids": tokens, "text": text, "raw_text": raw,
            "finish_reason": "eos" if ended else "length", "native_eos_id": eos, "native_separator": separator,
            "valid_bare_lower": valid, "format_invalid": not valid,
            "parsed_answer": answer if valid else None, "exact_correct": valid and answer == view.case["gold"],
            "semantic_correctness": "not_assessed", "shape_word_response": answer.lower() in shapes,
            "raw_shape_mentions": [s for s in shapes if re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", raw, re.I)],
            "steps": steps, "settings": {"max_new_tokens": AUDIT_BUDGET, "full_vocabulary": True,
            "patched": True, "do_sample": False, "use_cache": False, "fixed_row": view.row, "layer": layer}}


def family_results(cache, bundles, selected_rows, classify, *, audit=False, progress=None):
    for direction, (recipient, donors, mapping) in bundles.items():
        candidates = [c for c in recipient.case["candidates"] if c == c.lower()]
        for name in selected_rows:
            row = mapping[name]["row"]
            view = replace(recipient, row=row)
            reference, _ = cache.engine.score(view)
            for layer in cache.layers:
                base, native_meta = cache.capture(recipient, layer, row)
                for condition in CONDITIONS:
                    donor, payload, alignment = None, None, None
                    if condition != "native":
                        donor, alignment = donors[condition]
                        payload, _ = cache.capture(donor, layer, row)
                    scored, diagnostic = _score(cache, view, layer, payload, base)
                    if scored["candidate_token_ids"] != reference["candidate_token_ids"]:
                        raise ValueError("Condition changed recipient candidate events.")
                    if not scored["scores"] or any(not math.isfinite(v) for v in scored["scores"].values()):
                        raise ValueError("Missing/nonfinite recipient scores.")
                    parity = (cache.helper.self_score_check(reference, scored, classify, candidates)
                              if condition in ("native", "self") else None)
                    grading = classify(scored, candidates)
                    case, family = recipient.case, cache.family
                    selected = donor.case["target"] if donor else None
                    targets = {"primary_recipient_gold": case["gold"],
                               "recipient_anchor_color": family["palettes"][0][case["anchor"]],
                               "recipient_color_at_donor_target": family["palettes"][0][selected] if donor else None,
                               "donor_palette_color_at_donor_target": family["palettes"][donor.case["palette"]][selected] if donor else None,
                               "donor_answer_metadata_only": donor.case["gold"] if donor else None}
                    outcomes = {g: {key: value is not None and grading[g] == value for key, value in targets.items()}
                                for g in ("lower", "title", "combined")}
                    for g in outcomes:
                        outcomes[g]["wrong_role_counterfactual_hit"] = condition == "wrong_anchor_shape" and outcomes[g]["recipient_color_at_donor_target"]
                        outcomes[g]["disjoint_donor_color_transfer"] = condition in ("shape_disjoint", "color_sham", "wrong_anchor_shape") and outcomes[g]["donor_palette_color_at_donor_target"]
                    generation = (patched_greedy_audit(cache.engine, view, layer, payload,
                                  [n["shape"] for n in family["nodes"]], progress)
                                  if audit and direction == 1 and name == "finalassistant" and layer == 24 else None)
                    yield {"family_id": family["family_id"], "direction": direction, "edge": 1, "palette": 0,
                           "layer": layer, "row_name": name, "row": row, "row_map": mapping,
                           "condition": condition, "case": case, "score": scored, "grading": grading,
                           "native_unhooked_score": reference, "self_score_check": parity,
                           "targets": targets, "outcomes": outcomes, "native_capture": native_meta,
                           "native_capture_unchanged": True, "residual": diagnostic, "generation_audit": generation,
                           "donor": ({**cache.helper._donor_metadata(family, donor, alignment),
                                      "residual_vector": vector_metadata(payload)} if donor else None),
                           "scope": "zero_based_postblock_output_TEXT_row_not_QK_image_route",
                           "source_status": "native_preanswer_capture", "status": "completed"}


def summarize(rows, family_ids, layers, selected_rows, *, p0_passed, structural, audit=False, failure=None):
    schedule = {(f, d, layer, r, c) for f in family_ids for d in (1, -1) for layer in layers
                for r in selected_rows for c in CONDITIONS}
    keys = [(r["family_id"], r["direction"], r["layer"], r["row_name"], r["condition"]) for r in rows]
    structural_keys = {(s["family_id"], s["direction"], s["layer"]) for s in structural if s.get("passed")}
    structural_expected = {(f, d, layer) for f in family_ids for d in (1, -1) for layer in layers}
    structural_passed = (structural_keys == structural_expected and len(structural) == len(structural_expected)
                         and all(s.get("check", {}).get("passed") and s.get("same_image") and s.get("same_prefix") for s in structural))
    audits = [r for r in rows if r.get("generation_audit") is not None]

    def full_audit(result):
        generation = result["generation_audit"]
        settings, tokens = generation.get("settings", {}), generation.get("token_ids", [])
        eos, steps = generation.get("native_eos_id"), generation.get("steps", [])
        return (generation.get("status") == "completed" and type(eos) is int and eos >= 0
                and isinstance(tokens, list) and 1 <= len(tokens) <= AUDIT_BUDGET
                and all(type(t) is int and t >= 0 for t in tokens)
                and len(steps) == len(tokens)
                and all(s.get("step") == i + 1 and s.get("token_id") == t for i, (s, t) in enumerate(zip(steps, tokens, strict=True)))
                and generation.get("finish_reason") == ("eos" if tokens[-1] == eos else "length")
                and (tokens[-1] == eos or len(tokens) == AUDIT_BUDGET) and eos not in tokens[:-1]
                and settings.get("max_new_tokens") == AUDIT_BUDGET and settings.get("patched") is True
                and settings.get("full_vocabulary") is True and settings.get("do_sample") is False
                and settings.get("use_cache") is False and settings.get("layer") == 24
                and settings.get("fixed_row") == result.get("row"))

    audit_complete = (not audit and not audits) or (audit and len(audits) == 6
        and {r["condition"] for r in audits} == set(CONDITIONS)
        and all(r["family_id"] == family_ids[0] and r["direction"] == 1 and r["layer"] == 24
                and r["row_name"] == "finalassistant" and full_audit(r) for r in audits))
    groups = defaultdict(list)
    for result in rows:
        groups[(result["layer"], result["row_name"], result["direction"], result["condition"])].append(result)
    cells = [{"layer": layer, "row_name": r, "direction": d, "condition": c, "n_result": len(v),
              "n_family": len({x["family_id"] for x in v}), "expected_n_family": len(family_ids),
              "n_primary_hit_combined": sum(x["outcomes"]["combined"]["primary_recipient_gold"] for x in v)}
             for (layer, r, d, c), v in groups.items()]
    valid_axes = (1 <= len(layers) <= 4 and len(set(layers)) == len(layers)
                  and all(type(layer) is int and layer >= 0 for layer in layers)
                  and 2 <= len(selected_rows) <= 3 and len(set(selected_rows)) == len(selected_rows)
                  and {"task_noun", "finalassistant"} <= set(selected_rows)
                  and set(selected_rows) <= set(ROWS))
    complete = (valid_axes and bool(schedule) and 1 <= len(family_ids) <= 4 and len(set(family_ids)) == len(family_ids)
                and p0_passed and structural_passed and audit_complete and failure is None
                and len(keys) == len(set(keys)) == len(schedule) and set(keys) == schedule
                and all(r.get("status") == "completed" and r.get("native_capture_unchanged")
                        and r.get("source_status") == "native_preanswer_capture"
                        and (r["condition"] not in ("native", "self") or r.get("self_score_check", {}).get("passed")) for r in rows))
    return {"kind": "engineering_only_not_locked_trials", "counterhypothesis": "selector_information_in_TEXT_residual_not_QK_image_route",
            "complete": complete, "p0_passed": p0_passed, "structural_passed": structural_passed,
            "audit_requested": audit, "audit_complete": audit_complete, "n_audit": len(audits),
            "n_result": len(rows), "expected_n_result": len(schedule), "n_family": len(family_ids),
            "n_direction": 2, "n_layer": len(layers), "n_row": len(selected_rows), "n_condition": 6,
            "cells": cells, "failure": failure, "conditioning_filter": "none", "selection": "external_only",
            "confirmation": "not_attempted", "hypothesis_decision": "not_automated",
            "caveats": ["Residual substitution includes selector, task, format and other state; not selector-specific proof.",
                        "Pre-task traversal is a structural causal negative control, not a routing null.",
                        "Earlier-row edits propagate downstream; last-layer earlier-row edits may not reach final logits.",
                        "Aliases are prefix events, not semantic full-answer accuracy; invalid format is not semantic zero.",
                        "Same-image success alone cannot separate selector computation from donor/task copying.",
                        "No stage/weight mediation, held-out confirmation, training or scientific approval."]}


def main(argv=None):
    global _STARTED
    args = _parser().parse_args(argv)
    if args.audit and (24 not in args.layers or "finalassistant" not in args.rows):
        raise ValueError("--audit requires explicit layer24 and finalassistant row.")
    if _STARTED:
        raise RuntimeError("One checkpoint per fresh process; no second dispatch.")
    _STARTED = True
    from filelock import FileLock

    from vlm_pilot.data import load_dataset
    from vlm_pilot.io import atomic_json, identity, now, safe_path
    from vlm_pilot.plan import checkpoint_for, read_plan

    helper, helper_hash = _load_helper()
    classify, alias_hash = helper.load_classifier()
    plan_path, output = safe_path(args.plan), safe_path(args.output)
    if output.exists():
        raise FileExistsError("No overwrite/resume of diagnostic output.")
    plan = read_plan(plan_path, check_source=True)
    checkpoint = checkpoint_for(plan, args.checkpoint)
    families = [f for f in plan.dataset["families"] if f["split"] == "engineering"][:args.families]
    family_ids = [f["family_id"] for f in families]
    if len(family_ids) != args.families or len(set(family_ids)) != args.families:
        raise ValueError("Insufficient/duplicate engineering families; no split fallback.")
    root = safe_path(plan.spec.dataset)
    if identity(load_dataset(root)) != plan.dataset_sha256:
        raise ValueError("Dataset/image bytes differ from pinned plan.")
    inference = plan.spec.inference.model_copy(update={"answer_prefix": ""})
    source = {Path(__file__).name: _source_hash(Path(__file__)),
              "vlm_handoff_diagnostic.py": helper_hash, "vlm_research_followup.py": alias_hash}
    started, rows, structural, p0_passed, failure = time.perf_counter(), [], [], False, None

    def progress(stage, **details):
        elapsed, total = time.perf_counter() - started, args.families * 2 * len(args.rows) * len(args.layers) * 6
        print(json.dumps({"stage": stage, "device": args.device, "completed": len(rows), "total": total,
                          "elapsed_seconds": elapsed, "eta_seconds": elapsed * (total - len(rows)) / len(rows) if rows else None,
                          **details}, allow_nan=False), flush=True)

    lock = Path(tempfile.gettempdir()) / ("interp-vlm-" + args.device.replace(":", "-") + ".lock")
    with FileLock(str(lock), timeout=0):
        output.mkdir(parents=True, exist_ok=False)
        metadata = {"kind": "engineering_only_not_locked_trials", "created_at": now(), "pid": os.getpid(),
                    "plan": plan.model_dump(mode="json"), "plan_file_sha256": _source_hash(plan_path),
                    "diagnostic_source": source, "planned_inference": plan.spec.inference.model_dump(mode="json"),
                    "effective_inference": inference.model_dump(mode="json"), "model": None, "runtime": None,
                    "conditioning": "common_system_native_open_assistant", "answer_prefix": "",
                    "layers": args.layers, "rows": args.rows, "conditions": CONDITIONS, "family_ids": family_ids,
                    "scope": "zero_based_layer_POSTBLOCK_OUTPUT_TEXTROW_only_not_QK",
                    "audit_requested": args.audit, "allow_downloads": False, "source_check": True,
                    "donors_scored_or_generated": False, "selection_filter": "none", "lock": str(lock),
                    "cache_payload_bound_bytes": MAX_CAPTURE_BYTES, "peak_memory_fit_guaranteed": False}
        try:
            from vlm_pilot.runtime import Engine

            progress("cached_model_load_native_idle_guard")
            engine = Engine(checkpoint, inference, device=args.device, allow_downloads=False)
            for layer in args.layers:
                ResidualEngine(engine, layer)
                ResidualSession(engine.language_model, layer, 0)  # Validate; NO hook/device context.
            metadata.update(model=engine.model_profile, runtime=engine.runtime_profile)
            atomic_json(output / "metadata.json", metadata, immutable=True)
            color, _ = helper.paired_cases(families[0], 1, 1, 0)
            probe = dict(color, candidates=color["candidates"] + [c.title() for c in color["candidates"]])
            progress("native_P0_default_0_0")
            native_p0 = engine.preflight(root, families[0], probe)  # Original default no-op site0:0.
            atomic_json(output / "preflight.json", native_p0, immutable=True)
            if native_p0.get("passed") is not True:
                raise RuntimeError("Native default-site P0 failed.")
            for index, family in enumerate(families, 1):
                cache, family_rows, gates = FamilyCache(helper, engine, root, family, args.layers), [], []
                family_failure, bundles = None, {}
                try:
                    bundles = prepare_family(cache, structural, progress)
                    # All row/layer/direction capture/self gates precede this family's trials.
                    for direction, (recipient, _, mapping) in bundles.items():
                        for name in args.rows:
                            for layer in args.layers:
                                progress("residual_self_P0", family=family["family_id"], direction=direction, layer=layer, row_name=name)
                                gate = site_preflight(cache, recipient, layer, mapping[name]["row"], classify)
                                gates.append(dict(gate, direction=direction, row_name=name))
                    del recipient, mapping  # Do not retain GPU Prepareds across family boundaries.
                    if index == 1:
                        p0_passed = True
                    for result in family_results(cache, bundles, args.rows, classify, audit=args.audit and index == 1, progress=progress):
                        family_rows.append(result)
                        rows.append(result)
                        progress("conditions", family=family["family_id"], layer=result["layer"], row_name=result["row_name"], condition=result["condition"])
                except BaseException as error:
                    family_failure = {"type": type(error).__name__, "message": "Family failed; partial rows are incomplete."}
                    raise
                finally:
                    cache.clear()
                    bundles.clear()  # Release full donor/recipient prompts even if publication fails.
                    atomic_json(output / f"family-{index:04d}.json", {"family_id": family["family_id"], "rows": family_rows,
                                "self_preflight": gates, "structural": [s for s in structural if s["family_id"] == family["family_id"]],
                                "failure": family_failure, "status": "failed" if family_failure else "completed"}, immutable=True)
            if any(_source_hash(Path(__file__).resolve().with_name(name)) != sha for name, sha in source.items()):
                raise ValueError("Diagnostic/helper source changed during execution.")
        except BaseException as error:
            failure = {"type": type(error).__name__, "message": "Engineering diagnostic failed; no scientific approval."}
            raise
        finally:
            if not (output / "metadata.json").exists():
                atomic_json(output / "metadata.json", metadata, immutable=True)
            report = summarize(rows, family_ids, args.layers, args.rows, p0_passed=p0_passed,
                               structural=structural, audit=args.audit, failure=failure)
            atomic_json(output / "report.json", report, immutable=True)
        if not report["complete"]:
            raise RuntimeError("Full rows/self/structural/requested audit schedule incomplete.")
        progress("completed_engineering_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())