"""Bounded engineering-only earlier-row Q/K diagnostics, outside locked science.

Import is stdlib-only and inert. Explicit ``main(argv)`` loads ONE cached native
plan checkpoint with the ORIGINAL Inference object, including its answer prefix.
There is no download switch, generation, answer conditioning, padding, retry,
resume, weight edit, custom hook, or dense-site default. Engine.forward delegates
all instrumentation to the existing RoutingSession. Engine.preflight tests the
native final row at the supplied sites; every recipient/earlier-row/single-site
also has a separate capture-only and self-QK candidate-score gate BEFORE trials.

``row_map(engine, prepared)`` tokenizes rendered-prompt prefixes at the actual
traversal/destination word ends. Those IDs MUST be exact prefixes of the original
full IDs. A tokenizer with boundary rewriting/EOS insertion fails closed, rather
than inviting guessed offsets. ``align_rows`` first calls Engine.require_alignment
on full originals with allow_query_change=True, then maps each donor's OWN words.
``FamilyCache.capture`` uses dataclasses.replace ONLY for Prepared.row: full IDs,
embeddings, pixels and prompt are preserved. Scoring always uses the LAST logits,
not logits at the intervention row. Only single-token native candidate suffixes
are supported: multi-token answers fail BEFORE Engine.preflight can extend them.

Two families by default, at most four; both directions at e=1,p=0; three rows;
1..4 externally supplied whole-KV sites; exactly nine conditions. Thus four
families and two sites mean 432 result rows per checkpoint (default: 216).
One-family prepared cache <=30; detached CPU capture payload <=256 MiB including
one transient returned capture. Captures are recomputed per row/site and released.
These are payload bounds, NOT peak RSS/VRAM promises; native eager attention is
still dense internally. Crashes without a final report are not complete runs.

Same-palette single-flip donor gold equals the predicted hybrid: those hits alone
CANNOT distinguish hybrid computation from donor copying. Cancellation, shams,
raw scores and low native accuracy are reported without filtering or automatic
scientific approval. Earlier-row effects include downstream propagation; a late
layer's earlier-row edit can have no path to the last logits. V-swap targets the
donor-palette color at the recipient target; route permutation tests retention,
not an asserted permutation-specific semantic target. This is not confirmation,
stage attribution, full-answer correctness, or a protocol/site-selection runner.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import tempfile
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

__all__ = ["row_map", "align_rows", "parse_sites", "donor_cases", "target_for",
           "FamilyCache", "lowlevel_score", "family_results", "summarize", "main"]

ROWS = ("traversal", "destination", "finalprefix")
CONDITIONS = ("native", "self_qk", "qflip", "kflip", "qkcancel",
              "q_colorsham", "k_colorsham", "routepermute", "v_swap")
MAX_FAMILIES, MAX_PREPARED = 4, 30
MAX_CAPTURE_BYTES = 256 * 1024**2
SELF_ATOL = 1e-4
_TRAVERSAL = re.compile(r"\bfollow one arrow (forward|backward)\b")
_DESTINATION = re.compile(r"\bdestination\b")


def _site_pairs(value):
    message = "Supply 1–4 unique layer:KV-group sites, e.g. 16:0,24:0; no dense default."
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError(message)
    parts = value.split(",")
    if not 1 <= len(parts) <= 4 or any(
            re.fullmatch(r"[0-9]+:[0-9]+", part.strip()) is None for part in parts):
        raise argparse.ArgumentTypeError(message)
    pairs = tuple(tuple(map(int, part.strip().split(":"))) for part in parts)
    if len(set(pairs)) != len(pairs):
        raise argparse.ArgumentTypeError(message)
    return pairs


def parse_sites(value):
    """Return a bounded TUPLE of native whole-KV Sites, lazily, with no defaults."""
    pairs = _site_pairs(value)
    from vlm_pilot.attention import Site

    return tuple(Site(*pair) for pair in pairs)


def _validate_sites(sites):
    if not isinstance(sites, tuple) or not 1 <= len(sites) <= 4:
        raise ValueError("Sites must be an explicit tuple of 1–4 unique sites.")
    pairs = tuple((site.layer, site.kv_group) for site in sites)
    if len(set(pairs)) != len(pairs) or any(
            type(v) is not int or v < 0 for pair in pairs for v in pair):
        raise ValueError("Sites must be an explicit tuple of 1–4 unique sites.")


def _label(site):
    return f"{site.layer}:{site.kv_group}"


def select_families(dataset, count):
    """Only the first requested engineering families, never a split fallback."""
    if type(count) is not int or not 1 <= count <= MAX_FAMILIES:
        raise ValueError("Engineering family count must be 1..4.")
    selected = tuple(f for f in dataset["families"] if f["split"] == "engineering")[:count]
    if len(selected) != count or len({f["family_id"] for f in selected}) != count:
        raise ValueError("Insufficient/duplicate engineering families; no split fallback.")
    return selected


def original_case(family, edge, direction, palette):
    """Fresh copy of an ORIGINAL certified binding member, including its question."""
    from vlm_pilot.data import make_case

    if family["split"] != "engineering":
        raise ValueError("Only engineering families are permitted.")
    canonical = make_case(family, edge=edge, direction=direction, palette=palette)
    matches = [case for case in family["cases"] if case["case_id"] == canonical["case_id"]]
    if len(matches) != 1 or matches[0] != canonical:
        raise ValueError("Missing/changed original binding case; no rewritten questions.")
    return deepcopy(matches[0])


def donor_cases(family, recipient):
    """Same-palette (e,-d,p) Q and (-e,d,p) K; shared (e,d,1-p) color sham/V."""
    e, d, p = recipient["edge"], recipient["direction"], recipient["palette"]
    if (e, p) != (1, 0) or recipient != original_case(family, e, d, p):
        raise ValueError("Recipients must be original binding e=1,p=0, d=+1 or -1.")
    if set(family["palettes"][p]) & set(family["palettes"][1-p]):
        raise ValueError("Color sham palettes must be disjoint.")
    return {"query": original_case(family, e, -d, p),
            "key": original_case(family, -e, d, p),
            "colorsham": original_case(family, e, d, 1-p)}


def target_for(family, recipient, condition):
    """Geometry labels only; no gold/target enters preparation or a forward."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown diagnostic condition.")
    a, e, d, p = (recipient[k] for k in ("anchor", "edge", "direction", "palette"))
    hybrid_node = (a - e*d) % 3
    target_node = hybrid_node if condition in ("qflip", "kflip") else recipient["target"]
    target_palette = 1-p if condition == "v_swap" else p
    return {"gold": recipient["gold"], "gold_node": recipient["target"],
            "target": family["palettes"][target_palette][target_node],
            "target_node": target_node, "target_palette": target_palette,
            "hybrid_target": family["palettes"][p][hybrid_node], "hybrid_node": hybrid_node,
            "target_kind": "donor_palette_recipient_target" if condition == "v_swap" else
            "recipient_palette_flip" if condition in ("qflip", "kflip") else
            "recipient_gold_retention"}


def _unique(pattern, text, description):
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Token alignment failed: unique {description} required.")
    return matches[0]


def row_map(engine, prepared, evidence=None):
    """Map actual full-token prefixes at semantic word ends; mutate failure evidence.

    Requires the ORIGINAL full Prepared (its row is final), not an extended or
    previously row-replaced input. Each map records actual IDs, character end,
    last prefix ID and row, including failed prefix attempts when supplied.
    """
    record = evidence if evidence is not None else {}
    record.update(passed=False, rows={}, case_id=prepared.case["case_id"],
                  sequence_length=len(prepared.input_ids), original_row=prepared.row)
    ids = prepared.input_ids
    if (not isinstance(ids, tuple) or not ids or any(type(i) is not int or i < 0 for i in ids)
            or prepared.row != len(ids)-1 or prepared.embeddings.shape[:2] != (1, len(ids))
            or len(prepared.embeddings.shape) != 3):
        raise ValueError("Token alignment failed: original full IDs/embeddings/final row required.")
    if prepared.case["task"] != "binding" or prepared.case["image_id"] is None:
        raise ValueError("Token alignment failed: original image binding prompt required.")
    if prepared.prompt != engine._prompt(prepared.case["question"], True, "common"):
        raise ValueError("Token alignment failed: prompt changed from ORIGINAL inference/prefix.")
    if engine.settings.answer_prefix and not prepared.prompt.endswith(engine.settings.answer_prefix):
        raise ValueError("Token alignment failed: original answer prefix was lost.")
    actual = tuple(engine.tokenizer.encode(prepared.prompt, add_special_tokens=True))
    record["actual_full_token_ids"] = list(actual)
    if actual != ids:
        raise ValueError("Token alignment failed: actual full token IDs differ.")
    columns = tuple(i for i, token in enumerate(ids) if token == engine.img_token)
    if (len(columns) != 256 or columns != prepared.image_columns
            or columns[-1] > prepared.row):
        raise ValueError("Token alignment failed: actual full image columns differ.")
    question = prepared.case["question"]
    located = _unique(re.compile(re.escape(question)), prepared.prompt, "rendered question")
    traversal = _unique(_TRAVERSAL, question, "follow one arrow (forward|backward)")
    expected = "forward" if prepared.case["direction"] == 1 else "backward"
    if traversal.group(1) != expected:
        raise ValueError("Token alignment failed: traversal word disagrees with direction sign.")
    destination = _unique(_DESTINATION, question, "destination word")
    ends = {"traversal": located.start() + traversal.end(1),
            "destination": located.start() + destination.end(), "finalprefix": len(prepared.prompt)}
    for name in ROWS:
        end = ends[name]
        if not 0 < end <= len(prepared.prompt):
            raise ValueError("Token alignment failed: word end outside rendered prompt.")
        prefix_ids = tuple(engine.tokenizer.encode(prepared.prompt[:end], add_special_tokens=True))
        item = record["rows"][name] = {"character_end": end, "prefix_token_ids": list(prefix_ids),
            "prefix_length": len(prefix_ids), "row": len(prefix_ids)-1,
            "last_prefix_token_id": prefix_ids[-1] if prefix_ids else None, "exact_prefix": False}
        if not prefix_ids or len(prefix_ids) > len(ids) or prefix_ids != ids[:len(prefix_ids)]:
            raise ValueError(f"Token alignment failed at {name}: word-end IDs are not an exact prefix.")
        item["exact_prefix"] = True
        if item["row"] < columns[-1] or not 0 <= item["row"] < len(ids):
            raise ValueError(f"Token alignment failed at {name}: row precedes last image/outside prompt.")
    if len({record["rows"][name]["row"] for name in ROWS}) != len(ROWS):
        raise ValueError("Token alignment failed: semantic rows are not distinct.")
    record.update(passed=True, question_offset=located.start(), image_columns=list(columns))
    return record


def align_rows(engine, recipient, donor, evidence=None):
    """Native FULL alignment FIRST; donor semantic rows are independently mapped."""
    result = evidence if evidence is not None else {}
    result.update(passed=False, recipient_case_id=recipient.case["case_id"],
                  donor_case_id=donor.case["case_id"], allow_query_change=True)
    engine.require_alignment(recipient, donor, allow_query_change=True)
    result["native_full_alignment_passed"] = True
    result["recipient"] = {}
    recipient_map = row_map(engine, recipient, result["recipient"])
    result["donor"] = {}
    donor_map = row_map(engine, donor, result["donor"])
    # Restrict query changes to the canonical traversal word, not arbitrary same-length text.
    rq, dq = recipient.case["question"], donor.case["question"]
    word = _unique(_TRAVERSAL, rq, "recipient traversal")
    replacement = "forward" if donor.case["direction"] == 1 else "backward"
    expected = rq[:word.start(1)] + replacement + rq[word.end(1):]
    if dq != expected:
        raise ValueError("Token alignment failed: donor changed non-traversal question text.")
    if donor.case["direction"] == recipient.case["direction"] and donor.input_ids != recipient.input_ids:
        raise ValueError("Token alignment failed: image-only donor changed original full IDs.")
    for name in ROWS:
        if recipient_map["rows"][name]["row"] != donor_map["rows"][name]["row"]:
            raise ValueError(f"Token alignment failed at {name}: donor OWN word row differs.")
        # A second native check is made on row-only views, without extending inputs.
        engine.require_alignment(replace(recipient, row=recipient_map["rows"][name]["row"]),
                                 replace(donor, row=donor_map["rows"][name]["row"]),
                                 allow_query_change=True)
    result["passed"] = True
    return result


def single_candidate_tokens(engine, prepared):
    """Use native prefix-event suffix IDs, rejecting ALL candidate teacher forcing."""
    suffixes = engine.suffixes(prepared)
    if (not suffixes or list(suffixes) != prepared.case["candidates"]
            or any(len(tokens) != 1 or type(tokens[0]) is not int or tokens[0] < 0
                   for tokens in suffixes.values())):
        raise ValueError("Single-token native candidates required; no teacher-forced answer tails.")
    return suffixes


def check_routing(diagnostic, mode, atol):
    """No missing invariants: native reconstruction and recipient mass/text checks."""
    if (diagnostic.get("mode") != mode or diagnostic.get("captured_before_intervention") is not True
            or diagnostic.get("donor_finite") is not True
            or type(diagnostic.get("attention_row_updated")) is not bool):
        raise ValueError("Missing native-before-intervention routing evidence.")
    names = ("mass_before_after_maxdiff", "returned_attention_mass_maxdiff",
             "reconstruction_weight_maxdiff", "conditional_l1", "row_output_delta_rms",
             "text_weight_maxdiff")
    for name in names:
        value = diagnostic.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("Missing/nonfinite routing invariant.")
    if any(diagnostic[name] > atol for name in names[:3]) or diagnostic["text_weight_maxdiff"] != 0:
        raise ValueError("Native reconstruction or recipient image-mass/text invariant failed.")


def capture_bytes(capture):
    """Count ALL CPU tensor payloads, including hidden, mask and rotary metadata."""
    capture.validate()
    size = 0
    for value in vars(capture).values():
        if hasattr(value, "numel"):
            if value.device.type != "cpu":
                raise ValueError("Every retained native capture tensor must be on CPU.")
            size += value.numel() * value.element_size()
    if size > MAX_CAPTURE_BYTES:
        raise ValueError("Native capture exceeded the 256 MiB CPU payload limit.")
    return size


def _capture_matches(capture, prepared, row, site):
    capture.validate()
    if (capture.row != row or capture.sequence_length != len(prepared.input_ids)
            or capture.image_columns != prepared.image_columns or capture.site != site):
        raise ValueError("Capture changed full sequence/row/image/site alignment.")


class FamilyCache:
    """Single family, <=30 originals; <=256 MiB CPU captures across one row/site."""

    def __init__(self, engine, root, family, sites):
        _validate_sites(sites)
        if family["split"] != "engineering":
            raise ValueError("Family cache is engineering-only.")
        self.engine, self.root, self.family, self.sites = engine, root, family, sites
        self.prepared, self.captures, self.bytes = {}, {}, 0

    def prepare(self, case):
        if case["family_id"] != self.family["family_id"] or case["split"] != "engineering":
            raise ValueError("Prepared family cache cannot span families/splits.")
        key = case["case_id"]
        if key not in self.prepared:
            if len(self.prepared) >= MAX_PREPARED:
                raise ValueError("Prepared family cache exceeded 30 originals.")
            self.prepared[key] = self.engine.prepare(self.root, self.family, case, profile="common")
        elif self.prepared[key].case != case:
            raise ValueError("Case ID reused with different original conditioning/labels.")
        return self.prepared[key]

    def capture(self, prepared, row, site):
        if site not in self.sites or self.prepared.get(prepared.case["case_id"]) is not prepared:
            raise ValueError("Capture requires a cached original and an explicitly selected site.")
        key = prepared.case["case_id"], row, site
        if key not in self.captures:
            if self.captures and any(k[1:] != (row, site) for k in self.captures):
                raise ValueError("Release CPU captures before changing row/site scope.")
            # Only row changes. In particular, IDs/embeddings/pixels are NEVER truncated.
            _, captures, diagnostics = self.engine.forward(replace(prepared, row=row), sites=(site,))
            if set(captures) != {site} or set(diagnostics) != {site}:
                raise ValueError("Missing/excess single-site native capture.")
            check_routing(diagnostics[site], "capture", self.engine.settings.atol)
            capture = captures[site]
            _capture_matches(capture, prepared, row, site)
            size = capture_bytes(capture)
            if self.bytes + size > MAX_CAPTURE_BYTES:
                raise ValueError("Family captures exceeded the 256 MiB CPU payload limit.")
            self.captures[key], self.bytes = capture, self.bytes + size
        return self.captures[key]

    def check_returned(self, capture, base):
        import torch

        if self.bytes + capture_bytes(capture) > MAX_CAPTURE_BYTES:
            raise ValueError("Cache plus transient capture exceeds 256 MiB.")
        # Same-site output captures MUST still be native before the edit.
        if capture.metadata != base.metadata or any(not torch.equal(
                getattr(capture, name), getattr(base, name)) for name in ("q", "k", "v", "mass", "conditional")):
            raise ValueError("Returned capture is not the unchanged native recipient.")

    def clear_captures(self):
        self.captures.clear()
        self.bytes = 0
        gc.collect()  # Detached CPU tensors must not accumulate over sites/rows.

    def clear(self):
        self.clear_captures()
        self.prepared.clear()
        self.engine._features.clear()
        self.engine._family = None
        gc.collect()


def lowlevel_score(engine, prepared, row, *, site=None, action=None):
    """One full native forward; candidate logprobs from LAST logits, no answer tail.

    Returns a JSON score record plus an optional native CPU capture. It deliberately
    does not call Engine.score, which can append teacher-forced candidate tails.
    """
    import torch

    if (prepared.embeddings.shape[1] != len(prepared.input_ids) or type(row) is not int
            or not prepared.image_columns[-1] <= row < len(prepared.input_ids)
            or action is not None and site is None):
        raise ValueError("Low-level score requires full input, causal row and explicit action site.")
    suffixes = single_candidate_tokens(engine, prepared)
    started = time.perf_counter()
    logits, captures, diagnostics = engine.forward(replace(prepared, row=row),
        sites=() if site is None else (site,), actions=None if action is None else {site: action})
    if logits.ndim != 3 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
        raise ValueError("Invalid last-logit native score source.")
    last = logits[0, -1]  # NEVER logits[0,row]; keep=1 retains the original last prompt row.
    logprobs = torch.log_softmax(last.float(), -1)
    scores = {candidate: float(logprobs[tokens[0]].item()) for candidate, tokens in suffixes.items()}
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("Nonfinite single-token candidate scores.")
    maximum = max(scores.values())
    winners = [candidate for candidate, value in scores.items() if value == maximum]
    capture = None
    if site is not None:
        if set(captures) != {site} or set(diagnostics) != {site}:
            raise ValueError("Missing/excess single-site score capture.")
        capture = captures[site]
        _capture_matches(capture, prepared, row, site)
        capture_bytes(capture)
        check_routing(diagnostics[site], "capture" if action is None else action.mode, engine.settings.atol)
    elif captures or diagnostics:
        raise ValueError("Unhooked native score unexpectedly captured a site.")
    return {"scores": scores, "candidate_token_ids": {c: list(t) for c, t in suffixes.items()},
            "candidate_token_logprobs": {c: [v] for c, v in scores.items()},
            "prediction": winners[0] if len(winners) == 1 else None, "tied": len(winners) != 1,
            "last_logit_argmax": int(last.argmax().item()), "intervention_row": row,
            "scored_prompt_row": len(prepared.input_ids)-1, "sequence_length": len(prepared.input_ids),
            "forward_passes": 1, "elapsed_seconds": time.perf_counter()-started,
            "score_kind": "native_single_token_candidate_last_logit_logprob",
            "teacher_forced_answer": False,
            "routing": {_label(s): d for s, d in diagnostics.items()}}, capture


def self_score_check(reference, actual):
    """Absolute-only 1e-4 parity and BOTH candidate/global last-logit argmax."""
    if not reference.get("scores") or set(reference["scores"]) != set(actual.get("scores", {})):
        raise ValueError("Empty/mismatched native/self scores.")
    if reference["candidate_token_ids"] != actual["candidate_token_ids"]:
        raise ValueError("Native/self candidate events changed.")
    if any(not math.isfinite(v) for scores in (reference["scores"], actual["scores"]) for v in scores.values()):
        raise ValueError("Nonfinite native/self scores.")
    maximum = max(abs(reference["scores"][c]-actual["scores"][c]) for c in reference["scores"])
    argmax = all(reference[k] == actual[k] for k in ("prediction", "tied", "last_logit_argmax"))
    if maximum > SELF_ATOL or not argmax:
        raise ValueError("Native/self-QK score parity failed at atol=1e-4, rtol=0.")
    return {"passed": True, "atol": SELF_ATOL, "rtol": 0, "max_abs": maximum, "argmax_match": argmax}


def action_for(condition, captures, seed=1729):
    """Compose joint cancellation from TWO independent same-palette donors."""
    from vlm_pilot.attention import RouteAction

    if condition == "native":
        return None
    if condition == "self_qk":
        return RouteAction("qk", query=captures["self"], key=captures["self"])
    if condition == "qflip":
        return RouteAction("q", query=captures["query"])
    if condition == "kflip":
        return RouteAction("k", key=captures["key"])
    if condition == "qkcancel":
        return RouteAction("qk", query=captures["query"], key=captures["key"])
    if condition == "q_colorsham":
        return RouteAction("q", query=captures["colorsham"])
    if condition == "k_colorsham":
        return RouteAction("k", key=captures["colorsham"])
    if condition == "routepermute":
        return RouteAction("route_permute", seed=seed)
    if condition == "v_swap":
        return RouteAction("v", value=captures["colorsham"])
    raise ValueError("Unknown diagnostic condition.")


def _source_metadata(family, prepared):
    image = next(i for i in family["images"] if i["image_id"] == prepared.case["image_id"])
    return {"case": deepcopy(prepared.case), "image": deepcopy(image),
            "source_prompt_ids": list(prepared.input_ids), "rendered_prompt": prepared.prompt,
            "sequence_length": len(prepared.input_ids), "original_row": prepared.row,
            "image_columns": list(prepared.image_columns), "embedding_shape": list(prepared.embeddings.shape),
            "gold_used_in_conditioning": False, "answer_provided": False,
            "source_forward_kind": "full_original_prompt_native_capture_no_answer_tail"}


def family_results(cache, evidence, progress=None):
    """All directions/rows/sites/conditions, unfiltered; seal gates before yielding.

    ``evidence`` is populated incrementally (including failed alignment attempts),
    and must be saved by the caller even on error. Contains only JSON, not tensors.
    """
    engine, family = cache.engine, cache.family
    evidence.update(passed=False, source_prompts={}, actual_row_maps={}, alignments={}, score_preflights={})
    contexts = []
    # Certify ALL full donors and their own rows before ANY donor capture/trial.
    for direction in (1, -1):
        recipient = cache.prepare(original_case(family, 1, direction, 0))
        sources = {"self": recipient}
        for role, case in donor_cases(family, recipient.case).items():
            sources[role] = cache.prepare(case)
        for source in sources.values():
            evidence["source_prompts"][source.case["case_id"]] = _source_metadata(family, source)
        aligned = evidence["alignments"][str(direction)] = {}
        for role, donor in sources.items():
            aligned[role] = {}
            try:
                align_rows(engine, recipient, donor, aligned[role])
            finally:
                # Also expose partial ACTUAL maps when a prefix/own-row check fails.
                for part, source in (("recipient", recipient), ("donor", donor)):
                    mapped = aligned[role].get(part)
                    if mapped:
                        evidence["actual_row_maps"][source.case["case_id"]] = mapped
        single_candidate_tokens(engine, recipient)
        contexts.append((direction, recipient, sources, aligned))
    # Every direction/row/single-site gets its OWN score gate before family trials.
    for direction, recipient, _, aligned in contexts:
        for row_name in ROWS:
            row = aligned["self"]["recipient"]["rows"][row_name]["row"]
            reference, _ = lowlevel_score(engine, recipient, row)
            for site in cache.sites:
                key = f"{direction}:{row_name}:{_label(site)}"
                gate = evidence["score_preflights"][key] = {"passed": False, "native_unhooked": reference}
                if progress:
                    progress("earlier_row_self_P0", direction=direction, row_name=row_name, site=_label(site))
                try:
                    base = cache.capture(recipient, row, site)
                    for condition in ("native", "self_qk"):
                        action = action_for(condition, {"self": base})
                        result, capture = lowlevel_score(engine, recipient, row, site=site, action=action)
                        gate[condition] = {"score": result}
                        cache.check_returned(capture, base)
                        gate[condition]["parity"] = self_score_check(reference, result)
                        del capture
                    gate["passed"] = True
                finally:
                    # Remove local references as well as the dictionary's references.
                    base = action = capture = None
                    cache.clear_captures()
    evidence["passed"] = True
    for direction, recipient, sources, aligned in contexts:
        for row_name in ROWS:
            row = aligned["self"]["recipient"]["rows"][row_name]["row"]
            for site in cache.sites:
                gate = evidence["score_preflights"][f"{direction}:{row_name}:{_label(site)}"]
                caps = {}
                try:
                    # Each donor uses its OWN mapped row; equality was certified above.
                    for role, source in sources.items():
                        own_row = aligned[role]["donor"]["rows"][row_name]["row"]
                        caps[role] = cache.capture(source, own_row, site)
                    for condition in CONDITIONS:
                        action = action_for(condition, caps)
                        if condition in ("native", "self_qk"):
                            result = gate[condition]["score"]  # Reuse the sealed, already executed gate.
                            parity = gate[condition]["parity"]
                        else:
                            result, capture = lowlevel_score(engine, recipient, row, site=site, action=action)
                            cache.check_returned(capture, caps["self"])
                            del capture
                            parity = None
                        labels = target_for(family, recipient.case, condition)
                        roles = {"self_qk": ("self", "self"), "qflip": ("query",),
                                 "kflip": ("key",), "qkcancel": ("query", "key"),
                                 "q_colorsham": ("colorsham",), "k_colorsham": ("colorsham",),
                                 "v_swap": ("colorsham",)}.get(condition, ())
                        donors = {role: {**evidence["source_prompts"][sources[role].case["case_id"]],
                            "own_row_map": aligned[role]["donor"]["rows"][row_name],
                            "capture": caps[role].metadata, "alignment_passed": True,
                            "source_capture_scored": False, "donor_scored_separately": False,
                            "source_can_also_be_recipient": role in ("self", "query")} for role in set(roles)}
                        prediction = result["prediction"]
                        yield {"family_id": family["family_id"], "split": "engineering",
                            "case_id": recipient.case["case_id"], "edge": 1, "direction": direction, "palette": 0,
                            "site": _label(site), "row_name": row_name, "actual_row": row,
                            "condition": condition, "score": result, **labels, "prediction": prediction,
                            "gold_hit": prediction == labels["gold"], "target_hit": prediction == labels["target"],
                            "hybrid_hit": prediction == labels["hybrid_target"],
                            "donor_answer_copy": bool(donors) and prediction in {v["case"]["gold"] for v in donors.values()},
                            "anchor_copy": prediction == family["palettes"][0][recipient.case["anchor"]],
                            "donors": donors, "native_capture": caps["self"].metadata,
                            "native_capture_unchanged": True, "self_score_check": parity,
                            "all_row_site_gates_passed": True, "source_status": "native_original_full_prompt",
                            "gold_used_in_conditioning": False, "status": "completed"}
                finally:
                    caps.clear()
                    action = capture = None
                    cache.clear_captures()


def summarize(rows, family_ids, sites, *, p0_passed, failure=None):
    """Exact schedule completeness; site/row primary comparisons use ALL cases."""
    _validate_sites(sites)
    if not 1 <= len(family_ids) <= MAX_FAMILIES or len(set(family_ids)) != len(family_ids):
        raise ValueError("Summary requires 1..4 unique expected engineering family IDs.")
    expected = {(f, d, r, _label(s), c) for f in family_ids for d in (1, -1)
                for r in ROWS for s in sites for c in CONDITIONS}
    keys = [(r["family_id"], r["direction"], r["row_name"], r["site"], r["condition"]) for r in rows]
    complete = (p0_passed and failure is None and len(keys) == len(set(keys)) == len(expected)
                and set(keys) == expected and all(r.get("status") == "completed"
                    and r.get("split") == "engineering" and r.get("edge") == 1 and r.get("palette") == 0
                    and r.get("native_capture_unchanged") is True and r.get("all_row_site_gates_passed") is True
                    and r.get("gold_used_in_conditioning") is False
                    and r.get("source_status") == "native_original_full_prompt" for r in rows))
    groups = defaultdict(list)
    for row in rows:
        groups[row["site"], row["row_name"], row["condition"]].append(row)
    cells, primary = [], []
    for site in sites:
        for row_name in ROWS:
            counts = {}
            for condition in CONDITIONS:
                values = groups[_label(site), row_name, condition]
                cell = {"site": _label(site), "row_name": row_name, "condition": condition,
                        "n": len(values), "expected_n": 2*len(family_ids),
                        "n_family": len({v["family_id"] for v in values}),
                        **{name: sum(v[name] for v in values) for name in
                           ("gold_hit", "target_hit", "hybrid_hit", "anchor_copy", "donor_answer_copy")},
                        "conditional_l1_sum": sum(v["score"]["routing"][_label(site)]["conditional_l1"] for v in values)}
                cells.append(cell)
                counts[condition] = cell
            # Comparable flipped-color hits in shams/cancel, NOT mixed gold/target hit subtraction.
            primary.append({"site": _label(site), "row_name": row_name, "unfiltered": True,
                "expected_all_case_n": 2*len(family_ids),
                "all_conditions_n": {c: counts[c]["n"] for c in CONDITIONS},
                "native_gold_hits": counts["native"]["gold_hit"],
                "q_hybrid_hits": counts["qflip"]["hybrid_hit"],
                "k_hybrid_hits": counts["kflip"]["hybrid_hit"],
                "q_sham_hybrid_hits": counts["q_colorsham"]["hybrid_hit"],
                "k_sham_hybrid_hits": counts["k_colorsham"]["hybrid_hit"],
                "cancel_hybrid_hits": counts["qkcancel"]["hybrid_hit"],
                "cancel_gold_hits": counts["qkcancel"]["gold_hit"],
                "q_minus_sham_hybrid_hits": counts["qflip"]["hybrid_hit"]-counts["q_colorsham"]["hybrid_hit"],
                "k_minus_sham_hybrid_hits": counts["kflip"]["hybrid_hit"]-counts["k_colorsham"]["hybrid_hit"],
                "q_minus_cancel_hybrid_hits": counts["qflip"]["hybrid_hit"]-counts["qkcancel"]["hybrid_hit"],
                "k_minus_cancel_hybrid_hits": counts["kflip"]["hybrid_hit"]-counts["qkcancel"]["hybrid_hit"]})
    return {"kind": "engineering_only_earlier_routing_not_locked_trials", "complete": complete,
            "p0_passed": p0_passed, "failure": failure, "n_result": len(rows),
            "expected_n_result": len(expected), "missing": len(expected-set(keys)),
            "unexpected": len(set(keys)-expected), "duplicates": len(keys)-len(set(keys)),
            "family_ids": list(family_ids), "n_direction": 2, "n_row": 3, "n_site": len(sites), "n_condition": 9,
            "conditioning_filter": "none", "site_selection": "external_only", "cells": cells,
            "primary_site_and_row_all_in_call": primary, "hypothesis_decision": "not_automated",
            "confirmation": "not_attempted", "cautions": [
                "Low native binding floor: report all cases, not baseline-correct survivors.",
                "Same-palette Q/K single-flip donor answers equal hybrid labels: hits do not exclude copying.",
                "Cancellation uses independent Q and K donors; compare shams, cancellation and native scores jointly.",
                "Two directions/three rows/nine conditions are repeated measures, not independent families.",
                "Single-token candidate events are not free-generation/full-answer correctness.",
                "Earlier-row edits propagate downstream; late-layer earlier rows can have no last-logit path.",
                "V-swap uses donor-palette target; route permutation retention is not a mechanistic target prediction.",
                "Engineering only; no automatic site choice, stage attribution, confirmation or go.",
                "256 MiB capture payload and 30 prepared limits do not guarantee native eager peak-memory fit."]}


def _device(value):
    if re.fullmatch(r"cpu|cuda:[0-9]+", value) is None:
        raise argparse.ArgumentTypeError("Choose cpu or explicit cuda:N; no device fallback.")
    return value


def _site_string(value):
    _site_pairs(value)
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--families", type=int, choices=range(1, MAX_FAMILIES+1), default=2)
    parser.add_argument("--sites", required=True, type=_site_string)
    parser.add_argument("--device", default="cuda:0", type=_device)
    return parser


def device_lock_path(device):
    _device(device)
    return Path(tempfile.gettempdir()) / ("interp-vlm-" + device.replace(":", "-") + ".lock")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    """Explicit cached-only launch. Every partial/failure is sealed, never resumed."""
    args = _parser().parse_args(argv)
    from filelock import FileLock

    from vlm_pilot.data import load_dataset
    from vlm_pilot.io import atomic_json, identity, now, safe_path
    from vlm_pilot.plan import checkpoint_for, read_plan, source_digest

    sites = parse_sites(args.sites)
    plan_path, output = safe_path(args.plan), safe_path(args.output)
    if output.exists():
        raise FileExistsError("Diagnostic output exists; no overwrite/resume.")
    plan = read_plan(plan_path, check_source=True)
    checkpoint = checkpoint_for(plan, args.checkpoint)
    families = select_families(plan.dataset, args.families)
    root = safe_path(plan.spec.dataset)
    if identity(load_dataset(root)) != plan.dataset_sha256:
        raise ValueError("Dataset/image bytes differ from the original locked plan.")
    # EXACT original object; do not model_copy, replace prefix, dtype or system text.
    inference = plan.spec.inference
    started, rows, p0_passed, stage = time.perf_counter(), [], False, "initialization"
    metadata = {"created_at": now(), "kind": "engineering_only_earlier_routing_not_locked_trials",
        "plan": plan.model_dump(mode="json"), "plan_file_sha256": _sha(plan_path),
        "diagnostic_source_sha256": _sha(Path(__file__)), "locked_source_sha256": plan.source_sha256,
        "effective_inference": inference.model_dump(mode="json"), "original_inference_unchanged": True,
        "conditioning": "original_common_system_native_template_original_answer_prefix_no_answer",
        "checkpoint": args.checkpoint, "device": args.device, "sites": [_label(s) for s in sites],
        "rows": list(ROWS), "conditions": list(CONDITIONS), "family_ids": [f["family_id"] for f in families],
        "factors": {"edge": 1, "directions": [1, -1], "palette": 0}, "allow_downloads": False,
        "prepared_limit": MAX_PREPARED, "capture_limit_bytes": MAX_CAPTURE_BYTES,
        "model": None, "runtime": None, "family_evidence": {}, "complete": False}

    def progress(current, **details):
        nonlocal stage
        stage = current
        elapsed, done, total = time.perf_counter()-started, len(rows), args.families*2*3*len(sites)*9
        print(json.dumps({"stage": current, "device": args.device, "completed": done, "total": total,
            "elapsed_seconds": elapsed, "eta_seconds": elapsed*(total-done)/done if done else None,
            **details}, allow_nan=False), flush=True)

    # SAME native writer lock, timeout=0: no occupancy bypass or implicit lock retry.
    with FileLock(str(device_lock_path(args.device)), timeout=0):
        output.mkdir(parents=True, exist_ok=False)
        try:
            from vlm_pilot.runtime import Engine

            progress("cached_model_load")
            engine = Engine(checkpoint, inference, device=args.device, allow_downloads=False)
            metadata.update(model=engine.model_profile, runtime=engine.runtime_profile)
            cache = FamilyCache(engine, root, families[0], sites)
            try:
                prepared = cache.prepare(original_case(families[0], 1, 1, 0))
                # Reject multi-token suffixes BEFORE native P0's optional candidate-tail checks.
                single_candidate_tokens(engine, prepared)
                progress("native_final_row_P0", selected_sites=metadata["sites"])
                preflight = engine.preflight(root, families[0], prepared.case, sites=sites)
                atomic_json(output / "preflight.json", preflight, immutable=True)
                if preflight.get("passed") is not True:
                    raise RuntimeError("Native selected-final-site preflight failed; no trials.")
                p0_passed = True
            finally:
                prepared = None
                cache.clear()
            for index, family in enumerate(families, 1):
                cache, family_rows, family_failure = FamilyCache(engine, root, family, sites), [], None
                evidence = metadata["family_evidence"][family["family_id"]] = {}
                try:
                    progress("full_token_alignment", family=family["family_id"])
                    for row in family_results(cache, evidence, progress):
                        family_rows.append(row)
                        rows.append(row)
                        progress("conditions", family=family["family_id"], direction=row["direction"],
                                 row_name=row["row_name"], actual_row=row["actual_row"],
                                 site=row["site"], condition=row["condition"])
                except BaseException as error:
                    family_failure = {"type": type(error).__name__, "stage": stage,
                                      "message": "Family failed closed; alignment/P0/partial evidence retained."}
                    raise
                finally:
                    cache.clear()
                    atomic_json(output / f"family-{index:04d}.json", {"family_id": family["family_id"],
                        "rows": family_rows, "evidence": evidence, "failure": family_failure,
                        "status": "failed" if family_failure else "completed"}, immutable=True)
            if (_sha(Path(__file__)) != metadata["diagnostic_source_sha256"]
                    or _sha(plan_path) != metadata["plan_file_sha256"]
                    or source_digest() != plan.source_sha256):
                raise RuntimeError("Diagnostic/plan/package source changed during execution.")
            report = summarize(rows, metadata["family_ids"], sites, p0_passed=p0_passed)
            if not report["complete"]:
                raise RuntimeError("Partial schedule cannot earn completion.")
            metadata["complete"] = True
            atomic_json(output / "metadata.json", metadata, immutable=True)
            atomic_json(output / "report.json", report, immutable=True)
            progress("completed_engineering_only", output=str(output))
            return 0
        except BaseException as error:
            failure = {"type": type(error).__name__, "stage": stage,
                       "message": "Diagnostic failed closed; inspect saved alignment and score-gate evidence."}
            metadata.update(complete=False, failure=failure)
            if not (output / "metadata.json").exists():
                atomic_json(output / "metadata.json", metadata, immutable=True)
            if not (output / "report.json").exists():
                atomic_json(output / "report.json", summarize(rows, metadata["family_ids"], sites,
                    p0_passed=p0_passed, failure=failure), immutable=True)
            raise


if __name__ == "__main__":
    raise SystemExit(main())