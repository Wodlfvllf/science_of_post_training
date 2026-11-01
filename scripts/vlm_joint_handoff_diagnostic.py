"""Engineering-only SIMULTANEOUS task-Q handoff at one externally chosen layer.

Import and CLI parsing use only the standard library and the adjacent stdlib
handoff helper, loaded by its exact file path with before/after source hashes.
Model, tensor, classifier and locking dependencies are lazy. Supply --layer as
one zero-based integer: 24 is an external proposal, NOT a default or a search.
All native KV groups (1..8) at that layer are selected in EVERY hooked forward.
There are seven conditions, e=1, p=0 and both directions on 1..4 engineering
families, without source-answer scoring, correctness filtering or donor text.

FamilyCache subclasses the existing bounded helper. One native forward captures
ALL groups per prompt, not eight forwards. It keeps at most ten Prepared objects
and 512 MiB of CPU-owned native captures, including every optional tensor.
Native Engine.preflight and simultaneous capture/self-Q candidate-score P0 must
pass before trials. Each direction repeats unhooked/capture/self score parity.
Only recipient lowercase colors and Title aliases enter candidate scoring;
pooling is delegated to the existing classifier, with ties remaining None.

The first family's e=1,d=+1 recipient alone also gets seven unrestricted greedy
audits, at most 32 tokens each. Every step recomputes Engine.forward with the
SAME all-site actions and fixed original pre-answer row; only embeddings grow.
This is patched generation, not Engine.generate's unhooked path. Raw shape-word
detection is reported separately from EOS-terminated bare-lowercase correctness.

Per-group projected row RMS and route diagnostics are retained, NEVER summed to
claim a net projected delta: the engine does not expose its accumulated vector.
Joint intervention is NOT proof of synergy. No training, weight intervention,
locked trials, confirmation, selection, downloads, overwrite or resume. Run one
checkpoint per fresh process. Runtime uses the native per-device FileLock and
Engine's idle-GPU guard; no CUDA mem_get_info is called ahead of that guard.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path


def _load_helper():
    """Resolve the adjacent file only; do not import a same-named installed module."""
    import hashlib

    path = Path(__file__).resolve().with_name("vlm_handoff_diagnostic.py")
    payload = path.read_bytes()
    if not payload.strip():
        raise ValueError("Empty handoff helper cannot establish provenance.")
    checksum = hashlib.sha256(payload).hexdigest()
    spec = importlib.util.spec_from_file_location("_vlm_joint_handoff_helper", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the exact-path handoff helper.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError("Handoff helper changed during exact-path import.")
    return module, checksum


helper, HELPER_SHA256 = _load_helper()
CONDITIONS = helper.CONDITIONS + ("route_permute",)
MAX_PREPARED, MAX_CAPTURE_BYTES = helper.MAX_PREPARED, helper.MAX_CAPTURE_BYTES
AUDIT_BUDGET, SEED = 32, 1729
GRADES = ("lower", "title", "combined")
paired_cases = helper.paired_cases
donor_cases = helper.donor_cases
require_alignment = helper.require_alignment


def _layer(value):
    """Reject lists/ranges/floats/negative layers before optional imports."""
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("Supply one nonnegative zero-based integer layer.")
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Supply one bounded integer layer.") from None


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--families", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--layer", required=True, type=_layer)
    parser.add_argument("--device", type=helper._device, default="cuda:0")
    return parser


def _site_labels(sites):
    return tuple(f"{s.layer}:{s.kv_group}" for s in sites)


def _validate_sites(sites):
    """A contiguous tuple of all groups at ONE layer, never independent sites.

    Architecture completeness is established by all_sites(); this dependency-free
    check additionally rejects reordered, duplicated, sparse or cross-layer keys.
    """
    if not isinstance(sites, tuple) or not 1 <= len(sites) <= 8:
        raise ValueError("Joint routing requires a tuple of 1..8 all-layer KV groups.")
    if any(type(getattr(s, "layer", None)) is not int or s.layer < 0
           or type(getattr(s, "kv_group", None)) is not int for s in sites):
        raise ValueError("Joint sites require nonnegative native integer coordinates.")
    if (len(set(sites)) != len(sites) or len({s.layer for s in sites}) != 1
            or tuple(s.kv_group for s in sites) != tuple(range(len(sites)))):
        raise ValueError("Joint sites must include every KV group once at one layer.")


def all_sites(engine, layer):
    """Read actual native architecture; reject >8 groups or out-of-range layers."""
    config = engine.language_model.config
    count, layers = config.num_key_value_heads, config.num_hidden_layers
    if (type(layer) is not int or type(count) is not int or not 1 <= count <= 8
            or type(layers) is not int or not 0 <= layer < layers
            or len(engine.language_model.model.layers) != layers):
        raise ValueError("Selected layer/native KV-group count is outside the joint bound.")
    from vlm_pilot.attention import Site

    sites = tuple(Site(layer, group) for group in range(count))
    _validate_sites(sites)
    return sites


def build_actions(condition, captures, sites, seed=SEED):
    """Build ONE simultaneous action dictionary; keep each query Capture by identity.

    captures is a flat Site->native Capture map for the already selected donor
    condition (recipient for native/permutation). Exact keys and capture.site
    must agree. No tensors are copied, concatenated or silently reassigned.
    Native returns an empty dictionary but the caller still passes ALL sites.
    """
    _validate_sites(sites)
    if condition not in CONDITIONS:
        raise ValueError("Unknown joint handoff condition.")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Permutation seed must be a nonnegative signed-64-bit integer.")
    if not isinstance(captures, Mapping) or set(captures) != set(sites):
        raise ValueError("Capture keys must match ALL selected sites exactly.")
    if any(getattr(captures[s], "site", None) != s for s in sites):
        raise ValueError("Capture site does not match its selected dictionary key.")
    if condition == "native":
        return {}
    from vlm_pilot.attention import RouteAction

    if condition == "route_permute":
        return {s: RouteAction("route_permute", seed=seed) for s in sites}
    return {s: RouteAction("q", query=captures[s]) for s in sites}


def check_all_routing(diagnostics, sites, mode, atol, *, string_keys=False):
    """Certify every group, including finite/mass/text/native-before-edit evidence."""
    expected = _site_labels(sites) if string_keys else sites
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(expected):
        raise ValueError("Missing/extra all-group routing diagnostics.")
    for key in expected:
        helper.check_routing(diagnostics[key], mode, atol)
        if type(diagnostics[key].get("attention_row_updated")) is not bool:
            raise ValueError("Missing all-group attention-row update evidence.")


def _capture_size(capture, prepared, site):
    """Validate native coordinates and count all tensor bytes, not just Q/K/V."""
    capture.validate()
    if (capture.site != site or capture.row != prepared.row
            or capture.sequence_length != prepared.embeddings.shape[1]
            or capture.image_columns != prepared.image_columns):
        raise ValueError("Capture is not the selected prompt's native row/site.")
    size = 0
    for value in vars(capture).values():
        if hasattr(value, "numel"):
            if value.device.type != "cpu" or not value.is_contiguous():
                raise ValueError("All capture tensors must own contiguous CPU storage.")
            size += value.numel() * value.element_size()
    return size


class FamilyCache(helper.FamilyCache):
    """Bounded helper cache, with atomic ALL-group population from one forward."""

    def __init__(self, engine, root, family, sites):
        _validate_sites(sites)
        super().__init__(engine, root, family, sites)
        self.capture_diagnostics = {}

    def capture(self, prepared, site):
        if site not in self.sites:
            raise ValueError("Unselected site cannot enter the joint family cache.")
        case_id = prepared.case["case_id"]
        if self.prepared.get(case_id) is not prepared:
            raise ValueError("Capture must use the exact family-cached Prepared object.")
        keys = tuple((case_id, s) for s in self.sites)
        present = [key in self.captures for key in keys]
        if any(present) and not all(present):
            raise ValueError("Partial all-group cache cannot be repaired by another forward.")
        if not all(present):
            # Engine captures every same-layer group BEFORE any row intervention.
            _, captures, diagnostics = self.engine.forward(prepared, sites=self.sites)
            check_all_routing(diagnostics, self.sites, "capture", self.engine.settings.atol)
            if set(captures) != set(self.sites):
                raise ValueError("Native forward omitted/added joint group captures.")
            size = sum(_capture_size(captures[s], prepared, s) for s in self.sites)
            if self.bytes + size > MAX_CAPTURE_BYTES:
                raise ValueError("Native joint family capture cache exceeded 512 MiB.")
            # Publish only after EVERY group's diagnostics, layout and bound pass.
            self.captures.update({(case_id, s): captures[s] for s in self.sites})
            self.capture_diagnostics[case_id] = diagnostics
            self.bytes += size
        return self.captures[(case_id, site)]

    def capture_all(self, prepared):
        self.capture(prepared, self.sites[0])
        return {s: self.captures[(prepared.case["case_id"], s)] for s in self.sites}

    def clear(self):
        super().clear()
        self.capture_diagnostics.clear()


def _unchanged_native(base, captures, prepared, sites):
    """Same-layer joint edits must not alter ANY returned pre-edit native capture."""
    import torch

    if set(base) != set(sites) or set(captures) != set(sites):
        raise ValueError("Native comparison requires all selected group captures.")
    for site in sites:
        left, right = base[site], captures[site]
        _capture_size(right, prepared, site)
        if left.metadata != right.metadata or set(vars(left)) != set(vars(right)):
            raise ValueError("Joint score returned changed native capture metadata.")
        for name, value in vars(left).items():
            if hasattr(value, "numel") and not torch.equal(value, getattr(right, name)):
                raise ValueError("Joint score returned a changed native recipient tensor.")
    return True


def _check_score(result, prepared, sites, condition, atol):
    """Reject missing/nonfinite scores and changed recipient conditioning/events."""
    scores = result.get("scores", {})
    if (not scores or set(scores) != set(prepared.case["candidates"])
            or set(result.get("candidate_token_ids", {})) != set(scores)
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in scores.values())):
        raise ValueError("Missing/nonfinite/mismatched recipient color candidate scores.")
    maximum = max(scores.values())
    winners = [name for name, value in scores.items() if value == maximum]
    prediction = winners[0] if len(winners) == 1 else None
    if result.get("prediction") != prediction or result.get("tied") is not (len(winners) != 1):
        raise ValueError("Raw prediction must preserve exact ties as None.")
    if (result.get("input_ids") != list(prepared.input_ids)
            or result.get("rendered_prompt") != prepared.prompt
            or result.get("row") != prepared.row
            or result.get("image_columns") != list(prepared.image_columns)):
        raise ValueError("Joint score changed the shared recipient inputs or causal row.")
    mode = "capture" if condition == "native" else "route_permute" if condition == "route_permute" else "q"
    check_all_routing(result.get("routing"), sites, mode, atol,
                      string_keys=True)


def _joint_score(cache, recipient, condition, sources, base):
    actions = build_actions(condition, sources, cache.sites, SEED)
    # ONE score with ALL actions: never a loop of individual-site scores.
    result, captures = cache.engine.score(recipient, sites=cache.sites, actions=actions)
    _check_score(result, recipient, cache.sites, condition, cache.engine.settings.atol)
    _unchanged_native(base, captures, recipient, cache.sites)
    return result


def candidate_preflight(cache, probe, classify, report):
    """Final simultaneous native/capture/self-Q score P0, sealed even on failure."""
    recipient = cache.prepare(probe)
    report.update(passed=False, sites=list(_site_labels(cache.sites)), joint=True,
                  atol=helper.SELF_ATOL, rtol=0, conditions={})
    report["alignment"] = require_alignment(cache.engine, recipient, recipient)
    reference, _ = cache.engine.score(recipient)  # Explicitly UNHOOKED reference.
    report["native_unhooked_score"] = reference
    candidates = [c for c in probe["candidates"] if c == c.lower()]
    base = cache.capture_all(recipient)
    for condition in ("native", "self_q"):
        result = _joint_score(cache, recipient, condition, base, base)
        check = report["conditions"][condition] = {"score": result}
        check["parity"] = helper.self_score_check(reference, result, classify, candidates)
    report["passed"] = True


def counterfactuals(family, color, donor=None):
    """Recipient color at donor-selected node, NOT default-donor shape-as-gold."""
    selected = donor.case["target"] if donor is not None else None
    return {"target_gold": color["gold"],
            "anchor_color": family["palettes"][color["palette"]][color["anchor"]],
            "donor_selected_node": selected,
            "recipient_color_at_donor_selected_node": (
                family["palettes"][color["palette"]][selected] if selected is not None else None),
            "donor_payload_color": (
                family["palettes"][donor.case["palette"]][selected] if selected is not None else None),
            "donor_answer_metadata_only": donor.case["gold"] if donor is not None else None}


def _outcomes(grading, targets, condition):
    def matches(prediction, value):
        return value is not None and prediction == value

    return {g: {"hit": matches(grading[g], targets["target_gold"]),
                "anchor_copy": matches(grading[g], targets["anchor_color"]),
                "donor_answer_copy": matches(grading[g], targets["donor_answer_metadata_only"]),
                "donor_payload_color_copy": matches(grading[g], targets["donor_payload_color"]),
                "recipient_color_at_donor_selected_node_hit": matches(
                    grading[g], targets["recipient_color_at_donor_selected_node"]),
                "wrong_role_counterfactual_hit": condition == "wrong_anchor_shape_q" and matches(
                    grading[g], targets["recipient_color_at_donor_selected_node"]),
                "disjoint_donor_copy": condition in ("shape_q_disjoint", "color_sham",
                                                     "wrong_anchor_shape_q") and matches(
                    grading[g], targets["donor_payload_color"])} for g in GRADES}


def audit_text(text, raw_text, tokens, eos, candidates, gold, shapes):
    """Strict lower-case bare delivery and descriptive raw shape detection only."""
    answer = text.strip()
    ended = bool(tokens) and tokens[-1] == eos
    valid = ended and answer in candidates and answer == answer.lower()
    mentioned = [shape for shape in shapes
                 if re_search_word(shape, text)]
    return {"text": text, "raw_text": raw_text, "token_ids": list(tokens), "eos": ended,
            "finish_reason": "eos" if ended else "length", "valid_bare_lower": valid,
            "parsed_answer": answer if valid else None, "exact_correct": valid and answer == gold,
            "raw_shape_answer_detected": answer.lower() in shapes,
            "raw_shape_answer": answer.lower() if answer.lower() in shapes else None,
            "raw_shape_mentions": mentioned,
            "raw_shape_detection_is_correctness": False}


def re_search_word(word, text):
    import re

    return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.IGNORECASE) is not None


def patched_greedy_audit(engine, prepared, sites, actions, shapes, *, progress=None):
    """Full-vocabulary greedy decoding, fixed pre-answer row, same joint patch.

    Appended tokens are generated recipient tokens, never donor/gold tokens.
    Engine.forward performs its native append-only donor alignment adaptation.
    No model forward method/global hook is replaced and no net delta is inferred.
    """
    import torch

    _validate_sites(sites)
    if set(actions) not in (set(), set(sites)):
        raise ValueError("Greedy audit requires the same complete joint actions.")
    if len(prepared.input_ids) + AUDIT_BUDGET > engine.settings.max_context_tokens:
        raise ValueError("32-token audit would exceed the locked context; no truncation.")
    eos = engine.tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise ValueError("A single native tokenizer EOS ID is required.")
    current, tokens, steps = prepared, [], []
    started = time.perf_counter()
    mode = next(iter(actions.values())).mode if actions else "capture"
    for index in range(AUDIT_BUDGET):
        if current.row != prepared.row or current.input_ids != prepared.input_ids:
            raise ValueError("Greedy audit moved the pre-answer intervention row or IDs.")
        logits, captures, diagnostics = engine.forward(current, sites=sites, actions=actions)
        check_all_routing(diagnostics, sites, mode, engine.settings.atol)
        if set(captures) != set(sites):
            raise ValueError("Greedy audit omitted joint native captures.")
        for site in sites:
            _capture_size(captures[site], current, site)
        token = int(logits[0, -1].argmax().item())
        tokens.append(token)
        steps.append({"step": index + 1, "token_id": token, "row": current.row,
                      "sequence_length": current.embeddings.shape[1],
                      "routing": {label: diagnostics[s] for label, s in zip(_site_labels(sites), sites, strict=True)}})
        if progress is not None:
            progress("patched_greedy_audit", generated=len(tokens), token_budget=AUDIT_BUDGET)
        del logits, captures, diagnostics
        if token == eos or index + 1 == AUDIT_BUDGET:
            break
        with torch.no_grad():
            embedding = engine.language_model.get_input_embeddings()(
                torch.tensor([[token]], device=engine.device))
            current = replace(current, embeddings=torch.cat((current.embeddings, embedding), 1))
    result = audit_text(engine.tokenizer.decode(tokens, skip_special_tokens=True),
                        engine.tokenizer.decode(tokens, skip_special_tokens=False), tokens, eos,
                        [c for c in prepared.case["candidates"] if c == c.lower()],
                        prepared.case["gold"], shapes)
    result.update(status="completed", steps=steps, elapsed_seconds=time.perf_counter() - started,
                  settings={"max_new_tokens": AUDIT_BUDGET, "do_sample": False, "use_cache": False,
                            "eos_token_id": eos, "full_vocabulary": True, "joint_patched": True,
                            "sites": list(_site_labels(sites)), "fixed_row": prepared.row,
                            "net_projected_row_delta_available": False})
    return result


def family_results(cache, classify, *, audit=False, progress=None):
    """Yield 2*7 joint rows; score recipients only, capture every donor natively."""
    family, engine, sites = cache.family, cache.engine, cache.sites
    for direction in (1, -1):
        color, _ = paired_cases(family, 1, direction, 0)
        candidates = color["candidates"]
        probe = dict(color, candidates=candidates + [c.title() for c in candidates])
        recipient = cache.prepare(probe)
        donors = donor_cases(family, color)
        donors["self_q"] = probe  # Exact alias-scored object/ID, not another prompt.
        aligned = {}
        for name, case in donors.items():
            donor = cache.prepare(case)
            aligned[name] = donor, require_alignment(engine, recipient, donor)
        if progress is not None:
            progress("direction_native_self_gate", family=family["family_id"], direction=direction)
        reference, _ = engine.score(recipient)
        base = cache.capture_all(recipient)
        gated = {}
        # Finish BOTH local parity checks before any audit or donor intervention.
        # Reuse these scores below instead of doing two extra all-group forwards.
        for condition in ("native", "self_q"):
            result = _joint_score(cache, recipient, condition, base, base)
            gated[condition] = result, helper.self_score_check(reference, result, classify, candidates)
        for condition in CONDITIONS:
            if progress is not None:
                progress("joint_condition_start", family=family["family_id"],
                         direction=direction, condition=condition)
            donor, donor_meta, sources = None, None, base
            if condition in aligned:
                donor, alignment = aligned[condition]
                sources = cache.capture_all(donor)
                donor_meta = helper._donor_metadata(family, donor, alignment)
                donor_meta["captures"] = {label: sources[s].metadata
                                          for label, s in zip(_site_labels(sites), sites, strict=True)}
            if condition in gated:
                result, parity = gated[condition]
            else:
                result, parity = _joint_score(cache, recipient, condition, sources, base), None
            if result["candidate_token_ids"] != reference["candidate_token_ids"]:
                raise ValueError("Condition changed the shared recipient candidate events.")
            grading = classify(result, candidates)
            targets = counterfactuals(family, color, donor)
            generation = None
            if audit and direction == 1:
                actions = build_actions(condition, sources, sites, SEED)
                callback = (None if progress is None else
                            lambda stage, direction=direction, condition=condition, **values: progress(stage, family=family["family_id"],
                                                            direction=direction, condition=condition, **values))
                generation = patched_greedy_audit(engine, recipient, sites, actions,
                    [n["shape"] for n in family["nodes"]], progress=callback)
            yield {"family_id": family["family_id"], "direction": direction, "edge": 1,
                   "palette": 0, "layer": sites[0].layer, "sites": list(_site_labels(sites)),
                   "joint": True, "condition": condition, "case": color, "score": result,
                   "native_unhooked_score": reference, "grading": grading,
                   "target_gold": color["gold"], "counterfactuals": targets,
                   "outcomes": _outcomes(grading, targets, condition), "donor": donor_meta,
                   "native_captures": {label: base[s].metadata
                                       for label, s in zip(_site_labels(sites), sites, strict=True)},
                   "native_capture_unchanged": True, "self_score_check": parity,
                   "generation_audit": generation,
                   "changed_groups": [label for label in _site_labels(sites)
                                      if result["routing"][label]["conditional_l1"] > 0],
                   "intervention_scope": "all_KV_groups_one_layer_recipient_K_V_image_mass_text_kept",
                   "net_projected_row_delta_available": False, "joint_is_proof_of_synergy": False,
                   "source_status": "native_preanswer_capture", "status": "completed"}


def summarize(rows, expected_families, sites, *, p0_passed, checkpoint=None, failure=None):
    """No site multiplier: default complete schedule is 4*2*7=56 joint rows."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["direction"], row["condition"])].append(row)
    cells = []
    outcome_names = ("hit", "anchor_copy", "donor_answer_copy", "donor_payload_color_copy",
                     "recipient_color_at_donor_selected_node_hit", "wrong_role_counterfactual_hit",
                     "disjoint_donor_copy")
    for direction in (1, -1):
        for condition in CONDITIONS:
            values = groups[(direction, condition)]
            for grade in GRADES:
                cells.append({"direction": direction, "condition": condition, "grading": grade,
                              "n_family": len({r["family_id"] for r in values}),
                              "n_result": len(values), "expected_n_family": expected_families,
                              **{name: sum(r["outcomes"][grade][name] for r in values)
                                 for name in outcome_names}})
    expected = expected_families * 2 * len(CONDITIONS)
    family_ids = list(dict.fromkeys(r["family_id"] for r in rows))
    keys = {(r["family_id"], r["direction"], r["condition"]) for r in rows}
    schedule = {(f, d, c) for f in family_ids for d in (1, -1) for c in CONDITIONS}
    labels = list(_site_labels(sites))
    audits = [r for r in rows if r.get("generation_audit") is not None]
    audit_complete = (len(audits) == len(CONDITIONS)
                      and {r["condition"] for r in audits} == set(CONDITIONS)
                      and all(r["family_id"] == family_ids[0] and r["direction"] == 1
                              and r["generation_audit"].get("status") == "completed"
                              and r["generation_audit"]["settings"]["max_new_tokens"] == AUDIT_BUDGET
                              and r["generation_audit"]["settings"]["joint_patched"] is True
                              for r in audits))
    complete = (1 <= expected_families <= 4 and bool(sites) and p0_passed and failure is None
                and len(family_ids) == expected_families and len(keys) == len(rows) == expected
                and keys == schedule and audit_complete
                and all(r.get("status") == "completed" and r.get("joint") is True
                        and r.get("native_capture_unchanged") is True
                        and r.get("source_status") == "native_preanswer_capture"
                        and r.get("sites") == labels and r.get("edge") == 1 and r.get("palette") == 0
                        and set(r["score"]["routing"]) == set(labels) for r in rows))
    return {"kind": "engineering_only_not_locked_trials", "mode": "joint_query",
            "checkpoint": checkpoint, "complete": complete, "failure": failure,
            "p0_passed": p0_passed, "n_family": len(family_ids), "expected_n_family": expected_families,
            "n_result": len(rows), "expected_n_result": expected, "n_direction": 2,
            "n_condition": len(CONDITIONS), "sites": labels, "n_group": len(sites),
            "n_generation_audit": len(audits), "expected_n_generation_audit": 7,
            "generation_audit_complete": audit_complete, "cells": cells,
            "n_hit_combined": sum(r["outcomes"]["combined"]["hit"] for r in rows),
            "net_projected_row_delta_available": False, "joint_is_proof_of_synergy": False,
            "site_selection": "external_one_layer_all_native_groups_no_search",
            "conditioning_filter": "none", "hypothesis_decision": "not_automated",
            "confirmation": "not_attempted", "training": "not_attempted",
            "limitations": ["Joint intervention is not a synergy/necessity/weight-mediation proof",
                            "Per-group projected RMS values cannot establish a net projected delta",
                            "Color/Title pooling is a prefix-event diagnostic, not answer correctness",
                            "Shape donors are capture-only; default donor shape is not hybrid target gold",
                            "Wrong-role target is recipient color at donor-selected node",
                            "Unrestricted patched greedy audit covers first family, d=+1 only, 32 tokens",
                            "Native captures precede edits; no post-edit donor tensors are substituted",
                            "No confirmation, site search, auto-go, training or weight interventions",
                            "Bounded caches and memory-fit estimates do not guarantee peak fit"]}


def _failure(error, stage):
    """Never serialize exception text, request chains, credentials or traceback."""
    return {"type": type(error).__name__, "stage": stage,
            "message": "Joint diagnostic failed; partial sources are not complete or confirmed."}


def main(argv=None):
    args = _parser().parse_args(argv)
    from filelock import FileLock

    from vlm_pilot.io import atomic_json, identity, now, safe_path

    output = safe_path(args.output)
    if output.exists():
        raise FileExistsError("Joint diagnostic output exists; no overwrite or resume.")
    lock = Path(tempfile.gettempdir()) / ("interp-vlm-" + args.device.replace(":", "-") + ".lock")
    started, rows, sites, p0_passed, stage = time.perf_counter(), [], (), False, "initialization"

    def progress(value, **details):
        nonlocal stage
        stage = value
        elapsed, done, total = time.perf_counter() - started, len(rows), args.families * 2 * 7
        print(json.dumps({"stage": stage, "device": args.device, "completed": done, "total": total,
                          "elapsed_seconds": elapsed,
                          "eta_seconds": elapsed * (total - done) / done if done else None,
                          **details}, allow_nan=False), flush=True)

    # Same native pilot lock; default is /tmp/interp-vlm-cuda-0.lock.
    with FileLock(str(lock), timeout=0):
        output.mkdir(parents=True, exist_ok=False)
        metadata = {"kind": "engineering_only_not_locked_trials", "mode": "joint_query",
                    "created_at": now(), "checkpoint": args.checkpoint, "requested_layer": args.layer,
                    "requested_families": args.families, "device": args.device,
                    "plan": None, "effective_inference": None, "model": None, "runtime": None,
                    "conditions": list(CONDITIONS), "selection_filter": "none", "allow_downloads": False,
                    "source_check": True, "joint_is_proof_of_synergy": False,
                    "net_projected_row_delta_available": False}
        try:
            from vlm_pilot.data import load_dataset
            from vlm_pilot.plan import checkpoint_for, read_plan

            progress("source_and_plan_check")
            plan_path = safe_path(args.plan)
            plan_hash = helper._source_hash(plan_path)
            plan = read_plan(plan_path, check_source=True)
            if helper._source_hash(plan_path) != plan_hash:
                raise ValueError("Plan changed while loading diagnostic provenance.")
            checkpoint = checkpoint_for(plan, args.checkpoint)
            families = [f for f in plan.dataset["families"] if f["split"] == "engineering"][:args.families]
            if len(families) != args.families or len({f["family_id"] for f in families}) != args.families:
                raise ValueError("Insufficient unique engineering families; no split fallback.")
            root = safe_path(Path(plan.spec.dataset))
            if identity(load_dataset(root)) != plan.dataset_sha256:
                raise ValueError("Dataset/image bytes differ from the locked plan.")
            if helper._source_hash(Path(helper.__file__)) != HELPER_SHA256:
                raise ValueError("Exact-path handoff helper changed since import.")
            classify, alias_hash = helper.load_classifier()
            inference = plan.spec.inference.model_copy(update={"answer_prefix": ""})
            metadata.update(plan=plan.model_dump(mode="json"), plan_file_sha256=plan_hash,
                diagnostic_source={Path(__file__).name: helper._source_hash(Path(__file__)),
                                   "vlm_handoff_diagnostic.py": HELPER_SHA256,
                                   "vlm_research_followup.py": alias_hash},
                checkpoint_definition=checkpoint.model_dump(mode="json"),
                effective_inference=inference.model_dump(mode="json"),
                conditioning="common_system_native_open_assistant_no_response_no_answer_prefix",
                family_ids=[f["family_id"] for f in families],
                factors={"edge": 1, "directions": [1, -1], "palette": 0},
                expected_n_result=args.families * 2 * 7,
                greedy_audit={"family_id": families[0]["family_id"], "edge": 1, "direction": 1,
                              "conditions": list(CONDITIONS), "max_new_tokens": AUDIT_BUDGET},
                cache_bounds={"prepared_per_family": MAX_PREPARED, "groups_per_prompt": 8,
                              "capture_bytes_per_family": MAX_CAPTURE_BYTES})
            progress("fit_check")
            metadata["memory_fit"] = helper.memory_fit(checkpoint, inference, args.device)
            from vlm_pilot.runtime import Engine

            progress("cached_model_load")
            engine = Engine(checkpoint, inference, device=args.device, allow_downloads=False)
            metadata.update(model=engine.model_profile, runtime=engine.runtime_profile)
            sites = all_sites(engine, args.layer)
            metadata.update(sites=list(_site_labels(sites)), n_group=len(sites))
            atomic_json(output / "metadata.json", metadata, immutable=True)
            color, _ = paired_cases(families[0], 1, 1, 0)
            probe = dict(color, candidates=color["candidates"] + [c.title() for c in color["candidates"]])
            progress("native_all_group_P0", selected_sites=metadata["sites"])
            p0 = engine.preflight(root, families[0], probe, sites=sites)
            atomic_json(output / "preflight.json", p0, immutable=True)
            check_all_routing(p0.get("routing"), sites, "capture", engine.settings.atol,
                              string_keys=True)
            if p0.get("passed") is not True:
                raise RuntimeError("Native all-group P0 failed; no condition trials.")
            score_p0, cache = {"passed": False}, FamilyCache(engine, root, families[0], sites)
            progress("joint_self_score_P0")
            try:
                candidate_preflight(cache, probe, classify, score_p0)
            finally:
                atomic_json(output / "score-preflight.json", score_p0, immutable=True)
                cache.clear()
            p0_passed = True
            for index, family in enumerate(families, 1):
                cache, family_rows, failure = FamilyCache(engine, root, family, sites), [], None
                try:
                    progress("family_start", family=family["family_id"])
                    for row in family_results(cache, classify, audit=index == 1, progress=progress):
                        family_rows.append(row)
                        rows.append(row)
                        progress("joint_conditions", family=family["family_id"],
                                 direction=row["direction"], condition=row["condition"])
                except BaseException as error:
                    failure = _failure(error, stage)
                    raise
                finally:
                    atomic_json(output / f"family-{index:04d}.json",
                        {"family_id": family["family_id"], "checkpoint": args.checkpoint,
                         "rows": family_rows, "failure": failure,
                         "status": "failed" if failure else "completed"}, immutable=True)
                    atomic_json(output / f"report-families-{index:04d}.json",
                        summarize(rows, args.families, sites, p0_passed=p0_passed,
                                  checkpoint=args.checkpoint, failure=failure), immutable=True)
                    cache.clear()
            report = summarize(rows, args.families, sites, p0_passed=p0_passed, checkpoint=args.checkpoint)
            if not report["complete"]:
                raise RuntimeError("Nonempty full joint/audit schedule was not completed.")
            atomic_json(output / "report.json", report, immutable=True)
            progress("completed_engineering_only")
            return 0
        except BaseException as error:
            failure = _failure(error, stage)
            if not (output / "metadata.json").exists():
                metadata["failure"] = failure
                atomic_json(output / "metadata.json", metadata, immutable=True)
            if not (output / "report.json").exists():
                atomic_json(output / "report.json", summarize(rows, args.families, sites,
                    p0_passed=p0_passed, checkpoint=args.checkpoint, failure=failure), immutable=True)
            progress("failed_sealed", failure=failure)
            return 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        # Lock/path/dependency failures before artifact creation stay sanitized too.
        print("Joint diagnostic could not start; inspect environment, paths and device lock.", flush=True)
        code = 1
    raise SystemExit(code)