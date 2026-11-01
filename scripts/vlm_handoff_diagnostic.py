"""Ad hoc, engineering-only pre-answer task-Q interchange; no locked trials.

Importing this file uses only the standard library. ``paired_cases`` returns
fresh color/shape cases; ``parse_sites`` lazily constructs native whole-KV-group
Sites. The CLI uses cached checkpoints only and an open assistant turn (no
response or Answer: prefix), with the plan's common system message. This runtime
conditioning difference is recorded, not written back to the plan.

Each explicitly supplied site is tested independently on e=1, p=0, BOTH d=+1
and d=-1, without correctness filtering. Shape donors are never scored, generated
or teacher-forced: their gold/candidates are metadata, not conditioning. Only
recipient color candidates and their Title aliases enter scoring. Alias pooling
is a prefix-event diagnostic, not a new scientific endpoint or site selection.

Native Engine.preflight internally checks self-QK; this script additionally
checks capture-only and self-Q color scores against unhooked native scores at
absolute tolerance 1e-4, with unchanged raw and alias argmax. Engine/RouteAction
enforce native layout, causal/RoPE alignment and recipient K/V/image-mass rules.
Captures are native BEFORE edits; they are not post-intervention route tensors.
Engine.score may append recipient candidate tails at the unchanged causal row;
its append-only action adaptation is not padding a mismatched donor prompt.
RAM/VRAM checks are explicit estimates, NOT a guarantee of peak-memory fit.
Host available-RAM discovery requires Linux MemAvailable or SC_AVPHYS_PAGES;
unsupported hosts fail closed. No fallback, padding, resuming, overwrite,
validation/confirmation selection, downloads, weights editing or automatic go.
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
import time
from collections import defaultdict
from pathlib import Path

CONDITIONS = ("native", "self_q", "shape_q", "shape_q_disjoint", "color_sham",
              "wrong_anchor_shape_q")
MAX_PREPARED, MAX_CAPTURE_BYTES = 10, 512 * 1024**2
SELF_ATOL = 1e-4


def _paired_cases(family, edge, direction, palette, anchor=None):
    from vlm_pilot.data import SHAPES, make_case

    base = make_case(family, edge=edge, direction=direction, palette=palette, anchor=anchor)
    name = family["nodes"][base["anchor"]]["shape"]
    traversal = "forward" if direction == 1 else "backward"
    prefix = f"Starting at the {name}, follow one arrow {traversal}. "
    color = dict(base, case_id=base["case_id"] + "-handoff-color", answer_kind="color",
                 candidates=list(base["candidates"]),
                 question=prefix + "What is the destination color? Reply with one color name.")
    # Target/labels come from canonical geometry, never an edge-readout surrogate.
    shape = dict(base, case_id=base["case_id"] + "-handoff-shape", answer_kind="shape",
                 gold=family["nodes"][base["target"]]["shape"], candidates=list(SHAPES),
                 question=prefix + "What is the destination shape? Reply with one shape name.")
    return color, shape


def paired_cases(family, edge, direction, palette):
    """Return (color, shape) with original binding image/anchor/target/factors.

    The only question differences are the two task words. Shape gold is the
    destination node's shape even for an unknown-token shape; it is never added
    to a prompt or tokenizer vocabulary. Returned mutable lists are independent.
    """
    return _paired_cases(family, edge, direction, palette)


def donor_cases(family, color):
    """Build fixed controls; the wrong anchor is the recipient destination node."""
    e, d, p = color["edge"], color["direction"], color["palette"]
    _, shape = paired_cases(family, e, d, p)
    other_color, other_shape = paired_cases(family, e, d, 1-p)
    _, wrong_shape = _paired_cases(family, e, d, 1-p, anchor=color["target"])
    return {"self_q": color, "shape_q": shape, "shape_q_disjoint": other_shape,
            "color_sham": other_color, "wrong_anchor_shape_q": wrong_shape}


def _site_pairs(value):
    """Validate a bounded explicit string before importing optional model code."""
    message = "Supply 1–8 unique zero-based layer:KV-group sites, e.g. 8:0,16:1."
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError(message)
    parts = value.split(",")
    if len(parts) > 8 or any(re.fullmatch(r"[0-9]+:[0-9]+", p.strip()) is None for p in parts):
        raise argparse.ArgumentTypeError(message)
    pairs = tuple(tuple(map(int, p.strip().split(":"))) for p in parts)
    if len(set(pairs)) != len(pairs):
        raise argparse.ArgumentTypeError(message)
    return pairs


def parse_sites(value):
    """Lazily return native Sites, never a dense sweep or a guessed default."""
    pairs = _site_pairs(value)
    from vlm_pilot.attention import Site

    return tuple(Site(*pair) for pair in pairs)


def _site_string(value):
    _site_pairs(value)
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--families", type=int, choices=range(1, 9), default=4)
    # Keep parsing dependency-free; dispatch converts the already validated string.
    parser.add_argument("--sites", required=True, type=_site_string)
    parser.add_argument("--mode", choices=("query",), default="query")
    parser.add_argument("--device", default="cuda:0", type=_device)
    parser.add_argument("--include-route-permute", action="store_true")
    return parser


def _device(value):
    if re.fullmatch(r"cpu|cuda:[0-9]+", value) is None:
        raise argparse.ArgumentTypeError("Choose cpu or explicit cuda:N; no device fallback.")
    return value


def _source_hash(path):
    payload = path.read_bytes()
    if not payload.strip():
        raise ValueError("Empty diagnostic/scoring source cannot establish provenance.")
    return hashlib.sha256(payload).hexdigest()


def load_classifier():
    """Use the existing alias classifier by exact file, without executing its CLI."""
    path = Path(__file__).with_name("vlm_research_followup.py")
    before = _source_hash(path)
    spec = importlib.util.spec_from_file_location("_vlm_handoff_alias_classifier", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the existing response-alias classifier.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if _source_hash(path) != before or not callable(getattr(module, "classify_scores", None)):
        raise ValueError("Changed/failed classifier source cannot establish provenance.")
    return module.classify_scores, before


def _available_host_bytes():
    """Use available RAM, not total installed RAM or a guessed hardware label."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        match = re.search(r"^MemAvailable:\s+([0-9]+) kB$", meminfo.read_text(), re.MULTILINE)
        if match and int(match[1]) > 0:
            return int(match[1]) * 1024
    try:
        pages, size = os.sysconf("SC_AVPHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and size > 0:
            return pages * size
    except (AttributeError, OSError, ValueError):
        pass
    raise RuntimeError("Available host RAM cannot be verified on this host; fit check failed closed.")


def memory_fit(checkpoint, inference, device):
    """Reject insufficient CPU/GPU estimates BEFORE Engine loads any weights.

    Native parameter counts are architecture-specific, not a model-name guess.
    The budget includes one bounded family cache and dense eager row-hook clones.
    CUDA occupancy is checked by the SAME runtime guard Engine uses, before
    creating a CUDA context to query free VRAM. Engine repeats its own guard.
    """
    host = _available_host_bytes()
    parameters = {"2b": 2_348_347_392, "8b": 8_528_318_464, "fixture": 1_000_000}
    weight_bytes = parameters[checkpoint.family] * (4 if inference.dtype == "float32" else 2)
    workspace = (256 * 1024**2 if checkpoint.backend == "fixture" else
                 2 * 1024**3 + 12 * 32 * inference.max_context_tokens**2)
    # Reserve two weight copies on the host even when low_cpu_mem_usage is enabled.
    cpu_required = 2 * weight_bytes + MAX_CAPTURE_BYTES + workspace
    result = {"device": device, "host_available_bytes": host,
              "host_required_estimate_bytes": cpu_required, "weight_bytes": weight_bytes,
              "family_capture_budget_bytes": MAX_CAPTURE_BYTES,
              "workspace_estimate_bytes": workspace, "peak_fit_guaranteed": False}
    if host < cpu_required:
        raise RuntimeError("Insufficient available host RAM for the explicit diagnostic estimate.")
    if device.startswith("cuda:"):
        import shutil
        import subprocess

        import torch

        from vlm_pilot.runtime import verify_idle_gpu

        if checkpoint.backend == "fixture" or not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable or fixture is CPU-only; no fallback.")
        selected = torch.device(device)
        result["gpu_uuid"] = verify_idle_gpu(selected)
        # CUDA mem_get_info creates a context. In a PID namespace nvidia-smi may
        # report its host PID, which the later guard correctly treats as another
        # process. Query the exact GPU through nvidia-smi before Engine creates
        # its context; do not weaken the occupancy guard or guess a PID alias.
        memory = subprocess.run([shutil.which("nvidia-smi"), "-i", result["gpu_uuid"],
                                 "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=15)
        match = re.fullmatch(r"\s*([0-9]+)\s*,\s*([0-9]+)\s*", memory.stdout)
        if match is None:
            raise RuntimeError("GPU memory query returned an unrecognized response")
        free, total = (int(v) * 1024**2 for v in match.groups())
        required = weight_bytes + workspace
        result.update(gpu_free_bytes=free, gpu_total_bytes=total,
                      gpu_required_estimate_bytes=required)
        if free < required:
            raise RuntimeError("Insufficient free VRAM for the explicit diagnostic estimate.")
    return result


def require_alignment(engine, recipient, donor):
    """Certify FULL native IDs and per-word lengths, with no compensating padding.

    Engine's allow_query_change check certifies length/row/image columns. Extra
    checks restrict changed IDs to individually length-preserving task/anchor
    edits in the full rendered prompt. Isolated edits are re-tokenized in context,
    so equal total length cannot conceal unequal task-word token lengths.
    """
    engine.require_alignment(recipient, donor, allow_query_change=True)
    for prepared in (recipient, donor):
        if (prepared.row != len(prepared.input_ids)-1
                or prepared.embeddings.shape[1] != len(prepared.input_ids)
                or len(prepared.image_columns) != 256
                or tuple(engine.tokenizer.encode(prepared.prompt, add_special_tokens=True))
                != prepared.input_ids):
            raise ValueError("Not a full, unextended native pre-answer prompt with one image tile.")
        if engine.settings.answer_prefix or prepared.prompt != engine._prompt(
                prepared.case["question"], True, "common"):
            raise ValueError("A response/prefix was supplied; donors must be actual pre-answer captures.")
    rq, dq = recipient.case["question"], donor.case["question"]
    if recipient.prompt.count(rq) != 1:
        raise ValueError("Ambiguous recipient question in native template.")
    before, after = recipient.prompt.split(rq)
    expected = rq
    edits = []
    if donor.case["answer_kind"] == "shape":
        edits.extend((m.start(), m.end(), "shape") for m in re.finditer(r"\bcolor\b", rq))
        if len(edits) != 2:
            raise ValueError("Expected exactly two aligned task words.")
    if donor.case["anchor"] != recipient.case["anchor"]:
        start = len("Starting at the ")
        end = rq.index(",", start)
        other_start = len("Starting at the ")
        edits.append((start, end, dq[other_start:dq.index(",", other_start)]))
    allowed, descriptions = set(), []
    for start, end, replacement in sorted(edits, reverse=True):
        single = before + rq[:start] + replacement + rq[end:] + after
        ids = tuple(engine.tokenizer.encode(single, add_special_tokens=True))
        if len(ids) != len(recipient.input_ids):
            raise ValueError("Task/anchor word token lengths differ; no padding or truncation allowed.")
        changed = {i for i, (a, b) in enumerate(zip(recipient.input_ids, ids, strict=True)) if a != b}
        allowed.update(changed)
        descriptions.append({"old": rq[start:end], "new": replacement,
                             "changed_token_positions": sorted(changed)})
        expected = expected[:start] + replacement + expected[end:]
    if dq != expected or donor.prompt != before + dq + after:
        raise ValueError("Donor changed text outside the specified task/anchor words.")
    actual = {i for i, (a, b) in enumerate(zip(recipient.input_ids, donor.input_ids, strict=True))
              if a != b}
    if not actual <= allowed or actual.intersection(recipient.image_columns):
        raise ValueError("Full donor token IDs changed outside allowed task/anchor positions.")
    return {"passed": True, "allow_query_change": True, "full_input_ids_checked": True,
            "sequence_length": len(recipient.input_ids), "row": recipient.row,
            "image_columns": list(recipient.image_columns), "edits": descriptions,
            "changed_token_positions": sorted(actual)}


class FamilyCache:
    """At most ten prepared prompts and eight sites per prompt in ONE family."""

    def __init__(self, engine, root, family, sites):
        if not isinstance(sites, tuple) or not 1 <= len(sites) <= 8 or len(set(sites)) != len(sites):
            raise ValueError("Family cache requires 1–8 unique externally supplied sites.")
        self.engine, self.root, self.family, self.sites = engine, root, family, sites
        self.prepared, self.captures, self.bytes = {}, {}, 0

    def prepare(self, case):
        key = case["case_id"]
        if key not in self.prepared:
            if len(self.prepared) >= MAX_PREPARED:
                raise ValueError("Prepared family cache exceeded its fixed bound.")
            self.prepared[key] = self.engine.prepare(self.root, self.family, case, profile="common")
        elif self.prepared[key].case != case:
            raise ValueError("A cached case ID was reused with different scoring/conditioning data.")
        return self.prepared[key]

    def capture(self, prepared, site):
        if site not in self.sites:
            raise ValueError("Unselected site cannot enter the family capture cache.")
        key = prepared.case["case_id"], site
        if key not in self.captures:
            _, captures, diagnostics = self.engine.forward(prepared, sites=(site,))
            capture = captures[site]
            # Engine.forward keys diagnostics by native Site; Engine.score uses strings.
            check_routing(diagnostics[site], "capture", self.engine.settings.atol)
            capture.validate()
            # Count every stored tensor, including hidden inputs and rotary metadata.
            size = sum(v.numel()*v.element_size() for v in vars(capture).values()
                       if hasattr(v, "numel"))
            if self.bytes + size > MAX_CAPTURE_BYTES:
                raise ValueError("Native family capture cache exceeded 512 MiB.")
            self.bytes += size
            self.captures[key] = capture
        return self.captures[key]

    def clear(self):
        self.prepared.clear()
        self.captures.clear()
        self.bytes = 0
        # Engine itself otherwise retains the last family's image features.
        self.engine._features.clear()
        self.engine._family = None


def check_routing(diagnostic, mode, atol):
    """Missing/invalid native invariants fail, rather than masquerading as a source."""
    if (diagnostic.get("mode") != mode
            or diagnostic.get("captured_before_intervention") is not True
            or diagnostic.get("donor_finite") is not True):
        raise ValueError("Missing native-before-edit or finite routing evidence.")
    for name in ("mass_before_after_maxdiff", "returned_attention_mass_maxdiff",
                 "reconstruction_weight_maxdiff", "conditional_l1", "row_output_delta_rms",
                 "text_weight_maxdiff"):
        value = diagnostic.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("Missing/nonfinite routing invariant.")
    if (diagnostic["mass_before_after_maxdiff"] > atol
            or diagnostic["returned_attention_mass_maxdiff"] > atol
            or diagnostic["reconstruction_weight_maxdiff"] > atol
            or diagnostic["text_weight_maxdiff"] != 0):
        raise ValueError("Recipient image-mass/text or native reconstruction invariant failed.")


def self_score_check(reference, actual, classify, candidates):
    """Absolute-only P0 score parity and argmax; never relax the locked P0 gate."""
    if not reference.get("scores") or set(reference["scores"]) != set(actual.get("scores", {})):
        raise ValueError("Empty/mismatched source scores cannot pass self parity.")
    if reference["candidate_token_ids"] != actual["candidate_token_ids"]:
        raise ValueError("Self/native candidate events changed.")
    differences = [abs(reference["scores"][k]-actual["scores"][k]) for k in reference["scores"]]
    if any(not math.isfinite(v) for v in (*reference["scores"].values(), *actual["scores"].values())):
        raise ValueError("Nonfinite self/native scores.")
    left, right = classify(reference, candidates), classify(actual, candidates)
    argmax = reference["prediction"] == actual["prediction"] and all(
        left[g] == right[g] for g in ("lower", "title", "combined"))
    check = {"atol": SELF_ATOL, "rtol": 0, "max_abs": max(differences), "argmax_match": argmax}
    if check["max_abs"] > SELF_ATOL or not argmax:
        raise ValueError("Native/capture/self-Q color-score P0 parity failed.")
    return dict(check, passed=True)


def _donor_metadata(family, prepared, alignment):
    case = prepared.case
    image = next(i for i in family["images"] if i["image_id"] == case["image_id"])
    return {"case": case, "image": image, "input_ids": list(prepared.input_ids),
            "rendered_prompt": prepared.prompt, "row": prepared.row,
            "image_columns": list(prepared.image_columns), "alignment": alignment,
            "gold_used_in_conditioning": False, "response_provided": False,
            "donor_scored": False, "source_forward_kind": "capture_only_no_candidate_tail",
            "source_status": "native_preanswer_capture"}


def candidate_preflight(cache, probe, classify, report):
    """Populate a separately sealed native/capture/self-Q SCORE P0 before trials.

    ``report`` retains raw scores even when a parity/invariant check raises. The
    caller publishes it in a finally block and only then permits family trials.
    """
    import torch

    from vlm_pilot.attention import RouteAction

    recipient = cache.prepare(probe)
    report.update(passed=False, sites={}, atol=SELF_ATOL, rtol=0)
    report["alignment"] = require_alignment(cache.engine, recipient, recipient)
    reference, _ = cache.engine.score(recipient)
    report["native_unhooked_score"] = reference
    candidates = [c for c in probe["candidates"] if c == c.lower()]
    for site in cache.sites:
        label = f"{site.layer}:{site.kv_group}"
        base = cache.capture(recipient, site)
        checks = report["sites"][label] = {}
        for condition, action in (("native", None), ("self_q", RouteAction("q", query=base))):
            result, captures = cache.engine.score(recipient, sites=(site,),
                actions={site: action} if action is not None else None)
            checks[condition] = {"score": result}
            native = captures[site]
            native.validate()
            if native.metadata != base.metadata or not all(
                    torch.equal(getattr(base, name), getattr(native, name))
                    for name in ("q", "k", "v", "mass", "conditional")):
                raise ValueError("Score P0 returned a non-native recipient capture.")
            check_routing(result["routing"][label], "capture" if action is None else "q",
                          cache.engine.settings.atol)
            checks[condition]["parity"] = self_score_check(reference, result, classify, candidates)
    report["passed"] = True


def family_results(cache, conditions, classify):
    """Yield all unfiltered direction/site/condition rows; only score COLOR."""
    import torch

    from vlm_pilot.attention import RouteAction

    family, engine = cache.family, cache.engine
    for direction in (1, -1):
        color, _ = paired_cases(family, 1, direction, 0)
        candidates = color["candidates"]
        probe = dict(color, candidates=candidates + [c.title() for c in candidates])
        recipient = cache.prepare(probe)
        donors = donor_cases(family, color)
        # self_q must reuse precisely the alias-scored recipient, not a duplicate ID.
        donors["self_q"] = probe
        aligned = {}
        for name, case in donors.items():
            prepared = cache.prepare(case)
            aligned[name] = prepared, require_alignment(engine, recipient, prepared)
        reference, _ = engine.score(recipient)
        for site in cache.sites:
            base = cache.capture(recipient, site)
            for condition in conditions:
                donor, donor_meta, action = None, None, None
                if condition in aligned:
                    donor, alignment = aligned[condition]
                    source = cache.capture(donor, site)
                    donor_meta = _donor_metadata(family, donor, alignment)
                    donor_meta["capture"] = source.metadata
                    action = RouteAction("q", query=source)
                elif condition == "route_permute":
                    action = RouteAction("route_permute", seed=1729)
                result, captures = engine.score(recipient, sites=(site,),
                    actions={site: action} if action is not None else None)
                label = f"{site.layer}:{site.kv_group}"
                routing = result["routing"][label]
                check_routing(routing, "capture" if action is None else action.mode, engine.settings.atol)
                # A single site's intervention cannot change its own native capture.
                native = captures[site]
                native.validate()
                unchanged = all(torch.equal(getattr(base, n), getattr(native, n))
                                for n in ("q", "k", "v", "mass", "conditional"))
                if not unchanged or native.metadata != base.metadata:
                    raise ValueError("Returned capture is not the unchanged native recipient state.")
                if not result["scores"] or any(not math.isfinite(v) for v in result["scores"].values()):
                    raise ValueError("Empty/nonfinite color candidate source scores.")
                grading = classify(result, candidates)
                parity = self_score_check(reference, result, classify, candidates) if condition in (
                    "native", "self_q") else None
                anchor_color = family["palettes"][0][color["anchor"]]
                donor_payload = (family["palettes"][donor.case["palette"]][donor.case["target"]]
                                 if donor is not None else None)
                wrong_color = (family["palettes"][0][donor.case["target"]]
                               if condition == "wrong_anchor_shape_q" else None)
                outcomes = {g: {"hit": grading[g] == color["gold"],
                               "anchor_copy": grading[g] == anchor_color,
                               "donor_answer_copy": donor is not None and grading[g] == donor.case["gold"],
                               "donor_payload_color_copy": donor is not None and grading[g] == donor_payload,
                               "wrong_anchor_recipient_color_hit": wrong_color is not None and grading[g] == wrong_color}
                            for g in ("lower", "title", "combined")}
                yield {"family_id": family["family_id"], "direction": direction, "edge": 1,
                       "palette": 0, "site": label, "condition": condition, "case": color,
                       "score": result, "native_unhooked_score": reference,
                       "grading": grading, "outcomes": outcomes, "donor": donor_meta,
                       "native_capture": base.metadata, "native_capture_unchanged": unchanged,
                       "self_score_check": parity, "anchor_color": anchor_color,
                       "donor_payload_color": donor_payload, "wrong_anchor_recipient_color": wrong_color,
                       "intervention_scope": "recipient_K_V_mass_text_kept_conditional_image_route_only",
                       "changed_route": routing["conditional_l1"] > 0,
                       "attention_row_updated": routing["attention_row_updated"],
                       "source_status": "native_preanswer_capture", "status": "completed"}


def summarize(rows, expected_families, sites, conditions, *, p0_passed, failure=None):
    """Operational counts only. Empty/failed/partial sources never earn completion."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["site"], row["direction"], row["condition"])].append(row)
    cells = []
    for site in sites:
        label = f"{site.layer}:{site.kv_group}"
        for direction in (1, -1):
            for condition in conditions:
                values = groups[(label, direction, condition)]
                for grade in ("lower", "title", "combined"):
                    cells.append({"site": label, "direction": direction, "condition": condition,
                                  "grading": grade, "n_family": len({v["family_id"] for v in values}),
                                  "n_result": len(values), "expected_n_family": expected_families,
                                  **{name: sum(v["outcomes"][grade][name] for v in values) for name in (
                                      "hit", "anchor_copy", "donor_answer_copy", "donor_payload_color_copy",
                                      "wrong_anchor_recipient_color_hit")},
                                  "changed_route": sum(v["changed_route"] for v in values),
                                  "attention_row_updated": sum(v["attention_row_updated"] for v in values),
                                  "conditional_l1_sum": sum(v["score"]["routing"][label]["conditional_l1"] for v in values)})
    expected = expected_families * 2 * len(sites) * len(conditions)
    keys = {(r["family_id"], r["site"], r["direction"], r["condition"]) for r in rows}
    complete = (expected > 0 and p0_passed and failure is None and len(keys) == len(rows) == expected
                and all(c["n_family"] == expected_families for c in cells)
                and all(r.get("status") == "completed" and r.get("native_capture_unchanged")
                        and r.get("source_status") == "native_preanswer_capture" for r in rows))
    return {"kind": "engineering_only_not_locked_trials", "mode": "query", "complete": complete,
            "p0_passed": p0_passed, "failure": failure, "n_family": len({r["family_id"] for r in rows}),
            "n_recipient_case": len({(r["family_id"], r["direction"]) for r in rows}),
            "expected_n_recipient_case": expected_families * 2,
            "expected_n_family": expected_families, "n_direction": 2, "n_site": len(sites),
            "n_condition": len(conditions), "n_result": len(rows), "expected_n_result": expected,
            "n_hit_combined": sum(r["outcomes"]["combined"]["hit"] for r in rows), "cells": cells,
            "conditioning_filter": "none", "site_selection": "external_only",
            "hypothesis_decision": "not_automated", "confirmation": "not_attempted",
            "limitations": ["Development-only; no locked results or regrading",
                            "Color/Title aliases are distinct prefix events, not full-answer correctness",
                            "Shape donors are unscored; shape answer-copy is outside color candidates",
                            "Native captures are before edits; route change uses low-level diagnostics",
                            "No stage comparison, selection, validation or automatic go",
                            "RAM/VRAM estimates and P0 do not guarantee long-run fit or scientific validity"]}


def main(argv=None):
    args = _parser().parse_args(argv)
    from filelock import FileLock

    from vlm_pilot.data import load_dataset
    from vlm_pilot.io import atomic_json, identity, now, safe_path
    from vlm_pilot.plan import checkpoint_for, read_plan

    sites = parse_sites(args.sites)
    plan_path, output = safe_path(args.plan), safe_path(args.output)
    if output.exists():
        raise FileExistsError("Diagnostic output already exists; no overwrite or resume.")
    plan = read_plan(plan_path, check_source=True)
    checkpoint = checkpoint_for(plan, args.checkpoint)
    families = [f for f in plan.dataset["families"] if f["split"] == "engineering"][:args.families]
    if not families or len(families) != args.families:
        raise ValueError("Insufficient engineering source families; no split fallback.")
    root = safe_path(plan.spec.dataset)
    if identity(load_dataset(root)) != plan.dataset_sha256:
        raise ValueError("Dataset/image bytes differ from the locked plan.")
    classify, alias_hash = load_classifier()
    source = {Path(__file__).name: _source_hash(Path(__file__)),
              "vlm_research_followup.py": alias_hash}
    inference = plan.spec.inference.model_copy(update={"answer_prefix": ""})
    conditions = CONDITIONS + (("route_permute",) if args.include_route_permute else ())
    lock = Path(tempfile.gettempdir()) / ("interp-vlm-" + args.device.replace(":", "-") + ".lock")
    started, rows, p0_passed, metadata = time.perf_counter(), [], False, {}

    def progress(stage, **details):
        elapsed = time.perf_counter()-started
        done, total = len(rows), args.families * 2 * len(sites) * len(conditions)
        print(json.dumps({"stage": stage, "device": args.device, "completed": done,
                          "total": total, "elapsed_seconds": elapsed,
                          "eta_seconds": elapsed*(total-done)/done if done else None,
                          **details}, allow_nan=False), flush=True)

    # This is the exact same per-device lock path the native pilot writer uses.
    with FileLock(str(lock), timeout=0):
        output.mkdir(parents=True, exist_ok=False)
        metadata = {"kind": "engineering_only_not_locked_trials", "mode": args.mode,
                    "created_at": now(), "plan": plan.model_dump(mode="json"),
                    "plan_file_sha256": _source_hash(plan_path), "diagnostic_source": source,
                    "model": None, "runtime": None, "effective_inference": inference.model_dump(mode="json"),
                    "conditioning": "common_system_native_open_assistant_no_response_no_answer_prefix",
                    "sites": [f"{s.layer}:{s.kv_group}" for s in sites], "conditions": list(conditions),
                    "family_ids": [f["family_id"] for f in families], "factors": {"edge": 1, "directions": [1, -1], "palette": 0},
                    "allow_downloads": False, "source_check": True, "selection_filter": "none"}
        try:
            progress("fit_check")
            metadata["memory_fit"] = memory_fit(checkpoint, inference, args.device)
            from vlm_pilot.runtime import Engine

            progress("cached_model_load")
            engine = Engine(checkpoint, inference, device=args.device, allow_downloads=False)
            metadata.update(model=engine.model_profile, runtime=engine.runtime_profile)
            atomic_json(output / "metadata.json", metadata, immutable=True)
            progress("native_P0", selected_sites=metadata["sites"])
            color, _ = paired_cases(families[0], 1, 1, 0)
            probe = dict(color, candidates=color["candidates"] + [c.title() for c in color["candidates"]])
            p0 = engine.preflight(root, families[0], probe, sites=sites)
            atomic_json(output / "preflight.json", p0, immutable=True)
            if p0.get("passed") is not True:
                raise RuntimeError("Selected-site native P0 failed; no condition trials allowed.")
            score_p0, p0_cache = {"passed": False}, FamilyCache(engine, root, families[0], sites)
            try:
                candidate_preflight(p0_cache, probe, classify, score_p0)
            finally:
                atomic_json(output / "score-preflight.json", score_p0, immutable=True)
                p0_cache.clear()
            p0_passed = True
            for index, family in enumerate(families, 1):
                cache, family_rows = FamilyCache(engine, root, family, sites), []
                failure = None
                try:
                    for row in family_results(cache, conditions, classify):
                        family_rows.append(row)
                        rows.append(row)
                        progress("conditions", family=family["family_id"], site=row["site"],
                                 direction=row["direction"], condition=row["condition"])
                except BaseException as error:
                    failure = {"type": type(error).__name__, "message": "Family failed; partial rows are not a complete diagnostic."}
                    raise
                finally:
                    atomic_json(output / f"family-{index:04d}.json",
                                {"family_id": family["family_id"], "rows": family_rows,
                                 "status": "failed" if failure else "completed", "failure": failure}, immutable=True)
                    atomic_json(output / f"report-families-{index:04d}.json",
                                summarize(rows, args.families, sites, conditions,
                                          p0_passed=p0_passed, failure=failure), immutable=True)
                    cache.clear()
            report = summarize(rows, args.families, sites, conditions, p0_passed=p0_passed)
            if not report["complete"]:
                raise RuntimeError("Nonempty full condition schedule was not completed.")
            atomic_json(output / "report.json", report, immutable=True)
            progress("completed_engineering_only", report=str(output))
            return 0
        except BaseException as error:
            if not (output / "metadata.json").exists():
                atomic_json(output / "metadata.json", metadata, immutable=True)
            if not (output / "report.json").exists():
                failure = {"type": type(error).__name__, "message": "Diagnostic failed; no confirmed source or automatic go."}
                atomic_json(output / "report.json", summarize(rows, args.families, sites, conditions,
                            p0_passed=p0_passed, failure=failure), immutable=True)
            raise


if __name__ == "__main__":
    raise SystemExit(main())