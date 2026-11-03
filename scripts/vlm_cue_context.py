"""Fixed prospective cue/context DEVELOPMENT experiment after the critical audit.

Usage is explicit CLI execution with --protocol, --plan, --checkpoint, --output,
--device cuda:N and --cached-only. Imports and argument parsing are inert/stdlib.
This does NOT edit/relock the original pilot, train, select layers, resume, retry,
download or touch validation/confirmation. Engineering indices 8..15 are an
arbitrary fixed schedule, NOT fresh confirmation: engineering was used before.

Each family has eight e/d/p variants, ALL three fixed SHAPES cues and three
contexts (72 calls), plus shape/color questions in two matched frames (32 calls).
That is 104 stages/family, 832 score+generation stages/model, 1664 generations
for both endpoints, excluding mandatory P0. Every generation uses the ORIGINAL
common Answer: profile, native greedy EOS and 32 tokens, not a cheap two-token
decoder. Thinking/explanations may truncate or fail strict delivery; these are
retained, not operation errors, semantic-zero evidence or grounds for filtering.

ALL-node cue interventions include the oracle node only by exhaustive schedule;
oracle matches are labelled AFTER execution. They are NOT autonomous rescue.
Contexts supply an anchor/traversal request, NEVER the relation answer. The
primary is within-family CHANGE in cue compliance from lookup to each context,
then their equal average. Negative means context interference in this assay,
not a general mechanism. Independent external calls are not native internal
mediators; equal token caps are not equal FLOPs. Shape/color vocabularies differ.

The adjacent generated-decomposition helper is imported from exact bounded
bytes using importlib, not sys.path. Its native _run_stage and strict parser are
used with a private any-factor image-check adapter and fail-first-operation
guard. No inference, prefix, EOS, tolerance, scoring or decoder overrides occur.
All scheduled rows, raw scores/generations, prompts/tokens, metadata and exact
source texts are atomically sealed in finally, including unfinished families.
Only NEW runs/vlm-pilot/report-* roots are allowed. Batch skipping of complete
roots belongs outside this CLI. No automatic paper-ready or scientific go gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import tempfile
import time
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).absolute().parents[1]
COLORS = ("red", "blue", "green", "yellow", "purple", "orange")
SHAPES = ("triangle", "circle", "square")
CONTEXTS = ("lookup_only", "true_relation", "competing_anchor")
FACTORS = tuple((e, d, p) for e in (1, -1) for d in (1, -1) for p in (0, 1))
FAMILY_INDICES = tuple(range(8, 16))
GENERATION_BUDGET = 32
FINAL_INSTRUCTION = "Now inspect the {cue} only. What color is its chip? Reply with one color name."
RELATION_PREFIX = "Starting at the {anchor}, follow one arrow {traversal}. "
TASK_FRAMES = {
    "1": "Starting at the {anchor}, follow one arrow {traversal}. What is the destination {answer}? Reply with one {answer} name.",
    "2": "From the {anchor}, move one arrow {traversal}. Name the {answer} at the destination. Reply with one {answer} name.",
}
CHECKPOINT_PINS = {
    "8b-mpo": {"model_id": "OpenGVLab/InternVL3_5-8B-MPO", "revision": "bb33f08275aa566f3442c1f5002512ffb66d7952"},
    "8b-gspo": {"model_id": "OpenGVLab/InternVL3_5-8B", "revision": "9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79"},
}
FIXED_PROTOCOL = {
    "schema_version": 1, "kind": "cue_context_development_v1",
    "plan_id": "44f4c5dbae51af518f044c735dc96ae8f6b043cb3c7ca70bd0512fb3cf9fc050",
    "dataset_sha256": "cb93e6e3e4034750252ebce2d1d995e59c537f8a6bf6de1fc963b69b0e2e3de5",
    "split": "engineering", "family_offset": 8, "family_count": 8,
    "family_indices": list(FAMILY_INDICES),
    "selection": "arbitrary_fixed_indices_before_new_outputs_no_outcome_filters",
    "prior_use": "prior_engineering_used_not_confirmation",
    "factors": {"edges": [1, -1], "directions": [1, -1], "palettes": [0, 1]},
    "cues": list(SHAPES), "contexts": list(CONTEXTS),
    "vocabularies": {"COLORS": list(COLORS), "SHAPES": list(SHAPES)},
    "final_instruction": FINAL_INSTRUCTION, "relation_prefix": RELATION_PREFIX,
    "competing_anchor_rule": "(original_anchor+1)%3_independent_of_direction_and_gold",
    "cue_rule": "all_fixed_SHAPES_all_node_intervention_not_autonomous_rescue",
    "task_frames": TASK_FRAMES, "task_answers": ["shape", "color"],
    "checkpoints": ["8b-mpo", "8b-gspo"], "checkpoint_pins": CHECKPOINT_PINS,
    "inference": "original_unchanged_common_answer_prefix",
    "generation": {"max_new_tokens": 32, "do_sample": False, "use_cache": False,
                   "eos": "native_unchanged", "parser": "helper.strict_token_answer"},
    "stages_per_family": {"cue_context": 72, "task": 32, "total": 104},
    "stages_per_model": 832, "generations_both_models": 1664,
    "primary": "mean_family_half_true_plus_competing_cue_compliance_minus_lookup",
    "scope": "prospective_development_not_confirmation_no_mechanism_claim",
    "prohibited": ["downloads", "resume", "overwrite", "retries", "outcome_filters",
                   "training", "probes", "pilot_validation", "confirmation"],
}
_HELPER = None
__all__ = ["FIXED_PROTOCOL", "validate_protocol", "read_protocol", "select_families",
           "selected_cases", "cue_question", "make_schedule", "stage_counts",
           "grade_row", "evaluate_stage", "evaluate_family", "summarize", "main"]


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _safe(path):
    path = Path(path).expanduser().absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Traversal and symlinks are forbidden.")
    return path


def _source(path, maximum=1024**2):
    path = _safe(path)
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("Missing, empty or oversized source/protocol/image.")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if not 0 < len(payload) <= maximum:
        raise ValueError("Artifact changed size or exceeded its bound.")
    return payload


def _unique(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key.")
    return result


def _fixed(protocol):
    # Bytewise canonical equality rejects extras, duplicates (at read), bool/int
    # substitution, float caps, vocab/order changes and ANY alternate hash pins.
    if type(protocol) is not dict or _canonical(protocol) != _canonical(FIXED_PROTOCOL):
        raise ValueError("Protocol differs from the fixed cue/context development experiment.")


def read_protocol(path):
    protocol = json.loads(_source(path), object_pairs_hook=_unique)
    _fixed(protocol)
    return protocol


def validate_protocol(protocol, plan):
    """Require the concrete fixed pins, not merely any self-consistent new plan."""
    _fixed(protocol)
    if plan.plan_id != protocol["plan_id"] or plan.dataset_sha256 != protocol["dataset_sha256"]:
        raise ValueError("Protocol must match the original locked plan and dataset.")
    return deepcopy(protocol)


def _helper():
    """Private importlib module compiled from the SAME bytes embedded in metadata.

    Only its image-reference function is adapted for the complete e/d/p matrix.
    The original helper file/module and all pilot sources remain untouched.
    """
    global _HELPER
    path = _safe(Path(__file__).with_name("vlm_generated_decomposition.py"))
    payload = _source(path)
    key = (str(path), _sha(payload))
    if _HELPER is None:
        spec = importlib.util.spec_from_file_location("_cue_context_exact_adjacent", path)
        module = importlib.util.module_from_spec(spec)
        exec(compile(payload, str(path), "exec"), module.__dict__)
        module._image_reference = _image_reference
        _HELPER = (key, module)
    if _HELPER[0] != key:
        raise RuntimeError("Adjacent helper path/bytes changed; stop without retry.")
    return _HELPER[1]


def _vocabulary_check():
    colors, shapes = _helper()._vocabulary()
    if tuple(colors) != COLORS or tuple(shapes) != SHAPES:
        raise ValueError("Original renderer vocabulary differs from the fixed protocol.")


def stage_counts(families=8):
    """Return prospective counts; no token/FLOP/fit or runtime estimate implied."""
    if type(families) is not int or not 1 <= families <= 8:
        raise ValueError("Family count must be a strict integer in 1..8.")
    return {"families": families, "factors_per_family": 8, "cue_context_per_family": 72,
            "task_per_family": 32, "stages_per_family": 104,
            "cue_context": families * 72, "task": families * 32,
            "score_calls": families * 104, "generation_calls": families * 104,
            "total": families * 104, "max_new_tokens": 32, "includes_p0": False}


def _scene(family):
    """Bound a selected engineering scene before indexing; never choose by gold."""
    if (family.get("split") != "engineering" or type(family.get("index")) is not int
            or family["index"] not in FAMILY_INDICES):
        raise ValueError("Only fixed engineering indices 8..15 are enrolled.")
    anchor = family.get("anchor")
    if type(anchor) is not int or not 0 <= anchor < 3:
        raise ValueError("Anchor must be an integer node in 0..2.")
    nodes, palettes = family.get("nodes"), family.get("palettes")
    if (type(nodes) is not list or len(nodes) != 3
            or any(type(n) is not dict for n in nodes)
            or any(type(n.get("shape")) is not str or n["shape"] not in SHAPES for n in nodes)
            or sorted(n.get("shape", "") for n in nodes) != sorted(SHAPES)
            or type(palettes) is not list or len(palettes) != 2
            or any(type(p) is not list or len(p) != 3 for p in palettes)
            or any(type(c) is not str or c not in COLORS for p in palettes for c in p)
            or sorted(c for p in palettes for c in p) != sorted(COLORS)):
        raise ValueError("Require three unique SHAPES and two disjoint three-color palettes.")


def selected_cases(family):
    """All eight original binding cases in fixed e/d/p order, never gold-filtered."""
    _scene(family)
    from vlm_pilot.data import make_case

    cases = family.get("cases")
    if type(cases) is not list or any(type(c) is not dict for c in cases):
        raise ValueError("Original cases must be a list of case objects.")
    binding = [c for c in cases if c.get("task") == "binding"]
    if len(binding) != 8:
        raise ValueError("Exactly eight binding factors are required, without duplicates.")
    result = []
    for edge, direction, palette in FACTORS:
        expected = make_case(family, edge=edge, direction=direction, palette=palette)
        matches = [c for c in binding if (c.get("edge"), c.get("direction"), c.get("palette"))
                   == (edge, direction, palette)]
        if len(matches) != 1 or _canonical(matches[0]) != _canonical(expected):
            raise ValueError("Binding case factors/IDs/gold must match the original scene program.")
        result.append(deepcopy(expected))
    images = family.get("images")
    if (type(images) is not list or len(images) != 4
            or any(type(i) is not dict or type(i.get("image_id")) is not str for i in images)
            or len({i["image_id"] for i in images}) != 4):
        raise ValueError("Four unique original images are required.")
    for case in result:
        matches = [i for i in images if i["image_id"] == case["image_id"]]
        if (len(matches) != 1 or type(matches[0].get("edge")) is not int
                or type(matches[0].get("palette")) is not int
                or (matches[0]["edge"], matches[0]["palette"]) != (case["edge"], case["palette"])):
            raise ValueError("Binding case image does not match its e/p factors.")
    return result


def select_families(protocol, dataset):
    """Original ordered 24 engineering families; select precisely indices 8..15."""
    _fixed(protocol)
    engineering = [f for f in dataset["families"] if f["split"] == "engineering"]
    if (len(engineering) != 24 or [f["index"] for f in engineering] != list(range(24))
            or any(type(f["index"]) is not int for f in engineering)
            or len({f["family_id"] for f in engineering}) != 24):
        raise ValueError("Require the original ordered 24 engineering families; no fallback.")
    selected = engineering[8:16]
    ids = [c["case_id"] for family in selected for c in selected_cases(family)]
    if len(set(ids)) != 64:
        raise ValueError("Duplicate selected binding case IDs.")
    return selected


def cue_question(anchor_shape, direction, cue, context):
    """Pure prompt builder: no destination, gold, palette or model-answer input."""
    if (anchor_shape not in SHAPES or cue not in SHAPES or context not in CONTEXTS
            or type(direction) is not int or direction not in (1, -1)):
        raise ValueError("Require fixed vocabulary/context and signed integer direction.")
    final = FINAL_INSTRUCTION.format(cue=cue)
    if context == "lookup_only":
        return final
    return RELATION_PREFIX.format(anchor=anchor_shape,
                                  traversal="forward" if direction == 1 else "backward") + final


def _row(family, original, *, cue=None, context=None, answer=None, frame=None):
    anchor, target, palette = original["anchor"], original["target"], original["palette"]
    context_anchor = (None if context == "lookup_only" or context is None else
                      (anchor + 1) % 3 if context == "competing_anchor" else anchor)
    cue_node = next((i for i, n in enumerate(family["nodes"]) if n["shape"] == cue), None)
    original_color = family["palettes"][palette][target]
    gold = family["nodes"][target]["shape"] if answer == "shape" else original_color
    if cue is not None:
        question = cue_question(family["nodes"][anchor if context_anchor is None else context_anchor]["shape"],
                                original["direction"], cue, context)
        suffix, candidates = f"cue-{cue}-{context}", COLORS
    else:
        question = TASK_FRAMES[str(frame)].format(anchor=family["nodes"][anchor]["shape"],
            traversal="forward" if original["direction"] == 1 else "backward", answer=answer)
        suffix, candidates = f"task-{answer}-frame-{frame}", SHAPES if answer == "shape" else COLORS
    case = dict(deepcopy(original), case_id=original["case_id"] + "-cue-context-" + suffix,
                question=question, gold=gold, candidates=list(candidates), diagnostic=suffix)
    row = dict(_helper()._empty_stage(), family_id=family["family_id"], family_index=family["index"],
        case_id=case["case_id"], case=case, edge=original["edge"], direction=original["direction"],
        palette=palette, image_id=case["image_id"], kind="cue_context" if cue is not None else "task",
        cue=cue, cue_context=context, task_frame=frame, task_answer=answer,
        original_gold=gold, original_color_gold=original_color,
        cue_gold=None if cue_node is None else family["palettes"][palette][cue_node],
        destination_node=target, cue_node=cue_node, context_anchor=context_anchor, original_anchor=anchor,
        anchor_color=None if context_anchor is None else family["palettes"][palette][context_anchor],
        original_true_anchor_color=family["palettes"][palette][anchor],
        original_correct=False, cue_correct=False, target_hit=False, cue_following=False,
        score_original_correct=False, score_cue_correct=False, score_checked=False,
        cue_intervention_label="all_node_intervention_not_autonomous_rescue" if cue else None,
        oracle_cue_after_test=None)
    return row


def make_schedule(family):
    """104 deterministic fresh case IDs: 8 * (3 cues * 3 contexts + 2 * 2 tasks)."""
    _vocabulary_check()
    rows = []
    for case in selected_cases(family):
        for cue in SHAPES:  # FIXED vocabulary order, not node order or gold choice.
            for context in CONTEXTS:
                rows.append(_row(family, case, cue=cue, context=context))
        for frame in (1, 2):
            for answer in ("shape", "color"):
                rows.append(_row(family, case, answer=answer, frame=frame))
    if len(rows) != 104 or len({r["case_id"] for r in rows}) != 104:
        raise ValueError("The complete 104-stage family schedule must remain unique.")
    return rows


def _image_reference(root, family, case):
    """Check SAME original bytes before EVERY prepare, for ANY of eight factors."""
    matches = [i for i in family["images"] if i["image_id"] == case["image_id"]]
    if (len(matches) != 1 or type(case.get("edge")) is not int
            or type(case.get("direction")) is not int
            or type(case.get("palette")) is not int
            or type(matches[0].get("edge")) is not int
            or type(matches[0].get("palette")) is not int
            or (case["edge"], case["direction"], case["palette"]) not in FACTORS
            or (matches[0]["edge"], matches[0]["palette"]) != (case["edge"], case["palette"])):
        raise ValueError("Image must match the scheduled original e/d/p factors.")
    item = matches[0]
    relative = Path(item["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe image path.")
    payload = _source(_safe(root) / relative, maximum=8 * 1024**2)
    actual = _sha(payload)
    if actual != item["sha256"]:
        raise ValueError("Original image checksum changed, including cached native features.")
    return {"image_id": item["image_id"], "relative_path": item["relative_path"],
            "sha256": actual, "verified_before_prepare": True}


class _OperationStop(BaseException):
    """Escape the helper's Exception catches before ANY later native operation."""

    def __init__(self, operation, error):
        self.operation, self.error = operation, error
        super().__init__(f"{operation}: {type(error).__name__}: {str(error)[:2048]}")


class _Guard:
    """Native delegation only: validate raw returns before helper can continue."""

    def __init__(self, engine, stage):
        self.engine, self.stage = engine, stage
        self.tokenizer = engine.tokenizer

    def _prompt(self, *args):
        return self.engine._prompt(*args)

    def prepare(self, root, family, case, *, profile):
        try:
            # Retain at most ONE image's native features; no decoder/capture cache.
            if (self.engine._family != family["family_id"]
                    or set(self.engine._features) != {case["image_id"]}):
                _helper()._clear_features(self.engine)
            return self.engine.prepare(root, family, case, profile=profile)
        except Exception as error:
            raise _OperationStop("prepare", error) from None

    def score(self, prepared):
        try:
            result, captures = self.engine.score(prepared)  # ALL candidates; no hooks/actions.
            self.stage["score"] = _helper()._recordable(result)
            _helper()._check_score(result, captures, self.stage["input"], prepared.case["candidates"])
            self.stage["score_checked"] = True
            return result, captures
        except Exception as error:
            raise _OperationStop("score", error) from None

    def generate(self, prepared, budget):
        try:
            if type(budget) is not int or budget != 32:
                raise ValueError("Only the fixed 32-token native budget is allowed.")
            result = self.engine.generate(prepared, budget)
            self.stage["generation"] = _helper()._recordable(result)
            _helper()._check_generation(self.engine, result, budget)
            settings = result["settings"]
            if (type(settings["max_new_tokens"]) is not int
                    or type(settings["eos_token_id"]) is not int
                    or settings.get("profile") != "unrestricted_greedy_audit"):
                raise ValueError("Native integer cap/EOS and unrestricted greedy profile are required.")
            return result
        except Exception as error:
            raise _OperationStop("generate", error) from None


def grade_row(row):
    """Dual-grade STRICT delivered text, never a score winner/Engine parser/oracle.

    Oracle inclusion is evaluation metadata assigned only after an attempted
    stage; unstarted rows never masquerade as successful/observed oracle trials.
    """
    delivered = row["status"] in ("completed", "invalid_answer") and not row["errors"]
    parsed = _helper().strict_token_answer(row["generation"], row["case"]["candidates"]) if delivered else None
    original = parsed is not None and parsed == row["original_gold"]
    cued = parsed is not None and row["cue_node"] is not None and parsed == row["cue_gold"]
    score = row["score"].get("prediction") if row["score_checked"] else None
    row.update(parsed_answer=parsed, answer_valid=parsed is not None, original_correct=original,
               cue_correct=cued, target_hit=original, cue_following=cued, exact=original,
               score_original_correct=score is not None and score == row["original_gold"],
               score_cue_correct=score is not None and row["cue_node"] is not None and score == row["cue_gold"],
               oracle_cue_after_test=(row["cue_node"] == row["destination_node"]
                                      if row["attempted"] and row["cue_node"] is not None else None))
    return row


def evaluate_stage(engine, root, family, row, *, progress=None):
    """Use exact adjacent _run_stage; fail-first, invalid delivery continues."""
    if row["status"] != "not_started" or row["attempted"]:
        raise ValueError("No retry or prepared/generation-result reuse.")
    h = _helper()

    def guarded_progress(*args, **details):
        # A broken progress sink must not be swallowed by the helper's score
        # catch and followed by another native operation without publication.
        if progress is not None:
            try:
                progress(*args, **details)
            except Exception as error:
                raise _OperationStop("progress", error) from None

    try:
        h._run_stage(_Guard(engine, row), root, family, row["case"], row,
                     GENERATION_BUDGET, guarded_progress, row["case_id"])
    except _OperationStop as stop:
        row["errors"] = [dict(h._error(stop.error), operation=stop.operation)]
        raise RuntimeError(f"First {stop.operation} operation failed; schedule unfinished.") from None
    finally:
        grade_row(row)
    if row["errors"]:
        raise RuntimeError("First prepare/input operation failed; schedule unfinished.")
    return row


def _family_record(family):
    rows = make_schedule(family)
    return {"family_id": family["family_id"], "family_index": family["index"],
            "family_canonical_sha256": _sha(_canonical(family)), "family": deepcopy(family),
            "schedule_sha256": _sha(_canonical([r["case"] for r in rows])),
            "status": "not_started", "failure": None, "rows": rows}


def _counts(rows):
    n = len(rows)
    result = {"n": n, **{k: sum(r[k] is True for r in rows) for k in
        ("attempted", "answer_valid", "original_correct", "cue_correct",
         "score_original_correct", "score_cue_correct")}}
    result.update(cue_following=result["cue_correct"], target_hit=result["original_correct"],
        cue_compliance=result["cue_correct"] / n if n else None,
        target_hit_rate=result["original_correct"] / n if n else None,
        oracle_cue_matches_after_test=sum(r["oracle_cue_after_test"] is True for r in rows),
        cue_following_original_wrong=sum(r["cue_correct"] and not r["original_correct"] for r in rows),
        target_hit_cue_wrong=sum(r["cue_node"] is not None and r["original_correct"]
                                 and not r["cue_correct"] for r in rows),
        valid_neither=sum(r["cue_node"] is not None and r["answer_valid"]
                          and not r["original_correct"] and not r["cue_correct"] for r in rows),
        operation_errors=sum(len(r["errors"]) for r in rows),
        **{s: sum(r["status"] == s for r in rows) for s in
           ("completed", "invalid_answer", "failed", "running", "not_started")})
    return result


def _shape_pairs(rows, frame):
    """Raw relational SHAPE correctness, including paired reversal/palette joints.

    A pair/quad is consistent only if EVERY outcome equals its relation gold,
    not merely if two guesses differ. Both edge and query reversals are exposed.
    """
    shape = {(r["edge"], r["direction"], r["palette"]): r for r in rows
             if r["kind"] == "task" and r["task_answer"] == "shape" and r["task_frame"] == frame}
    if len(shape) != 8:
        raise ValueError("All eight relational shape outcomes must remain denominators.")

    def group(keys):
        items = [shape[k] for k in keys]
        return {"factors": [list(k) for k in keys], "case_ids": [r["case_id"] for r in items],
                "gold": [r["original_gold"] for r in items],
                "answers": [r["parsed_answer"] for r in items],
                "statuses": [r["status"] for r in items],
                "all_exact": all(r["original_correct"] for r in items)}

    groups = {
        "direction_reversal": [group([(e, 1, p), (e, -1, p)]) for e in (1, -1) for p in (0, 1)],
        "edge_reversal": [group([(1, d, p), (-1, d, p)]) for d in (1, -1) for p in (0, 1)],
        "palette": [group([(e, d, 0), (e, d, 1)]) for e in (1, -1) for d in (1, -1)],
        "joint_direction_reversal_palette": [group([(e, d, p) for d in (1, -1) for p in (0, 1)]) for e in (1, -1)],
        "joint_edge_reversal_palette": [group([(e, d, p) for e in (1, -1) for p in (0, 1)]) for d in (1, -1)],
    }
    return {"task_frame": frame, "n_outcomes": 8, "exact": sum(r["original_correct"] for r in shape.values()),
            "all_eight_exact": all(r["original_correct"] for r in shape.values()),
            "groups": {k: {"n": len(v), "all_exact": sum(g["all_exact"] for g in v), "pairs_or_quads": v}
                       for k, v in groups.items()}}


def _family_summary(record):
    # Reconstruct evaluation from raw delivered generations, not stored flags.
    rows = [grade_row(deepcopy(r)) for r in record["rows"]]
    expected = make_schedule(record["family"])
    if (record["family_id"] != record["family"]["family_id"]
            or record["family_index"] != record["family"]["index"]
            or record["family_canonical_sha256"] != _sha(_canonical(record["family"]))
            or len(rows) != 104
            or _canonical([r["case"] for r in rows]) != _canonical([r["case"] for r in expected])
            or record["schedule_sha256"] != _sha(_canonical([r["case"] for r in expected]))):
        raise ValueError("Family/schedule canonical identity changed; no hidden filtering.")
    # Evaluation fields must also match the canonical scene, not imported flags.
    keys = ("family_id", "family_index", "case_id", "edge", "direction", "palette", "image_id",
            "kind", "cue", "cue_context", "task_frame", "task_answer", "original_gold",
            "cue_gold", "destination_node", "cue_node", "context_anchor", "original_anchor",
            "anchor_color", "original_true_anchor_color", "original_color_gold")
    if any(any(r[k] != e[k] for k in keys) for r, e in zip(rows, expected, strict=True)):
        raise ValueError("Row evaluation fields differ from the canonical scheduled scene.")
    cells = {context: _counts([r for r in rows if r["cue_context"] == context]) for context in CONTEXTS}
    changes = {c: cells[c]["cue_compliance"] - cells["lookup_only"]["cue_compliance"] for c in CONTEXTS[1:]}
    groups = [{"cue": cue, "context": context, **_counts([r for r in rows
              if r["cue"] == cue and r["cue_context"] == context])}
              for cue in SHAPES for context in CONTEXTS]
    task = {f"{answer}_frame_{frame}": _counts([r for r in rows if r["task_answer"] == answer
             and r["task_frame"] == frame]) for frame in (1, 2) for answer in ("shape", "color")}
    paired = []
    for edge, direction, palette in FACTORS:
        items = [r for r in rows if r["kind"] == "task" and r["task_answer"] == "shape"
                 and (r["edge"], r["direction"], r["palette"]) == (edge, direction, palette)]
        paired.append({"factors": [edge, direction, palette], "answers": [r["parsed_answer"] for r in items],
                       "gold": [r["original_gold"] for r in items],
                       "both_frames_exact": all(r["original_correct"] for r in items)})
    complete = all(r["status"] in ("completed", "invalid_answer") and not r["errors"] for r in rows)
    return {"family_id": record["family_id"], "family_index": record["family_index"],
            "family_canonical_sha256": record["family_canonical_sha256"],
            "schedule_sha256": record["schedule_sha256"],
            "counts": _counts(rows), "contexts": cells, "cue_context_groups": groups,
            "tasks": task, "cue_compliance_change": changes,
            "primary_value": sum(changes.values()) / 2, "execution_complete": complete,
            "shape_relational": {"frames": [_shape_pairs(rows, f) for f in (1, 2)],
                                 "paired_frames_n": 8, "paired_frames": paired,
                                 "paired_frames_both_exact": sum(p["both_frames_exact"] for p in paired)}}


def summarize(records, *, checkpoint, p0_passed=False, failure=None):
    """All EIGHT families/832 stages, even after model-load/P0/operation failure."""
    if (checkpoint not in CHECKPOINT_PINS or len(records) != 8
            or [r["family_index"] for r in records] != list(FAMILY_INDICES)
            or len({r["family_id"] for r in records}) != 8):
        raise ValueError("All eight fixed families and a matched endpoint are required.")
    families = [_family_summary(r) for r in records]
    rows = [grade_row(deepcopy(row)) for record in records for row in record["rows"]]
    complete = p0_passed and failure is None and all(f["execution_complete"] for f in families)
    changes = {c: sum(f["cue_compliance_change"][c] for f in families) / 8 for c in CONTEXTS[1:]}
    return {"kind": FIXED_PROTOCOL["kind"], "checkpoint": checkpoint, "n_family": 8,
        "n": 832, "expected_schedule": stage_counts(), "counts": _counts(rows),
        "contexts": {c: _counts([r for r in rows if r["cue_context"] == c]) for c in CONTEXTS},
        "cue_context_groups": [{"cue": cue, "context": context,
            **_counts([r for r in rows if r["cue"] == cue and r["cue_context"] == context])}
            for cue in SHAPES for context in CONTEXTS],
        "factor_cue_context_groups": [{"edge": e, "direction": d, "palette": p,
            "cue": cue, "context": context,
            **_counts([r for r in rows if (r["edge"], r["direction"], r["palette"]) == (e, d, p)
                       and r["cue"] == cue and r["cue_context"] == context])}
            for e, d, p in FACTORS for cue in SHAPES for context in CONTEXTS],
        "tasks": {f"{a}_frame_{f}": _counts([r for r in rows if r["task_answer"] == a and r["task_frame"] == f])
                  for f in (1, 2) for a in ("shape", "color")},
        "paired_families": families, "cue_compliance_change": changes,
        "primary": FIXED_PROTOCOL["primary"], "primary_value": sum(changes.values()) / 2,
        "primary_interpretable_as_complete_schedule": complete,
        "primary_interpretation": "negative_is_context_interference_here_not_general_mechanism",
        "p0_passed": p0_passed, "execution_complete": complete, "unfinished": not complete,
        "failure": failure, "selection_filter": "none", "confirmation": "untouched",
        "prior_use": FIXED_PROTOCOL["prior_use"], "hypothesis_decision": "not_automated",
        "statistics": "raw_counts_and_equal_weight_paired_family_values_no_paper_ready_gate",
        "limitations": ["Engineering previously used; prospective prompts are not confirmation",
            "All-node/oracle-cue intervention is not autonomous rescue",
            "Invalid/truncated strict delivery is not evidence of semantically wrong reasoning",
            "SHAPES has three answers, COLORS six; syntax matching does not establish common internal code",
            "Equal token caps are not equal FLOPs or equal cost; thinking may truncate",
            "Independent external calls are not native internal mediator identification",
            "P0 does not promise hardware fit, runtime, mechanistic novelty or scientific success"]}


def evaluate_family(engine, root, family, output, *, record=None, metadata=None, progress=None):
    """One atomic immutable ALL-row family artifact in finally, never retries."""
    from vlm_pilot.io import atomic_json

    output = _safe(output)
    if output.exists():
        raise FileExistsError("Family artifact exists; no overwrite/resume/model work.")
    record = _family_record(family) if record is None else record
    record["status"] = "running"
    try:
        _helper()._clear_features(engine)
        for row in record["rows"]:
            evaluate_stage(engine, root, family, row, progress=progress)
        return record
    except BaseException as error:
        record["failure"] = _helper()._error(error)
        raise
    finally:
        record["status"] = "unfinished" if record["failure"] else "completed"
        record["sealed"] = True
        record["counts"] = _family_summary(record)
        record["metadata"] = deepcopy(metadata)
        try:
            atomic_json(output, _helper()._recordable(record), immutable=True)
        finally:
            _helper()._clear_features(engine)


def _device(value):
    if re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", value) is None:
        raise argparse.ArgumentTypeError("Require explicit cuda:N; no CPU/automatic fallback.")
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("protocol", "plan", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, choices=tuple(CHECKPOINT_PINS))
    parser.add_argument("--device", required=True, type=_device)
    parser.add_argument("--cached-only", required=True, action="store_true")
    return parser


def _checkpoint_gate(plan, name, checkpoint_for):
    # Verify BOTH native pinned members of the matched lineage, before a model.
    pair = {n: checkpoint_for(plan, n) for n in CHECKPOINT_PINS}
    for n, checkpoint in pair.items():
        pin = CHECKPOINT_PINS[n]
        if (checkpoint.backend != "internvl" or checkpoint.family != "8b"
                or checkpoint.stage != n[3:] or checkpoint.model_id != pin["model_id"]
                or checkpoint.revision != pin["revision"]):
            raise ValueError("Require the original matched native 8B MPO/GSPO full revision pins.")
    return pair[name]


def _source_snapshots():
    _helper()  # Verify that the executed private helper still has these bytes.
    result = {}
    for path in (Path(__file__), Path(__file__).with_name("vlm_generated_decomposition.py")):
        payload = _source(path)
        result["scripts/" + path.name] = {"sha256": _sha(payload), "size_bytes": len(payload),
                                       "text": payload.decode("utf-8")}
    return result


def main(argv=None):
    args = _parser().parse_args(argv)  # Help/malformed CLI: no dependency/model imports.
    protocol_path, plan_path, output = map(_safe, (args.protocol, args.plan, args.output))
    if output.parent != _safe(REPO / "runs/vlm-pilot") or not re.fullmatch(r"report-[A-Za-z0-9_-]+", output.name):
        raise ValueError("Use a NEW runs/vlm-pilot/report-* root for backup visibility.")
    if output.exists():
        raise FileExistsError("Output exists; no overwrite/resume. Batch skips belong outside CLI.")
    protocol_bytes = _source(protocol_path)
    protocol = json.loads(protocol_bytes, object_pairs_hook=_unique)
    _fixed(protocol)
    from filelock import FileLock

    from vlm_pilot.data import load_dataset
    from vlm_pilot.io import atomic_json, identity, now
    from vlm_pilot.plan import checkpoint_for, read_plan

    plan_bytes = _source(plan_path, 32 * 1024**2)
    plan = read_plan(plan_path, check_source=True)  # Preserve original science/identity.
    if _source(plan_path, 32 * 1024**2) != plan_bytes:
        raise ValueError("Plan file changed during validation; no model work permitted.")
    validate_protocol(protocol, plan)
    checkpoint = _checkpoint_gate(plan, args.checkpoint, checkpoint_for)
    root = _safe(plan.spec.dataset)
    dataset = load_dataset(root, verify_images=False)  # Validates ALL original SHA bytes.
    if identity(dataset) != protocol["dataset_sha256"]:
        raise ValueError("Verified dataset differs from the original fixed plan.")
    families = select_families(protocol, dataset)
    records = [_family_record(f) for f in families]  # Populate all denominators BEFORE load.
    inference = plan.spec.inference  # SAME object: no copy/override/relock.
    if inference.answer_prefix != "Answer:":
        raise ValueError("Require the original common Answer: prefix unchanged.")
    sources = _source_snapshots()
    metadata = {"kind": FIXED_PROTOCOL["kind"], "created_at": now(), "protocol": deepcopy(protocol),
        "protocol_file_sha256": _sha(protocol_bytes), "protocol_canonical_sha256": _sha(_canonical(protocol)),
        "plan": plan.model_dump(mode="json"), "plan_file_sha256": _sha(plan_bytes),
        "source_check": True, "pilot_source_sha256": plan.source_sha256,
        "source_files": sources, "source_sha256": _sha(_canonical({k: v["sha256"] for k, v in sources.items()})),
        "checkpoint": args.checkpoint, "matched_checkpoint_pins": deepcopy(CHECKPOINT_PINS),
        "model": None, "runtime": None, "original_inference": inference.model_dump(mode="json"),
        "effective_inference": inference.model_dump(mode="json"), "device_override": args.device,
        "allow_downloads": False, "cached_only": True, "conditioning": "original_common_answer_prefix",
        "max_new_tokens": 32, "do_sample": False, "use_cache": False, "eos_override": False,
        "sites": [], "case_hooks": False, "full_captures": False,
        "image_cache": "at_most_one_native_image_no_decoder_states",
        "p0": "usual_native_embedding_capture_self_ALL_COLORS_plus_SHAPES",
        "expected_schedule": stage_counts(), "family_indices": list(FAMILY_INDICES),
        "family_ids": [f["family_id"] for f in families], "selection_filter": "none",
        "prior_use": FIXED_PROTOCOL["prior_use"], "confirmation": "untouched"}
    start, engine, p0_passed, failure = time.perf_counter(), None, False, None

    def progress(stage, **details):
        rows = [r for f in records for r in f["rows"]]
        done = sum(r["status"] in ("completed", "invalid_answer") for r in rows)
        elapsed = time.perf_counter() - start
        event = {"stage": stage, "device": args.device, "completed": done, "total": 832,
                 "attempted": sum(r["attempted"] for r in rows), "elapsed_seconds": elapsed,
                 "eta_seconds": elapsed * (832 - done) / done if done else None,
                 "eta_kind": "observed_stage_extrapolation_not_time_or_fit_promise", "sealed": False,
                 **details}
        print(json.dumps(event, allow_nan=False), flush=True)
        return event

    lock = Path(tempfile.gettempdir()) / ("interp-vlm-" + args.device.replace(":", "-") + ".lock")
    with FileLock(str(lock), timeout=0):  # Native lifetime guard BEFORE importing/creating Engine.
        output.mkdir(parents=True, exist_ok=False)
        try:
            progress("cached_native_model_load")
            engine = _helper().cached_engine_class()(checkpoint, inference, device=args.device, allow_downloads=False)
            metadata.update(model=engine.model_profile, runtime=engine.runtime_profile)
            p0 = {"passed": False}
            try:
                probe = dict(selected_cases(families[0])[0], candidates=list(COLORS + SHAPES))
                _image_reference(root, families[0], probe)
                progress("native_P0_all_fixed_candidates")
                p0 = engine.preflight(root, families[0], probe)
                _helper()._require_preflight(p0, COLORS + SHAPES)
                p0_passed = True
                metadata["p0_passed"] = True
            except BaseException as error:
                p0["error"] = _helper()._error(error)
                raise
            finally:
                atomic_json(output / "preflight.json", _helper()._recordable(p0), immutable=True)
                _helper()._clear_features(engine)
            for family, record in zip(families, records, strict=True):
                evaluate_family(engine, root, family, output / f"family-{family['index']:04d}.json",
                                record=record, metadata=metadata, progress=progress)
                progress("family_published", family_index=family["index"], family_id=family["family_id"])
        except BaseException as error:
            failure = _helper()._error(error)
            raise
        finally:
            # Seal unstarted families too: no missing denominator after early failure.
            metadata.update(elapsed_seconds=time.perf_counter() - start, p0_passed=p0_passed,
                            failure=failure, sealed=True)
            for record in records:
                path = output / f"family-{record['family_index']:04d}.json"
                if not path.exists():
                    record.update(status="unfinished", failure=failure, sealed=True, metadata=deepcopy(metadata))
                    record["counts"] = _family_summary(record)
                    atomic_json(path, _helper()._recordable(record), immutable=True)
            if engine is not None:
                _helper()._clear_features(engine)
            report = summarize(records, checkpoint=args.checkpoint, p0_passed=p0_passed, failure=failure)
            report.update(metadata=deepcopy(metadata), metadata_sha256=_sha(_canonical(metadata)))
            report["family_artifacts"] = []
            for record in records:
                path = output / f"family-{record['family_index']:04d}.json"
                payload = _source(path, 32 * 1024**2)
                report["family_artifacts"].append({"path": path.name, "sha256": _sha(payload),
                    "size_bytes": len(payload), "family_index": record["family_index"],
                    "family_canonical_sha256": record["family_canonical_sha256"],
                    "schedule_sha256": record["schedule_sha256"]})
            atomic_json(output / "metadata.json", _helper()._recordable(metadata), immutable=True)
            event = progress("sealed_complete" if report["execution_complete"] else "sealed_unfinished")
            event.update(sealed=True, unfinished=report["unfinished"], failure=failure,
                         expected_schedule=stage_counts(), counts=report["counts"])
            atomic_json(output / "progress.json", event, immutable=True)
            atomic_json(output / "report.json", _helper()._recordable(report), immutable=True)
        return 0 if report["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())