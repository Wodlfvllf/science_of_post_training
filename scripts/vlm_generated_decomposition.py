"""Engineering-only generated-selector backup for C2 if text residuals are null.

Explicit CLI execution loads ONE cached, pinned native Engine, with the ORIGINAL
Inference object, common conversation profile and UNCHANGED answer prefix. There
are no layer selections, hooks or captures in case scoring/generation. Mandatory
Engine.preflight alone runs the usual native/capture/self P0, with ALL fixed color
and shape candidates, before any cases. No downloads, training, confirmation,
oracle cue, gold-only teacher forcing, stage contrast, resume or overwrite.

The first 1..4 engineering families use their original e=+1,p=0 image and BOTH
query directions: default four families means eight cases per model. Each case
scores all fixed candidates and calls Engine.generate(prepared, 32) on the direct
color question and the destination-SHAPE question. This is Engine's unrestricted
native greedy, full-prefix-recomputed, UNCACHED generation, not model.chat(), a
custom decoder, best-candidate selection or answer extraction. The actual native
prompt, token IDs, image SHA, score inputs and raw generated tokens are recorded.

Only an EOS-terminated bare generated SHAPES word (casefold, outer whitespace,
at most ONE final period) permits two NEW prepares on the SAME checksummed image:
a color readout cued by the MODEL PREDICTION, and a wrong-selector control cued by
prediction+1 in the FIXED data.SHAPES cyclic order. Both are scored over ALL
data.COLORS and generated at 32 tokens; both are evaluated against the ORIGINAL
destination color, never the cued node's color. Source shape gold is evaluation
metadata only. Candidate scoring necessarily includes gold among the whole fixed
vocabulary; it never supplies a gold-only answer tail to generation or a cue.

Every step has a distinct case_id; there is no prepared-prompt reuse across steps.
CachedEngine retains at most ONE image's native features per family, not decoder
states/full captures. Each family is published atomically in finally, including
failed/invalid/partial stages. Reports retain ALL scheduled cases in denominators,
including invalid shapes, skipped chains and interrupted/unstarted cases. Finite
P0 and these bounds are not a runtime, RAM/VRAM-fit or scientific-validity promise.
Import and argument parsing are stdlib-only and perform no model generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

GENERATION_BUDGET = 32
STAGES = ("direct", "shape", "label_chain", "wrongcue_chain")
WRONG_SELECTOR_RULE = "prediction_plus_1_fixed_SHAPES_cyclic"
__all__ = ["strict_token_answer", "selected_cases", "shape_case", "next_shape", "cue_case",
           "evaluate_case", "evaluate_family", "summarize", "cached_engine_class", "main"]


def _vocabulary():
    # The renderer's fixed vocabulary, never a family permutation/learned vocabulary.
    from vlm_pilot.data import COLORS, SHAPES

    return COLORS, SHAPES


def strict_token_answer(generation, candidates):
    """Return a canonical bare answer or None; ignore Engine's permissive parser.

    This function does not extract a subsequence or choose a scored candidate.
    finish_reason='eos' is required even for apparently complete text. Exactly
    one optional period is allowed, not rstrip('.')'s unlimited periods. The
    caller separately checks native raw token/settings/EOS consistency.
    """
    if not isinstance(generation, Mapping) or generation.get("finish_reason") != "eos":
        return None
    text = generation.get("text")
    if not isinstance(text, str):
        return None
    canonical = {word.casefold(): word for word in candidates}
    answer = text.strip().casefold()
    if re.fullmatch(r"[a-z]+\.?", answer) is None:
        return None
    return canonical.get(answer.removesuffix("."))


def selected_cases(family):
    """Select the TWO original binding cases, e=+1,p=0; never filter by gold."""
    if family.get("split") != "engineering":
        raise ValueError("Only engineering families are allowed; no split fallback.")
    result = []
    for direction in (1, -1):
        matches = [c for c in family["cases"] if c["task"] == "binding"
                   and c["edge"] == 1 and c["palette"] == 0 and c["direction"] == direction]
        if len(matches) != 1:
            raise ValueError("Require exactly one original e=+1,p=0 case per direction.")
        result.append(deepcopy(matches[0]))
    if result[0]["image_id"] != result[1]["image_id"]:
        raise ValueError("Both query directions must use the same original image.")
    return result


def shape_case(family, case):
    """Question uses anchor/direction only; target shape stays evaluation metadata."""
    _, shapes = _vocabulary()
    anchor = family["nodes"][case["anchor"]]["shape"]
    traversal = "forward" if case["direction"] == 1 else "backward"
    return dict(deepcopy(case), case_id=case["case_id"] + "-generated-shape",
                question=(f"Starting at the {anchor}, follow one arrow {traversal}. "
                          "What shape is the destination? Reply with one shape name."),
                gold=family["nodes"][case["target"]]["shape"], candidates=list(shapes),
                diagnostic="generated_destination_shape")


def next_shape(prediction):
    """Prediction+1 in data.SHAPES, NOT nodes order and NOT target/oracle+1."""
    _, shapes = _vocabulary()
    if prediction not in shapes:
        raise ValueError("Cue requires a valid canonical model-generated SHAPES prediction.")
    return shapes[(shapes.index(prediction) + 1) % len(shapes)]


def cue_case(case, prediction, *, wrong_selector=False):
    """Construct a fresh same-image readout from prediction; keep ORIGINAL color gold.

    No family/nodes argument is needed: neither cue can consult an oracle selector.
    The case_id includes the actual prediction even in the wrong-selector control.
    """
    colors, shapes = _vocabulary()
    if prediction not in shapes:
        raise ValueError("Invalid shape must not queue any label readout.")
    cue = next_shape(prediction) if wrong_selector else prediction
    stage = "wrongcue-prediction-plus-1" if wrong_selector else "label-chain"
    return dict(deepcopy(case), case_id=f"{case['case_id']}-generated-{stage}-{prediction}-cue-{cue}",
                question=f"What color is the chip at the {cue}? Reply with one color name.",
                candidates=list(colors), diagnostic=stage, predicted_shape=prediction,
                cued_shape=cue, selector_rule=WRONG_SELECTOR_RULE if wrong_selector
                else "model_generated_prediction")


def _error(error):
    # Keep the actual bounded error, without a traceback or chained Hub exception.
    return {"type": type(error).__name__, "message": str(error)[:2048]}


def _recordable(value):
    """Preserve malformed raw outputs without permitting NaN/Inf in durable JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, Mapping):
        return {str(k): _recordable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_recordable(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    return {"unsupported_type": type(value).__name__}


def _empty_stage():
    return {"status": "not_started", "attempted": False, "input": None, "score": None,
            "generation": None, "parsed_answer": None, "answer_valid": False,
            "exact": False, "score_exact": False, "errors": []}


def _new_record(family, case):
    return {"family_id": family["family_id"], "case_id": case["case_id"],
            "direction": case["direction"], "original_case": deepcopy(case),
            "evaluation": {"original_destination_color": case["gold"],
                           "destination_shape": family["nodes"][case["target"]]["shape"],
                           "target_node": case["target"]},
            "status": "not_started", "stages": {name: _empty_stage() for name in STAGES}}


def _image_reference(root, family, case):
    matches = [i for i in family["images"] if i["image_id"] == case["image_id"]]
    if len(matches) != 1 or matches[0]["edge"] != 1 or matches[0]["palette"] != 0:
        raise ValueError("Every step requires the original family e=+1,p=0 image.")
    item = matches[0]
    path = Path(root).absolute() / item["relative_path"]
    if (Path(item["relative_path"]).is_absolute() or ".." in path.parts
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("Unsafe image path.")
    if not path.is_file() or not 0 < path.stat().st_size <= 8 * 1024**2:
        raise ValueError("Missing, empty or oversized original image.")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise ValueError("Original image checksum changed, including cached-feature steps.")
    return {"image_id": item["image_id"], "relative_path": item["relative_path"],
            "sha256": actual, "verified_before_prepare": True}


def _prepared_input(engine, prepared, image):
    ids = prepared.input_ids
    if (not isinstance(ids, tuple) or not ids or any(type(i) is not int or i < 0 for i in ids)
            or prepared.row != len(ids) - 1
            or prepared.prompt != engine._prompt(prepared.case["question"], True, "common")):
        raise ValueError("Prepare must retain actual original common-prefix native inputs.")
    return {"case": deepcopy(prepared.case), "question": prepared.case["question"],
            "rendered_prompt": prepared.prompt, "input_ids": list(ids), "row": prepared.row,
            "image_columns": list(prepared.image_columns), "image": image,
            "profile": "common"}


def _check_score(result, captures, inputs, candidates):
    if not isinstance(result, Mapping) or captures or result.get("routing") != {}:
        raise ValueError("Case scoring must be native/unhooked, without captures or routing.")
    scores, tokens = result.get("scores"), result.get("candidate_token_ids")
    if (not isinstance(scores, Mapping) or set(scores) != set(candidates)
            or not isinstance(tokens, Mapping) or set(tokens) != set(candidates)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in scores.values())
            or any(not isinstance(v, (list, tuple)) or not v
                   or any(type(t) is not int or t < 0 for t in v) for v in tokens.values())):
        raise ValueError("Require finite scores/token IDs for ALL fixed candidates.")
    for name in ("rendered_prompt", "input_ids", "row", "image_columns"):
        if result.get(name) != inputs[name]:
            raise ValueError("Score input differs from its own fresh prepare.")
    winners = [c for c in candidates if scores[c] == max(scores.values())]
    if (result.get("prediction") != (winners[0] if len(winners) == 1 else None)
            or result.get("tied") is not (len(winners) != 1)):
        raise ValueError("Scored candidate ties must remain None.")


def _check_generation(engine, result, budget):
    if not isinstance(result, Mapping) or not isinstance(result.get("text"), str):
        raise ValueError("Native generation must return raw text and tokens.")
    tokens, settings = result.get("token_ids"), result.get("settings")
    eos = engine.tokenizer.eos_token_id
    if (not isinstance(tokens, list) or not 1 <= len(tokens) <= budget
            or any(type(t) is not int or t < 0 for t in tokens)
            or not isinstance(settings, Mapping) or settings.get("max_new_tokens") != budget
            or settings.get("do_sample") is not False or settings.get("use_cache") is not False
            or settings.get("eos_token_id") != eos):
        raise ValueError("Native generation must be greedy/uncached with the actual fixed budget.")
    reason = "eos" if tokens[-1] == eos else "length"
    if (result.get("finish_reason") != reason or eos in tokens[:-1]
            or reason == "length" and len(tokens) != budget):
        raise ValueError("Raw tokens disagree with native EOS/budget finish reason.")


def _require_preflight(p0, candidates):
    """Require the usual Engine gate's actual finite/self/all-candidate evidence.

    This validates its existing report, never changes native tolerances, selects
    a layer, introduces another hooked pass or replaces Engine.preflight.
    """
    if not isinstance(p0, Mapping) or p0.get("passed") is not True:
        raise RuntimeError("Native/self/all-candidate P0 failed; no cases permitted.")
    checks = p0.get("checks", {})
    for name in ("native_embedding", "capture_only", "self_qk"):
        check = checks.get(name, {})
        maximum = check.get("max_abs")
        if (check.get("passed") is not True or check.get("finite") is not True
                or check.get("argmax_match") is not True or check.get("mismatches") != 0
                or type(maximum) not in (int, float) or not math.isfinite(maximum)):
            raise RuntimeError("Incomplete or nonfinite native/self P0 evidence; no cases permitted.")
    suffixes = p0.get("candidate_suffixes", {})
    if set(suffixes) != set(candidates) or any(
            not isinstance(tokens, (list, tuple)) or not tokens
            or any(type(t) is not int or t < 0 for t in tokens) for tokens in suffixes.values()):
        raise RuntimeError("P0 must cover ALL known color/shape candidate suffixes.")
    extended = p0.get("multi_token_candidate_checks", {})
    if set(extended) != {c for c, tokens in suffixes.items() if len(tokens) > 1} or any(
            check.get("passed") is not True
            or type(check.get("max_abs")) not in (int, float)
            or not math.isfinite(check["max_abs"]) for check in extended.values()):
        raise RuntimeError("Missing/failed native multi-token candidate self P0 evidence.")


def _run_stage(engine, root, family, case, stage, budget, progress, name):
    """Populate in place BEFORE risky operations, retaining score/gen partial failures."""
    start = time.perf_counter()
    stage.update(status="running", attempted=True, case=deepcopy(case))
    try:
        if progress:
            progress(name, operation="prepare", case_id=case["case_id"])
        # Even cached native features must correspond to unchanged ORIGINAL bytes.
        image = _image_reference(root, family, case)
        prepared = engine.prepare(root, family, case, profile="common")
        stage["input"] = _prepared_input(engine, prepared, image)
        try:
            if progress:
                progress(name, operation="score", case_id=case["case_id"])
            result, captures = engine.score(prepared)  # No sites/actions, no gold-only suffix.
            stage["score"] = _recordable(result)
            _check_score(result, captures, stage["input"], case["candidates"])
            stage["score_exact"] = result["prediction"] == case["gold"]
        except Exception as error:
            stage["errors"].append(dict(_error(error), operation="score"))
        # A score failure cannot silently hide a raw native generation attempt.
        try:
            if progress:
                progress(name, operation="generate", case_id=case["case_id"])
            generation = engine.generate(prepared, budget)
            stage["generation"] = _recordable(generation)
            _check_generation(engine, generation, budget)
            prediction = strict_token_answer(generation, case["candidates"])
            stage.update(parsed_answer=prediction, answer_valid=prediction is not None,
                         exact=prediction is not None and prediction == case["gold"])
            if prediction is None:
                stage["answer_failure"] = "invalid_or_truncated_bare_answer"
        except Exception as error:
            stage["errors"].append(dict(_error(error), operation="generate"))
    except Exception as error:
        stage["errors"].append(dict(_error(error), operation="prepare"))
    except BaseException as error:
        stage["errors"].append(dict(_error(error), operation="interrupted"))
        raise
    finally:
        stage["elapsed_seconds"] = time.perf_counter() - start
        stage["status"] = ("failed" if stage["errors"] else "completed"
                           if stage["answer_valid"] else "invalid_answer")


def evaluate_case(engine, root, family, case, budget=GENERATION_BUDGET, *, record=None,
                  progress=None):
    """Return the four-stage record; valid GENERATED shape alone gates both cues.

    engine implements prepare(root,family,case,profile), score(prepared)->(dict,{}),
    generate(prepared,budget)->dict, _prompt(), tokenizer.eos_token_id. Tiny spies
    need no model/tensor libraries. Optional record is populated in place so an
    enclosing family finally can publish it even on KeyboardInterrupt/SystemExit.
    Native operation errors are recorded, not replaced by best-candidate answers.
    """
    if type(budget) is not int or budget != GENERATION_BUDGET:
        raise ValueError("This engineering diagnostic has a fixed 32-token budget.")
    if case not in selected_cases(family):
        raise ValueError("Use an original engineering e=+1,p=0 binding case.")
    colors, _ = _vocabulary()
    record = _new_record(family, case) if record is None else record
    stages = record["stages"]
    record["status"] = "running"
    try:
        direct = dict(deepcopy(case), case_id=case["case_id"] + "-generated-direct",
                      candidates=list(colors))
        _run_stage(engine, root, family, direct, stages["direct"], budget, progress, "direct")
        source = shape_case(family, case)
        _run_stage(engine, root, family, source, stages["shape"], budget, progress, "shape")
        prediction = stages["shape"]["parsed_answer"]
        if prediction is None:
            for name in ("label_chain", "wrongcue_chain"):
                stages[name].update(status="skipped", skip_reason="invalid_or_failed_generated_shape")
        else:
            for name, wrong in (("label_chain", False), ("wrongcue_chain", True)):
                cue = cue_case(case, prediction, wrong_selector=wrong)
                _run_stage(engine, root, family, cue, stages[name], budget, progress, name)
        return record
    finally:
        record["status"] = ("failed" if any(s["status"] in ("failed", "invalid_answer", "running",
                                                            "not_started") for s in stages.values())
                            else "completed")


def cached_engine_class():
    """Lazily subclass the ORIGINAL Engine; change no inference/decoder behavior."""
    from vlm_pilot.runtime import Engine

    class CachedEngine(Engine):
        def prepare(self, root, family, case, *, profile="common"):
            if profile != "common":
                raise ValueError("Only unchanged common conditioning is allowed.")
            prepared = super().prepare(root, family, case, profile=profile)
            # Native Engine already keys image features by family and image_id.
            if len(self._features) > 1 or set(self._features) != {case["image_id"]}:
                raise ValueError("Family feature cache must contain only the original image.")
            return prepared

    return CachedEngine


def _clear_features(engine):
    engine._features.clear()
    engine._family = None


def _family_record(family):
    return {"family_id": family["family_id"], "status": "not_started", "failure": None,
            "rows": [_new_record(family, case) for case in selected_cases(family)]}


def summarize(family_records, *, p0_passed=False, failure=None):
    """All scheduled cases are denominators; score wins NEVER become delivered answers."""
    rows = [row for family in family_records for row in family["rows"]]
    n = len(rows)
    cells = {}
    for name in STAGES:
        stages = [row["stages"][name] for row in rows]
        exact = sum(stage["exact"] is True for stage in stages)
        cells[name] = {"n": n, "attempted": sum(s["attempted"] is True for s in stages),
                       "valid": sum(s["answer_valid"] is True for s in stages),
                       "exact": exact, "accuracy": exact / n if n else None,
                       "score_exact": sum(s["score_exact"] is True for s in stages),
                       "completed": sum(s["status"] == "completed" for s in stages),
                       "failed": sum(s["status"] == "failed" for s in stages),
                       "invalid_answer": sum(s["status"] == "invalid_answer" for s in stages),
                       "skipped": sum(s["status"] == "skipped" for s in stages),
                       "not_started": sum(s["status"] == "not_started" for s in stages)}
    return {"kind": "engineering_only_generated_decomposition", "n": n,
            "n_family": len(family_records), "n_case": n, "p0_passed": p0_passed,
            "family_ids": [f["family_id"] for f in family_records], "stages": cells,
            "direct_exact": cells["direct"]["exact"], "shape_correct": cells["shape"]["exact"],
            "label_chain": cells["label_chain"]["exact"],
            "wrongcue_chain": cells["wrongcue_chain"]["exact"],
            "invalid_shape": n - cells["shape"]["valid"],
            "invalid_shape_definition": "all scheduled cases without a valid generated shape, including errors/unstarted",
            "execution_complete": p0_passed and failure is None and all(
                s["status"] in ("completed", "invalid_answer", "skipped")
                for row in rows for s in row["stages"].values()) and bool(rows),
            "failure": failure, "conditioning_filter": "none",
            "chain_gold": "original_destination_color_for_both_cues",
            "wrong_selector_rule": WRONG_SELECTOR_RULE,
            "stage_cross": "not_computed", "hypothesis_decision": "not_automated",
            "confirmation": "not_attempted", "max_new_tokens": GENERATION_BUDGET,
            "limitations": ["Engineering only; no locked-result regrading or scientific go",
                            "Invalid/truncated bare delivery is a failure, not semantic-zero evidence",
                            "Scores are whole fixed-candidate prefix events, not generated answers",
                            "Wrong cue is relative to prediction; it can coincide with the true destination",
                            "P0 and bounded work do not guarantee time or memory fit"]}


def evaluate_family(engine, root, family, output, *, record=None, progress=None):
    """Always atomically publish this family's ALL rows, including partial failures."""
    from vlm_pilot.io import atomic_json, safe_path

    output = safe_path(Path(output))
    if output.exists():
        raise FileExistsError("Family output exists; no model work, overwrite or resume.")
    record = _family_record(family) if record is None else record
    record["status"] = "running"
    try:
        _clear_features(engine)
        for case, row in zip(selected_cases(family), record["rows"], strict=True):
            evaluate_case(engine, root, family, case, record=row, progress=progress)
        return record
    except BaseException as error:
        record["failure"] = _error(error)
        raise
    finally:
        record["status"] = ("failed" if record["failure"] or any(
            row["status"] != "completed" for row in record["rows"]) else "completed")
        record["counts"] = summarize([record], p0_passed=True, failure=record["failure"])
        try:
            atomic_json(output, record, immutable=True)
        finally:
            _clear_features(engine)


def _device(value):
    if re.fullmatch(r"cpu|cuda:[0-9]+", value) is None:
        raise argparse.ArgumentTypeError("Choose cpu or explicit cuda:N; no device fallback.")
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--families", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--device", type=_device, default="cuda:0")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)  # --help/bad arguments never import models.
    from filelock import FileLock

    from vlm_pilot.data import load_dataset
    from vlm_pilot.io import atomic_json, identity, now, safe_path
    from vlm_pilot.plan import checkpoint_for, read_plan

    output, plan_path = safe_path(args.output), safe_path(args.plan)
    if output.exists():
        raise FileExistsError("New output directory required; no overwrite or resume.")
    plan = read_plan(plan_path, check_source=True)
    checkpoint = checkpoint_for(plan, args.checkpoint)
    if checkpoint.backend != "internvl":
        raise ValueError("This diagnostic requires the pinned native InternVL checkpoint.")
    families = [f for f in plan.dataset["families"] if f["split"] == "engineering"][:args.families]
    if len(families) != args.families:
        raise ValueError("Insufficient engineering families; no discovery/confirmation fallback.")
    root = safe_path(plan.spec.dataset)
    if identity(load_dataset(root, verify_images=False)) != plan.dataset_sha256:
        raise ValueError("Dataset/image bytes differ from the original locked plan.")
    records = [_family_record(f) for f in families]  # All cases, even after early failure.
    inference = plan.spec.inference  # NO model_copy, prefix/system/dtype override or relock.
    colors, shapes = _vocabulary()
    lock_path = Path(tempfile.gettempdir()) / ("interp-vlm-" + args.device.replace(":", "-") + ".lock")
    start, engine, p0_passed, failure = time.perf_counter(), None, False, None

    def progress(stage, **details):
        done = sum(row["status"] in ("completed", "failed") for f in records for row in f["rows"])
        total, elapsed = args.families * 2, time.perf_counter() - start
        print(json.dumps({"stage": stage, "device": args.device, "completed": done,
                          "total": total, "elapsed_seconds": elapsed,
                          "eta_seconds": elapsed * (total - done) / done if done else None,
                          "eta_kind": "observed_case_extrapolation_not_time_promise", **details}), flush=True)

    # EXACT native writer guard, acquired before importing/constructing a model.
    # No mem_get_info, CUDA context creation or guessed fit check precedes Engine.
    with FileLock(str(lock_path), timeout=0):
        output.mkdir(parents=True, exist_ok=False)
        metadata = {"kind": "engineering_only_generated_decomposition", "created_at": now(),
                    "plan": plan.model_dump(mode="json"),
                    "plan_file_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    "diagnostic_source": {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                    "source_check": True, "model": None, "runtime": None,
                    "original_inference": inference.model_dump(mode="json"),
                    "effective_inference": inference.model_dump(mode="json"),
                    "device_override": args.device, "conditioning": "common_answer_prefix_unchanged",
                    "max_new_tokens": GENERATION_BUDGET, "do_sample": False, "use_cache": False,
                    "sites": [], "case_hooks": False, "full_captures": False,
                    "p0": "usual_Engine_preflight_native_capture_self_all_known_candidates",
                    "requested_families": args.families, "family_ids": [f["family_id"] for f in families],
                    "factors": {"edge": 1, "palette": 0, "directions": [1, -1]},
                    "candidates": {"COLORS": list(colors), "SHAPES": list(shapes)},
                    "allow_downloads": False, "selection_filter": "none"}
        try:
            progress("cached_native_model_load")
            engine = cached_engine_class()(checkpoint, inference, device=args.device, allow_downloads=False)
            metadata.update(model=engine.model_profile, runtime=engine.runtime_profile)
            probe = dict(selected_cases(families[0])[0], candidates=list(colors + shapes))
            p0 = {"passed": False}
            try:
                progress("native_P0_all_known_candidates")
                p0 = engine.preflight(root, families[0], probe)
                _require_preflight(p0, colors + shapes)
                p0_passed = True
            except BaseException as error:
                p0["error"] = _error(error)
                raise
            finally:
                atomic_json(output / "preflight.json", _recordable(p0), immutable=True)
                _clear_features(engine)
            for index, (family, record) in enumerate(zip(families, records, strict=True), 1):
                evaluate_family(engine, root, family, output / f"family-{index:04d}.json",
                                record=record, progress=progress)
                progress("family_published", family_id=family["family_id"])
        except BaseException as error:
            failure = _error(error)
            raise
        finally:
            # Seal every scheduled family, even when model load/P0/interruption blocks it.
            for index, record in enumerate(records, 1):
                path = output / f"family-{index:04d}.json"
                if not path.exists():
                    record.update(status="failed", failure=failure)
                    record["counts"] = summarize([record], p0_passed=p0_passed, failure=failure)
                    atomic_json(path, record, immutable=True)
            if engine is not None:
                _clear_features(engine)
            metadata["elapsed_seconds"] = time.perf_counter() - start
            atomic_json(output / "metadata.json", metadata, immutable=True)
            atomic_json(output / "report.json", summarize(records, p0_passed=p0_passed,
                                                          failure=failure), immutable=True)
        report = summarize(records, p0_passed=p0_passed)
        progress("engineering_diagnostic_published", report=str(output), counts={
            k: report[k] for k in ("n", "direct_exact", "shape_correct", "label_chain",
                                  "wrongcue_chain", "invalid_shape")})
        return 0 if report["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())