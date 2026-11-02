"""Frozen C2 DEVELOPMENT replication, not confirmation or a scientific go gate.

CLI requires a JSON protocol equal to FIXED_PROTOCOL plus the locked plan_id and
dataset_sha256. Only engineering indices 4..23, two original e=+1,p=0 directions,
are enrolled. Import/argument parsing are stdlib-only; execution is cached-only,
native/unhooked, original inference, one image per family, no training or probes.
Each family retains the adjacent C2 helper's four raw stages and four independent
controls. Invalid delivery counts as incorrect, not an operation failure. Any
operation failure stops work and finally seals every planned denominator.
Outputs must be NEW runs/vlm-pilot/report-* directories: existing backup rules
include their JSON and metadata. Source snapshots are embedded because the old
backup script's explicit script allowlist does not include this new script.
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
CONTROLS = {"direct_budget64": 64, "composed_one_turn64": 64,
            "wasted_call_repeat32": 32, "selectorless_fixedcue32": 32}
FIXED_RULE = "SHAPES[(family.index+direction_sign)%len(SHAPES)]"
FIXED_PROTOCOL = {
    "schema_version": 1, "kind": "c2_development_replication_v1",
    "split": "engineering", "family_offset": 4, "family_count": 20,
    "excluded_family_indices": [0, 1, 2, 3],
    "factors": {"edge": 1, "palette": 0, "directions": [1, -1]},
    "budgets": {"original": 32, **CONTROLS}, "checkpoints": ["8b-mpo", "8b-gspo"],
    "primary": "primary_chain_minus_direct_8b_gspo", "inference": "original_unchanged",
    "selectorless_rule": FIXED_RULE, "scope": "development_replication_not_confirmation",
    "prohibited": ["cpt", "probes", "training", "pilot_validation", "confirmation"],
}
_HELPER = None
__all__ = ["validate_protocol", "select_families", "control_cases", "evaluate_controls",
           "summarize", "main"]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _safe(path):
    path = Path(path).expanduser().absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlinks and traversal are forbidden.")
    return path


def _source(path, maximum=1024**2):
    path = _safe(path)
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("Missing or oversized source/protocol.")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if not 0 < len(payload) <= maximum:
        raise ValueError("Source/protocol size changed.")
    return payload


def _unique(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate protocol JSON key.")
    return result


def _helper():
    """Import ONLY adjacent exact bytes, lazily; refuse subsequent source drift."""
    global _HELPER
    path = _safe(Path(__file__).with_name("vlm_generated_decomposition.py"))
    payload = _source(path)
    key = (str(path), _sha(payload))
    if _HELPER is None:
        spec = importlib.util.spec_from_file_location("_c2_replication_adjacent", path)
        module = importlib.util.module_from_spec(spec)
        exec(compile(payload, str(path), "exec"), module.__dict__)
        _HELPER = (key, module)
    if _HELPER[0] != key:
        raise RuntimeError("Adjacent C2 helper path/hash changed.")
    return _HELPER[1]


def _fixed(protocol):
    if type(protocol) is not dict:
        raise ValueError("Protocol must be a JSON object.")
    pins = {k: protocol.get(k) for k in ("plan_id", "dataset_sha256")}
    if any(type(v) is not str or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in pins.values()):
        raise ValueError("Protocol requires full plan and dataset hashes.")
    # Canonical equality rejects extras, booleans-as-integers and floating budgets.
    if _canonical(protocol) != _canonical({**FIXED_PROTOCOL, **pins}):
        raise ValueError("Protocol differs from the frozen development replication.")


def validate_protocol(protocol, plan):
    """Reject every varied/extra value, including seeds, splits and model overrides."""
    _fixed(protocol)
    if protocol["plan_id"] != plan.plan_id or protocol["dataset_sha256"] != plan.dataset_sha256:
        raise ValueError("Protocol plan_id/dataset_sha256 must match the locked plan.")
    return deepcopy(protocol)


def select_families(protocol, dataset):
    """Select EXACT indices 4..23, never rank/filter outcomes or fall back splits."""
    _fixed(protocol)
    engineering = [f for f in dataset["families"] if f["split"] == "engineering"]
    if ([f["index"] for f in engineering] != list(range(24))
            or any(type(f["index"]) is not int for f in engineering)
            or len({f["family_id"] for f in engineering}) != 24):
        raise ValueError("Require the original ordered 24 engineering families.")
    families, seen = engineering[4:24], set()
    for family in families:
        for case in _helper().selected_cases(family):
            if (case["family_id"] != family["family_id"] or case["split"] != "engineering"
                    or case["case_id"] in seen):
                raise ValueError("Duplicate or mismatched scheduled case IDs.")
            seen.add(case["case_id"])
    return families


def control_cases(family, case):
    """No target/palette/gold lookup: anchor, direction and fixed index cue ONLY.

    Gold is merely copied for later evaluation. Fixed cue may equal the anchor or
    destination; never reject/reselect it. Repeat uses the already executed C2
    shape inquiry as its wasted call, but MUST perform a NEW direct generation.
    """
    colors, shapes = _helper()._vocabulary()
    if type(case["direction"]) is not int or case["direction"] not in (1, -1):
        raise ValueError("Require an original signed direction.")
    anchor = family["nodes"][case["anchor"]]["shape"]
    traversal = "forward" if case["direction"] == 1 else "backward"
    cue = shapes[(family["index"] + case["direction"]) % len(shapes)]
    questions = {
        "direct_budget64": case["question"], "wasted_call_repeat32": case["question"],
        "composed_one_turn64": (f"Starting at the {anchor}, follow one arrow {traversal}. "
            "First determine which shape is the destination, then report the color of its chip. "
            "Reply with one color name."),
        "selectorless_fixedcue32": f"What color is the chip at the {cue}? Reply with one color name.",
    }
    result = {name: dict(deepcopy(case), case_id=f"{case['case_id']}-{name}",
                        question=questions[name], candidates=list(colors), diagnostic=name)
              for name in CONTROLS}
    result["selectorless_fixedcue32"].update(cued_shape=cue, selector_rule=FIXED_RULE)
    return result


def _control_record(family):
    h = _helper()
    return {"family_id": family["family_id"], "index": family["index"], "failure": None,
            "rows": [dict(h._new_record(family, c), stages={k: h._empty_stage() for k in CONTROLS})
                     for c in h.selected_cases(family)]}


class _StopOperations(BaseException):
    """Escape the original helper's intentional per-operation Exception capture."""


def _has_errors(record):
    return any(s["errors"] for r in record["rows"] for s in r["stages"].values())


def _control_stage(engine, root, family, case, stage, budget, progress, name):
    """Own 32/64 runner; all COLORS scored, strict raw native EOS generation."""
    h, start = _helper(), time.perf_counter()
    colors, _ = h._vocabulary()
    if name not in CONTROLS or type(budget) is not int or budget != CONTROLS[name]:
        raise ValueError("Control budgets are frozen.")
    stage.update(status="running", attempted=True, case=deepcopy(case))
    try:
        progress(name, operation="prepare", case_id=case["case_id"])
        image = h._image_reference(root, family, case)
        prepared = engine.prepare(root, family, case, profile="common")
        stage["input"] = h._prepared_input(engine, prepared, image)
        for operation in ("score", "generate"):
            try:
                progress(name, operation=operation, case_id=case["case_id"])
                if operation == "score":
                    result, captures = engine.score(prepared)
                    stage["score"] = h._recordable(result)
                    h._check_score(result, captures, stage["input"], colors)
                    stage["score_exact"] = result["prediction"] == case["gold"]
                else:
                    result = engine.generate(prepared, budget)
                    stage["generation"] = h._recordable(result)
                    h._check_generation(engine, result, budget)
                    answer = h.strict_token_answer(result, colors)
                    stage.update(parsed_answer=answer, answer_valid=answer is not None,
                                 exact=answer is not None and answer == case["gold"])
            except Exception as error:
                stage["errors"].append(dict(h._error(error), operation=operation))
                break  # No later native forward after an operation error.
    except BaseException as error:
        stage["errors"].append(dict(h._error(error), operation="prepare_or_interrupted"))
        raise
    finally:
        stage["elapsed_seconds"] = time.perf_counter() - start
        stage["status"] = ("failed" if stage["errors"] else "completed"
                           if stage["answer_valid"] else "invalid_answer")
    if stage["errors"]:
        raise RuntimeError("Control operation failed; no later controls/families permitted.")


def evaluate_controls(engine, root, family, base, output, *, record=None, progress=None):
    """Publish full two-case partial controls in finally; never reuse direct output."""
    from vlm_pilot.io import atomic_json, safe_path

    h, output = _helper(), safe_path(Path(output))
    if output.exists():
        raise FileExistsError("New control file required; no overwrite/resume.")
    record = _control_record(family) if record is None else record
    progress = progress or (lambda *a, **k: None)
    try:
        if _has_errors(base):
            raise RuntimeError("C2 operation failure blocks controls.")
        for case, row, original in zip(h.selected_cases(family), record["rows"], base["rows"], strict=True):
            if original["case_id"] != case["case_id"] or not original["stages"]["shape"]["attempted"]:
                raise ValueError("Controls require the same case's actual prior shape inquiry.")
            row["status"] = "running"
            for name, control in control_cases(family, case).items():
                _control_stage(engine, root, family, control, row["stages"][name], CONTROLS[name], progress, name)
            direct, repeat = original["stages"]["direct"], row["stages"]["wasted_call_repeat32"]
            def signature(stage):
                return {k: stage["generation"].get(k) for k in ("text", "token_ids", "finish_reason", "settings")}
            row["repeat_audit"] = {"reference_case_id": direct["case"]["case_id"],
                "shape_call_case_id": original["stages"]["shape"]["case"]["case_id"],
                "conditioned_on_shape_output": False, "actual_generate": True,
                "same_input_ids": direct["input"]["input_ids"] == repeat["input"]["input_ids"],
                "same_prompt": direct["input"]["rendered_prompt"] == repeat["input"]["rendered_prompt"],
                "same_raw_generation": signature(direct) == signature(repeat)}
            cue = row["stages"]["selectorless_fixedcue32"]["case"]["cued_shape"]
            row["evaluation"]["fixedcue_equals_destination"] = cue == row["evaluation"]["destination_shape"]
            # Evaluation ONLY, after all generation: distinguish lookup from selection.
            node = next(i for i, n in enumerate(family["nodes"]) if n["shape"] == cue)
            row["evaluation"]["fixedcue_color"] = family["palettes"][0][node]
            row["evaluation"]["fixedcue_lookup_exact"] = row["stages"]["selectorless_fixedcue32"]["parsed_answer"] == family["palettes"][0][node]
            row["status"] = "completed"  # Invalid delivery is counted, not an operation error.
        return record
    except BaseException as error:
        record["failure"] = h._error(error)
        raise
    finally:
        atomic_json(output, h._recordable(record), immutable=True)
        h._clear_features(engine)


def summarize(records, controls, *, checkpoint, p0_passed=False, failure=None):
    """Counts and matched two-case family values only; no automatic CI/go decision."""
    h, cells, paired = _helper(), {}, []
    if len(records) != 20 or len(controls) != 20:
        raise ValueError("All twenty planned families are denominators, even unstarted.")
    stages = (*h.STAGES, *CONTROLS)
    merged = []
    for index, (base, control) in enumerate(zip(records, controls, strict=True), 4):
        if base["family_id"] != control["family_id"] or control["index"] != index:
            raise ValueError("Family pairing/order changed.")
        if len(base["rows"]) != 2 or len(control["rows"]) != 2:
            raise ValueError("Both original directions must remain scheduled.")
        rows = []
        for direction, a, b in zip((1, -1), base["rows"], control["rows"], strict=True):
            if a["case_id"] != b["case_id"] or a["direction"] != b["direction"] or a["direction"] != direction:
                raise ValueError("Case IDs/directions changed; no skipping allowed.")
            rows.append({**a["stages"], **b["stages"]})
        merged.extend(rows)
        values = {k: sum(r[k]["exact"] is True for r in rows) / 2 for k in stages}
        paired.append({"family_index": index, "family_id": base["family_id"],
            "case_ids": [r["case_id"] for r in base["rows"]], "values": values,
            "chain_minus_direct": values["label_chain"] - values["direct"],
            "chain_minus_controls": {k: values["label_chain"] - values[k] for k in CONTROLS}})
    for name in stages:
        items = [r[name] for r in merged]
        cells[name] = {"n": 40, **{k: sum(s[k] is True for s in items)
            for k in ("attempted", "answer_valid", "exact", "score_exact")},
            "valid": sum(s["answer_valid"] is True for s in items),
            "accuracy": sum(s["exact"] is True for s in items) / 40,
            **{k: sum(s["status"] == k for s in items) for k in
               ("completed", "invalid_answer", "skipped", "failed", "running", "not_started")},
            "operation_errors": sum(len(s["errors"]) for s in items)}
    cross = {k: sum(r["shape"]["exact"] is shape and r["label_chain"]["exact"] is chain for r in merged)
             for k, shape, chain in (("shape_correct_chain_correct", True, True),
                 ("shape_wrong_chain_correct", False, True), ("shape_correct_chain_wrong", True, False))}
    cross.update(chain_correct_direct_wrong=sum(r["label_chain"]["exact"] and not r["direct"]["exact"] for r in merged),
                 direct_correct_chain_wrong=sum(r["direct"]["exact"] and not r["label_chain"]["exact"] for r in merged))
    return {"kind": FIXED_PROTOCOL["kind"], "n": 40, "n_case": 40, "n_family": 20, "checkpoint": checkpoint,
        "stages": cells, "paired_families": paired, "comparisons": cross, "failure": failure,
        "direct_exact": cells["direct"]["exact"], "shape_correct": cells["shape"]["exact"],
        "label_chain": cells["label_chain"]["exact"], "wrongcue_chain": cells["wrongcue_chain"]["exact"],
        "invalid_shape": 40 - cells["shape"]["valid"],
        "invalid_shape_definition": "all planned cases without a valid generated shape, including unstarted/errors",
        "primary": FIXED_PROTOCOL["primary"], "primary_value": sum(p["chain_minus_direct"] for p in paired) / 20
        if checkpoint == "8b-gspo" else None, "p0_passed": p0_passed,
        "chain_minus_controls": {k: (cells["label_chain"]["exact"] - cells[k]["exact"]) / 40 for k in CONTROLS},
        "fixedcue_matches_destination": sum(r["evaluation"].get("fixedcue_equals_destination") is True
            for f in controls for r in f["rows"]), "fixedcue_nominal_selector_chance": 1 / 3,
        "fixedcue_lookup_exact": sum(r["evaluation"].get("fixedcue_lookup_exact") is True for f in controls for r in f["rows"]),
        "execution_complete": p0_passed and failure is None and all(
            s["status"] in ("completed", "invalid_answer", "skipped") and not s["errors"]
            for r in merged for s in r.values()), "confirmation": "not_attempted",
        "hypothesis_decision": "not_automated", "statistics": "counts_and_paired_family_values_only"}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "protocol", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, choices=("8b-mpo", "8b-gspo"))
    parser.add_argument("--device", required=True, type=_device)
    return parser


def _device(value):
    if re.fullmatch(r"cpu|cuda:[0-9]+", value) is None:
        raise argparse.ArgumentTypeError("Choose cpu or explicit cuda:N; no fallback.")
    return value


def main(argv=None):
    args = _parser().parse_args(argv)
    from filelock import FileLock

    from vlm_pilot.data import load_dataset
    from vlm_pilot.io import atomic_json, identity, now
    from vlm_pilot.plan import checkpoint_for, read_plan, source_digest

    output, plan_path, protocol_path = map(_safe, (args.output, args.plan, args.protocol))
    if output.parent != REPO / "runs/vlm-pilot" or not re.fullmatch(r"report-[A-Za-z0-9_-]+", output.name):
        raise ValueError("Output must be a new runs/vlm-pilot/report-* backup-eligible directory.")
    if output.exists():
        raise FileExistsError("New output required; no overwrite or resume.")
    plan_sha = _sha(_source(plan_path, 32 * 1024**2))
    plan = read_plan(plan_path, check_source=True)
    protocol_bytes = _source(protocol_path)
    protocol = validate_protocol(json.loads(protocol_bytes, object_pairs_hook=_unique), plan)
    checkpoint = checkpoint_for(plan, args.checkpoint)
    if checkpoint.backend != "internvl" or checkpoint.family != "8b" or checkpoint.stage != args.checkpoint[3:]:
        raise ValueError("Only the plan's pinned native 8B MPO/GSPO checkpoints are allowed.")
    families, h = select_families(protocol, plan.dataset), _helper()
    root, inference = _safe(plan.spec.dataset), plan.spec.inference
    if identity(load_dataset(root, verify_images=False)) != plan.dataset_sha256:
        raise ValueError("Dataset differs from locked plan.")
    records, controls = [h._family_record(f) for f in families], [_control_record(f) for f in families]
    sources = {p.name: {"path": str(_safe(p)), "sha256": _sha(payload := _source(p)), "text": payload.decode()}
               for p in (Path(__file__), Path(h.__file__))}
    metadata = {"kind": protocol["kind"], "created_at": now(), "checkpoint": args.checkpoint, "plan": plan.model_dump(mode="json"),
        "plan_file_sha256": plan_sha, "protocol": protocol, "protocol_json": protocol_bytes.decode(),
        "protocol_sha256": _sha(protocol_bytes), "source_manifest": sources,
        "source_check": True, "original_inference": inference.model_dump(mode="json"),
        "effective_inference": inference.model_dump(mode="json"), "device_override": args.device,
        "model": None, "runtime": None, "allow_downloads": False, "sites": [], "case_hooks": False,
        "full_captures": False, "conditioning": "common_answer_prefix_unchanged",
        "schedule": [{"index": f["index"], "family_id": f["family_id"],
                      "case_ids": [c["case_id"] for c in h.selected_cases(f)]} for f in families]}
    start, engine, passed, failure = time.perf_counter(), None, False, None

    def unchanged():
        if (source_digest() != plan.source_sha256 or _sha(_source(plan_path, 32 * 1024**2)) != metadata["plan_file_sha256"]
                or _sha(_source(protocol_path)) != metadata["protocol_sha256"]
                or inference.model_dump(mode="json") != metadata["original_inference"]
                or _HELPER[0][1] != sources[Path(h.__file__).name]["sha256"]
                or any(_sha(_source(s["path"])) != s["sha256"] for s in sources.values())):
            raise RuntimeError("Plan/protocol/source/inference changed during execution.")

    def progress(stage, **details):
        done = sum(r["status"] == "completed" for f in controls for r in f["rows"])
        elapsed = time.perf_counter() - start
        print(json.dumps({"stage": stage, "device": args.device, "completed": done, "total": 40,
            "elapsed_seconds": elapsed, "eta_seconds": elapsed * (40 - done) / done if done else None,
            "eta_kind": "observed_case_extrapolation_not_time_promise", **details}), flush=True)

    lock = Path(tempfile.gettempdir()) / ("interp-vlm-" + args.device.replace(":", "-") + ".lock")
    with FileLock(str(lock), timeout=0):  # No CUDA context or memory query before Engine.
        output.mkdir(parents=True, exist_ok=False)
        try:
            atomic_json(output / "report-start.json", metadata, immutable=True)
            unchanged()
            progress("cached_native_model_load")
            engine = h.cached_engine_class()(checkpoint, inference, device=args.device, allow_downloads=False)
            metadata.update(model=engine.model_profile, runtime=engine.runtime_profile)
            colors, shapes = h._vocabulary()
            p0 = {"passed": False}
            try:
                progress("native_P0_all_known_colors_shapes")
                p0 = engine.preflight(root, families[0], dict(h.selected_cases(families[0])[0], candidates=list(colors + shapes)))
                h._require_preflight(p0, colors + shapes)
                unchanged()
                passed = True
            except BaseException as error:
                p0 = {"raw_preflight": h._recordable(p0), "passed": False, "error": h._error(error)}
                raise
            finally:
                atomic_json(output / "preflight.json", h._recordable(p0), immutable=True)
                h._clear_features(engine)
            for family, base, control in zip(families, records, controls, strict=True):
                def guarded_progress(stage, base=base, **details):
                    if _has_errors(base):
                        raise _StopOperations("C2 operation failure; no further native forwards.")
                    progress(stage, **details)
                unchanged()
                try:
                    h.evaluate_family(engine, root, family, output / f"family-{family['index']:04d}.json",
                                      record=base, progress=guarded_progress)
                except _StopOperations as error:
                    raise RuntimeError("C2 operation failure; stopped.") from error
                if _has_errors(base):
                    raise RuntimeError("C2 operation failure; stopped.")
                unchanged()
                evaluate_controls(engine, root, family, base, output / f"controls-{family['index']:04d}.json",
                                  record=control, progress=progress)
                progress("family_published", family_id=family["family_id"])
            unchanged()
        except BaseException as error:
            failure = h._error(error)
            raise
        finally:
            for family, base, control in zip(families, records, controls, strict=True):
                for prefix, record in (("family", base), ("controls", control)):
                    path = output / f"{prefix}-{family['index']:04d}.json"
                    if not path.exists():
                        record["failure"] = failure
                        atomic_json(path, h._recordable(record), immutable=True)
            if engine is not None:
                h._clear_features(engine)
            metadata.update(elapsed_seconds=time.perf_counter() - start, failure=failure)
            atomic_json(output / "metadata.json", h._recordable(metadata), immutable=True)
            report = summarize(records, controls, checkpoint=args.checkpoint, p0_passed=passed, failure=failure)
            atomic_json(output / "report.json", h._recordable(report), immutable=True)
        progress("development_replication_published", counts={k: v["exact"] for k, v in report["stages"].items()})
        return 0 if report["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())