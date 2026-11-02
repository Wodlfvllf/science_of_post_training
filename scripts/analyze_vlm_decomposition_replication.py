"""Read-only, model-free audit of two COMPLETE frozen C2 development endpoints.

API: analyze(mpo, gspo, output) validates before creating a NEW output directory,
then returns the JSON report. CLI: --mpo PATH --gspo PATH --output PATH. Import
and validation use stdlib only; statistics lazily load exact stats source/NumPy.
Only trusted, byte-matched adjacent sources are executed, never embedded code or
an Engine. Inputs, including the embedded dataset, need no original image paths.
Hashes certify consistency, NOT execution, token decoding, or image attestation.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO = Path(__file__).absolute().parents[1]
SEED, SAMPLES = 20261102, 10000
PINS = {"8b-mpo": ("OpenGVLab/InternVL3_5-8B-MPO", "bb33f08275aa566f3442c1f5002512ffb66d7952"),
        "8b-gspo": ("OpenGVLab/InternVL3_5-8B", "9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79")}
SOURCES = ("vlm_decomposition_replication.py", "vlm_generated_decomposition.py")
DOCUMENTS = {"metadata.json", "report.json", "preflight.json", "report-start.json"} | {
    f"{prefix}-{i:04d}.json" for prefix in ("family", "controls") for i in range(4, 24)}
__all__ = ["analyze", "main"]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, message):
    require(canonical(actual) == canonical(expected), message)


def path_safe(path):
    path = Path(path).expanduser().absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            "Symlink/traversal paths are forbidden")
    return path


def pairs(items):
    result = dict(items)
    require(len(items) == len(result), "Duplicate JSON key")
    return result


def decode(payload):
    result = json.loads(payload, object_pairs_hook=pairs)
    canonical(result)  # Reject NaN, infinity and overflowing JSON float literals.
    return result


def read(path, inventory):
    path = path_safe(path)
    before = path.stat()
    require(path.is_file() and 0 < before.st_size <= 32 * 1024**2, "Missing/oversized input")
    with path.open("rb") as stream:
        payload = stream.read(32 * 1024**2 + 1)
    after = path.stat()
    def signature(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    require(signature(before) == signature(after) and len(payload) == before.st_size,
            "Input changed while reading")  # Reading may legitimately update atime.
    entry = {"size_bytes": len(payload), "sha256": sha(payload)}
    require(str(path) not in inventory or inventory[str(path)] == entry, "Input changed between reads")
    inventory[str(path)] = entry
    return payload


def tools(inventory):
    """Use exact local stdlib-only helpers; inject only the literal vocabulary."""
    snapshots = {name: read(REPO / "scripts" / name, inventory) for name in SOURCES}
    data = read(REPO / "src/vlm_pilot/data.py", inventory)
    vocabulary = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in ast.parse(data).body if isinstance(node, ast.Assign)
                  and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in ("COLORS", "SHAPES")}
    modules = []
    for name in SOURCES:
        module = ModuleType("_audit_" + name[:-3])
        module.__file__ = str(REPO / "scripts" / name)
        exec(compile(snapshots[name], module.__file__, "exec"), module.__dict__)
        modules.append(module)
    runner, helper = modules
    helper._vocabulary = lambda: (vocabulary["COLORS"], vocabulary["SHAPES"])
    runner._helper = lambda: helper
    files = sorted((REPO / "src/vlm_pilot").rglob("*.py"))
    source_hash = sha(canonical([(p.relative_to(REPO / "src/vlm_pilot").as_posix(),
                                  sha(read(p, inventory))) for p in files]))
    return runner, helper, snapshots, source_hash, files


def metadata_check(docs, checkpoint, runner, helper, snapshots, source_hash):
    m, start, p0 = (docs[k] for k in ("metadata.json", "report-start.json", "preflight.json"))
    plan, protocol, model, runtime = (m[k] for k in ("plan", "protocol", "model", "runtime"))
    runner.validate_protocol(protocol, SimpleNamespace(**plan))
    equal(sha(canonical(plan["dataset"])), plan["dataset_sha256"], "Embedded dataset hash")
    spec = {k: v for k, v in plan["spec"].items() if k != "dataset"}
    equal(sha(canonical({"spec": spec, "dataset": plan["dataset_sha256"],
                         "source": plan["source_sha256"]})), plan["plan_id"], "Plan identity")
    equal(plan["source_sha256"], source_hash, "Locked package source changed")
    require(re.fullmatch(r"[0-9a-f]{64}", m["plan_file_sha256"]) is not None, "Plan file hash")
    equal(decode(m["protocol_json"].encode()), protocol, "Protocol text differs")
    equal(sha(m["protocol_json"].encode()), m["protocol_sha256"], "Protocol text hash")
    require(set(m["source_manifest"]) == set(snapshots), "Source inventory")
    for name, payload in snapshots.items():
        entry = m["source_manifest"][name]
        require(entry["text"].encode() == payload and entry["sha256"] == sha(payload), "Source hash/text differs")
    for key, expected in {"checkpoint": checkpoint, "kind": protocol["kind"], "source_check": True,
                          "failure": None, "allow_downloads": False, "sites": [], "case_hooks": False,
                          "full_captures": False, "conditioning": "common_answer_prefix_unchanged"}.items():
        equal(m[key], expected, "Metadata " + key)
    inference = plan["spec"]["inference"]
    equal(m["original_inference"], inference, "Original inference differs")
    equal(m["effective_inference"], inference, "Effective inference differs")
    require(inference["dtype"] == "float32" and inference["image_size"] == 448
            and bool(inference["answer_prefix"]), "Original FP32/single-tile prefix required")
    pins = [p for p in plan["spec"]["checkpoints"] if p["name"] == checkpoint]
    require(len(pins) == 1, "Unique pinned checkpoint required")
    pin = pins[0]
    require((pin["model_id"], pin["revision"]) == PINS[checkpoint]
            and pin["family"] == "8b" and pin["stage"] == checkpoint[3:]
            and pin["backend"] == "internvl", "Wrong pinned checkpoint definition")
    equal(model["checkpoint"], pin, "Model checkpoint differs from plan")
    equal(model["common_inference"], inference, "Model inference differs")
    require(model["scientific_fixture"] is False and model["dtype"] == "torch.float32"
            and model["image_tokens"] == 256 and model["tile_grid"] == [16, 16], "Native model provenance")
    require(model["weight_files"] and model["tokenizer_assets"] and model["model_code_sha256"], "Missing model provenance")
    hashes = [model[k] for k in ("lm_config_sha256", "tokenizer_sha256")]
    hashes += list(model["tokenizer_assets"].values()) + list(model["model_code_sha256"].values())
    for weight in model["weight_files"]:
        require(type(weight["size_bytes"]) is int and weight["size_bytes"] > 0, "Weight size")
        hashes.append(weight["sha256"])
    require(all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes), "Model provenance hashes")
    require(isinstance(runtime, dict) and runtime.get("packages") and runtime.get("python")
            and runtime.get("device") == m["device_override"], "Missing/runtime device provenance")
    helper._require_preflight(p0, sum(helper._vocabulary(), ()))
    equal(p0["runtime"], runtime, "Preflight runtime differs")
    equal(p0["model"], model, "Preflight model differs")
    require(p0["fixture_only"] is False, "Fixture preflight")
    initial = {k: v for k, v in m.items() if k not in ("elapsed_seconds", "failure")}
    initial.update(model=None, runtime=None)
    equal(start, initial, "Start/final metadata differs")
    families = runner.select_families(protocol, plan["dataset"])
    ids = [f["family_id"] for f in plan["dataset"]["families"]]
    require(len(ids) == len(set(ids)), "Overlapping embedded families")
    schedule = [{"index": f["index"], "family_id": f["family_id"],
                 "case_ids": [c["case_id"] for c in helper.selected_cases(f)]} for f in families]
    equal(m["schedule"], schedule, "Metadata schedule differs")
    return m, families


def stage_check(stage, case, family, budget, helper, inference, state):
    """Regrade raw delivery; recorded parser/exact/score flags are assertions only."""
    equal(stage["case"], case, "Stage case/predicted cue differs")
    require(stage["attempted"] is True and stage["errors"] == [], "Missing/failed operation")
    inputs, score, generation = (stage[k] for k in ("input", "score", "generation"))
    equal(inputs["case"], case, "Prepared case differs")
    require(inputs["question"] == case["question"] and inputs["profile"] == "common", "Prepared question/profile")
    prompt, ids, columns = (inputs[k] for k in ("rendered_prompt", "input_ids", "image_columns"))
    require(type(prompt) is str and prompt.endswith(inference["answer_prefix"])
            and prompt.count(case["question"]) == 1 and inference["system_message"] in prompt
            and prompt.count("<IMG_CONTEXT>") == 256, "Original rendered prompt/prefix required")
    skeleton = prompt.replace(case["question"], "{QUESTION}")
    equal(skeleton, state.setdefault("prompt_skeleton", skeleton), "Prompt template changed")
    require(type(ids) is list and ids and all(type(t) is int and t >= 0 for t in ids)
            and type(inputs["row"]) is int and inputs["row"] == len(ids) - 1
            and len(ids) + budget <= inference["max_context_tokens"], "Prepared token IDs/context")
    require(type(columns) is list and len(columns) == 256 and columns == sorted(set(columns))
            and all(type(i) is int and 0 <= i < len(ids) for i in columns)
            and len({ids[i] for i in columns}) == 1, "Image columns")
    image = [i for i in family["images"] if i["image_id"] == case["image_id"]]
    require(len(image) == 1, "Original image reference")
    expected_image = {k: image[0][k] for k in ("image_id", "relative_path", "sha256")}
    equal(inputs["image"], {**expected_image, "verified_before_prepare": True}, "Image provenance differs")
    helper._check_score(score, {}, inputs, case["candidates"])
    require(score["score_kind"] == "complete_candidate_sequence_logprob", "Candidate score kind")
    logs = score["candidate_token_logprobs"]
    require(set(logs) == set(case["candidates"]), "All candidate logprob sequences required")
    for candidate, values in logs.items():
        require(type(values) is list and len(values) == len(score["candidate_token_ids"][candidate])
                and all(type(v) in (int, float) and math.isfinite(v) and v <= 0 for v in values)
                and math.isclose(sum(values), score["scores"][candidate], rel_tol=1e-12, abs_tol=1e-12),
                "Candidate logprob sums differ")
    eos = generation["settings"]["eos_token_id"]
    require(type(eos) is int and eos >= 0, "EOS token must be an integer")
    equal(eos, state.setdefault("eos", eos), "EOS token changed")
    require(type(generation["settings"]["max_new_tokens"]) is int, "Integer generation budget required")
    helper._check_generation(SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=eos)), generation, budget)
    parsed = helper.strict_token_answer(generation, case["candidates"])
    expected = {"parsed_answer": parsed, "answer_valid": parsed is not None,
                "exact": parsed is not None and parsed == case["gold"],
                "score_exact": score["prediction"] == case["gold"],
                "status": "completed" if parsed is not None else "invalid_answer"}
    for key, value in expected.items():
        equal(stage[key], value, "Raw reconstruction differs: " + key)
    return parsed


def endpoint(docs, checkpoint, runner, helper, snapshots, source_hash):
    m, families = metadata_check(docs, checkpoint, runner, helper, snapshots, source_hash)
    bases, controls, state = [], [], {}
    colors, shapes = helper._vocabulary()
    for family in families:
        i = family["index"]
        base, control = (docs[f"{p}-{i:04d}.json"] for p in ("family", "controls"))
        for record in (base, control):
            require(record["family_id"] == family["family_id"] and record["failure"] is None
                    and len(record["rows"]) == 2, "Missing/failed family or direction")
        equal(control["index"], i, "Control family index")
        for case, row, cr in zip(helper.selected_cases(family), base["rows"], control["rows"], strict=True):
            require(case["target"] == (case["anchor"] + case["direction"]) % 3
                    and case["gold"] == family["palettes"][0][case["target"]]
                    and set(n["shape"] for n in family["nodes"]) == set(shapes), "Embedded geometry/gold")
            skeleton = helper._new_record(family, case)
            for record in (row, cr):
                for key in ("family_id", "case_id", "direction", "original_case"):
                    equal(record[key], skeleton[key], "Raw case schedule differs")
            equal(row["evaluation"], skeleton["evaluation"], "Shape/color evaluation geometry differs")
            require(set(row["stages"]) == set(helper.STAGES) and set(cr["stages"]) == set(runner.CONTROLS), "Stage schedule")
            direct = dict(deepcopy(case), case_id=case["case_id"] + "-generated-direct", candidates=list(colors))
            stage_check(row["stages"]["direct"], direct, family, 32, helper, m["original_inference"], state)
            prediction = stage_check(row["stages"]["shape"], helper.shape_case(family, case), family, 32,
                                     helper, m["original_inference"], state)
            for name, wrong in (("label_chain", False), ("wrongcue_chain", True)):
                stage = row["stages"][name]
                if prediction is None:
                    equal(stage, dict(helper._empty_stage(), status="skipped",
                          skip_reason="invalid_or_failed_generated_shape"), "Invalid shape must skip BOTH cues")
                else:
                    cue = helper.cue_case(case, prediction, wrong_selector=wrong)
                    stage_check(stage, cue, family, 32, helper, m["original_inference"], state)
            for name, cc in runner.control_cases(family, case).items():
                stage_check(cr["stages"][name], cc, family, runner.CONTROLS[name], helper, m["original_inference"], state)
            direct, repeat = row["stages"]["direct"], cr["stages"]["wasted_call_repeat32"]
            for key in ("input_ids", "rendered_prompt", "image_columns"):
                equal(direct["input"][key], repeat["input"][key], "Fresh repeat input differs")
            def signature(stage):
                return {k: stage["generation"][k] for k in ("text", "token_ids", "finish_reason", "settings")}
            equal(signature(direct), signature(repeat), "Fresh repeat raw generation differs")
            equal(cr["repeat_audit"], {"reference_case_id": direct["case"]["case_id"],
                  "shape_call_case_id": row["stages"]["shape"]["case"]["case_id"],
                  "conditioned_on_shape_output": False, "actual_generate": True,
                  "same_input_ids": True, "same_prompt": True, "same_raw_generation": True}, "Repeat audit differs")
            fixed = cr["stages"]["selectorless_fixedcue32"]
            cue = fixed["case"]["cued_shape"]
            color = family["palettes"][0][next(j for j, n in enumerate(family["nodes"]) if n["shape"] == cue)]
            equal(cr["evaluation"], {**skeleton["evaluation"],
                  "fixedcue_equals_destination": cue == skeleton["evaluation"]["destination_shape"],
                  "fixedcue_color": color, "fixedcue_lookup_exact": fixed["parsed_answer"] == color}, "Fixed cue evaluation differs")
            equal(cr["status"], "completed", "Incomplete control row")
            equal(row["status"], "failed" if any(s["status"] == "invalid_answer" for s in row["stages"].values())
                  else "completed", "Main row delivery status differs")
        equal(base["status"], "failed" if any(r["status"] != "completed" for r in base["rows"])
              else "completed", "Family delivery status differs")
        equal(base["counts"], helper.summarize([base], p0_passed=True), "Family counts differ")
        bases.append(base)
        controls.append(control)
    reconstructed = runner.summarize(bases, controls, checkpoint=checkpoint, p0_passed=True)
    equal(docs["report.json"], reconstructed, "Report counts/summary differ from raw reconstruction")
    require(reconstructed["execution_complete"], "Incomplete endpoint")
    return m, families, bases, controls, reconstructed, state


def interval(values, *, alpha=.05, bound=1):
    # No installed project/environment or stale module cache is required.
    source = REPO / "src/vlm_pilot/stats.py"
    stats = ModuleType("_c2_audit_stats")
    exec(compile(read(source, {}), str(source), "exec"), stats.__dict__)
    result = stats.family_difference_interval(values, bound=bound, samples=SAMPLES, seed=SEED, alpha=alpha)
    # Never present a degenerate bootstrap as certainty, even for discrete samples.
    if result["ci95"][0] == result["ci95"][1]:
        epsilon = bound * math.sqrt(2 * math.log(2 / alpha) / len(values))
        result.update(ci95=[max(-bound, result["difference"] - epsilon), min(bound, result["difference"] + epsilon)],
                      interval_method="conservative_hoeffding")
    result["interval"] = result.pop("ci95")  # The secondary interval is 99%, not 95%.
    result.update(samples=SAMPLES, seed=SEED)
    return result


def statistics(result):
    m, families, bases, controls, counts, _ = result
    names = ("direct_budget64", "composed_one_turn64", "selectorless_fixedcue32", "wrongcue_chain", "wasted_call_repeat32")
    differences = {name: {} for name in ("direct", *names)}
    pairs_out = {name: Counter(rescue=0, regression=0, both_correct=0, both_wrong=0) for name in differences}
    delivery, cue_following, rows_out = {}, Counter(), []
    decomposition = Counter({k: 0 for k in (
        "invalid_selector", "correct_selector_faithful_lookup", "correct_selector_unfaithful_lookup",
        "wrong_selector_faithful_lookup", "wrong_selector_unfaithful_lookup")})
    cross = Counter({f"shape_{s}_chain_{c}": 0 for s in ("correct", "wrong") for c in ("correct", "wrong")})
    for family, base, control in zip(families, bases, controls, strict=True):
        family_values = {name: [] for name in differences}
        for row, cr in zip(base["rows"], control["rows"], strict=True):
            stages = {**row["stages"], **cr["stages"]}
            chain, shape = (stages[k]["exact"] for k in ("label_chain", "shape"))
            cross[f"shape_{'correct' if shape else 'wrong'}_chain_{'correct' if chain else 'wrong'}"] += 1
            selector = stages["shape"]["parsed_answer"]
            if selector is None:
                decomposition["invalid_selector"] += 1
            else:
                selected_node = next(j for j, node in enumerate(family["nodes"]) if node["shape"] == selector)
                indicated_color = family["palettes"][0][selected_node]
                faithful = stages["label_chain"]["parsed_answer"] == indicated_color
                decomposition[f"{'correct' if shape else 'wrong'}_selector_{'faithful' if faithful else 'unfaithful'}_lookup"] += 1
            for name in differences:
                other = stages[name]["exact"]
                family_values[name].append(int(chain) - int(other))
                pairs_out[name]["both_correct" if chain and other else "rescue" if chain else "regression" if other else "both_wrong"] += 1
            for name, stage in stages.items():
                item = delivery.setdefault(name, {"n": 40, "reasons": Counter(), "generated_lengths": [], "input_lengths": []})
                generation = stage["generation"]
                reason = "skipped_invalid_shape" if generation is None else "truncated_length" if generation["finish_reason"] == "length" else "valid" if stage["answer_valid"] else "eos_nonbare_or_out_of_vocabulary"
                item["reasons"][reason] += 1
                item["generated_lengths"].append(len(generation["token_ids"]) if generation else 0)
                item["input_lengths"].append(len(stage["input"]["input_ids"]) if generation else 0)
                cued_shape = cued_color = follows_cue = None
                cue_is_destination = None
                if name in ("label_chain", "wrongcue_chain", "selectorless_fixedcue32") and generation:
                    cue = stage["case"]["cued_shape"]
                    node = next(j for j, n in enumerate(family["nodes"]) if n["shape"] == cue)
                    cued_shape, cued_color = cue, family["palettes"][0][node]
                    follows_cue = stage["parsed_answer"] == cued_color
                    cue_is_destination = cue == row["evaluation"]["destination_shape"]
                    cue_following[name] += follows_cue
                rows_out.append({"checkpoint": m["checkpoint"], "family_id": family["family_id"],
                    "family_index": family["index"], "case_id": row["case_id"], "direction": row["direction"],
                    "stage": name, "parsed_answer": stage["parsed_answer"], "exact": stage["exact"],
                    "delivery": reason, "generated_tokens": item["generated_lengths"][-1], "input_tokens": item["input_lengths"][-1],
                    "generated_selector": selector, "selector_correct": shape,
                    "cued_shape": cued_shape, "cued_color": cued_color,
                    "cue_is_destination": cue_is_destination, "lookup_follows_actual_cue": follows_cue})
        for name, values in family_values.items():
            differences[name][family["family_id"]] = sum(values) / 2
    for item in delivery.values():
        item.update(generated_tokens_total=sum(item["generated_lengths"]), input_tokens_total=sum(item["input_lengths"]))
    return {"counts": counts, "chain_minus_direct": interval(differences["direct"]),
            "secondary_exploratory": {k: interval(differences[k], alpha=.01) for k in names},
            "paired_cases": pairs_out, "shape_chain_cross": cross,
            "selector_lookup_partition": {"n": sum(decomposition.values()), "counts": decomposition,
                "interpretation": "External generated pipeline only; not a native internal selector measurement"},
            "delivery_and_cost": delivery,
            "successful_cue_following_counts": {k: {"exact": cue_following[k], "n": 40}
                for k in ("label_chain", "wrongcue_chain", "selectorless_fixedcue32")}}, rows_out


def analyze(mpo, gspo, output):
    """Validate complete inputs without writes; publish no-clobber report and seal."""
    roots, output = [path_safe(p) for p in (mpo, gspo)], path_safe(output)
    require(roots[0] != roots[1] and not any(a in b.parents or b in a.parents
            for a, b in ((roots[0], roots[1]), *((r, output) for r in roots))), "Overlapping input/output paths")
    if output.exists():
        raise FileExistsError("Output exists; no overwrite/resume")
    inventory, input_inodes = {}, set()
    runner, helper, snapshots, source_hash, source_files = tools(inventory)
    read(Path(__file__), inventory)
    endpoints = []
    for root, checkpoint in zip(roots, PINS, strict=True):
        require(root.is_dir() and {p.name for p in root.iterdir()} == DOCUMENTS, "Require exactly 40 raw documents plus metadata/report/preflight/report-start; no aliases")
        docs = {}
        for name in sorted(DOCUMENTS):
            path = path_safe(root / name)
            inode = (path.stat().st_dev, path.stat().st_ino)
            require(inode not in input_inodes, "Aliased input file")
            input_inodes.add(inode)
            docs[name] = decode(read(path, inventory))
        endpoints.append(endpoint(docs, checkpoint, runner, helper, snapshots, source_hash))
    before, after = (e[0] for e in endpoints)
    for key in ("plan", "plan_file_sha256", "protocol", "protocol_json", "protocol_sha256",
                "original_inference", "effective_inference", "runtime", "device_override"):
        equal(before[key], after[key], "Endpoint mismatch: " + key)
    equal(endpoints[0][-1], endpoints[1][-1], "Endpoint prompt/EOS mismatch")
    require(before["model"]["checkpoint"] != after["model"]["checkpoint"], "Checkpoint definitions must differ")
    summaries, csv_rows = {}, []
    for e in endpoints:
        summaries[e[0]["checkpoint"]], rows = statistics(e)
        csv_rows.extend(rows)
    primary = deepcopy(summaries["8b-gspo"]["chain_minus_direct"])
    primary.update(material_lead=primary["difference"] >= .15 and primary["interval"][0] > 0,
                   gate="descriptive_resource_gate_not_automatic_paper_ready", threshold=.15)
    contrasts = [summaries[k]["chain_minus_direct"]["family_differences"] for k in PINS]
    interaction = interval({k: contrasts[1][k] - contrasts[0][k] for k in contrasts[0]}, bound=2)
    report = {"kind": "c2_development_replication_model_free_audit_v1", "primary_gspo": primary,
              "endpoints": summaries, "gspo_minus_mpo_chain_direct_interaction": interaction,
              "multiplicity": "GSPO five exploratory secondary 99% intervals; Bonferroni .05/5, no sign-flip p-values; MPO descriptive",
              "all_three_cue_content_intervention": "untested", "scientific_mechanism": "not_established",
              "optimizer_specific_claim": False, "confirmation": "not_attempted",
              "caveats": ["Development families 4..23, not globally untouched or confirmation data.",
                  "Strict delivery is not semantic reasoning accuracy; invalid cases remain in all denominators.",
                  "Actual token lengths are reported; budgets do not match compute/FLOPs or account for all scoring cost.",
                  "No native mechanism, shape synergy mechanism, optimizer causality or publication readiness follows.",
                  "Hashes are consistency evidence, not independent execution/image/token-decoding attestation; original plan-file hash is recorded, not independently re-read."],
              "provenance": {"plan_id": before["plan"]["plan_id"], "protocol": before["protocol"],
                  "source_sha256": source_hash, "runtime": before["runtime"], "runtime_ignored_fields": [],
                  "inference": before["original_inference"], "models": {e[0]["checkpoint"]: e[0]["model"] for e in endpoints},
                  "inputs_and_sources": inventory, "inventory_sha256": sha(canonical(inventory))}}
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    markdown = ("# C2 development replication audit\n\n"
        f"GSPO chain − direct: {primary['difference']:+.3f}; 95% family interval {primary['interval']}.\n\n"
        f"Descriptive material gate: {primary['material_lead']}; not an automatic paper-ready decision.\n\n"
        "| Endpoint/stage | Exact / 40 | Invalid | Skipped | Generated tokens |\n|---|---:|---:|---:|---:|\n")
    for checkpoint, summary in summaries.items():
        for name, cell in summary["counts"]["stages"].items():
            markdown += f"| {checkpoint}/{name} | {cell['exact']} / 40 | {cell['invalid_answer']} | {cell['skipped']} | {summary['delivery_and_cost'][name]['generated_tokens_total']} |\n"
    markdown += "\n" + "\n".join("- " + c for c in report["caveats"]) + "\n\nAll-three-cue intervention: untested. Secondary intervals are exploratory (99%, five GSPO contrasts). MPO is descriptive; stage interaction is not an optimizer claim.\n"

    def unchanged():
        require(sorted((REPO / "src/vlm_pilot").rglob("*.py")) == source_files, "Package source inventory changed")
        for root in roots:
            require({p.name for p in root.iterdir()} == DOCUMENTS, "Input file list changed")
        check = {}
        for path in inventory:
            read(path, check)
        equal(check, inventory, "Input/source changed during analysis")

    unchanged()  # No input locks, caches, temporary files or other read-side writes.
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"report.json": canonical(report), "cases.csv": text.getvalue().encode(), "REPORT.md": markdown.encode()}
    for name, payload in artifacts.items():
        with (output / name).open("xb") as stream:
            stream.write(payload)
    require({p.name for p in output.iterdir()} == set(artifacts), "Output inventory changed")
    for name, payload in artifacts.items():
        require(read(output / name, {}) == payload, "Output bytes changed before sealing")
    unchanged()
    seal = {"input_inventory_sha256": report["provenance"]["inventory_sha256"],
            "before_after_equal": True, "files": {k: {"sha256": sha(v), "size_bytes": len(v)} for k, v in artifacts.items()}}
    with (output / "integrity.json").open("xb") as stream:
        stream.write(canonical(seal))  # Published LAST; a failed write/mutation has no success seal.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("mpo", "gspo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.mpo, args.gspo, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())