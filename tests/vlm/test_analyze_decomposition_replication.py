"""Synthetic JSON contracts ONLY: no GPU, Engine, network, subprocess or real run.

Pinned names/model manifests below are fabricated consistency fixtures, explicitly
NOT evidence that a checkpoint executed. Geometry comes from data._family; prompts
and token IDs are tiny semantic toys, not a native tokenizer or image attestation.
"""
from __future__ import annotations

import builtins
import csv
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from vlm_pilot.data import _family

PATH = Path(__file__).absolute().parents[2] / "scripts/analyze_vlm_decomposition_replication.py"


def load():
    spec = importlib.util.spec_from_file_location("analyze_c2_test", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load()


def write(path, value):
    path.write_bytes(audit.canonical(value))


def edit(root, name, mutate):
    path = root / name
    value = json.loads(path.read_bytes())
    mutate(value)
    write(path, value)


@pytest.fixture
def reports(tmp_path):
    runner, h, snapshots, source_hash, _ = audit.tools({})
    colors, shapes = h._vocabulary()
    families = [_family(31, "engineering", i) for i in range(24)]
    for family in families:
        for image in family["images"]:
            image["sha256"] = hashlib.sha256(image["image_id"].encode()).hexdigest()
    dataset = {"families": families, "synthetic_test_only": True}
    inference = {"dtype": "float32", "device": "cuda:0", "image_size": 448,
                 "max_context_tokens": 1024, "system_message": "SYNTHETIC TEST ONLY",
                 "answer_prefix": "Answer:", "atol": .0001, "rtol": .0001}
    pins = [{"name": name, "family": "8b", "stage": name[3:], "backend": "internvl",
             "model_id": model_id, "revision": revision, "fixture_seed": 17}
            for name, (model_id, revision) in audit.PINS.items()]
    spec = {"dataset": "/synthetic-test-only/not-read", "inference": inference,
            "checkpoints": pins, "seed": 31, "name": "synthetic-test-only", "max_new_tokens": 64}
    data_hash = audit.sha(audit.canonical(dataset))
    plan_id = audit.sha(audit.canonical({"spec": {k: v for k, v in spec.items() if k != "dataset"},
                                       "dataset": data_hash, "source": source_hash}))
    plan = {"schema_version": 1, "plan_id": plan_id, "created_at": "test-only",
            "source_sha256": source_hash, "dataset_sha256": data_hash, "dataset": dataset, "spec": spec}
    protocol = {**deepcopy(runner.FIXED_PROTOCOL), "plan_id": plan_id, "dataset_sha256": data_hash}
    runtime = {"python": "synthetic-test-only", "device": "cuda:0", "packages": {"torch": "NOT LOADED"},
               "gpu_uuid": "synthetic", "pid": 123, "threads": 4}

    def stage(family, case, answer, budget=32, *, truncated=False):
        prompt = (inference["system_message"] + "\n<img>" + "<IMG_CONTEXT>" * 256
                  + "</img>\n" + case["question"] + "\nassistant\nAnswer:")
        # Stable question-based IDs guarantee a NEW repeat record has equal input.
        ids = [1] + [9] * 256 + [sum(case["question"].encode()) + 20, 10]
        image = next(i for i in family["images"] if i["image_id"] == case["image_id"])
        inputs = {"case": deepcopy(case), "question": case["question"], "rendered_prompt": prompt,
                  "input_ids": ids, "row": len(ids) - 1, "image_columns": list(range(1, 257)),
                  "image": {**{k: image[k] for k in ("image_id", "relative_path", "sha256")},
                            "verified_before_prepare": True}, "profile": "common"}
        candidates = case["candidates"]
        score = {"scores": {c: -float(i + 1) for i, c in enumerate(candidates)},
                 "candidate_token_ids": {c: [20 + i] for i, c in enumerate(candidates)},
                 "candidate_token_logprobs": {c: [-float(i + 1)] for i, c in enumerate(candidates)},
                 "prediction": candidates[0], "tied": False, "routing": {},
                 "score_kind": "complete_candidate_sequence_logprob",
                 **{k: inputs[k] for k in ("rendered_prompt", "input_ids", "row", "image_columns")}}
        gen = {"text": answer, "token_ids": [11] * budget if truncated else [11, 2],
               "finish_reason": "length" if truncated else "eos",
               "parsed_answer": "UNTRUSTED FALLBACK", "correct": True,
               "settings": {"max_new_tokens": budget, "eos_token_id": 2,
                            "do_sample": False, "use_cache": False}}
        parsed = h.strict_token_answer(gen, candidates)
        return {**h._empty_stage(), "case": deepcopy(case), "input": inputs, "score": score,
                "generation": gen, "attempted": True, "parsed_answer": parsed,
                "answer_valid": parsed is not None, "exact": parsed is not None and parsed == case["gold"],
                "score_exact": candidates[0] == case["gold"],
                "status": "completed" if parsed is not None else "invalid_answer", "elapsed_seconds": .01}

    for checkpoint, pin in zip(audit.PINS, pins, strict=True):
        root = tmp_path / checkpoint
        root.mkdir()
        bases, controls = [], []
        for family in families[4:]:
            base, control = h._family_record(family), runner._control_record(family)
            for case, row, cr in zip(h.selected_cases(family), base["rows"], control["rows"], strict=True):
                wrong = next(c for c in colors if c != case["gold"])
                direct_answer = case["gold"] if family["index"] % 3 == 0 else wrong
                row["stages"]["direct"] = stage(family, dict(case, case_id=case["case_id"] + "-generated-direct"), direct_answer)
                invalid = family["index"] == 4 and case["direction"] == 1
                shape_answer = "The shape is triangle." if invalid else shapes[0]
                row["stages"]["shape"] = stage(family, h.shape_case(family, case), shape_answer)
                for name, wrongcue in (("label_chain", False), ("wrongcue_chain", True)):
                    if invalid:
                        row["stages"][name].update(status="skipped", skip_reason="invalid_or_failed_generated_shape")
                    else:
                        answer = case["gold"] if not wrongcue and (checkpoint == "8b-gspo" or family["index"] % 2 == 0) else wrong
                        row["stages"][name] = stage(family, h.cue_case(case, shapes[0], wrong_selector=wrongcue), answer)
                for name, cc in runner.control_cases(family, case).items():
                    cr["stages"][name] = stage(family, cc, direct_answer, runner.CONTROLS[name],
                        truncated=name == "direct_budget64" and family["index"] == 5 and case["direction"] == 1)
                cr["repeat_audit"] = {"reference_case_id": row["stages"]["direct"]["case"]["case_id"],
                    "shape_call_case_id": row["stages"]["shape"]["case"]["case_id"],
                    "conditioned_on_shape_output": False, "actual_generate": True,
                    "same_input_ids": True, "same_prompt": True, "same_raw_generation": True}
                fixed = cr["stages"]["selectorless_fixedcue32"]
                cue = fixed["case"]["cued_shape"]
                cue_color = family["palettes"][0][next(i for i, n in enumerate(family["nodes"]) if n["shape"] == cue)]
                cr["evaluation"].update(fixedcue_equals_destination=cue == cr["evaluation"]["destination_shape"],
                    fixedcue_color=cue_color, fixedcue_lookup_exact=fixed["parsed_answer"] == cue_color)
                cr["status"] = "completed"
                row["status"] = "failed" if invalid else "completed"
            base["status"] = "failed" if family["index"] == 4 else "completed"
            base["counts"] = h.summarize([base], p0_passed=True)
            write(root / f"family-{family['index']:04d}.json", base)
            write(root / f"controls-{family['index']:04d}.json", control)
            bases.append(base)
            controls.append(control)
        model = {"checkpoint": pin, "common_inference": inference, "scientific_fixture": False,
                 "dtype": "torch.float32", "image_tokens": 256, "tile_grid": [16, 16],
                 "lm_config_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
                 "model_code_sha256": {"SYNTHETIC_TEST_ONLY": "c" * 64},
                 "tokenizer_assets": {"SYNTHETIC_TEST_ONLY": "d" * 64},
                 "weight_files": [{"name": "SYNTHETIC_TEST_ONLY", "size_bytes": 1, "sha256": "e" * 64}]}
        p0 = {"passed": True, "fixture_only": False, "model": model, "runtime": runtime,
              "checks": {k: {"passed": True, "finite": True, "argmax_match": True,
                             "mismatches": 0, "max_abs": 0.0}
                         for k in ("native_embedding", "capture_only", "self_qk")},
              "candidate_suffixes": {c: [20 + i] for i, c in enumerate(colors + shapes)},
              "multi_token_candidate_checks": {}}
        metadata = {"checkpoint": checkpoint, "kind": protocol["kind"], "created_at": "test-only",
            "source_check": True, "failure": None, "allow_downloads": False, "sites": [],
            "case_hooks": False, "full_captures": False, "conditioning": "common_answer_prefix_unchanged",
            "plan": plan, "plan_file_sha256": audit.sha(audit.canonical(plan)), "protocol": protocol,
            "protocol_json": audit.canonical(protocol).decode(), "protocol_sha256": audit.sha(audit.canonical(protocol)),
            "source_manifest": {k: {"path": str(audit.REPO / "scripts" / k), "text": v.decode(), "sha256": audit.sha(v)}
                                for k, v in snapshots.items()},
            "original_inference": inference, "effective_inference": inference, "device_override": "cuda:0",
            "model": model, "runtime": runtime, "elapsed_seconds": 1.0,
            "schedule": [{"index": f["index"], "family_id": f["family_id"],
                          "case_ids": [c["case_id"] for c in h.selected_cases(f)]} for f in families[4:]]}
        start = {k: v for k, v in metadata.items() if k not in ("elapsed_seconds", "failure")}
        write(root / "report-start.json", {**start, "model": None, "runtime": None})
        write(root / "metadata.json", metadata)
        write(root / "preflight.json", p0)
        write(root / "report.json", runner.summarize(bases, controls, checkpoint=checkpoint, p0_passed=True))
    return tmp_path / "8b-mpo", tmp_path / "8b-gspo", tmp_path / "analysis"


def test_complete_raw_reconstruction_costs_interaction_and_no_clobber(reports):
    mpo, gspo, out = reports
    before = {p: p.read_bytes() for root in (mpo, gspo) for p in root.iterdir()}
    result = audit.analyze(mpo, gspo, out)
    assert before == {p: p.read_bytes() for p in before}
    primary = result["primary_gspo"]
    assert primary["families"] == 20 and primary["seed"] == 20261102 and primary["samples"] == 10000
    assert primary["difference"] == pytest.approx((39 - 12) / 40)
    g = result["endpoints"]["8b-gspo"]
    assert g["counts"]["invalid_shape"] == g["counts"]["stages"]["label_chain"]["skipped"] == 1
    assert g["delivery_and_cost"]["direct_budget64"]["reasons"]["truncated_length"] == 1
    assert g["delivery_and_cost"]["direct_budget64"]["generated_tokens_total"] == 39 * 2 + 64
    assert g["paired_cases"]["direct"]["rescue"] == 27 and g["paired_cases"]["direct"]["regression"] == 0
    assert sum(g["shape_chain_cross"].values()) == 40
    partition = g["selector_lookup_partition"]
    assert partition["n"] == sum(partition["counts"].values()) == 40
    assert partition["counts"]["invalid_selector"] == 1
    assert partition["counts"]["wrong_selector_unfaithful_lookup"] > 0
    assert all(v["alpha"] == .01 for v in g["secondary_exploratory"].values())
    assert result["all_three_cue_content_intervention"] == "untested" and not result["optimizer_specific_claim"]
    with (out / "cases.csv").open() as stream:
        rows = list(csv.DictReader(stream))
        assert len(rows) == 640
        for row in rows:
            if row["cued_shape"]:
                assert (row["lookup_follows_actual_cue"] == "True") == (row["parsed_answer"] == row["cued_color"])
                if row["cue_is_destination"] == "True":
                    assert row["lookup_follows_actual_cue"] == row["exact"]
    seal = json.loads((out / "integrity.json").read_bytes())
    assert seal["before_after_equal"]
    for name, signature in seal["files"].items():
        payload = (out / name).read_bytes()
        assert signature == {"sha256": audit.sha(payload), "size_bytes": len(payload)}
    with pytest.raises(FileExistsError):
        audit.analyze(mpo, gspo, out)


@pytest.mark.parametrize("name,mutate", [
    ("report.json", lambda r: r.update(label_chain=40)),
    ("report.json", lambda r: r["stages"]["direct"].update(n=39)),
    ("report.json", lambda r: r.update(execution_complete=False)),
    ("family-0004.json", lambda r: r["rows"][0]["stages"]["shape"].update(parsed_answer="triangle", answer_valid=True)),
    ("family-0004.json", lambda r: r["rows"][0]["stages"]["wrongcue_chain"].update(attempted=True)),
    ("family-0005.json", lambda r: r["rows"][0]["stages"]["wrongcue_chain"]["case"].update(cued_shape="hexagon")),
    ("family-0005.json", lambda r: r["rows"][0]["stages"]["shape"]["case"].update(gold="hexagon")),
    ("family-0005.json", lambda r: r["rows"][0]["stages"]["label_chain"]["case"].update(gold="ORACLE")),
    ("family-0005.json", lambda r: r["rows"][0]["stages"]["direct"].update(status="failed", errors=[{"type": "failure"}])),
    ("family-0005.json", lambda r: r["rows"].reverse()),
    ("family-0005.json", lambda r: r["rows"][0]["stages"]["direct"]["input"].update(rendered_prompt="Answer:gold")),
    ("family-0005.json", lambda r: r["rows"][0]["stages"]["direct"]["score"]["scores"].pop("red")),
    ("family-0005.json", lambda r: r["rows"][0]["stages"]["direct"]["score"]["candidate_token_logprobs"].update(red=[-999.])),
    ("controls-0005.json", lambda r: r["rows"][0]["stages"]["direct_budget64"]["generation"]["settings"].update(max_new_tokens=63)),
    ("controls-0005.json", lambda r: r["rows"][0]["stages"]["direct_budget64"]["generation"].update(token_ids=[11] * 63)),
    ("controls-0005.json", lambda r: r["rows"][0]["stages"]["wasted_call_repeat32"]["generation"].update(
        text=r["rows"][0]["stages"]["wasted_call_repeat32"]["generation"]["text"] + "!")),
    ("controls-0005.json", lambda r: r["rows"][0]["stages"]["wasted_call_repeat32"]["generation"].update(token_ids=[12, 2])),
    ("controls-0005.json", lambda r: r["rows"][0]["stages"]["wasted_call_repeat32"]["generation"]["settings"].update(max_new_tokens=64)),
    ("controls-0005.json", lambda r: r["rows"][0]["stages"]["wasted_call_repeat32"]["case"].update(case_id=r["rows"][0]["case_id"])),
    ("metadata.json", lambda r: r["runtime"].update(pid=999)),
    ("metadata.json", lambda r: r["effective_inference"].update(dtype="bfloat16")),
    ("metadata.json", lambda r: r["source_manifest"][audit.SOURCES[0]].update(sha256="f" * 64)),
    ("metadata.json", lambda r: r["model"]["checkpoint"].update(name="8b-mpo")),
    ("metadata.json", lambda r: r["plan"]["dataset"]["families"].append(deepcopy(r["plan"]["dataset"]["families"][4]))),
    ("metadata.json", lambda r: r["protocol"].update(family_offset=0)),
    ("metadata.json", lambda r: r["schedule"][0].update(case_ids=[])),
])
def test_tampering_partial_fallback_cues_sources_runtime_budgets_rejected(reports, name, mutate):
    mpo, gspo, out = reports
    edit(gspo, name, mutate)
    with pytest.raises((ValueError, RuntimeError)):
        audit.analyze(mpo, gspo, out)
    assert not out.exists()


@pytest.mark.parametrize("mode", ("missing", "alias", "symlink", "zip", "duplicate_json", "nonfinite"))
def test_input_inventory_strict_and_no_output_on_failure(reports, mode):
    mpo, gspo, out = reports
    path = gspo / "family-0004.json"
    if mode == "missing":
        path.unlink()
    elif mode == "alias":
        (gspo / "family-4.json").write_bytes(path.read_bytes())
    elif mode == "symlink":
        path.unlink()
        path.symlink_to(mpo / path.name)
    elif mode == "zip":
        (gspo / "backup.zip").write_bytes(b"not allowed")
    elif mode == "duplicate_json":
        path.write_text('{"rows":[],"rows":[]}')
    else:
        path.write_text('{"scores":{"red":1e999}}')
    with pytest.raises(ValueError):
        audit.analyze(mpo, gspo, out)
    assert not out.exists()


def test_mutation_during_statistics_prevents_publication(reports, monkeypatch):
    mpo, gspo, out = reports
    original = audit.statistics

    def changed(result):
        value = original(result)
        edit(gspo, "report.json", lambda r: r.update(n=39))
        return value

    monkeypatch.setattr(audit, "statistics", changed)
    with pytest.raises(ValueError, match="changed"):
        audit.analyze(mpo, gspo, out)
    assert not out.exists()


@pytest.mark.parametrize("key,value", (("pid", 999), ("gpu_uuid", "different"), ("threads", 8)))
def test_coherent_endpoint_runtime_change_ignores_no_fields(reports, key, value):
    mpo, gspo, out = reports
    # Keep each endpoint internally consistent; only BETWEEN-endpoint equality fails.
    for name in ("metadata.json", "preflight.json"):
        edit(gspo, name, lambda r: r["runtime"].update({key: value}))
    with pytest.raises(ValueError, match="Endpoint mismatch: runtime"):
        audit.analyze(mpo, gspo, out)
    assert not out.exists()


def test_repeat_input_ids_compared_independently_of_audit_flags(reports):
    mpo, gspo, out = reports

    def mutate(record):
        stage = record["rows"][0]["stages"]["wasted_call_repeat32"]
        for item in (stage["input"], stage["score"]):
            item["input_ids"][-1] += 1

    edit(gspo, "controls-0005.json", mutate)
    with pytest.raises(ValueError, match="repeat input"):
        audit.analyze(mpo, gspo, out)
    assert not out.exists()


def test_constant_intervals_are_bounded_conservative_not_zero_width():
    for alpha, bound in ((.05, 1), (.01, 1), (.05, 2)):
        result = audit.interval({str(i): float(bound) for i in range(20)}, alpha=alpha, bound=bound)
        assert result["interval_method"] == "conservative_hoeffding"
        assert -bound <= result["interval"][0] < result["interval"][1] <= bound


def test_import_help_and_validation_never_import_models_or_eager_numpy(reports, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "numpy", "PIL", "filelock", "pydantic", "yaml"}
        assert name not in {"vlm_pilot.runtime", "vlm_pilot.plan"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = load()
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    mpo, gspo, out = reports
    # Invalid input fails BEFORE the only optional statistical import.
    (gspo / "family-0004.json").unlink()
    with pytest.raises(ValueError):
        module.analyze(mpo, gspo, out)
    with pytest.raises(SystemExit):
        module.main(["--mpo", str(mpo), "--gspo", str(gspo), "--output", str(out), "--seed", "1"])


def test_overlap_refused_without_footprint(reports):
    mpo, gspo, out = reports
    for a, b, target in ((mpo, mpo, out), (mpo, gspo, mpo / "analysis"), (mpo, gspo, mpo.parent)):
        with pytest.raises(ValueError, match="Overlapping"):
            audit.analyze(a, b, target)
    assert not out.exists()