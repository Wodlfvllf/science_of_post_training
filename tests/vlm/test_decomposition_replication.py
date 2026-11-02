"""Model-free frozen-development contracts; no terminals, network or model loads.

TinyFakeEngine exercises real adjacent-helper records and immutable JSON writes.
Only temporary checksummed bytes stand in for images; no rendering is needed.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlm_pilot.data import COLORS, SHAPES
from vlm_pilot.data import _family as data_family
from vlm_pilot.io import identity

PATH = Path(__file__).absolute().parents[2] / "scripts/vlm_decomposition_replication.py"


def load_module():
    spec = importlib.util.spec_from_file_location("replication_contracts", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rep = load_module()


@pytest.fixture
def frozen(tmp_path):
    dataset = {"families": [data_family(31, "engineering", i) for i in range(24)]}
    for family in dataset["families"]:
        for image in family["images"]:
            path = tmp_path / image["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = image["image_id"].encode()
            path.write_bytes(payload)
            image["sha256"] = hashlib.sha256(payload).hexdigest()
    plan = SimpleNamespace(plan_id="a" * 64, dataset_sha256=identity(dataset), dataset=dataset)
    protocol = {**deepcopy(rep.FIXED_PROTOCOL), "plan_id": plan.plan_id,
                "dataset_sha256": plan.dataset_sha256}
    return tmp_path, plan, protocol


@pytest.mark.parametrize("patch", [
    {"seed": 7}, {"seeds": []}, {"family_offset": 0}, {"family_count": 4},
    {"family_count": 20.0}, {"schema_version": True}, {"schema_version": 2},
    {"split": "discovery"}, {"split": "pilot_validation"}, {"split": "confirmation"},
    {"kind": "confirmation"}, {"primary": "other"}, {"inference": "changed"},
    {"allow_downloads": True}, {"resume": True}, {"checkpoints": ["8b-cpt"]},
    {"factors": {"edge": -1, "palette": 0, "directions": [1, -1]}},
    {"budgets": {"original": 64, **rep.CONTROLS}}, {"selectorless_rule": "target_plus_1"},
    {"excluded_family_indices": []}, {"prohibited": []},
])
def test_frozen_protocol_rejects_all_extra_or_varied_values(frozen, patch):
    _, plan, protocol = frozen
    with pytest.raises(ValueError):
        rep.validate_protocol({**protocol, **patch}, plan)


@pytest.mark.parametrize("key", ("plan_id", "dataset_sha256"))
def test_protocol_requires_exact_plan_and_dataset_pins(frozen, key):
    _, plan, protocol = frozen
    assert rep.validate_protocol(protocol, plan) == protocol
    changed = deepcopy(protocol)
    changed[key] = "b" * 64
    with pytest.raises(ValueError, match="locked plan"):
        rep.validate_protocol(changed, plan)
    changed[key] = "short"
    with pytest.raises(ValueError, match="hashes"):
        rep.validate_protocol(changed, plan)


def test_exact20_selection_excludes_tuned4_and_other_splits(frozen):
    _, plan, protocol = frozen
    original = deepcopy(plan.dataset)
    dataset = deepcopy(original)
    dataset["families"] += [data_family(31, s, 24 + i) for i, s in enumerate(
        ("discovery", "pilot_validation", "confirmation"))]
    selected = rep.select_families(protocol, dataset)
    assert [f["index"] for f in selected] == list(range(4, 24))
    cases = [c for f in selected for c in rep._helper().selected_cases(f)]
    assert len({c["case_id"] for c in cases}) == 40
    assert [c["direction"] for c in cases] == [1, -1] * 20
    assert all(c["edge"] == 1 and c["palette"] == 0 for c in cases)
    assert len({c["image_id"] for c in cases}) == 20
    assert plan.dataset == original
    for mutated in (original["families"][1:], original["families"][::-1]):
        with pytest.raises(ValueError, match="24 engineering"):
            rep.select_families(protocol, {"families": mutated})


def test_control_templates_deterministic_nonoracle_and_fixed_vocabulary(frozen):
    _, plan, protocol = frozen
    for family in rep.select_families(protocol, plan.dataset):
        for case in rep._helper().selected_cases(family):
            controls = rep.control_cases(family, case)
            assert controls == rep.control_cases(family, case)
            direction = "forward" if case["direction"] == 1 else "backward"
            anchor = family["nodes"][case["anchor"]]["shape"]
            assert controls["composed_one_turn64"]["question"] == (
                f"Starting at the {anchor}, follow one arrow {direction}. First determine which shape "
                "is the destination, then report the color of its chip. Reply with one color name.")
            cue = SHAPES[(family["index"] + case["direction"]) % len(SHAPES)]
            assert controls["selectorless_fixedcue32"]["cued_shape"] == cue
            assert controls["selectorless_fixedcue32"]["question"] == f"What color is the chip at the {cue}? Reply with one color name."
            assert all(c["candidates"] == list(COLORS) and c["image_id"] == case["image_id"]
                       and c["gold"] == case["gold"] for c in controls.values())
            assert len({c["case_id"] for c in controls.values()}) == 4
            changed, changed_case = deepcopy(family), dict(case, gold="NOT_A_COLOR", target=999)
            changed.pop("palettes")
            changed["nodes"][case["target"]]["shape"] = "ORACLE_TRAP"
            other = rep.control_cases(changed, changed_case)
            assert [c["question"] for c in other.values()] == [c["question"] for c in controls.values()]
            assert controls["direct_budget64"]["question"] == controls["wasted_call_repeat32"]["question"] == case["question"]


class TinyFakeEngine:
    """Native protocol only; answers depend on questions, NEVER evaluation gold."""
    def __init__(self, *, invalid_shape=False, failure=None):
        self.invalid_shape, self.failure = invalid_shape, failure
        self.prepares, self.generations, self.scores = [], [], []
        self.tokenizer = SimpleNamespace(eos_token_id=2)
        self._features, self._family = {}, None
        self.model_profile, self.runtime_profile = {"native_test_double": True}, {"device": "cuda:0"}

    def _prompt(self, question, has_image, profile):
        assert has_image is True and profile == "common"
        return "unchanged system\n<image>\n" + question + "\nAnswer:"

    def prepare(self, root, family, case, *, profile):
        self.prepares.append(deepcopy(case))
        self._features = {case["image_id"]: "one-frame"}
        self._family = family["family_id"]
        return SimpleNamespace(case=case, prompt=self._prompt(case["question"], True, profile),
            input_ids=(1, 5, sum(case["question"].encode())), row=2, image_columns=(1,))

    def score(self, prepared):
        self.scores.append(prepared)
        if self.failure == "base" and prepared.case["case_id"].endswith("-generated-direct"):
            raise RuntimeError("base score operation failed")
        words = prepared.case["candidates"]
        return {"scores": {c: -float(i) for i, c in enumerate(words)}, "prediction": words[0],
            "tied": False, "candidate_token_ids": {c: [20 + i] for i, c in enumerate(words)},
            "rendered_prompt": prepared.prompt, "input_ids": list(prepared.input_ids),
            "row": prepared.row, "image_columns": [1], "routing": {}}, {}

    def generate(self, prepared, budget):
        self.generations.append((deepcopy(prepared.case), budget))
        if self.failure == "control" and budget == 64:
            raise RuntimeError("control generate operation failed")
        shape = prepared.case["candidates"] == list(SHAPES)
        text = "The shape is triangle." if shape and self.invalid_shape else SHAPES[0] if shape else COLORS[0]
        if budget == 64 and self.failure == "invalid_control":
            text = "The color is red."
        truncated = budget == 64 and self.failure == "length_control"
        return {"text": text, "token_ids": [11] * budget if truncated else [11, 2],
            "finish_reason": "length" if truncated else "eos",
            "parsed_answer": "deliberately_wrong", "correct": True,
            "settings": {"max_new_tokens": budget, "do_sample": False, "use_cache": False, "eos_token_id": 2}}


def environment(monkeypatch, frozen, *, mode=None, checkpoint="8b-gspo"):
    from vlm_pilot import data
    from vlm_pilot import plan as plans

    root, plan, protocol = frozen
    inference = SimpleNamespace(model_dump=lambda **k: {"answer_prefix": "Answer:", "dtype": "float32"})
    pin = SimpleNamespace(name=checkpoint, family="8b", backend="internvl", stage=checkpoint[3:], revision="c" * 40)
    plan.source_sha256 = "d" * 64
    plan.spec = SimpleNamespace(dataset=root, inference=inference)
    payload = {"plan_id": plan.plan_id, "dataset_sha256": plan.dataset_sha256, "revision": pin.revision}
    plan.model_dump = lambda **k: deepcopy(payload)
    plan_path, protocol_path = root / "plan.json", root / "protocol.json"
    plan_path.write_text(json.dumps(payload))
    protocol_path.write_text(json.dumps(protocol, indent=2))
    state = {"locked": False, "engines": [], "source_checks": [], "locks": [], "p0": []}

    def read_plan(path, *, check_source):
        assert path == plan_path
        state["source_checks"].append(check_source)
        return plan

    class Lock:
        def __init__(self, path, timeout):
            assert timeout == 0
            state["locks"].append(path)

        def __enter__(self):
            state["locked"] = True

        def __exit__(self, *args):
            state["locked"] = False

    class Native(TinyFakeEngine):
        def __init__(self, received, settings, *, device, allow_downloads):
            assert state["locked"] and received is pin and settings is inference
            assert device == "cuda:0" and allow_downloads is False
            if mode == "load":
                raise RuntimeError("cached model unavailable")
            super().__init__(invalid_shape=mode == "invalid", failure=mode)
            state["engines"].append(self)

        def preflight(self, received_root, family, case):
            assert state["locked"] and received_root == root and family["index"] == 4
            assert not self.prepares and case["candidates"] == list(COLORS + SHAPES)
            state["p0"].append(case)
            if mode == "drift":
                protocol_path.write_text(protocol_path.read_text() + "\n")
            return {"passed": mode != "p0", "checks": {k: {"passed": True, "finite": True,
                "argmax_match": True, "mismatches": 0, "max_abs": 0.0}
                for k in ("native_embedding", "capture_only", "self_qk")},
                "candidate_suffixes": {c: [10 + i] for i, c in enumerate(COLORS + SHAPES)},
                "multi_token_candidate_checks": {}}

    monkeypatch.setattr(rep, "REPO", root)
    monkeypatch.setattr(plans, "read_plan", read_plan)
    monkeypatch.setattr(plans, "checkpoint_for", lambda p, n: pin if p is plan and n == checkpoint else None)
    monkeypatch.setattr(plans, "source_digest", lambda: plan.source_sha256)
    monkeypatch.setattr(data, "load_dataset", lambda p, verify_images: plan.dataset)
    monkeypatch.setitem(sys.modules, "filelock", SimpleNamespace(FileLock=Lock))
    monkeypatch.setattr(rep._helper(), "cached_engine_class", lambda: Native)
    output = root / "runs/vlm-pilot/report-replication"
    argv = ["--plan", str(plan_path), "--protocol", str(protocol_path), "--checkpoint", checkpoint,
            "--output", str(output), "--device", "cuda:0"]
    return state, output, argv


@pytest.mark.parametrize("mode,per_case", ((None, 8), ("invalid", 6), ("invalid_control", 8), ("length_control", 8)))
def test_native_full_schedule_actual_repeat_and_invalid_nonfatal(monkeypatch, frozen, mode, per_case):
    state, output, argv = environment(monkeypatch, frozen, mode=mode)
    assert rep.main(argv) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["execution_complete"] and report["n"] == 40 and report["n_family"] == 20
    assert all(c["n"] == 40 and not c["operation_errors"] for c in report["stages"].values())
    assert report["invalid_shape"] == (40 if mode == "invalid" else 0)
    assert report["stages"]["label_chain"]["skipped"] == (40 if mode == "invalid" else 0)
    if mode in ("invalid_control", "length_control"):
        assert report["stages"]["direct_budget64"]["invalid_answer"] == 40
        assert report["stages"]["composed_one_turn64"]["valid"] == 0
    engine = state["engines"][0]
    assert len(engine.generations) == per_case * 40
    assert len(engine.prepares) == len(engine.scores) == len(engine.generations)
    assert {c["family_id"] for c in engine.prepares} == {f["family_id"] for f in frozen[1].dataset["families"][4:]}
    assert sum(b == 64 for _, b in engine.generations) == 80
    assert all(c["candidates"] in (list(COLORS), list(SHAPES)) for c in engine.prepares)
    assert engine._features == {} and state["source_checks"] == [True] and not state["locked"]
    assert state["locks"] == [str(Path(tempfile.gettempdir()) / "interp-vlm-cuda-0.lock")]
    assert len(report["paired_families"]) == 20
    assert all(len(p["case_ids"]) == 2 for p in report["paired_families"])
    assert report["primary_value"] == sum(p["chain_minus_direct"] for p in report["paired_families"]) / 20
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["original_inference"] == metadata["effective_inference"]
    assert metadata["protocol"] == frozen[2] and metadata["runtime"] == engine.runtime_profile
    assert metadata["protocol_sha256"] == hashlib.sha256(metadata["protocol_json"].encode()).hexdigest()
    assert (output / "report-start.json").is_file() and len(list(output.glob("family-*.json"))) == 20
    assert [s["index"] for s in metadata["schedule"]] == list(range(4, 24))
    for snapshot in metadata["source_manifest"].values():
        assert snapshot["sha256"] == hashlib.sha256(snapshot["text"].encode()).hexdigest()
    for path in output.glob("controls-*.json"):
        for row in json.loads(path.read_text())["rows"]:
            audit = row["repeat_audit"]
            assert audit["actual_generate"] and not audit["conditioned_on_shape_output"]
            assert audit["same_prompt"] and audit["same_input_ids"] and audit["same_raw_generation"]
            assert all(s["generation"]["token_ids"] and set(s["score"]["scores"]) == set(COLORS)
                       for s in row["stages"].values())
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        rep.main(argv)
    assert len(state["engines"]) == 1 and (output / "report.json").read_bytes() == before


@pytest.mark.parametrize("mode", ("load", "p0", "base", "control", "drift"))
def test_failures_stop_and_seal_all40_without_dropping_cases(monkeypatch, frozen, mode):
    state, output, argv = environment(monkeypatch, frozen, mode=mode)
    with pytest.raises(RuntimeError):
        rep.main(argv)
    report = json.loads((output / "report.json").read_text())
    assert not report["execution_complete"] and report["failure"]
    assert report["n"] == 40 and report["n_family"] == 20
    assert len(list(output.glob("family-*.json"))) == len(list(output.glob("controls-*.json"))) == 20
    assert all(c["n"] == 40 for c in report["stages"].values())
    assert not state["locked"] and (output / "metadata.json").is_file()
    if state["engines"]:
        engine = state["engines"][0]
        assert engine._features == {}
        assert all(c["family_id"] == frozen[1].dataset["families"][4]["family_id"] for c in engine.prepares)
        if mode in ("p0", "drift"):
            assert not engine.prepares
        if mode == "base":
            assert len(engine.prepares) == 1 and not engine.generations
        if mode == "control":
            partial = json.loads((output / "controls-0004.json").read_text())
            assert partial["failure"] and partial["rows"][0]["stages"]["direct_budget64"]["score"]
            assert partial["rows"][1]["status"] == "not_started"


@pytest.mark.parametrize("mutation", (
    lambda r: r["settings"].update(max_new_tokens=32),
    lambda r: r["settings"].update(use_cache=True),
    lambda r: r.update(token_ids=[11], finish_reason="eos"),
))
def test_control64_checks_actual_settings_and_tokens(monkeypatch, frozen, mutation):
    _, output, argv = environment(monkeypatch, frozen)
    native = TinyFakeEngine.generate

    def generate(self, prepared, budget):
        result = native(self, prepared, budget)
        if budget == 64:
            mutation(result)
        return result

    monkeypatch.setattr(TinyFakeEngine, "generate", generate)
    with pytest.raises(RuntimeError, match="Control operation"):
        rep.main(argv)
    stage = json.loads((output / "controls-0004.json").read_text())["rows"][0]["stages"]["direct_budget64"]
    assert stage["status"] == "failed" and stage["generation"] and stage["errors"]


@pytest.mark.parametrize("raw", ('{"schema_version":1,"schema_version":1}', '{"x":NaN}', '[]'))
def test_protocol_json_fail_closed_before_model(monkeypatch, frozen, raw):
    state, output, argv = environment(monkeypatch, frozen)
    Path(argv[3]).write_text(raw)
    with pytest.raises(ValueError):
        rep.main(argv)
    assert not state["engines"] and not output.exists()


def test_import_help_and_bad_arguments_are_stdlib_only(monkeypatch):
    native = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "numpy", "transformers", "vlm_pilot", "filelock", "PIL"}
        return native(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = load_module()
    assert module._HELPER is None
    with pytest.raises(SystemExit) as info:
        module.main(["--help"])
    assert info.value.code == 0
    argv = ["--plan", "p", "--protocol", "freeze", "--checkpoint", "8b-mpo", "--output", "o", "--device", "cuda:0"]
    for option in ("--seed", "--families", "--resume", "--confirmation", "--allow-downloads", "--budget"):
        with pytest.raises(SystemExit):
            module.main(argv + [option])
    for bad in ("8b-cpt", "2b-gspo"):
        with pytest.raises(SystemExit):
            module.main([*argv[:5], bad, *argv[6:]])
    with pytest.raises(SystemExit):
        module.main(argv[:2] + argv[4:])


def test_exact_adjacent_helper_hash_and_backup_output_safety(monkeypatch, frozen):
    module, root = load_module(), frozen[0]
    adjacent = root / "vlm_generated_decomposition.py"
    adjacent.write_bytes(PATH.with_name(adjacent.name).read_bytes())
    monkeypatch.setattr(module, "__file__", str(root / PATH.name))
    helper = module._helper()
    assert Path(helper.__file__) == adjacent
    adjacent.write_text(adjacent.read_text() + "\n# source drift\n")
    with pytest.raises(RuntimeError, match="path/hash"):
        module._helper()
    state, output, argv = environment(monkeypatch, frozen)
    argv[7] = str(output.with_name("not-a-report"))
    with pytest.raises(ValueError, match="backup-eligible"):
        rep.main(argv)
    assert not state["engines"]


def test_mpo_not_primary_and_case_pairing_cannot_change(monkeypatch, frozen):
    _, output, argv = environment(monkeypatch, frozen, checkpoint="8b-mpo")
    assert rep.main(argv) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["checkpoint"] == "8b-mpo" and report["primary_value"] is None
    families = rep.select_families(frozen[2], frozen[1].dataset)
    records, controls = [rep._helper()._family_record(f) for f in families], [rep._control_record(f) for f in families]
    partial = rep.summarize(records, controls, checkpoint="8b-gspo")
    assert partial["n"] == 40 and partial["invalid_shape"] == 40 and not partial["execution_complete"]
    assert all(c["not_started"] == 40 for c in partial["stages"].values())
    controls[0]["rows"].reverse()
    with pytest.raises(ValueError, match="Case IDs/directions"):
        rep.summarize(records, controls, checkpoint="8b-gspo")


def test_control_image_recheck_and_interruption_finally(monkeypatch, frozen):
    _, output, argv = environment(monkeypatch, frozen)
    generate = TinyFakeEngine.generate

    def interrupt(self, prepared, budget):
        if budget == 64:
            raise KeyboardInterrupt("operator stopped control")
        return generate(self, prepared, budget)

    monkeypatch.setattr(TinyFakeEngine, "generate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        rep.main(argv)
    saved = json.loads((output / "controls-0004.json").read_text())
    assert saved["failure"]["type"] == "KeyboardInterrupt"
    stage = saved["rows"][0]["stages"]["direct_budget64"]
    assert stage["score"] and stage["status"] == "failed" and stage["generation"] is None
    root, plan, _ = frozen
    family = plan.dataset["families"][4]
    case = rep._helper().selected_cases(family)[0]
    image = next(i for i in family["images"] if i["image_id"] == case["image_id"])
    (root / image["relative_path"]).write_bytes(b"corrupt")
    engine, stage = TinyFakeEngine(), rep._helper()._empty_stage()
    with pytest.raises(ValueError, match="checksum"):
        rep._control_stage(engine, root, family, rep.control_cases(family, case)["direct_budget64"],
                           stage, 64, lambda *a, **k: None, "direct_budget64")
    assert stage["errors"] and not engine.prepares