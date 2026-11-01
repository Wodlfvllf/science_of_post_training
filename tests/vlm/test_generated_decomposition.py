"""Model-free generated-selector contracts; no torch, pretrained models or network.

TinyFakeEngine spies on every separate prepare/score/generate. Tests distinguish
raw native EOS delivery from both Engine's parser and candidate-score winners.
Temporary image bytes test the same-image SHA guard without rendering any model.
"""
from __future__ import annotations

import argparse
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

from vlm_pilot.data import COLORS, SHAPES, _family

PATH = Path(__file__).resolve().parents[2] / "scripts/vlm_generated_decomposition.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generated_decomposition_tests", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnostic = load_module()


def generation(text, *, finish="eos"):
    # Deliberately wrong Engine parser fields: the script must grade RAW text.
    return {"text": text, "token_ids": [11, 2] if finish == "eos" else [11] * 32,
            "finish_reason": finish, "parsed_answer": "triangle", "valid": True,
            "correct": True, "elapsed_seconds": 0.01,
            "settings": {"max_new_tokens": 32, "do_sample": False,
                         "use_cache": False, "eos_token_id": 2}}


class TinyFakeEngine:
    """Only the documented Engine protocol; score always favors an unrelated word."""

    def __init__(self, outputs):
        self.outputs = deepcopy(outputs)
        self.prepares, self.scores, self.generations = [], [], []
        self.tokenizer = SimpleNamespace(eos_token_id=2)
        self.settings = SimpleNamespace(answer_prefix="Answer:", system_message="original system")
        self._features, self._family = {}, None
        self.score_error = None

    def _prompt(self, question, has_image, profile):
        assert has_image is True and profile == "common"
        return "original system\n<image>\n" + question + "\nassistant\nAnswer:"

    def prepare(self, root, family, case, *, profile):
        assert profile == "common"
        self.prepares.append(deepcopy(case))
        if self._family != family["family_id"]:
            self._features.clear()
            self._family = family["family_id"]
        self._features[case["image_id"]] = "native-image-features"
        number = len(self.prepares)
        return SimpleNamespace(case=case, prompt=self._prompt(case["question"], True, profile),
                               input_ids=(1, 5, number + 10), row=2, image_columns=(1,))

    def score(self, prepared):
        self.scores.append(prepared)
        if self.score_error is not None:
            raise self.score_error
        candidates = prepared.case["candidates"]
        scores = {word: -float(i + 1) for i, word in enumerate(candidates)}
        return {"scores": scores, "prediction": candidates[0], "tied": False,
                "candidate_token_ids": {word: [i + 20] for i, word in enumerate(candidates)},
                "input_ids": list(prepared.input_ids), "rendered_prompt": prepared.prompt,
                "row": prepared.row, "image_columns": list(prepared.image_columns),
                "routing": {}, "score_kind": "complete_candidate_sequence_logprob"}, {}

    def generate(self, prepared, budget):
        assert budget == 32
        self.generations.append(prepared)
        result = self.outputs.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def scene(tmp_path):
    family = _family(31, "engineering", 0)
    # Deliberately permute nodes relative to fixed SHAPES to catch oracle/order use.
    for node, shape in zip(family["nodes"], reversed(SHAPES), strict=True):
        node["shape"] = shape
    from vlm_pilot.data import make_case

    family["cases"] = [make_case(family, edge=e, direction=d, palette=p)
                       for e in (1, -1) for d in (1, -1) for p in (0, 1)]
    for image in family["images"]:
        path = tmp_path / image["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("checksummed-image-" + image["image_id"]).encode())
        image["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return tmp_path, family, diagnostic.selected_cases(family)[0]


@pytest.mark.parametrize("word", SHAPES + COLORS)
@pytest.mark.parametrize("surface", (str.upper, str.title, lambda s: " \n" + s + ".\t "))
def test_strict_bare_casefold_one_period(word, surface):
    candidates = SHAPES if word in SHAPES else COLORS
    assert diagnostic.strict_token_answer(generation(surface(word)), candidates) == word


@pytest.mark.parametrize("text", (
    "The shape is triangle.", "I think triangle", "<think>triangle</think>triangle",
    "triangle because the arrow", "triangle circle", "triangle..", "triangle .",
    "triangle!", "'triangle'", '"triangle"', "**triangle**", "(triangle)",
    "triangle\nexplanation", "triangle.\ntriangle", "Answer: triangle", "", "hexagon",
))
def test_strict_rejects_extraction_explanation_think_extra_punctuation(text):
    assert diagnostic.strict_token_answer(generation(text), SHAPES) is None


@pytest.mark.parametrize("finish", ("length", "stop", "error", None))
def test_strict_rejects_truncated_even_with_engine_valid_best_answer(finish):
    result = generation("TRIANGLE.")
    result["finish_reason"] = finish
    assert diagnostic.strict_token_answer(result, SHAPES) is None


@pytest.mark.parametrize("result", ({}, {"text": None, "finish_reason": "eos"}, None, "triangle"))
def test_strict_malformed_raw_output(result):
    assert diagnostic.strict_token_answer(result, SHAPES) is None


def test_original_cases_and_exact_shape_template_fixed_vocabulary(scene):
    _, family, _ = scene
    cases = diagnostic.selected_cases(family)
    assert len(cases) == 2 and [c["direction"] for c in cases] == [1, -1]
    before = deepcopy(family)
    for case in cases:
        source = diagnostic.shape_case(family, case)
        anchor = family["nodes"][case["anchor"]]["shape"]
        target = family["nodes"][case["target"]]["shape"]
        traversal = "forward" if case["direction"] == 1 else "backward"
        assert source["question"] == (f"Starting at the {anchor}, follow one arrow {traversal}. "
                                      "What shape is the destination? Reply with one shape name.")
        assert target not in source["question"] and case["gold"] not in source["question"]
        assert source["gold"] == target and source["candidates"] == list(SHAPES)
        assert source["image_id"] == case["image_id"] and source["case_id"] != case["case_id"]
    assert family == before


def test_predicted_cues_not_gold_and_wrongselector_prediction_plus1_fixed_order(scene):
    _, family, case = scene
    target = family["nodes"][case["target"]]["shape"]
    prediction = next(s for s in SHAPES if s != target)
    for wrong in (False, True):
        cue = diagnostic.cue_case(case, prediction, wrong_selector=wrong)
        expected = SHAPES[(SHAPES.index(prediction) + 1) % len(SHAPES)] if wrong else prediction
        assert cue["question"] == f"What color is the chip at the {expected}? Reply with one color name."
        assert cue["gold"] == case["gold"]  # Not cued-node color, even for wrong prediction.
        assert cue["predicted_shape"] == prediction and cue["cued_shape"] == expected
        assert prediction in cue["case_id"] and cue["candidates"] == list(COLORS)
        assert cue["image_id"] == case["image_id"]
        if wrong:
            assert "prediction-plus-1" in cue["case_id"]
            assert cue["selector_rule"] == diagnostic.WRONG_SELECTOR_RULE
    for i, shape in enumerate(SHAPES):
        assert diagnostic.next_shape(shape) == SHAPES[(i + 1) % 3]


def test_evaluate_case_fresh_native_prepares_raw_source_prompt_and_original_gold(scene):
    root, family, case = scene
    source_gold = family["nodes"][case["target"]]["shape"]
    prediction = next(s for s in SHAPES if s != source_gold)
    engine = TinyFakeEngine([generation(case["gold"]), generation(prediction.upper() + "."),
                             generation(case["gold"]), generation(case["gold"])])
    progress = []
    row = diagnostic.evaluate_case(engine, root, family, case,
                                   progress=lambda stage, **k: progress.append((stage, k)))
    assert len(engine.prepares) == len(engine.scores) == len(engine.generations) == 4
    assert len({c["case_id"] for c in engine.prepares}) == 4
    assert all(a is b for a, b in zip(engine.scores, engine.generations, strict=True))
    assert engine.prepares[0]["question"] == case["question"]
    assert source_gold not in engine.prepares[1]["question"]
    assert engine.prepares[2]["question"] == f"What color is the chip at the {prediction}? Reply with one color name."
    assert all(c["gold"] == case["gold"] for c in (engine.prepares[0], *engine.prepares[2:]))
    assert row["stages"]["shape"]["parsed_answer"] == prediction
    assert row["stages"]["shape"]["generation"]["token_ids"] == [11, 2]
    assert row["stages"]["shape"]["exact"] is False
    assert row["stages"]["label_chain"]["exact"] is True
    assert row["stages"]["wrongcue_chain"]["exact"] is True
    images = [s["input"]["image"] for s in row["stages"].values()]
    assert all(i == images[0] for i in images)
    for stage in row["stages"].values():
        assert stage["score"]["rendered_prompt"] == stage["input"]["rendered_prompt"]
        assert stage["score"]["input_ids"] == stage["input"]["input_ids"]
        assert stage["input"]["rendered_prompt"].endswith("Answer:")
    assert [k["operation"] for _, k in progress] == ["prepare", "score", "generate"] * 4
    assert engine.settings.answer_prefix == "Answer:" and len(engine._features) == 1


@pytest.mark.parametrize("shape_output", (
    generation("The shape is triangle."), generation("triangle", finish="length"),
    generation("triangle.."), generation("<think>triangle"), RuntimeError("native raw-output error"),
))
def test_invalid_shape_no_cue_queued_no_bestcandidate_fallback(scene, shape_output):
    root, family, case = scene
    engine = TinyFakeEngine([generation("red"), shape_output])
    row = diagnostic.evaluate_case(engine, root, family, case)
    assert len(engine.prepares) == len(engine.generations) == 2
    assert engine.prepares[1]["candidates"] == list(SHAPES)
    assert row["stages"]["shape"]["parsed_answer"] is None
    assert row["status"] == "failed"
    for name in ("label_chain", "wrongcue_chain"):
        assert row["stages"][name]["status"] == "skipped"
        assert row["stages"][name]["attempted"] is False
        assert row["stages"][name]["exact"] is False
    if isinstance(shape_output, Exception):
        assert row["stages"]["shape"]["errors"][0]["message"] == "native raw-output error"


def test_bad_direct_or_label_raw_output_never_uses_score_prediction(scene):
    root, family, case = scene
    shape = family["nodes"][case["target"]]["shape"]
    engine = TinyFakeEngine([generation("The destination color is red."), generation(shape),
                             generation(case["gold"], finish="length"), generation("red..")])
    row = diagnostic.evaluate_case(engine, root, family, case)
    assert len(engine.prepares) == 4
    for name in ("direct", "label_chain", "wrongcue_chain"):
        assert row["stages"][name]["parsed_answer"] is None
        assert row["stages"][name]["exact"] is False
        assert row["stages"][name]["status"] == "invalid_answer"


def test_cued_node_color_is_not_chain_evaluation_gold(scene):
    root, family, case = scene
    prediction = family["nodes"][case["anchor"]]["shape"]
    wrongcue = diagnostic.next_shape(prediction)
    wrongnode = next(i for i, n in enumerate(family["nodes"]) if n["shape"] == wrongcue)
    assert wrongnode != case["target"]
    palette = family["palettes"][0]
    engine = TinyFakeEngine([generation(case["gold"]), generation(prediction),
                             generation(palette[case["anchor"]]), generation(palette[wrongnode])])
    row = diagnostic.evaluate_case(engine, root, family, case)
    assert row["stages"]["direct"]["exact"] is True
    for name in ("shape", "label_chain", "wrongcue_chain"):
        assert row["stages"][name]["answer_valid"] is True
        assert row["stages"][name]["exact"] is False
    assert row["evaluation"]["original_destination_color"] == case["gold"]


def test_score_error_does_not_hide_generation_or_override_generated_selector(scene):
    root, family, case = scene
    prediction = SHAPES[-1]
    engine = TinyFakeEngine([generation("red"), generation(prediction), generation("red"), generation("red")])
    engine.score_error = RuntimeError("score failed")
    row = diagnostic.evaluate_case(engine, root, family, case)
    assert len(engine.generations) == 4
    assert row["stages"]["shape"]["parsed_answer"] == prediction
    assert all(s["errors"][0]["operation"] == "score" for s in row["stages"].values())
    assert row["status"] == "failed"


@pytest.mark.parametrize("mutate", (
    lambda r: r["settings"].update(use_cache=True),
    lambda r: r["settings"].update(do_sample=True),
    lambda r: r["settings"].update(max_new_tokens=64),
    lambda r: r.update(token_ids=[11], finish_reason="eos"),
    lambda r: r.update(token_ids=[2, 11, 2]),
))
def test_generation_native_settings_and_actual_eos_fail_closed(scene, mutate):
    root, family, case = scene
    result = generation("triangle")
    mutate(result)
    engine = TinyFakeEngine([generation("red"), result])
    row = diagnostic.evaluate_case(engine, root, family, case)
    assert len(engine.prepares) == 2 and row["stages"]["shape"]["parsed_answer"] is None
    assert row["stages"]["shape"]["status"] == "failed"


def test_cached_step_rechecks_same_original_image_sha(scene):
    root, family, case = scene
    item = next(i for i in family["images"] if i["image_id"] == case["image_id"])
    engine = TinyFakeEngine([generation("red")])
    native_generate = engine.generate

    def changed(prepared, budget):
        result = native_generate(prepared, budget)
        (root / item["relative_path"]).write_bytes(b"changed-after-first-native-prepare")
        return result

    engine.generate = changed
    row = diagnostic.evaluate_case(engine, root, family, case)
    assert len(engine.prepares) == 1
    assert "checksum" in row["stages"]["shape"]["errors"][0]["message"]
    assert row["stages"]["label_chain"]["status"] == "skipped"


def test_all8_denominators_include_invalid_shapes_skipped_and_unstarted(scene):
    _, family, _ = scene
    records = [diagnostic._family_record(dict(family, family_id=f"scheduled-{i}")) for i in range(4)]
    row = records[0]["rows"][0]
    row["stages"]["direct"].update(status="completed", attempted=True, answer_valid=True, exact=True)
    row["stages"]["shape"].update(status="invalid_answer", attempted=True)
    for name in ("label_chain", "wrongcue_chain"):
        row["stages"][name]["status"] = "skipped"
    report = diagnostic.summarize(records, p0_passed=True)
    assert report["n"] == report["n_case"] == 8 and report["n_family"] == 4
    assert report["direct_exact"] == 1 and report["shape_correct"] == 0
    assert report["label_chain"] == report["wrongcue_chain"] == 0
    assert report["invalid_shape"] == 8
    assert all(s["n"] == 8 for s in report["stages"].values())
    assert report["stages"]["direct"]["accuracy"] == 1 / 8
    assert report["execution_complete"] is False and report["stage_cross"] == "not_computed"
    assert report["confirmation"] == "not_attempted"


def test_family_finally_publishes_interrupted_partial_and_clears_features(scene, tmp_path):
    root, family, _ = scene
    engine = TinyFakeEngine([generation("red"), KeyboardInterrupt("operator stop")])
    output = tmp_path / "published-family.json"
    with pytest.raises(KeyboardInterrupt, match="operator stop"):
        diagnostic.evaluate_family(engine, root, family, output)
    saved = json.loads(output.read_text())
    assert saved["status"] == "failed" and saved["failure"]["type"] == "KeyboardInterrupt"
    assert len(saved["rows"]) == saved["counts"]["n"] == 2
    first, second = saved["rows"]
    assert first["stages"]["direct"]["generation"]["token_ids"] == [11, 2]
    assert first["stages"]["shape"]["input"]["rendered_prompt"].endswith("Answer:")
    assert first["stages"]["shape"]["status"] == "failed"
    assert second["status"] == "not_started"
    assert engine._features == {} and engine._family is None
    original_bytes = output.read_bytes()
    refused_engine = TinyFakeEngine([])
    with pytest.raises(FileExistsError):
        diagnostic.evaluate_family(refused_engine, root, family, output)
    assert output.read_bytes() == original_bytes
    assert refused_engine.prepares == []


def test_cli_and_import_inert_no_optional_models_or_other_scripts(monkeypatch):
    native_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy", "transformers", "vlm_pilot", "filelock", "PIL"}:
            raise AssertionError("optional import during import/argument parsing")
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = load_module()
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--help"])
    assert exit_info.value.code == 0
    args = module._parser().parse_args(["--plan", "p", "--checkpoint", "8b-mpo", "--output", "new"])
    assert args.families == 4 and args.device == "cuda:0"
    for value in ("0", "5"):
        with pytest.raises(SystemExit):
            module.main(["--plan", "p", "--checkpoint", "8b-mpo", "--output", "new", "--families", value])
    for option in ("--layers", "--confirmation", "--allow-downloads", "--resume", "--budget"):
        with pytest.raises(SystemExit):
            module.main(["--plan", "p", "--checkpoint", "8b-mpo", "--output", "new", option])
    for value in ("cuda", "auto", "mps", "cuda:-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            module._device(value)


def test_cached_native_subclass_retains_single_family_feature_without_overrides(monkeypatch, scene):
    root, family, case = scene
    monkeypatch.setitem(sys.modules, "vlm_pilot.runtime", SimpleNamespace(Engine=TinyFakeEngine))
    cached = diagnostic.cached_engine_class()
    assert cached.__name__ == "CachedEngine" and cached.__bases__ == (TinyFakeEngine,)
    assert "generate" not in cached.__dict__ and "score" not in cached.__dict__
    engine = cached([])
    engine.prepare(root, family, case, profile="common")
    engine.prepare(root, family, diagnostic.shape_case(family, case), profile="common")
    assert len(engine._features) == 1
    with pytest.raises(ValueError, match="common"):
        engine.prepare(root, family, case, profile="native")


@pytest.mark.parametrize("budget", (0, 1, 31, 33, 64, True))
def test_no_budget_override_even_callable(scene, budget):
    root, family, case = scene
    engine = TinyFakeEngine([])
    with pytest.raises(ValueError, match="fixed 32"):
        diagnostic.evaluate_case(engine, root, family, case, budget)
    assert engine.prepares == []


def _cli_environment(monkeypatch, scene, tmp_path, *, load_error=None, p0_passed=True):
    """Exercise explicit dispatch with fake native construction, real immutable JSON."""
    from vlm_pilot import data as data_module
    from vlm_pilot import plan as plan_module
    from vlm_pilot.io import identity

    root, family, _ = scene
    settings = {"answer_prefix": "Answer:", "system_message": "original system",
                "dtype": "float32", "device": "cuda:0", "atol": 1e-4, "rtol": 1e-4}
    inference = SimpleNamespace(model_dump=lambda **kwargs: deepcopy(settings))
    dataset = {"families": [family]}
    plan_payload = {"plan_id": "original-pinned-plan", "dataset": dataset,
                    "source_sha256": "original-source", "spec": {"inference": settings}}
    plan = SimpleNamespace(spec=SimpleNamespace(inference=inference, dataset=root),
                           dataset=dataset, dataset_sha256=identity(dataset),
                           model_dump=lambda **kwargs: deepcopy(plan_payload))
    checkpoint = SimpleNamespace(backend="internvl", name="8b-mpo")
    state = {"locked": False, "constructors": [], "preflights": [], "locks": [],
             "source_checks": [], "dataset_checks": [], "engine": None}

    def read_plan(path, *, check_source):
        assert path == tmp_path / "pinned-plan.json"
        state["source_checks"].append(check_source)
        return plan

    def checkpoint_for(received, name):
        assert received is plan and name == "8b-mpo"
        return checkpoint

    def load_dataset(path, *, verify_images):
        assert path == root and verify_images is False
        state["dataset_checks"].append(path)
        return dataset

    class Guard:
        def __init__(self, path, timeout):
            assert timeout == 0
            state["locks"].append(path)

        def __enter__(self):
            assert not state["locked"]
            state["locked"] = True
            return self

        def __exit__(self, *args):
            state["locked"] = False

    class NativeSpy(TinyFakeEngine):
        def __init__(self, received_checkpoint, received_inference, *, device, allow_downloads):
            assert state["locked"] is True  # Same lock BEFORE native model construction.
            assert received_checkpoint is checkpoint and received_inference is inference
            assert device == "cuda:0" and allow_downloads is False
            state["constructors"].append((received_checkpoint, received_inference))
            if load_error:
                raise load_error
            outputs = []
            for case in diagnostic.selected_cases(family):
                shape = family["nodes"][case["target"]]["shape"]
                outputs += [generation(case["gold"]), generation(shape),
                            generation(case["gold"]), generation(case["gold"])]
            super().__init__(outputs)
            self.model_profile = {"checkpoint": "8b-mpo", "native": True}
            self.runtime_profile = {"device": "cuda:0", "native_test_double": True}
            state["engine"] = self

        def preflight(self, received_root, received_family, probe):
            assert state["locked"] and not self.prepares and not self.generations
            assert received_root == root and received_family is family
            assert probe["candidates"] == list(COLORS + SHAPES)
            state["preflights"].append(deepcopy(probe))
            return {"passed": p0_passed, "checks": {name: {"finite": True, "passed": p0_passed,
                "argmax_match": True, "mismatches": 0, "max_abs": 0.0}
                for name in ("native_embedding", "capture_only", "self_qk")},
                "candidate_suffixes": {c: [i + 20] for i, c in enumerate(COLORS + SHAPES)},
                "multi_token_candidate_checks": {}}

    monkeypatch.setattr(plan_module, "read_plan", read_plan)
    monkeypatch.setattr(plan_module, "checkpoint_for", checkpoint_for)
    monkeypatch.setattr(data_module, "load_dataset", load_dataset)
    monkeypatch.setitem(sys.modules, "filelock", SimpleNamespace(FileLock=Guard))
    monkeypatch.setitem(sys.modules, "vlm_pilot.runtime", SimpleNamespace(Engine=NativeSpy))
    plan_path = tmp_path / "pinned-plan.json"
    plan_path.write_text(json.dumps(plan_payload))
    output = tmp_path / "new-native-diagnostic"
    argv = ["--plan", str(plan_path), "--checkpoint", "8b-mpo", "--output", str(output),
            "--families", "1", "--device", "cuda:0"]
    return state, output, argv, plan_payload, settings


def test_cli_original_inference_lock_native_p0_all_candidates_and_durable_provenance(
        monkeypatch, scene, tmp_path):
    state, output, argv, plan_payload, settings = _cli_environment(monkeypatch, scene, tmp_path)
    assert diagnostic.main(argv) == 0
    assert state["locked"] is False and len(state["constructors"]) == 1
    assert state["source_checks"] == [True] and len(state["preflights"]) == 1
    expected_lock = Path(tempfile.gettempdir()) / "interp-vlm-cuda-0.lock"
    assert state["locks"] == [str(expected_lock)]
    engine = state["engine"]
    assert len(engine.generations) == 8 and len(engine.prepares) == 8
    assert engine._features == {} and engine._family is None
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["plan"] == plan_payload
    assert metadata["original_inference"] == metadata["effective_inference"] == settings
    assert metadata["allow_downloads"] is False and metadata["sites"] == []
    assert metadata["runtime"] == engine.runtime_profile and metadata["model"] == engine.model_profile
    assert metadata["diagnostic_source"] == {PATH.name: hashlib.sha256(PATH.read_bytes()).hexdigest()}
    assert metadata["plan_file_sha256"] == hashlib.sha256(Path(argv[1]).read_bytes()).hexdigest()
    report = json.loads((output / "report.json").read_text())
    assert report["n"] == 2 and report["execution_complete"] is True
    assert report["direct_exact"] == report["shape_correct"] == report["label_chain"] == 2
    assert report["wrongcue_chain"] == 2 and report["invalid_shape"] == 0
    assert report["stage_cross"] == "not_computed"
    saved = json.loads((output / "family-0001.json").read_text())
    assert len(saved["rows"]) == 2 and saved["status"] == "completed"
    # Existing CLI output is refused BEFORE a second model constructor or P0.
    original = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError, match="no overwrite or resume"):
        diagnostic.main(argv)
    assert len(state["constructors"]) == 1 and (output / "report.json").read_bytes() == original


@pytest.mark.parametrize("failure_stage", ("load", "p0"))
def test_cli_failure_before_cases_publishes_all_scheduled_family_denominators(
        monkeypatch, scene, tmp_path, failure_stage):
    state, output, argv, _, _ = _cli_environment(
        monkeypatch, scene, tmp_path,
        load_error=RuntimeError("cached model unavailable") if failure_stage == "load" else None,
        p0_passed=failure_stage != "p0")
    with pytest.raises(RuntimeError):
        diagnostic.main(argv)
    assert state["locked"] is False
    report = json.loads((output / "report.json").read_text())
    saved = json.loads((output / "family-0001.json").read_text())
    assert report["n"] == saved["counts"]["n"] == len(saved["rows"]) == 2
    assert report["p0_passed"] is False and report["execution_complete"] is False
    assert report["invalid_shape"] == 2 and report["failure"]["type"] == "RuntimeError"
    assert all(cell["n"] == 2 and cell["attempted"] == 0 for cell in report["stages"].values())
    assert saved["status"] == "failed" and (output / "metadata.json").is_file()
    if failure_stage == "p0":
        assert state["engine"].prepares == state["engine"].generations == []
        assert json.loads((output / "preflight.json").read_text())["passed"] is False


@pytest.mark.parametrize("mutation", (
    lambda p: p["checks"]["native_embedding"].update(finite=False),
    lambda p: p["checks"]["self_qk"].update(mismatches=1),
    lambda p: p["checks"]["capture_only"].update(argmax_match=False),
    lambda p: p["checks"]["self_qk"].update(max_abs=float("nan")),
    lambda p: p["candidate_suffixes"].pop("square"),
    lambda p: p["candidate_suffixes"].update(square=[50, 51]),
))
def test_native_p0_requires_actual_finite_allself_and_allcandidate_evidence(mutation):
    p0 = {"passed": True,
          "checks": {name: {"finite": True, "passed": True, "argmax_match": True,
                            "mismatches": 0, "max_abs": 0.0}
                     for name in ("native_embedding", "capture_only", "self_qk")},
          "candidate_suffixes": {c: [i + 20] for i, c in enumerate(COLORS + SHAPES)},
          "multi_token_candidate_checks": {}}
    diagnostic._require_preflight(p0, COLORS + SHAPES)
    mutation(p0)
    with pytest.raises(RuntimeError):
        diagnostic._require_preflight(p0, COLORS + SHAPES)