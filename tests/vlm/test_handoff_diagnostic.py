"""Model-free handoff contracts; optional fixture TOKENIZER tests load no model.

These tests are authored only; running them is a separate explicit action.
No GPU, model download, generation, source edit or locked trial is required.
"""
from __future__ import annotations

import argparse
import builtins
import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlm_pilot.data import COLORS, SHAPES, _family, make_case
from vlm_pilot.schemas import PilotError

PATH = Path(__file__).resolve().parents[2] / "scripts/vlm_handoff_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("handoff_diagnostic_tests", PATH)
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


@pytest.mark.parametrize("edge", (1, -1))
@pytest.mark.parametrize("direction", (1, -1))
@pytest.mark.parametrize("palette", (0, 1))
def test_paired_cases_exact_original_binding_factors_and_labels(edge, direction, palette):
    family = _family(17, "engineering", 2)
    original = deepcopy(family)
    base = make_case(family, edge=edge, direction=direction, palette=palette)
    color, shape = handoff.paired_cases(family, edge, direction, palette)
    name = family["nodes"][base["anchor"]]["shape"]
    direction_word = "forward" if direction == 1 else "backward"
    for case, kind in ((color, "color"), (shape, "shape")):
        for key in ("family_id", "split", "task", "image_id", "edge", "direction", "palette", "anchor", "target"):
            assert case[key] == base[key]
        assert case["question"] == (f"Starting at the {name}, follow one arrow {direction_word}. "
                                    f"What is the destination {kind}? Reply with one {kind} name.")
        assert case["answer_kind"] == kind
        assert case["gold"] in case["candidates"]
    assert color["gold"] == base["gold"]
    assert color["candidates"] == list(COLORS)
    assert shape["gold"] == family["nodes"][base["target"]]["shape"]
    assert shape["candidates"] == list(SHAPES)
    assert shape["gold"] != family["nodes"][base["anchor"]]["shape"]
    assert shape["gold"] not in shape["question"]
    assert color["gold"] not in color["question"]
    assert shape["question"] == color["question"].replace("color", "shape")
    assert len({color["case_id"], shape["case_id"], base["case_id"]}) == 3
    color["candidates"].append("unrelated")
    assert "unrelated" not in shape["candidates"]
    assert family == original


@pytest.mark.parametrize("direction", (1, -1))
def test_donor_conditions_same_geometry_and_disjoint_payload(direction):
    family = _family(17, "engineering", 0)
    color, _ = handoff.paired_cases(family, 1, direction, 0)
    donors = handoff.donor_cases(family, color)
    assert set(donors) == set(handoff.CONDITIONS) - {"native"}
    assert donors["self_q"] == color
    assert donors["shape_q"]["image_id"] == color["image_id"]
    for condition in ("shape_q_disjoint", "color_sham", "wrong_anchor_shape_q"):
        assert donors[condition]["palette"] == 1
        assert donors[condition]["image_id"] != color["image_id"]
        assert donors[condition]["edge"] == 1
        assert donors[condition]["direction"] == direction
    for condition in ("self_q", "shape_q", "shape_q_disjoint", "color_sham"):
        assert donors[condition]["anchor"] == color["anchor"]
        assert donors[condition]["target"] == color["target"]
    wrong = donors["wrong_anchor_shape_q"]
    assert wrong["anchor"] == color["target"]
    assert wrong["target"] == (wrong["anchor"] + direction) % 3
    assert wrong["gold"] == family["nodes"][wrong["target"]]["shape"]
    assert donors["color_sham"]["question"] == color["question"]
    assert not set(family["palettes"][0]) & set(family["palettes"][1])


@pytest.mark.parametrize("value", ("", " ", "0", "-1:0", "0:-1", "0:0,", ",0:0",
                                     "0:0,,1:1", "0:0,00:0", "all", "0:0:1", "1.0:0",
                                     ",".join(f"{i}:0" for i in range(9))))
def test_site_string_rejects_missing_dense_duplicate_or_overbound(value):
    with pytest.raises(argparse.ArgumentTypeError, match="1–8 unique"):
        handoff._site_pairs(value)


def test_import_and_cli_parsing_do_not_import_optional_runtime(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "numpy", "PIL", "vlm_pilot", "filelock"}:
            raise AssertionError("Optional runtime imported before explicit dispatch.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("handoff_lazy_import_test", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module._parser().parse_args(["--plan", "unused-plan", "--checkpoint", "2b-mpo",
                                      "--output", "new-output", "--sites", "0:0"])
    assert args.sites == "0:0"


def test_site_parser_returns_only_explicit_native_whole_groups():
    pytest.importorskip("torch")
    from vlm_pilot.attention import Site

    assert handoff.parse_sites(" 8:0,16:1 ") == (Site(8, 0), Site(16, 1))
    assert len(handoff.parse_sites(",".join(f"{i}:0" for i in range(8)))) == 8


def test_cli_defaults_bounds_and_query_only_without_optional_imports():
    parser = handoff._parser()
    required = ["--plan", "unused-plan", "--checkpoint", "2b-mpo", "--output", "new-output", "--sites", "8:0"]
    args = parser.parse_args(required)
    assert args.families == 4
    assert args.mode == "query"
    assert args.sites == "8:0"
    assert not args.include_route_permute
    for extra in (["--families", "0"], ["--families", "9"], ["--mode", "key"], ["--device", "auto"]):
        with pytest.raises(SystemExit):
            parser.parse_args(required + extra)
    with pytest.raises(SystemExit):
        parser.parse_args(required[:-2])


def _fixture_alignment():
    pytest.importorskip("torch")
    from vlm_pilot.runtime import Engine, FixtureTokenizer

    tokenizer = FixtureTokenizer()

    def prompt(question, has_image, profile):
        assert has_image and profile == "common"
        return ("<|im_start|>user\n<img>" + "<IMG_CONTEXT>"*256 + "</img>\n" + question
                + "<|im_end|>\n<|im_start|>assistant\n")

    engine = SimpleNamespace(tokenizer=tokenizer, _prompt=prompt,
                             settings=SimpleNamespace(answer_prefix=""),
                             require_alignment=Engine.require_alignment)

    def prepare(case):
        text = prompt(case["question"], True, "common")
        ids = tuple(tokenizer.encode(text, add_special_tokens=True))
        columns = tuple(i for i, token in enumerate(ids) if token == tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))
        return SimpleNamespace(case=case, prompt=text, input_ids=ids, row=len(ids)-1,
                               image_columns=columns, embeddings=SimpleNamespace(shape=(1, len(ids), 1)))

    return engine, prepare


@pytest.mark.parametrize("direction", (1, -1))
def test_optional_fixture_tokenizer_full_alignment_all_donors(direction):
    engine, prepare = _fixture_alignment()
    family = _family(17, "engineering", 0)
    color, _ = handoff.paired_cases(family, 1, direction, 0)
    recipient = prepare(color)
    for name, case in handoff.donor_cases(family, color).items():
        donor = prepare(case)
        check = handoff.require_alignment(engine, recipient, donor)
        assert check["passed"] and check["full_input_ids_checked"]
        assert len(donor.input_ids) == len(recipient.input_ids)
        assert donor.row == recipient.row and donor.image_columns == recipient.image_columns
        assert donor.prompt.endswith("<|im_start|>assistant\n")
        assert "Answer:" not in donor.prompt
        if name in ("self_q", "color_sham"):
            assert check["changed_token_positions"] == []


def test_unknown_destination_shape_gold_is_not_leaked_or_vocabulary_extended():
    engine, prepare = _fixture_alignment()
    family = _family(17, "engineering", 0)
    target = (family["anchor"] + 1) % 3
    family["nodes"][target]["shape"] = "pentagon_unknown"
    color, shape = handoff.paired_cases(family, 1, 1, 0)
    vocab = engine.tokenizer.get_vocab()
    assert shape["gold"] == "pentagon_unknown"
    assert shape["gold"] not in shape["question"]
    assert handoff.require_alignment(engine, prepare(color), prepare(shape))["passed"]
    assert engine.tokenizer.get_vocab() == vocab
    assert shape["gold"] not in vocab
    # Malformed ad hoc geometry is not silently normalized into certified SHAPES.
    assert shape["candidates"] == list(SHAPES)


def test_optional_alignment_rejects_length_positions_prefix_and_unrelated_text():
    engine, prepare = _fixture_alignment()
    family = _family(17, "engineering", 0)
    color, shape = handoff.paired_cases(family, 1, 1, 0)
    recipient = prepare(color)
    for change in (dict(row=recipient.row-1), dict(image_columns=recipient.image_columns[1:]),
                   dict(input_ids=recipient.input_ids[:-1])):
        donor = prepare(shape)
        donor.__dict__.update(change)
        with pytest.raises((ValueError, PilotError)):
            handoff.require_alignment(engine, recipient, donor)
    engine.settings.answer_prefix = "Answer:"
    with pytest.raises(ValueError, match="response/prefix"):
        handoff.require_alignment(engine, recipient, prepare(shape))
    engine.settings.answer_prefix = ""
    changed = dict(shape, question=shape["question"].replace("forward", "backward"))
    with pytest.raises(ValueError, match="outside"):
        handoff.require_alignment(engine, recipient, prepare(changed))


def test_individual_task_word_expansion_fails_without_padding():
    engine, prepare = _fixture_alignment()
    family = _family(17, "engineering", 0)
    color, shape = handoff.paired_cases(family, 1, 1, 0)
    encode = engine.tokenizer.encode
    shape_id = engine.tokenizer.convert_tokens_to_ids("shape")

    def expanded(text, add_special_tokens=True):
        return [token for original in encode(text, add_special_tokens=add_special_tokens)
                for token in ((original, original) if original == shape_id else (original,))]

    engine.tokenizer.encode = expanded
    with pytest.raises((ValueError, PilotError), match="align|length"):
        handoff.require_alignment(engine, prepare(color), prepare(shape))


def test_dynamic_alias_classifier_and_absolute_self_parity():
    classify, sha = handoff.load_classifier()
    assert len(sha) == 64
    result = {"scores": {"red": -2., "Red": -3., "blue": -4., "Blue": -5.},
              "candidate_token_ids": {"red": [1], "Red": [2], "blue": [3], "Blue": [4]},
              "prediction": "red"}
    assert handoff.self_score_check(result, deepcopy(result), classify, ["red", "blue"])["passed"]
    changed = deepcopy(result)
    changed["scores"]["Red"] += 2e-4
    with pytest.raises(ValueError, match="parity"):
        handoff.self_score_check(result, changed, classify, ["red", "blue"])
    with pytest.raises(ValueError, match="Empty"):
        handoff.self_score_check({"scores": {}}, result, classify, ["red", "blue"])
    changed = deepcopy(result)
    changed["scores"]["Red"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        handoff.self_score_check(result, changed, classify, ["red", "blue"])


def test_empty_or_failed_source_report_is_never_complete():
    sites = (SimpleNamespace(layer=8, kv_group=0),)
    for passed in (False, True):
        report = handoff.summarize([], 4, sites, handoff.CONDITIONS, p0_passed=passed)
        assert not report["complete"]
        assert report["n_family"] == report["n_result"] == report["n_hit_combined"] == 0
        assert report["expected_n_result"] == 4*2*6
        assert len(report["cells"]) == 2*6*3
        assert report["confirmation"] == "not_attempted"
        assert report["site_selection"] == "external_only"
    report = handoff.summarize([], 1, sites, handoff.CONDITIONS, p0_passed=True,
                              failure={"type": "RuntimeError"})
    assert not report["complete"]


def test_complete_counts_require_every_unconditional_direction_and_unique_row():
    site = SimpleNamespace(layer=8, kv_group=0)
    rows = []
    for family in ("f0", "f1"):
        for direction in (1, -1):
            for condition in handoff.CONDITIONS:
                outcomes = {g: {"hit": direction == 1, "anchor_copy": direction == -1,
                               "donor_answer_copy": False, "donor_payload_color_copy": False,
                               "wrong_anchor_recipient_color_hit": False}
                            for g in ("lower", "title", "combined")}
                rows.append({"family_id": family, "direction": direction, "site": "8:0",
                             "condition": condition, "outcomes": outcomes, "changed_route": False,
                             "attention_row_updated": False, "status": "completed",
                             "source_status": "native_preanswer_capture", "native_capture_unchanged": True,
                             "score": {"routing": {"8:0": {"conditional_l1": 0.}}}})
    report = handoff.summarize(rows, 2, (site,), handoff.CONDITIONS, p0_passed=True)
    assert report["complete"]
    assert report["n_result"] == 24 and report["n_hit_combined"] == 12
    assert report["n_family"] == 2 and report["n_recipient_case"] == 4
    assert all(c["n_family"] == 2 for c in report["cells"])
    assert not handoff.summarize(rows[:-1], 2, (site,), handoff.CONDITIONS, p0_passed=True)["complete"]
    assert not handoff.summarize(rows + [rows[0]], 2, (site,), handoff.CONDITIONS, p0_passed=True)["complete"]
    assert not handoff.summarize(rows, 2, (site,), handoff.CONDITIONS, p0_passed=False)["complete"]
    assert not handoff.summarize(rows, 2, (site,), handoff.CONDITIONS, p0_passed=True,
                                 failure={"type": "RuntimeError"})["complete"]


def test_missing_or_changed_native_routing_invariants_fail_closed():
    diagnostic = {"mode": "q", "captured_before_intervention": True, "donor_finite": True,
                  "mass_before_after_maxdiff": 0., "returned_attention_mass_maxdiff": 0.,
                  "reconstruction_weight_maxdiff": 0., "conditional_l1": 0.2,
                  "row_output_delta_rms": 0.1, "text_weight_maxdiff": 0.}
    handoff.check_routing(diagnostic, "q", 1e-4)
    for changes in ({"captured_before_intervention": False}, {"donor_finite": False},
                    {"text_weight_maxdiff": 1e-10}, {"mass_before_after_maxdiff": 2e-4},
                    {"conditional_l1": float("nan")}, {"mode": "v"}):
        with pytest.raises(ValueError):
            handoff.check_routing(dict(diagnostic, **changes), "q", 1e-4)


def test_cpu_memory_fit_estimate_is_explicit_and_fails_closed(monkeypatch):
    checkpoint = SimpleNamespace(family="2b", backend="internvl")
    inference = SimpleNamespace(dtype="float32", max_context_tokens=1024)
    monkeypatch.setattr(handoff, "_available_host_bytes", lambda: 64*1024**3)
    fit = handoff.memory_fit(checkpoint, inference, "cpu")
    assert fit["weight_bytes"] == 2_348_347_392*4
    assert fit["host_required_estimate_bytes"] > fit["weight_bytes"]*2
    assert not fit["peak_fit_guaranteed"]
    monkeypatch.setattr(handoff, "_available_host_bytes", lambda: 1)
    with pytest.raises(RuntimeError, match="host RAM"):
        handoff.memory_fit(checkpoint, inference, "cpu")