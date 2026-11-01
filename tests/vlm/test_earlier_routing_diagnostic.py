"""Authored model-free contracts; no test execution or model/download is needed.

Word-tokenizer and dataclass Prepared stand-ins exercise exact rendered-prefix
mapping, strict FULL alignment and row-only replacement. Optional tensor tests
use tiny CPU tensors with a fake Engine, NEVER instantiate a model. CLI failure
publication uses mocked plan/data/runtime APIs and no CUDA/download/real plan.
"""
from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vlm_pilot.data import _family, make_case

PATH = Path(__file__).resolve().parents[2] / "scripts/vlm_earlier_routing_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("earlier_routing_contracts", PATH)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class WordTokenizer:
    """Deterministic mock: actual word/punctuation tokens, no offsets API."""

    pattern = re.compile(r"<IMG_CONTEXT>|[A-Za-z_]+|[^\w\s]")

    def encode(self, text, add_special_tokens=True):
        # Stable text-derived IDs avoid vocabulary expansion from answers.
        words = self.pattern.findall(text)
        tokens = [99 if w == "<IMG_CONTEXT>" else
                  100 + int.from_bytes(w.encode(), "big") for w in words]
        return ([1] if add_special_tokens else []) + tokens


@dataclass(frozen=True)
class MockPrepared:
    case: dict
    prompt: str
    input_ids: tuple
    embeddings: object
    image_columns: tuple
    row: int
    pixels: object = None


def mock_engine(tokenizer=None, prefix="Answer:"):
    tokenizer = tokenizer or WordTokenizer()

    def prompt(question, has_image, profile):
        assert has_image is True and profile == "common"
        return "system original\n" + "<IMG_CONTEXT>"*256 + "\nuser " + question + "\nassistant\n" + prefix

    calls = []

    def require_alignment(recipient, donor, *, allow_query_change=False):
        calls.append((recipient.row, donor.row, allow_query_change))
        if (len(recipient.input_ids) != len(donor.input_ids) or recipient.row != donor.row
                or recipient.image_columns != donor.image_columns):
            raise ValueError("Native FULL alignment failed.")
        if not allow_query_change and recipient.input_ids != donor.input_ids:
            raise ValueError("Native query tokens changed.")

    engine = SimpleNamespace(tokenizer=tokenizer, img_token=99,
        settings=SimpleNamespace(answer_prefix=prefix, atol=1e-4), _prompt=prompt,
        require_alignment=require_alignment, alignment_calls=calls, _features={}, _family=None)

    def prepare(root, family, case, profile="common"):
        text = prompt(case["question"], True, profile)
        ids = tuple(tokenizer.encode(text, add_special_tokens=True))
        return MockPrepared(case, text, ids, SimpleNamespace(shape=(1, len(ids), 8)),
                            tuple(i for i, token in enumerate(ids) if token == 99), len(ids)-1,
                            pixels=object())

    engine.prepare = prepare
    engine.suffixes = lambda prepared: {c: (10+i,) for i, c in enumerate(prepared.case["candidates"])}
    return engine


def prepared_pair(direction=1, role="query", tokenizer=None):
    engine = mock_engine(tokenizer)
    family = _family(17, "engineering", 0)
    recipient_case = diagnostic.original_case(family, 1, direction, 0)
    donor_case = diagnostic.donor_cases(family, recipient_case)[role]
    return (engine, family, engine.prepare(None, family, recipient_case),
            engine.prepare(None, family, donor_case))


@pytest.mark.parametrize("direction", (1, -1))
@pytest.mark.parametrize("role", ("query", "key", "colorsham"))
def test_original_cases_donor_factors_labels_and_no_answer_leak(direction, role):
    engine, family, recipient, donor = prepared_pair(direction, role)
    original = deepcopy(family)
    assert recipient.case == make_case(family, edge=1, direction=direction, palette=0)
    e, d, p = {"query": (1, -direction, 0), "key": (-1, direction, 0),
               "colorsham": (1, direction, 1)}[role]
    assert donor.case == make_case(family, edge=e, direction=d, palette=p)
    assert recipient.prompt.endswith("Answer:") and donor.prompt.endswith("Answer:")
    for prepared in (recipient, donor):
        assert prepared.case["gold"] not in prepared.prompt
        for color in prepared.case["candidates"]:
            assert color not in prepared.prompt
    assert family == original
    assert engine.settings.answer_prefix == "Answer:"


@pytest.mark.parametrize("direction", (1, -1))
def test_geometry_targets_hybrid_cancel_shams_and_v_donor_palette(direction):
    _, family, recipient, _ = prepared_pair(direction)
    donors = diagnostic.donor_cases(family, recipient.case)
    gold_node, hybrid_node = (recipient.case["anchor"]+direction) % 3, (recipient.case["anchor"]-direction) % 3
    for condition in diagnostic.CONDITIONS:
        labels = diagnostic.target_for(family, recipient.case, condition)
        assert labels["gold_node"] == gold_node
        assert labels["gold"] == family["palettes"][0][gold_node]
        assert labels["hybrid_target"] == family["palettes"][0][hybrid_node]
        if condition in ("qflip", "kflip"):
            assert labels["target"] == donors["query"]["gold"] == donors["key"]["gold"]
            assert labels["target_node"] == hybrid_node and labels["target_palette"] == 0
            assert labels["target"] != labels["gold"]
        elif condition == "v_swap":
            assert labels["target"] == donors["colorsham"]["gold"]
            assert labels["target_node"] == gold_node and labels["target_palette"] == 1
            assert labels["target"] not in family["palettes"][0]
        else:
            assert labels["target"] == recipient.case["gold"]
    assert not set(family["palettes"][0]) & set(family["palettes"][1])


@pytest.mark.parametrize("direction", (1, -1))
def test_actual_full_prefix_row_map_at_word_ends_not_guessed_offsets(direction):
    engine, _, recipient, _ = prepared_pair(direction)
    result = diagnostic.row_map(engine, recipient)
    assert result["passed"] and result["actual_full_token_ids"] == list(recipient.input_ids)
    previous = recipient.image_columns[-1]
    for name in diagnostic.ROWS:
        item = result["rows"][name]
        prefix = recipient.prompt[:item["character_end"]]
        actual = engine.tokenizer.encode(prefix, add_special_tokens=True)
        assert item["prefix_token_ids"] == actual
        assert actual == list(recipient.input_ids[:len(actual)])
        assert item["row"] == len(actual)-1 and item["last_prefix_token_id"] == actual[-1]
        assert previous < item["row"] < len(recipient.input_ids)
        previous = item["row"]
    assert recipient.prompt[:result["rows"]["traversal"]["character_end"]].endswith(
        "forward" if direction == 1 else "backward")
    assert recipient.prompt[:result["rows"]["destination"]["character_end"]].endswith("destination")
    assert result["rows"]["finalprefix"]["row"] == recipient.row
    assert recipient.prompt.endswith("Answer:")


@pytest.mark.parametrize("direction", (1, -1))
@pytest.mark.parametrize("role", ("query", "key", "colorsham"))
def test_full_native_alignment_first_then_donor_own_semantic_rows(direction, role):
    engine, _, recipient, donor = prepared_pair(direction, role)
    result = diagnostic.align_rows(engine, recipient, donor)
    assert engine.alignment_calls[0] == (recipient.row, donor.row, True)
    assert len(engine.alignment_calls) == 4
    assert result["passed"] and result["native_full_alignment_passed"]
    assert result["recipient"]["case_id"] == recipient.case["case_id"]
    assert result["donor"]["case_id"] == donor.case["case_id"]
    for name in diagnostic.ROWS:
        assert result["recipient"]["rows"][name]["row"] == result["donor"]["rows"][name]["row"]
    if role == "query":
        assert result["recipient"]["rows"]["traversal"]["last_prefix_token_id"] != result["donor"]["rows"]["traversal"]["last_prefix_token_id"]
    else:
        assert recipient.input_ids == donor.input_ids


class PrefixAlteringTokenizer(WordTokenizer):
    def encode(self, text, add_special_tokens=True):
        ids = super().encode(text, add_special_tokens)
        if text.endswith(("forward", "backward", "destination")):
            ids[-1] += 7  # Boundary rewriting is NOT an offset to guess around.
        return ids


def test_prefix_altering_tokenizer_fail_closed_saves_actual_attempt():
    engine, _, recipient, _ = prepared_pair(tokenizer=PrefixAlteringTokenizer())
    evidence = {}
    with pytest.raises(ValueError, match="not an exact prefix"):
        diagnostic.row_map(engine, recipient, evidence)
    assert not evidence["passed"]
    assert not evidence["rows"]["traversal"]["exact_prefix"]
    assert evidence["rows"]["traversal"]["last_prefix_token_id"] != recipient.input_ids[
        evidence["rows"]["traversal"]["row"]]


@pytest.mark.parametrize("change", ("length", "row", "images", "embeddings", "ids", "prompt"))
def test_full_input_alignment_and_original_prefix_fail_closed(change):
    engine, _, recipient, donor = prepared_pair()
    modified = {
        "length": replace(donor, input_ids=donor.input_ids[:-1]),
        "row": replace(donor, row=donor.row-1),
        "images": replace(donor, image_columns=donor.image_columns[1:]),
        "embeddings": replace(donor, embeddings=SimpleNamespace(shape=(1, len(donor.input_ids)-1, 8))),
        "ids": replace(donor, input_ids=(42, *donor.input_ids[1:])),
        "prompt": replace(donor, prompt=donor.prompt[:-len("Answer:")]),
    }[change]
    with pytest.raises(ValueError, match="alignment|prefix"):
        diagnostic.align_rows(engine, recipient, modified)


class CompensatingTokenizer(WordTokenizer):
    def encode(self, text, add_special_tokens=True):
        ids = super().encode(text, add_special_tokens)
        if "backward" in text:
            at = ids.index(super().encode("backward", False)[0])
            ids.insert(at, 888)  # Make traversal row different.
            if "Reply" in text:
                ids.remove(super().encode("Reply", False)[0])  # Keep FULL length equal.
        return ids


def test_same_full_length_cannot_hide_donor_own_word_row_mismatch():
    engine, _, recipient, donor = prepared_pair(tokenizer=CompensatingTokenizer())
    assert len(recipient.input_ids) == len(donor.input_ids)
    evidence = {}
    with pytest.raises(ValueError, match="donor OWN word row differs"):
        diagnostic.align_rows(engine, recipient, donor, evidence)
    assert evidence["native_full_alignment_passed"] and not evidence["passed"]


@pytest.mark.parametrize("question", (
    "Starting at the circle, follow one arrow forward. What color?",
    "follow one arrow forward follow one arrow forward destination",
    "follow one arrow forward destination destination",
    "follow one arrow backward destination",
))
def test_missing_ambiguous_or_sign_wrong_semantic_words(question):
    engine, family, recipient, _ = prepared_pair()
    altered = engine.prepare(None, family, dict(recipient.case, question=question))
    with pytest.raises(ValueError, match="unique|direction sign"):
        diagnostic.row_map(engine, altered)


def test_question_occurrence_must_be_unique_in_rendered_prompt():
    engine, _, recipient, _ = prepared_pair()
    text = recipient.prompt + recipient.case["question"] + "Answer:"
    engine._prompt = lambda *args: text
    ids = tuple(engine.tokenizer.encode(text))
    changed = replace(recipient, prompt=text, input_ids=ids, row=len(ids)-1,
                      embeddings=SimpleNamespace(shape=(1, len(ids), 8)))
    with pytest.raises(ValueError, match="unique rendered question"):
        diagnostic.row_map(engine, changed)


def test_rows_before_last_actual_image_fail_not_image_offset_guess():
    engine, _, recipient, _ = prepared_pair()
    text = recipient.case["question"] + "<IMG_CONTEXT>"*256 + "Answer:"
    engine._prompt = lambda *args: text
    ids = tuple(engine.tokenizer.encode(text))
    altered = replace(recipient, prompt=text, input_ids=ids, row=len(ids)-1,
        image_columns=tuple(i for i, token in enumerate(ids) if token == 99),
        embeddings=SimpleNamespace(shape=(1, len(ids), 8)))
    with pytest.raises(ValueError, match="precedes last image"):
        diagnostic.row_map(engine, altered)


@pytest.mark.parametrize("value", ("", " ", "all", "0", "-1:0", "0:-1", "0:0,", "0:0,,1:0",
                                     "0:0,00:0", "0:0:0", "1.0:0", "0:0,1:0,2:0,3:0,4:0"))
def test_sites_bounded_explicit_no_dense_or_duplicate(value):
    with pytest.raises(argparse.ArgumentTypeError, match="1–4 unique"):
        diagnostic._site_pairs(value)


def test_lazy_script_and_dependency_free_cli_parse(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy", "PIL", "transformers", "vlm_pilot", "filelock"}:
            raise AssertionError("Import/CLI parsing must not start an optional runtime.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("earlier_routing_lazy_test", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ["--plan", "unused", "--checkpoint", "8b-mpo", "--output", "new", "--sites", "16:0,24:0"]
    args = module._parser().parse_args(required)
    assert args.families == 2 and args.device == "cuda:0"
    assert module._site_pairs(args.sites) == ((16, 0), (24, 0))
    for extra in (["--families", "0"], ["--families", "5"], ["--device", "auto"],
                  ["--allow-downloads"], ["--answer-prefix", ""], ["--split", "discovery"]):
        with pytest.raises(SystemExit):
            module._parser().parse_args(required + extra)
    with pytest.raises(SystemExit):
        module._parser().parse_args(required[:-2])


@dataclass(frozen=True)
class MockSite:
    layer: int
    kv_group: int


@dataclass
class MockAction:
    mode: str
    query: object = None
    key: object = None
    value: object = None
    seed: int = 0


def mock_attention(monkeypatch):
    module = ModuleType("vlm_pilot.attention")
    module.Site, module.RouteAction = MockSite, MockAction
    monkeypatch.setitem(sys.modules, "vlm_pilot.attention", module)


def test_native_site_tuple_and_joint_independent_donor_actions(monkeypatch):
    mock_attention(monkeypatch)
    assert diagnostic.parse_sites("16:0,24:0") == (MockSite(16, 0), MockSite(24, 0))
    caps = {name: object() for name in ("self", "query", "key", "colorsham")}
    assert diagnostic.action_for("native", caps) is None
    action = diagnostic.action_for("qkcancel", caps)
    assert action.mode == "qk" and action.query is caps["query"] and action.key is caps["key"]
    assert action.query is not action.key
    assert diagnostic.action_for("self_qk", caps).query is caps["self"]
    assert diagnostic.action_for("qflip", caps).query is caps["query"]
    assert diagnostic.action_for("kflip", caps).key is caps["key"]
    assert diagnostic.action_for("q_colorsham", caps).query is caps["colorsham"]
    assert diagnostic.action_for("k_colorsham", caps).key is caps["colorsham"]
    assert diagnostic.action_for("v_swap", caps).value is caps["colorsham"]
    assert diagnostic.action_for("routepermute", caps).mode == "route_permute"
    assert diagnostic.action_for("routepermute", caps).seed == 1729


def test_bounds_family_cache_identity_and_prepared_limit(monkeypatch):
    engine, family, recipient, _ = prepared_pair()
    cache = diagnostic.FamilyCache(engine, None, family, (MockSite(16, 0),))
    assert cache.prepare(recipient.case) is cache.prepare(recipient.case)
    with pytest.raises(ValueError, match="Case ID reused"):
        cache.prepare(dict(recipient.case, gold="unrelated"))
    cache.prepared = {str(i): object() for i in range(30)}
    with pytest.raises(ValueError, match="30 originals"):
        cache.prepare(recipient.case)
    for count in (0, 5, True):
        with pytest.raises(ValueError, match="1..4"):
            diagnostic.select_families({"families": [family]}, count)
    with pytest.raises(ValueError, match="no split fallback"):
        diagnostic.select_families({"families": [family, _family(17, "discovery", 1)]}, 2)
    for sites in ([], (), [MockSite(0, 0)], (MockSite(0, 0),)*2,
                  tuple(MockSite(i, 0) for i in range(5))):
        with pytest.raises(ValueError, match="explicit tuple"):
            diagnostic.FamilyCache(engine, None, family, sites)
    assert diagnostic.MAX_CAPTURE_BYTES == 256*1024**2
    assert diagnostic.device_lock_path("cuda:0") == Path(__import__("tempfile").gettempdir()) / "interp-vlm-cuda-0.lock"


def test_single_token_only_no_engine_score_generate_or_candidate_tail():
    engine, _, recipient, _ = prepared_pair()
    assert len(diagnostic.single_candidate_tokens(engine, recipient)) == 6
    engine.suffixes = lambda p: {c: (1, 2) for c in p.case["candidates"]}
    with pytest.raises(ValueError, match="no teacher-forced answer tails"):
        diagnostic.single_candidate_tokens(engine, recipient)


def score_record():
    return {"scores": {"red": -1., "blue": -2.}, "candidate_token_ids": {"red": [1], "blue": [2]},
            "prediction": "red", "tied": False, "last_logit_argmax": 1}


def test_absolute_self_score_parity_and_global_candidate_argmax():
    native = score_record()
    assert diagnostic.self_score_check(native, deepcopy(native))["passed"]
    actual = deepcopy(native)
    actual["scores"]["red"] += 0.9e-4
    assert diagnostic.self_score_check(native, actual)["passed"]
    for change in ("delta", "nan", "candidate_argmax", "vocab_argmax", "events"):
        actual = deepcopy(native)
        if change == "delta":
            actual["scores"]["red"] += 1.1e-4
        elif change == "nan":
            actual["scores"]["red"] = float("nan")
        elif change == "candidate_argmax":
            actual["prediction"] = "blue"
        elif change == "vocab_argmax":
            actual["last_logit_argmax"] = 2
        else:
            actual["candidate_token_ids"]["red"] = [3]
        with pytest.raises(ValueError):
            diagnostic.self_score_check(native, actual)


def routing(mode="capture"):
    return {"mode": mode, "captured_before_intervention": True, "donor_finite": True,
            "attention_row_updated": False, "mass_before_after_maxdiff": 0.,
            "returned_attention_mass_maxdiff": 0., "reconstruction_weight_maxdiff": 0.,
            "conditional_l1": 0., "row_output_delta_rms": 0., "text_weight_maxdiff": 0.}


def test_missing_invalid_native_routing_evidence_fails():
    diagnostic.check_routing(routing(), "capture", 1e-4)
    for changes in ({"text_weight_maxdiff": 1e-10}, {"reconstruction_weight_maxdiff": 2e-4},
                    {"conditional_l1": float("nan")}, {"captured_before_intervention": False},
                    {"mode": "q"}, {"attention_row_updated": None}):
        with pytest.raises(ValueError):
            diagnostic.check_routing(dict(routing(), **changes), "capture", 1e-4)


class MockCapture:
    def __init__(self, prepared, row, site, size=1):
        self.row, self.site, self.sequence_length = row, site, len(prepared.input_ids)
        self.image_columns = prepared.image_columns
        self.payload = SimpleNamespace(numel=lambda: size, element_size=lambda: 1,
                                       device=SimpleNamespace(type="cpu"))

    def validate(self):
        return None

    @property
    def metadata(self):
        return {"row": self.row, "sequence_length": self.sequence_length,
                "image_columns": list(self.image_columns), "site": diagnostic._label(self.site)}


def test_capture_row_only_replace_keeps_full_embeddings_ids_pixels_and_byte_bound():
    engine, family, recipient, _ = prepared_pair()
    site = MockSite(16, 0)
    cache = diagnostic.FamilyCache(engine, None, family, (site,))
    prepared = cache.prepare(recipient.case)
    mapped = diagnostic.row_map(engine, prepared)
    earlier = mapped["rows"]["traversal"]["row"]
    calls = []

    def forward(view, *, sites):
        calls.append(view)
        assert sites == (site,) and view.row == earlier
        assert view.embeddings is prepared.embeddings and view.input_ids is prepared.input_ids
        assert view.pixels is prepared.pixels and view.prompt is prepared.prompt
        return None, {site: MockCapture(view, view.row, site)}, {site: routing()}

    engine.forward = forward
    capture = cache.capture(prepared, earlier, site)
    assert capture is cache.capture(prepared, earlier, site) and len(calls) == 1
    assert prepared.row == len(prepared.input_ids)-1
    with pytest.raises(ValueError, match="Release CPU captures"):
        cache.capture(prepared, prepared.row, site)
    cache.clear_captures()
    assert cache.bytes == 0 and not cache.captures
    capture.payload.numel = lambda: diagnostic.MAX_CAPTURE_BYTES+1
    with pytest.raises(ValueError, match="256 MiB"):
        diagnostic.capture_bytes(capture)
    capture.payload.numel = lambda: 1
    capture.payload.device.type = "cuda"
    with pytest.raises(ValueError, match="CPU"):
        diagnostic.capture_bytes(capture)
    cache.clear()
    assert not cache.prepared and not engine._features and engine._family is None


def test_optional_cpu_tensor_lowlevel_scores_last_logits_not_intervention_row():
    torch = pytest.importorskip("torch")
    engine, _, recipient, _ = prepared_pair()
    earlier = diagnostic.row_map(engine, recipient)["rows"]["traversal"]["row"]
    seen = []
    tokens = diagnostic.single_candidate_tokens(engine, recipient)
    winner = recipient.case["candidates"][-1]
    logits = torch.zeros((1, len(recipient.input_ids), 32))
    logits[0, earlier, tokens[recipient.case["candidates"][0]][0]] = 20.
    logits[0, -1, tokens[winner][0]] = 10.

    def forward(view, *, sites, actions):
        seen.append(view)
        assert sites == () and actions is None and view.row == earlier
        assert view.embeddings is recipient.embeddings and view.input_ids is recipient.input_ids
        assert view.pixels is recipient.pixels
        return logits, {}, {}

    engine.forward = forward
    engine.score = engine.generate = lambda *args, **kwargs: pytest.fail("Answer-tail API used")
    result, capture = diagnostic.lowlevel_score(engine, recipient, earlier)
    assert capture is None and len(seen) == 1 and result["prediction"] == winner
    assert result["scored_prompt_row"] == recipient.row and result["intervention_row"] == earlier
    assert result["forward_passes"] == 1 and result["teacher_forced_answer"] is False
    assert result["scores"][winner] == pytest.approx(torch.log_softmax(logits[0, -1].float(), -1)[tokens[winner][0]].item())


def fake_rows(family_ids, sites):
    rows = []
    for family_id in family_ids:
        for direction in (1, -1):
            for row_name in diagnostic.ROWS:
                for site in sites:
                    for condition in diagnostic.CONDITIONS:
                        rows.append({"family_id": family_id, "direction": direction, "row_name": row_name,
                            "site": diagnostic._label(site), "condition": condition, "split": "engineering",
                            "edge": 1, "palette": 0, "status": "completed", "gold_hit": False,
                            "target_hit": condition in ("qflip", "kflip"),
                            "hybrid_hit": condition in ("qflip", "kflip"),
                            "anchor_copy": True, "donor_answer_copy": False,
                            "all_row_site_gates_passed": True, "native_capture_unchanged": True,
                            "gold_used_in_conditioning": False, "source_status": "native_original_full_prompt",
                            "score": {"routing": {diagnostic._label(site): routing()}}})
    return rows


def test_432_exact_schedule_all_in_call_low_floor_and_partial_not_complete():
    sites, ids = (MockSite(16, 0), MockSite(24, 0)), ("f0", "f1", "f2", "f3")
    rows = fake_rows(ids, sites)
    result = diagnostic.summarize(rows, ids, sites, p0_passed=True)
    assert result["complete"] and result["n_result"] == result["expected_n_result"] == 432
    assert result["n_condition"] == 9 and len(result["primary_site_and_row_all_in_call"]) == 6
    for primary in result["primary_site_and_row_all_in_call"]:
        assert primary["unfiltered"] and primary["expected_all_case_n"] == 8
        assert primary["native_gold_hits"] == 0
        assert primary["q_hybrid_hits"] == primary["k_hybrid_hits"] == 8
        assert primary["q_minus_sham_hybrid_hits"] == primary["k_minus_sham_hybrid_hits"] == 8
        assert primary["cancel_hybrid_hits"] == primary["cancel_gold_hits"] == 0
        assert all(n == 8 for n in primary["all_conditions_n"].values())
    assert result["hypothesis_decision"] == "not_automated" and result["conditioning_filter"] == "none"
    assert any("Low native" in text for text in result["cautions"])
    for partial in ([], rows[:-1], rows+[rows[0]], rows[:-1]+[rows[0]]):
        assert not diagnostic.summarize(partial, ids, sites, p0_passed=True)["complete"]
    assert not diagnostic.summarize(rows, ids, sites, p0_passed=False)["complete"]
    assert not diagnostic.summarize(rows, ids, sites, p0_passed=True, failure={"type": "ValueError"})["complete"]
    invalid = deepcopy(rows)
    invalid[0]["gold_used_in_conditioning"] = True
    assert not diagnostic.summarize(invalid, ids, sites, p0_passed=True)["complete"]


def test_failed_family_alignment_yields_no_condition_trials_and_retains_evidence():
    engine, family, _, _ = prepared_pair(tokenizer=PrefixAlteringTokenizer())
    cache = diagnostic.FamilyCache(engine, None, family, (MockSite(16, 0),))
    engine.forward = lambda *args, **kwargs: pytest.fail("Forward before alignment gate")
    evidence = {}
    try:
        with pytest.raises(ValueError, match="not an exact prefix"):
            list(diagnostic.family_results(cache, evidence))
        assert not evidence["passed"] and evidence["source_prompts"]
        assert not evidence["score_preflights"]
        assert not evidence["alignments"]["1"]["self"]["recipient"]["passed"]
        attempted = evidence["actual_row_maps"][diagnostic.original_case(family, 1, 1, 0)["case_id"]]
        assert not attempted["passed"] and attempted["rows"]["traversal"]["prefix_token_ids"]
    finally:
        cache.clear()


def mock_family_execution(monkeypatch):
    """Mock the scoring seam, but exercise real cache, gates, schedule and labels."""
    mock_attention(monkeypatch)
    engine, family, _, _ = prepared_pair()
    sites = (MockSite(16, 0), MockSite(24, 0))
    cache = diagnostic.FamilyCache(engine, None, family, sites)
    events = []

    def native_capture(prepared, row, site):
        capture = MockCapture(prepared, row, site)
        capture.case = prepared.case
        return capture

    def forward(view, *, sites):
        assert isinstance(sites, tuple) and len(sites) == 1
        events.append(("capture", view.case["case_id"], view.row, diagnostic._label(sites[0])))
        assert view.embeddings.shape[1] == len(view.input_ids)
        return None, {sites[0]: native_capture(view, view.row, sites[0])}, {sites[0]: routing()}

    def checked(capture, base):
        assert cache.bytes + diagnostic.capture_bytes(capture) <= diagnostic.MAX_CAPTURE_BYTES
        assert capture.metadata == base.metadata

    def score(actual_engine, prepared, row, *, site=None, action=None):
        assert actual_engine is engine
        events.append(("score", prepared.case["direction"], row,
                       diagnostic._label(site) if site else None, action.mode if action else None))
        # Native scores deliberately have zero binding accuracy; never filter them out.
        native = family["palettes"][0][prepared.case["anchor"]]
        prediction = native
        if action is not None:
            if action.mode == "q" and action.query.case["direction"] != prepared.case["direction"]:
                prediction = diagnostic.target_for(family, prepared.case, "qflip")["target"]
            elif action.mode == "k" and action.key.case["edge"] != prepared.case["edge"]:
                prediction = diagnostic.target_for(family, prepared.case, "kflip")["target"]
            elif action.mode == "qk" and action.query.case != prepared.case:
                prediction = prepared.case["gold"]
            elif action.mode == "v":
                prediction = diagnostic.target_for(family, prepared.case, "v_swap")["target"]
        tokens = diagnostic.single_candidate_tokens(engine, prepared)
        result = {"scores": {c: -1. if c == prediction else -4. for c in tokens},
            "candidate_token_ids": {c: list(t) for c, t in tokens.items()},
            "prediction": prediction, "tied": False, "last_logit_argmax": tokens[prediction][0],
            "routing": {diagnostic._label(site): routing(action.mode if action else "capture")} if site else {},
            "scored_prompt_row": prepared.row, "intervention_row": row,
            "teacher_forced_answer": False, "forward_passes": 1}
        return result, native_capture(prepared, row, site) if site else None

    engine.forward = forward
    cache.check_returned = checked
    monkeypatch.setattr(diagnostic, "lowlevel_score", score)
    return cache, events


def test_all_own_row_site_self_gates_precede_any_trial_full_nine_conditions(monkeypatch):
    cache, events = mock_family_execution(monkeypatch)
    evidence = {}
    generator = diagnostic.family_results(cache, evidence)
    try:
        first = next(generator)
        assert evidence["passed"] and len(evidence["score_preflights"]) == 2*3*2
        assert all(gate["passed"] and gate["native"]["parity"]["passed"]
                   and gate["self_qk"]["parity"]["passed"] for gate in evidence["score_preflights"].values())
        # Native/self gates were actual forwards, not assumed from final-row Engine.preflight.
        scored = [e for e in events if e[0] == "score"]
        assert len(scored) == 2*3 + 2*3*2*2
        rows = [first, *generator]
        report = diagnostic.summarize(rows, (cache.family["family_id"],), cache.sites, p0_passed=True)
        assert report["complete"] and report["n_result"] == 108
        assert {r["condition"] for r in rows} == set(diagnostic.CONDITIONS)
        assert not any(r["gold_hit"] for r in rows if r["condition"] in ("native", "self_qk"))
        for row in rows:
            assert row["score"]["scored_prompt_row"] >= row["actual_row"]
            assert row["self_score_check"] is not None if row["condition"] in ("native", "self_qk") else row["self_score_check"] is None
            if row["condition"] in ("qflip", "kflip", "qkcancel", "v_swap"):
                assert row["target_hit"]
            for donor in row["donors"].values():
                assert donor["own_row_map"]["row"] == row["actual_row"]
                assert len(donor["source_prompt_ids"]) == row["native_capture"]["sequence_length"]
                assert donor["gold_used_in_conditioning"] is False and donor["answer_provided"] is False
            if row["condition"] == "qkcancel":
                assert set(row["donors"]) == {"query", "key"}
        assert not cache.captures and cache.bytes == 0
    finally:
        generator.close()
        cache.clear()


def test_failed_earlier_self_score_retains_raw_gate_and_yields_no_trials(monkeypatch):
    cache, _ = mock_family_execution(monkeypatch)
    original = diagnostic.lowlevel_score

    def changed(engine, prepared, row, *, site=None, action=None):
        result, capture = original(engine, prepared, row, site=site, action=action)
        if action is not None and action.mode == "qk":
            result["scores"][prepared.case["candidates"][0]] += 2e-4
        return result, capture

    monkeypatch.setattr(diagnostic, "lowlevel_score", changed)
    evidence = {}
    try:
        with pytest.raises(ValueError, match="score parity failed"):
            list(diagnostic.family_results(cache, evidence))
        assert not evidence["passed"] and len(evidence["score_preflights"]) == 1
        gate = next(iter(evidence["score_preflights"].values()))
        assert not gate["passed"] and gate["self_qk"]["score"]["scores"]
        assert "parity" not in gate["self_qk"]
        assert not cache.captures and cache.bytes == 0
    finally:
        cache.clear()


def test_cli_alignment_failure_publishes_failed_report_original_inference_and_no_retry(tmp_path, monkeypatch):
    """Fake native API: only a temporary diagnostic output is written by the test."""
    mock_attention(monkeypatch)
    engine, family, _, _ = prepared_pair(tokenizer=PrefixAlteringTokenizer())
    inference = SimpleNamespace(answer_prefix="Answer:", model_dump=lambda **kwargs: {"answer_prefix": "Answer:"})
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("original plan bytes")
    output = tmp_path / "output"
    plan = SimpleNamespace(dataset={"families": [family]}, dataset_sha256="dataset-hash",
        source_sha256="original-package-hash", spec=SimpleNamespace(dataset=tmp_path, inference=inference),
        model_dump=lambda **kwargs: {"original_plan": True})
    engines, preflights, locks = [], [], []

    def construct(checkpoint, effective, *, device, allow_downloads):
        assert effective is inference and effective.answer_prefix == "Answer:"
        assert allow_downloads is False and device == "cpu"
        engines.append(engine)
        return engine

    def preflight(root, scene, case, *, sites):
        preflights.append((case, sites))
        return {"passed": True, "tested_final_sites": [diagnostic._label(s) for s in sites]}

    engine.preflight = preflight
    engine.model_profile, engine.runtime_profile = {"native": True}, {"device": "cpu"}
    engine.forward = lambda *args, **kwargs: pytest.fail("Forward after token alignment failure")
    runtime = ModuleType("vlm_pilot.runtime")
    runtime.Engine = construct
    monkeypatch.setitem(sys.modules, "vlm_pilot.runtime", runtime)
    import vlm_pilot.data as data
    import vlm_pilot.io as io
    import vlm_pilot.plan as planning

    monkeypatch.setattr(data, "load_dataset", lambda root: plan.dataset)
    monkeypatch.setattr(io, "identity", lambda value: "dataset-hash")
    monkeypatch.setattr(planning, "read_plan", lambda path, check_source: plan)
    monkeypatch.setattr(planning, "checkpoint_for", lambda p, name: SimpleNamespace(name=name))

    class Lock:
        def __init__(self, path, timeout):
            locks.append((path, timeout))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    import filelock

    monkeypatch.setattr(filelock, "FileLock", Lock)
    with pytest.raises(ValueError, match="not an exact prefix"):
        diagnostic.main(["--plan", str(plan_file), "--checkpoint", "8b-mpo", "--output", str(output),
                         "--families", "1", "--sites", "16:0,24:0", "--device", "cpu"])
    report = json.loads((output / "report.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    sealed = json.loads((output / "family-0001.json").read_text())
    assert not report["complete"] and report["n_result"] == 0 and report["expected_n_result"] == 108
    assert report["failure"]["type"] == "ValueError" and sealed["status"] == "failed"
    assert not metadata["complete"] and metadata["original_inference_unchanged"]
    assert metadata["effective_inference"]["answer_prefix"] == "Answer:"
    assert metadata["plan"] == {"original_plan": True}
    assert metadata["diagnostic_source_sha256"] == diagnostic._sha(PATH)
    failed = metadata["family_evidence"][family["family_id"]]["alignments"]["1"]["self"]["recipient"]
    assert not failed["passed"] and failed["rows"]["traversal"]["prefix_token_ids"]
    assert len(engines) == len(preflights) == len(locks) == 1 and locks[0][1] == 0
    assert preflights[0][1] == (MockSite(16, 0), MockSite(24, 0))
    assert plan_file.read_text() == "original plan bytes"
    with pytest.raises(FileExistsError):
        diagnostic.main(["--plan", str(plan_file), "--checkpoint", "8b-mpo", "--output", str(output),
                         "--sites", "16:0", "--device", "cpu"])
    assert len(engines) == 1