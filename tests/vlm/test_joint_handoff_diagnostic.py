"""Lightweight contracts for the one-layer joint diagnostic; authored, not run.

Only stdlib geometry/classifier code and pytest are imported. Fake attention and
tensor interfaces exercise action identity, batching, bounds and generation
without importing torch/Transformers/numpy/Pillow or loading any model. These
spies establish dispatch contracts, NOT real checkpoint numerical qualification.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.util
import sys
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vlm_pilot.data import COLORS, SHAPES, _family, make_case

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/vlm_joint_handoff_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("joint_handoff_tests", PATH)
joint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(joint)


@dataclass(frozen=True)
class FakeSite:
    layer: int
    kv_group: int


@dataclass
class FakeAction:
    mode: str
    query: object = None
    seed: int = 0


@pytest.fixture
def fake_attention(monkeypatch):
    """Replace only the lazy import seam, never import real attention dependencies."""
    module = ModuleType("vlm_pilot.attention")
    module.Site, module.RouteAction = FakeSite, FakeAction
    monkeypatch.setitem(sys.modules, "vlm_pilot.attention", module)
    return tuple(FakeSite(24, group) for group in range(8))


def test_exact_adjacent_helper_provenance_and_import_without_optional_dependencies(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "numpy", "PIL", "filelock", "vlm_pilot"}:
            raise AssertionError("Optional dependency imported during import/CLI parsing.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("joint_lazy_contract", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = PATH.with_name("vlm_handoff_diagnostic.py")
    assert Path(module.helper.__file__) == expected
    assert module.HELPER_SHA256 == hashlib.sha256(expected.read_bytes()).hexdigest()
    assert module.FamilyCache.__bases__ == (module.helper.FamilyCache,)
    assert module.MAX_CAPTURE_BYTES == 512 * 1024**2
    assert module.MAX_PREPARED == 10
    classifier, checksum = module.helper.load_classifier()
    assert callable(classifier) and len(checksum) == 64
    args = module._parser().parse_args([
        "--plan", "unused-plan", "--checkpoint", "8b-gspo", "--output", "unused-output", "--layer", "24"])
    assert args.layer == 24 and args.families == 4 and args.device == "cuda:0"


def test_cli_requires_one_external_layer_and_has_exact_family_bounds():
    required = ["--plan", "unused-plan", "--checkpoint", "8b-gspo", "--output", "unused-output"]
    with pytest.raises(SystemExit):
        joint._parser().parse_args(required)
    for layer in ("", "-1", "24,25", "24:0", "24..25", "24.0", "all", "２４", "+24", " 24"):
        with pytest.raises(argparse.ArgumentTypeError):
            joint._layer(layer)
    for size in ("0", "5", "8"):
        with pytest.raises(SystemExit):
            joint._parser().parse_args(required + ["--layer", "24", "--families", size])
    assert joint._parser().parse_args(required + ["--layer", "0", "--families", "1"]).layer == 0
    assert len(joint.CONDITIONS) == 7
    assert joint.CONDITIONS == ("native", "self_q", "shape_q", "shape_q_disjoint", "color_sham",
                                "wrong_anchor_shape_q", "route_permute")


@pytest.mark.parametrize("direction", (1, -1))
def test_paired_and_control_cases_clone_raw_inputs_without_optional_imports(direction, monkeypatch):
    family = _family(17, "engineering", 0)
    raw = deepcopy(family)
    original = make_case(family, edge=1, direction=direction, palette=0)
    color, shape = joint.paired_cases(family, 1, direction, 0)
    donors = joint.donor_cases(family, color)
    for case in (color, shape):
        for key in ("image_id", "edge", "direction", "palette", "anchor", "target"):
            assert case[key] == original[key]
    assert shape["question"] == color["question"].replace("color", "shape")
    assert shape["gold"] == family["nodes"][color["target"]]["shape"]
    assert shape["candidates"] == list(SHAPES)
    assert color["candidates"] == list(COLORS)
    assert color["gold"] not in color["question"]
    assert shape["gold"] not in shape["question"]
    for name in ("shape_q_disjoint", "color_sham", "wrong_anchor_shape_q"):
        donor = donors[name]
        assert donor["image_id"] != color["image_id"] and donor["palette"] == 1
        assert donor["edge"] == 1 and donor["direction"] == direction
    wrong = donors["wrong_anchor_shape_q"]
    assert wrong["anchor"] == color["target"]
    assert wrong["target"] == (wrong["anchor"] + direction) % 3
    for case in donors.values():
        case["candidates"].append("mutation_test")
    color["candidates"].clear()
    assert shape["candidates"] == list(SHAPES) and family == raw


@pytest.mark.parametrize("condition", joint.CONDITIONS)
def test_build_actions_is_one_all_group_map_with_site_query_identity(condition, fake_attention):
    sites = fake_attention
    captures = {s: SimpleNamespace(site=s, q=object()) for s in sites}
    actions = joint.build_actions(condition, captures, sites, 1729)
    if condition == "native":
        assert actions == {}
    else:
        assert tuple(actions) == sites and len(actions) == 8
        for site, action in actions.items():
            if condition == "route_permute":
                assert action.mode == "route_permute" and action.seed == 1729 and action.query is None
            else:
                assert action.mode == "q" and action.query is captures[site]
                assert action.query.site == site
                assert action.query.q is captures[site].q


def test_build_actions_rejects_partial_wrong_and_swapped_capture_keys(fake_attention):
    sites = fake_attention
    captures = {s: SimpleNamespace(site=s) for s in sites}
    for bad in ({s: captures[s] for s in sites[:-1]},
                dict(captures, unrelated=SimpleNamespace(site=sites[0])),
                {FakeSite(23, s.kv_group): captures[s] for s in sites}):
        with pytest.raises(ValueError, match="keys"):
            joint.build_actions("shape_q", bad, sites, 1729)
    swapped = dict(captures)
    swapped[sites[0]], swapped[sites[1]] = swapped[sites[1]], swapped[sites[0]]
    with pytest.raises(ValueError, match="site"):
        joint.build_actions("self_q", swapped, sites, 1729)
    for bad_sites in (sites + (sites[0],), tuple(reversed(sites)), sites[1:],
                      (sites[0], FakeSite(25, 1)), list(sites)):
        with pytest.raises(ValueError):
            joint.build_actions("shape_q", captures, bad_sites, 1729)
    for seed in (-1, 2**63, True, 1.5):
        with pytest.raises(ValueError, match="seed"):
            joint.build_actions("route_permute", captures, sites, seed)
    with pytest.raises(ValueError, match="Unknown"):
        joint.build_actions("weight_swap", captures, sites, 1729)


@pytest.mark.parametrize("groups", (1, 2, 8))
def test_architecture_drives_all_groups_at_exact_external_layer(groups, fake_attention):
    engine = SimpleNamespace(language_model=SimpleNamespace(
        config=SimpleNamespace(num_key_value_heads=groups, num_hidden_layers=36),
        model=SimpleNamespace(layers=[None] * 36)))
    assert joint.all_sites(engine, 24) == tuple(FakeSite(24, i) for i in range(groups))
    for layer in (-1, 36, True):
        with pytest.raises(ValueError):
            joint.all_sites(engine, layer)
    engine.language_model.config.num_key_value_heads = 9
    with pytest.raises(ValueError):
        joint.all_sites(engine, 24)


class FakeTensor:
    """Only the tensor bookkeeping interface used by cache/dispatch contracts."""

    def __init__(self, count=4, *, shape=(1, 3, 2), device="cpu"):
        self.count, self.shape, self.device = count, shape, SimpleNamespace(type=device)

    def numel(self):
        return self.count

    def element_size(self):
        return 4

    def is_contiguous(self):
        return True


@dataclass(frozen=True)
class FakePrepared:
    case: dict
    prompt: str
    input_ids: tuple
    embeddings: FakeTensor
    image_columns: tuple
    row: int


class FakeCapture:
    def __init__(self, site, prepared):
        self.site, self.row = site, prepared.row
        self.sequence_length = prepared.embeddings.shape[1]
        self.image_columns = prepared.image_columns
        self.q, self.input_image = FakeTensor(4), FakeTensor(12)

    def validate(self):
        assert self.q.device.type == self.input_image.device.type == "cpu"

    @property
    def metadata(self):
        return {"site": {"layer": self.site.layer, "kv_group": self.site.kv_group}, "row": self.row,
                "sequence_length": self.sequence_length, "image_columns": list(self.image_columns)}


def diagnostic(mode="capture"):
    return {"mode": mode, "captured_before_intervention": True, "donor_finite": True,
            "mass_before_after_maxdiff": 0., "returned_attention_mass_maxdiff": 0.,
            "reconstruction_weight_maxdiff": 0., "conditional_l1": 0.2 if mode == "q" else 0.,
            "row_output_delta_rms": 0.1 if mode == "q" else 0., "text_weight_maxdiff": 0.,
            "attention_row_updated": mode == "q"}


class SpyEngine:
    """Fake score results; captures are not scientific tensors or probabilities."""

    def __init__(self):
        self.settings = SimpleNamespace(atol=1e-4, answer_prefix="", max_context_tokens=1024)
        self._features, self._family = {}, None
        self.forward_calls, self.score_calls = [], []

    def prepare(self, root, family, case, *, profile):
        assert profile == "common"
        return FakePrepared(case, case["question"], (1, 3, 7), FakeTensor(), (0, 1), 2)

    def forward(self, prepared, *, sites=(), actions=None):
        self.forward_calls.append((prepared, sites, actions))
        mode = next(iter(actions.values())).mode if actions else "capture"
        return None, {s: FakeCapture(s, prepared) for s in sites}, {s: diagnostic(mode) for s in sites}

    def score(self, prepared, *, sites=(), actions=None):
        self.score_calls.append((prepared, sites, actions))
        candidates = prepared.case["candidates"]
        scores = {name: -1. - index for index, name in enumerate(candidates)}
        mode = next(iter(actions.values())).mode if actions else "capture"
        result = {"scores": scores, "prediction": candidates[0], "tied": False,
                  "candidate_token_ids": {name: [index + 4] for index, name in enumerate(candidates)},
                  "input_ids": list(prepared.input_ids), "rendered_prompt": prepared.prompt,
                  "row": prepared.row, "image_columns": list(prepared.image_columns),
                  "routing": {f"{s.layer}:{s.kv_group}": diagnostic(mode) for s in sites}}
        return result, {s: FakeCapture(s, prepared) for s in sites}


@pytest.fixture
def fake_tensor_import(monkeypatch):
    module = ModuleType("torch")
    module.equal = lambda left, right: (left.count, left.shape) == (right.count, right.shape)
    module.no_grad = nullcontext
    module.tensor = lambda values, **kwargs: values
    module.cat = lambda values, dimension: FakeTensor(
        shape=(1, sum(value.shape[1] for value in values), values[0].shape[2]))
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def test_cache_captures_all_eight_groups_in_one_forward_and_counts_hidden_payload(fake_attention):
    engine, sites = SpyEngine(), fake_attention
    cache = joint.FamilyCache(engine, ROOT, {"family_id": "f0"}, sites)
    prepared = cache.prepare({"case_id": "p0", "question": "pre-answer"})
    first = cache.capture(prepared, sites[3])
    assert len(engine.forward_calls) == 1 and engine.forward_calls[0][1] is sites
    assert len(cache.captures) == 8 and cache.bytes == 8 * (4 + 12) * 4
    assert first is cache.capture_all(prepared)[sites[3]]
    assert len(engine.forward_calls) == 1
    cache.clear()
    assert not cache.prepared and not cache.captures and not cache.capture_diagnostics and cache.bytes == 0


def test_cache_bounds_fail_atomically_and_prepared_limit_is_inherited(fake_attention, monkeypatch):
    engine, sites = SpyEngine(), fake_attention
    cache = joint.FamilyCache(engine, ROOT, {"family_id": "f0"}, sites)
    prepared = cache.prepare({"case_id": "p0", "question": "pre-answer"})
    monkeypatch.setattr(joint, "MAX_CAPTURE_BYTES", 8 * 64 - 1)
    with pytest.raises(ValueError, match="512 MiB"):
        cache.capture_all(prepared)
    assert cache.bytes == 0 and not cache.captures
    for index in range(1, 10):
        cache.prepare({"case_id": f"p{index}", "question": "pre-answer"})
    assert len(cache.prepared) == 10
    with pytest.raises(ValueError, match="fixed bound"):
        cache.prepare({"case_id": "p10", "question": "pre-answer"})
    with pytest.raises(ValueError, match="different"):
        cache.prepare({"case_id": "p0", "question": "changed"})


def test_cache_rejects_unselected_foreign_and_partial_entries(fake_attention):
    engine, sites = SpyEngine(), fake_attention
    cache = joint.FamilyCache(engine, ROOT, {"family_id": "f0"}, sites)
    prepared = cache.prepare({"case_id": "p0", "question": "pre-answer"})
    with pytest.raises(ValueError, match="Unselected"):
        cache.capture(prepared, FakeSite(24, 8))
    foreign = engine.prepare(ROOT, cache.family, prepared.case, profile="common")
    with pytest.raises(ValueError, match="exact"):
        cache.capture(foreign, sites[0])
    cache.captures[("p0", sites[0])] = FakeCapture(sites[0], prepared)
    with pytest.raises(ValueError, match="Partial"):
        cache.capture(prepared, sites[0])
    assert not engine.forward_calls


@pytest.mark.parametrize("changes", ({"donor_finite": False}, {"text_weight_maxdiff": 1e-10},
                                    {"conditional_l1": float("nan")},
                                    {"row_output_delta_rms": float("inf")},
                                    {"mass_before_after_maxdiff": 2e-4},
                                    {"attention_row_updated": None}))
def test_every_group_invariant_is_checked_not_just_first(changes, fake_attention):
    sites = fake_attention
    values = {s: diagnostic("q") for s in sites}
    joint.check_all_routing(values, sites, "q", 1e-4)
    values[sites[-1]].update(changes)
    with pytest.raises(ValueError):
        joint.check_all_routing(values, sites, "q", 1e-4)
    with pytest.raises(ValueError, match="Missing/extra"):
        joint.check_all_routing({s: diagnostic() for s in sites[:-1]}, sites, "capture", 1e-4)


def test_joint_candidate_preflight_uses_unhooked_then_two_all_group_scores(
        fake_attention, fake_tensor_import, monkeypatch):
    engine, sites = SpyEngine(), fake_attention
    cache = joint.FamilyCache(engine, ROOT, {"family_id": "f0"}, sites)
    monkeypatch.setattr(joint, "require_alignment", lambda *args: {"passed": True})
    classify, _ = joint.helper.load_classifier()
    probe = {"case_id": "p0", "question": "pre-answer", "candidates": ["red", "blue", "Red", "Blue"]}
    report = {}
    joint.candidate_preflight(cache, probe, classify, report)
    assert report["passed"] and report["joint"]
    assert report["atol"] == 1e-4 and report["rtol"] == 0
    assert len(engine.forward_calls) == 1
    assert len(engine.score_calls) == 3
    assert engine.score_calls[0][1:] == ((), None)
    assert engine.score_calls[1][1] is sites and engine.score_calls[1][2] == {}
    assert engine.score_calls[2][1] is sites and set(engine.score_calls[2][2]) == set(sites)
    assert all(a.mode == "q" for a in engine.score_calls[2][2].values())


def test_family_dispatch_never_scores_donor_answers_and_audits_only_first_direction(
        fake_attention, fake_tensor_import, monkeypatch):
    engine, sites, family = SpyEngine(), fake_attention, _family(17, "engineering", 0)
    cache = joint.FamilyCache(engine, ROOT, family, sites)
    alignments, audits = [], []

    def alignment(engine, recipient, donor):
        alignments.append((recipient, donor))
        return {"passed": True, "full_input_ids_checked": True}

    def audit(engine, prepared, selected, actions, shapes, **kwargs):
        audits.append((prepared, selected, actions))
        return {"status": "completed", "settings": {"max_new_tokens": 32, "joint_patched": True}}

    monkeypatch.setattr(joint, "require_alignment", alignment)
    monkeypatch.setattr(joint, "patched_greedy_audit", audit)
    classify, _ = joint.helper.load_classifier()
    rows = list(joint.family_results(cache, classify, audit=True))
    assert len(rows) == 14 and len(engine.score_calls) == 16
    assert len(engine.forward_calls) == 10 and len(cache.prepared) == 10
    assert len(alignments) == 10 and len(audits) == 7
    assert all(prepared.case["answer_kind"] == "color" for prepared, _, _ in engine.score_calls)
    assert all(prepared.case["direction"] == 1 and selected is sites for prepared, selected, _ in audits)
    for prepared, selected, actions in engine.score_calls:
        assert prepared.case["candidates"] == list(COLORS) + [c.title() for c in COLORS]
        assert selected in ((), sites)
        if selected:
            assert actions == {} or set(actions) == set(sites)
    for row in rows:
        assert row["sites"] == list(joint._site_labels(sites)) and row["joint"]
        assert row["target_gold"] == row["case"]["gold"]
        assert row["generation_audit"] is None or row["direction"] == 1
        assert not row["net_projected_row_delta_available"] and not row["joint_is_proof_of_synergy"]
        if row["donor"] is not None:
            assert not row["donor"]["donor_scored"] and not row["donor"]["response_provided"]
    assert joint.summarize(rows, 1, sites, p0_passed=True, checkpoint="8b-gspo")["complete"]
    raw = deepcopy(rows)
    repeated = joint.summarize(rows, 1, sites, p0_passed=True)
    assert rows == raw and repeated["expected_n_result"] == 14


@pytest.mark.parametrize("direction", (1, -1))
def test_wrong_role_hybrid_and_disjoint_copy_use_colors_not_shape_gold(direction):
    family = _family(17, "engineering", 0)
    color, _ = joint.paired_cases(family, 1, direction, 0)
    donors = joint.donor_cases(family, color)
    wrong = SimpleNamespace(case=donors["wrong_anchor_shape_q"])
    targets = joint.counterfactuals(family, color, wrong)
    recipient_wrong = family["palettes"][0][wrong.case["target"]]
    assert targets["target_gold"] == color["gold"]
    assert targets["recipient_color_at_donor_selected_node"] == recipient_wrong
    assert recipient_wrong != color["gold"] and recipient_wrong != wrong.case["gold"]
    grading = {g: recipient_wrong for g in joint.GRADES}
    outcomes = joint._outcomes(grading, targets, "wrong_anchor_shape_q")
    assert all(v["wrong_role_counterfactual_hit"] and not v["hit"] for v in outcomes.values())
    assert all(not v["donor_answer_copy"] for v in outcomes.values())
    donor = SimpleNamespace(case=donors["shape_q_disjoint"])
    targets = joint.counterfactuals(family, color, donor)
    assert targets["recipient_color_at_donor_selected_node"] == color["gold"]
    assert targets["donor_payload_color"] not in family["palettes"][0]
    grading = {g: targets["donor_payload_color"] for g in joint.GRADES}
    assert all(v["disjoint_donor_copy"] and not v["hit"] for v in
               joint._outcomes(grading, targets, "shape_q_disjoint").values())


def test_alias_classifier_keeps_ties_none_and_does_not_mutate_raw_score():
    classify, _ = joint.helper.load_classifier()
    result = {"scores": {"red": -2., "Red": -3., "blue": -2., "Blue": -3.},
              "candidate_token_ids": {"red": [1], "Red": [2], "blue": [3], "Blue": [4]},
              "prediction": None, "tied": True}
    raw = deepcopy(result)
    grading = classify(result, ["red", "blue"])
    assert all(grading[g] is None for g in joint.GRADES)
    assert result == raw
    assert joint.helper.self_score_check(result, deepcopy(result), classify, ["red", "blue"])["passed"]


@pytest.mark.parametrize("text,eos,valid,correct,shape", (
    ("red", True, True, True, False), (" red\n", True, True, True, False),
    ("Red", True, False, False, False), ("red.", True, False, False, False),
    ("red", False, False, False, False), ("triangle", True, False, False, True),
    ("The triangle is red", True, False, False, False)))
def test_raw_greedy_delivery_is_separate_from_alias_or_shape_detection(text, eos, valid, correct, shape):
    result = joint.audit_text(text, text + ("<eos>" if eos else ""), [7, 2] if eos else [7],
                              2, ["red", "blue"], "red", list(SHAPES))
    assert result["valid_bare_lower"] is valid and result["exact_correct"] is correct
    assert result["raw_shape_answer_detected"] is shape
    if "triangle" in text:
        assert result["raw_shape_mentions"] == ["triangle"]
    assert not result["raw_shape_detection_is_correctness"]


def test_patched_greedy_recomputes_same_all_site_actions_at_original_row(
        fake_attention, fake_tensor_import):
    sites, engine = fake_attention, SpyEngine()
    prepared = engine.prepare(ROOT, {}, {"case_id": "p0", "question": "pre-answer",
        "gold": "red", "candidates": ["red", "blue", "Red", "Blue"]}, profile="common")
    actions = joint.build_actions("shape_q", {s: FakeCapture(s, prepared) for s in sites}, sites, 1729)
    calls = []

    class Logits:
        def __init__(self, token):
            self.token = token

        def __getitem__(self, index):
            return self

        def argmax(self):
            return SimpleNamespace(item=lambda: self.token)

    def forward(current, *, sites, actions):
        calls.append((current, sites, actions))
        token = 7 if len(calls) == 1 else 2
        return Logits(token), {s: FakeCapture(s, current) for s in sites}, {s: diagnostic("q") for s in sites}

    engine.forward, engine.device = forward, "cpu"
    engine.language_model = SimpleNamespace(get_input_embeddings=lambda: lambda ids: FakeTensor(shape=(1, 1, 2)))
    engine.tokenizer = SimpleNamespace(eos_token_id=2, decode=lambda tokens, **kwargs: "red")
    updates = []
    result = joint.patched_greedy_audit(engine, prepared, sites, actions, list(SHAPES),
                                      progress=lambda stage, **details: updates.append(details))
    assert len(calls) == 2 and result["eos"] and result["exact_correct"]
    assert calls[0][0] is prepared
    assert calls[1][0].embeddings.shape[1] == prepared.embeddings.shape[1] + 1
    assert all(current.row == prepared.row and current.input_ids == prepared.input_ids
               and selected is sites and applied is actions for current, selected, applied in calls)
    assert len(updates) == 2 and result["settings"]["max_new_tokens"] == 32
    assert all(set(step["routing"]) == set(joint._site_labels(sites)) for step in result["steps"])
    assert not result["settings"]["net_projected_row_delta_available"]


def test_empty_partial_failed_sources_cannot_complete_and_default_is_56(fake_attention):
    for p0 in (False, True):
        report = joint.summarize([], 4, fake_attention, p0_passed=p0, checkpoint="8b-gspo")
        assert not report["complete"] and report["n_result"] == 0
        assert report["expected_n_result"] == 56 and report["expected_n_generation_audit"] == 7
        assert report["checkpoint"] == "8b-gspo" and len(report["cells"]) == 2 * 7 * 3
        assert report["confirmation"] == report["training"] == "not_attempted"
        assert not report["joint_is_proof_of_synergy"]
    secret = RuntimeError("SECRET request URL and token must never be serialized")
    failure = joint._failure(secret, "joint_conditions")
    assert "SECRET" not in str(failure) and "token" not in str(failure)
    assert failure["type"] == "RuntimeError"
    assert not joint.summarize([], 4, fake_attention, p0_passed=True, failure=failure)["complete"]


def test_full_default_schedule_requires_56_unique_rows_and_seven_first_case_audits(fake_attention):
    sites, rows = fake_attention, []
    names = ("hit", "anchor_copy", "donor_answer_copy", "donor_payload_color_copy",
             "recipient_color_at_donor_selected_node_hit", "wrong_role_counterfactual_hit",
             "disjoint_donor_copy")
    for family in range(4):
        for direction in (1, -1):
            for condition in joint.CONDITIONS:
                rows.append({"family_id": f"f{family}", "direction": direction, "condition": condition,
                             "edge": 1, "palette": 0, "sites": list(joint._site_labels(sites)),
                             "joint": True, "native_capture_unchanged": True, "status": "completed",
                             "source_status": "native_preanswer_capture",
                             "score": {"routing": {label: diagnostic() for label in joint._site_labels(sites)}},
                             "outcomes": {g: {name: False for name in names} for g in joint.GRADES},
                             "generation_audit": ({"status": "completed", "settings": {
                                 "max_new_tokens": 32, "joint_patched": True}}
                                 if family == 0 and direction == 1 else None)})
    original = deepcopy(rows)
    result = joint.summarize(rows, 4, sites, p0_passed=True, checkpoint="8b-gspo")
    assert result["complete"] and result["n_result"] == result["expected_n_result"] == 56
    assert result["n_generation_audit"] == 7 and result["generation_audit_complete"]
    assert all(cell["n_family"] == 4 and cell["n_result"] == 4 for cell in result["cells"])
    assert rows == original
    for bad in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
        assert not joint.summarize(bad, 4, sites, p0_passed=True)["complete"]
    assert not joint.summarize(rows, 4, sites, p0_passed=False)["complete"]
    assert not joint.summarize(rows, 4, sites, p0_passed=True,
                               failure={"type": "RuntimeError"})["complete"]
    for change in ({"native_capture_unchanged": False}, {"sites": ["24:0"]},
                   {"generation_audit": None}, {"edge": -1}, {"joint": False}):
        bad = deepcopy(rows)
        bad[0].update(change)
        assert not joint.summarize(bad, 4, sites, p0_passed=True)["complete"]
    bad = deepcopy(rows)
    bad[7]["generation_audit"], bad[0]["generation_audit"] = bad[0]["generation_audit"], None
    assert not joint.summarize(bad, 4, sites, p0_passed=True)["complete"]