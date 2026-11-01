"""Offline authored-only contracts; no pretrained model, GPU or network needed."""
from __future__ import annotations

import argparse
import builtins
import importlib.util
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/vlm_text_state_diagnostic.py"


def load_module():
    spec = importlib.util.spec_from_file_location("text_state_diagnostic_tests", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnostic = load_module()


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")

    class TinyLayer(torch.nn.Module):
        def __init__(self, tuple_output=False):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((4,), 2.0))
            self.tuple_output, self.rest, self.fail = tuple_output, object(), False
            self.last = None

        def forward(self, value):
            if self.fail:
                raise RuntimeError("native failure")
            self.last = value * self.weight + 1
            return (self.last, self.rest, None) if self.tuple_output else self.last

    class TinyLanguage(torch.nn.Module):
        def __init__(self, tuple_output=False):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([TinyLayer(tuple_output), TinyLayer()])
            self.model.norm = torch.nn.LayerNorm(4)
            self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=2, use_cache=False,
                                          output_hidden_states=False, output_attentions=False)

        def forward(self, value, **kwargs):
            for layer in self.model.layers:
                result = layer(value)
                value = result[0] if isinstance(result, tuple) else result
            return value

    return torch, TinyLanguage, TinyLayer


@pytest.mark.parametrize("tuple_output", (False, True))
def test_capture_self_owned_cpu_postblock_and_restore(tiny, tuple_output):
    torch, Language, _ = tiny
    model = Language(tuple_output).eval()
    x = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4).requires_grad_()
    native = model(x)
    weights = {name: (id(p), p.detach().clone()) for name, p in model.named_parameters()}
    forward = model.model.layers[0].forward.__func__
    with diagnostic.ResidualSession(model, 0, 2) as session:
        captured_result = model(x)
    assert torch.equal(native, captured_result)
    assert torch.equal(session.capture, (x.detach() * 2 + 1)[:, 2:3])
    assert session.capture.shape == (1, 1, 4) and session.capture.device.type == "cpu"
    assert not session.capture.requires_grad and session.capture._base is None
    assert session.capture.data_ptr() != model.model.layers[0].last[:, 2:3].data_ptr()
    donor = session.capture.clone().requires_grad_()
    with diagnostic.ResidualSession(model, 0, 2, donor) as patched:
        result = model(x)
    assert torch.equal(native, result)
    assert diagnostic.exact_vector(session.capture, patched.capture)["passed"]
    assert patched.diagnostic["output_delta_rms"] == patched.diagnostic["embedding_delta_rms"] == 0
    assert model.model.layers[0].forward.__func__ is forward
    for name, p in model.named_parameters():
        assert id(p) == weights[name][0] and torch.equal(p, weights[name][1])
    session.capture.zero_()
    assert torch.equal(model(x), native)
    assert not model._forward_pre_hooks and not model.model.layers[0]._forward_hooks


@pytest.mark.parametrize("tuple_output", (False, True))
def test_exact_substitution_row_only_preserves_native_tuple_rest(tiny, tuple_output):
    torch, Language, _ = tiny
    model = Language(tuple_output).eval()
    layer = model.model.layers[0]
    x = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    donor = torch.tensor([[[8., 9., 10., 11.]]], dtype=torch.float64)
    original_donor = donor.clone()
    with diagnostic.ResidualSession(model, 0, 3, donor) as session:
        donor.zero_()  # Session owns its input, including CPU donors.
        result = layer(x)
    edited = result[0] if tuple_output else result
    assert torch.equal(edited[:, 3:4], original_donor.float())
    assert torch.equal(session.capture, layer.last[:, 3:4])
    assert edited.data_ptr() != layer.last.data_ptr()
    for row in (0, 1, 2, 4, 5):
        assert torch.equal(edited[:, row], layer.last[:, row])
    assert session.diagnostic["output_delta_rms"] > 0
    if tuple_output:
        assert type(result) is tuple and result[1] is layer.rest and result[2] is None
    assert torch.equal(layer(x)[0] if tuple_output else layer(x), x * 2 + 1)


def test_cleanup_native_hook_error_partial_setup_and_nesting(tiny, monkeypatch):
    torch, Language, _ = tiny
    model = Language().eval()
    layer = model.model.layers[0]
    x = torch.ones(1, 3, 4)
    with diagnostic.ResidualSession(model, 0, 1):
        with pytest.raises(ValueError, match="Nested"):
            with diagnostic.ResidualSession(model, 0, 1):
                pass
        model(x)
    assert not diagnostic._ACTIVE
    layer.fail = True
    with pytest.raises(RuntimeError, match="native failure"):
        with diagnostic.ResidualSession(model, 0, 1):
            model(x)
    layer.fail = False
    with pytest.raises(ValueError, match="postblock"):
        with diagnostic.ResidualSession(model, 0, 9):
            model(x)
    register = layer.register_forward_hook
    monkeypatch.setattr(layer, "register_forward_hook", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("register failure")))
    enabled = torch.is_grad_enabled()
    with pytest.raises(RuntimeError, match="register failure"):
        with diagnostic.ResidualSession(model, 0, 1):
            pass
    assert torch.is_grad_enabled() is enabled
    assert not model._forward_pre_hooks and not layer._forward_hooks and not diagnostic._ACTIVE
    monkeypatch.setattr(layer, "register_forward_hook", register)
    with diagnostic.ResidualSession(model, 0, 1):
        model(x)


def test_reject_donor_shape_nan_inf_sparse_and_oversize(tiny):
    torch, Language, _ = tiny
    model = Language().eval()
    for donor in (torch.zeros(4), torch.zeros(1, 2, 4), torch.zeros(1, 1, 5),
                  torch.full((1, 1, 4), float("nan")), torch.full((1, 1, 4), float("inf")),
                  torch.zeros(1, 1, 4, dtype=torch.int64), torch.zeros(1, 1, 4).to_sparse()):
        with pytest.raises(ValueError, match="Donor"):
            diagnostic.ResidualSession(model, 0, 1, donor)
    model.config.hidden_size = (1 << 20) // 4 + 1
    with pytest.raises(ValueError, match="1 MiB"):
        diagnostic.ResidualSession(model, 0, 1, torch.zeros(1, 1, model.config.hidden_size))


def test_eval_layout_recorders_and_no_ambient_rng(tiny):
    torch, Language, _ = tiny
    model = Language()
    with pytest.raises(ValueError, match="eval"):
        with diagnostic.ResidualSession(model, 0, 1):
            pass
    model.eval()
    before = torch.random.get_rng_state().clone()
    enabled = torch.is_grad_enabled()
    with diagnostic.ResidualSession(model, 0, 1):
        model(torch.ones(1, 3, 4))
    assert torch.equal(before, torch.random.get_rng_state()) and torch.is_grad_enabled() is enabled
    with pytest.raises(ValueError, match="recorders"):
        with diagnostic.ResidualSession(model, 0, 1):
            model(torch.ones(1, 3, 4), output_hidden_states=True)
    for x in (torch.ones(2, 3, 4), torch.ones(1, 3, 5), torch.ones(3, 4)):
        with pytest.raises((ValueError, RuntimeError)):
            with diagnostic.ResidualSession(model, 0, 1):
                model(x)
    assert not model._forward_pre_hooks and not model.model.layers[0]._forward_hooks


def test_stdlib_only_inert_import_and_parser(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy", "transformers", "vlm_pilot", "filelock", "PIL"}:
            raise AssertionError("optional import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = load_module()
    args = module._parser().parse_args(["--plan", "unused", "--checkpoint", "8b-mpo", "--output", "new",
                                       "--layers", "8,16,24,32", "--rows", "task_noun,finalassistant"])
    assert args.families == 4 and not args.audit and args.layers == (8, 16, 24, 32)
    assert args.rows == ("task_noun", "finalassistant") and not module._STARTED
    assert module._load_helper()[1] == module._source_hash(PATH.with_name("vlm_handoff_diagnostic.py"))


@pytest.mark.parametrize("value", ("", "all", "8,08", "-1", "1.0", "1,2,3,4,5", "8,"))
def test_bounded_layers(value):
    with pytest.raises(argparse.ArgumentTypeError):
        diagnostic.parse_layers(value)


@pytest.mark.parametrize("value", ("", "destination", "task_noun,task_noun", "finalassistant,",
                                   "task_noun", "finalassistant", "traversal"))
def test_explicit_rows(value):
    with pytest.raises(argparse.ArgumentTypeError):
        diagnostic.parse_rows(value)


def _row_fixture():
    """Char tokenizer: color/shape equal word lengths; no vocabulary mutation."""
    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [999 if c == "\ufffc" else ord(c) for c in text]

    def prompt(question, image, profile):
        assert image and profile == "common"
        return "system\n" + "\ufffc" * 256 + "\nuser\n" + question + "\nassistant\n"

    @dataclass
    class Prepared:
        case: dict
        prompt: str
        input_ids: tuple
        embeddings: object
        image_columns: tuple
        row: int

    engine = SimpleNamespace(tokenizer=Tokenizer(), img_token=999, _prompt=prompt,
                             settings=SimpleNamespace(answer_prefix=""))

    def prepare(case):
        text = prompt(case["question"], True, "common")
        ids = tuple(engine.tokenizer.encode(text))
        return Prepared(case, text, ids, SimpleNamespace(shape=(1, len(ids), 4)),
                        tuple(i for i, t in enumerate(ids) if t == 999), len(ids) - 1)

    calls = []

    def alignment(a, b, allow_query_change=False):
        calls.append((a.row, b.row, allow_query_change))
        assert a.row == len(a.input_ids) - 1 and b.row == len(b.input_ids) - 1
        if len(a.input_ids) != len(b.input_ids) or a.image_columns != b.image_columns:
            raise ValueError("full alignment length")

    engine.require_alignment = alignment
    return engine, prepare, calls


def test_first_task_noun_wordend_full_alignment_and_no_gold_leak():
    from vlm_pilot.data import _family

    helper, _ = diagnostic._load_helper()
    engine, prepare, calls = _row_fixture()
    family = _family(17, "engineering", 0)
    color, shape = helper.paired_cases(family, 1, 1, 0)
    shape["gold"] = "NEVER_CONDITION_THIS"
    original = deepcopy(shape)
    recipient, donor = prepare(color), prepare(shape)
    aligned = diagnostic.align_rows(helper, engine, recipient, donor)
    assert calls and aligned["all_rows_same_positions"]
    mapping = aligned["recipient_rows"]
    expected_end = recipient.prompt.index("destination color") + len("destination color")
    assert mapping["task_noun"]["character_end"] == expected_end
    assert mapping["finalassistant"]["row"] == recipient.row
    assert mapping["traversal"]["prefix_token_ids"] == aligned["donor_rows"]["traversal"]["prefix_token_ids"]
    assert shape == original and shape["gold"] not in donor.prompt
    assert donor.input_ids == tuple(engine.tokenizer.encode(donor.prompt))
    encode = engine.tokenizer.encode
    engine.tokenizer.encode = lambda text, **kw: encode(text, **kw) + ([1234] if text.endswith("color") else [])
    with pytest.raises(ValueError, match="exact actual"):
        diagnostic.row_map(engine, recipient)


def test_six_conditions_and_geometric_metadata():
    from vlm_pilot.data import _family

    helper, _ = diagnostic._load_helper()
    family = _family(17, "engineering", 0)
    recipient, _ = helper.paired_cases(family, 1, 1, 0)
    sources = helper.donor_cases(family, recipient)
    assert diagnostic.CONDITIONS == ("native", "self", "shape_same", "shape_disjoint", "color_sham", "wrong_anchor_shape")
    assert sources["shape_q"]["image_id"] == recipient["image_id"]
    assert sources["wrong_anchor_shape_q"]["anchor"] == recipient["target"]
    assert sources["wrong_anchor_shape_q"]["palette"] == 1
    assert not set(family["palettes"][0]) & set(family["palettes"][1])


def _complete_rows(families, layers, rows):
    results, structural = [], []
    for family in families:
        for direction in (1, -1):
            for layer in layers:
                structural.append({"family_id": family, "direction": direction, "layer": layer,
                                   "passed": True, "same_image": True, "same_prefix": True, "check": {"passed": True}})
                for name in rows:
                    for condition in diagnostic.CONDITIONS:
                        results.append({"family_id": family, "direction": direction, "layer": layer,
                                        "row_name": name, "row": 10, "condition": condition, "status": "completed",
                                        "native_capture_unchanged": True, "source_status": "native_preanswer_capture",
                                        "self_score_check": {"passed": True}, "generation_audit": None,
                                        "outcomes": {"combined": {"primary_recipient_gold": False}}})
    return results, structural


def test_expected_counts_completion_structural_self_and_optin_audit():
    families, layers, rows = ("f0", "f1"), (8, 16, 24, 32), ("task_noun", "finalassistant")
    results, structural = _complete_rows(families, layers, rows)

    def report(values=results, **changes):
        return diagnostic.summarize(values, families, layers, rows,
                                    **dict({"p0_passed": True, "structural": structural}, **changes))

    summary = report()
    assert summary["complete"] and summary["expected_n_result"] == 2 * 2 * 2 * 4 * 6
    assert summary["n_result"] == 192 and not summary["audit_requested"] and summary["n_audit"] == 0
    assert not report(results[:-1])["complete"] and not report(results + results[:1])["complete"]
    assert not report(p0_passed=False)["complete"] and not report(structural=structural[:-1])["complete"]
    assert not report(audit=True)["complete"] and not report(failure={"type": "Error"})["complete"]
    bad = deepcopy(results)
    bad[0]["self_score_check"] = {"passed": False}
    assert not report(bad)["complete"]
    for result in results:
        if result["family_id"] == "f0" and result["direction"] == 1 and result["layer"] == 24 and result["row_name"] == "finalassistant":
            result["generation_audit"] = {"status": "completed", "token_ids": [151645], "native_eos_id": 151645,
                                          "steps": [{"step": 1, "token_id": 151645}], "finish_reason": "eos",
                                          "settings": {"max_new_tokens": 32, "patched": True, "layer": 24,
                                                       "fixed_row": 10, "full_vocabulary": True,
                                                       "do_sample": False, "use_cache": False}}
    assert report(audit=True)["complete"] and not report()["complete"]
    broken = deepcopy(results)
    for result in broken:
        if result["generation_audit"]:
            result["generation_audit"]["steps"] = []
            break
    assert not report(broken, audit=True)["complete"]


def test_native_score_adapter_scopes_each_tail_forward_and_captures_donors_only(tiny):
    """Use the existing Engine.score implementation, not a duplicate test scorer."""
    from vlm_pilot.runtime import Engine

    torch, Language, _ = tiny

    class Qwen3ForCausalLM(Language):
        """Explicit tiny runtime-adapter stand-in; no pretrained construction."""

        def __init__(self):
            super().__init__()
            self.config._attn_implementation = "eager"
            self.embedding = torch.nn.Embedding.from_pretrained(torch.arange(16.).reshape(4, 4))

        def get_input_embeddings(self):
            return self.embedding

    class FakeEngine:
        score = Engine.score

        def __init__(self):
            self.language_model = Qwen3ForCausalLM().eval()
            self.device, self.calls = torch.device("cpu"), []
            self.settings = SimpleNamespace(answer_prefix="", max_context_tokens=2048)
            self._features, self._family = {}, "f0"

        def suffixes(self, prepared):
            assert prepared.case["answer_kind"] == "color", "donor was scored"
            return {"red": (0, 1), "Red": (2,), "blue": (3,), "Blue": (1,)}

        def forward(self, prepared, *, keep=1, **kwargs):
            self.calls.append((prepared.case["case_id"], prepared.row, prepared.input_ids,
                               prepared.embeddings.shape[1], len(self.language_model.model.layers[0]._forward_hooks)))
            value = self.language_model(prepared.embeddings)
            return value[:, -keep:], {}, {}

    engine = FakeEngine()
    _, prepare, _ = _row_fixture()
    from vlm_pilot.data import _family

    helper, _ = diagnostic._load_helper()
    family = _family(17, "engineering", 0)
    color, shape = helper.paired_cases(family, 1, 1, 0)
    color = dict(color, candidates=["red", "Red", "blue", "Blue"])
    prepared = prepare(color)
    prepared.embeddings = torch.ones(1, len(prepared.input_ids), 4)
    native, _ = engine.score(prepared)
    adapter = diagnostic.ResidualEngine(engine, 0)
    hooked, _ = adapter.score(prepared)
    assert native["scores"] == hooked["scores"] and len(adapter.sessions) == 2
    assert diagnostic.exact_vector(adapter.sessions[0].capture, adapter.sessions[1].capture)["passed"]
    assert engine.calls[-1][1:3] == (prepared.row, prepared.input_ids)
    assert engine.calls[-1][3] == len(prepared.input_ids) + 1 and engine.calls[-1][4] == 1
    assert not engine.language_model._forward_pre_hooks and not engine.language_model.model.layers[0]._forward_hooks
    with pytest.raises(ValueError, match="Qwen3"):
        diagnostic.ResidualEngine(SimpleNamespace(language_model=Language().eval()), 0)
    donor = prepare(dict(shape, gold="NEVER_FORWARD_GOLD"))
    donor.embeddings = torch.ones(1, len(donor.input_ids), 4)
    cache = diagnostic.FamilyCache(helper, engine, Path("unused"), family, (0,))
    cache.prepared[donor.case["case_id"]] = donor
    vector, metadata = cache.capture(donor, 0, donor.row)
    assert vector.shape == (1, 1, 4) and metadata["native_before_edit"]
    assert engine.calls[-1][0] == donor.case["case_id"] and engine.calls[-1][3] == len(donor.input_ids)
    assert donor.case["gold"] not in donor.prompt and vector.requires_grad is False
    cache.clear()
    assert not cache.prepared and not cache.captures and cache.bytes == 0


def test_raw_hash_parity_is_stricter_than_numerical_tolerance(tiny):
    torch, _, _ = tiny
    left = torch.zeros(1, 1, 4)
    right = torch.tensor([[[-0.0, 0.0, 0.0, 0.0]]])
    assert torch.equal(left, right)
    assert diagnostic.vector_metadata(left)["sha256"] != diagnostic.vector_metadata(right)["sha256"]
    with pytest.raises(ValueError, match="bytes differ"):
        diagnostic.exact_vector(left, right)