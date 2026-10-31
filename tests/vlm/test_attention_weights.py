"""Offline CPU engineering contracts, not research predictions or GPU validation.

Tiny random native Qwen3 fixtures exercise the verified HF 4.57.6 interface:
post-RMSNorm RoPE, consecutive-query-head GQA, additive masks, and the eager
(projected output, dense weights) return. No pretrained model is instantiated,
no downloads are made, and no tests are executed by the implementing agent.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, apply_rotary_pos_emb

from vlm_pilot.attention import Capture, RouteAction, RoutingSession, Site
from vlm_pilot.weights import (
    MappingTensorSource,
    SafetensorSource,
    WeightSwap,
    identical_input_replay,
)

IMAGE_COLUMNS = (1, 2, 3)
REVISION = "a" * 40


def _tiny_model(*, sliding: bool = False) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=41, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, attention_dropout=0.0, use_cache=False,
        attn_implementation="eager", use_sliding_window=sliding,
        sliding_window=3, max_window_layers=0,
    )
    # Fixture construction must not change the caller's global RNG state.
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        return Qwen3ForCausalLM(config).to(device="cpu", dtype=torch.float32).eval()


@pytest.fixture
def language_model():
    return _tiny_model()


@pytest.fixture
def prompt():
    return torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)


@pytest.fixture
def site():
    # KV group one controls query heads two AND three, not just one of them.
    return Site(layer=0, kv_group=1)


def _hooks(model):
    return {
        name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
        for name, module in model.named_modules()
    }


def _parameters(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _assert_parameters(model, expected):
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, expected[name]), name


def _inputs(model, prompt):
    """Prepare native layer-zero inputs, including the actual causal additive mask."""
    with torch.no_grad():
        hidden = model.model.layers[0].input_layernorm(model.model.embed_tokens(prompt))
        positions = torch.arange(prompt.shape[1]).unsqueeze(0)
        embeddings = model.model.rotary_emb(hidden, positions)
        future = torch.ones((prompt.shape[1], prompt.shape[1]), dtype=torch.bool).triu(1)
        mask = hidden.new_zeros((1, 1, prompt.shape[1], prompt.shape[1]))
        mask.masked_fill_(future, torch.finfo(hidden.dtype).min)
    return {
        "hidden_states": hidden, "position_embeddings": embeddings,
        "position_ids": positions, "attention_mask": mask, "use_cache": False,
    }


def _capture(model, site, inputs, *, columns=IMAGE_COLUMNS, row=None):
    row = inputs["hidden_states"].shape[1] - 1 if row is None else row
    session = RoutingSession(model, (site,), row=row, image_columns=columns)
    with torch.no_grad(), session:
        output = model.model.layers[site.layer].self_attn(**inputs)
    return session.captures[site], output, session.diagnostics[site]


def _intervene(model, site, inputs, action, *, columns=IMAGE_COLUMNS, row=None):
    row = inputs["hidden_states"].shape[1] - 1 if row is None else row
    session = RoutingSession(model, (site,), row=row, image_columns=columns, actions={site: action})
    with torch.no_grad(), session:
        output = model.model.layers[site.layer].self_attn(**inputs)
    return session.captures[site], output, session.diagnostics[site]


def _selected(weights, capture):
    start = capture.site.kv_group * capture.q.shape[1]
    return weights[:, start : start + capture.q.shape[1], capture.row : capture.row + 1]


def _outside(weights, capture):
    columns = [index for index in range(capture.sequence_length) if index not in capture.image_columns]
    return _selected(weights, capture)[..., columns]


def _expected_delta(model, site, context_delta):
    dim, group_queries = model.config.head_dim, 2
    start = site.kv_group * group_queries * dim
    weight = model.model.layers[site.layer].self_attn.o_proj.weight[:, start : start + group_queries * dim]
    flat = context_delta.transpose(1, 2).reshape(1, 1, group_queries * dim)
    return F.linear(flat, weight, bias=None)


def _source(model, *, changed=True):
    tensors = {
        f"language_model.{name}": parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    if changed:
        for layer in range(model.config.num_hidden_layers):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                key = f"language_model.model.layers.{layer}.self_attn.{name}.weight"
                tensor = tensors[key]
                delta = torch.linspace(0.02, 0.11, tensor.numel()).reshape(tensor.shape)
                tensors[key] = tensor + delta
    return MappingTensorSource(tensors), tensors


def test_capture_matches_native_normalization_rope_gqa_and_cpu_ownership(language_model, prompt, site):
    inputs = _inputs(language_model, prompt)
    attn = language_model.model.layers[0].self_attn
    hooks, parameters = _hooks(language_model), _parameters(language_model)
    original_forward = attn.forward.__func__
    capture, output, diagnostics = _capture(language_model, site, inputs)
    assert isinstance(capture, Capture)
    with torch.no_grad():
        native = attn(**inputs)
        hidden = inputs["hidden_states"]
        q = attn.q_norm(attn.q_proj(hidden).view(1, 7, 4, 8)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden).view(1, 7, 2, 8)).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *inputs["position_embeddings"])
        v = attn.v_proj(hidden).view(1, 7, 2, 8).transpose(1, 2)
    assert torch.equal(output[0], native[0])
    assert torch.equal(output[1], native[1])
    torch.testing.assert_close(capture.q, q[:, 2:4, -1:], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(capture.k, k[:, 1:2, IMAGE_COLUMNS], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(capture.v, v[:, 1:2, IMAGE_COLUMNS], atol=0, rtol=0)
    native_image = native[1][:, 2:4, -1:, IMAGE_COLUMNS]
    torch.testing.assert_close(capture.mass, native_image.sum(-1, keepdim=True), atol=0, rtol=0)
    torch.testing.assert_close(capture.conditional, native_image / capture.mass, atol=0, rtol=0)
    assert capture.q.shape == (1, 2, 1, 8) and capture.k.shape == (1, 1, 3, 8)
    assert capture.metadata["sequence_length"] == 7 and capture.site == site
    for name in ("q", "k", "v", "mass", "conditional", "cos_row", "sin_image", "input_rows", "input_image"):
        assert getattr(capture, name).device.type == "cpu"
        assert not getattr(capture, name).requires_grad
    saved_input, saved_cos = capture.input_rows.clone(), capture.cos_row.clone()
    inputs["hidden_states"].zero_()
    inputs["position_embeddings"][0].zero_()
    assert torch.equal(capture.input_rows, saved_input) and torch.equal(capture.cos_row, saved_cos)
    assert diagnostics["row_output_delta_rms"] == 0
    assert diagnostics["reconstruction_weight_maxdiff"] < 1e-6
    assert _hooks(language_model) == hooks and attn.forward.__func__ is original_forward
    _assert_parameters(language_model, parameters)


@pytest.mark.parametrize("mode", ["q", "k", "qk", "v", "distribution"])
def test_self_clamps_are_bitexact_p0(language_model, prompt, site, mode):
    inputs = _inputs(language_model, prompt)
    capture, native, _ = _capture(language_model, site, inputs)
    fields = {"q": {"query": capture}, "k": {"key": capture},
              "qk": {"query": capture, "key": capture}, "v": {"value": capture},
              "distribution": {"distribution": capture}}
    _, output, diagnostics = _intervene(language_model, site, inputs, RouteAction(mode, **fields[mode]))
    assert torch.equal(output[0], native[0]) and torch.equal(output[1], native[1])
    assert diagnostics["row_output_delta_rms"] == 0
    assert diagnostics["mass_before_after_maxdiff"] == 0


def test_full_language_model_capture_and_self_logits_are_bitexact(language_model, prompt, site):
    with torch.no_grad():
        native = language_model(input_ids=prompt, use_cache=False).logits
        capture_session = RoutingSession(language_model, (site,), row=6, image_columns=IMAGE_COLUMNS)
        with capture_session:
            captured = language_model(input_ids=prompt, use_cache=False).logits
        own = capture_session.captures[site]
        with RoutingSession(language_model, (site,), row=6, image_columns=IMAGE_COLUMNS,
                            actions={site: RouteAction("qk", query=own, key=own)}):
            self_clamp = language_model(input_ids=prompt, use_cache=False).logits
    assert torch.equal(native, captured) and torch.equal(native, self_clamp)


@pytest.mark.parametrize("mode", ["q", "k", "qk"])
def test_qk_changes_only_image_conditional_keeps_recipient_values_and_mass(language_model, prompt, site, mode):
    inputs = _inputs(language_model, prompt)
    recipient, native, _ = _capture(language_model, site, inputs)
    donor_prompt = prompt.clone()
    donor_prompt[0, -1], donor_prompt[0, 2] = 19, 23
    donor, _, _ = _capture(language_model, site, _inputs(language_model, donor_prompt))
    fields = {"q": {"query": donor}, "k": {"key": donor}, "qk": {"query": donor, "key": donor}}
    captured, output, diagnostics = _intervene(language_model, site, inputs, RouteAction(mode, **fields[mode]))
    q = donor.q if mode in ("q", "qk") else recipient.q
    k = donor.k if mode in ("k", "qk") else recipient.k
    expected_conditional = torch.softmax(q @ k.transpose(-1, -2) * (8**-0.5), dim=-1)
    actual_image = _selected(output[1], recipient)[..., IMAGE_COLUMNS]
    torch.testing.assert_close(actual_image / recipient.mass, expected_conditional, atol=2e-7, rtol=1e-6)
    assert torch.equal(captured.v, recipient.v) and torch.equal(captured.q, recipient.q)
    assert torch.equal(_outside(output[1], recipient), _outside(native[1], recipient))
    assert torch.equal(output[1][:, :2], native[1][:, :2])
    assert torch.equal(output[1][:, :, :6], native[1][:, :, :6])
    assert torch.equal(output[0][:, :6], native[0][:, :6])
    old_image = _selected(native[1], recipient)[..., IMAGE_COLUMNS]
    delta = _expected_delta(language_model, site, (actual_image - old_image) @ recipient.v)
    torch.testing.assert_close(output[0][:, -1:], native[0][:, -1:] + delta, atol=1e-8, rtol=1e-6)
    assert diagnostics["mass_before_after_maxdiff"] < 1e-7
    assert diagnostics["text_weight_maxdiff"] == 0 and diagnostics["donor_finite"]


def test_distribution_ignores_donor_mass_and_updates_dense_return(language_model, prompt, site):
    inputs = _inputs(language_model, prompt)
    recipient, native, _ = _capture(language_model, site, inputs)
    conditional = torch.tensor([[[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]])
    donor = replace(recipient, conditional=conditional, mass=recipient.mass * 0.25)
    captured, output, diagnostics = _intervene(
        language_model, site, inputs, RouteAction("distribution", distribution=donor),
    )
    expected = conditional * recipient.mass
    torch.testing.assert_close(_selected(output[1], recipient)[..., IMAGE_COLUMNS], expected, atol=0, rtol=0)
    assert torch.equal(captured.conditional, recipient.conditional)
    assert torch.equal(_outside(output[1], recipient), _outside(native[1], recipient))
    assert torch.equal(output[0][:, :6], native[0][:, :6])
    assert diagnostics["attention_row_updated"]
    assert diagnostics["mass_before_after_maxdiff"] == 0
    assert diagnostics["row_output_delta_rms"] > 0


def test_nonfinal_selected_row_only_is_edited(language_model, prompt, site):
    inputs = _inputs(language_model, prompt)
    recipient, native, _ = _capture(language_model, site, inputs, row=5)
    _, output, diagnostics = _intervene(language_model, site, inputs, RouteAction("uniform"), row=5)
    assert torch.equal(output[0][:, :5], native[0][:, :5])
    assert torch.equal(output[0][:, 6:], native[0][:, 6:])
    assert torch.equal(output[1][:, :, :5], native[1][:, :, :5])
    assert torch.equal(output[1][:, :, 6:], native[1][:, :, 6:])
    assert torch.equal(_outside(output[1], recipient), _outside(native[1], recipient))
    assert diagnostics["mass_before_after_maxdiff"] < 1e-7


def test_q_routing_reuses_recipient_additive_image_bias(language_model, prompt, site):
    inputs = _inputs(language_model, prompt)
    inputs["attention_mask"][..., 6, 2] = -1.3
    recipient, _, _ = _capture(language_model, site, inputs)
    donor_prompt = prompt.clone()
    donor_prompt[0, -1] = 21
    donor_inputs = _inputs(language_model, donor_prompt)
    donor_inputs["attention_mask"] = inputs["attention_mask"].clone()
    donor, _, _ = _capture(language_model, site, donor_inputs)
    _, output, _ = _intervene(language_model, site, inputs, RouteAction("q", query=donor))
    mask = recipient.mask_row[..., IMAGE_COLUMNS]
    expected = torch.softmax(donor.q @ recipient.k.transpose(-1, -2) * (8**-0.5) + mask, -1)
    actual = _selected(output[1], recipient)[..., IMAGE_COLUMNS] / recipient.mass
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)


def test_capture_only_none_and_missing_nonfinal_causal_mask(language_model, prompt, site):
    inputs = _inputs(language_model, prompt)
    inputs["attention_mask"] = None
    attn = language_model.model.layers[0].self_attn
    with torch.no_grad():
        native = attn(**inputs)
        with RoutingSession(language_model, (site,), row=6, image_columns=IMAGE_COLUMNS,
                            actions=None) as session:
            captured = attn(**inputs)
    assert torch.equal(native[0], captured[0]) and torch.equal(native[1], captured[1])
    assert session.diagnostics[site]["row_output_delta_rms"] == 0
    with pytest.raises(ValueError, match="actual causal"):
        _capture(language_model, site, inputs, row=5)


def test_v_is_full_palette_change_not_a_route_change(language_model, prompt, site):
    inputs = _inputs(language_model, prompt)
    recipient, native, _ = _capture(language_model, site, inputs)
    donor_v = recipient.v + torch.linspace(-0.4, 0.6, recipient.v.numel()).reshape(recipient.v.shape)
    donor = replace(recipient, v=donor_v)
    captured, output, diagnostics = _intervene(language_model, site, inputs, RouteAction("v", value=donor))
    old_image = _selected(native[1], recipient)[..., IMAGE_COLUMNS]
    delta = _expected_delta(language_model, site, old_image @ (donor.v - recipient.v))
    torch.testing.assert_close(output[0][:, -1:], native[0][:, -1:] + delta, atol=1e-8, rtol=1e-6)
    assert torch.equal(output[0][:, :6], native[0][:, :6])
    assert torch.equal(output[1], native[1]) and torch.equal(captured.v, recipient.v)
    assert diagnostics["conditional_l1"] == 0 and diagnostics["row_output_delta_rms"] > 0


@pytest.mark.parametrize("mode", ["route_permute", "value_permute", "uniform"])
def test_controls_are_deterministic_and_preserve_rng_mass_and_support(language_model, prompt, site, mode):
    inputs = _inputs(language_model, prompt)
    recipient, native, _ = _capture(language_model, site, inputs)
    rng = torch.random.get_rng_state().clone()
    _, first, diag = _intervene(language_model, site, inputs, RouteAction(mode, seed=17))
    _, second, _ = _intervene(language_model, site, inputs, RouteAction(mode, seed=17))
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(_outside(first[1], recipient), _outside(native[1], recipient))
    assert diag["mass_before_after_maxdiff"] < 1e-7
    permutation = torch.randperm(3, generator=torch.Generator().manual_seed(17))
    if mode == "value_permute":
        assert torch.equal(first[1], native[1])
        old = _selected(native[1], recipient)[..., IMAGE_COLUMNS]
        expected = _expected_delta(language_model, site, old @ (recipient.v[:, :, permutation] - recipient.v))
        torch.testing.assert_close(first[0][:, -1:], native[0][:, -1:] + expected, atol=1e-8, rtol=1e-6)
    else:
        conditional = recipient.conditional[..., permutation] if mode == "route_permute" else torch.full_like(recipient.conditional, 1 / 3)
        torch.testing.assert_close(_selected(first[1], recipient)[..., IMAGE_COLUMNS] / recipient.mass,
                                   conditional, atol=2e-7, rtol=1e-6)


def test_multiple_gqa_groups_are_captured_before_any_edit(language_model, prompt):
    inputs = _inputs(language_model, prompt)
    sites = (Site(0, 0), Site(0, 1))
    baseline = {}
    for site in sites:
        baseline[site], _, _ = _capture(language_model, site, inputs)
    with torch.no_grad(), RoutingSession(
        language_model, sites, row=6, image_columns=IMAGE_COLUMNS,
        actions={site: RouteAction("uniform") for site in sites},
    ) as session:
        language_model.model.layers[0].self_attn(**inputs)
    for site in sites:
        assert torch.equal(session.captures[site].conditional, baseline[site].conditional)
        assert session.diagnostics[site]["query_heads"] == [site.kv_group * 2, site.kv_group * 2 + 1]


@pytest.mark.parametrize("layer,group", [(-1, 0), (0, -1), (True, 0), (0, True)])
def test_site_requires_nonnegative_integer_indices(layer, group):
    with pytest.raises(ValueError):
        Site(layer, group)


@pytest.mark.parametrize("site", [Site(9, 0), Site(0, 2)])
def test_site_range_error_leaves_no_hooks(language_model, site):
    before = _hooks(language_model)
    with pytest.raises(ValueError), RoutingSession(language_model, (site,), row=6, image_columns=IMAGE_COLUMNS):
        pass
    assert _hooks(language_model) == before


@pytest.mark.parametrize("columns", [(), (2, 1), (1, 1), (1, 7), (-1, 2)])
def test_invalid_image_columns_reject(language_model, site, columns):
    with pytest.raises(ValueError):
        RoutingSession(language_model, (site,), row=6, image_columns=columns)


@pytest.mark.parametrize("kind", ["columns", "row", "sequence", "site", "rotary", "positions", "missing_rotary", "heads"])
def test_unaligned_donor_rejects_and_immediately_removes_hooks(language_model, prompt, site, kind):
    inputs = _inputs(language_model, prompt)
    donor, _, _ = _capture(language_model, site, inputs)
    if kind == "columns":
        donor = replace(donor, image_columns=(0, 2, 3))
    elif kind == "row":
        donor = replace(donor, row=5)
    elif kind == "sequence":
        donor = replace(donor, sequence_length=8, position_ids=torch.arange(8)[None],
                        mask_row=torch.cat((donor.mask_row, torch.zeros(1, 2, 1, 1)), -1))
    elif kind == "site":
        donor = replace(donor, site=Site(1, 1))
    elif kind == "rotary":
        donor = replace(donor, cos_row=donor.cos_row + 0.01)
    elif kind == "positions":
        donor = replace(donor, position_ids=donor.position_ids + 1)
    elif kind == "missing_rotary":
        donor = replace(donor, cos_image=None)
    else:
        donor = replace(donor, q=donor.q.repeat(1, 2, 1, 1),
                        conditional=donor.conditional.repeat(1, 2, 1, 1),
                        mass=donor.mass.repeat(1, 2, 1, 1), mask_row=donor.mask_row.repeat(1, 2, 1, 1))
    before = _hooks(language_model)
    session = RoutingSession(language_model, (site,), row=6, image_columns=IMAGE_COLUMNS,
                             actions={site: RouteAction("q", query=donor)})
    with session:
        with pytest.raises(ValueError, match="alignment|rotary"):
            language_model.model.layers[0].self_attn(**inputs)
        assert _hooks(language_model) == before
    assert _hooks(language_model) == before


@pytest.mark.parametrize("field", ["q", "k", "v", "conditional", "mass"])
def test_mutated_nonfinite_donor_rejects(language_model, prompt, site, field):
    inputs = _inputs(language_model, prompt)
    donor, _, _ = _capture(language_model, site, inputs)
    getattr(donor, field).flatten()[0] = float("nan")
    before = _hooks(language_model)
    with pytest.raises(ValueError, match="finite"):
        _intervene(language_model, site, inputs, RouteAction("qk", query=donor, key=donor))
    assert _hooks(language_model) == before


def test_finite_donor_conversion_overflow_rejects(language_model, prompt, site):
    inputs = _inputs(language_model, prompt)
    donor, _, _ = _capture(language_model, site, inputs)
    donor = replace(donor, q=torch.full(donor.q.shape, 1e100, dtype=torch.float64))
    before = _hooks(language_model)
    with pytest.raises(ValueError, match="converted donor q.*finite"):
        _intervene(language_model, site, inputs, RouteAction("q", query=donor))
    assert _hooks(language_model) == before


def test_unused_donors_and_invalid_capture_distributions_reject(language_model, prompt, site):
    donor, _, _ = _capture(language_model, site, _inputs(language_model, prompt))
    with pytest.raises(ValueError, match="does not use"):
        RouteAction("uniform", query=donor)
    with pytest.raises(ValueError, match="nonnegative distribution"):
        replace(donor, conditional=torch.zeros_like(donor.conditional))
    with pytest.raises(ValueError, match="mass"):
        replace(donor, mass=torch.zeros_like(donor.mass))
    with pytest.raises(ValueError, match="unsupported"):
        RouteAction(["q"])


@pytest.mark.parametrize("mode,fields", [("bad", {}), ("q", {}), ("qk", {}), ("v", {}), ("distribution", {})])
def test_action_requires_explicit_correct_donors(mode, fields):
    with pytest.raises(ValueError):
        RouteAction(mode, **fields)


@pytest.mark.parametrize("use_cache", [False, True])
def test_even_empty_cache_rejects_before_native_mutation(language_model, prompt, site, use_cache):
    cache = DynamicCache(config=language_model.config)
    before = _hooks(language_model)
    with pytest.raises(ValueError, match="cache"), RoutingSession(
        language_model, (site,), row=6, image_columns=IMAGE_COLUMNS,
    ):
        language_model(input_ids=prompt, use_cache=use_cache, past_key_values=cache)
    assert cache.get_seq_length() == 0 and _hooks(language_model) == before
    inputs = _inputs(language_model, prompt)
    inputs["past_key_values"] = cache
    with pytest.raises(ValueError, match="cache"):
        _capture(language_model, site, inputs)
    assert cache.get_seq_length() == 0


def test_true_or_default_cache_policy_rejects(language_model, prompt, site):
    with pytest.raises(ValueError, match="use_cache=False"), RoutingSession(
        language_model, (site,), row=6, image_columns=IMAGE_COLUMNS,
    ):
        language_model(input_ids=prompt, use_cache=True)
    language_model.config.use_cache = True
    with pytest.raises(ValueError, match="use_cache=False"), RoutingSession(
        language_model, (site,), row=6, image_columns=IMAGE_COLUMNS,
    ):
        language_model(input_ids=prompt)


@pytest.mark.parametrize("flag", ["output_attentions", "output_hidden_states"])
def test_hf_recording_rejects_before_forward_wrappers_are_installed(language_model, prompt, site, flag):
    attn = language_model.model.layers[0].self_attn
    had_override = "forward" in attn.__dict__
    before, original = _hooks(language_model), attn.forward.__func__
    with pytest.raises(ValueError, match="recording precedes hooks"), RoutingSession(
        language_model, (site,), row=6, image_columns=IMAGE_COLUMNS,
    ):
        language_model(input_ids=prompt, use_cache=False, **{flag: True})
    assert attn.forward.__func__ is original and ("forward" in attn.__dict__) == had_override
    assert _hooks(language_model) == before


def test_batched_train_dropout_sdpa_and_non_qwen_reject(language_model, prompt, site):
    with pytest.raises(ValueError, match="batch"), RoutingSession(
        language_model, (site,), row=6, image_columns=IMAGE_COLUMNS,
    ):
        language_model(input_ids=prompt.repeat(2, 1), use_cache=False)
    language_model.train()
    with pytest.raises(ValueError, match="eval"):
        with RoutingSession(language_model, (site,), row=6, image_columns=IMAGE_COLUMNS):
            pass
    language_model.eval()
    attn = language_model.model.layers[0].self_attn
    attn.attention_dropout = 0.1
    with pytest.raises(ValueError, match="dropout"):
        with RoutingSession(language_model, (site,), row=6, image_columns=IMAGE_COLUMNS):
            pass
    attn.attention_dropout = 0.0
    language_model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        with RoutingSession(language_model, (site,), row=6, image_columns=IMAGE_COLUMNS):
            pass
    with pytest.raises(ValueError, match="Qwen3"):
        with RoutingSession(nn.Linear(32, 32), (site,), row=6, image_columns=IMAGE_COLUMNS):
            pass


@pytest.mark.parametrize("kind", ["image", "future", "boolean", "nonfinite_hidden", "positive_inf"])
def test_actual_mask_support_and_finite_inputs_are_required(language_model, prompt, site, kind):
    inputs = _inputs(language_model, prompt)
    row = 6
    if kind == "image":
        inputs["attention_mask"][..., 6, 2] = float("-inf")
    elif kind == "future":
        row = 5
        inputs["attention_mask"][..., row, 6] = 0
    elif kind == "boolean":
        inputs["attention_mask"] = inputs["attention_mask"] == 0
    elif kind == "nonfinite_hidden":
        inputs["hidden_states"][0, 0, 0] = float("nan")
    else:
        inputs["attention_mask"][..., 6, 2] = float("inf")
    before = _hooks(language_model)
    with pytest.raises(ValueError):
        _capture(language_model, site, inputs, row=row)
    assert _hooks(language_model) == before


def test_model_mode_changes_and_cpu_autocast_reject_without_hook_leaks(language_model, prompt, site):
    inputs = _inputs(language_model, prompt)
    before = _hooks(language_model)
    with RoutingSession(language_model, (site,), row=6, image_columns=IMAGE_COLUMNS):
        language_model.model.layers[0].self_attn.train()
        with pytest.raises(ValueError, match="eval"):
            language_model.model.layers[0].self_attn(**inputs)
        assert _hooks(language_model) == before
    language_model.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        with RoutingSession(language_model, (site,), row=6, image_columns=IMAGE_COLUMNS):
            pass
    assert _hooks(language_model) == before


def test_sliding_attention_uses_native_mask_not_an_invented_full_window(prompt, site):
    model = _tiny_model(sliding=True)
    with torch.no_grad():
        native = model(input_ids=prompt, use_cache=False).logits
        with RoutingSession(model, (site,), row=6, image_columns=(4, 5)) as session:
            output = model(input_ids=prompt, use_cache=False).logits
    assert torch.equal(native, output)
    assert bool((session.captures[site].mask_row[..., :4] < -1e20).all())
    before = _hooks(model)
    with pytest.raises(ValueError, match="masked|sliding"), RoutingSession(
        model, (site,), row=6, image_columns=(1, 2),
    ):
        model(input_ids=prompt, use_cache=False)
    assert _hooks(model) == before


def test_nested_routing_guard_and_body_exception_cleanup(language_model, prompt, site):
    before = _hooks(language_model)
    with pytest.raises(RuntimeError, match="body"), RoutingSession(
        language_model, (site,), row=6, image_columns=IMAGE_COLUMNS,
    ):
        with pytest.raises(RuntimeError, match="nested"):
            with RoutingSession(language_model, (Site(0, 0),), row=6, image_columns=IMAGE_COLUMNS):
                pass
        language_model(input_ids=prompt, use_cache=False)
        raise RuntimeError("body failure")
    assert _hooks(language_model) == before
    _capture(language_model, site, _inputs(language_model, prompt))


def test_mapping_source_snapshots_and_native_prefix_is_mandatory(language_model):
    source, tensors = _source(language_model)
    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    original = source.read_tensor(name)
    tensors[name].zero_()
    read = source.read_tensor(name)
    read.zero_()
    assert torch.equal(source.read_tensor(name), original)
    with pytest.raises(ValueError, match="native"):
        source.read_tensor("model.layers.0.self_attn.q_proj.weight")


def test_full_swap_exact_group_slices_shared_norm_scope_and_restore(language_model, site):
    source, tensors = _source(language_model)
    before = _parameters(language_model)
    ids = {name: id(parameter) for name, parameter in language_model.named_parameters()}
    attn = language_model.model.layers[0].self_attn
    original_forward = attn.forward.__func__
    swap = WeightSwap(language_model, site, source, projections=("q_proj", "k_proj", "v_proj", "o_proj"), include_norms=True)
    expected = {name: tensor.clone() for name, tensor in before.items()}
    selections = {"q_proj": (slice(16, 32), slice(None)), "k_proj": (slice(8, 16), slice(None)),
                  "v_proj": (slice(8, 16), slice(None)), "o_proj": (slice(None), slice(16, 32)),
                  "q_norm": (slice(None),), "k_norm": (slice(None),)}
    for name, selection in selections.items():
        key = f"model.layers.0.self_attn.{name}.weight"
        expected[key][selection] = tensors[f"language_model.{key}"][selection]
    with swap:
        _assert_parameters(language_model, expected)
        assert swap.summary["changedtensors"] == 6 and swap.summary["active"]
        assert swap.summary["shared_norm_scope"] == "layerwide_all_query_heads_and_all_kv_heads"
        assert swap.summary["paramcount"] == 1552 and not swap.summary["biases_swapped"]
        for report in swap.summary["slices"]:
            assert report["name"].startswith("language_model.model.layers.0.self_attn.")
            assert len(report["source_checksum"]) == 64
            assert report["source_slice_checksum"] == report["applied_slice_checksum"]
            assert report["source_slice_checksum"] != report["native_slice_checksum"]
            assert math.isfinite(report["normalized_delta_norm"])
        assert attn.forward.__func__ is original_forward
        assert ids == {name: id(parameter) for name, parameter in language_model.named_parameters()}
    _assert_parameters(language_model, before)
    assert swap.summary["restored_bitexact"] and not swap.summary["active"]


def test_swap_copies_converted_source_endpoint_directly(language_model, site):
    _, tensors = _source(language_model, changed=False)
    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    tensors[name] = tensors[name].double() + 1e-7
    source = MappingTensorSource(tensors)
    before = _parameters(language_model)
    with WeightSwap(language_model, site, source, projections=("q_proj",)) as swap:
        actual = language_model.model.layers[0].self_attn.q_proj.weight[16:32]
        assert torch.equal(actual, tensors[name][16:32].float())
        assert swap.summary["slices"][0]["source_dtype"] == "torch.float64"
        assert swap.summary["norm_scope"] == "selected_slices_only"
    _assert_parameters(language_model, before)


@pytest.mark.parametrize("include_norms", [False, True])
def test_sham_permutates_delta_not_source_weights_preserves_norm_and_rng(language_model, site, include_norms):
    source, tensors = _source(language_model)
    before = _parameters(language_model)
    rng = torch.random.get_rng_state().clone()
    snapshots = []
    for _ in range(2):
        swap = WeightSwap(language_model, site, source, sham=True, seed=97, include_norms=include_norms)
        with swap:
            snapshots.append(_parameters(language_model))
            generator = torch.Generator().manual_seed(97)
            for report in swap.summary["slices"]:
                key = report["name"].removeprefix("language_model.")
                selection = (slice(None),) if "norm" in key else (
                    slice(report["slice"]["start"], report["slice"]["stop"]), slice(None),
                )
                original = before[key][selection]
                endpoint = tensors[report["name"]][selection]
                delta = endpoint.double() - original.double()
                permutation = torch.randperm(delta.numel(), generator=generator)
                intended = delta.flatten()[permutation].reshape(delta.shape)
                expected = (original.double() + intended).to(original.dtype)
                assert torch.equal(snapshots[-1][key][selection], expected)
                assert report["intended_delta_frobnorm"] == pytest.approx(report["source_delta_frobnorm"], rel=1e-12)
                assert report["delta_frobnorm"] == pytest.approx(report["source_delta_frobnorm"], rel=1e-6)
                if "q_proj" in key:
                    shuffled_source = endpoint.flatten()[permutation].reshape(endpoint.shape)
                    assert not torch.equal(expected, shuffled_source)
            assert swap.summary["intended_delta_norm_difference"] < 1e-12
        _assert_parameters(language_model, before)
        assert swap.summary["restored_bitexact"]
    for name in before:
        assert torch.equal(snapshots[0][name], snapshots[1][name])
    assert torch.equal(rng, torch.random.get_rng_state())
    # Every parameter outside reported support remained native while the sham was active.
    for name in before:
        if "model.layers.0.self_attn.q_proj.weight" == name:
            assert torch.equal(snapshots[0][name][:16], before[name][:16])
        elif "model.layers.0.self_attn.k_proj.weight" == name:
            assert torch.equal(snapshots[0][name][:8], before[name][:8])
        elif include_norms and name in ("model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight"):
            continue
        else:
            assert torch.equal(snapshots[0][name], before[name])


def test_swap_restores_after_exception_and_nested_group_guard(language_model, site):
    source, _ = _source(language_model)
    before = _parameters(language_model)
    swap = WeightSwap(language_model, site, source, include_norms=True)
    with pytest.raises(RuntimeError, match="intentional"), swap:
        current = _parameters(language_model)
        with pytest.raises(RuntimeError, match="nested"):
            with WeightSwap(language_model, Site(0, 0), source):
                pass
        _assert_parameters(language_model, current)
        raise RuntimeError("intentional failure")
    _assert_parameters(language_model, before)
    assert swap.summary["restored_bitexact"]
    with WeightSwap(language_model, site, source):
        pass
    _assert_parameters(language_model, before)


@pytest.mark.parametrize("kind", ["missing", "shape", "nan", "short_prefix"])
def test_invalid_source_stages_no_partial_mutation_and_releases_guard(language_model, site, kind):
    _, tensors = _source(language_model)
    before = _parameters(language_model)
    bad_name = "language_model.model.layers.0.self_attn.k_proj.weight"
    if kind == "missing":
        del tensors[bad_name]
        source = MappingTensorSource(tensors)
    elif kind == "shape":
        tensors[bad_name] = tensors[bad_name][:8]
        source = MappingTensorSource(tensors)
    elif kind == "nan":
        class NonfiniteSource:
            def read_tensor(self, name):
                result = tensors[name].clone()
                if name == bad_name:
                    result.flatten()[0] = float("inf")
                return result
        source = NonfiniteSource()
    else:
        source = MappingTensorSource({name.removeprefix("language_model."): tensor for name, tensor in tensors.items()})
    with pytest.raises((KeyError, ValueError)):
        with WeightSwap(language_model, site, source):
            pass
    _assert_parameters(language_model, before)
    valid, _ = _source(language_model)
    with WeightSwap(language_model, site, valid):
        pass
    _assert_parameters(language_model, before)


def test_mapping_nonfinite_reject_and_weight_swap_eval_only(language_model, site):
    with pytest.raises(ValueError, match="finite"):
        MappingTensorSource({"weight": torch.tensor([float("nan")])})
    source, _ = _source(language_model)
    before = _parameters(language_model)
    language_model.train()
    with pytest.raises(ValueError, match="eval"):
        with WeightSwap(language_model, site, source):
            pass
    _assert_parameters(language_model, before)


def test_same_input_replay_reports_real_mediator_deltas_and_restores(language_model, prompt, site):
    capture, _, _ = _capture(language_model, site, _inputs(language_model, prompt))
    before = _parameters(language_model)
    source, _ = _source(language_model)
    report = identical_input_replay(language_model, site, source, capture, include_norms=True)
    assert report["q_rms_delta"] > 0 and report["k_rms_delta"] > 0
    assert 0 <= report["conditional_routing_l1"] <= 2
    assert len(report["conditional_routing_l1_per_query_head"]) == 2
    assert report["native_capture_reconstruction"]["q_rms"] < 1e-6
    assert report["native_capture_reconstruction"]["k_rms"] < 1e-6
    assert report["native_capture_reconstruction"]["conditional_l1"] < 1e-6
    assert report["weight_swap"]["restored_bitexact"]
    _assert_parameters(language_model, before)
    own_source, _ = _source(language_model, changed=False)
    own = identical_input_replay(language_model, site, own_source, capture)
    assert own["q_rms_delta"] == 0 and own["k_rms_delta"] == 0
    assert own["conditional_routing_l1"] == 0 and own["weight_swap"]["changedtensors"] == 0
    _assert_parameters(language_model, before)


def test_same_input_replay_failure_restores_weights(language_model, prompt, site, monkeypatch):
    import vlm_pilot.weights as weights_module

    capture, _, _ = _capture(language_model, site, _inputs(language_model, prompt))
    source, _ = _source(language_model)
    before = _parameters(language_model)
    original = weights_module._replay_qk
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("replay failure")
        return original(*args)

    monkeypatch.setattr(weights_module, "_replay_qk", fail_second)
    with pytest.raises(RuntimeError, match="replay failure"):
        identical_input_replay(language_model, site, source, capture)
    _assert_parameters(language_model, before)


def test_same_input_replay_real_finite_projection_overflow_restores(language_model, prompt, site):
    capture, _, _ = _capture(language_model, site, _inputs(language_model, prompt))
    _, tensors = _source(language_model, changed=False)
    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    # All stored weights remain finite, but this aligned-sign dot product overflows FP32.
    aligned = capture.input_rows[0, 0].sign() * (torch.finfo(torch.float32).max / 2)
    tensors[name] = aligned.unsqueeze(0).expand_as(tensors[name]).contiguous().clone()
    source = MappingTensorSource(tensors)
    before = _parameters(language_model)
    with pytest.raises(ValueError, match="finite"):
        identical_input_replay(language_model, site, source, capture, projections=("q_proj",))
    _assert_parameters(language_model, before)
    with WeightSwap(language_model, site, MappingTensorSource({
        f"language_model.{name}": tensor for name, tensor in before.items()
    })):
        pass
    _assert_parameters(language_model, before)


@pytest.mark.parametrize("field", ["input_rows", "input_image", "cos_row", "mask_row"])
def test_same_input_replay_requires_bounded_audit_inputs(language_model, prompt, site, field):
    capture, _, _ = _capture(language_model, site, _inputs(language_model, prompt))
    capture = replace(capture, **{field: None})
    before = _parameters(language_model)
    source, _ = _source(language_model)
    with pytest.raises(ValueError, match="requires"):
        identical_input_replay(language_model, site, source, capture)
    _assert_parameters(language_model, before)


def _cached_checkpoint(tmp_path, monkeypatch, *, allow_downloads=False):
    from safetensors.torch import save_file

    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    unused = "language_model.model.layers.0.self_attn.k_proj.weight"
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file({name: tensor, unused: torch.tensor([float("nan")])}, str(shard))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({
        "metadata": {"total_size": 52}, "weight_map": {name: shard.name, unused: shard.name},
    }), encoding="utf-8")
    calls = []

    def cached_download(**kwargs):
        calls.append(kwargs)
        assert kwargs["revision"] == REVISION
        assert kwargs["local_files_only"] is (not allow_downloads)
        assert kwargs["repo_type"] == "model"
        assert kwargs["repo_id"] == "fixture/checkpoint"
        assert kwargs["filename"] in (index.name, shard.name)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr("huggingface_hub.hf_hub_download", cached_download)
    return name, unused, tensor, index, shard, calls


@pytest.mark.parametrize("allow_downloads", [False, True])
def test_safetensor_source_is_lazy_pinned_selected_cpu_read(tmp_path, monkeypatch, allow_downloads):
    name, unused, expected, _, _, calls = _cached_checkpoint(tmp_path, monkeypatch, allow_downloads=allow_downloads)

    def forbid_model(*args, **kwargs):
        raise AssertionError("a tensor source must not instantiate models or load pickle")

    monkeypatch.setattr(Qwen3ForCausalLM, "from_pretrained", forbid_model)
    monkeypatch.setattr(torch, "load", forbid_model)
    source = SafetensorSource("fixture/checkpoint", REVISION, allow_downloads=allow_downloads)
    assert calls == []
    first = source.read_tensor(name)
    assert first.device.type == "cpu" and torch.equal(first, expected)
    first.zero_()
    assert torch.equal(source.read_tensor(name), expected)
    assert [call["filename"] for call in calls].count("model.safetensors.index.json") == 1
    with pytest.raises(ValueError, match="finite"):
        source.read_tensor(unused)


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40, "../checkpoint"])
def test_safetensor_requires_commit_pin(revision):
    with pytest.raises(ValueError, match="pinned"):
        SafetensorSource("fixture/checkpoint", revision)


def test_safetensor_pin_and_download_policy_are_read_only():
    source = SafetensorSource("fixture/checkpoint", REVISION)
    for field, value in (("repo_id", "fixture/other"), ("revision", "b" * 40), ("allow_downloads", True)):
        with pytest.raises(AttributeError):
            setattr(source, field, value)


@pytest.mark.parametrize("filename", ["../bad.safetensors", "/bad.safetensors", "x/y.safetensors", "x\\y.safetensors", "https://host/x.safetensors", "model..safetensors"])
def test_safetensor_unsafe_index_paths_reject_before_shard_access(tmp_path, monkeypatch, filename):
    name, _, _, index, _, calls = _cached_checkpoint(tmp_path, monkeypatch)
    index.write_text(json.dumps({"weight_map": {name: filename}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        SafetensorSource("fixture/checkpoint", REVISION).read_tensor(name)
    assert len(calls) == 1


@pytest.mark.parametrize("cap", ["MAX_INDEX_BYTES", "MAX_SHARD_BYTES", "MAX_TENSOR_BYTES"])
def test_safetensor_enforces_size_caps(tmp_path, monkeypatch, cap):
    name, _, _, _, _, _ = _cached_checkpoint(tmp_path, monkeypatch)
    monkeypatch.setattr(SafetensorSource, cap, 8)
    with pytest.raises(ValueError, match="size cap"):
        SafetensorSource("fixture/checkpoint", REVISION).read_tensor(name)


def test_safetensor_duplicate_keys_and_nonfinite_index_metadata_reject(tmp_path, monkeypatch):
    name, _, _, index, shard, _ = _cached_checkpoint(tmp_path, monkeypatch)
    index.write_text(f'{{"weight_map":{{"{name}":"{shard.name}","{name}":"{shard.name}"}}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        SafetensorSource("fixture/checkpoint", REVISION).read_tensor(name)
    index.write_text(json.dumps({"metadata": {"total_size": float("nan")}, "weight_map": {name: shard.name}}), encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        SafetensorSource("fixture/checkpoint", REVISION).read_tensor(name)