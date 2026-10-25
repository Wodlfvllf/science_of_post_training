"""Run the real tiny HF decoder through the portable study CLI, entirely offline.

Predictions are scientifically meaningless. Tests check exact seeds/tokens,
model-free completed resume, safe parsing boundaries, and numerical cache parity.
The shared session fixture selects one CPU thread for efficient tiny forwards.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from interp.cli import main
from interp.posttrain.models import preflight_inputs, render_prompt
from interp.posttrain.plan import lock_study
from interp.posttrain.preflight import cache_parity
from interp.posttrain.runner import resume_study, run_study
from interp.posttrain.schemas import Checkpoint, Job, Prompt
from interp.posttrain.storage import read_snapshot

ROOT = Path(__file__).resolve().parents[2]


def test_real_cli_partial_resume_and_report(bundle, tmp_path, capsys, monkeypatch):
    """A remote-style CLI job and library replay retain identical logical outputs."""
    config = ROOT / "configs/posttrain/smoke.yaml"
    locked = tmp_path / "plan.json"
    base, post, replay = (tmp_path / name for name in ("base", "post", "replay"))
    assert main(["posttrain", "preview", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["planned_trials"] == 8
    assert main(["posttrain", "lock", str(config), "--output", str(locked)]) == 0
    capsys.readouterr()
    assert main(["posttrain", "run", str(locked), "--checkpoint", "base",
                 "--output-dir", str(base), "--device", "cpu", "--max-new-trials", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["completed_trials"] == 1
    assert resume_study(base, device="cpu")["new_attempts"] == 3
    assert run_study(locked, Job(checkpoint="post"), post, device="cpu")["completed_trials"] == 4
    assert run_study(locked, Job(checkpoint="base"), replay, device="cpu")["completed_trials"] == 4
    first = [(row.trial, row.result.generation, row.result.grade) for row in read_snapshot(base).records]
    repeated = [(row.trial, row.result.generation, row.result.grade) for row in read_snapshot(replay).records]
    assert first == repeated

    def forbidden(*args, **kwargs):
        """Inspection and already complete resume must never allocate another model."""
        raise AssertionError("Model loaded unexpectedly.")

    monkeypatch.setattr("interp.posttrain.models.load_checkpoint", forbidden)
    locked.unlink()
    assert resume_study(base, device="cpu")["new_attempts"] == 0
    assert main(["posttrain", "inspect", str(base)]) == 0
    capsys.readouterr()
    output = tmp_path / "report"
    assert main(["posttrain", "analyze", str(base), str(post), "--output", str(output),
                 "--bootstrap-samples", "20"]) == 0
    assert json.loads(capsys.readouterr().out)["incomplete"] is False
    report = json.loads((output / "report.json").read_text())
    assert report["cells"][0]["included_trials"] == 4
    assert len(report["comparisons"]) == 1


def test_prompt_rendering_never_includes_gold_or_skips_missing_template(bundle, tmp_path):
    """Unassisted input uses only question text, while native formatting must be explicit."""
    plan = lock_study(ROOT / "configs/posttrain/smoke.yaml", tmp_path / "plan.json")
    problem = plan.problems[0].model_copy(update={"answer": "999999"})
    rendered, ids = render_prompt(bundle, plan.spec.checkpoints[0], problem, plan.spec.prompts[0])
    assert "999999" not in rendered and ids
    post = Checkpoint(name="post", stage="rl", backend="fixture", model_id="offline-tiny-gpt2", revision="fixture-v1")
    with pytest.raises(ValueError, match="real chat template"):
        render_prompt(bundle, post, problem, Prompt(name="native", format="native"))
    # Explicitly supply a fixture-only template and verify special-token ownership.
    original = bundle.tokenizer.chat_template
    try:
        bundle.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}{{ eos_token }}"
        text, native = render_prompt(bundle, post, problem, Prompt(name="native", format="native"))
        assert native == tuple(bundle.tokenizer.encode(text, add_special_tokens=False))
        assert native[0] == bundle.tokenizer.bos_token_id
        assert native.count(bundle.tokenizer.bos_token_id) == 1
    finally:
        bundle.tokenizer.chat_template = original


def test_all_cues_cross_correct_and_error_drafts_without_gold_leak(bundle):
    """Checking assistance does not identify which prefix actually contains an error."""
    from interp.posttrain.data import make_repair_problems

    pair = make_repair_problems(count=1)
    checkpoint = Checkpoint(name="base", stage="base", backend="fixture", model_id="offline-tiny-gpt2", revision="fixture-v1")
    for cue in ("none", "neutral", "check"):
        prompt = Prompt(name=cue, cue=cue, instruction="Continue.")
        texts = [render_prompt(bundle, checkpoint, row, prompt)[0] for row in pair]
        assert pair[0].draft in texts[0] and pair[1].draft in texts[1]
        assert texts[0].replace(pair[0].draft, "DRAFT") == texts[1].replace(pair[1].draft, "DRAFT")
        assert f"Final answer: {pair[0].answer}" not in texts[0]


def test_tokenization_overflow_is_error_not_truncation(bundle, tmp_path):
    """Validate all cells before starting research work, including the largest budget."""
    plan = lock_study(ROOT / "configs/posttrain/smoke.yaml", tmp_path / "plan.json")
    short = replace(bundle, context_limit=8)
    with pytest.raises(ValueError, match="exceeds context"):
        preflight_inputs(short, plan.spec.checkpoints[0], plan)


def test_cached_sample_replays_and_restores_state(bundle):
    """KV caching changes speed, not trial lifetime or caller-owned randomness."""
    import torch

    from interp.models import generate
    from interp.schemas import GenerationSpec, PromptSpec

    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    settings = GenerationSpec(max_new_tokens=5, do_sample=True, use_cache=True)
    first = generate(bundle, (1, 22, 23), settings, PromptSpec(), seed=33)
    second = generate(bundle, (1, 22, 23), settings, PromptSpec(), seed=33)
    assert first.token_ids == second.token_ids and first.effective_config["use_cache"] is True
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)
    assert cache_parity(bundle, (1, 22, 23, 24, 25))["cache_logits_close"]


def test_tiny_olmo3_cache_parity_on_cpu(bundle):
    """Exercise the actual Olmo3 implementation offline, not only a GPT-2 surrogate.

    Tiny random architecture parity is engineering evidence only; it cannot verify
    downloaded weights, GPU kernels, the production tokenizer, or long contexts.
    """
    import torch
    from transformers import Olmo3Config, Olmo3ForCausalLM

    config = Olmo3Config(
        vocab_size=len(bundle.tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        eos_token_id=2, bos_token_id=1, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        model = Olmo3ForCausalLM(config).eval()
    olmo = replace(bundle, model=model)
    measured = cache_parity(olmo, (1, 10, 11, 12, 13, 14))
    assert measured["cache_logits_close"]