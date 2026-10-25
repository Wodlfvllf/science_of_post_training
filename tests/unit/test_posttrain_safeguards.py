"""Regression tests for scientific failure modes found during independent review.

These checks keep an attractive benchmark number from hiding unfinished thinking,
contradictory final expressions, inherited decoding defaults, or duplicate GPU jobs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from interp.posttrain.data import validate_problems
from interp.posttrain.plan import lock_study
from interp.posttrain.runner import run_study
from interp.posttrain.schemas import Job, Problem
from interp.posttrain.scoring import grade_answer

ROOT = Path(__file__).resolve().parents[2]


def _problem() -> Problem:
    """Provide an independently known answer without a generated solution trace."""
    return Problem(id="p", source_id="s", group_id="g", split="pilot", task="math",
                   question="What is six times seven?", answer="42", provenance="fixture")


@pytest.mark.parametrize("text", [r"\boxed{42}+1", r"\boxed{42} is wrong; the answer is 43"])
def test_boxed_subexpression_is_not_a_correct_final(text):
    """A correct-looking boxed intermediate value cannot rescue contradictory output."""
    grade = grade_answer(text, _problem())
    assert not grade.correct and grade.extraction == "invalid"


@pytest.mark.parametrize(("text", "initial"), [
    ("<think>Final answer: 42", False),
    ("Final answer: 42", True),
    ("</think>Final answer: 42<think>unfinished", True),
])
def test_unclosed_thinking_cannot_earn_final_credit(text, initial):
    """A length-bounded thinking rollout is a sample, but not a delivered final answer."""
    grade = grade_answer(text, _problem(), initial_thinking=initial)
    assert grade.unfinished_thinking and not grade.correct and grade.answer is None


def test_template_opening_closes_before_a_real_final_answer():
    """The native assistant template can supply the opening tag outside generated text."""
    grade = grade_answer("Working. </think>Final answer: 42", _problem(), initial_thinking=True)
    assert grade.correct and not grade.unfinished_thinking


def test_one_source_cannot_be_split_into_independent_groups():
    """Matching source seeds cannot masquerade as independent bootstrap observations."""
    second = _problem().model_copy(update={"id": "p2", "group_id": "g2", "question": "What is 40 + 2?"})
    with pytest.raises(ValueError, match="one group_id"):
        validate_problems((_problem(), second))


def test_duplicate_job_lock_precedes_model_allocation(tmp_path, monkeypatch):
    """A second invocation must not allocate another 7B model before failing its lock."""
    plan = lock_study(ROOT / "configs/posttrain/smoke.yaml", tmp_path / "plan.json")
    directory = tmp_path / "run"
    directory.mkdir()

    def forbidden(*args, **kwargs):
        """Make model allocation observable without requiring Torch/GPU initialization."""
        raise AssertionError("Model allocation happened before writer lock.")

    monkeypatch.setattr("interp.posttrain.models.load_checkpoint", forbidden)
    with FileLock(str(directory / ".writer.lock"), timeout=0):
        with pytest.raises(Timeout):
            run_study(plan, Job(checkpoint="base"), directory, device="cpu")


@pytest.mark.parametrize("options", [
    {"max_seconds": float("nan")}, {"max_seconds": float("inf")},
    {"shutdown_margin_seconds": float("nan")}, {"flush_records": True},
])
def test_invalid_operational_limits_fail_before_model_work(tmp_path, options):
    """NaN deadlines cannot accidentally remove the remote-session time limit."""
    with pytest.raises(ValueError):
        run_study(tmp_path / "nonexistent-plan", Job(checkpoint="base"), tmp_path / "run", **options)


def test_generation_does_not_inherit_checkpoint_penalties(bundle, monkeypatch):
    """Pass an explicit neutral config without Transformers merging model defaults."""
    from interp.models import generate
    from interp.schemas import GenerationSpec, PromptSpec

    settings = GenerationSpec(max_new_tokens=4, do_sample=True, top_k=0, use_cache=True)
    expected = generate(bundle, (1, 33, 34), settings, PromptSpec(), seed=123)
    # Modify only checkpoint defaults; the experiment's explicit settings are unchanged.
    monkeypatch.setattr(bundle.model.generation_config, "repetition_penalty", 9.0)
    monkeypatch.setattr(bundle.model.generation_config, "top_k", 1)
    monkeypatch.setattr(bundle.model.generation_config, "min_new_tokens", 10)
    observed = generate(bundle, (1, 33, 34), settings, PromptSpec(), seed=123)
    assert observed.token_ids == expected.token_ids
    assert observed.effective_config["repetition_penalty"] == 1.0
    assert observed.effective_config["top_k"] == 0