"""Diagnose post-load failures offline without leaking dependency messages.

Artificial logits exercise tolerance/NaN/Inf accounting; tiny real decoder tests
elsewhere cover ordinary execution. An explicit diagnostic-only mode must never
weaken the source lock for actual runs or alter model/data pins after an update.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from interp.artifacts import ArtifactError
from interp.cli import main
from interp.posttrain import plan as planning
from interp.posttrain import preflight, runner
from interp.posttrain.errors import StudyError, study_stage
from interp.posttrain.models import load_checkpoint
from interp.posttrain.schemas import Checkpoint, Inference, Job

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs/posttrain/smoke.yaml"


def _logit_probe(bundle, *, delta=0.0, nonfinite=None, dtype="float32"):
    """Provide known logits for cache diagnostics only, never fake research results."""
    import torch

    class Probe(torch.nn.Module):
        """Supply deterministic full/cached outputs and ordinary module-mode semantics."""

        def __init__(self):
            """Expose one parameter so dtype detection uses the ordinary model interface."""
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=getattr(torch, dtype)))

        def forward(self, input_ids, attention_mask, use_cache, past_key_values=None):
            """Change only the first cached logit, preserving the argmax for finite cases."""
            logits = self.anchor.new_tensor([0.0, 1.0, 2.0]).expand(
                1, input_ids.shape[1], 3,
            ).clone()
            if use_cache and past_key_values is not None:
                logits[..., 0] += delta
            if nonfinite is not None:
                # Matching infinities intentionally expose torch.isclose's permissive case.
                logits[..., 0] = nonfinite
            return SimpleNamespace(logits=logits, past_key_values=object() if use_cache else None)

    return replace(bundle, model=Probe())


def test_stage_hides_dependency_details_but_preserves_cause():
    """Public output names the operation; deliberate in-process debugging retains the cause."""
    original = ValueError("synthetic-private-dependency-detail")
    with pytest.raises(StudyError, match="Checkpoint loading failed \\(ValueError\\)") as caught:
        with study_stage("Checkpoint loading"):
            raise original
    assert caught.value.__cause__ is original
    assert "synthetic-private" not in str(caught.value)


@pytest.mark.parametrize("error", [StudyError("Controlled explanation"), KeyboardInterrupt()])
def test_stage_preserves_controlled_errors_and_interrupts(error):
    """Safe explanations and cancellation must not be relabeled as dependency faults."""
    with pytest.raises(type(error)) as caught:
        with study_stage("Checkpoint loading"):
            raise error
    assert caught.value is error


def test_cache_parity_reports_exact_finite_matches(bundle):
    """A passing diagnostic includes finite counts and zero error without changing mode."""
    probe = _logit_probe(bundle)
    assert probe.model.training
    report = preflight.cache_parity(probe, (1, 2, 3, 4))
    assert report["cache_logits_close"] and report["cache_logits_finite"]
    assert report["compared_logits"] == 9 and report["mismatched_logits"] == 0
    assert report["max_absolute_logit_difference"] == report["max_tolerance_ratio"] == 0.0
    assert report["cache_argmax_equal"] and probe.model.training


@pytest.mark.parametrize(("dtype", "tolerance"), [("float32", 1e-4), ("bfloat16", 0.05)])
def test_cache_parity_does_not_waive_mismatch_for_matching_argmax(bundle, dtype, tolerance):
    """BF16's existing tolerance remains unchanged; a common greedy token cannot waive it."""
    report = preflight.cache_parity(_logit_probe(bundle, delta=0.125, dtype=dtype), (1, 2, 3, 4))
    assert report["cache_argmax_equal"] and report["cache_logits_finite"]
    assert not report["cache_logits_close"] and report["mismatched_logits"] == 3
    assert report["max_absolute_logit_difference"] == 0.125
    assert report["max_tolerance_ratio"] == pytest.approx(0.125 / (tolerance * 1.125))
    assert report["atol"] == report["rtol"] == tolerance


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_logits_fail_without_invalid_json_or_false_zero_error(bundle, value):
    """Matching infinities cannot pass and NaNs cannot silently leave maximum error at zero."""
    report = preflight.cache_parity(_logit_probe(bundle, nonfinite=value), (1, 2, 3, 4))
    assert not report["cache_logits_close"] and not report["cache_logits_finite"]
    assert not report["cache_argmax_equal"] and report["mismatched_logits"] == 3
    assert report["max_absolute_logit_difference"] is None
    assert report["max_tolerance_ratio"] is None
    json.dumps(report, allow_nan=False)


def test_cache_forward_error_retains_stage_and_restores_module_mode(bundle, monkeypatch):
    """Even a dependency failure inside the parity loop must leave caller-owned modes intact."""
    probe = _logit_probe(bundle)
    monkeypatch.setattr(probe.model, "forward", Mock(side_effect=ValueError("private input")))
    with pytest.raises(StudyError, match="Cache-parity forward passes") as caught:
        preflight.cache_parity(probe, (1, 2, 3))
    assert "private input" not in str(caught.value)
    assert probe.model.training


@pytest.mark.parametrize("failure", ["parity", "replay", "both", "nonfinite"])
def test_cli_reports_numerical_failures_instead_of_generic_valueerror(
    bundle, tmp_path, monkeypatch, capsys, failure,
):
    """Expose counts/settings, distinguish replay, keep failed preflight nonzero and read-only."""
    path = tmp_path / "plan.json"
    planning.lock_study(SMOKE, path)
    before = path.read_bytes()
    monkeypatch.setattr(preflight, "load_checkpoint", Mock(return_value=bundle))
    measurements = preflight.cache_parity(_logit_probe(
        bundle, delta=0.125 if failure in {"parity", "both"} else 0.0,
        nonfinite=float("inf") if failure == "nonfinite" else None,
    ), (1, 2, 3, 4))
    monkeypatch.setattr(preflight, "cache_parity", Mock(return_value=measurements))
    generation = Mock(side_effect=[
        SimpleNamespace(token_ids=(1, 2)),
        SimpleNamespace(token_ids=(1, 3) if failure in {"replay", "both"} else (1, 2)),
    ])
    monkeypatch.setattr(preflight, "generate", generation)
    assert main(["posttrain", "preflight", str(path), "--checkpoint", "base", "--device", "cpu"]) == 1
    output = capsys.readouterr()
    assert not output.out and "Preflight failed:" in output.err
    assert "operation failed" not in output.err
    assert ("cache logit parity" in output.err) == (failure != "replay")
    assert ("seeded token replay" in output.err) == (failure in {"replay", "both"})
    details = json.loads(output.err[output.err.index("{"):])
    assert details["mismatched_logits"] == measurements["mismatched_logits"]
    assert details["plan_source_matches"] and not details["diagnostic_only"]
    assert generation.call_count == (0 if failure == "nonfinite" else 2)
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


def test_diagnostic_mode_reports_changed_source_without_weakening_run_lock(
    bundle, tmp_path, monkeypatch, capsys,
):
    """An update can diagnose the same cached checkpoint without rewriting a locked plan."""
    path = tmp_path / "plan.json"
    plan = planning.lock_study(SMOKE, path)
    before = path.read_bytes()
    changed = "0" * 64 if plan.source_sha256 != "0" * 64 else "1" * 64
    for module in (planning, preflight, runner):
        monkeypatch.setattr(module, "source_digest", lambda: changed)
    loader = Mock(return_value=bundle)
    monkeypatch.setattr(preflight, "load_checkpoint", loader)
    arguments = ["posttrain", "preflight", str(path), "--checkpoint", "base", "--device", "cpu"]
    assert main(arguments) == 1
    assert "code changed" in capsys.readouterr().err
    loader.assert_not_called()
    assert main([*arguments, "--diagnostic"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["diagnostic_only"] and not report["plan_source_matches"]
    assert report["plan_source_sha256"] == plan.source_sha256
    assert report["current_source_sha256"] == changed
    assert loader.call_args.args[0] == plan.spec.checkpoints[0]
    assert loader.call_args.kwargs["allow_downloads"] is False
    with pytest.raises(ArtifactError, match="source changed"):
        runner.run_study(plan, Job(checkpoint="base"), tmp_path / "run", device="cpu")
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["weights", "vocabulary", "fingerprint"])
def test_postload_stage_errors_expose_safe_location_only(bundle, monkeypatch, failure):
    """Mock HF acquisition while exercising the real post-load validation/provenance code."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    monkeypatch.setattr(AutoConfig, "from_pretrained", Mock(return_value=bundle.model.config))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(return_value=bundle.tokenizer))
    loader = Mock(return_value=bundle.model)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", loader)
    if failure == "weights":
        loader.side_effect = ValueError("private remote request details")
        expected = "Checkpoint weight and generation-config loading"
    elif failure == "vocabulary":
        rows = bundle.model.get_input_embeddings().num_embeddings
        monkeypatch.setattr(bundle.tokenizer, "get_vocab", lambda: {"fake-token": rows})
        expected = f"maximum ID={rows}, embedding rows={rows}"
    else:
        monkeypatch.setattr(bundle.model.generation_config, "to_dict", lambda: {"fake": float("inf")})
        expected = "Checkpoint profile fingerprinting"
    checkpoint = Checkpoint(name="base", stage="base", model_id="artificial/model",
                            revision="a" * 40, tokenizer_id="artificial/model", tokenizer_revision="a" * 40)
    with pytest.raises(StudyError, match=expected) as caught:
        load_checkpoint(checkpoint, Inference(dtype="float32", attention="eager"), device_name="cpu")
    assert "private remote request details" not in str(caught.value)
    assert loader.call_args.kwargs["use_safetensors"] is True
    assert loader.call_args.kwargs["trust_remote_code"] is False
    assert loader.call_args.kwargs["local_files_only"] is True