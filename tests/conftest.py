# Shared offline fixtures: small synthetic records for storage tests, real tiny CPU GPT-2 for scoring.
"""Isolated pytest fixtures; tests never authenticate, download, or access research GPUs."""

from __future__ import annotations

import socket
from dataclasses import replace
from pathlib import Path

import pytest

from interp.config import compute_run_id, prepare_run, with_execution
from interp.runtime import utc_now
from interp.schemas import CandidateScore, RunMetadata, RunSpec, TrialResult

# Anchor checked-in engineering data independently of the test process working directory.
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    """Disable HF downloads and reject outbound socket connections in every test."""
    # Tell established HF libraries to use local files only even if a test forgets an opt-in.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # Apply the same offline policy to Transformers.
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def deny_network(*args, **kwargs):
        """Make accidental network access an immediate test failure."""
        # Never send authentication or model requests from the offline suite.
        raise AssertionError("Offline tests must not open network connections.")

    # Reject direct outbound TCP connections, not only HTTP-library convenience calls.
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    # Reject the alternate socket connection interface too.
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)


@pytest.fixture
def prepared(tmp_path):
    """Provide a small scoring-only schedule with a unique temporary run directory."""
    # Validate the exact checked-in paired engineering observations.
    inputs = with_execution(prepare_run(ROOT / "configs" / "smoke.yaml"), output_dir=tmp_path / "run")
    # Disable generation only for artificial record/storage unit tests.
    raw = inputs.spec.model_dump(mode="python")
    # Let the fake ordinary trial function focus on token-score/attempt persistence.
    raw["generation"]["enabled"] = False
    # Publish one attempt per shard so fault boundaries are easy to inspect.
    raw["execution"]["shard_records"] = 1
    # Revalidate the altered scientific fixture specification.
    spec = RunSpec.model_validate(raw)
    # Compute a new identity because scoring-only is a different scientific recipe setting.
    identity = compute_run_id(spec, inputs.dataset_sha256, inputs.source_sha256)
    # Return the frozen artificial-record test preparation.
    return replace(inputs, spec=spec, run_id=identity)


@pytest.fixture
def metadata(prepared):
    """Build complete provenance for fake numerical records without loading a model."""
    # Label artificial records explicitly rather than pretending they came from real model execution.
    return RunMetadata(run_id=prepared.run_id, created_at=utc_now(), spec=prepared.spec, dataset_sha256=prepared.dataset_sha256, source_sha256=prepared.source_sha256, runtime_profile={"kind": "offline-record-test"}, model_profile={"kind": "artificial-score-fixture"}, examples=prepared.examples)


@pytest.fixture
def fake_trial():
    """Provide deterministic, schema-valid paired outputs for runner/storage tests only."""

    def execute(example, replicate, seed, should_stop):
        """Return artificial likelihoods, not model predictions, with a known paired change."""
        # Fix a short auditable prompt token prefix for persistence/alignment invariants.
        prefix = (1,)
        # Give the first candidate a controlled condition-dependent synthetic likelihood.
        target_value = -1.0 if example.condition == "baseline" else -2.0
        # Preserve exact candidate labels/texts from the actual dataset definitions.
        first = CandidateScore(candidate_id=example.candidates[0].id, text=example.candidates[0].text, input_ids=(*prefix, 5), attention_mask=(1, 1), continuation_start=1, token_logprobs=(target_value,), sum_logprob=target_value, mean_logprob=target_value)
        # Give the second candidate an unequal-length synthetic suffix with a known total.
        second = CandidateScore(candidate_id=example.candidates[1].id, text=example.candidates[1].text, input_ids=(*prefix, 6, 7), attention_mask=(1, 1, 1), continuation_start=1, token_logprobs=(-2.0, -2.0), sum_logprob=-4.0, mean_logprob=-2.0)
        # Use the dataset's declared target rather than hardcoding a semantic label.
        by_id = {first.candidate_id: first.sum_logprob, second.candidate_id: second.sum_logprob}
        # Identify the alternative by exclusion from the declared target label.
        other = next(candidate.id for candidate in example.candidates if candidate.id != example.target_candidate)
        # Return a complete scoring-only result with the correct raw likelihood margin.
        return TrialResult(input_ids=prefix, attention_mask=(1,), candidate_scores=(first, second), target_logprob_margin=by_id[example.target_candidate] - by_id[other])

    # Expose one ordinary function, exactly as the real recipe seam expects.
    return execute


@pytest.fixture(scope="session")
def bundle():
    """Construct the real tiny CPU decoder/tokenizer once, with no checkpoint downloads."""
    # Import optional model dependencies only for tests that actually need numerical execution.
    import torch

    from interp.models import load_model
    from interp.schemas import ModelSpec

    # Preserve surrounding CPU-thread policy while making the tiny test model efficient.
    original_threads = torch.get_num_threads()
    # A single thread avoids excessive thread-launch overhead for this engineering fixture.
    torch.set_num_threads(1)
    # Ensure the surrounding numerical setting is restored even after test failures.
    try:
        # Build standard HF model objects entirely from local configuration and byte vocabulary.
        yield load_model(ModelSpec())
    # Leave a developer's interactive process policy unchanged after the test session.
    finally:
        # Restore the previous CPU numerical/threading profile.
        torch.set_num_threads(original_threads)
