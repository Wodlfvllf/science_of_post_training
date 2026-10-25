"""Model-free numerical/report regressions using explicitly artificial rollouts.

Most structural edge cases inject typed snapshots at the verified-reader boundary;
one round-trip uses StudyStore to exercise real checksums and grade validation.
No test needs Torch, NumPy, a tokenizer, model weights, network, or a GPU. Synthetic
token IDs/text are bookkeeping fixtures, never represented as model measurements.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from interp.data import canonical_bytes, digest_bytes
from interp.posttrain import metrics
from interp.posttrain.data import make_repair_problems
from interp.posttrain.metrics import analyze_runs, paired_bootstrap, pass_at_k, write_report
from interp.posttrain.plan import job_identity, schedule, study_identity
from interp.posttrain.schemas import (
    Attempt,
    Checkpoint,
    Inference,
    Job,
    Problem,
    Prompt,
    Rollout,
    StudyMetadata,
    StudyPlan,
    StudySpec,
)
from interp.posttrain.scoring import grade_answer
from interp.posttrain.storage import Snapshot, StudyStore
from interp.schemas import CheckpointManifest, GenerationRecord, GenerationSpec

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _plan(*, count=2, samples=4, task="math", prompts=None, group_ids=None):
    """Build an actually valid locked fixture plan with an intentionally optional SFT stage."""
    if task == "repair":
        problems = make_repair_problems(count=count, seed=17)
    else:
        problems = tuple(Problem(
            id=f"q-{index}", source_id=f"source-{index}",
            group_id=group_ids[index] if group_ids else f"group-{index}",
            split="pilot", task="math", question=f"What is {index} plus forty-two?",
            answer=str(index + 42), provenance="artificial-metrics-fixture",
        ) for index in range(count))
    spec = StudySpec(
        name="metrics-fixture", hypothesis="Artificial bookkeeping only; no scientific result.",
        dataset=Path("unused-source.jsonl"),
        checkpoints=tuple(Checkpoint(
            name=name, stage=stage, backend="fixture", model_id="offline-tiny-gpt2",
            revision="fixture-v1",
        ) for name, stage in (("base", "base"), ("middle", "sft"), ("post", "rl"))),
        prompts=prompts or (Prompt(name="unusual-completion-name"),), budgets=(4,), samples=samples,
        inference=Inference(dtype="float32", attention="eager", max_context_tokens=64),
        generation=GenerationSpec(max_new_tokens=4, do_sample=True, use_cache=True),
    )
    dataset_sha = digest_bytes(canonical_bytes([problem.model_dump(mode="json") for problem in problems]))
    source_sha = "1" * 64
    return StudyPlan(
        study_id=study_identity(spec, dataset_sha, source_sha), created_at=_NOW,
        source_sha256=source_sha, dataset_sha256=dataset_sha, spec=spec, problems=problems,
    )


def _attempt(metadata, trial, text, *, status="completed", finish="eos"):
    """Create bounded raw output whose grade and effective settings satisfy StudyStore.

    Cancellations deliberately carry a correctly graded partial result. Analysis
    must still exclude them until a completed attempt fills that logical sample.
    """
    problem = next(row for row in metadata.plan.problems if row.id == trial.problem_id)
    settings = metadata.plan.spec.generation
    effective = {
        "max_new_tokens": trial.budget, "do_sample": True, "use_cache": settings.use_cache,
        "num_beams": 1, "num_return_sequences": 1, "repetition_penalty": 1.0,
        "temperature": settings.temperature, "top_p": settings.top_p, "top_k": settings.top_k,
        "eos_token_id": 0,
    }
    finish = "cancelled" if status == "cancelled" else finish
    tokens = (7,) * trial.budget if finish == "length" else (7, 0)
    result = None if status == "failed" else Rollout(
        rendered_prompt="Artificial input, not an executed model prompt.", input_ids=(11, 12, 13),
        generation=GenerationRecord(token_ids=tokens, text=text, finish_reason=finish,
                                    effective_config=effective),
        answer_text=text, grade=grade_answer(text, problem), elapsed_seconds=2.0,
    )
    return Attempt(
        run_id=metadata.run_id, trial=trial, attempt_id=uuid4(), started_at=_NOW, finished_at=_NOW,
        status=status, result=result, error_type="FixtureFailure" if status == "failed" else None,
    )


def _snapshot(plan, *, checkpoint="base", job=None, outcome=None, runtime=None):
    """Construct a typed snapshot; outcome(trial,problem)=None deliberately omits a trial."""
    job = job or Job(checkpoint=checkpoint)
    metadata = StudyMetadata(
        run_id=job_identity(plan, job), created_at=_NOW, plan=plan, job=job,
        runtime_profile=runtime if runtime is not None else {
            "device": "cpu", "threads": 1, "packages": {"torch": "synthetic-not-imported"},
        },
        model_profile={"kind": "artificial-records", "dtype": "torch.float32", "attention": "eager"},
    )
    records = []
    problems = {problem.id: problem for problem in plan.problems}
    for trial in schedule(plan, job):
        problem = problems[trial.problem_id]
        text = outcome(trial, problem) if outcome is not None else f"Final answer: {problem.answer}"
        if text is not None:
            records.append(_attempt(metadata, trial, text))
    manifest = CheckpointManifest(
        run_id=metadata.run_id, metadata_sha256=digest_bytes(canonical_bytes(metadata.model_dump(mode="json"))),
        updated_at=_NOW, completed_trial_ids=tuple(record.trial.trial_id for record in records),
    )
    return Snapshot(metadata, manifest, tuple(records))


def _analyze(monkeypatch, tmp_path, snapshots, **options):
    """Replace only the verified-reader boundary; retain all cross-job analysis logic."""
    indexed = {(tmp_path / f"input-{index}").resolve(): snapshot for index, snapshot in enumerate(snapshots)}
    monkeypatch.setattr(metrics, "read_snapshot", indexed.__getitem__)
    options.setdefault("bootstrap_samples", 40)
    return analyze_runs(list(indexed), **options)


def test_pass_at_k_is_exact_unbiased_not_plugin():
    """Without-replacement inclusion probability differs from a fitted Bernoulli model."""
    assert pass_at_k(4, 1, 2) == 0.5
    assert pass_at_k(4, 1, 2) != 1 - (1 - 1 / 4) ** 2
    assert pass_at_k(4, 1, 1) == 0.25
    assert pass_at_k(4, 1, 4) == 1.0
    assert pass_at_k(4, 0, 4) == 0.0
    assert pass_at_k(4, 4, 2) == 1.0
    assert pass_at_k(1024, 1, 64) == 64 / 1024


@pytest.mark.parametrize("counts", [
    (0, 0, 1), (-1, 0, 1), (4, -1, 1), (4, 5, 1), (4, 1, 0), (4, 1, 5),
    (True, 1, 1), (4, True, 1), (4, 1, False), (4.0, 1, 2), (4, 1.0, 2), (4, 1, 2.0),
])
def test_pass_at_k_rejects_malformed_counts_and_extrapolation(counts):
    """No coercion, negative successes, missing denominator, or k beyond observed n."""
    with pytest.raises(ValueError):
        pass_at_k(*counts)


@pytest.mark.parametrize("values", [[], [True], [float("nan")], [float("inf")], ["1"], None])
def test_bootstrap_rejects_nonfinite_or_empty_differences(values):
    """Undefined paired samples fail clearly instead of returning invented zero-width CIs."""
    with pytest.raises(ValueError, match="values"):
        paired_bootstrap(values)


@pytest.mark.parametrize("options", [
    {"bootstrap_samples": 0}, {"bootstrap_samples": -1}, {"bootstrap_samples": 1.0},
    {"bootstrap_samples": True}, {"bootstrap_samples": 100_001},
    {"seed": -1}, {"seed": True}, {"seed": 2**32}, {"seed": 1.5},
])
def test_bootstrap_integer_options_fail_before_reading(monkeypatch, tmp_path, options):
    """Both public entry points enforce the same positive-replica and seed bounds."""
    def unexpected_read(path):
        raise AssertionError("Invalid options must not read snapshots.")

    monkeypatch.setattr(metrics, "read_snapshot", unexpected_read)
    with pytest.raises(ValueError):
        paired_bootstrap([0.0], **options)
    with pytest.raises(ValueError):
        analyze_runs([tmp_path], **options)


def test_paired_bootstrap_is_isolated_deterministic_and_not_independent():
    """Constant within-group differences stay constant despite heterogeneous paired levels."""
    before = random.getstate()
    # Reference levels [0, .5] and target levels [.25, .75] differ by .25 in each group.
    assert paired_bootstrap([0.25, 0.25], bootstrap_samples=40) == (0.25, 0.25)
    first = paired_bootstrap([-1.0, 0.0, 0.5, 1.0], bootstrap_samples=73, seed=7)
    assert first == paired_bootstrap([-1.0, 0.0, 0.5, 1.0], bootstrap_samples=73, seed=7)
    assert random.getstate() == before
    assert paired_bootstrap([0.75], bootstrap_samples=1) == (0.75, 0.75)


def test_percentile_algorithm_is_explicit_inclusive_linear_interpolation():
    """The endpoints and a two-replicate interval distinguish interpolation conventions."""
    assert metrics._percentile([0.0, 10.0], 0.025) == 0.25
    assert metrics._percentile([0.0, 10.0], 0.975) == 9.75
    assert metrics._percentile([3.0], 0.975) == 3.0


def test_bootstrap_large_finite_differences_do_not_overflow_the_mean():
    """Finite inputs need not overflow merely because an unscaled intermediate sum would."""
    assert paired_bootstrap([1e308, 1e308], bootstrap_samples=5) == (1e308, 1e308)
    assert metrics._percentile([-1e308, 1e308], 0.5) == 0.0
    with pytest.raises(ValueError, match="finite float"):
        paired_bootstrap([10**400])


def test_missing_samples_never_create_survivor_pass_at_k(monkeypatch, tmp_path):
    """A lone missing sample invalidates the whole question, not merely its denominator."""
    plan = _plan(count=1)
    snapshot = _snapshot(plan, outcome=lambda trial, problem: (
        None if trial.sample == 3 else f"Final answer: {problem.answer}"
    ))
    with pytest.raises(ValueError, match="Incomplete checkpoint schedule.*1/4 missing"):
        _analyze(monkeypatch, tmp_path, [snapshot])
    report = _analyze(monkeypatch, tmp_path, [snapshot], allow_incomplete=True)
    cell, row = report["cells"][0], report["problems"][0]
    assert cell["pass_at_1"] is None and cell["pass_at_1_ci95"] is None
    assert cell["pass_at_k"] == {"1": None, "2": None, "4": None}
    assert cell["included_trials"] == 0 and cell["excluded_completed_trials"] == 3
    assert cell["planned_problems"] == 1 and cell["complete_problems"] == 0
    assert row["correct_samples"] == 3 and row["pass_at_1"] is None
    assert row["missing_sample_reasons"] == {"no_attempt": 1}
    assert row["exclusion_reasons"] == ["incomplete_group", "no_attempt"]


def test_missing_repair_variant_excludes_both_but_retains_other_groups(monkeypatch, tmp_path):
    """Complete error/correct pairs, not surviving variants, define analysis inclusion."""
    plan = _plan(task="repair")
    omitted = plan.problems[1].id
    snapshot = _snapshot(plan, outcome=lambda trial, problem: (
        None if problem.id == omitted and trial.sample == 3 else f"Final answer: {problem.answer}"
    ))
    report = _analyze(monkeypatch, tmp_path, [snapshot], allow_incomplete=True)
    cells = {cell["prefix"]: cell for cell in report["cells"]}
    for cell in cells.values():
        assert cell["included_problems"] == 1 and cell["excluded_problems"] == 1
        assert cell["pass_at_1"] == 1.0 and cell["included_trials"] == 4
    assert cells["correct"]["complete_problems"] == 2
    assert cells["correct"]["excluded_completed_trials"] == 4
    assert cells["error"]["excluded_completed_trials"] == 3
    dropped = [row for row in report["problems"] if row["group_id"] == plan.problems[0].group_id]
    assert len(dropped) == 2 and all(not row["included"] for row in dropped)
    assert all(row["pass_at_1"] is None for row in dropped)


def test_incompleteness_is_local_to_prompt_budget_not_whole_checkpoint(monkeypatch, tmp_path):
    """Opt-in partial reports can include another fully completed prompt condition."""
    plan = _plan(count=1, prompts=(Prompt(name="finished"), Prompt(name="unfinished")))
    snapshot = _snapshot(plan, outcome=lambda trial, problem: (
        None if trial.prompt == "unfinished" and trial.sample == 3 else f"Final answer: {problem.answer}"
    ))
    report = _analyze(monkeypatch, tmp_path, [snapshot], allow_incomplete=True)
    cells = {cell["prompt"]: cell for cell in report["cells"]}
    assert cells["finished"]["pass_at_1"] == 1.0
    assert cells["unfinished"]["pass_at_1"] is None


def test_failed_cancelled_attempts_are_not_completed_but_retry_can_fill_sample(monkeypatch, tmp_path):
    """Even a correct partial cancellation cannot rescue a missing logical trial."""
    snapshot = _snapshot(_plan(count=1))
    missing = snapshot.records[-1]
    failed = _attempt(snapshot.metadata, missing.trial, "", status="failed")
    cancelled = _attempt(snapshot.metadata, missing.trial, "Final answer: 42", status="cancelled")
    partial = replace(snapshot, records=(*snapshot.records[:-1], failed, cancelled))
    with pytest.raises(ValueError, match="Incomplete"):
        _analyze(monkeypatch, tmp_path, [partial])
    report = _analyze(monkeypatch, tmp_path, [partial], allow_incomplete=True)
    row = report["problems"][0]
    assert row["completed_samples"] == 3 and row["correct_samples"] == 3
    assert row["failed_attempts"] == row["cancelled_attempts"] == 1
    assert row["missing_sample_reasons"] == {"failed_and_cancelled_without_completion": 1}
    retried = replace(partial, records=(*partial.records, missing))
    completed = _analyze(monkeypatch, tmp_path, [retried])["cells"][0]
    assert completed["pass_at_1"] == 1.0 and completed["included_trials"] == 4
    assert completed["failed_attempts"] == completed["cancelled_attempts"] == 1


def test_parse_format_and_length_failures_remain_bounded_denominator(monkeypatch, tmp_path):
    """Two successes, a missing answer, and invalid final yield 2/4, not 2/2."""
    snapshot = _snapshot(_plan(count=1), outcome=lambda trial, problem: (
        f"Final answer: {problem.answer}", rf"\boxed{{{problem.answer}}}",
        "No declared final", "Final answer: not-a-number",
    )[trial.sample])
    length = _attempt(snapshot.metadata, snapshot.records[-1].trial, "Final answer: not-a-number", finish="length")
    snapshot = replace(snapshot, records=(*snapshot.records[:-1], length))
    cell = _analyze(monkeypatch, tmp_path, [snapshot])["cells"][0]
    assert cell["included_trials"] == 4 and cell["pass_at_1"] == 0.5
    assert cell["pass_at_k"]["2"] == pytest.approx(5 / 6)
    assert cell["format_ok_rate"] == 0.25 and cell["parse_failure_rate"] == 0.5
    assert cell["length_stop_rate"] == 0.25 and cell["oracle_coverage"] == 1.0
    assert cell["mean_generated_tokens"] == 2.5 and cell["mean_input_tokens"] == 3.0
    assert cell["mean_total_tokens"] == 5.5 and cell["mean_elapsed_seconds"] == 2.0


@pytest.mark.parametrize(("answers", "accuracy", "abstention"), [
    (("42", "43", None, None), 0.0, 1.0),
    (("42", "42", None, None), 1.0, 0.0),
    ((None, None, None, None), 0.0, 1.0),
])
def test_majority_vote_ties_abstain_missing_answers_are_not_votes(
    monkeypatch, tmp_path, answers, accuracy, abstention,
):
    """Only parsed answers vote; ties/no votes abstain and remain incorrect at question level."""
    snapshot = _snapshot(_plan(count=1), outcome=lambda trial, problem: (
        f"Final answer: {answers[trial.sample]}" if answers[trial.sample] is not None else "missing"
    ))
    report = _analyze(monkeypatch, tmp_path, [snapshot])
    cell = report["cells"][0]
    assert cell["majority_vote_accuracy"] == accuracy
    assert cell["majority_abstention_rate"] == abstention
    assert report["problems"][0]["parsed_votes"] == sum(answer is not None for answer in answers)


def test_repair_claim_rates_are_descriptive_observations(monkeypatch, tmp_path):
    """Equation labels can coexist with propagated errors without implying causal detection."""
    def outcome(trial, problem):
        a, b, c = problem.operands
        claims = (f"{a}*{b}={a*b}", f"{a}*{b}={a*b+1}",
                  f"{a}*{b}={a*b}\n{a}*{b}={a*b+1}", "No equation restatement.")
        answer = problem.proposed_product + c if trial.sample == 1 else problem.answer
        return f"{claims[trial.sample]}\nFinal answer: {answer}"

    report = _analyze(monkeypatch, tmp_path, [_snapshot(_plan(count=1, task="repair"), outcome=outcome)])
    for cell in report["cells"]:
        assert all(cell[f"step_claim_{claim}_rate"] == 0.25 for claim in metrics._CLAIMS)
        assert cell["propagated_error_rate"] == (0.25 if cell["prefix"] == "error" else 0.0)


def test_endpoint_only_study_is_complete_and_checkpoint_ci_is_paired(monkeypatch, tmp_path):
    """An absent intermediate stage is intentional; paired levels yield a constant difference."""
    plan = _plan()
    base = _snapshot(plan, outcome=lambda trial, problem: (
        f"Final answer: {problem.answer}" if trial.sample < 2 * int(problem.id[-1]) else "Final answer: -1"
    ))
    post = _snapshot(plan, checkpoint="post", outcome=lambda trial, problem: (
        f"Final answer: {problem.answer}" if trial.sample < 1 + 2 * int(problem.id[-1]) else "Final answer: -1"
    ))
    report = _analyze(monkeypatch, tmp_path, [base, post])
    assert report["checkpoints_present"] == ["base", "post"]
    assert report["checkpoints_absent"] == ["middle"] and not report["incomplete"]
    comparison = report["comparisons"][0]
    assert comparison["difference_unit"] == "proportion"
    assert comparison["pass_at_1_difference"] == 0.25
    assert comparison["pass_at_1_difference_ci95"] == [0.25, 0.25]
    assert comparison["matched_groups"] == comparison["matched_problems"] == 2
    assert comparison["solved_overlap"] == {"both": 1, "base_only": 0, "post_only": 1, "neither": 0}


def test_solved_overlap_contains_all_four_descriptive_categories(monkeypatch, tmp_path):
    """Observed successes, not mean correctness thresholds, define sampled overlap."""
    plan = _plan(count=4)
    base = _snapshot(plan, outcome=lambda trial, problem: (
        f"Final answer: {problem.answer}" if problem.id in ("q-0", "q-1") and trial.sample == 0 else "bad"
    ))
    post = _snapshot(plan, checkpoint="post", outcome=lambda trial, problem: (
        f"Final answer: {problem.answer}" if problem.id in ("q-0", "q-2") and trial.sample == 0 else "bad"
    ))
    comparison = _analyze(monkeypatch, tmp_path, [base, post])["comparisons"][0]
    assert comparison["solved_overlap"] == {"both": 1, "base_only": 1, "post_only": 1, "neither": 1}
    assert "descriptive sampled" in comparison["overlap_interpretation"]


def test_comparisons_use_intersection_of_complete_groups(monkeypatch, tmp_path):
    """Differences cannot compare disjoint survivor questions or their unmatched means."""
    plan = _plan(count=3)
    base = _snapshot(plan, outcome=lambda trial, problem: None if problem.id == "q-0" else "Final answer: -1")
    post = _snapshot(plan, checkpoint="post", outcome=lambda trial, problem: (
        None if problem.id == "q-1" else f"Final answer: {problem.answer}"
    ))
    report = _analyze(monkeypatch, tmp_path, [base, post], allow_incomplete=True)
    contrast = report["comparisons"][0]
    assert contrast["matched_groups"] == 1 and contrast["excluded_groups"] == 2
    assert contrast["reference_unmatched_problems"] == contrast["target_unmatched_problems"] == 1
    assert contrast["pass_at_1_difference"] == 1.0
    assert contrast["pass_at_1_difference_ci95"] == [1.0, 1.0]


def test_no_matched_groups_produces_no_comparison_estimate(monkeypatch, tmp_path):
    """Empty overlap remains auditable without assigning a zero effect or fake interval."""
    plan = _plan(count=1)
    base = _snapshot(plan, outcome=lambda trial, problem: None)
    post = _snapshot(plan, checkpoint="post")
    comparison = _analyze(monkeypatch, tmp_path, [base, post], allow_incomplete=True)["comparisons"][0]
    assert comparison["matched_groups"] == 0
    assert comparison["pass_at_1_difference"] is comparison["pass_at_1_difference_ci95"] is None


def test_group_bootstrap_preserves_unequal_question_weights(monkeypatch, tmp_path):
    """Two questions in one group cannot be resampled independently or given half the point weight."""
    plan = _plan(count=3, samples=1, group_ids=("related", "related", "independent"))
    snapshot = _snapshot(plan, outcome=lambda trial, problem: (
        f"Final answer: {problem.answer}" if problem.id == "q-2" else "bad"
    ))
    original = metrics._percentile
    observed = []

    def capture(ordered, probability):
        observed.extend(ordered)
        return original(ordered, probability)

    monkeypatch.setattr(metrics, "_percentile", capture)
    cell = _analyze(monkeypatch, tmp_path, [snapshot])["cells"][0]
    assert cell["pass_at_1"] == pytest.approx(1 / 3)
    assert cell["included_groups"] == 2
    # Drawing two clusters allows only (A,A), (A,B), or (B,B), with unequal sizes retained.
    assert set(observed) == {0.0, 1 / 3, 1.0}


def test_repair_cue_contrasts_use_definitions_not_guessed_names(monkeypatch, tmp_path):
    """Arbitrary valid labels still generate check-minus-neutral for both draft variants."""
    prompts = (Prompt(name="control-xyz", cue="neutral"), Prompt(name="intervention-abc", cue="check"))
    plan = _plan(count=1, task="repair", prompts=prompts)
    snapshot = _snapshot(plan, outcome=lambda trial, problem: (
        f"Final answer: {problem.answer}" if trial.prompt == "intervention-abc" else "bad"
    ))
    comparisons = _analyze(monkeypatch, tmp_path, [snapshot])["comparisons"]
    assert {row["prefix"] for row in comparisons} == {"correct", "error"}
    for row in comparisons:
        assert row["kind"] == "check_minus_neutral"
        assert (row["reference_prompt"], row["target_prompt"]) == ("control-xyz", "intervention-abc")
        assert row["pass_at_1_difference"] == 1.0


def test_native_minus_common_math_uses_actual_format_definitions(monkeypatch, tmp_path):
    """The completion/native intervention does not depend on fixed configuration labels."""
    plan = _plan(count=1, prompts=(Prompt(name="plain-abc"), Prompt(name="chat-xyz", format="native")))
    report = _analyze(monkeypatch, tmp_path, [_snapshot(plan)])
    row = report["comparisons"][0]
    assert row["kind"] == "native_minus_common"
    assert (row["reference_prompt"], row["target_prompt"]) == ("plain-abc", "chat-xyz")


def test_k_grid_includes_n_even_when_not_power_of_two(monkeypatch, tmp_path):
    """Finite estimates stop at the declared count and include that count as sampled oracle coverage."""
    cell = _analyze(monkeypatch, tmp_path, [_snapshot(_plan(count=1, samples=5))])["cells"][0]
    assert set(cell["pass_at_k"]) == {"1", "2", "4", "5"}


def test_empty_and_duplicate_input_paths_fail_before_read(monkeypatch, tmp_path):
    """Canonical path aliases cannot make one run appear to provide independent jobs."""
    def unexpected_read(path):
        raise AssertionError("Invalid directory list must not read snapshots.")

    monkeypatch.setattr(metrics, "read_snapshot", unexpected_read)
    with pytest.raises(ValueError, match="nonempty"):
        analyze_runs([])
    with pytest.raises(ValueError, match="Duplicate input paths"):
        analyze_runs([tmp_path, tmp_path / "nested" / ".."])
    with pytest.raises(ValueError, match="boolean"):
        analyze_runs([tmp_path], allow_incomplete=1)


def test_mixed_studies_and_incompatible_embedded_science_rejected(monkeypatch, tmp_path):
    """IDs and embedded science are both compared; relocation/timestamps alone are not science."""
    plan = _plan()
    base, post = _snapshot(plan), _snapshot(plan, checkpoint="post")
    other = _snapshot(_plan(samples=5), checkpoint="post")
    with pytest.raises(ValueError, match="study_id"):
        _analyze(monkeypatch, tmp_path, [base, other])
    # Deliberately bypass the verified reader to exercise the analyzer's second-line comparison.
    changed_spec = plan.spec.model_copy(update={"hypothesis": "Different scientific claim"})
    changed_plan = plan.model_copy(update={"spec": changed_spec})
    changed = replace(post, metadata=post.metadata.model_copy(update={"plan": changed_plan}))
    with pytest.raises(ValueError, match="specs/plans"):
        _analyze(monkeypatch, tmp_path, [base, changed])
    relocated = plan.model_copy(update={
        "created_at": datetime(2026, 2, 1, tzinfo=UTC),
        "spec": plan.spec.model_copy(update={"dataset": tmp_path / "not-needed.jsonl"}),
    })
    moved = replace(post, metadata=post.metadata.model_copy(update={"plan": relocated}))
    assert not _analyze(monkeypatch, tmp_path, [base, moved])["incomplete"]


@pytest.mark.parametrize("change", [
    {"device": "cuda"}, {"threads": 2}, {"packages": {"torch": "another-version"}},
    {"gpu": "different-chip"}, {"allow_tf32_matmul": True},
])
def test_runtime_profile_differences_are_not_silently_pooled(monkeypatch, tmp_path, change):
    """Hardware, libraries, and numerical settings are comparison compatibility constraints."""
    plan = _plan()
    base = _snapshot(plan)
    post = _snapshot(plan, checkpoint="post", runtime={**base.metadata.runtime_profile, **change})
    with pytest.raises(ValueError, match="numerical runtime profiles"):
        _analyze(monkeypatch, tmp_path, [base, post])


def test_device_ordinals_are_ignored_but_actual_model_precision_is_not(monkeypatch, tmp_path):
    """CUDA_VISIBLE_DEVICES renumbering is not a numerical-profile change."""
    plan = _plan()
    base = _snapshot(plan, runtime={"device": "cuda:0", "gpu": "fake-chip"})
    post = _snapshot(plan, checkpoint="post", runtime={"device": "cuda:7", "gpu": "fake-chip"})
    assert _analyze(monkeypatch, tmp_path, [base, post])["runtime_profile"]["device"] == "cuda"
    changed = replace(post, metadata=post.metadata.model_copy(update={
        "model_profile": {**post.metadata.model_profile, "dtype": "torch.bfloat16"},
    }))
    with pytest.raises(ValueError, match="numerical runtime profiles"):
        _analyze(monkeypatch, tmp_path, [base, changed])


def test_duplicate_empty_job_is_rejected_without_success_overlap(monkeypatch, tmp_path):
    """Copying two incomplete directories never creates independent worker partitions."""
    snapshot = _snapshot(_plan(), outcome=lambda trial, problem: None)
    with pytest.raises(ValueError, match="job identities"):
        _analyze(monkeypatch, tmp_path, [snapshot, snapshot], allow_incomplete=True)


def test_mixed_partitioning_rejected_even_with_no_records(monkeypatch, tmp_path):
    """A change in modulo partition count cannot be repaired by pooling surviving trials."""
    plan = _plan()
    unsharded = _snapshot(plan, outcome=lambda trial, problem: None)
    sharded = _snapshot(plan, job=Job(checkpoint="base", shard_index=1, num_shards=2),
                        outcome=lambda trial, problem: None)
    with pytest.raises(ValueError, match="Mixed partitioning"):
        _analyze(monkeypatch, tmp_path, [unsharded, sharded], allow_incomplete=True)


def test_full_schedule_required_even_when_each_supplied_shard_is_complete(monkeypatch, tmp_path):
    """Completion is measured against all plan groups, not merely provided shard schedules."""
    plan = _plan(count=1)
    partitions = [_snapshot(plan, job=Job(checkpoint="base", shard_index=index, num_shards=2))
                  for index in range(2)]
    empty = next(snapshot for snapshot in partitions if not snapshot.records)
    with pytest.raises(ValueError, match="4/4 missing"):
        _analyze(monkeypatch, tmp_path, [empty])
    report = _analyze(monkeypatch, tmp_path, [empty], allow_incomplete=True)
    assert report["completion"][0]["missing_trials"] == 4
    assert len(report["completion"][0]["shards_absent"]) == 1
    complete = _analyze(monkeypatch, tmp_path, partitions)
    assert complete["completion"][0]["shards_absent"] == []
    assert complete["cells"][0]["included_problems"] == 1


def test_model_profile_changes_within_checkpoint_shards_are_rejected(monkeypatch, tmp_path):
    """Different weight/tokenizer fingerprints on the same named checkpoint cannot pool."""
    plan = _plan()
    left = _snapshot(plan, job=Job(checkpoint="base", shard_index=0, num_shards=2))
    right = _snapshot(plan, job=Job(checkpoint="base", shard_index=1, num_shards=2))
    right = replace(right, metadata=right.metadata.model_copy(update={
        "model_profile": {**right.metadata.model_profile, "tokenizer_sha256": "2" * 64},
    }))
    with pytest.raises(ValueError, match="model profiles within"):
        _analyze(monkeypatch, tmp_path, [left, right], allow_incomplete=True)


@pytest.mark.parametrize("different_trial_id", [False, True])
def test_duplicate_successful_observations_are_rejected(monkeypatch, tmp_path, different_trial_id):
    """Neither changed UUIDs nor forged trial IDs can duplicate one logical sample."""
    plan = _plan(count=1)
    original = _snapshot(plan).records[0]
    copied_trial = original.trial.model_copy(update={"trial_id": "f" * 64}) if different_trial_id else original.trial
    duplicate = original.model_copy(update={"attempt_id": uuid4(), "trial": copied_trial})
    # The real reader rejects bad shard assignments; here we isolate cross-snapshot defenses.
    left = replace(_snapshot(plan, job=Job(checkpoint="base", shard_index=0, num_shards=2)), records=(original,))
    right = replace(_snapshot(plan, job=Job(checkpoint="base", shard_index=1, num_shards=2)), records=(duplicate,))
    with pytest.raises(ValueError, match="Duplicate successful logical"):
        _analyze(monkeypatch, tmp_path, [left, right], allow_incomplete=True)


def test_duplicate_failed_attempt_id_and_invalid_sample_count_rejected(monkeypatch, tmp_path):
    """Retry bookkeeping and arithmetic samples remain explicit even at the mock reader seam."""
    snapshot = _snapshot(_plan(count=1))
    failed = _attempt(snapshot.metadata, snapshot.records[0].trial, "", status="failed")
    with pytest.raises(ValueError, match="Duplicate attempt"):
        _analyze(monkeypatch, tmp_path, [replace(snapshot, records=(failed, failed))], allow_incomplete=True)
    bad = snapshot.records[0].model_copy(update={
        "trial": snapshot.records[0].trial.model_copy(update={"sample": 4}),
    })
    with pytest.raises(ValueError, match="sample indices"):
        _analyze(monkeypatch, tmp_path, [replace(snapshot, records=(bad,))], allow_incomplete=True)


def test_real_store_roundtrip_reports_are_new_atomic_derived_outputs(tmp_path):
    """Validated raw grades/checksums flow through the real reader into five small artifacts."""
    snapshot = _snapshot(_plan(count=1))
    run = tmp_path / "run"
    with StudyStore(run).writer(snapshot.metadata) as store:
        for attempt in snapshot.records:
            store.append(attempt)
    before = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    report = analyze_runs([run], bootstrap_samples=20)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    output = tmp_path / "derived" / "report"
    write_report(report, output)
    assert {path.name for path in output.iterdir()} == {
        "report.json", "cells.csv", "comparisons.csv", "problems.csv", "report.md",
    }
    assert json.loads((output / "report.json").read_text()) == report
    with (output / "cells.csv").open(newline="") as handle:
        cell = next(csv.DictReader(handle))
    assert cell["pass_at_k.2"] == "1.0" and cell["included_trials"] == "4"
    assert (output / "comparisons.csv").read_text().strip() == "kind"
    text = (output / "report.md").read_text()
    assert "FLOPs" in text and "training seeds" in text and "no significance" in text.lower()
    assert "| Checkpoint |" in text and "95% CI" in text
    for existing in (output, run):
        with pytest.raises(ValueError, match="already exists"):
            write_report(report, existing)
    assert before == {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}


def test_report_refuses_empty_existing_directory_and_dangling_symlink(monkeypatch, tmp_path):
    """No-overwrite applies even to empty destinations and links to absent targets."""
    report = _analyze(monkeypatch, tmp_path, [_snapshot(_plan(count=1))])
    empty = tmp_path / "empty"
    empty.mkdir()
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "absent")
    for output in (empty, link):
        with pytest.raises(ValueError, match="already exists"):
            write_report(report, output)


def test_report_escapes_markdown_and_formula_like_csv_text(monkeypatch, tmp_path):
    """Derived labels cannot add raw HTML/table rows or spreadsheet formulas."""
    report = _analyze(monkeypatch, tmp_path, [_snapshot(_plan(count=1))])
    report["study_name"] = "<script>bad</script>|new\nrow"
    report["cells"][0]["prompt"] = "=1+1"
    output = tmp_path / "escaped-report"
    write_report(report, output)
    with (output / "cells.csv").open(newline="") as handle:
        assert next(csv.DictReader(handle))["prompt"] == "'=1+1"
    markdown = (output / "report.md").read_text()
    assert "<script>" not in markdown
    assert "&lt;script&gt;bad&lt;/script&gt;\\|new row" in markdown
    # JSON remains faithful to the report; escaping applies only to human/spreadsheet rendering.
    assert json.loads((output / "report.json").read_text())["cells"][0]["prompt"] == "=1+1"


def test_report_rejects_nonfinite_json_before_creating_output(monkeypatch, tmp_path):
    """Unavailable estimates use null, never nonstandard NaN values or partial serialization."""
    report = _analyze(monkeypatch, tmp_path, [_snapshot(_plan(count=1))])
    report["cells"][0]["pass_at_1"] = float("nan")
    output = tmp_path / "invalid-report"
    with pytest.raises(ValueError):
        write_report(report, output)
    assert not output.exists()