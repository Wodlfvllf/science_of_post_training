"""Complete offline pilot wiring; random models cannot validate the hypotheses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_pilot.analysis import analyze_runs
from vlm_pilot.cli import main
from vlm_pilot.data import generate_dataset
from vlm_pilot.io import ResultStore, atomic_json
from vlm_pilot.plan import (
    ROUTING_CONDITIONS,
    freeze_protocol,
    lock_plan,
    read_plan,
    read_protocol,
    schedule,
    validate_job,
)
from vlm_pilot.runner import execute_plan, inspect_run
from vlm_pilot.schemas import Job, PilotError, Selection, SiteSpec


@pytest.fixture
def pilot(tmp_path):
    data = tmp_path / "data"
    generate_dataset(data, engineering=1, discovery=2, pilot_validation=2, confirmation=2)
    config = tmp_path / "config.yaml"
    config.write_text("""name: fixture-pilot
dataset: data
max_new_tokens: 2
inference: {device: cpu, dtype: float32, max_context_tokens: 1024}
checkpoints:
  - {name: before, backend: fixture, family: fixture, stage: mpo, model_id: offline-random-qwen3, revision: fixture-v1, fixture_seed: 17}
  - {name: after, backend: fixture, family: fixture, stage: gspo, model_id: offline-random-qwen3, revision: fixture-v1, fixture_seed: 19}
""")
    return lock_plan(config, tmp_path / "plan.json"), tmp_path, config


def test_plan_schedule_ids_factors_partitions_and_tamper(pilot):
    plan, root, _ = pilot
    assert read_plan(root / "plan.json", check_source=True) == plan
    job = Job(checkpoint="before", phase="routing", split="discovery", sites=(SiteSpec(layer=0, kv_group=0),))
    trials = [row for _, _, row in schedule(plan, job)]
    assert len(trials) == 2*len(ROUTING_CONDITIONS)
    assert len({r["trial_id"] for r in trials}) == len(trials)
    parts = [r for shard in range(2) for _, _, r in schedule(plan,
             job.model_copy(update={"num_shards": 2, "shard_index": shard}))]
    assert {r["trial_id"] for r in parts} == {r["trial_id"] for r in trials}
    cases = {r["condition"]: r for r in trials[:len(ROUTING_CONDITIONS)]}
    assert cases["q_flip"]["target"] != cases["q_flip"]["gold"]
    assert cases["qk_cancel"]["target"] == cases["qk_cancel"]["gold"]
    assert cases["v_swap"]["target"] != cases["v_swap"]["gold"]
    raw = json.loads((root / "plan.json").read_text())
    raw["spec"]["inference"]["answer_prefix"] = "Changed:"
    (root / "tampered.json").write_text(json.dumps(raw))
    with pytest.raises(PilotError, match="identity"):
        read_plan(root / "tampered.json")


def test_frozen_protocol_required_and_rejects_site_or_source_drift(pilot):
    plan, root, _ = pilot
    site = SiteSpec(layer=0, kv_group=0)
    job = Job(checkpoint="before", phase="weights", split="confirmation", sites=(site,),
              source_checkpoint="after")
    with pytest.raises(PilotError, match="protocol"):
        validate_job(plan, job)
    protocol = freeze_protocol(plan, (
        Selection(checkpoint="before", sites=(site,), source_checkpoint="after"),
        Selection(checkpoint="after", sites=(site,), source_checkpoint="before"),
    ), root / "protocol.json", "Tiny random engineering selections, not scientific evidence.")
    validate_job(plan, job, protocol)
    assert read_protocol(root / "protocol.json") == protocol
    with pytest.raises(PilotError, match="protocol"):
        validate_job(plan, job.model_copy(update={"sites": (SiteSpec(layer=1, kv_group=0),)}), protocol)
    with pytest.raises(PilotError, match="adjacent"):
        validate_job(plan, job.model_copy(update={"source_checkpoint": "before"}), protocol)


def test_baseline_resume_analysis_and_completed_model_free_fast_path(pilot, monkeypatch):
    plan, root, _ = pilot
    before, after = root / "before", root / "after"
    job = Job(checkpoint="before", phase="baseline", split="discovery", generate=True)
    result = execute_plan(plan, job, before, max_new_trials=1)
    assert result["completed"] == 1 and result["missing"] == 23 and result["fixture_only"]
    with pytest.raises(PilotError, match="Incomplete"):
        analyze_runs([before], root / "incomplete")
    result = execute_plan(plan, job, before)
    assert result["completed"] == 24 and result["new_attempts"] == 23
    execute_plan(plan, Job(checkpoint="after", phase="baseline", split="discovery", generate=True), after)
    report = analyze_runs([before, after], root / "report", bootstrap_samples=20)
    assert report["complete"] and report["fixture_only"] and report["stage_contrasts"]
    assert report["hypothesis_decision"] == "not_automated"
    assert (root / "report" / "REPORT.md").is_file()
    import vlm_pilot.runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed resume must not allocate a model")

    monkeypatch.setattr(vlm_pilot.runtime, "Engine", forbidden)
    assert execute_plan(plan, job, before)["model_loaded"] is False
    assert execute_plan(plan, job, before, max_new_trials=0)["model_loaded"] is False
    assert inspect_run(before)["complete"]


def test_routing_and_weights_end_to_end_with_saved_capture(pilot):
    plan, root, _ = pilot
    site = SiteSpec(layer=0, kv_group=0)
    routing = Job(checkpoint="before", phase="routing", split="discovery", sites=(site,),
                  save_captures=True)
    output = root / "routing"
    result = execute_plan(plan, routing, output)
    assert result["complete"] and result["completed"] == 2*len(ROUTING_CONDITIONS)
    records = ResultStore(output).read().records
    assert all(r["artifacts"] for r in records)
    self_rows = [r for r in records if r["condition"] == "self_qk"]
    assert all(r["routing"]["0:0"]["text_weight_maxdiff"] == 0 for r in self_rows)
    weights = Job(checkpoint="after", phase="weights", split="discovery", sites=(site,),
                  source_checkpoint="before")
    result = execute_plan(plan, weights, root / "weights")
    assert result["complete"] and result["completed"] == 34
    records = ResultStore(root / "weights").read().records
    edited = [r for r in records if "weight_swap" in r]
    assert edited and all(r["weight_swap"]["restored_bitexact"] for r in edited)
    assert any("identical_input_replay" in r for r in records)
    assert all(r["weight_direction"] == "rollback" for r in records)
    report = analyze_runs([root / "weights"], root / "weight-report", bootstrap_samples=20)
    assert report["condition_contrasts"] and report["h3_selectivity"]
    selected = {(r["family_id"], r["case_id"], r["condition"]): r for r in records}
    for row in records:
        if row["condition"] == "weight_sham" and row["task"] == "binding":
            clamped = selected[(row["family_id"], row["case_id"], "sham_clamp")]
            assert [s["applied_slice_checksum"] for s in row["weight_swap"]["slices"]] == [
                s["applied_slice_checksum"] for s in clamped["weight_swap"]["slices"]]


def test_failure_persists_without_completion_and_resumes(pilot, monkeypatch):
    plan, root, _ = pilot
    from vlm_pilot.runner import FamilyExecutor

    original = FamilyExecutor.execute

    def fail(*args):
        raise PilotError("Controlled test failure")

    monkeypatch.setattr(FamilyExecutor, "execute", fail)
    job = Job(checkpoint="before", phase="baseline", split="discovery")
    with pytest.raises(PilotError, match="failed"):
        execute_plan(plan, job, root / "failed", max_new_trials=1)
    assert inspect_run(root / "failed")["completed"] == 0
    assert inspect_run(root / "failed")["failed_attempts"] == 1
    monkeypatch.setattr(FamilyExecutor, "execute", original)
    assert execute_plan(plan, job, root / "failed", max_new_trials=1)["completed"] == 1


def test_orphans_ignored_and_semantic_or_checksum_corruption_rejected(pilot):
    plan, root, _ = pilot
    job = Job(checkpoint="before", phase="baseline", split="discovery")
    output = root / "run"
    execute_plan(plan, job, output, max_new_trials=1)
    atomic_json(output / "records" / "orphan.json", {"ignored": True}, immutable=True)
    assert inspect_run(output)["completed"] == 1
    store = ResultStore(output).read()
    ref = store.manifest["records"][0]
    file = output / ref["path"]
    file.write_bytes(file.read_bytes() + b" ")
    with pytest.raises(PilotError, match="checksum"):
        inspect_run(output)


def test_cli_help_preview_and_no_intervention_confirmation_without_protocol(pilot, capsys):
    plan, root, config = pilot
    assert main(["preview", str(config)]) == 0
    assert '"model_loaded": false' in capsys.readouterr().out
    assert main(["run", str(root / "plan.json"), "--checkpoint", "before", "--phase", "routing",
                 "--split", "confirmation", "--sites", "0:0", "--output", str(root / "blocked")]) == 1
    assert not (root / "blocked").exists()
    assert "protocol" in capsys.readouterr().err


def test_active_text_source_is_not_part_of_separate_pilot(pilot):
    from interp.config import source_digest as text_digest
    from vlm_pilot.plan import source_digest

    assert source_digest() != text_digest()
    assert Path(__file__).resolve().parents[2] / "src" / "interp" != Path(__file__).parent


def test_component_audit_and_zero_budget_identity_guard(pilot):
    from vlm_pilot.audit import audit_components
    from vlm_pilot.plan import checkpoint_for

    plan, root, _ = pilot
    report = audit_components(checkpoint_for(plan, "before"), checkpoint_for(plan, "after"))
    assert report["bitexact_equal"] and report["fixture_only"]
    job = Job(checkpoint="before", phase="baseline", split="discovery")
    execute_plan(plan, job, root / "identity", max_new_trials=1)
    with pytest.raises(PilotError, match="different"):
        execute_plan(plan, job.model_copy(update={"checkpoint": "after"}), root / "identity",
                     max_new_trials=0)


def test_capture_preflight_failure_blocks_all_trials_and_saves_measurements(pilot, monkeypatch):
    import vlm_pilot.runtime

    plan, root, _ = pilot

    def fail_preflight(*args, **kwargs):
        return {"passed": False, "checks": {"controlled_failure": True}}

    monkeypatch.setattr(vlm_pilot.runtime.Engine, "preflight", fail_preflight)
    with pytest.raises(PilotError, match="preflight"):
        execute_plan(plan, Job(checkpoint="before", phase="baseline", split="discovery"),
                     root / "no-trials")
    assert inspect_run(root / "no-trials")["completed"] == 0
    assert list((root / "no-trials").glob("preflight-*.json"))


def test_complete_unchanged_routing_sites_have_joint_signature_and_no_zero_width_ci(pilot):
    plan, root, _ = pilot
    site = SiteSpec(layer=0, kv_group=0)
    directories = []
    for checkpoint in ("before", "after"):
        output = root / (checkpoint + "-routing")
        execute_plan(plan, Job(checkpoint=checkpoint, phase="routing", split="discovery",
                              sites=(site,)), output)
        directories.append(output)
    report = analyze_runs(directories, root / "routing-report", bootstrap_samples=20)
    assert report["complete"] and report["routing_signatures"]
    assert report["routing_stage_interactions"]
    assert report["routing_stage_interactions"][0]["families"] == 2


def test_semantic_tamper_with_rehashed_bytes_is_still_rejected(pilot):
    from vlm_pilot.io import identity

    plan, root, _ = pilot
    output = root / "semantic"
    execute_plan(plan, Job(checkpoint="before", phase="baseline", split="discovery"), output,
                 max_new_trials=1)
    store = ResultStore(output).read()
    row = store.records[0]
    row["candidate_token_logprobs"] = {}
    sha = identity(row)
    path = f"records/{row['trial_id']}-{sha}.json"
    atomic_json(output / path, row, immutable=True)
    store.manifest["records"][0] = {"trial_id": row["trial_id"], "sha256": sha,
                                   "path": path, "size_bytes": (output / path).stat().st_size}
    atomic_json(output / "manifest.json", store.manifest)
    with pytest.raises(PilotError, match="candidates"):
        inspect_run(output)


def test_p0_rejects_within_tolerance_argmax_change(pilot, monkeypatch):
    import torch

    from vlm_pilot.attention import Site
    from vlm_pilot.plan import checkpoint_for
    from vlm_pilot.runtime import Engine

    plan, _, _ = pilot
    family = plan.dataset["families"][0]
    case = family["cases"][0]
    engine = Engine(checkpoint_for(plan, "before"), plan.spec.inference)
    prepared = engine.prepare(plan.spec.dataset, family, case)
    _, captures, diagnostics = engine.forward(prepared, sites=(Site(0, 0),))
    native = torch.zeros((1, 1, 128))
    native[..., 0] = 1.
    native[..., 1] = 1. - .00001
    flipped = native.clone()
    flipped[..., 1] = 1. + .00001

    def forward(prepared, *, sites=(), actions=None, keep=1):
        result = flipped if sites else native
        return result.expand(1, keep, 128), captures if sites else {}, diagnostics if sites else {}

    monkeypatch.setattr(engine, "forward", forward)
    monkeypatch.setattr(engine.model, "forward", lambda prepared: native)
    report = engine.preflight(plan.spec.dataset, family, case, sites=(Site(0, 0),))
    assert not report["passed"]
    assert report["checks"]["capture_only"]["mismatches"] == 0
    assert not report["checks"]["capture_only"]["argmax_match"]


def test_protocol_identity_uses_normalized_numeric_fields(pilot):
    plan, root, _ = pilot
    protocol = freeze_protocol(plan, (
        Selection(checkpoint="before", sites=(SiteSpec(layer=0, kv_group=0),), source_checkpoint="after"),
    ), root / "numeric-protocol.json", "Offline fixture only.", preservation_margin=0)
    assert read_protocol(root / "numeric-protocol.json") == protocol