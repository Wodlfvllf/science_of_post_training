"""Model-free reports with explicit completeness and family-paired uncertainty."""

from __future__ import annotations

import csv
from collections import defaultdict
from io import StringIO
from pathlib import Path
from statistics import mean

from .io import ResultStore, atomic_bytes, atomic_json, identity
from .plan import checkpoint_for
from .runner import inspect_run, validate_records
from .schemas import PilotError, Plan
from .stats import family_difference_interval, paired_summary, simulate_power


def _condition_contrast(rows, before, after, *, metric="gold_hit", samples=2000):
    normalized = [dict(row, checkpoint=row["condition"], condition="paired_condition")
                  for row in rows if row["condition"] in (before, after)]
    return paired_summary(normalized, before, after, metric=metric, bootstrap_samples=samples)


def analyze_runs(roots: list[Path], output: Path, *, allow_incomplete=False,
                 bootstrap_samples=2000) -> dict:
    if output.exists():
        raise PilotError("Report output exists; select a new directory.")
    rows, jobs, plans, protocols = [], [], {}, {}
    seen_trials = set()
    for root in roots:
        store = ResultStore(root).read()
        expected = validate_records(store)
        summary = inspect_run(root)
        if not summary["complete"] and not allow_incomplete:
            raise PilotError("Incomplete pilot jobs cannot produce a complete report. "
                             "Use --allow-incomplete for clearly labelled descriptive output.")
        jobs.append(summary)
        protocol = store.metadata.get("protocol")
        if protocol is not None:
            split = summary["job"]["split"]
            previous = protocols.setdefault(split, protocol)
            if previous["protocol_id"] != protocol["protocol_id"]:
                raise PilotError("Do not combine different frozen protocols within one split.")
        plan = Plan.model_validate(store.metadata["plan"])
        plans[plan.plan_id] = plan
        # Insert explicit missing trials: observed subsets must not erase denominators.
        completed = {row["trial_id"]: row for row in store.records if row["status"] == "completed"}
        for trial_id, planned in expected.items():
            if trial_id in seen_trials:
                raise PilotError("Overlapping logical trials across input runs; do not pool snapshots.")
            seen_trials.add(trial_id)
            row = completed.get(trial_id, dict(planned, status="missing"))
            rows.append(row)
    if len(plans) != 1:
        raise PilotError("Compare jobs from exactly one locked scientific plan.")
    plan = next(iter(plans.values()))
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in ("checkpoint", "split", "phase", "task", "condition", "site",
                                          "prompt_profile", "generation_audit", "operation"))
        groups[key].append(row)
    cells = []
    for key, values in sorted(groups.items(), key=repr):
        successes = [r for r in values if r["status"] == "completed"]
        by_family = defaultdict(list)
        for row in successes:
            by_family[row["family_id"]].append(row)
        fields = ("checkpoint", "split", "phase", "task", "condition", "site", "prompt_profile",
                "generation_audit", "operation")
        audited = [r["generation"] for r in successes if "generation" in r]
        cells.append(dict(zip(fields, key, strict=True)) | {
            "planned_rows": len(values), "completed_rows": len(successes),
            "source_families": len({r["family_id"] for r in values}),
            "observed_target_hit": mean(r["target_hit"] for r in successes) if successes else None,
            "observed_gold_hit": mean(r["gold_hit"] for r in successes) if successes else None,
            "complete": len(values) == len(successes),
            "elapsed_seconds": sum(r["elapsed_seconds"] for r in successes),
            "generation_audited_rows": len(audited),
            "generation_valid_fraction": mean(g["valid"] for g in audited) if audited else None,
            "generation_correct_fraction": mean(g["correct"] for g in audited) if audited else None,
            "generation_truncated_rows": sum(g["finish_reason"] == "length" for g in audited),
        })
    stage_contrasts = []
    conditions = {(r["split"], r["phase"], r["task"], r["condition"], r["site"],
                   r["prompt_profile"], r["generation_audit"], r["operation"]) for r in rows}
    for family in {c.family for c in plan.spec.checkpoints}:
        stages = {c.stage: c.name for c in plan.spec.checkpoints if c.family == family}
        for before, after in (("cpt", "sft"), ("sft", "mpo"), ("mpo", "gspo")):
            if before not in stages or after not in stages:
                continue
            for key in sorted(conditions, key=repr):
                selected = [r for r in rows if (r["split"], r["phase"], r["task"],
                            r["condition"], r["site"], r["prompt_profile"], r["generation_audit"],
                            r["operation"]) == key
                            and r["checkpoint"] in (stages[before], stages[after])]
                if {r["checkpoint"] for r in selected} != {stages[before], stages[after]}:
                    continue
                result = paired_summary(selected, stages[before], stages[after],
                                        bootstrap_samples=bootstrap_samples)
                stage_contrasts.append({"family": family, "split": key[0], "phase": key[1],
                    "task": key[2], "condition": key[3], "site": key[4],
                    "prompt_profile": key[5], "generation_audit": key[6], "operation": key[7], **result})
    controls = []
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["checkpoint"], row["split"], row["phase"], row["task"], row["site"],
                  row["prompt_profile"], row["generation_audit"], row["operation"])].append(row)
    for key, values in sorted(buckets.items(), key=repr):
        pairs = (("route_permute", "q_flip"), ("route_permute", "k_flip"),
                 ("qk_color_sham", "qk_cancel")) if key[2] == "routing" else (
                 ("native", "swap"), ("weight_sham", "swap"),
                 ("swap", "swap_clamp"), ("swap", "wrong_clamp"),
                 ("native", "route_only"), ("weight_sham", "sham_clamp")) if key[2] == "weights" else ()
        for before, after in pairs:
            if not {before, after} <= {r["condition"] for r in values}:
                continue
            if (before, after) == ("weight_sham", "sham_clamp"):
                checksum = {}
                for row in values:
                    if row["condition"] in (before, after) and row["status"] == "completed":
                        applied = tuple(r["applied_slice_checksum"] for r in row["weight_swap"]["slices"])
                        key_family = row["family_id"], row["case_id"]
                        if checksum.setdefault(key_family, applied) != applied:
                            raise PilotError("Sham/clamp comparison used different weight perturbations.")
            result = _condition_contrast(values, before, after,
                metric="target_hit" if key[2] == "routing" else "gold_hit", samples=bootstrap_samples)
            controls.append({"checkpoint": key[0], "split": key[1], "phase": key[2],
                             "task": key[3], "site": key[4], "prompt_profile": key[5],
                             "generation_audit": key[6], "operation": key[7], **result})
    interactions = []
    for family in {c.family for c in plan.spec.checkpoints}:
        pair = {c.stage: c.name for c in plan.spec.checkpoints if c.family == family}
        if not {"mpo", "gspo"} <= set(pair):
            continue
        routing_keys = {(r["split"], r["site"], r["operation"]) for r in rows if r["phase"] == "routing"
                        and r["checkpoint"] in (pair["mpo"], pair["gspo"])}
        for split, site, operation in sorted(routing_keys):
            selected = [r for r in rows if r["phase"] == "routing" and r["split"] == split
                        and r["site"] == site and r["operation"] == operation and r["condition"] in (
                            "q_flip", "k_flip", "route_permute")
                        and r["checkpoint"] in (pair["mpo"], pair["gspo"])]
            index = {(r["checkpoint"], r["family_id"], r["condition"]): r for r in selected}
            differences = {}
            all_families = {r["family_id"] for r in selected}
            for source_family in sorted(all_families):
                stage_values = []
                for checkpoint in (pair["mpo"], pair["gspo"]):
                    records = [index.get((checkpoint, source_family, condition)) for condition in (
                               "q_flip", "k_flip", "route_permute")]
                    if not all(r and r["status"] == "completed" for r in records):
                        break
                    stage_values.append((records[0]["target_hit"] + records[1]["target_hit"]) / 2
                                        - records[2]["target_hit"])
                if len(stage_values) == 2:
                    differences[source_family] = stage_values[1] - stage_values[0]
            selection = next((s for s in protocols.get(split, {}).get("selections", [])
                              if s["checkpoint"] == pair["mpo"]), None)
            sites_count = len(selection["sites"]) if selection is not None else 1
            estimate = family_difference_interval(differences, samples=bootstrap_samples,
                                                 alpha=.05/sites_count)
            if split == "discovery":
                estimate["prospective_power"] = simulate_power(differences, samples=bootstrap_samples,
                                                               alpha=.05/sites_count)
            missing = sorted(all_families - set(differences))
            if missing:
                estimate["state"] = "incomplete"
            interactions.append({"family": family, "split": split, "site": site, "operation": operation,
                "before": pair["mpo"], "after": pair["gspo"],
                "estimand": "stage_change_in_mean_Q_K_hybrid_hits_minus_route_permutation_hits",
                "missing_families": missing, **estimate})
    signatures = []
    for key, values in buckets.items():
        if key[2] != "routing":
            continue
        families = defaultdict(dict)
        for row in values:
            families[row["family_id"]][row["condition"]] = row
        hits, payload, copy_rates = [], [], []
        missing = []
        for source_family, mapping in families.items():
            required = ("q_flip", "k_flip", "qk_cancel", "q_samepalette", "k_samepalette",
                        "qk_samepalette")
            if not all(mapping.get(c) and mapping[c]["status"] == "completed" for c in required):
                missing.append(source_family)
                continue
            hits.append(all(mapping[c]["target_hit"] for c in required))
            payload.append(all(mapping[c]["recipient_payload_answer"] for c in required))
            copy_rates.extend(mapping[c]["donor_answer_copy"] for c in ("q_flip", "k_flip", "qk_cancel"))
        signatures.append({"checkpoint": key[0], "split": key[1], "site": key[4], "operation": key[7],
            "joint_flip_cancellation_success": mean(hits) if hits else None,
            "recipient_payload_fraction": mean(payload) if payload else None,
            "disjoint_donor_copy_fraction": mean(copy_rates) if copy_rates else None,
            "complete_families": len(hits), "missing_families": sorted(missing)})
    h1_selectivity = []
    for split in sorted({r["split"] for r in rows if r["phase"] == "baseline"}):
        for family in {c.family for c in plan.spec.checkpoints}:
            pair = {c.stage: c.name for c in plan.spec.checkpoints if c.family == family}
            if not {"mpo", "gspo"} <= set(pair):
                continue
            data = [r for r in rows if r["phase"] == "baseline" and r["split"] == split
                    and r["prompt_profile"] == "common" and not r["generation_audit"]
                    and r["checkpoint"] in (pair["mpo"], pair["gspo"])
                    and r["task"] in ("binding", "color_readout")]
            groups_by_family = defaultdict(list)
            for row in data:
                groups_by_family[row["family_id"]].append(row)
            differences, missing = {}, []
            for source_family, values in groups_by_family.items():
                if any(r["status"] != "completed" for r in values):
                    missing.append(source_family)
                    continue
                means = {}
                for checkpoint in (pair["mpo"], pair["gspo"]):
                    for task in ("binding", "color_readout"):
                        selected = [r for r in values if r["checkpoint"] == checkpoint and r["task"] == task]
                        if selected:
                            means[(checkpoint, task)] = mean(r["gold_hit"] for r in selected)
                if len(means) != 4:
                    missing.append(source_family)
                    continue
                differences[source_family] = (means[(pair["gspo"], "binding")]
                    - means[(pair["mpo"], "binding")] - means[(pair["gspo"], "color_readout")]
                    + means[(pair["mpo"], "color_readout")])
            estimate = family_difference_interval(differences, samples=bootstrap_samples)
            if missing:
                estimate["state"] = "incomplete"
            h1_selectivity.append({"family": family, "split": split,
                "estimand": "MPO_to_GSPO_binding_change_minus_cued_color_readout_change",
                "missing_families": missing, **estimate})
    h3_selectivity = []
    weight_groups = defaultdict(list)
    for row in rows:
        if row["phase"] == "weights":
            weight_groups[(row["checkpoint"], row["split"], row["site"], row["operation"])].append(row)
    for key, values in weight_groups.items():
        sources = {r["source_checkpoint"] for r in values}
        if len(sources) != 1:
            raise PilotError("Weight contrast must use one exact source checkpoint.")
        by_family = defaultdict(dict)
        for row in values:
            by_family[row["family_id"]][(row["task"], row["condition"])] = row
        for control_task in ("color_readout", "edge_readout", "text_binding"):
            differences, missing = {}, []
            for family_id, mapping in by_family.items():
                required = [(task, condition) for task in ("binding", control_task)
                            for condition in ("native", "swap")]
                if not all(mapping.get(k) and mapping[k]["status"] == "completed" for k in required):
                    missing.append(family_id)
                    continue
                differences[family_id] = (mapping[("binding", "swap")]["gold_hit"]
                    - mapping[("binding", "native")]["gold_hit"]
                    - mapping[(control_task, "swap")]["gold_hit"]
                    + mapping[(control_task, "native")]["gold_hit"])
            estimate = family_difference_interval(differences, samples=bootstrap_samples)
            if missing:
                estimate["state"] = "incomplete"
            h3_selectivity.append({"checkpoint": key[0], "split": key[1], "site": key[2],
                "operation": key[3], "source_checkpoint": next(iter(sources)),
                "control_task": control_task,
                "estimand": "weight_swap_binding_effect_minus_control_task_effect",
                "missing_families": missing, **estimate})
    comparison_complete = all(not result["missing_families"] for result in (
        *interactions, *h1_selectivity, *h3_selectivity)) and all(
        result["missing_before"] == 0 and result["missing_after"] == 0
        for result in (*stage_contrasts, *controls))
    if not comparison_complete and not allow_incomplete:
        raise PilotError("Paired comparison coverage is incomplete across checkpoints/shards; "
                         "use --allow-incomplete for descriptive output only.")
    report = {"schema_version": 1, "plan_id": plan.plan_id,
              "complete": all(j["complete"] for j in jobs) and comparison_complete,
              "jobs_complete": all(j["complete"] for j in jobs),
              "comparisons_complete": comparison_complete, "jobs": jobs, "cells": cells,
              "stage_contrasts": stage_contrasts, "condition_contrasts": controls,
              "routing_stage_interactions": interactions,
              "routing_signatures": signatures, "frozen_protocols": protocols,
              "h1_selectivity": h1_selectivity,
              "h3_selectivity": h3_selectivity,
              "fixture_only": all(checkpoint_for(plan, j["job"]["checkpoint"]).backend == "fixture"
                                  for j in jobs),
              "hypothesis_decision": "not_automated",
              "limitations": ["Candidate ranking is not unrestricted generation accuracy.",
                              "No correctness-based source filtering is used.",
                              "Intervals resample independent source families, not edits.",
                              "Null control tests do not prove equivalence or preserved perception.",
                              "Endpoint surgery does not isolate optimizer/data/reward causal effects.",
                              "A random fixture has no scientific meaning."]}
    output.mkdir(parents=True)
    atomic_json(output / "report.json", report, immutable=True)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(cells[0]) if cells else ["checkpoint"])
    writer.writeheader()
    writer.writerows(cells)
    atomic_bytes(output / "cells.csv", buffer.getvalue().encode(), immutable=True)
    lines = ["# Reading-or-Routing pilot report", "",
             f"Plan: {plan.plan_id}", f"Complete: {report['complete']}",
             f"Random fixture only: {report['fixture_only']}", "",
             "## Source-family paired stage changes", "",
             "| Pair | Phase/task | Condition/site | Families | Change | 95% interval |",
             "| --- | --- | --- | ---: | ---: | --- |"]
    for result in stage_contrasts:
        difference = result["difference"]
        lines.append(f"| {result['before']} → {result['after']} | {result['phase']}/{result['task']} "
                     f"| {result['condition']}/{result['site']} | {result['paired_families']} "
                     f"| {difference if difference is not None else 'NA'} | {result['ci95']} |")
    lines += ["", "## Limitations", "", *("- " + text for text in report["limitations"])]
    atomic_bytes(output / "REPORT.md", "\n".join(lines).encode(), immutable=True)
    atomic_json(output / "integrity.json", {"report_sha256": identity(report),
                "input_runs": len(roots), "logical_trials": len(seen_trials)}, immutable=True)
    return report