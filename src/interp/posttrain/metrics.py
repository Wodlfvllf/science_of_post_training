"""Inspect numerical studies without loading models or starting inference.

``analyze_runs`` reads verified publication snapshots, joins disjoint jobs, and
estimates bounded-rollout outcomes only from complete independent problem groups.
``write_report`` publishes small derived JSON/CSV/Markdown artifacts in a *new*
directory. Neither entry point changes a run, imports a numerical/model library,
downloads data, nor makes significance or latent-capability claims.
"""

from __future__ import annotations

import csv
import html
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any

from ..artifacts import atomic_write
from .schemas import Attempt, Problem, StudyPlan
from .storage import read_snapshot

_CLAIMS = ("correct", "incorrect", "mixed", "not_observed")
_LIMITATIONS = [
    "Tokens are not equal FLOPs or latency; elapsed time is observed rollout time, not training cost.",
    "Group bootstraps are conditional on observed rollouts, not training seeds or new rollouts.",
    "Finite pass@k and sampled oracle coverage are not capability boundaries; no arbitrary extrapolation.",
    "Oracle coverage uses a gold checker, not deployable answer selection.",
    "Solved overlap is descriptive sampled overlap, not overlap of latent capabilities.",
    "Assisted repair is not autonomous detection of naturally occurring errors.",
    "Step claims and propagated-error matches are observable text labels, not causal detectors.",
    "Missing claims mean not observed, not absence of checking; prompt contrasts are not causal isolation.",
    "One checkpoint lineage does not isolate training algorithms; benchmark contamination is not excluded.",
    "Incomplete-group exclusions may be selective; available complete groups need not represent the plan.",
    "Intervals with few groups can be degenerate or unstable; no significance claims are made.",
]


def _bootstrap_options(bootstrap_samples: int, seed: int) -> None:
    """Require bounded integer resampling and an isolated, reproducible 32-bit seed."""
    # Booleans are integers in Python, but are not intentional experiment settings.
    if type(bootstrap_samples) is not int or not 1 <= bootstrap_samples <= 100_000:
        raise ValueError("bootstrap_samples must be an integer from 1 to 100000 (zero is not supported).")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a nonnegative 32-bit integer.")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Return the unbiased finite-sample estimator ``1 - C(n-c,k)/C(n,k)``.

    ``n`` must be a positive integer, ``c`` an integer in [0,n], and ``k`` an
    integer in [1,n]; booleans and coerced floats are refused. Counts must describe
    a complete set of planned bounded samples, not just surviving attempts. The
    caller enforces that design constraint; this helper checks arithmetic bounds.
    Integer combinations are exact until the final division. No extrapolation
    beyond n or plug-in ``1-(1-c/n)**k`` approximation is performed.
    """
    if any(type(value) is not int for value in (n, c, k)):
        raise ValueError("n, c, and k must be integers, not booleans or floats.")
    if n < 1 or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Require n >= 1, 0 <= c <= n, and 1 <= k <= n.")
    # Subtract integers first to avoid cancellation when the success rate is tiny.
    total = math.comb(n, k)
    return (total - math.comb(n - c, k)) / total


def _percentile(ordered: list[float], probability: float) -> float:
    """Linearly interpolate sorted values at zero-based rank (B-1)*probability.

    This is the inclusive/type-7 percentile convention, including the endpoints.
    A single bootstrap replicate returns that replicate at every percentile.
    """
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if ordered[lower] == ordered[upper]:
        return ordered[lower]
    weight = position - lower
    # A convex combination avoids overflowing upper-lower for large opposite signs.
    return math.fsum(((1 - weight) * ordered[lower], weight * ordered[upper]))


def _group_interval(
    groups: list[list[float]], bootstrap_samples: int, seed: int,
) -> tuple[float, float] | None:
    """Resample whole groups, retaining the question-weighted mean for unequal sizes.

    Each draw samples G group indices with replacement. A drawn group brings all
    its question values, so the replicate is sum(values)/number of questions,
    *not* an unweighted mean of group means. Empty analyses have no interval.
    """
    if not groups:
        return None
    totals = [math.fsum(group) for group in groups]
    sizes = [len(group) for group in groups]
    rng = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        # Matched conditions have already been differenced within each question.
        selected = [rng.randrange(len(groups)) for _ in groups]
        count = sum(sizes[index] for index in selected)
        # Scale before adding so finite large paired differences do not overflow their mean.
        estimates.append(math.fsum(totals[index] / count for index in selected))
    estimates.sort()
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def paired_bootstrap(
    values: list[float], *, bootstrap_samples: int = 1000, seed: int = 2026,
) -> tuple[float, float]:
    """Return a 95% percentile CI for the mean of independent *paired differences*.

    Supply one finite numeric difference per independent equal-weight group, not
    two separately resampled condition arrays or individual correlated rollouts.
    ``analyze_runs`` additionally handles groups containing unequal numbers of
    questions through a question-weighted cluster bootstrap. Sampling uses a local
    ``random.Random`` and never changes global RNG state. Sorted replicate means
    use linear interpolation at ranks (B-1)*0.025 and (B-1)*0.975; replicas are not
    returned. Require 1..100000 integer replicas and a 32-bit nonnegative seed.
    Zero replicas, empty values, booleans, NaN, and infinity raise ValueError.
    """
    _bootstrap_options(bootstrap_samples, seed)
    if not isinstance(values, list) or not values or any(
        type(value) not in (int, float) for value in values
    ):
        raise ValueError("values must be a nonempty list of finite numeric paired differences.")
    try:
        normalized = [float(value) for value in values]
    except OverflowError as error:
        raise ValueError("Paired differences must have a finite float representation.") from error
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("values must contain only finite numeric paired differences.")
    interval = _group_interval([[value] for value in normalized], bootstrap_samples, seed)
    assert interval is not None
    return interval


def _profile(profile: dict) -> dict:
    """Ignore only a top-level device ordinal, retaining every numerical/hardware field."""
    result = dict(profile)
    device = result.get("device")
    # Current runners already save the device type; accept older explicit ordinals too.
    if isinstance(device, str) and re.fullmatch(r"[a-z][a-z0-9_]*:[0-9]+", device):
        result["device"] = device.rsplit(":", 1)[0]
    return result


def _science(plan: StudyPlan) -> dict:
    """Compare embedded science, excluding only creation time and the unused data path."""
    return plan.model_dump(mode="json", exclude={"created_at": True, "spec": {"dataset"}})


def _problem_row(
    problem: Problem, attempts: list[Attempt], n: int, checkpoint: str, prompt: str, budget: int,
) -> dict[str, Any]:
    """Keep audit counts/costs even when missing samples make all estimates unavailable.

    Failed/cancelled attempts never supply successes or votes, even if they carry
    partial output. Retries may fill a missing logical sample once. Parse failures,
    nonconforming formats, and length stops in completed rollouts *do* consume n.
    """
    completed = [attempt for attempt in attempts if attempt.status == "completed"]
    samples = {attempt.trial.sample for attempt in completed}
    if len(samples) != len(completed) or not samples <= set(range(n)):
        raise ValueError("Duplicate completed samples or sample indices outside the planned count.")
    results = [attempt.result for attempt in completed]
    if any(result is None for result in results):
        raise ValueError("Completed attempt has no rollout.")
    statuses: dict[int, set[str]] = defaultdict(set)
    for attempt in attempts:
        statuses[attempt.trial.sample].add(attempt.status)
    reasons: Counter[str] = Counter()
    for sample in set(range(n)) - samples:
        seen = statuses[sample]
        if "failed" in seen and "cancelled" in seen:
            reasons["failed_and_cancelled_without_completion"] += 1
        elif "failed" in seen:
            reasons["failed_without_completion"] += 1
        elif "cancelled" in seen:
            reasons["cancelled_without_completion"] += 1
        else:
            reasons["no_attempt"] += 1
    # Snapshot validation guarantees results and canonical numeric answer strings.
    grades = [result.grade for result in results if result is not None]
    votes = Counter(grade.answer for grade in grades if grade.answer is not None)
    maximum_votes = max(votes.values(), default=0)
    winners = [answer for answer, count in votes.items() if count == maximum_votes]
    winner = winners[0] if len(winners) == 1 else None
    counts = {
        "correct_samples": sum(grade.correct for grade in grades),
        "format_ok_samples": sum(grade.format_ok for grade in grades),
        "parse_failure_samples": sum(grade.extraction in ("missing", "invalid") for grade in grades),
        "length_stop_samples": sum(result.generation.finish_reason == "length" for result in results),
        "unfinished_thinking_samples": sum(grade.unfinished_thinking for grade in grades),
        "propagated_error_samples": sum(grade.propagated_error for grade in grades),
        **{f"step_claim_{claim}_samples": sum(grade.step_claim == claim for grade in grades)
           for claim in _CLAIMS},
    }
    generated = sum(len(result.generation.token_ids) for result in results)
    inputs = sum(len(result.input_ids) for result in results)
    complete = len(samples) == n
    return {
        "checkpoint": checkpoint, "prompt": prompt, "budget": budget, "prefix": problem.prefix,
        "problem_id": problem.id, "source_id": problem.source_id, "group_id": problem.group_id,
        "planned_samples": n, "completed_samples": len(samples), "missing_samples": n - len(samples),
        "failed_attempts": sum(attempt.status == "failed" for attempt in attempts),
        "cancelled_attempts": sum(attempt.status == "cancelled" for attempt in attempts),
        "problem_complete": complete, "included": False, "exclusion_reasons": [],
        "missing_sample_reasons": dict(reasons), **counts,
        "parsed_votes": sum(votes.values()), "distinct_answers": len(votes),
        "vote_tied": len(winners) > 1, "no_votes": not votes,
        "majority_abstained": winner is None,
        "majority_correct": winner == problem.answer if complete else None,
        "observed_solved": counts["correct_samples"] > 0,
        "observed_generated_tokens": generated, "observed_input_tokens": inputs,
        "observed_total_tokens": generated + inputs,
        "observed_elapsed_seconds": math.fsum(result.elapsed_seconds for result in results),
        "pass_at_1": counts["correct_samples"] / n if complete else None,
    }


def _cell(rows: list[dict], bootstrap_samples: int, seed: int) -> dict:
    """Aggregate included questions with their full n, retaining explicit exclusion counts."""
    first = rows[0]
    included = [row for row in rows if row["included"]]
    groups: dict[str, list[float]] = defaultdict(list)
    for row in included:
        groups[row["group_id"]].append(row["pass_at_1"])
    size, n = len(included), first["planned_samples"]
    denominator = size * n
    ks = sorted({k for k in (1, 2, 4, 8, 16, 32, 64, n) if k <= n})
    estimates = {
        str(k): math.fsum(pass_at_k(n, row["correct_samples"], k) for row in included) / size
        if size else None for k in ks
    }
    result = {
        key: first[key] for key in ("checkpoint", "prompt", "budget", "prefix")
    }
    result.update(
        planned_problems=len(rows), complete_problems=sum(row["problem_complete"] for row in rows),
        included_problems=size, excluded_problems=len(rows) - size,
        planned_groups=len({row["group_id"] for row in rows}), included_groups=len(groups),
        excluded_groups=len({row["group_id"] for row in rows}) - len(groups),
        samples_per_problem=n, planned_trials=len(rows) * n, included_trials=denominator,
        completed_trials=sum(row["completed_samples"] for row in rows),
        excluded_completed_trials=sum(row["completed_samples"] for row in rows if not row["included"]),
        missing_trials=sum(row["missing_samples"] for row in rows),
        failed_attempts=sum(row["failed_attempts"] for row in rows),
        cancelled_attempts=sum(row["cancelled_attempts"] for row in rows),
        exclusions_by_reason=dict(Counter(reason for row in rows for reason in row["exclusion_reasons"])),
        pass_at_1=estimates["1"], pass_at_k=estimates,
        pass_at_1_ci95=_group_interval(list(groups.values()), bootstrap_samples, seed),
        oracle_coverage=sum(row["observed_solved"] for row in included) / size if size else None,
        majority_vote_accuracy=sum(row["majority_correct"] for row in included) / size if size else None,
        majority_abstention_rate=sum(row["majority_abstained"] for row in included) / size if size else None,
    )
    for metric in ("format_ok", "parse_failure", "length_stop", "unfinished_thinking", "propagated_error",
                   *(f"step_claim_{claim}" for claim in _CLAIMS)):
        result[f"{metric}_rate"] = (
            sum(row[f"{metric}_samples"] for row in included) / denominator if denominator else None
        )
    for metric in ("generated_tokens", "input_tokens", "total_tokens", "elapsed_seconds"):
        result[f"mean_{metric}"] = (
            math.fsum(row[f"observed_{metric}"] for row in included) / denominator
            if denominator else None
        )
    return result


def _comparison(
    reference: list[dict], target: list[dict], kind: str, bootstrap_samples: int, seed: int,
) -> dict:
    """Pair identical complete groups before differencing and resampling questions.

    The base/post overlap names mean reference/target roles, respectively, including
    non-base-stage checkpoint pairs and prompt contrasts. They describe at least
    one observed success among n samples, never a latent capability classification.
    """
    left = {row["problem_id"]: row for row in reference if row["included"]}
    right = {row["problem_id"]: row for row in target if row["included"]}
    matched = sorted(left.keys() & right.keys())
    groups: dict[str, list[float]] = defaultdict(list)
    overlap = dict.fromkeys(("both", "base_only", "post_only", "neither"), 0)
    for problem_id in matched:
        base, post = left[problem_id], right[problem_id]
        groups[base["group_id"]].append(post["pass_at_1"] - base["pass_at_1"])
        solved = (base["observed_solved"], post["observed_solved"])
        label = {(True, True): "both", (True, False): "base_only",
                 (False, True): "post_only", (False, False): "neither"}[solved]
        overlap[label] += 1
    planned_groups = {row["group_id"] for row in reference}
    return {
        "kind": kind, "reference_checkpoint": reference[0]["checkpoint"],
        "target_checkpoint": target[0]["checkpoint"], "reference_prompt": reference[0]["prompt"],
        "target_prompt": target[0]["prompt"], "budget": reference[0]["budget"],
        "prefix": reference[0]["prefix"], "difference_direction": "target_minus_reference",
        "difference_unit": "proportion", "matched_problems": len(matched),
        "matched_groups": len(groups), "excluded_groups": len(planned_groups) - len(groups),
        "reference_unmatched_problems": len(left) - len(matched),
        "target_unmatched_problems": len(right) - len(matched),
        "pass_at_1_difference": math.fsum(value for group in groups.values() for value in group)
        / len(matched) if matched else None,
        "pass_at_1_difference_ci95": _group_interval(list(groups.values()), bootstrap_samples, seed),
        "solved_overlap": overlap,
        "overlap_interpretation": "descriptive sampled; base=reference, post=target; at least one success",
    }


def analyze_runs(
    directories: list[Path], *, allow_incomplete: bool = False,
    bootstrap_samples: int = 1000, seed: int = 2026,
) -> dict:
    """Read compatible jobs and return a JSON-safe, audit-friendly bounded study report.

    Every represented checkpoint must complete the *full* embedded study schedule,
    including groups on unprovided shards, unless ``allow_incomplete=True``. Entire
    checkpoints may intentionally be absent. Within each checkpoint/prompt/budget,
    a group is included only if every problem (both repair variants included) has
    exactly spec.samples completed logical trials. Failed/cancelled retries never
    count as successful samples; completed parsing/format failures and length stops
    remain in the denominator. Incomplete rows retain counts but no estimates.

    Reject duplicate canonical paths, overlapping job identities, duplicate successful
    logical observations, mixed partition counts within a checkpoint, mixed study IDs,
    changed embedded science, changed numerical runtime profiles (except device
    ordinals), and changed model profiles within a checkpoint. Different checkpoint
    weights/tokenizers are expected; declared actual dtype/attention must still agree.
    Snapshot reading owns checksum, schedule, grade, and record-integrity validation.

    Cells are question-weighted; CIs resample independent group IDs with all questions
    retained. Checkpoint pairs follow plan order, target minus reference. Repair cue
    contrasts use actual check/neutral definitions within a format; math contrasts
    use native/completion definitions, not guessed prompt names. All estimates and
    differences use proportions, not percentages. See ``paired_bootstrap`` for
    integer bounds and the explicit percentile algorithm. No bootstrap arrays are
    included. This function only reads; output publication is a separate operation.
    """
    _bootstrap_options(bootstrap_samples, seed)
    if type(allow_incomplete) is not bool:
        raise ValueError("allow_incomplete must be a boolean.")
    if not isinstance(directories, list) or not directories:
        raise ValueError("directories must be a nonempty list of paths.")
    paths = [Path(directory).expanduser().resolve() for directory in directories]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate input paths, including aliases, are not allowed.")
    snapshots = [read_snapshot(path) for path in sorted(paths)]
    plan = snapshots[0].metadata.plan
    science = _science(plan)
    runtime = _profile(snapshots[0].metadata.runtime_profile)
    numerical = {key: snapshots[0].metadata.model_profile.get(key) for key in ("dtype", "attention")}
    partitions: dict[str, int] = {}
    shards: dict[str, set[int]] = defaultdict(set)
    models: dict[str, dict] = {}
    jobs, run_ids, attempts_seen, successful_ids, successful_trials = set(), set(), set(), set(), set()
    observations: dict[tuple, list[Attempt]] = defaultdict(list)
    for snapshot in snapshots:
        metadata, job = snapshot.metadata, snapshot.metadata.job
        if metadata.plan.study_id != plan.study_id:
            raise ValueError("Cannot mix study_id values in one analysis.")
        if _science(metadata.plan) != science:
            raise ValueError("Incompatible embedded specs/plans in study snapshots.")
        if _profile(metadata.runtime_profile) != runtime or {
            key: metadata.model_profile.get(key) for key in ("dtype", "attention")
        } != numerical:
            raise ValueError("Incompatible numerical runtime profiles; do not pool these jobs.")
        if partitions.setdefault(job.checkpoint, job.num_shards) != job.num_shards:
            raise ValueError("Mixed partitioning: num_shards differs within a checkpoint.")
        identity = (job.checkpoint, job.shard_index)
        if identity in jobs or metadata.run_id in run_ids:
            raise ValueError("Duplicate or overlapping job identities, even without completed trials.")
        jobs.add(identity)
        run_ids.add(metadata.run_id)
        shards[job.checkpoint].add(job.shard_index)
        profile = _profile(metadata.model_profile)
        if models.setdefault(job.checkpoint, profile) != profile:
            raise ValueError("Incompatible model profiles within a checkpoint.")
        for attempt in snapshot.records:
            trial = attempt.trial
            key = (job.checkpoint, trial.problem_id, trial.prompt, trial.budget)
            logical = (*key, trial.sample)
            if attempt.status == "completed":
                if logical in successful_trials or trial.trial_id in successful_ids:
                    raise ValueError("Duplicate successful logical trial/observation across jobs.")
                successful_trials.add(logical)
                successful_ids.add(trial.trial_id)
            if attempt.attempt_id in attempts_seen:
                raise ValueError("Duplicate attempt identity across snapshots.")
            attempts_seen.add(attempt.attempt_id)
            observations[key].append(attempt)
    checkpoints = [checkpoint.name for checkpoint in plan.spec.checkpoints if checkpoint.name in partitions]
    absent = [checkpoint.name for checkpoint in plan.spec.checkpoints if checkpoint.name not in partitions]
    rows: list[dict] = []
    cells_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for checkpoint, prompt, budget in product(checkpoints, plan.spec.prompts, plan.spec.budgets):
        groups: dict[str, list[dict]] = defaultdict(list)
        for problem in plan.problems:
            row = _problem_row(problem, observations[(checkpoint, problem.id, prompt.name, budget)],
                               plan.spec.samples, checkpoint, prompt.name, budget)
            groups[problem.group_id].append(row)
        for group_rows in groups.values():
            reasons = sorted({reason for row in group_rows for reason in row["missing_sample_reasons"]})
            for row in group_rows:
                row["included"] = not reasons
                row["exclusion_reasons"] = ["incomplete_group", *reasons] if reasons else []
                if reasons:
                    # Even a complete variant gets no estimate if its partner/group is missing.
                    row["pass_at_1"] = row["majority_correct"] = None
                rows.append(row)
                cells_by_key[(checkpoint, prompt.name, budget, row["prefix"])].append(row)
    completion = []
    for checkpoint in checkpoints:
        selected = [row for row in rows if row["checkpoint"] == checkpoint]
        completion.append({
            "checkpoint": checkpoint, "num_shards": partitions[checkpoint],
            "shards_present": sorted(shards[checkpoint]),
            "shards_absent": sorted(set(range(partitions[checkpoint])) - shards[checkpoint]),
            "planned_trials": sum(row["planned_samples"] for row in selected),
            "completed_trials": sum(row["completed_samples"] for row in selected),
            "missing_trials": sum(row["missing_samples"] for row in selected),
        })
    incomplete = [item for item in completion if item["missing_trials"]]
    if incomplete and not allow_incomplete:
        detail = "; ".join(f"{item['checkpoint']}: {item['missing_trials']}/{item['planned_trials']} missing"
                           for item in incomplete)
        raise ValueError(f"Incomplete checkpoint schedule ({detail}). Use allow_incomplete=True "
                         "to report only available COMPLETE groups; partial samples are never estimated.")
    cells = [_cell(cell_rows, bootstrap_samples, seed) for cell_rows in cells_by_key.values()]
    comparisons = []
    prefixes = sorted({problem.prefix for problem in plan.problems})
    for (left, right), prompt, budget, prefix in product(
        combinations(checkpoints, 2), plan.spec.prompts, plan.spec.budgets, prefixes,
    ):
        comparisons.append(_comparison(cells_by_key[(left, prompt.name, budget, prefix)],
                                       cells_by_key[(right, prompt.name, budget, prefix)],
                                       "checkpoint", bootstrap_samples, seed))
    task = plan.problems[0].task
    for reference, target in product(plan.spec.prompts, repeat=2):
        kind = None
        if task == "repair" and reference.cue == "neutral" and target.cue == "check" and reference.format == target.format:
            kind = "check_minus_neutral"
        elif task == "math" and reference.format == "completion" and target.format == "native":
            kind = "native_minus_common"
        if kind is not None:
            for checkpoint, budget, prefix in product(checkpoints, plan.spec.budgets, prefixes):
                comparisons.append(_comparison(cells_by_key[(checkpoint, reference.name, budget, prefix)],
                                               cells_by_key[(checkpoint, target.name, budget, prefix)],
                                               kind, bootstrap_samples, seed))
    report = {
        "schema_version": 1, "study_id": plan.study_id, "study_name": plan.spec.name,
        "task": task, "split": plan.problems[0].split, "input_directories": [str(path) for path in sorted(paths)],
        "allow_incomplete": allow_incomplete, "incomplete": bool(incomplete),
        "checkpoints_present": checkpoints, "checkpoints_absent": absent, "completion": completion,
        "runtime_profile": runtime, "model_profiles": models,
        "prompts": [prompt.model_dump(mode="json") for prompt in plan.spec.prompts],
        "science": science,
        "method": {
            "estimate_unit": "proportion", "weighting": "equal questions; group-cluster bootstrap",
            "bootstrap_samples": bootstrap_samples, "seed": seed, "confidence_level": 0.95,
            "percentile": "sorted replicate means; linear interpolation at zero-based rank (B-1)*q",
            "group_rule": "all planned samples for every group member within checkpoint/prompt/budget",
            "majority_vote": "unique most frequent parsed answer; ties/no votes abstain, no strict >n/2 rule",
            "audit_costs": "observed_* costs cover completed attempts only, including excluded groups",
            "exclusion_counts": "problem counts by group-level reason; reasons can overlap",
            "checkpoint_direction": "later listed checkpoint minus earlier listed checkpoint in plan",
        },
        "cells": cells, "comparisons": comparisons, "problems": rows, "limitations": list(_LIMITATIONS),
    }
    # Round-trip removes tuples and refuses NaN/Infinity rather than emitting nonstandard JSON.
    return json.loads(json.dumps(report, allow_nan=False))


def _flatten(row: dict, prefix: str = "") -> dict:
    """Flatten nested metric names for CSV; encode small list fields as JSON strings."""
    flat = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = json.dumps(value, ensure_ascii=False) if isinstance(value, (tuple, list)) else value
    return flat


def _csv_bytes(rows: list[dict], empty_fields: list[str]) -> bytes:
    """Render portable CSV, quoting normally and neutralizing formula-like text cells."""
    flat = [_flatten(row) for row in rows]
    fields = list(dict.fromkeys(key for row in flat for key in row)) or empty_fields
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in flat:
        # Preserve numeric negatives; only text can masquerade as a spreadsheet formula.
        safe = {key: "'" + value if isinstance(value, str) and (
            value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n"))
        ) else value for key, value in row.items()}
        writer.writerow(safe)
    return stream.getvalue().encode("utf-8")


def _markdown(value: Any) -> str:
    """Escape untrusted text as one inert Markdown table cell, including raw HTML."""
    text = html.escape(str(value), quote=True).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\`*_|\[\]])", r"\\\1", text)


def write_report(report: dict, output: Path) -> None:
    """Publish report.json, cells.csv, comparisons.csv, problems.csv, and report.md.

    The output directory must not exist (including an empty directory or dangling
    symlink). Exclusive mkdir is the no-overwrite boundary; each file is then
    atomically published. A failed write may leave a partial *new* report directory
    for inspection, never modify an earlier report/run. JSON serialization and table
    rendering happen before creating the directory. CSVs flatten nested metric names,
    leave unavailable values blank, and guard formula-like strings; Markdown omits
    exact answer text and escapes labels. Files contain no bootstrap-replicate arrays.
    """
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError("Report output already exists; choose a NEW directory, never a run directory.")
    payloads = {"report.json": (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")}
    for table in ("cells", "comparisons", "problems"):
        payloads[f"{table}.csv"] = _csv_bytes(report[table], ["kind"] if table == "comparisons" else ["checkpoint"])
    lines = [f"# Study report: {_markdown(report['study_name'])}", "",
             "All accuracy estimates/differences are proportions. Unavailable estimates are None.",
             "Only COMPLETE groups enter estimates; see JSON/CSVs for all exclusion reasons and counts.",
             f"Incomplete represented schedules: {_markdown(report['incomplete'])}.",
             f"Checkpoints present: {_markdown(', '.join(report['checkpoints_present']))}.",
             f"Checkpoints absent: {_markdown(', '.join(report['checkpoints_absent']))}.", "",
             "| Checkpoint | Prompt | Budget | Prefix | Included / planned | Pass@1 | 95% CI |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["cells"]:
        values = [row["checkpoint"], row["prompt"], row["budget"], row["prefix"],
                  f"{row['included_problems']} / {row['planned_problems']}", row["pass_at_1"], row["pass_at_1_ci95"]]
        lines.append("| " + " | ".join(_markdown(value) for value in values) + " |")
    lines.extend(["", "## Paired comparisons", "",
                  "Target minus reference; overlap in CSV/JSON is descriptive sampled overlap only.", "",
                  "| Contrast | Reference | Target | Groups | Difference | 95% CI |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for row in report["comparisons"]:
        values = [f"{row['kind']} / {row['budget']} / {row['prefix']}",
                  f"{row['reference_checkpoint']} / {row['reference_prompt']}",
                  f"{row['target_checkpoint']} / {row['target_prompt']}", row["matched_groups"],
                  row["pass_at_1_difference"], row["pass_at_1_difference_ci95"]]
        lines.append("| " + " | ".join(_markdown(value) for value in values) + " |")
    lines.extend(["", "## Uncertainty and caveats", "", _markdown(report["method"]["percentile"]),
                  f"Bootstrap replicates: {report['method']['bootstrap_samples']}; seed: {report['method']['seed']}.",
                  "", *(f"- {_markdown(caveat)}" for caveat in report["limitations"]), ""])
    payloads["report.md"] = "\n".join(lines).encode("utf-8")
    try:
        # exist_ok=False also protects against a destination created after the first check.
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("Report output already exists; choose a NEW directory.") from error
    for name, payload in payloads.items():
        atomic_write(output / name, payload, immutable=True)