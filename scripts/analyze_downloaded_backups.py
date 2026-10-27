"""Offline, CPU-only audit of full posttraining backups; never execute archive code.

Uses the installed/local snapshot validator and metrics, not archived modules.
Inputs are read-only. Every output is exclusively created under a fresh directory.
Timeline snapshots are repeated publication boundaries, never pooled observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import stat
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath

from interp.config import source_digest
from interp.data import canonical_bytes
from interp.posttrain.metrics import analyze_runs, write_report
from interp.posttrain.plan import read_plan, schedule
from interp.posttrain.storage import read_snapshot


def digest(payload):
    """Hash exact bytes, without executing or interpreting source."""
    return hashlib.sha256(payload).hexdigest()


def file_digest(path):
    """Stream receipt checks without loading an archive into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Fail on overwrite and reject nonstandard floating point JSON."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    """Keep machine-readable audits small and spreadsheet-safe."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            safe = {}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    value = "'" + value
                safe[key] = value
            writer.writerow(safe)


def safe_members(archive):
    """Reject traversal, absolute/Windows paths, duplicates, links and special files.

    Limits are deliberately generous for these small data-only backups, not a
    general archive utility. All names are checked, even names never extracted.
    """
    members = {}
    total = 0
    for item in archive.infolist():
        name = item.filename
        parts = name.rstrip("/").split("/")
        mode = item.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if (not name or PurePosixPath(name).is_absolute() or "\\" in name
                or any(part in ("", ".", "..") or ":" in part for part in parts)
                or name in members or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
                or item.flag_bits & 1):
            raise ValueError(f"Unsafe, duplicate, special or encrypted ZIP entry: {name!r}")
        if item.file_size > 64 * 1024**2:
            raise ValueError("ZIP member exceeds 64 MiB")
        total += item.file_size
        members[name] = item
    if total > 2 * 1024**3 or len(members) > 100_000:
        raise ValueError("ZIP exceeds extraction bounds")
    return members


def extract_selected(archive, names, destination):
    """Extract an explicit data allowlist to a NEW root using exclusive file writes."""
    members = safe_members(archive)
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    for name in sorted(names):
        if name not in members or members[name].is_dir():
            raise ValueError("Selected artifact is absent or not a file")
        path = destination.joinpath(*PurePosixPath(name).parts)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Extraction escapes its root")
        path.parent.mkdir(parents=True, exist_ok=True)
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("Linked extraction path")
        with path.open("xb") as stream:
            stream.write(archive.read(name))


def inspect_archive(path):
    """Verify CRCs, manifest hashes/counts and backup boundaries without extraction."""
    with zipfile.ZipFile(path) as archive:
        members = safe_members(archive)
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failed: {bad}")
        backup = json.loads(archive.read("BACKUP.json"))
        source_entries = [(name.removeprefix("src/interp/"), digest(archive.read(name)))
                          for name in sorted(members)
                          if name.startswith("src/interp/") and name.endswith(".py")]
        source_hash = digest(canonical_bytes(source_entries))
        runs, selected = [], {"BACKUP.json"}
        for name in sorted(members):
            if not re.fullmatch(r"runs/posttrain/[^/]+/manifest\.json", name):
                continue
            parent = str(PurePosixPath(name).parent)
            manifest = json.loads(archive.read(name))
            metadata_name = parent + "/metadata.json"
            metadata_bytes = archive.read(metadata_name)
            if digest(metadata_bytes) != manifest["metadata_sha256"]:
                raise ValueError("Metadata checksum mismatch")
            metadata = json.loads(metadata_bytes)
            if metadata["run_id"] != manifest["run_id"]:
                raise ValueError("Metadata/manifest identity mismatch")
            if source_hash != metadata["plan"]["source_sha256"]:
                raise ValueError("Archived source does not match the locked source")
            selected.update((name, metadata_name))
            records = []
            for index, ref in enumerate(manifest["shards"]):
                if ref["name"] != f"records-{index:06d}-{ref['sha256']}.jsonl":
                    raise ValueError("Invalid shard name/index")
                shard_name = parent + "/" + ref["name"]
                payload = archive.read(shard_name)
                lines = payload.splitlines()
                if (digest(payload) != ref["sha256"] or len(payload) != ref["size_bytes"]
                        or not payload.endswith(b"\n") or len(lines) != ref["record_count"]):
                    raise ValueError("Shard checksum/size/record count mismatch")
                records.extend(json.loads(line) for line in lines)
                selected.add(shard_name)
            completed = [row["trial"]["trial_id"] for row in records if row["status"] == "completed"]
            if (len(set(completed)) != len(completed)
                    or sorted(completed) != sorted(manifest["completed_trial_ids"])
                    or len({row["attempt_id"] for row in records}) != len(records)):
                raise ValueError("Duplicate attempts or manifest completion mismatch")
            run_name = PurePosixPath(parent).name
            if backup["completed_trials"].get(run_name) != len(completed):
                raise ValueError("BACKUP completion disagrees with manifest")
            runs.append({
                "directory": parent, "run_id": metadata["run_id"],
                "checkpoint": metadata["job"]["checkpoint"], "study_id": metadata["plan"]["study_id"],
                "metadata_sha256": manifest["metadata_sha256"], "source_sha256": source_hash,
                "manifest_updated_at": manifest["updated_at"], "shards": len(manifest["shards"]),
                "completed": len(completed), "statuses": dict(Counter(row["status"] for row in records)),
                "completed_trial_ids": sorted(completed),
                "generated_tokens": sum(len(row["result"]["generation"]["token_ids"])
                                        for row in records if row["result"]),
                "generation_seconds": sum(row["result"]["elapsed_seconds"] for row in records if row["result"]),
            })
        if set(backup["completed_trials"]) != {PurePosixPath(run["directory"]).name for run in runs}:
            raise ValueError("BACKUP run list differs from manifests")
        # Only inert data and published result reports are eligible, never source/config/scripts.
        selected.update(name for name in members if re.fullmatch(r"data/posttrain/[^/]+\.(json|jsonl)", name))
        selected.update(name for name in members if re.fullmatch(r"runs/posttrain/[^/]*report[^/]*/[^/]+\.(json|csv|md)", name))
        return {"captured_at": backup["captured_at"], "scope": backup["scope"],
                "entry_count": len(members), "source_sha256": source_hash, "runs": runs}, selected


def describe(rows):
    """Summarize observed completed rollouts, keeping every denominator explicit."""
    complete = [row for row in rows if row.status == "completed"]
    results = [row.result for row in complete]
    durations = [row.elapsed_seconds for row in results]
    total = sum(durations)
    tokens = sum(len(row.generation.token_ids) for row in results)
    return {
        "attempts": len(rows), "statuses": dict(Counter(row.status for row in rows)),
        "completed": len(complete), "correct": sum(row.grade.correct for row in results),
        "parsed": sum(row.grade.answer is not None for row in results),
        "extraction": dict(Counter(row.grade.extraction for row in results)),
        "format_ok": sum(row.grade.format_ok for row in results),
        "unfinished_thinking": sum(row.grade.unfinished_thinking for row in results),
        "finish_reasons": dict(Counter(row.generation.finish_reason for row in results)),
        "input_tokens": sum(len(row.input_ids) for row in results), "generated_tokens": tokens,
        "generation_seconds": total, "tokens_per_generation_second": tokens / total if total else None,
        "trials_per_generation_hour": len(complete) * 3600 / total if total else None,
        "seconds_per_trial_mean": statistics.mean(durations) if durations else None,
        "seconds_per_trial_median": statistics.median(durations) if durations else None,
        "seconds_per_trial_min": min(durations, default=None),
        "seconds_per_trial_max": max(durations, default=None),
    }


def custom_audit(snapshots, report, output):
    """Add observed timing, schedule selection, paired sensitivity and text audit data."""
    audit = {"runs": [], "cells": [], "balanced_first_complete_problems": [],
             "prompt_identity": [], "raw_case_selection": "First chronological case in each declared category; not random."}
    trials, cases = [], []
    for saved in snapshots:
        metadata, plan = saved.metadata, saved.metadata.plan
        checkpoint = metadata.job.checkpoint
        completed = [row for row in saved.records if row.status == "completed"]
        problems = {row.id: row for row in plan.problems}
        positions = {row.id: i + 1 for i, row in enumerate(plan.problems)}
        ordered = list(schedule(plan, metadata.job))
        done = {row.trial.trial_id for row in completed}
        pending = [row for row in ordered if row.trial_id not in done]
        start = min(row.started_at for row in saved.records)
        end = max(row.finished_at for row in saved.records)
        wall = (end - start).total_seconds()
        per_problem = Counter(row.trial.problem_id for row in completed)
        per_problem_planned = Counter(row.problem_id for row in ordered)
        run = {"checkpoint": checkpoint, "run_id": metadata.run_id, **describe(list(saved.records)),
               "planned_trials": len(ordered), "missing_trials": len(pending),
               "first_attempt": start.isoformat(), "last_attempt": end.isoformat(), "wall_span_seconds": wall,
               "trials_per_wall_hour": len(completed) * 3600 / wall,
               "tokens_per_wall_second": sum(len(row.result.generation.token_ids) for row in completed) / wall,
               "attempt_seconds": sum((row.finished_at - row.started_at).total_seconds() for row in saved.records),
               "touched_problems": len(per_problem), "planned_problems": len(problems),
               "fully_complete_problems": sum(per_problem[key] == count for key, count in per_problem_planned.items()),
               "completed_schedule_prefix": [row.trial.trial_id for row in completed] == [row.trial_id for row in ordered[:len(completed)]],
               "next_trial": pending[0].model_dump(mode="json") if pending else None,
               "last_problem_position": max(positions[key] for key in per_problem),
               "observed_problem_positions": sorted(positions[key] for key in per_problem),
               "model_profile": metadata.model_profile, "runtime_profile": metadata.runtime_profile,
               "dataset_provenance": sorted({row.provenance for row in plan.problems}),
               "initial_thinking_count": sum(row.result.initial_thinking for row in completed)}
        audit["runs"].append(run)
        by_cell = defaultdict(list)
        for row in saved.records:
            by_cell[(row.trial.prompt, row.trial.budget)].append(row)
            if row.result is not None:
                result = row.result
                trials.append({"checkpoint": checkpoint, "problem_id": row.trial.problem_id,
                               "problem_position": positions[row.trial.problem_id],
                               "prompt": row.trial.prompt, "budget": row.trial.budget, "sample": row.trial.sample,
                               "trial_id": row.trial.trial_id, "attempt_id": str(row.attempt_id), "status": row.status,
                               "correct": result.grade.correct, "gold": problems[row.trial.problem_id].answer,
                               "answer": result.grade.answer, "extraction": result.grade.extraction,
                               "unfinished_thinking": result.grade.unfinished_thinking,
                               "finish_reason": result.generation.finish_reason,
                               "generated_tokens": len(result.generation.token_ids), "elapsed_seconds": result.elapsed_seconds,
                               "started_at": row.started_at.isoformat(), "finished_at": row.finished_at.isoformat()})
        for prompt in plan.spec.prompts:
            for budget in plan.spec.budgets:
                rows = by_cell[(prompt.name, budget)]
                row = {"checkpoint": checkpoint, "prompt": prompt.name, "budget": budget, **describe(rows)}
                row["planned"] = sum(t.prompt == prompt.name and t.budget == budget for t in ordered)
                row["missing"] = row["planned"] - row["completed"]
                # Conditional arithmetic, not a forecast interval: unseen questions can be slower.
                row["conditional_remaining_generation_hours_at_observed_cell_mean"] = (
                    row["missing"] * row["seconds_per_trial_mean"] / 3600 if row["completed"] else None)
                audit["cells"].append(row)
        # Identical base rendering/seeds can produce duplicate common/native outputs.
        lookup = {(a.trial.problem_id, a.trial.prompt, a.trial.budget, a.trial.sample): a for a in completed}
        pairs = [(a, lookup[(a.trial.problem_id, "native", a.trial.budget, a.trial.sample)]) for a in completed
                 if a.trial.prompt == "common" and (a.trial.problem_id, "native", a.trial.budget, a.trial.sample) in lookup]
        audit["prompt_identity"].append({"checkpoint": checkpoint, "paired_samples": len(pairs),
                                         "same_input_ids": sum(a.result.input_ids == b.result.input_ids for a, b in pairs),
                                         "same_output_tokens": sum(a.result.generation.token_ids == b.result.generation.token_ids for a, b in pairs),
                                         "same_seed": sum(a.trial.seed == b.trial.seed for a, b in pairs)})
        categories = {
            "correct": lambda a: a.result.grade.correct,
            "parsed_incorrect": lambda a: a.result.grade.answer is not None and not a.result.grade.correct,
            "invalid": lambda a: a.result.grade.extraction == "invalid",
            "missing_not_thinking": lambda a: a.result.grade.extraction == "missing" and not a.result.grade.unfinished_thinking,
            "unfinished_thinking": lambda a: a.result.grade.unfinished_thinking,
            "correct_length_stop": lambda a: a.result.grade.correct and a.result.generation.finish_reason == "length",
        }
        for category, predicate in categories.items():
            case = next((row for row in completed if predicate(row)), None)
            if case:
                cases.append({"checkpoint": checkpoint, "category": category,
                              "problem_position": positions[case.trial.problem_id],
                              "problem": problems[case.trial.problem_id].model_dump(mode="json"),
                              "attempt": case.model_dump(mode="json")})
    # A common, complete question intersection removes the obvious schedule-denominator difference.
    included = defaultdict(set)
    for row in report["problems"]:
        if row["included"]:
            included[(row["checkpoint"], row["prompt"], row["budget"])].add(row["problem_id"])
    common = set.intersection(*included.values()) if included else set()
    audit["balanced_problem_ids"] = sorted(common)
    for saved in snapshots:
        for prompt in saved.metadata.plan.spec.prompts:
            for budget in saved.metadata.plan.spec.budgets:
                rows = [a for a in saved.records if a.trial.problem_id in common and a.trial.prompt == prompt.name and a.trial.budget == budget]
                audit["balanced_first_complete_problems"].append({"checkpoint": saved.metadata.job.checkpoint,
                    "prompt": prompt.name, "budget": budget, "problems": len(common), **describe(rows)})
    write_csv(output / "trials.csv", trials)
    write_json(output / "representative-cases.json", cases)
    write_json(output / "audit.json", audit)
    return audit


def main():
    """Verify every receipt, extract only the latest valid snapshot, analyze once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("downloads", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    local_source_before = source_digest()
    integrity, timeline, selections = [], [], {}
    archives = sorted(args.downloads.glob("*.zip"))
    receipts = sorted(args.downloads.glob("*.sha256"))
    previous = {}
    for receipt in receipts:
        record = {"receipt": receipt.name, "valid": False}
        try:
            if receipt.is_symlink() or not receipt.is_file():
                raise ValueError("Receipt is not a regular file")
            match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\\r\n]+\.zip)\n?", receipt.read_text())
            if not match or receipt.name != match[2] + ".sha256":
                raise ValueError("Malformed receipt or mismatched filename")
            path = args.downloads / match[2]
            if path.is_symlink() or not path.is_file():
                raise ValueError("Missing or linked ZIP")
            actual = file_digest(path)
            record.update(archive=path.name, expected_sha256=match[1], actual_sha256=actual, size_bytes=path.stat().st_size)
            if actual != match[1]:
                raise ValueError("ZIP SHA256 receipt mismatch")
            boundary, selected = inspect_archive(path)
            for run in boundary["runs"]:
                ids = set(run.pop("completed_trial_ids"))
                prior = previous.get(run["run_id"])
                run["new_completions_since_previous_snapshot"] = len(ids - prior[0]) if prior else len(ids)
                if prior and (not prior[0] <= ids or run["metadata_sha256"] != prior[1]):
                    raise ValueError("Regressed completion or changed metadata")
                previous[run["run_id"]] = (ids, run["metadata_sha256"])
            record.update(valid=True, crc_verified=True, manifest_checksums_verified=True)
            timeline.append({"archive": path.name, **boundary})
            selections[path.name] = selected
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        integrity.append(record)
    for path in archives:
        if not (args.downloads / (path.name + ".sha256")).is_file():
            integrity.append({"archive": path.name, "valid": False, "error": "Missing receipt"})
    write_json(args.output / "integrity.json", integrity)
    timeline.sort(key=lambda row: datetime.fromisoformat(row["captured_at"]))
    write_json(args.output / "timeline.json", timeline)
    write_csv(args.output / "timeline.csv", [
        {"archive": item["archive"], "captured_at": item["captured_at"], **run}
        for item in timeline for run in item["runs"]])
    if not timeline:
        raise ValueError("No valid backups; see integrity.json")
    latest = timeline[-1]
    recovery = args.output / "recovered"
    with zipfile.ZipFile(args.downloads / latest["archive"]) as archive:
        extract_selected(archive, selections[latest["archive"]], recovery)
    directories = [recovery / run["directory"] for run in latest["runs"]]
    snapshots = [read_snapshot(path) for path in directories]
    plans = [read_plan(path) for path in (recovery / "data/posttrain").glob("*plan.json")]
    strict_error = None
    try:
        analyze_runs(directories)
    except ValueError as error:
        strict_error = str(error)
        if "Incomplete checkpoint schedule" not in strict_error:
            raise
    report = analyze_runs(directories, allow_incomplete=True, bootstrap_samples=2000, seed=2026)
    write_report(report, args.output / "standard-report")
    audit = custom_audit(snapshots, report, args.output)
    local_source_after = source_digest()
    if local_source_before != local_source_after:
        raise ValueError("Local package source changed during analysis")
    summary = {"archives": len(archives), "receipts": len(receipts),
               "valid_archives": len(timeline), "verification_failures": [row for row in integrity if not row["valid"]],
               "latest_archive": latest["archive"], "latest_captured_at": latest["captured_at"],
               "selected_files": len(selections[latest["archive"]]),
               "local_source_before": local_source_before, "local_source_after": local_source_after,
               "archived_source": latest["source_sha256"], "study_id": report["study_id"],
               "validated_snapshots": len(snapshots), "validated_standalone_plans": len(plans),
               "strict_analysis_rejection": strict_error,
               "checkpoints_present": report["checkpoints_present"], "checkpoints_absent": report["checkpoints_absent"],
               "completion": report["completion"], "model_modules_imported": [name for name in ("torch", "transformers") if name in sys.modules],
               "analysis_script_sha256": file_digest(Path(__file__)),
               "python": sys.version, "cells": audit["cells"]}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "cells"}, indent=2))


if __name__ == "__main__":
    main()