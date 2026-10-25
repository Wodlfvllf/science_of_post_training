"""Thin explicit commands for dataset preparation, freezing, remote jobs, and reports.

Use ``interp posttrain --help``. Inputs/data/results stay local unless acquisition
or weight downloads are explicitly enabled. No credentials enter arguments or
saved configurations. SIGTERM requests a cooperative checkpointed stop and the
previous signal handler is restored on exit.
"""

from __future__ import annotations

import argparse
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def add_commands(commands: Any) -> None:
    """Register nested CLI syntax without importing model, dataset, or Hub libraries."""
    root = commands.add_parser("posttrain", help="Personal maths and error-repair studies.")
    sub = root.add_subparsers(dest="study_command", required=True)
    data = sub.add_parser("data", help="Explicitly prepare numerical problems as immutable JSONL.")
    data.add_argument("kind", choices=("gsm8k", "repair"))
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--count", type=int, default=50, help="Number of sources (repair emits two rows each).")
    data.add_argument("--split", choices=("development", "pilot", "test"), default="pilot")
    data.add_argument("--seed", type=int, default=2026)
    data.add_argument("--revision", default="main", help="GSM8K dataset ref, resolved to a full SHA.")
    data.add_argument("--allow-downloads", action="store_true")
    preview = sub.add_parser("preview", help="Validate data/config and count total work; no model or writes.")
    preview.add_argument("config", type=Path)
    lock = sub.add_parser("lock", help="Freeze data, recursive code identity, and model revisions; no weights.")
    lock.add_argument("config", type=Path)
    lock.add_argument("--output", type=Path, required=True)
    lock.add_argument("--allow-downloads", action="store_true", help="Resolve floating refs through Hub metadata.")
    run = sub.add_parser("run", help="Run/resume one checkpoint and group partition from a locked plan.")
    run.add_argument("plan", type=Path)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    _execution_arguments(run)
    resume = sub.add_parser("resume", help="Resume from archived data; original config/JSONL are unnecessary.")
    resume.add_argument("run_dir", type=Path)
    _execution_arguments(resume)
    inspect = sub.add_parser("inspect", help="Verify one job's checksums, grades, and completeness.")
    inspect.add_argument("run_dir", type=Path)
    analyze = sub.add_parser("analyze", help="Join jobs and write paired metrics; no model loading.")
    analyze.add_argument("run_dirs", nargs="+", type=Path)
    analyze.add_argument("--output", type=Path, required=True, help="NEW directory for JSON/CSV/Markdown reports.")
    analyze.add_argument("--allow-incomplete", action="store_true", help="Report complete groups only, with exclusions.")
    analyze.add_argument("--bootstrap-samples", type=int, default=1000)
    analyze.add_argument("--seed", type=int, default=2026)
    preflight = sub.add_parser("preflight", help="Explicit short model check: lengths, cache parity, seeded replay.")
    preflight.add_argument("plan", type=Path)
    preflight.add_argument("--checkpoint", required=True)
    preflight.add_argument("--device", default="cuda:0")
    preflight.add_argument("--allow-downloads", action="store_true")


def _execution_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose operational limits only; scientific overrides require a newly locked plan."""
    parser.add_argument("--device", default="cuda:0", help="Explicit cpu or cuda:N; never auto-fallback.")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--max-new-trials", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--shutdown-margin-seconds", type=float, default=30.0)
    parser.add_argument("--flush-records", type=int, default=1, help="Unpublished samples at risk on a hard kill.")


@contextmanager
def _termination_guard() -> Iterator[Any]:
    """Latch SIGTERM within an explicit CLI invocation and restore caller-owned state."""
    stopped = False

    def request_stop(signum: int, frame: Any) -> None:
        """Do not write from a signal callback; let the bounded generator/runner stop."""
        nonlocal stopped
        stopped = True

    previous = signal.signal(signal.SIGTERM, request_stop)
    try:
        yield lambda: stopped
    finally:
        signal.signal(signal.SIGTERM, previous)


def dispatch(arguments: argparse.Namespace) -> dict[str, Any]:
    """Execute exactly the requested action through the shared project library API."""
    from .plan import lock_study, preview

    operation = arguments.study_command
    if operation == "data":
        from .data import gsm8k_problems, make_repair_problems, write_problems

        options = dict(count=arguments.count, split=arguments.split, seed=arguments.seed)
        if arguments.kind == "gsm8k":
            problems = gsm8k_problems(
                **options, revision=arguments.revision, allow_downloads=arguments.allow_downloads,
            )
        else:
            problems = make_repair_problems(**options)
        write_problems(arguments.output, problems)
        return {"output": str(arguments.output), "problems": len(problems),
                "groups": len({row.group_id for row in problems}), "split": arguments.split}
    if operation == "preview":
        return preview(arguments.config)
    if operation == "lock":
        plan = lock_study(arguments.config, arguments.output, allow_downloads=arguments.allow_downloads)
        return {"study_id": plan.study_id, "output": str(arguments.output),
                "checkpoints": [row.model_dump(mode="json") for row in plan.spec.checkpoints]}
    if operation == "analyze":
        from .metrics import analyze_runs, write_report

        report = analyze_runs(
            arguments.run_dirs, allow_incomplete=arguments.allow_incomplete,
            bootstrap_samples=arguments.bootstrap_samples, seed=arguments.seed,
        )
        write_report(report, arguments.output)
        return {"study_id": report["study_id"], "output": str(arguments.output),
                "incomplete": report["incomplete"], "cells": len(report["cells"]),
                "comparisons": len(report["comparisons"])}
    if operation == "preflight":
        from .preflight import preflight_checkpoint

        return preflight_checkpoint(
            arguments.plan, arguments.checkpoint, device=arguments.device,
            allow_downloads=arguments.allow_downloads,
        )
    from .runner import inspect_job, resume_study, run_study
    from .schemas import Job

    if operation == "inspect":
        return inspect_job(arguments.run_dir)
    options = {
        key: getattr(arguments, key) for key in (
            "device", "allow_downloads", "max_new_trials", "max_seconds",
            "shutdown_margin_seconds", "flush_records",
        )
    }
    with _termination_guard() as stop:
        if operation == "resume":
            return resume_study(arguments.run_dir, should_cancel=stop, **options)
        job = Job(checkpoint=arguments.checkpoint, shard_index=arguments.shard_index,
                  num_shards=arguments.num_shards)
        return run_study(arguments.plan, job, arguments.output_dir, should_cancel=stop, **options)