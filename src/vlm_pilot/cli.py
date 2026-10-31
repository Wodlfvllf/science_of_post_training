"""Thin, model-free-on-import CLI; model downloads/execution are explicit actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .schemas import Job, PilotError, Selection, SiteSpec


def _sites(value: str) -> tuple[SiteSpec, ...]:
    try:
        if any(len(pair.split(":")) != 2 for pair in value.split(",")):
            raise ValueError
        sites = tuple(SiteSpec(layer=int(pair.split(":")[0]), kv_group=int(pair.split(":")[1]))
                      for pair in value.split(",") if pair)
        if not sites or len({s.label for s in sites}) != len(sites):
            raise ValueError
        return sites
    except (ValueError, IndexError) as error:
        raise argparse.ArgumentTypeError("Sites use zero-based layer:KV-group pairs, e.g. 8:0,16:1.") from error


def _parser():
    parser = argparse.ArgumentParser(prog="interp-vlm", description="Explicit Reading-or-Routing pilot.")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="Generate new auditable single-image source families.")
    data.add_argument("output", type=Path)
    data.add_argument("--seed", type=int, default=20261031)
    for split, count in (("engineering", 24), ("discovery", 40), ("pilot-validation", 60),
                         ("confirmation", 200)):
        data.add_argument(f"--{split}", type=int, default=count)
    data.add_argument("--contact-sheet", type=Path)
    preview = commands.add_parser("preview", help="Count declared science without models/network.")
    preview.add_argument("config", type=Path)
    lock = commands.add_parser("lock", help="Freeze data, full checkpoint pins, science and pilot source.")
    lock.add_argument("config", type=Path)
    lock.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze", help="Freeze discovered sites before validation/confirmation.")
    freeze.add_argument("plan", type=Path)
    for family in ("2b", "8b", "fixture"):
        freeze.add_argument(f"--sites-{family}", type=_sites)
    freeze.add_argument("--projections", default="q_proj,k_proj")
    freeze.add_argument("--include-norms", action="store_true")
    freeze.add_argument("--routing-effect-screen", type=float, default=.15)
    freeze.add_argument("--weight-effect-screen", type=float, default=.10)
    freeze.add_argument("--preservation-margin", type=float, default=.05)
    freeze.add_argument("--note", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    preflight = commands.add_parser("preflight", help="P0 native/capture/self parity; no research trials.")
    preflight.add_argument("plan", type=Path)
    preflight.add_argument("--checkpoint", required=True)
    preflight.add_argument("--sites", type=_sites)
    preflight.add_argument("--data-root", type=Path)
    preflight.add_argument("--device")
    preflight.add_argument("--allow-downloads", action="store_true")
    preflight.add_argument("--output", type=Path)
    audit = commands.add_parser("audit-weights", help="Verify actual ViT/projector equality; no model loads.")
    audit.add_argument("plan", type=Path)
    audit.add_argument("--before", required=True)
    audit.add_argument("--after", required=True)
    audit.add_argument("--components", default="vision_model,mlp1")
    audit.add_argument("--allow-downloads", action="store_true")
    audit.add_argument("--output", type=Path)
    run = commands.add_parser("run", help="Run/resume one explicit job; auto-preflight is mandatory.")
    run.add_argument("plan", type=Path)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--phase", choices=("baseline", "routing", "weights"), default="baseline")
    run.add_argument("--split", choices=("engineering", "discovery", "pilot_validation", "confirmation"),
                     default="discovery")
    run.add_argument("--sites", type=_sites, default=())
    run.add_argument("--source-checkpoint")
    run.add_argument("--projections", default="q_proj,k_proj")
    run.add_argument("--include-norms", action="store_true")
    run.add_argument("--protocol", type=Path)
    run.add_argument("--data-root", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device")
    run.add_argument("--allow-downloads", action="store_true")
    run.add_argument("--max-new-trials", type=int)
    run.add_argument("--max-seconds", type=float)
    run.add_argument("--generate", action="store_true")
    run.add_argument("--prompt-profile", choices=("common", "native"), default="common")
    run.add_argument("--save-captures", action="store_true")
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    inspect = commands.add_parser("inspect", help="Read verified completion; imports no Torch.")
    inspect.add_argument("run", type=Path)
    analyze = commands.add_parser("analyze", help="Family-paired reports; no model reruns.")
    analyze.add_argument("runs", nargs="+", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--allow-incomplete", action="store_true")
    analyze.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def dispatch(args):
    from .data import contact_sheet, generate_dataset, load_dataset
    from .io import atomic_json, identity
    from .plan import (
        checkpoint_for,
        freeze_protocol,
        load_spec,
        read_plan,
        read_protocol,
        schedule,
    )

    if args.command == "data":
        raw = generate_dataset(args.output, seed=args.seed, engineering=args.engineering,
                               discovery=args.discovery, pilot_validation=args.pilot_validation,
                               confirmation=args.confirmation)
        if args.contact_sheet:
            contact_sheet(args.output, args.contact_sheet)
        return {"families": len(raw["families"]), "split_counts": raw["split_counts"],
                "image_question_cases": len(raw["families"]) * 12,
                "dataset_sha256": identity(raw), "model_loaded": False}
    if args.command == "preview":
        spec = load_spec(args.config)
        raw = load_dataset(spec.dataset, verify_images=False)
        return {"checkpoints": [c.name for c in spec.checkpoints], "families": len(raw["families"]),
                "split_counts": raw["split_counts"],
                "baseline_trials": len(raw["families"]) * 12 * len(spec.checkpoints),
                "tokenization_checked": False, "model_loaded": False}
    if args.command == "lock":
        from .plan import lock_plan

        plan = lock_plan(args.config, args.output)
        return {"plan_id": plan.plan_id, "source_sha256": plan.source_sha256,
                "dataset_sha256": plan.dataset_sha256, "model_loaded": False}
    if args.command == "inspect":
        from .runner import inspect_run

        return inspect_run(args.run)
    if args.command == "analyze":
        from .analysis import analyze_runs

        result = analyze_runs(args.runs, args.output, allow_incomplete=args.allow_incomplete,
                              bootstrap_samples=args.bootstrap_samples)
        return {"plan_id": result["plan_id"], "complete": result["complete"],
                "cells": len(result["cells"]), "paired_stage_contrasts": len(result["stage_contrasts"]),
                "fixture_only": result["fixture_only"], "hypothesis_decision": "not_automated"}
    plan = read_plan(args.plan, check_source=not (args.command == "run" and args.max_new_trials == 0))
    if args.command == "audit-weights":
        from .audit import audit_components

        report = audit_components(checkpoint_for(plan, args.before), checkpoint_for(plan, args.after),
            allow_downloads=args.allow_downloads, components=tuple(args.components.split(",")))
        if args.output:
            atomic_json(args.output, report, immutable=True)
        return report
    if args.command == "freeze":
        entries = []
        for checkpoint in plan.spec.checkpoints:
            sites = getattr(args, "sites_" + checkpoint.family)
            if sites is None:
                continue
            opposite = "mpo" if checkpoint.stage == "gspo" else "gspo" if checkpoint.stage == "mpo" else None
            source = next((c.name for c in plan.spec.checkpoints if c.family == checkpoint.family
                           and c.stage == opposite), None)
            entries.append(Selection(checkpoint=checkpoint.name, sites=sites,
                           source_checkpoint=source, projections=tuple(args.projections.split(",")),
                           include_norms=args.include_norms))
        protocol = freeze_protocol(plan, tuple(entries), args.output, args.note,
            routing_effect_screen=args.routing_effect_screen,
            weight_effect_screen=args.weight_effect_screen, preservation_margin=args.preservation_margin)
        return {"protocol_id": protocol.protocol_id, "checkpoints": [e.checkpoint for e in entries],
                "model_loaded": False}
    if args.command == "preflight":
        from .attention import Site
        from .runtime import Engine

        root = args.data_root or plan.spec.dataset
        if identity(load_dataset(root)) != plan.dataset_sha256:
            raise PilotError("Dataset differs from the locked plan.")
        checkpoint = checkpoint_for(plan, args.checkpoint)
        engine = Engine(checkpoint, plan.spec.inference,
                        allow_downloads=args.allow_downloads, device=args.device)
        family = next((f for f in plan.dataset["families"] if f["split"] == "engineering"),
                      plan.dataset["families"][0])
        report = engine.preflight(root, family, family["cases"][0],
                                 tuple(Site(s.layer, s.kv_group) for s in (args.sites or ())))
        if args.output:
            atomic_json(args.output, report, immutable=True)
        return report
    from .runner import execute_plan

    job = Job(checkpoint=args.checkpoint, phase=args.phase, split=args.split, sites=args.sites,
              source_checkpoint=args.source_checkpoint, projections=tuple(args.projections.split(",")),
              include_norms=args.include_norms, generate=args.generate,
              prompt_profile=args.prompt_profile, save_captures=args.save_captures,
              num_shards=args.num_shards, shard_index=args.shard_index)
    protocol = read_protocol(args.protocol) if args.protocol else None
    # Enumerating verifies a nonempty schedule before model work.
    if not any(schedule(plan, job)):
        raise PilotError("The selected split/shard is empty.")
    return execute_plan(plan, job, args.output, dataset_root=args.data_root,
                        allow_downloads=args.allow_downloads, device=args.device, protocol=protocol,
                        max_new_trials=args.max_new_trials, max_seconds=args.max_seconds)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = dispatch(args)
    except KeyboardInterrupt:
        print("Interrupted; published records are resumable. Inspect the local manifest.", file=sys.stderr)
        return 130
    except PilotError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        print(f"{type(error).__name__}: pilot operation failed; check arguments, data and optional "
              "dependencies. Inspect saved attempts; no silent fallback.", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 1 if result.get("passed") is False else 0