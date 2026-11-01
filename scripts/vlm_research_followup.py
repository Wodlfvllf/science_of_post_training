"""Explicit engineering diagnostics outside the locked science package.

No automatic model work on import. This does not emit locked ResultStore trials,
modify the source/config/data/plan, or select confirmation outcomes. Every result
stores the diagnostic source hash, native plan, runtime and exact scoring inputs.
Run each checkpoint in a fresh process to release all CUDA contexts on exit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def variants(family, case):
    """Diagnose selection before color handoff; keep original image and gold.

    New wordings are engineering development, not a regrading of the baseline.
    Explicit direction means outgoing for +1 and incoming for -1.
    """
    shapes = [n["shape"] for n in family["nodes"]]
    anchor, target = shapes[case["anchor"]], shapes[case["target"]]
    relation = (f"Follow the arrow pointing away from the {anchor}." if case["direction"] == 1
                else f"Follow the incoming arrow backward from the {anchor} to its other end.")
    questions = {
        "original_color": (case["question"], case["gold"], case["candidates"]),
        "destination_shape": (f"Starting at the {anchor}, follow one arrow "
                              f"{'forward' if case['direction'] == 1 else 'backward'}. "
                              "What shape is the destination? Reply with one shape name.", target, shapes),
        "explicit_shape": (relation + " Which shape is at the other end? Reply with one shape name.",
                           target, shapes),
        "explicit_color": (relation + " What color is the chip at that other end? Reply with one color name.",
                           case["gold"], case["candidates"]),
        "anchor_readout": (f"What color is the chip at the {anchor}? Reply with one color name.",
                           family["palettes"][case["palette"]][case["anchor"]], case["candidates"]),
        "destination_readout": (f"What color is the chip at the {target}? Reply with one color name.",
                                case["gold"], case["candidates"]),
    }
    return [dict(case, case_id=case["case_id"] + "-diagnostic-" + name, question=q,
                 gold=gold, candidates=list(candidates), diagnostic=name)
            for name, (q, gold, candidates) in questions.items()]


def classify_scores(result, candidates):
    """Report lowercase, title-case and distinct-event combined scores separately.

    Combining surface prefixes is a diagnostic endpoint, not full-answer scoring.
    Case aliases with identical token sequences are counted only once.
    """
    import math

    groups = {c: {} for c in candidates}
    for c in candidates:
        for alias in (c, c.title()):
            groups[c][tuple(result["candidate_token_ids"][alias])] = result["scores"][alias]
    pooled = {}
    for c, events in groups.items():
        values = list(events.values())
        m = max(values)
        pooled[c] = m + math.log(sum(math.exp(v - m) for v in values))

    def winner(scores):
        top = max(scores.values())
        names = [c for c, v in scores.items() if v == top]
        return names[0] if len(names) == 1 else None

    return {"lower": winner({c: result["scores"][c] for c in candidates}),
            "title": winner({c: result["scores"][c.title()] for c in candidates}),
            "combined": winner(pooled), "combined_scores": pooled}


def scored(engine, root, family, case):
    probe = dict(case, candidates=case["candidates"] + [c.title() for c in case["candidates"]])
    prepared = engine.prepare(root, family, probe, profile="common")
    result, _ = engine.score(prepared)
    grading = classify_scores(result, case["candidates"])
    return {"family_id": family["family_id"], "case": case, "score": result,
            "grading": grading,
            "correct": {k: grading[k] == case["gold"] for k in ("lower", "title", "combined")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "decompose", "projector"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 1 <= args.families <= 24 or args.output.exists():
        parser.error("Use 1–24 engineering families and a new output directory")
    if args.mode == "projector" and not args.source_checkpoint:
        parser.error("Projector replay needs the exact matched source checkpoint")

    import tempfile
    from collections import defaultdict
    from copy import deepcopy
    from statistics import mean

    import torch
    from filelock import FileLock

    from vlm_pilot.io import atomic_json, safe_path
    from vlm_pilot.plan import checkpoint_for, read_plan
    from vlm_pilot.runtime import Engine

    plan = read_plan(args.plan, check_source=True)
    checkpoint = checkpoint_for(plan, args.checkpoint)
    families = [f for f in plan.dataset["families"] if f["split"] == "engineering"][:args.families]
    if len(families) != args.families:
        raise ValueError("Insufficient engineering families")
    output = safe_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    scripts = (Path(__file__), Path(__file__).with_name("vlm_projector_diagnostic.py"))
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}
    lock = Path(tempfile.gettempdir()) / ("interp-vlm-" + args.device.replace(":", "-") + ".lock")
    with FileLock(str(lock), timeout=0):
        engine = Engine(checkpoint, plan.spec.inference, device=args.device, allow_downloads=False)
        root = plan.spec.dataset
        probe = deepcopy(families[0]["cases"][0])
        probe["candidates"] += [c.title() for c in probe["candidates"]]
        p0 = engine.preflight(root, families[0], probe)
        atomic_json(output / "preflight.json", p0, immutable=True)
        if not p0["passed"]:
            raise RuntimeError("Native/self/candidate diagnostic P0 failed")
        metadata = {"kind": "engineering_only_not_locked_trials", "mode": args.mode,
                    "plan": plan.model_dump(mode="json"), "model": engine.model_profile,
                    "runtime": engine.runtime_profile, "diagnostic_source": source_hashes,
                    "requested_families": args.families, "conditioning": "common",
                    "limitations": ["Development-only", "Alias score is a prefix-event diagnostic",
                                    "No optimizer-level or general-natural-image claim"]}
        atomic_json(output / "metadata.json", metadata, immutable=True)
        if args.mode == "qualify":
            from PIL import Image
            from torchvision import transforms as T
            from torchvision.transforms.functional import InterpolationMode

            checks = []
            for family in families:
                prepared = engine.prepare(root, family, family["cases"][0])
                item = next(i for i in family["images"] if i["image_id"] == prepared.case["image_id"])
                # Independent implementation of the published one-tile RGB/BICUBIC
                # ImageNet transform. This is not validation of dynamic tiling.
                transform = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                                       T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
                with Image.open(root / item["relative_path"]) as image:
                    reference = transform(image.convert("RGB"))[None].to(engine.device, engine.dtype)
                difference = float((reference - prepared.pixels).abs().max().item())
                checks.append({"family_id": family["family_id"], "max_abs": difference,
                               "bitexact": torch.equal(reference, prepared.pixels)})
            import importlib

            module = importlib.import_module(type(engine.model).__module__)
            template = module.get_conv_template(engine.model.template)
            separator = template.sep.strip()
            eos_id = engine.tokenizer.convert_tokens_to_ids(separator)
            report = {"pixel_checks": checks, "native_separator": separator,
                      "native_separator_id": eos_id, "tokenizer_eos": engine.tokenizer.eos_token_id,
                      "eos_match": eos_id == engine.tokenizer.eos_token_id,
                      "transform_definition": "single RGB tile torchvision BICUBIC ImageNet normalize"}
            atomic_json(output / "report.json", report, immutable=True)
            print(json.dumps(report))
            if any(not c["bitexact"] for c in checks) or not report["eos_match"]:
                raise RuntimeError("Independent pixels or native EOS differ")
            return

        conditions, source_meta, swaps = ["native"], None, []
        if args.mode == "projector":
            import importlib.util

            helper = Path(__file__).with_name("vlm_projector_diagnostic.py")
            module_spec = importlib.util.spec_from_file_location("projector_followup_helper", helper)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            ProjectorSwap, load_projector = module.ProjectorSwap, module.load_projector

            counterpart = checkpoint_for(plan, args.source_checkpoint)
            if counterpart.family != checkpoint.family or counterpart.stage == checkpoint.stage:
                raise ValueError("Use distinct matched architecture stage endpoints")
            source, source_meta = load_projector(counterpart)
            native = {"mlp1." + n: p.detach().cpu().clone() for n, p in engine.model.mlp1.named_parameters()}
            conditions += ["self", "endpoint", "delta_sham"]
        totals = defaultdict(list)
        for condition in conditions:
            context = None
            if args.mode == "projector" and condition != "native":
                context = ProjectorSwap(engine.model, native if condition == "self" else source,
                                        sham=condition == "delta_sham", seed=1729)
                context.__enter__()
            try:
                # Projector changes require re-extraction and fresh Prepared objects;
                # never use old cached embeddings to evaluate a supposedly new model.
                engine._features.clear()
                engine._family = None
                for family in families:
                    rows = []
                    for case in family["cases"]:
                        selected = (variants(family, case) if args.mode == "decompose"
                                    and case["task"] == "binding" else [case])
                        for diagnostic in selected:
                            row = scored(engine, root, family, diagnostic)
                            row["condition"] = condition
                            rows.append(row)
                    atomic_json(output / (condition + "-" + family["family_id"] + ".json"),
                                {"condition": condition, "rows": rows}, immutable=True)
                    buckets = defaultdict(list)
                    for row in rows:
                        task = row["case"].get("diagnostic", row["case"]["task"])
                        buckets[task].append(row)
                    for task, values in buckets.items():
                        for grade in ("lower", "title", "combined"):
                            totals[(condition, task, grade)].append(mean(v["correct"][grade] for v in values))
                    print(json.dumps({"condition": condition, "family": family["family_id"], "rows": len(rows)}), flush=True)
            finally:
                if context is not None:
                    context.__exit__(None, None, None)
                    if not context.summary["restored_bitexact"]:
                        raise RuntimeError("Projector restoration failed")
                    swaps.append(deepcopy(context.summary))
                engine._features.clear()
                engine._family = None
        report = {"source_projector": source_meta, "swaps": swaps,
                  "cells": [{"condition": c, "task": t, "grading": g,
                             "families": len(v), "accuracy": mean(v), "family_values": v}
                            for (c, t, g), v in totals.items()]}
        atomic_json(output / "report.json", report, immutable=True)
        print(json.dumps({"report": str(output), "cells": report["cells"]}), flush=True)


if __name__ == "__main__":
    main()