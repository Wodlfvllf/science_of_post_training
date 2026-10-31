"""Model-free locked science, explicit site protocols, and stable trial schedules."""

from __future__ import annotations

from pathlib import Path

import yaml

from .data import binding_case, load_dataset, make_case, representative_case
from .io import atomic_json, digest, identity, now, read_json, safe_path
from .schemas import Job, PilotError, Plan, Protocol, Selection, StudySpec

ROUTING_CONDITIONS = (
    "capture", "self_qk", "q_flip", "k_flip", "qk_cancel", "v_swap",
    "q_color_sham", "k_color_sham", "qk_color_sham", "route_permute",
    "value_permute", "uniform", "q_samepalette", "k_samepalette", "qk_samepalette",
    "wrong_anchor_q", "wrong_anchor_k",
)
WEIGHT_CONDITIONS = (
    "native", "swap", "weight_sham", "swap_clamp", "wrong_clamp",
    "sham_clamp", "route_only", "route_sham",
)


def source_digest() -> str:
    root = Path(__file__).resolve().parent
    return identity([(p.relative_to(root).as_posix(), digest(p.read_bytes()))
                     for p in sorted(root.rglob("*.py"))])


def _yaml_pairs(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result or key == "<<":
            raise PilotError("YAML keys must be unique strings; merges are forbidden.")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


def load_spec(path: Path) -> StudySpec:
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_pairs)
    path = safe_path(path)
    if path.stat().st_size > 1024**2:
        raise PilotError("Pilot YAML exceeds 1 MiB.")
    spec = StudySpec.model_validate(yaml.load(path.read_text(), Loader=Loader))
    raw = spec.model_dump(mode="python")
    raw["dataset"] = (path.parent / spec.dataset).resolve()
    return StudySpec.model_validate(raw)


def plan_identity(spec: StudySpec, data_sha: str, code_sha: str) -> str:
    return identity({"spec": spec.model_dump(mode="json", exclude={"dataset"}),
                     "dataset": data_sha, "source": code_sha})


def lock_plan(config: Path, output: Path) -> Plan:
    output = safe_path(output)
    if output.exists():
        raise PilotError("Plan output already exists; use it or choose a new path.")
    spec = load_spec(config)
    dataset = load_dataset(spec.dataset)
    if dataset["image_size"] != spec.inference.image_size:
        raise PilotError("Dataset and fixed single-tile inference sizes disagree.")
    data_sha, code_sha = identity(dataset), source_digest()
    plan = Plan(plan_id=plan_identity(spec, data_sha, code_sha), created_at=now(),
                source_sha256=code_sha, dataset_sha256=data_sha, spec=spec, dataset=dataset)
    atomic_json(output, plan.model_dump(mode="json"), immutable=True)
    return plan


def validate_plan(plan: Plan) -> None:
    if (identity(plan.dataset) != plan.dataset_sha256
            or plan_identity(plan.spec, plan.dataset_sha256, plan.source_sha256) != plan.plan_id):
        raise PilotError("Plan data or science identity is inconsistent.")


def read_plan(path: Path, *, check_source: bool = False) -> Plan:
    plan = Plan.model_validate(read_json(path))
    validate_plan(plan)
    if check_source and source_digest() != plan.source_sha256:
        raise PilotError("Pilot source changed; restore it or lock a new pilot plan.")
    return plan


def checkpoint_for(plan: Plan, name: str):
    for checkpoint in plan.spec.checkpoints:
        if checkpoint.name == name:
            return checkpoint
    raise PilotError("Selected checkpoint is absent from the locked plan.")


def job_identity(plan: Plan, job: Job) -> str:
    return identity({"plan_id": plan.plan_id, "job": job.model_dump(mode="json")})


def validate_job(plan: Plan, job: Job, protocol: Protocol | None = None) -> None:
    validate_plan(plan)
    checkpoint = checkpoint_for(plan, job.checkpoint)
    if job.source_checkpoint is not None:
        source = checkpoint_for(plan, job.source_checkpoint)
        if (source.family != checkpoint.family or source.backend != checkpoint.backend
                or {source.stage, checkpoint.stage} != {"mpo", "gspo"}):
            raise PilotError("Weight tests require a matched adjacent MPO/final pair.")
    gated = job.split == "confirmation" or (
        job.split == "pilot_validation" and job.phase != "baseline"
    )
    if gated and protocol is None:
        raise PilotError("Validation/confirmation requires an explicitly frozen site protocol.")
    if protocol is not None:
        validate_protocol(protocol)
        if protocol.plan_id != plan.plan_id:
            raise PilotError("Protocol belongs to a different pilot plan.")
        entry = next((s for s in protocol.selections if s.checkpoint == job.checkpoint), None)
        if entry is None:
            raise PilotError("Checkpoint is not enrolled in the frozen protocol.")
        if job.phase != "baseline" and (
            entry.sites != job.sites or entry.projections != job.projections
            or entry.include_norms != job.include_norms
            or job.phase == "weights" and entry.source_checkpoint != job.source_checkpoint
        ):
            raise PilotError("Selected intervention disagrees with the frozen protocol.")


def freeze_protocol(plan: Plan, selections: tuple[Selection, ...], output: Path,
                    note: str, *, routing_effect_screen=.15, weight_effect_screen=.10,
                    preservation_margin=.05) -> Protocol:
    if not note.strip() or not selections:
        raise PilotError("Record the discovery selection rationale before freezing a protocol.")
    names = [entry.checkpoint for entry in selections]
    if len(set(names)) != len(names):
        raise PilotError("Duplicate protocol checkpoint selections.")
    for entry in selections:
        checkpoint_for(plan, entry.checkpoint)
        validate_job(plan, Job(checkpoint=entry.checkpoint, phase="routing", split="discovery",
                               sites=entry.sites, projections=entry.projections,
                               include_norms=entry.include_norms))
        if entry.source_checkpoint is not None:
            validate_job(plan, Job(checkpoint=entry.checkpoint, phase="weights", split="discovery",
                                   sites=entry.sites, source_checkpoint=entry.source_checkpoint))
    content = {"plan_id": plan.plan_id, "selections": [s.model_dump(mode="json") for s in selections],
               "note": note, "primary_endpoint": "routing_stage_interaction",
               "primary_size": "fixture" if all(c.family == "fixture" for c in plan.spec.checkpoints)
               else "8b", "routing_effect_screen": routing_effect_screen,
               "weight_effect_screen": weight_effect_screen,
               "preservation_margin": preservation_margin,
               "multiplicity": "bonferroni_primary_sites"}
    protocol = Protocol(protocol_id="0" * 64, created_at=now(), **content)
    normalized = protocol.model_dump(mode="json", exclude={"created_at", "protocol_id", "schema_version"})
    protocol = protocol.model_copy(update={"protocol_id": identity(normalized)})
    atomic_json(output, protocol.model_dump(mode="json"), immutable=True)
    return protocol


def validate_protocol(protocol: Protocol) -> None:
    raw = protocol.model_dump(mode="json", exclude={"created_at", "protocol_id", "schema_version"})
    if identity(raw) != protocol.protocol_id:
        raise PilotError("Protocol identity is inconsistent.")


def read_protocol(path: Path) -> Protocol:
    protocol = Protocol.model_validate(read_json(path))
    validate_protocol(protocol)
    return protocol


def families_for(plan: Plan, job: Job):
    for family in plan.dataset["families"]:
        if family["split"] == job.split and (
            int(digest(family["family_id"].encode()), 16) % job.num_shards == job.shard_index
        ):
            yield family


def _trial(plan: Plan, job: Job, case: dict, condition: str, site, target: str) -> dict:
    key = {"plan": plan.plan_id, "checkpoint": job.checkpoint, "phase": job.phase,
           "case_id": case["case_id"], "condition": condition,
           "site": site.label if site is not None else None,
           "source": job.source_checkpoint, "projections": job.projections,
           "include_norms": job.include_norms, "generate": job.generate,
           "prompt_profile": job.prompt_profile, "save_captures": job.save_captures}
    operation = identity({"phase": job.phase, "projections": job.projections,
                          "include_norms": job.include_norms,
                          "prompt_profile": job.prompt_profile, "generate": job.generate,
                          "save_captures": job.save_captures})
    seed_condition = ("weight_sham" if condition in ("weight_sham", "sham_clamp") else condition)
    return {"trial_id": identity(key), "checkpoint": job.checkpoint,
            "family_id": case["family_id"], "case_id": case["case_id"],
            "split": case["split"], "task": case["task"], "phase": job.phase,
            "condition": condition, "site": key["site"], "gold": case["gold"],
            "edge": case["edge"], "direction": case["direction"],
            "palette": case["palette"], "anchor": case["anchor"], "target_node": case["target"],
            "prompt_profile": job.prompt_profile, "generation_audit": job.generate,
            "operation": operation, "projections": list(job.projections),
            "include_norms": job.include_norms,
            "target": target, "source_checkpoint": job.source_checkpoint,
            "seed": int(identity([plan.spec.seed, case["family_id"], seed_condition])[:15], 16)}


def schedule(plan: Plan, job: Job):
    """Family-major schedule; no correctness filtering or hidden best-site selection."""
    for family in families_for(plan, job):
        if job.phase == "baseline":
            for case in family["cases"]:
                yield family, case, _trial(plan, job, case, "native", None, case["gold"])
            continue
        recipient = representative_case(family)
        alternate = binding_case(family, recipient["edge"], -recipient["direction"],
                                 recipient["palette"])
        value_donor = binding_case(family, recipient["edge"], recipient["direction"],
                                   1 - recipient["palette"])
        for site in job.sites:
            if job.phase == "routing":
                for condition in ROUTING_CONDITIONS:
                    target = (alternate["gold"] if condition in (
                        "q_flip", "k_flip", "q_samepalette", "k_samepalette", "route_permute",
                        "value_permute", "uniform", "wrong_anchor_q"
                    ) else value_donor["gold"] if condition == "v_swap" else recipient["gold"])
                    yield family, recipient, _trial(plan, job, recipient, condition, site, target)
            else:
                for condition in WEIGHT_CONDITIONS:
                    yield family, recipient, _trial(plan, job, recipient, condition, site,
                                                     recipient["gold"])
                # Preserve readout/text controls at the same representative e/d/p factors.
                for task in ("color_readout", "edge_readout", "text_binding"):
                    case = make_case(family, edge=recipient["edge"],
                                     direction=recipient["direction"], palette=recipient["palette"],
                                     task=task)
                    for condition in ("native", "swap", "weight_sham"):
                        yield family, case, _trial(plan, job, case, condition, site, case["gold"])