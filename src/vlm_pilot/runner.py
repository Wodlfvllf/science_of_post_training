"""Explicit single-writer pilot execution; failed trials persist and never count."""

from __future__ import annotations

import copy
import tempfile
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

from filelock import FileLock

from .data import binding_case, load_dataset, make_case
from .io import ResultStore, atomic_json, identity, now, safe_path
from .plan import checkpoint_for, families_for, job_identity, schedule, source_digest, validate_job
from .schemas import Job, PilotError, Plan, Protocol


def validate_records(store: ResultStore) -> dict[str, dict]:
    """Verify semantic schedule and likelihood-derived labels, not only file hashes."""
    plan = Plan.model_validate(store.metadata["plan"])
    job = Job.model_validate(store.metadata["job"])
    protocol_raw = store.metadata.get("protocol")
    protocol = Protocol.model_validate(protocol_raw) if protocol_raw is not None else None
    validate_job(plan, job, protocol)
    if job_identity(plan, job) != store.metadata["run_id"]:
        raise PilotError("Stored job does not match its run identity.")
    expected = {row["trial_id"]: row for _, _, row in schedule(plan, job)}
    for row in store.records:
        planned = expected.get(row["trial_id"])
        if planned is None or any(row.get(k) != v for k, v in planned.items()):
            raise PilotError("Saved trial disagrees with the locked schedule.")
        if row["status"] == "completed":
            import math

            family = next(f for f in plan.dataset["families"] if f["family_id"] == row["family_id"])
            canonical_case = make_case(family, edge=row["edge"], direction=row["direction"],
                                      palette=row["palette"], anchor=row["anchor"], task=row["task"])
            keys = set(canonical_case["candidates"])
            scores = row["scores"]
            if keys != set(scores) or keys != set(row["candidate_token_ids"]) or keys != set(
                row["candidate_token_logprobs"]):
                raise PilotError("Saved successful trial has invalid candidates.")
            maximum = max(scores.values())
            winners = [key for key, value in scores.items() if value == maximum]
            prediction = winners[0] if len(winners) == 1 else None
            if (row["prediction"] != prediction or row["tied"] != (prediction is None)
                    or row["target_hit"] != (prediction == row["target"])
                    or row["gold_hit"] != (prediction == row["gold"])):
                raise PilotError("Saved likelihoods disagree with their derived correctness.")
            for name, values in row["candidate_token_logprobs"].items():
                ids = row["candidate_token_ids"][name]
                if (not values or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v > 0
                                       for v in values)
                        or any(type(token) is not int or token < 0 for token in ids)
                        or sum(values) != scores[name] or len(values) != len(ids)):
                    raise PilotError("Saved candidate sequence scores are inconsistent.")
            if job.generate:
                generated = row.get("generation")
                if not generated or len(generated["token_ids"]) > plan.spec.max_new_tokens:
                    raise PilotError("Missing or invalid generation audit.")
                finished = generated["token_ids"][-1] == generated["settings"]["eos_token_id"]
                parsed = generated["text"].strip().lower().rstrip(".").strip()
                valid = finished and parsed in keys
                if (generated["valid"] != valid or generated["correct"] != (valid and parsed == row["gold"])
                        or generated["parsed_answer"] != (parsed if valid else None)
                        or generated["finish_reason"] != ("eos" if finished else "length")):
                    raise PilotError("Saved raw generation disagrees with grading/termination.")
    return expected


def inspect_run(root: Path) -> dict:
    store = ResultStore(root).read()
    expected = validate_records(store)
    failed = [row for row in store.records if row["status"] != "completed"]
    return {"run_id": store.metadata["run_id"], "job": store.metadata["job"],
            "planned": len(expected), "completed": len(store.completed),
            "missing": len(expected) - len(store.completed), "failed_attempts": len(failed),
            "complete": len(expected) == len(store.completed),
            "fixture_only": store.metadata["model"]["scientific_fixture"],
            "manifest_boundary": store.manifest["updated_at"],
            "remote_backup_verified": False}


class CachedSource:
    """At most 1 GiB of selected full projection tensors, never a second GPU model."""

    def __init__(self, source):
        self.source, self.cache, self.size, self.fingerprints = source, {}, 0, {}
        self.repo_id, self.revision = getattr(source, "repo_id", None), getattr(source, "revision", None)

    def read_tensor(self, name):
        from .weights import _checksum

        if name not in self.cache:
            tensor = self.source.read_tensor(name)
            checksum = _checksum(tensor)
            if name in self.fingerprints and self.fingerprints[name] != checksum:
                raise PilotError("Selected source tensor bytes changed during the run.")
            self.fingerprints[name] = checksum
            size = tensor.numel() * tensor.element_size()
            if self.size + size > 1024**3:
                self.cache.clear()
                self.size = 0
            self.cache[name] = tensor
            self.size += size
        return self.cache[name].clone()


class FamilyExecutor:
    """Reuse four image features and bounded donors within one independent family."""

    def __init__(self, engine, root, job, source=None):
        self.engine, self.root, self.job, self.source = engine, root, job, source
        self.family_id = None
        self.prepared, self.captures = {}, {}

    def prepare(self, family, case):
        if self.family_id != family["family_id"]:
            self.prepared.clear()
            self.captures.clear()
            self.family_id = family["family_id"]
        if case["case_id"] not in self.prepared:
            self.prepared[case["case_id"]] = self.engine.prepare(
                self.root, family, case, profile=self.job.prompt_profile)
        return self.prepared[case["case_id"]]

    def capture(self, family, case, site):
        key = case["case_id"], site
        if key not in self.captures:
            prepared = self.prepare(family, case)
            _, captures, _ = self.engine.forward(prepared, sites=(site,))
            self.captures[key] = captures[site]
        return self.captures[key]

    def routing(self, family, case, trial, site):
        from .attention import RouteAction

        engine, condition = self.engine, trial["condition"]
        prepared = self.prepare(family, case)
        e, d, p = case["edge"], case["direction"], case["palette"]
        base = self.capture(family, case, site)
        donors = {}

        def donor(role, edge, direction, palette, anchor=None):
            item = make_case(family, edge=edge, direction=direction, palette=palette, anchor=anchor)
            other = self.prepare(family, item)
            engine.require_alignment(prepared, other, allow_query_change=direction != d or anchor is not None)
            donors[role] = {"case_id": item["case_id"], "image_id": item["image_id"],
                            "gold": item["gold"], "input_ids": list(other.input_ids)}
            return self.capture(family, item, site)

        action = None
        if condition == "self_qk":
            action = RouteAction("qk", query=base, key=base)
        elif condition == "q_flip":
            action = RouteAction("q", query=donor("query", e, -d, 1-p))
        elif condition == "k_flip":
            action = RouteAction("k", key=donor("key", -e, d, 1-p))
        elif condition == "qk_cancel":
            action = RouteAction("qk", query=donor("query", e, -d, 1-p),
                                 key=donor("key", -e, d, 1-p))
        elif condition == "q_samepalette":
            action = RouteAction("q", query=donor("query", e, -d, p))
        elif condition == "k_samepalette":
            action = RouteAction("k", key=donor("key", -e, d, p))
        elif condition == "qk_samepalette":
            action = RouteAction("qk", query=donor("query", e, -d, p),
                                 key=donor("key", -e, d, p))
        elif condition in ("wrong_anchor_q", "wrong_anchor_k"):
            if condition == "wrong_anchor_q":
                action = RouteAction("q", query=donor("wrong_query", e, d, 1-p,
                                                      anchor=(case["anchor"] + e*d) % 3))
            else:
                action = RouteAction("k", key=donor("wrong_key", e, d, 1-p,
                                                    anchor=(case["anchor"] + e*d) % 3))
        elif condition == "v_swap":
            action = RouteAction("v", value=donor("value", e, d, 1-p))
        elif condition == "q_color_sham":
            action = RouteAction("q", query=donor("query", e, d, 1-p))
        elif condition == "k_color_sham":
            action = RouteAction("k", key=donor("key", e, d, 1-p))
        elif condition == "qk_color_sham":
            other = donor("query_key", e, d, 1-p)
            action = RouteAction("qk", query=other, key=other)
        elif condition in ("route_permute", "value_permute", "uniform"):
            action = RouteAction(condition, seed=trial["seed"])
        result, captures = engine.score(prepared, sites=(site,),
                                         actions={site: action} if action is not None else None)
        result["donors"] = donors
        result["donor_answer_copy"] = bool(donors) and result["prediction"] in {
            donor["gold"] for donor in donors.values()}
        result["recipient_payload_answer"] = result["prediction"] in family["palettes"][p]
        return result, captures

    def weights(self, family, case, trial, site):
        from .attention import RouteAction
        from .weights import WeightSwap, identical_input_replay

        engine, condition = self.engine, trial["condition"]
        prepared = self.prepare(family, case)
        sites = (site,) if prepared.image_columns else ()
        native = self.capture(family, case, site) if sites else None
        options = dict(projections=self.job.projections, include_norms=self.job.include_norms)
        artifacts, notes = {}, {}
        if condition == "native":
            result, artifacts = engine.score(prepared, sites=sites)
        elif condition == "route_sham":
            result, artifacts = engine.score(prepared, sites=sites,
                actions={site: RouteAction("route_permute", seed=trial["seed"])})
        elif condition == "route_only":
            swap = WeightSwap(engine.language_model, site, self.source, **options)
            with swap:
                _, routed, _ = engine.forward(prepared, sites=sites)
                source_route = routed[site]
            result, artifacts = engine.score(prepared, sites=sites,
                actions={site: RouteAction("distribution", distribution=source_route)})
            notes["weight_swap"] = copy.deepcopy(swap.summary)
        else:
            sham = condition in ("weight_sham", "sham_clamp")
            if native is not None and condition == "swap":
                notes["identical_input_replay"] = identical_input_replay(
                    engine.language_model, site, self.source, native, **options)
            reference = native
            if condition == "wrong_clamp":
                wrong = binding_case(family, case["edge"], -case["direction"], case["palette"])
                engine.require_alignment(prepared, self.prepare(family, wrong), allow_query_change=True)
                reference = self.capture(family, wrong, site)
            action = (RouteAction("distribution", distribution=reference)
                      if condition in ("swap_clamp", "wrong_clamp", "sham_clamp") else None)
            swap = WeightSwap(engine.language_model, site, self.source, sham=sham,
                               seed=trial["seed"], **options)
            with swap:
                result, artifacts = engine.score(prepared, sites=sites,
                    actions={site: action} if action is not None else None)
            notes["weight_swap"] = copy.deepcopy(swap.summary)
        result.update(notes)
        result["weight_direction"] = ("rollback" if engine.checkpoint.stage == "gspo" else "replay")
        result["clamp_scope"] = "conditional_image_distribution_only_recipient_mass_and_values"
        return result, artifacts

    def execute(self, family, case, trial):
        from .attention import Site

        if self.job.phase == "baseline":
            prepared = self.prepare(family, case)
            result, captures = self.engine.score(prepared)
            if self.job.generate:
                result["generation"] = self.engine.generate(prepared, self.budget)
            return result, captures
        layer, group = map(int, trial["site"].split(":"))
        site = Site(layer, group)
        if self.job.phase == "routing":
            return self.routing(family, case, trial, site)
        return self.weights(family, case, trial, site)


def execute_plan(plan: Plan, job: Job, output: Path, *, dataset_root: Path | None = None,
                 allow_downloads=False, device=None, protocol=None, max_new_trials=None,
                 max_seconds=None) -> dict:
    """Budget limits are operational; expected denominators remain the full schedule."""
    validate_job(plan, job, protocol)
    if max_new_trials != 0 and source_digest() != plan.source_sha256:
        raise PilotError("Pilot source changed; restore it or lock a new plan before execution.")
    if max_new_trials is not None and (type(max_new_trials) is not int or max_new_trials < 0):
        raise PilotError("max_new_trials must be a nonnegative integer.")
    if max_seconds is not None and not 0 < max_seconds < float("inf"):
        raise PilotError("max_seconds must be positive and finite.")
    output = safe_path(output)
    run_id = job_identity(plan, job)
    expected = list(schedule(plan, job))
    if (output / "metadata.json").exists():
        previous = ResultStore(output).read()
        if previous.metadata["run_id"] != run_id:
            raise PilotError("Output already contains a different checkpoint/job.")
    if max_new_trials == 0:
        return dict(inspect_run(output), model_loaded=False, new_attempts=0) if (
            output / "metadata.json").exists() else {
            "run_id": run_id, "planned": len(expected), "completed": 0, "missing": len(expected),
            "new_attempts": 0, "model_loaded": False}
    if (output / "manifest.json").exists():
        existing = ResultStore(output).read()
        validate_records(existing)
        if existing.metadata["run_id"] != run_id:
            raise PilotError("Output already contains a different checkpoint/job.")
        if len(existing.completed) == len(expected):
            return dict(inspect_run(output), new_attempts=0, model_loaded=False)
    root = safe_path(dataset_root or plan.spec.dataset)
    dataset = load_dataset(root)
    if identity(dataset) != plan.dataset_sha256:
        raise PilotError("Dataset relocation/content disagrees with the archived plan.")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with ExitStack() as locks:
        locks.enter_context(FileLock(str(output / ".writer.lock"), timeout=0))
        selected_device = device or plan.spec.inference.device
        if selected_device.startswith("cuda:"):
            gpu_lock = Path(tempfile.gettempdir()) / ("interp-vlm-" + selected_device.replace(":", "-") + ".lock")
            locks.enter_context(FileLock(str(gpu_lock), timeout=0))
        from .attention import Site
        from .runtime import Engine, FixtureModel, save_capture
        from .weights import MappingTensorSource, SafetensorSource

        engine = Engine(checkpoint_for(plan, job.checkpoint), plan.spec.inference,
                        allow_downloads=allow_downloads, device=device)
        source = None
        if job.source_checkpoint:
            other = checkpoint_for(plan, job.source_checkpoint)
            if other.backend == "fixture":
                counterpart = FixtureModel(other).eval()
                source = MappingTensorSource({"language_model." + name: tensor
                                              for name, tensor in counterpart.language_model.named_parameters()})
                del counterpart
            else:
                source = SafetensorSource(other.model_id, other.revision, allow_downloads)
            source = CachedSource(source)
            names = job.projections + (("q_norm", "k_norm") if job.include_norms else ())
            for selected_site in job.sites:
                for name in names:
                    source.read_tensor(f"language_model.model.layers.{selected_site.layer}."
                                       f"self_attn.{name}.weight")
        store = ResultStore(output)
        created = (ResultStore(output).read().metadata["created_at"]
                   if (output / "metadata.json").exists() else now())
        metadata = {"schema_version": 1, "run_id": run_id, "created_at": created,
                    "plan": plan.model_dump(mode="json"), "job": job.model_dump(mode="json"),
                    "protocol": protocol.model_dump(mode="json") if protocol else None,
                    "model": engine.model_profile, "runtime": engine.runtime_profile,
                    "weight_source_fingerprints": source.fingerprints if source else {}}
        store.initialize(metadata)
        validate_records(store)
        family = next(iter(families_for(plan, job)), None)
        if family is None:
            raise PilotError("This job has no source families; choose a nonempty split/shard.")
        preflight_case = next(c for c in family["cases"] if c["task"] == "binding")
        sites = tuple(Site(s.layer, s.kv_group) for s in job.sites)
        preflight = engine.preflight(root, family, preflight_case, sites)
        preflight_path = output / f"preflight-{uuid.uuid4().hex}.json"
        atomic_json(preflight_path, preflight, immutable=True)
        if not preflight["passed"]:
            raise PilotError("VLM preflight failed; inspect saved preflight measurements before running.")
        executor = FamilyExecutor(engine, root, job, source)
        executor.budget = plan.spec.max_new_tokens
        attempts = 0
        for family, case, trial in expected:
            if trial["trial_id"] in store.completed:
                continue
            if (max_new_trials is not None and attempts >= max_new_trials
                    or max_seconds is not None and time.perf_counter() - start >= max_seconds):
                break
            row = dict(trial, run_id=run_id, attempt_id=uuid.uuid4().hex, created_at=now())
            try:
                result, captures = executor.execute(family, case, trial)
                artifacts = save_capture(output, trial["trial_id"], captures) if job.save_captures else []
                row.update(result, status="completed", artifacts=artifacts,
                           target_hit=result["prediction"] == trial["target"],
                           gold_hit=result["prediction"] == trial["gold"])
            except KeyboardInterrupt:
                store.append(dict(row, status="cancelled", error_type="KeyboardInterrupt"))
                raise
            except Exception as error:
                safe_detail = str(error) if isinstance(error, PilotError) else None
                store.append(dict(row, status="failed", error_type=type(error).__name__,
                                  controlled_error=safe_detail))
                raise PilotError(f"Pilot trial failed ({type(error).__name__}) in {job.phase}/"
                                 f"{trial['condition']}; {safe_detail or 'inspect saved attempts'}. "
                                 "No success counted.") from error
            store.append(row)
            attempts += 1
        report = dict(inspect_run(output), new_attempts=attempts, model_loaded=True,
                      invocation_seconds=time.perf_counter() - start)
    return report