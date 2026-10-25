"""Freeze a portable study before sending it to a remote GPU machine.

Preview is model/network-free. Locking floating checkpoint refs needs an explicit
network opt-in and downloads no weights. The archived plan embeds labels and exact
questions, so execution never rereads the original dataset or follows a moving ref.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from ..artifacts import ArtifactError, atomic_write
from ..config import _check_yaml_keys, source_digest
from ..data import canonical_bytes, digest_bytes
from ..runtime import trial_seed, utc_now
from .data import load_problems, validate_problems
from .schemas import Checkpoint, Job, Problem, StudyPlan, StudySpec, Trial


def load_spec(path: Path) -> StudySpec:
    """Read safe YAML and resolve only the dataset path, relative to this config."""
    # Reuse the existing duplicate/merge/cycle guards rather than a second YAML dialect.
    path = path.expanduser().resolve()
    content = path.read_text(encoding="utf-8")
    node = yaml.compose(content, Loader=yaml.SafeLoader)
    if not isinstance(node, yaml.MappingNode):
        raise ValueError("Study configuration must be a mapping.")
    _check_yaml_keys(node, set())
    raw = yaml.safe_load(content)
    spec = StudySpec.model_validate(raw)
    raw = spec.model_dump(mode="python")
    raw["dataset"] = (path.parent / spec.dataset.expanduser()).resolve()
    return StudySpec.model_validate(raw)


def _validate_design(spec: StudySpec, problems: tuple[Problem, ...]) -> None:
    """Require one task and one evaluation role per study to prevent silent pooling."""
    # Keep reports interpretable without hidden per-row task selection.
    validate_problems(problems)
    if len({row.task for row in problems}) != 1 or len({row.split for row in problems}) != 1:
        raise ValueError("One study must contain exactly one task and one split.")
    if problems[0].task == "math" and any(prompt.cue != "none" for prompt in spec.prompts):
        raise ValueError("Checking cues belong only to the assisted repair study.")


def preview(path: Path) -> dict[str, Any]:
    """Count the complete experiment before allocating models or writing anything."""
    # Surface worst-case generated tokens so the user can bound remote cost first.
    spec = load_spec(path)
    problems = load_problems(spec.dataset)
    _validate_design(spec, problems)
    trials = len(problems) * len(spec.checkpoints) * len(spec.prompts) * spec.samples
    return {
        "name": spec.name, "task": problems[0].task, "split": problems[0].split,
        "groups": len({row.group_id for row in problems}), "problems": len(problems),
        "checkpoints": [item.name for item in spec.checkpoints],
        "planned_trials": trials * len(spec.budgets),
        "max_generated_tokens": trials * sum(spec.budgets),
        "tokenization_checked": False, "weights_downloaded": False,
    }


def _resolve(checkpoint: Checkpoint, allow_downloads: bool) -> Checkpoint:
    """Pin model and tokenizer refs independently, without downloading their contents."""
    # Deterministic local fixture construction has no remote identity to resolve.
    if checkpoint.backend == "fixture":
        return checkpoint
    cache: dict[tuple[str, str], str] = {}

    def revision(repo: str, ref: str) -> str:
        """Resolve a floating branch only after explicit metadata-network permission."""
        if re.fullmatch(r"[0-9a-f]{40}", ref):
            return ref
        if not allow_downloads:
            raise ValueError("Floating refs need lock --allow-downloads; no weights are fetched.")
        if (repo, ref) not in cache:
            from huggingface_hub import HfApi

            sha = HfApi().model_info(repo, revision=ref).sha
            if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
                raise ValueError("Model ref did not resolve to a full commit SHA.")
            cache[repo, ref] = sha
        return cache[repo, ref]

    # Default tokenizer source follows the model, but an explicit source gets its own pin.
    model_sha = revision(checkpoint.model_id, checkpoint.revision)
    tokenizer_repo = checkpoint.tokenizer_id or checkpoint.model_id
    tokenizer_ref = checkpoint.tokenizer_revision or checkpoint.revision
    raw = checkpoint.model_dump(mode="python")
    raw.update(revision=model_sha, tokenizer_id=tokenizer_repo,
               tokenizer_revision=revision(tokenizer_repo, tokenizer_ref))
    return Checkpoint.model_validate(raw)


def study_identity(spec: StudySpec, dataset_sha: str, code_sha: str) -> str:
    """Bind actual data/code and all science, excluding only the original local path."""
    science = spec.model_dump(mode="json", exclude={"dataset"})
    return digest_bytes(canonical_bytes({"spec": science, "data": dataset_sha, "code": code_sha}))


def lock_study(config: Path, output: Path, *, allow_downloads: bool = False) -> StudyPlan:
    """Create an immutable execution plan; refuse overwrite and snapshot code recursively.

    Floating refs may be resolved via metadata requests only. Commit the code and
    copy this plan to every worker before execution. A new code/data/prompt/budget
    choice needs a new plan, rather than quietly mixing incompatible observations.
    """
    # Refuse accidental replacement before making even a metadata network request.
    if output.exists() or output.is_symlink():
        raise ArtifactError("Plan output already exists; use it or choose a new path.")
    spec = load_spec(config)
    problems = load_problems(spec.dataset)
    _validate_design(spec, problems)
    raw = spec.model_dump(mode="python")
    raw["checkpoints"] = [
        _resolve(item, allow_downloads).model_dump(mode="python") for item in spec.checkpoints
    ]
    spec = StudySpec.model_validate(raw)
    dataset_sha = digest_bytes(canonical_bytes([row.model_dump(mode="json") for row in problems]))
    code_sha = source_digest()
    plan = StudyPlan(
        study_id=study_identity(spec, dataset_sha, code_sha), created_at=utc_now(),
        source_sha256=code_sha, dataset_sha256=dataset_sha, spec=spec, problems=problems,
    )
    payload = canonical_bytes(plan.model_dump(mode="json"))
    if len(payload) > 64 * 1024 * 1024:
        raise ValueError("Plan exceeds the supported 64 MiB snapshot size.")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(output, payload, immutable=True)
    return plan


def validate_plan(plan: StudyPlan) -> None:
    """Recompute historical science without requiring current code or source data."""
    _validate_design(plan.spec, plan.problems)
    sha = digest_bytes(canonical_bytes([row.model_dump(mode="json") for row in plan.problems]))
    if sha != plan.dataset_sha256 or study_identity(plan.spec, sha, plan.source_sha256) != plan.study_id:
        raise ArtifactError("Study plan data or scientific identity is inconsistent.")


def read_plan(path: Path, *, check_source: bool = False) -> StudyPlan:
    """Read a bounded portable snapshot; execution, unlike inspection, pins current code."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ArtifactError("Plan must be a regular file no larger than 64 MiB.")
    plan = StudyPlan.model_validate_json(path.read_bytes())
    validate_plan(plan)
    if check_source and source_digest() != plan.source_sha256:
        raise ArtifactError("Plan code changed; restore the original code or lock a new study.")
    return plan


def checkpoint_for(plan: StudyPlan, job: Job) -> Checkpoint:
    """Resolve an explicitly selected checkpoint name without fuzzy matching."""
    for checkpoint in plan.spec.checkpoints:
        if checkpoint.name == job.checkpoint:
            return checkpoint
    raise ValueError("Checkpoint name is absent from the locked plan.")


def job_identity(plan: StudyPlan, job: Job) -> str:
    """Identify one single-writer partition; individual trial IDs ignore partitioning."""
    checkpoint_for(plan, job)
    return digest_bytes(canonical_bytes([plan.study_id, job.model_dump(mode="json")]))


def belongs(problem: Problem, job: Job) -> bool:
    """Keep correlated sources and both draft variants on one deterministic worker."""
    return int(digest_bytes(problem.group_id.encode("utf-8")), 16) % job.num_shards == job.shard_index


def schedule(plan: StudyPlan, job: Job) -> Iterator[Trial]:
    """Yield stable problem-major observations; a retry repeats its own isolated seed.

    Seeds match within a source/sample across checkpoints/prompts/variants/budgets.
    This is a matched-randomness design, not independence across those conditions.
    Each budget runs independently; no shorter result is silently sliced from a long run.
    """
    checkpoint_for(plan, job)
    for problem in plan.problems:
        if not belongs(problem, job):
            continue
        for prompt in plan.spec.prompts:
            for budget in plan.spec.budgets:
                for sample in range(plan.spec.samples):
                    identity = [plan.study_id, job.checkpoint, problem.id, prompt.name, budget, sample]
                    yield Trial(
                        trial_id=digest_bytes(canonical_bytes(identity)), problem_id=problem.id,
                        prompt=prompt.name, budget=budget, sample=sample,
                        seed=trial_seed(plan.spec.seed, problem.source_id, sample),
                    )