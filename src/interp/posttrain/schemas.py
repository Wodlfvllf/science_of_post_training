"""Frozen contracts for numerical maths rollouts and matched error-repair trials.

These records intentionally do not impersonate the old two-candidate scoring
recipe. A study plan embeds its entire dataset and immutable checkpoint revisions;
each independently scheduled GPU job has its own checksum-verified checkpoint.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, JsonValue, StrictBool, StrictInt, model_validator

from ..schemas import (
    Digest,
    GenerationRecord,
    GenerationSpec,
    Identifier,
    Record,
    Seed,
    Text,
    TokenId,
)

# Keep development, exploratory pilot, and final evaluation explicitly distinct.
StudySplit = Literal["development", "pilot", "test"]


class Problem(Record):
    """One unassisted question or one version of a matched arithmetic draft.

    ``answer`` is an exact numeric string, never a solution shown to the model.
    Related correct/error variants must share a source and group. Repair-specific
    fields describe a generated, independently checkable multiplication step.
    """

    # Stable observation, source, and independent resampling unit.
    id: Identifier
    source_id: Identifier
    group_id: Identifier
    split: StudySplit
    task: Literal["math", "repair"]
    # Preserve exact question and gold label separately from the optional draft.
    question: Text
    answer: Text
    provenance: Text
    prefix: Literal["none", "correct", "error"] = "none"
    draft: str | None = None
    # Generated repair problems are (a * b) + c; no arbitrary expression execution.
    operands: tuple[StrictInt, StrictInt, StrictInt] | None = None
    proposed_product: StrictInt | None = None

    @model_validator(mode="after")
    def check_task(self) -> Self:
        """Refuse ambiguous drafts and numerically inconsistent repair labels."""
        # Ordinary maths must not accidentally receive solution-bearing assistance.
        if self.task == "math":
            if self.prefix != "none" or any(
                item is not None for item in (self.draft, self.operands, self.proposed_product)
            ):
                raise ValueError("Math problems must not contain repair fields.")
        else:
            # Generated diagnostic labels are verified independently of the generator.
            if self.prefix == "none" or not self.draft or self.operands is None or self.proposed_product is None:
                raise ValueError("Repair problems require a labeled draft and operands.")
            a, b, c = self.operands
            if self.answer != str(a * b + c):
                raise ValueError("Repair gold answer disagrees with its operands.")
            if (self.proposed_product == a * b) != (self.prefix == "correct"):
                raise ValueError("Repair prefix label disagrees with its proposed product.")
            if self.draft != f"First calculate the items in the boxes: {a} * {b} = {self.proposed_product}.":
                raise ValueError("Repair draft must contain exactly the declared first step.")
        return self


class Checkpoint(Record):
    """One stage in a documented lineage; revisions become full SHAs when locked."""

    # Labels are descriptive; they do not prove that training algorithms were isolated.
    name: Identifier
    stage: Literal["base", "sft", "dpo", "rl"]
    backend: Literal["huggingface", "fixture"] = "huggingface"
    model_id: Text
    revision: Text = "main"
    tokenizer_id: Text | None = None
    tokenizer_revision: Text | None = None
    initialization_seed: Seed = 17

    @model_validator(mode="after")
    def check_fixture(self) -> Self:
        """Make artificial weights impossible to confuse with a named real checkpoint."""
        # Fixture identities must be conspicuous in plans and reports.
        if self.backend == "fixture" and (
            self.model_id != "offline-tiny-gpt2" or self.revision != "fixture-v1"
            or self.tokenizer_id is not None or self.tokenizer_revision is not None
        ):
            raise ValueError("Fixture checkpoints must use offline-tiny-gpt2 / fixture-v1.")
        return self


class Prompt(Record):
    """A predeclared prompt condition, with cue interventions separate from formatting.

    ``native`` means tokenizer chat formatting for post-trained stages, and the
    completion format for the base. It is not an assertion of optimal elicitation.
    Chat rendering must have an actual template; no template is guessed.
    """

    name: Identifier
    format: Literal["completion", "native"] = "completion"
    cue: Literal["none", "neutral", "check"] = "none"
    instruction: Text = "Solve the problem. Show your calculations. End with Final answer: <number>."
    # None means no explicit system message; the tokenizer may implement its own default.
    system: Text | None = None
    add_bos: StrictBool = False

    @model_validator(mode="after")
    def check_format(self) -> Self:
        """Prevent templates from duplicating BOS or silently dropping a system message."""
        if self.format == "native" and self.add_bos:
            raise ValueError("Native templates own special tokens; add_bos must be false.")
        if self.format == "completion" and self.system is not None:
            raise ValueError("Completion prompts do not accept a system message.")
        return self


class Inference(Record):
    """Scientific numerical controls, independent of the machine's CUDA ordinal."""

    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    attention: Literal["eager", "sdpa"] = "sdpa"
    max_context_tokens: Annotated[StrictInt, Field(ge=2)] = 16384


class StudySpec(Record):
    """One bounded dataset × checkpoints × prompts × token budgets × samples design."""

    schema_version: Literal[1] = 1
    name: Identifier
    hypothesis: Text
    dataset: Path
    # All configured checkpoints belong to the comparison, even when run separately.
    checkpoints: Annotated[tuple[Checkpoint, ...], Field(min_length=1)]
    prompts: Annotated[tuple[Prompt, ...], Field(min_length=1)]
    budgets: Annotated[tuple[Annotated[StrictInt, Field(ge=1)], ...], Field(min_length=1)]
    samples: Annotated[StrictInt, Field(ge=1, le=1024)] = 8
    seed: Seed = 2026
    # max_new_tokens is replaced by each explicitly enumerated budget, never truncated.
    generation: GenerationSpec = Field(default_factory=lambda: GenerationSpec(
        max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.95, use_cache=True
    ))
    inference: Inference = Field(default_factory=Inference)

    @model_validator(mode="after")
    def check_design(self) -> Self:
        """Reject accidental repeated cells, irrelevant settings, and fake replications."""
        for values in (
            [item.name for item in self.checkpoints], [item.name for item in self.prompts],
            list(self.budgets),
        ):
            if len(values) != len(set(values)):
                raise ValueError("Checkpoint names, prompt names, and budgets must be unique.")
        if not self.generation.enabled or self.generation.max_new_tokens != self.budgets[0]:
            raise ValueError("Enable generation and set max_new_tokens to the first budget.")
        if not self.generation.do_sample and self.samples != 1:
            raise ValueError("Repeated greedy outputs are not independent samples; use samples=1.")
        if max(self.budgets) >= self.inference.max_context_tokens:
            raise ValueError("Generation budgets must leave room for a nonempty prompt.")
        return self


class StudyPlan(Record):
    """Portable, model-free snapshot; loading it never follows the original data path."""

    schema_version: Literal[1] = 1
    study_id: Digest
    created_at: AwareDatetime
    source_sha256: Digest
    dataset_sha256: Digest
    spec: StudySpec
    problems: tuple[Problem, ...]

    @model_validator(mode="after")
    def check_revisions(self) -> Self:
        """Execution plans may not contain moving HF branches or implicit tokenizer refs."""
        for checkpoint in self.spec.checkpoints:
            if checkpoint.backend == "huggingface" and (
                re.fullmatch(r"[0-9a-f]{40}", checkpoint.revision) is None
                or checkpoint.tokenizer_id is None
                or re.fullmatch(r"[0-9a-f]{40}", checkpoint.tokenizer_revision or "") is None
            ):
                raise ValueError("Lock full model and tokenizer revisions before running.")
        return self


class Job(Record):
    """One checkpoint and one disjoint problem-group partition of a locked study."""

    checkpoint: Identifier
    shard_index: Annotated[StrictInt, Field(ge=0)] = 0
    num_shards: Annotated[StrictInt, Field(ge=1, le=1024)] = 1

    @model_validator(mode="after")
    def check_partition(self) -> Self:
        """Reject empty or overlapping-by-construction ordinal declarations."""
        if self.shard_index >= self.num_shards:
            raise ValueError("shard_index must be smaller than num_shards.")
        return self


class Trial(Record):
    """One rollout, with stable identity unaffected by scheduling or GPU partitioning."""

    trial_id: Digest
    problem_id: Identifier
    prompt: Identifier
    budget: Annotated[StrictInt, Field(ge=1)]
    sample: Annotated[StrictInt, Field(ge=0)]
    seed: Seed


class Grade(Record):
    """Exact numerical grading; no LLM judge, loose last-number extraction, or eval()."""

    answer: str | None
    correct: StrictBool
    format_ok: StrictBool
    extraction: Literal["final", "boxed", "hash", "missing", "invalid"]
    # A reasoning-only marker before closing </think> is not a delivered final answer.
    unfinished_thinking: StrictBool = False
    # This is an observable equation claim, not proof of an internal error detector.
    step_claim: Literal["correct", "incorrect", "mixed", "not_observed"] = "not_observed"
    propagated_error: StrictBool = False


class Rollout(Record):
    """Auditable successful or partial model output, before attempt bookkeeping."""

    rendered_prompt: Text
    input_ids: Annotated[tuple[TokenId, ...], Field(min_length=1)]
    generation: GenerationRecord
    # Clean decoding is for inspection; grading retains raw text and thinking boundaries.
    answer_text: str
    # Native chat templates can open <think> before the first generated token.
    initial_thinking: StrictBool = False
    grade: Grade
    elapsed_seconds: Annotated[float, Field(ge=0)]


class Attempt(Record):
    """Retries retain distinct UUIDs; only complete rollouts satisfy a logical trial."""

    run_id: Digest
    trial: Trial
    attempt_id: UUID
    started_at: AwareDatetime
    finished_at: AwareDatetime
    status: Literal["completed", "failed", "cancelled"]
    result: Rollout | None = None
    error_type: str | None = None

    @model_validator(mode="after")
    def check_status(self) -> Self:
        """Never publish interrupted output as a successful bounded sample."""
        if self.finished_at < self.started_at:
            raise ValueError("Attempt timestamps are reversed.")
        if self.status == "completed" and (
            self.result is None or self.error_type is not None
            or self.result.generation.finish_reason == "cancelled"
        ):
            raise ValueError("Completed attempts require a complete rollout and no error.")
        if self.status == "failed" and not self.error_type:
            raise ValueError("Failed attempts require a safe error category.")
        return self


class StudyMetadata(Record):
    """A job's locked science plus actual loaded model/tokenizer/numerical fingerprints."""

    schema_version: Literal[1] = 1
    run_id: Digest
    created_at: AwareDatetime
    plan: StudyPlan
    job: Job
    model_profile: dict[str, JsonValue]
    runtime_profile: dict[str, JsonValue]