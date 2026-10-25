# Use this module to validate experiment inputs and saved records before execution.
# These contracts deliberately cover paired text experiments, not every future method.
"""Validated, serializable contracts for the first behavioral research workflow.

Unknown fields, non-finite numbers, invalid candidate boundaries, and inconsistent
success records are rejected. Records are frozen to discourage accidental mutation;
callers must also treat their nested dictionaries as immutable. Text is never stripped
or normalized: a space can change tokenization and therefore the experiment.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

# Restrict identifiers so they are safe labels, not filesystem paths.
Identifier = Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")]
# Require complete SHA-256 hashes for content identity.
Digest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
# Preserve exact text while rejecting empty strings.
Text = Annotated[StrictStr, Field(min_length=1)]
# Reject negative, boolean, or fractional token IDs.
TokenId = Annotated[StrictInt, Field(ge=0)]
# Limit attention masks to their actual binary meaning.
MaskValue = Annotated[StrictInt, Field(ge=0, le=1)]
# Keep seeds compatible with common 32-bit RNG interfaces.
Seed = Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]
# Split names are explicit even though this release does not fit probes.
Split = Literal["train", "validation", "test", "inspection"]


class Record(BaseModel):
    """Base for frozen records with strict field names and finite numeric values."""

    # Fail on typos and NaN/inf rather than silently changing a scientific setting.
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True, allow_inf_nan=False)


class Candidate(Record):
    """One exact continuation, including any intended leading whitespace."""

    # Name the answer independently of its token count.
    id: Identifier
    # Keep the continuation byte-for-byte as supplied by the dataset.
    text: Text


class Example(Record):
    """One condition of a paired prompt with two candidate continuations.

    ``source_id`` and ``group_id`` track related observations; ``pair_id`` joins the
    two conditions. ``label_provenance`` states where the target designation came
    from. Dataset-wide uniqueness, pairing, and split checks live in ``data.py``.
    """

    # Give each condition an immutable dataset-level identity.
    id: Identifier
    # Preserve the source observation that generated this variant.
    source_id: Identifier
    # Group correlated observations for future split and uncertainty calculations.
    group_id: Identifier
    # Join exactly one observation from each declared condition.
    pair_id: Identifier
    # Name the controlled experimental condition.
    condition: Identifier
    # Prevent derived variants from crossing evaluation splits.
    split: Split = "inspection"
    # Store the exact already-rendered text; chat rendering is not supported yet.
    text: Text
    # This first recipe compares exactly two candidate sequences.
    candidates: tuple[Candidate, Candidate]
    # Declare which candidate is the target before scoring anything.
    target_candidate: Identifier
    # Record the origin and meaning of the target designation.
    label_provenance: Text

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        """Reject duplicate candidates or a target absent from the candidate set."""
        # Collect candidate labels without altering their texts.
        labels = {candidate.id for candidate in self.candidates}
        # Require two genuinely distinct choices, not duplicated labels or strings.
        if len(labels) != 2 or self.candidates[0].text == self.candidates[1].text:
            # Stop before a meaningless zero-margin comparison can be run.
            raise ValueError("Candidates must have distinct IDs and distinct texts.")
        # Check that the supervised target actually names one of the choices.
        if self.target_candidate not in labels:
            # Surface a dataset error rather than guessing the target.
            raise ValueError("target_candidate must name one of the two candidates.")
        # Return the unchanged validated observation.
        return self


class DatasetSpec(Record):
    """A local JSONL dataset and an optional expected semantic content digest."""

    # Name the dataset separately from its location on a particular machine.
    id: Identifier
    # Resolve relative paths against the configuration directory, not shell cwd.
    path: Path
    # Allow the user to pin the validated content before an important run.
    expected_sha256: Digest | None = None


class ModelSpec(Record):
    """Explicit CPU/CUDA profile for a tiny fixture or a pinned HF GPT-2 model.

    Real models must use complete commit SHAs. Only GPT-2 is verified in this first
    release; remote code, pickle weights, quantization, MPS, and device maps are not
    enabled. The fixture is random and exists solely for engineering tests.
    """

    # Select an offline fixture or the explicitly requested pretrained-model path.
    backend: Literal["fixture", "huggingface"] = "fixture"
    # Use a clearly named non-research fixture by default.
    model_id: Text = "offline-tiny-gpt2"
    # Version the fixture recipe; HF execution instead requires a full commit SHA.
    revision: Text = "fixture-v1"
    # Default to the tokenizer from the same repository as the model.
    tokenizer_id: Text | None = None
    # Default to the model revision only when both repository IDs match.
    tokenizer_revision: Text | None = None
    # Seed random fixture weights separately from per-trial sampling.
    initialization_seed: Seed = 17
    # Never choose a GPU implicitly just because one happens to be available.
    device: Literal["cpu", "cuda"] = "cpu"
    # Make reduced precision an explicit experiment setting.
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    # Start with reference attention rather than an unverified optimized kernel.
    attention: Literal["eager"] = "eager"

    @model_validator(mode="after")
    def validate_backend(self) -> Self:
        """Reject ambiguous revisions and unsupported fixture/numerical profiles."""
        # Keep CPU execution in the verified reference precision.
        if self.device == "cpu" and self.dtype != "float32":
            # Do not silently emulate a GPU precision profile on the Mac.
            raise ValueError("The initial CPU profile supports float32 only.")
        # Apply stronger invariants to the fixed engineering fixture.
        if self.backend == "fixture":
            # Disallow names/settings that would falsely describe fixture weights.
            if (self.model_id, self.revision) != ("offline-tiny-gpt2", "fixture-v1"):
                # Make fixture identity unambiguous in every artifact.
                raise ValueError("Use offline-tiny-gpt2 at fixture-v1 for the fixture backend.")
            # Keep the fixture entirely offline and CPU-only.
            if self.device != "cpu" or self.tokenizer_id is not None or self.tokenizer_revision:
                # Require pretrained-model execution for other tokenizer/device choices.
                raise ValueError("The fixture uses its own tokenizer and the CPU profile.")
        # Validate immutable revisions for actual pretrained weights.
        else:
            # Require a complete lowercase git SHA rather than a mutable branch/tag.
            if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
                # Prevent resuming a run against silently updated model weights.
                raise ValueError("HF model revision must be a full 40-character commit SHA.")
            # A different tokenizer repository cannot inherit the model's commit.
            if self.tokenizer_id not in (None, self.model_id) and self.tokenizer_revision is None:
                # Ask for an explicit independent tokenizer identity.
                raise ValueError("A different tokenizer repository needs its own pinned revision.")
            # Validate a supplied tokenizer revision by the same immutable rule.
            if self.tokenizer_revision and re.fullmatch(r"[0-9a-f]{40}", self.tokenizer_revision) is None:
                # Refuse branch names for tokenizer provenance too.
                raise ValueError("HF tokenizer revision must be a full commit SHA.")
        # Return the unchanged model specification.
        return self


class PromptSpec(Record):
    """Tokenization choices for exact raw text, with no automatic truncation."""

    # Prepend precisely one tokenizer BOS token if requested.
    add_bos: StrictBool = True
    # Reject oversized sequences; never truncate a scientific input silently.
    max_sequence_tokens: Annotated[StrictInt, Field(ge=2)] = 256


class GenerationSpec(Record):
    """Optional, bounded single-sequence generation in addition to candidate scoring."""

    # Permit scoring-only experiments without generating unused text.
    enabled: StrictBool = True
    # Bound output length independently from the prompt length.
    max_new_tokens: Annotated[StrictInt, Field(ge=1)] = 8
    # Default to greedy decoding for reference reproducibility.
    do_sample: StrictBool = False
    # Apply temperature only when multinomial sampling is explicitly enabled.
    temperature: Annotated[StrictFloat, Field(gt=0)] = 1.0
    # Keep all probability mass unless the user specifies nucleus sampling.
    top_p: Annotated[StrictFloat, Field(gt=0, le=1)] = 1.0
    # Zero disables top-k filtering when sampling.
    top_k: Annotated[StrictInt, Field(ge=0)] = 0
    # Keep the old reference default uncached; research rollouts can explicitly opt in.
    use_cache: StrictBool = False

    @model_validator(mode="after")
    def validate_sampling(self) -> Self:
        """Fail on sampling-only settings that greedy decoding would otherwise ignore."""
        # Detect ineffective settings rather than logging and silently ignoring them.
        if not self.do_sample and (self.temperature, self.top_p, self.top_k) != (1.0, 1.0, 0):
            # Require the intended decoding strategy to be stated explicitly.
            raise ValueError("Non-neutral sampling settings require do_sample: true.")
        # Return the unchanged decoding specification.
        return self


class ExecutionSpec(Record):
    """Operational settings excluded from experimental identity.

    Time limits are cooperative, not process watchdogs. A running forward pass may
    exceed its budget; the runner checks between trials and generation steps.
    """

    # Choose a single-writer run directory; relative paths use the config directory.
    output_dir: Path = Path("../runs/smoke")
    # Cap fresh attempts in this invocation without changing logical trial IDs.
    max_new_trials: Annotated[StrictInt, Field(ge=1)] | None = None
    # Bound wall time for this invocation, including explicit model loading.
    max_seconds: Annotated[StrictFloat, Field(gt=0)] | None = None
    # Leave headroom for local finalization and a subsequent explicit upload.
    shutdown_margin_seconds: Annotated[StrictFloat, Field(ge=0)] = 5.0
    # Seal small result shards rather than uploading or writing one file per token.
    shard_records: Annotated[StrictInt, Field(ge=1, le=1000)] = 8
    # Bound each shard and reject a record larger than this limit.
    max_shard_bytes: Annotated[StrictInt, Field(ge=1024, le=16 * 1024 * 1024)] = 4 * 1024 * 1024

    @model_validator(mode="after")
    def validate_margin(self) -> Self:
        """Require enough time for useful work before the requested shutdown margin."""
        # Check a time limit only if one was supplied.
        if self.max_seconds is not None and self.shutdown_margin_seconds >= self.max_seconds:
            # Reject a budget that could never schedule a trial.
            raise ValueError("shutdown_margin_seconds must be smaller than max_seconds.")
        # Return the unchanged operational settings.
        return self


class RunSpec(Record):
    """One hypothesis, its alternative, and a predeclared paired behavioral test."""

    # Fail explicitly on unsupported saved/configuration schema versions.
    schema_version: Literal[1] = 1
    # Name the experiment for humans without using the name as a content identity.
    name: Identifier
    # State the prediction before observing model outputs.
    hypothesis: Text
    # Preserve the strongest simple competing explanation.
    alternative: Text
    # Pin the interpretation of the ordinary trial function used by this recipe.
    recipe_version: Literal["paired-continuations-v1"] = "paired-continuations-v1"
    # Locate and optionally pin the dataset.
    dataset: DatasetSpec
    # Declare exactly the two conditions required in every pair.
    conditions: tuple[Identifier, Identifier] = ("baseline", "changed")
    # Record controls as research notes rather than implying causal validity.
    controls: tuple[Text, ...] = ("paired source examples",)
    # Pin model, tokenizer, precision, and attention choices.
    model: ModelSpec = Field(default_factory=ModelSpec)
    # Declare BOS insertion and maximum accepted sequence length.
    prompt: PromptSpec = Field(default_factory=PromptSpec)
    # Fix generation settings independently of model repository defaults.
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    # Preserve the raw sequence log-probability margin as the primary metric.
    primary_metric: Literal["target_logprob_margin"] = "target_logprob_margin"
    # Give each repeat its own stable identity and isolated seed.
    replicates: Annotated[StrictInt, Field(ge=1)] = 1
    # Seed trial scheduling/sampling independently of fixture weight initialization.
    seed: Seed = 2026
    # Separate operational budgets/locations from scientific identity.
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)

    @model_validator(mode="after")
    def validate_conditions(self) -> Self:
        """Require distinct condition names so pairs cannot collapse to one label."""
        # Compare the declared labels without guessing baseline semantics.
        if self.conditions[0] == self.conditions[1]:
            # Reject an experiment with no distinguishable conditions.
            raise ValueError("The two conditions must be distinct.")
        # Return the unchanged validated run settings.
        return self


class CandidateScore(Record):
    """Auditable teacher-forced likelihood for one exact candidate token suffix.

    ``continuation_start`` is an unpadded sequence index. The logit at index
    ``continuation_start - 1`` predicts the first candidate token. No EOS likelihood
    is added unless EOS is actually part of the specified continuation.
    """

    # Join this result to the original candidate label.
    candidate_id: Identifier
    # Preserve the exact text that was tokenized in prompt context.
    text: Text
    # Record the complete unpadded teacher-forced model input.
    input_ids: tuple[TokenId, ...]
    # Preserve the actual attention mask, not an inferred token count.
    attention_mask: tuple[MaskValue, ...]
    # Require at least one prefix token to predict the continuation.
    continuation_start: Annotated[StrictInt, Field(ge=1)]
    # Keep one finite natural-log probability per continuation token.
    token_logprobs: tuple[StrictFloat, ...]
    # Save the unnormalized sequence likelihood, used by the primary margin.
    sum_logprob: StrictFloat
    # Also expose length-normalized likelihood without confusing it with the sum.
    mean_logprob: StrictFloat

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        """Verify token/mask boundaries and check aggregate scores against token scores."""
        # Derive the expected scored suffix length from the saved boundary.
        count = len(self.input_ids) - self.continuation_start
        # This recipe writes unpadded inputs; padded utility tests are separate.
        if len(self.attention_mask) != len(self.input_ids) or any(v != 1 for v in self.attention_mask):
            # Refuse a mask that cannot describe the saved unpadded sequence.
            raise ValueError("Saved candidate scores require aligned, unpadded attention masks.")
        # Require one score for every requested continuation token.
        if count < 1 or len(self.token_logprobs) != count:
            # Make causal-shift errors detectable at the artifact boundary.
            raise ValueError("Candidate boundary and token_logprobs do not align.")
        # Log probabilities cannot be positive except for negligible round-off.
        if any(value > 1e-6 for value in self.token_logprobs):
            # Reject raw logits mislabeled as probabilities.
            raise ValueError("token_logprobs must be natural-log probabilities.")
        # Recompute the sum in ordinary Python for independent validation.
        total = math.fsum(self.token_logprobs)
        # Check both persisted reductions against the per-token observations.
        if not math.isclose(self.sum_logprob, total, rel_tol=1e-6, abs_tol=1e-6) or not math.isclose(
            self.mean_logprob, total / count, rel_tol=1e-6, abs_tol=1e-6
        ):
            # Reject corrupted or incorrectly reduced aggregate likelihoods.
            raise ValueError("Aggregate likelihoods disagree with token_logprobs.")
        # Return the unchanged validated score.
        return self


class GenerationRecord(Record):
    """Raw generated tokens/text and the exact effective HF decoding configuration."""

    # Include a generated EOS token when present.
    token_ids: tuple[TokenId, ...]
    # Decode without stripping special tokens or cleaning whitespace.
    text: StrictStr
    # Distinguish ordinary bounded output from a cooperatively interrupted trial.
    finish_reason: Literal["eos", "length", "cancelled"]
    # Record defaults as well as explicitly supplied generation settings.
    effective_config: dict[str, JsonValue]


class TrialResult(Record):
    """Output of an ordinary paired-trial function, before runner bookkeeping."""

    # Preserve the exact prompt token IDs shared by candidate scoring and generation.
    input_ids: tuple[TokenId, ...]
    # Preserve the prompt attention mask used for generation.
    attention_mask: tuple[MaskValue, ...]
    # Retain both candidates even when only their difference is plotted.
    candidate_scores: tuple[CandidateScore, CandidateScore]
    # Save target summed log likelihood minus alternative summed log likelihood.
    target_logprob_margin: StrictFloat
    # Allow scoring-only runs to omit generation entirely.
    generation: GenerationRecord | None = None

    @model_validator(mode="after")
    def validate_prompt(self) -> Self:
        """Check that both candidates were scored after this exact prompt prefix."""
        # Require a nonempty prefix and aligned mask.
        if not self.input_ids or len(self.input_ids) != len(self.attention_mask):
            # Reject an unconditioned or misaligned trial output.
            raise ValueError("A trial needs nonempty, aligned prompt tokens and mask.")
        # Verify each candidate against the preserved prompt boundary.
        for score in self.candidate_scores:
            # Compare token identities, not just compatible tensor shapes.
            if score.continuation_start != len(self.input_ids) or score.input_ids[: score.continuation_start] != self.input_ids:
                # Stop an inconsistent result from becoming a completed trial.
                raise ValueError("Candidate scoring did not preserve the exact prompt prefix.")
        # Return the unchanged paired-trial result.
        return self


class TrialRecord(Record):
    """One execution attempt; only a valid completed attempt satisfies a logical trial."""

    # Pin the saved record format.
    schema_version: Literal[1] = 1
    # Join the attempt to its immutable experiment identity.
    run_id: Digest
    # Identify the logical observation independently of retries and execution order.
    trial_id: Digest
    # Distinguish reruns without duplicating logical completion.
    attempt_id: UUID
    # Preserve the original example identity.
    example_id: Identifier
    # Keep pair/group/source labels directly accessible for result inspection.
    pair_id: Identifier
    # Preserve the independent group label.
    group_id: Identifier
    # Preserve source lineage for future resampling analyses.
    source_id: Identifier
    # Preserve the experimental condition.
    condition: Identifier
    # Keep inspection and held-out evaluation outputs distinguishable.
    split: Split
    # Use zero-based replicate indices.
    replicate: Annotated[StrictInt, Field(ge=0)]
    # Preserve the isolated trial seed, shared within each paired replicate.
    seed: Seed
    # Distinguish success, error, and interruption explicitly.
    status: Literal["completed", "failed", "cancelled"]
    # Use timezone-aware timestamps for cross-machine provenance.
    started_at: AwareDatetime
    # Preserve when the execution attempt finished or was interrupted.
    finished_at: AwareDatetime
    # Keep raw prompt text even when model execution fails before tokenization.
    input_text: Text
    # Retain complete outputs or a bounded, interrupted generation when available.
    result: TrialResult | None = None
    # Store exception type only, avoiding accidental credentials in exception messages.
    error_type: StrictStr | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        """Require valid outputs for success and a coherent timestamp/status combination."""
        # Compare chronological execution timestamps.
        if self.finished_at < self.started_at:
            # Reject impossible bookkeeping rather than repairing it silently.
            raise ValueError("finished_at must not precede started_at.")
        # Successful attempts must include a complete, non-interrupted result.
        if self.status == "completed" and (
            self.result is None or self.error_type is not None or (
                self.result.generation is not None and self.result.generation.finish_reason == "cancelled"
            )
        ):
            # Prevent a partial generation or error from counting as completed work.
            raise ValueError("A completed trial needs complete outputs and no error.")
        # Failed attempts must say which exception class occurred.
        if self.status == "failed" and not self.error_type:
            # Preserve at least a safe error category for subsequent inspection.
            raise ValueError("A failed trial requires error_type.")
        # Return the unchanged validated execution attempt.
        return self


class RunMetadata(Record):
    """Resolved configuration, dataset snapshot, and execution provenance for one run."""

    # Version the resolved snapshot separately from transient user settings.
    schema_version: Literal[1] = 1
    # Bind science, validated examples, and package source into one content identity.
    run_id: Digest
    # Record initial creation without including this timestamp in scientific identity.
    created_at: AwareDatetime
    # Keep the resolved settings that created the run.
    spec: RunSpec
    # Pin the canonical validated dataset content, not machine-local file locations.
    dataset_sha256: Digest
    # Pin the installed package's actual Python source.
    source_sha256: Digest
    # Refuse incompatible numerical/library/hardware profiles on resume.
    runtime_profile: dict[str, JsonValue]
    # Save loaded model-config and tokenizer fingerprints and immutable revisions.
    model_profile: dict[str, JsonValue]
    # Restore independently of the original dataset path.
    examples: tuple[Example, ...]


class ShardRef(Record):
    """One immutable JSONL shard acknowledged by a local or remote manifest."""

    # Permit only the content-addressed filenames produced by the writer.
    name: Annotated[StrictStr, Field(pattern=r"^records-[0-9]{6}-[0-9a-f]{64}\.jsonl$")]
    # Check file content before accepting any completed trial from it.
    sha256: Digest
    # Detect truncation before parsing JSONL.
    size_bytes: Annotated[StrictInt, Field(ge=1)]
    # Check the number of persisted attempts against the parsed records.
    record_count: Annotated[StrictInt, Field(ge=1)]


class CheckpointManifest(Record):
    """The publication boundary; unreferenced files are not completed work."""

    # Refuse unknown checkpoint formats.
    schema_version: Literal[1] = 1
    # Bind all references to one scientific run.
    run_id: Digest
    # Verify the resolved configuration/dataset/provenance snapshot too.
    metadata_sha256: Digest
    # List only sealed immutable artifacts.
    shards: tuple[ShardRef, ...] = ()
    # Advance completion only when the corresponding records are published.
    completed_trial_ids: tuple[Digest, ...] = ()
    # Record when this publication boundary was created.
    updated_at: AwareDatetime


class SyncReceipt(Record):
    """Local acknowledgement of a specific remotely committed checkpoint, not a promise."""

    # Record the explicitly selected results repository, never an authentication token.
    repo_id: Text
    # Record the immutable run prefix used in that repository.
    prefix: Text
    # Pin restore to the acknowledged full remote commit.
    revision: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{40}$")]
    # Bind the receipt to one experiment.
    run_id: Digest
    # Identify the exact remotely published manifest bytes.
    manifest_sha256: Digest
    # Preserve the durable boundary even if subsequent local trials finish.
    completed_trial_ids: tuple[Digest, ...]
    # Record the acknowledgement time separately from trial execution.
    acknowledged_at: AwareDatetime
