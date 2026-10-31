"""Model-free, immutable contracts for the single-image hypothesis pilot."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

Index = Annotated[StrictInt, Field(ge=0)]
Seed = Annotated[StrictInt, Field(ge=0, lt=2**63)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PilotError(RuntimeError):
    """A safe, actionable pilot error, without dependency secrets/tracebacks."""


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Checkpoint(Frozen):
    name: str
    family: Literal["2b", "8b", "fixture"]
    stage: Literal["cpt", "sft", "mpo", "gspo"]
    backend: Literal["internvl", "fixture"] = "internvl"
    model_id: str
    revision: str
    fixture_seed: Seed = 17

    @field_validator("name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value) is None:
            raise ValueError("Checkpoint names must be short lowercase identifiers.")
        return value

    @model_validator(mode="after")
    def native_pin(self) -> Checkpoint:
        if self.backend == "internvl":
            if (re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*",
                             self.model_id) is None or ".." in self.model_id):
                raise ValueError("Use an explicit owner/model identifier.")
            if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
                raise ValueError("Native model and remote code require a full commit revision.")
            if self.family == "fixture":
                raise ValueError("An actual model must not be labelled as the fixture.")
        elif self.family != "fixture" or self.model_id != "offline-random-qwen3":
            raise ValueError("Fixture checkpoints must be explicitly labelled random/offline.")
        return self


class Inference(Frozen):
    dtype: Literal["float32", "bfloat16"] = "float32"
    device: str = "cuda:0"
    image_size: Literal[448] = 448
    max_context_tokens: Annotated[StrictInt, Field(ge=384, le=4096)] = 1024
    system_message: str = "You are a helpful assistant."
    answer_prefix: str = "Answer:"
    atol: Annotated[float, Field(gt=0, le=0.001, allow_inf_nan=False)] = 0.0001
    rtol: Annotated[float, Field(gt=0, le=0.001, allow_inf_nan=False)] = 0.0001

    @field_validator("device")
    @classmethod
    def explicit_device(cls, value: str) -> str:
        if value != "cpu" and re.fullmatch(r"cuda:[0-9]+", value) is None:
            raise ValueError("Choose cpu or an explicit cuda:N device; no automatic fallback.")
        return value


class StudySpec(Frozen):
    name: str = "reading-or-routing-pilot"
    dataset: Path
    seed: Seed = 20261031
    checkpoints: tuple[Checkpoint, ...]
    inference: Inference = Field(default_factory=Inference)
    max_new_tokens: Annotated[StrictInt, Field(ge=1, le=512)] = 64

    @model_validator(mode="after")
    def unique_checkpoints(self) -> StudySpec:
        names = [item.name for item in self.checkpoints]
        stages = [(item.family, item.stage) for item in self.checkpoints]
        if not names or len(names) != len(set(names)) or len(stages) != len(set(stages)):
            raise ValueError("Checkpoint names and family/stage combinations must be unique.")
        return self


class SiteSpec(Frozen):
    layer: Index
    kv_group: Index

    @property
    def label(self) -> str:
        return f"{self.layer}:{self.kv_group}"


class Job(Frozen):
    checkpoint: str
    phase: Literal["baseline", "routing", "weights"]
    split: Literal["engineering", "discovery", "pilot_validation", "confirmation"]
    sites: tuple[SiteSpec, ...] = ()
    source_checkpoint: str | None = None
    projections: tuple[Literal["q_proj", "k_proj", "v_proj", "o_proj"], ...] = (
        "q_proj", "k_proj",
    )
    include_norms: bool = False
    generate: bool = False
    prompt_profile: Literal["common", "native"] = "common"
    save_captures: bool = False
    num_shards: Annotated[StrictInt, Field(ge=1, le=128)] = 1
    shard_index: Index = 0

    @model_validator(mode="after")
    def intervention_contract(self) -> Job:
        if self.shard_index >= self.num_shards:
            raise ValueError("shard_index must be less than num_shards.")
        if len({site.label for site in self.sites}) != len(self.sites) or len(self.sites) > 64:
            raise ValueError("Sites must be unique and bounded; no implicit dense sweep.")
        if self.phase != "baseline" and not self.sites:
            raise ValueError("Routing/weight jobs need explicitly selected layer:KV-group sites.")
        if self.phase == "baseline" and self.sites:
            raise ValueError("Baseline jobs have no selected intervention sites.")
        if self.phase == "weights" and self.source_checkpoint is None:
            raise ValueError("Weight replay/rollback needs an explicit source checkpoint.")
        if self.phase != "weights" and self.source_checkpoint is not None:
            raise ValueError("Only weight jobs have a weight source.")
        if self.phase != "baseline" and (self.generate or self.prompt_profile != "common"):
            raise ValueError("Interventions are uncached common-prefix scoring, not CoT generation.")
        if not self.projections or len(set(self.projections)) != len(self.projections):
            raise ValueError("Select unique projection names.")
        return self


class Plan(Frozen):
    schema_version: Literal[1] = 1
    plan_id: Digest
    created_at: str
    source_sha256: Digest
    dataset_sha256: Digest
    spec: StudySpec
    dataset: dict


class Selection(Frozen):
    checkpoint: str
    sites: tuple[SiteSpec, ...]
    source_checkpoint: str | None = None
    projections: tuple[Literal["q_proj", "k_proj", "v_proj", "o_proj"], ...] = (
        "q_proj", "k_proj",
    )
    include_norms: bool = False


class Protocol(Frozen):
    schema_version: Literal[1] = 1
    protocol_id: Digest
    plan_id: Digest
    created_at: str
    selections: tuple[Selection, ...]
    note: str
    primary_endpoint: Literal["routing_stage_interaction"] = "routing_stage_interaction"
    primary_size: Literal["8b", "fixture"] = "8b"
    routing_effect_screen: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = .15
    weight_effect_screen: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = .10
    preservation_margin: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = .05
    multiplicity: Literal["bonferroni_primary_sites"] = "bonferroni_primary_sites"

    @model_validator(mode="after")
    def bounded_confirmatory_sites(self) -> Protocol:
        if any(not entry.sites or len(entry.sites) > 2 for entry in self.selections):
            raise ValueError("Freeze at most two candidate groups per checkpoint before validation.")
        return self