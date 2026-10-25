"""One-model, one-sequence HF execution for the two personal research recipes.

KV caching is local to each call; seed identity survives interruptions and GPU
partitioning. There is no automatic device mapping, remote code, quantization,
activation capture, or cross-hardware reproducibility claim. Start with a short
preflight on the actual GPU before spending a full study budget.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..data import canonical_bytes, digest_bytes
from ..models import ModelBundle, generate, load_model
from ..schemas import GenerationSpec, ModelSpec, PromptSpec
from .schemas import Checkpoint, Inference, Problem, Prompt, Rollout, StudyPlan, Trial
from .scoring import grade_answer


def load_checkpoint(
    checkpoint: Checkpoint, settings: Inference, *, device_name: str = "cuda:0",
    allow_downloads: bool = False,
) -> ModelBundle:
    """Load one pinned dense decoder safely; refuse unsupported devices and dtypes.

    The shipped research configs use Olmo 3. Native Transformers GPT-2, Qwen2,
    Llama, and Olmo3 implementations are accepted, with real pretrained parity
    still a user-run preflight requirement. Fast tokenizer/config content is hashed.
    ``CUDA_VISIBLE_DEVICES`` can select a physical GPU; no other GPUs are allocated.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # CPU support exists for small fixtures/tests, not as an automatic GPU fallback.
    device = torch.device(device_name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA are supported by this research recipe.")
    if device.type == "cpu" and settings.dtype != "float32":
        raise ValueError("CPU execution requires explicit float32.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; no CPU fallback is performed.")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        with torch.cuda.device(device):
            if settings.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("Selected GPU cannot execute the requested bfloat16 profile.")
    if checkpoint.backend == "fixture":
        if device.type != "cpu" or settings.attention != "eager":
            raise ValueError("The engineering fixture uses CPU float32 eager attention.")
        return load_model(ModelSpec(initialization_seed=checkpoint.initialization_seed))
    # The execution plan already pins full SHAs; no moving references enter loading.
    config = AutoConfig.from_pretrained(
        checkpoint.model_id, revision=checkpoint.revision, trust_remote_code=False,
        local_files_only=not allow_downloads,
    )
    if config.model_type not in {"gpt2", "qwen2", "llama", "olmo3"} or config.is_encoder_decoder:
        raise ValueError("Unsupported architecture; add a native implementation and parity tests.")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.tokenizer_id, revision=checkpoint.tokenizer_revision, trust_remote_code=False,
        local_files_only=not allow_downloads, use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("Research snapshots require a serializable fast tokenizer.")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint.model_id, revision=checkpoint.revision, config=config,
        trust_remote_code=False, use_safetensors=True, local_files_only=not allow_downloads,
        torch_dtype=getattr(torch, settings.dtype), attn_implementation=settings.attention,
    )
    model.to(device)
    model.eval()
    if max(tokenizer.get_vocab().values()) >= model.get_input_embeddings().num_embeddings:
        raise ValueError("Tokenizer IDs exceed the checkpoint embedding vocabulary.")
    context_limit = int(model.config.max_position_embeddings)
    # Paths/caches do not affect scientific identity; actual serialized content does.
    config_dict = model.config.to_dict()
    config_dict.pop("_name_or_path", None)
    profile: dict[str, Any] = {
        "backend": "huggingface", "model_id": checkpoint.model_id,
        "revision": checkpoint.revision, "tokenizer_id": checkpoint.tokenizer_id,
        "tokenizer_revision": checkpoint.tokenizer_revision,
        "tokenizer_sha256": digest_bytes(tokenizer.backend_tokenizer.to_str().encode()),
        "chat_template_sha256": digest_bytes(canonical_bytes(tokenizer.chat_template)),
        "special_tokens_sha256": digest_bytes(canonical_bytes(tokenizer.special_tokens_map)),
        "config_sha256": digest_bytes(canonical_bytes(config_dict)),
        "generation_defaults_sha256": digest_bytes(canonical_bytes(model.generation_config.to_dict())),
        "dtype": str(next(model.parameters()).dtype), "attention": settings.attention,
        "model_type": config.model_type, "context_limit": context_limit,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return ModelBundle(model, tokenizer, device, context_limit, profile)


def render_prompt(
    bundle: ModelBundle, checkpoint: Checkpoint, problem: Problem, prompt: Prompt,
) -> tuple[str, tuple[int, ...]]:
    """Render exact input without gold labels, silent chat fallback, or truncation.

    Repair drafts are supplied as task context, not as teacher-forced model
    reasoning. This makes the diagnostic usable across base/chat checkpoints
    while keeping its assisted nature explicit. Both draft variants get every cue.
    """
    # Only these question/draft fields are allowed into the prompt, never answer/provenance.
    body = f"{prompt.instruction}\n\nProblem:\n{problem.question}"
    if problem.task == "repair":
        body += f"\n\nPartial solution:\n{problem.draft}"
        if prompt.cue == "neutral":
            body += "\nContinue the solution from the previous step."
        elif prompt.cue == "check":
            body += "\nCheck the previous step, then finish the solution."
    # Apply exactly the declared template convention, not one inferred from a name.
    if prompt.format == "native" and checkpoint.stage != "base":
        if not bundle.tokenizer.chat_template:
            raise ValueError("Native condition requires a checkpoint with a real chat template.")
        messages = []
        if prompt.system is not None:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": body})
        rendered = bundle.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = tuple(bundle.tokenizer.encode(rendered, add_special_tokens=False))
    else:
        rendered = f"{body}\n\nSolution:\n"
        ids = tuple(bundle.tokenizer.encode(rendered, add_special_tokens=False))
        if prompt.add_bos:
            if bundle.tokenizer.bos_token_id is None:
                raise ValueError("Requested BOS is absent from the tokenizer.")
            ids = (bundle.tokenizer.bos_token_id, *ids)
    if not ids:
        raise ValueError("Model prompt must contain at least one token.")
    return rendered, ids


def preflight_inputs(bundle: ModelBundle, checkpoint: Checkpoint, plan: StudyPlan) -> dict[str, int]:
    """Validate every prompt length before any trial can become a partial study.

    This performs tokenization only. It ensures that the longest budget can fit
    without silently discarding question tokens or reducing the requested cap.
    """
    limit = min(bundle.context_limit, plan.spec.inference.max_context_tokens)
    longest = 0
    for problem in plan.problems:
        for prompt in plan.spec.prompts:
            _, ids = render_prompt(bundle, checkpoint, problem, prompt)
            longest = max(longest, len(ids))
            if len(ids) + max(plan.spec.budgets) > limit:
                raise ValueError("A prompt plus its full generation budget exceeds context; lock a new plan.")
    return {"longest_prompt_tokens": longest, "context_limit": limit}


def execute_trial(
    bundle: ModelBundle, checkpoint: Checkpoint, problem: Problem, prompt: Prompt,
    plan: StudyPlan, trial: Trial, should_stop: Callable[[], bool],
) -> Rollout:
    """Generate once with an isolated seed, retain exact tokens, and grade numerically."""
    # Reuse the shared HF generation path rather than inventing a custom decoder.
    rendered, ids = render_prompt(bundle, checkpoint, problem, prompt)
    raw = plan.spec.generation.model_dump(mode="python")
    raw["max_new_tokens"] = trial.budget
    settings = GenerationSpec.model_validate(raw)
    prompt_settings = PromptSpec(add_bos=False, max_sequence_tokens=plan.spec.inference.max_context_tokens)
    started = time.monotonic()
    generated = generate(
        bundle, ids, settings, prompt_settings, seed=trial.seed, should_stop=should_stop,
    )
    # The shared generator transfers tokens to CPU before returning, synchronizing CUDA timing.
    elapsed = time.monotonic() - started
    answer_text = bundle.tokenizer.decode(
        generated.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    # Grade raw text so an actual </think> boundary cannot vanish as a skipped special token.
    initial_thinking = rendered.rstrip().endswith("<think>")
    grade = grade_answer(generated.text, problem, initial_thinking=initial_thinking)
    return Rollout(
        rendered_prompt=rendered, input_ids=ids, generation=generated,
        answer_text=answer_text, initial_thinking=initial_thinking, grade=grade, elapsed_seconds=elapsed,
    )