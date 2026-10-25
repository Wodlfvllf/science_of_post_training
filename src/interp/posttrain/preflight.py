"""Short numerical checks to run on the actual research GPU before long experiments.

No task accuracy is inferred here. Cache parity is checked on a short fixed token
suffix; stochastic replay checks identical tokens under identical seeds. Neither
is a proof of long-context or cross-hardware numerical equivalence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import ModelBundle, generate
from ..runtime import inference_scope, runtime_profile
from ..schemas import GenerationSpec, PromptSpec
from .models import load_checkpoint, preflight_inputs, render_prompt
from .plan import checkpoint_for, read_plan
from .schemas import Job


def cache_parity(bundle: ModelBundle, tokens: tuple[int, ...]) -> dict[str, Any]:
    """Compare last-token logits for full recomputation versus a fresh per-call KV cache.

    Up to eight final prompt tokens are decoded incrementally; earlier tokens are
    prefetched once. Tolerances are explicit and dtype-dependent. Return measured
    discrepancies as well as allclose; model state/modes are restored on failure.
    """
    import torch

    if len(tokens) < 2:
        raise ValueError("Cache parity needs at least two prompt tokens.")
    inputs = torch.tensor([tokens], dtype=torch.long, device=bundle.device)
    start = max(1, len(tokens) - 8)
    maximum = 0.0
    allclose = True
    tolerance = 1e-4 if next(bundle.model.parameters()).dtype == torch.float32 else 0.05
    with inference_scope(bundle.model):
        cached = bundle.model(input_ids=inputs[:, :start], attention_mask=torch.ones_like(inputs[:, :start]), use_cache=True)
        state = cached.past_key_values
        for index in range(start, len(tokens)):
            full = bundle.model(input_ids=inputs[:, :index + 1], attention_mask=torch.ones_like(inputs[:, :index + 1]), use_cache=False)
            cached = bundle.model(
                input_ids=inputs[:, index:index + 1], attention_mask=torch.ones_like(inputs[:, :index + 1]),
                past_key_values=state, use_cache=True,
            )
            state = cached.past_key_values
            left, right = full.logits[:, -1].float(), cached.logits[:, -1].float()
            maximum = max(maximum, float((left - right).abs().max()))
            allclose = allclose and bool(torch.allclose(left, right, atol=tolerance, rtol=tolerance))
    return {"cache_logits_close": allclose, "max_absolute_logit_difference": maximum,
            "atol": tolerance, "rtol": tolerance, "checked_suffix_tokens": len(tokens) - start}


def preflight_checkpoint(
    plan_path: Path, checkpoint_name: str, *, device: str = "cuda:0", allow_downloads: bool = False,
) -> dict[str, Any]:
    """Load a selected checkpoint, check all prompt lengths, short cache parity, and replay.

    This explicit command downloads weights only with opt-in and writes no research
    trials. A failed check raises rather than returning a false successful exit code.
    """
    plan = read_plan(plan_path, check_source=True)
    checkpoint = checkpoint_for(plan, Job(checkpoint=checkpoint_name))
    bundle = load_checkpoint(checkpoint, plan.spec.inference, device_name=device, allow_downloads=allow_downloads)
    try:
        lengths = preflight_inputs(bundle, checkpoint, plan)
        _, tokens = render_prompt(bundle, checkpoint, plan.problems[0], plan.spec.prompts[0])
        parity = cache_parity(bundle, tokens[:64])
        raw = plan.spec.generation.model_dump(mode="python")
        raw["max_new_tokens"] = min(8, min(plan.spec.budgets))
        settings = GenerationSpec.model_validate(raw)
        prompt_settings = PromptSpec(add_bos=False, max_sequence_tokens=plan.spec.inference.max_context_tokens)
        first = generate(bundle, tokens, settings, prompt_settings, seed=plan.spec.seed)
        second = generate(bundle, tokens, settings, prompt_settings, seed=plan.spec.seed)
        replay = first.token_ids == second.token_ids
        if not parity["cache_logits_close"] or not replay:
            raise ValueError("Preflight cache parity or seeded replay failed; do not start a full run.")
        return {"study_id": plan.study_id, "checkpoint": checkpoint_name,
                "model_profile": bundle.profile, "runtime_profile": runtime_profile(bundle.device),
                **lengths, **parity, "seeded_tokens_equal": replay,
                "scope": "short numerical smoke only; no task-accuracy or long-context claim"}
    finally:
        del bundle