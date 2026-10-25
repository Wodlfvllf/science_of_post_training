"""Short numerical checks to run on the actual research GPU before long experiments.

No task accuracy is inferred here. Cache parity is checked on a short fixed token
suffix; stochastic replay checks identical tokens under identical seeds. Neither
is a proof of long-context or cross-hardware numerical equivalence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import source_digest
from ..models import ModelBundle, generate
from ..runtime import inference_scope, runtime_profile
from ..schemas import GenerationSpec, PromptSpec
from .errors import StudyError, study_stage
from .models import load_checkpoint, preflight_inputs, render_prompt
from .plan import checkpoint_for, read_plan
from .schemas import Inference, Job


def cache_parity(bundle: ModelBundle, tokens: tuple[int, ...]) -> dict[str, Any]:
    """Compare last-token logits for full recomputation versus a fresh per-call KV cache.

    Up to eight final prompt tokens are decoded incrementally; earlier tokens are
    prefetched once. Tolerances are explicit and dtype-dependent. Return measured
    discrepancies as well as allclose; model state/modes are restored on failure.
    Every compared logit must be finite. Report mismatch counts, tolerance-normalized
    errors, and argmax agreement separately; argmax agreement does not waive allclose.
    """
    import torch

    if len(tokens) < 2:
        raise StudyError("Cache parity needs at least two prompt tokens.")
    inputs = torch.tensor([tokens], dtype=torch.long, device=bundle.device)
    start = max(1, len(tokens) - 8)
    maximum = 0.0
    maximum_ratio = 0.0
    finite_logits = True
    argmax_equal = True
    compared = mismatched = 0
    tolerance = 1e-4 if next(bundle.model.parameters()).dtype == torch.float32 else 0.05
    with study_stage("Cache-parity forward passes"), inference_scope(bundle.model):
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
            finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            finite_logits = finite_logits and finite
            compared += left.numel()
            # isclose alone accepts matching infinities; nonfinite logits must always fail.
            matches = torch.isfinite(left) & torch.isfinite(right) & torch.isclose(
                left, right, atol=tolerance, rtol=tolerance,
            )
            mismatched += int((~matches).sum())
            argmax_equal = argmax_equal and finite and bool((left.argmax(-1) == right.argmax(-1)).all())
            if finite:
                # Double precision is diagnostic-only, after the locked forward passes.
                # It prevents overflow in differences of otherwise finite FP32 logits.
                difference = (left.double() - right.double()).abs()
                maximum = max(maximum, float(difference.max()))
                ratio = difference / (tolerance + tolerance * right.double().abs())
                maximum_ratio = max(maximum_ratio, float(ratio.max()))
    return {
        "cache_logits_close": finite_logits and mismatched == 0,
        "cache_logits_finite": finite_logits,
        "max_absolute_logit_difference": maximum if finite_logits else None,
        "max_tolerance_ratio": maximum_ratio if finite_logits else None,
        "compared_logits": compared, "mismatched_logits": mismatched,
        "cache_argmax_equal": argmax_equal, "atol": tolerance, "rtol": tolerance,
        "checked_suffix_tokens": len(tokens) - start,
    }


def preflight_checkpoint(
    plan_path: Path, checkpoint_name: str, *, device: str = "cuda:0", allow_downloads: bool = False,
    diagnostic: bool = False, diagnostic_dtype: str | None = None,
    diagnostic_attention: str | None = None,
) -> dict[str, Any]:
    """Load a selected checkpoint, check all prompt lengths, short cache parity, and replay.

    This explicit command downloads weights only with opt-in and writes no research
    trials. A failed check raises rather than returning a false successful exit code.
    Diagnostic mode may inspect an older code-pinned plan after a fix, but reports
    both source hashes and is not approval to execute that plan with changed code.
    Run/resume source enforcement is unchanged. Dataset and model pins remain exact.
    Explicit diagnostic dtype/attention overrides apply only to this short check.
    They require diagnostic mode and report planned/effective settings separately;
    neither the plan, generation settings, nor numerical tolerances are rewritten.
    """
    # Refuse scientific overrides before loading a plan or allocating any model.
    if not diagnostic and (diagnostic_dtype is not None or diagnostic_attention is not None):
        raise StudyError("Diagnostic profile overrides require --diagnostic; run settings stay locked.")
    plan = read_plan(plan_path, check_source=not diagnostic)
    profile = plan.spec.inference.model_dump(mode="python")
    if diagnostic_dtype is not None:
        profile["dtype"] = diagnostic_dtype
    if diagnostic_attention is not None:
        profile["attention"] = diagnostic_attention
    with study_stage("Diagnostic profile validation"):
        inference = Inference.model_validate(profile)
    current_source = source_digest()
    provenance = {
        "diagnostic_only": diagnostic,
        "plan_source_matches": current_source == plan.source_sha256,
        "plan_source_sha256": plan.source_sha256, "current_source_sha256": current_source,
        "planned_inference": plan.spec.inference.model_dump(mode="json"),
        "effective_inference": inference.model_dump(mode="json"),
        "diagnostic_profile_changed": inference != plan.spec.inference,
    }
    checkpoint = checkpoint_for(plan, Job(checkpoint=checkpoint_name))
    bundle = load_checkpoint(checkpoint, inference, device_name=device, allow_downloads=allow_downloads)
    try:
        with study_stage("Runtime profiling"):
            runtime = runtime_profile(bundle.device)
        lengths = preflight_inputs(bundle, checkpoint, plan)
        _, tokens = render_prompt(bundle, checkpoint, plan.problems[0], plan.spec.prompts[0])
        parity = cache_parity(bundle, tokens[:64])
        raw = plan.spec.generation.model_dump(mode="python")
        raw["max_new_tokens"] = min(8, min(plan.spec.budgets))
        settings = GenerationSpec.model_validate(raw)
        prompt_settings = PromptSpec(add_bos=False, max_sequence_tokens=plan.spec.inference.max_context_tokens)
        # Nonfinite logits cannot support a meaningful sampling diagnostic.
        replay = None
        if parity["cache_logits_finite"]:
            with study_stage("Seeded generation replay"):
                first = generate(bundle, tokens, settings, prompt_settings, seed=plan.spec.seed)
                second = generate(bundle, tokens, settings, prompt_settings, seed=plan.spec.seed)
                replay = first.token_ids == second.token_ids
        if not parity["cache_logits_close"] or not replay:
            failed = []
            if not parity["cache_logits_close"]:
                failed.append("cache logit parity")
            if replay is False:
                failed.append("seeded token replay")
            # Only measured numbers/booleans and declared numerical settings enter this message.
            details = {"checkpoint": checkpoint.name, "model_revision": checkpoint.revision,
                       "dtype": inference.dtype, "attention": inference.attention,
                       "runtime_profile": runtime,
                       **provenance, **lengths, **parity, "seeded_tokens_equal": replay}
            raise StudyError(
                f"Preflight failed: {', '.join(failed)}. No research trials were started.\n"
                "Keep the numerical guard enabled; use these measurements to diagnose the device/profile.\n"
                + json.dumps(details, indent=2, allow_nan=False)
            )
        return {"study_id": plan.study_id, "checkpoint": checkpoint_name,
            "model_profile": bundle.profile, "runtime_profile": runtime,
            **provenance, **lengths, **parity, "seeded_tokens_equal": replay,
            "scope": "diagnostic only; not execution approval for a changed-source or changed-profile plan" if diagnostic
            else "short numerical smoke only; no task-accuracy or long-context claim"}
    finally:
        del bundle