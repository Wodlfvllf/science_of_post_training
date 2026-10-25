# Load models, score exact continuations, or generate only through explicit calls here.
# The default fixture builds random tiny GPT-2 weights and a tokenizer entirely offline.
"""A reference GPT-2 execution path with auditable token boundaries and likelihoods.

The first implementation intentionally scores candidates one at a time without padding
or caching. ``continuation_logprobs`` separately supports left/right padded tensors for
correctness tests and later batching. Unsupported model families fail before weights
are loaded; pretrained execution uses safe tensors, pinned revisions, and no remote code.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .data import canonical_bytes, digest_bytes
from .runtime import CancellationRequested, inference_scope, isolated_seed
from .schemas import (
    Candidate,
    CandidateScore,
    Example,
    GenerationRecord,
    GenerationSpec,
    ModelSpec,
    PromptSpec,
    RunSpec,
    TrialResult,
)

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerFast


@dataclass(frozen=True)
class ModelBundle:
    """A loaded model/tokenizer with explicit limits and numerical artifact identity.

    Underlying objects remain accessible for later research, but callers must not
    mutate them during a run. No hooks, global registry, or persistent cache is added.
    """

    # Preserve direct access to the actual reference HF model.
    model: PreTrainedModel
    # Preserve the exact fast tokenizer used by scoring and generation.
    tokenizer: PreTrainedTokenizerFast
    # Record the actual execution device selected explicitly by ModelSpec.
    device: torch.device
    # Bound input lengths by the loaded architecture's positional capacity.
    context_limit: int
    # Preserve config/tokenizer content hashes and immutable model identities.
    profile: dict[str, Any]


def load_model(spec: ModelSpec, *, allow_downloads: bool = False) -> ModelBundle:
    """Explicitly build the offline fixture or load one pinned HF GPT-2 checkpoint.

    Args:
        spec: Validated model/tokenizer/device/precision choices.
        allow_downloads: False uses cached HF files only. True explicitly allows
            model/tokenizer downloads, but never uploads or remote Python execution.

    Returns:
        A reference model in evaluation mode and an unchanged tokenizer.

    Raises:
        ImportError: The optional model dependencies are not installed.
        ValueError: Hardware, architecture, or tokenizer capability is unsupported.
        OSError: A pinned checkpoint is unavailable locally or on the allowed Hub path.
    """
    # Import heavyweight dependencies only after an explicit model-loading action.
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # Refuse a requested GPU rather than silently falling back to CPU.
    if spec.device == "cuda" and not torch.cuda.is_available():
        # Preserve the intended numerical experiment profile.
        raise ValueError("CUDA was requested but is not available in this environment.")
    # Refuse unsupported GPU bfloat16 execution before allocating model weights.
    if spec.device == "cuda" and spec.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        # Do not emulate or silently substitute a different precision.
        raise ValueError("The selected CUDA GPU does not support the requested bfloat16 profile.")
    # Pin a concrete first-GPU device instead of an ambiguous current-device label.
    device = torch.device("cuda:0" if spec.device == "cuda" else "cpu")
    # Translate only the three supported explicit dtype names.
    dtype = getattr(torch, spec.dtype)
    # Construct the local fixture without any from_pretrained/network operation.
    if spec.backend == "fixture":
        # Create a random tiny model and a complete byte-level tokenizer offline.
        model, tokenizer = _offline_fixture(spec.initialization_seed)
    # Load real checkpoints only from the validated immutable HF identities.
    else:
        # Inspect architecture before spending memory on unsupported weights.
        config = AutoConfig.from_pretrained(spec.model_id, revision=spec.revision, trust_remote_code=False, local_files_only=not allow_downloads)
        # Restrict this first verified execution path to a dense GPT-2 decoder.
        if config.model_type != "gpt2" or config.is_encoder_decoder:
            # Make extension requirements explicit instead of claiming universal support.
            raise ValueError("Only GPT-2 is verified in this release; add parity tests before another family.")
        # Use exactly the configured tokenizer repository and pinned revision.
        tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_id or spec.model_id, revision=spec.tokenizer_revision or spec.revision, trust_remote_code=False, use_fast=True, local_files_only=not allow_downloads)
        # Load safe weights with the same immutable model revision and reference attention.
        model = AutoModelForCausalLM.from_pretrained(spec.model_id, revision=spec.revision, config=config, trust_remote_code=False, use_safetensors=True, local_files_only=not allow_downloads, torch_dtype=dtype, attn_implementation=spec.attention)
    # Require tokenizer serialization/boundary capabilities used by this recipe.
    if not tokenizer.is_fast:
        # Refuse an unverified slow-tokenizer behavior rather than approximating boundaries.
        raise ValueError("This release requires a fast tokenizer.")
    # Move the model explicitly; no automatic sharding, quantization, or device map is used.
    model.to(device=device, dtype=dtype)
    # Select evaluation mode for the shared behavioral baseline.
    model.eval()
    # Ensure every tokenizer token can be represented by the loaded embedding table.
    if max(tokenizer.get_vocab().values()) >= model.get_input_embeddings().num_embeddings:
        # Detect incompatible tokenizer/model pairs before an embedding lookup fails.
        raise ValueError("Tokenizer token IDs exceed the model's embedding vocabulary.")
    # Record the full tokenizer algorithm, vocab, and special-token behavior.
    tokenizer_sha = digest_bytes(tokenizer.backend_tokenizer.to_str().encode("utf-8"))
    # Fingerprint actual loaded model configuration, not only the requested model name.
    config_sha = digest_bytes(canonical_bytes(model.config.to_dict()))
    # Preserve the model architecture's verified positional limit.
    context_limit = int(model.config.max_position_embeddings)
    # Save enough immutable/numerical identity to reject incompatible resumes.
    profile = {
        "backend": spec.backend,  # Distinguish random engineering weights from pretrained weights.
        "model_id": spec.model_id,  # Preserve the actual selected checkpoint repository.
        "revision": spec.revision,  # Pin the weights or deterministic fixture recipe.
        "tokenizer_id": spec.tokenizer_id or spec.model_id,  # Record the tokenizer source explicitly.
        "tokenizer_revision": spec.tokenizer_revision or spec.revision,  # Pin tokenizer content too.
        "tokenizer_sha256": tokenizer_sha,  # Detect locally modified tokenizer implementations.
        "config_sha256": config_sha,  # Detect altered model settings in cached snapshots.
        "dtype": str(next(model.parameters()).dtype),  # Record actual, not merely requested, precision.
        "attention": spec.attention,  # Preserve the verified attention implementation.
        "context_limit": context_limit,  # Preserve architecture-imposed length limits.
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),  # Record model size.
        "initialization_seed": spec.initialization_seed if spec.backend == "fixture" else None,  # Identify fixture weights.
    }
    # Return plain model access without adding persistent state or executing an experiment.
    return ModelBundle(model, tokenizer, device, context_limit, profile)


def _offline_fixture(seed: int) -> tuple[Any, Any]:
    """Build a 2-layer random GPT-2 and byte-level tokenizer without downloading files.

    Every byte has a vocabulary entry and there are no BPE merges. Consequently,
    Unicode and leading whitespace are preserved without a hand-written tokenizer.
    The initialization scope restores surrounding Torch RNG state.
    """
    # Import optional execution libraries only during explicit fixture construction.
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    # Reserve special IDs independently of ordinary byte tokens.
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    # Sort the byte alphabet so fixture token IDs do not depend on set iteration order.
    for token in sorted(pre_tokenizers.ByteLevel.alphabet()):
        # Give every ordinary byte a stable unique ID following the specials.
        vocabulary[token] = len(vocabulary)
    # Reuse the standard tokenizer library instead of a custom text encoder.
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    # Avoid artificial leading spaces and disable regex splitting to retain exact bytes.
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    # Decode byte tokens through the standard byte-level decoder.
    backend.decoder = decoders.ByteLevel()
    # Expose the same HF fast-tokenizer interface used by real checkpoints.
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", model_max_length=256)
    # Keep this engineering model small enough for ordinary offline CPU tests.
    config = GPT2Config(vocab_size=len(vocabulary), n_positions=256, n_ctx=256, n_embd=32, n_layer=2, n_head=2, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False)
    # Select reference attention explicitly even if Transformers changes its default.
    config._attn_implementation = "eager"
    # Protect surrounding CPU RNG state during deterministic random-weight initialization.
    with torch.random.fork_rng(devices=[]):
        # Seed CPU only, avoiding unintended global CUDA/MPS RNG changes.
        torch.random.default_generator.manual_seed(seed)
        # Instantiate random weights locally rather than loading a pretrained checkpoint.
        model = GPT2LMHeadModel(config)
    # Return the fixture objects without retaining any RNG scope or hooks.
    return model, tokenizer


def encode_prompt(bundle: ModelBundle, text: str, settings: PromptSpec) -> tuple[int, ...]:
    """Encode exact raw text, inserting only the explicitly requested BOS token.

    Automatic tokenizer special-token insertion and truncation are disabled.
    No chat template or thinking-mode behavior is inferred from the model name.
    """
    # Ask the actual tokenizer for raw-text token IDs without hidden added specials.
    ids = tuple(bundle.tokenizer.encode(text, add_special_tokens=False))
    # Apply the declared BOS policy exactly once outside tokenizer defaults.
    if settings.add_bos:
        # Read the tokenizer's actual BOS definition.
        bos = bundle.tokenizer.bos_token_id
        # Refuse a missing BOS rather than guessing EOS or another special token.
        if bos is None:
            # Make the required special-token convention explicit to the caller.
            raise ValueError("add_bos requires a tokenizer with a defined BOS token.")
        # Prefix precisely the explicitly requested token.
        ids = (bos, *ids)
    # Require a real conditioning token for next-token likelihoods/generation.
    if not ids:
        # Reject an undefined unconditioned continuation score.
        raise ValueError("At least one prompt/BOS token is required.")
    # Enforce both user and architecture length limits without truncation.
    if len(ids) > min(settings.max_sequence_tokens, bundle.context_limit):
        # Preserve the user's original scientific prompt rather than silently shortening it.
        raise ValueError("Prompt exceeds the declared or model context limit.")
    # Return unpadded logical sequence positions for later alignment.
    return ids


def continuation_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    continuation_starts: Sequence[int],
) -> tuple[tuple[float, ...], ...]:
    """Extract correctly shifted continuation log probabilities from model logits.

    Args:
        logits: A ``[batch, padded_sequence, vocabulary]`` tensor.
        input_ids: Matching integer token IDs, including the complete continuation.
        attention_mask: Matching binary mask; each sequence's real tokens must form
            one contiguous span. Both left and right padding are accepted. The caller
            must produce correctly positioned logits: GPT-2 padded forwards require
            attention-mask-derived position IDs to match the unpadded reference.
        continuation_starts: Prefix lengths in *unpadded* sequence coordinates,
            each at least one and smaller than that sequence's real-token count.

    Returns:
        CPU Python natural-log probabilities, one tuple per continuation. Softmax
        is evaluated in float32 even when the model forward uses reduced precision.

    Raises:
        ValueError: Shapes, masks, IDs, boundaries, or selected logits are invalid.
    """
    # Import Torch only when a caller explicitly supplies numerical model outputs.
    import torch

    # Validate all tensor dimensions before indexing them.
    if logits.ndim != 3 or input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or logits.shape[:2] != input_ids.shape:
        # Fail clearly instead of broadcasting misaligned sequences.
        raise ValueError("Expected aligned [B,S,V] logits and [B,S] tokens/masks.")
    # Require integer model tokens and matching batch-boundary declarations.
    if input_ids.dtype != torch.long or len(continuation_starts) != input_ids.shape[0]:
        # Refuse fractional/broadcast token IDs or missing per-sequence boundaries.
        raise ValueError("Token IDs must be int64 with one boundary per sequence.")
    # Reject non-binary masks before using them to locate real sequence positions.
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        # Make attention-mask semantics auditable rather than truthiness-based.
        raise ValueError("Attention masks must contain only zero and one.")
    # Require inputs on the same device so indexing cannot mix CPU/GPU coordinates.
    if logits.device != input_ids.device or input_ids.device != attention_mask.device:
        # Do not hide an accidental device transfer in scientific scoring.
        raise ValueError("Logits, token IDs, and masks must share one device.")
    # Collect variable-length continuation outputs without counting padded tokens.
    results: list[tuple[float, ...]] = []
    # Score each sequence using its own unpadded prefix length.
    for row, start in enumerate(continuation_starts):
        # Locate actual tokens even if padding is on the left.
        positions = torch.nonzero(attention_mask[row], as_tuple=False).flatten()
        # Require an actual integer prefix length and at least one continuation token.
        if isinstance(start, bool) or not isinstance(start, int) or not 1 <= start < len(positions):
            # Reject zero-prefix or empty candidate scores.
            raise ValueError("Continuation starts must be nonempty unpadded prefix lengths.")
        # A real-token span must be contiguous; internal holes have ambiguous shift semantics.
        if len(positions) > 1 and not bool((positions[1:] - positions[:-1] == 1).all()):
            # Avoid treating masked internal tokens as though the model never consumed them.
            raise ValueError("Real tokens must form one contiguous span.")
        # Select logits immediately preceding each continuation token, not those at the token itself.
        prediction_logits = logits[row, positions[start - 1 : -1]].float()
        # Select the exact requested teacher-forced target suffix.
        targets = input_ids[row, positions[start:]]
        # Reject invalid vocabulary indices and non-finite distributions before reduction.
        if bool((targets < 0).any()) or bool((targets >= logits.shape[-1]).any()) or not bool(torch.isfinite(prediction_logits).all()):
            # Do not silently replace NaNs or treat arbitrary logits as valid probabilities.
            raise ValueError("Scored logits must be finite and target tokens inside the vocabulary.")
        # Normalize each selected next-token distribution in explicit float32 coordinates.
        log_probs = torch.log_softmax(prediction_logits, dim=-1)
        # Gather only the log probabilities of the actually requested suffix tokens.
        selected = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Detach and transfer only the requested scalars, not a full vocabulary cache.
        results.append(tuple(float(value) for value in selected.detach().cpu().tolist()))
    # Return ordinary serializable observations for record validation.
    return tuple(results)


def score_continuations(
    bundle: ModelBundle,
    prompt: str,
    candidates: Sequence[Candidate],
    settings: PromptSpec,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[CandidateScore, ...]:
    """Teacher-force exact candidate strings after the same unchanged prompt tokens.

    Tokenize ``prompt + candidate`` jointly, then require the original prompt IDs
    to be an exact prefix. If BPE merges across that boundary, fail rather than
    silently score a different conditioning prompt or independently encoded answer.
    Leading answer whitespace often makes intended boundaries explicit.
    """
    # Import tensor operations only for an explicit scoring request.
    import torch

    # Establish the exact prefix used by both scoring and subsequent generation.
    prefix = encode_prompt(bundle, prompt, settings)
    # Tokenize every candidate before spending compute on any forward pass.
    encoded: list[tuple[Candidate, tuple[int, ...]]] = []
    # Preserve the candidate ordering supplied by the validated dataset.
    for candidate in candidates:
        # Jointly tokenize text so continuation tokens reflect the actual prompt context.
        raw = tuple(bundle.tokenizer.encode(prompt + candidate.text, add_special_tokens=False))
        # Insert only the declared BOS, exactly as for the standalone prompt.
        full = (bundle.tokenizer.bos_token_id, *raw) if settings.add_bos else raw
        # Refuse a changed prefix or an empty token suffix rather than approximate the boundary.
        if full[: len(prefix)] != prefix or len(full) <= len(prefix):
            # Make subtle tokenizer-boundary errors visible to the researcher.
            raise ValueError("Continuation tokenization changed the prompt prefix or produced no suffix.")
        # Apply both configured and architecture context limits to full scored inputs.
        if len(full) > min(settings.max_sequence_tokens, bundle.context_limit):
            # Never shorten a candidate just to make it fit.
            raise ValueError("Prompt plus continuation exceeds the context limit.")
        # Save this immutable tokenized candidate for the explicit forward loop.
        encoded.append((candidate, full))
    # Collect auditable per-candidate records without allocating a cross-example cache.
    scores: list[CandidateScore] = []
    # Scope evaluation/autograd modes and restore them on any execution failure.
    with inference_scope(bundle.model):
        # Score conservatively one full sequence at a time, without padding or KV state.
        for candidate, full in encoded:
            # Honor an explicit deadline/cancellation between synchronous forward passes.
            if should_stop is not None and should_stop():
                # Leave this logical trial incomplete so resume can rerun it with its own seed.
                raise CancellationRequested("Stopped before completing candidate scoring.")
            # Materialize exactly the saved teacher-forced input on the selected device.
            tokens = torch.tensor([full], dtype=torch.long, device=bundle.device)
            # Use an explicit all-real-token mask, including when pad and EOS share an ID.
            mask = torch.ones_like(tokens)
            # Call the module normally so future scoped registered hooks will execute.
            outputs = bundle.model(input_ids=tokens, attention_mask=mask, use_cache=False, return_dict=True)
            # Apply the correct causal shift and retain only candidate token likelihoods.
            values = continuation_logprobs(outputs.logits, tokens, mask, [len(prefix)])[0]
            # Reduce with a transparent Python sum for artifact validation and manual tests.
            total = math.fsum(values)
            # Persist exact inputs, boundaries, masks, and both clearly named reductions.
            scores.append(CandidateScore(candidate_id=candidate.id, text=candidate.text, input_ids=full, attention_mask=(1,) * len(full), continuation_start=len(prefix), token_logprobs=values, sum_logprob=total, mean_logprob=total / len(values)))
    # Return complete candidate observations; no full logits are retained.
    return tuple(scores)


def generate(
    bundle: ModelBundle,
    input_ids: tuple[int, ...],
    settings: GenerationSpec,
    prompt_settings: PromptSpec,
    *,
    seed: int,
    should_stop: Callable[[], bool] | None = None,
) -> GenerationRecord:
    """Generate one bounded continuation with explicit defaults and isolated randomness.

    KV caching follows the explicit settings (off for the paired reference recipe).
    Cancellation is cooperative between decode steps, not
    a hard timeout for a running forward pass. EOS and length limits are saved
    distinctly; an interrupted generation is never a completed logical trial.
    """
    # Import optional numerical/generation helpers only for this explicit operation.
    import torch
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

    # Require a complete prompt and enough declared/model capacity for the whole budget.
    if not input_ids or len(input_ids) + settings.max_new_tokens > min(prompt_settings.max_sequence_tokens, bundle.context_limit):
        # Do not silently reduce max_new_tokens or truncate the prompt.
        raise ValueError("The prompt plus generation budget exceeds the context limit.")
    # Stop before starting a generation if the caller's budget has already expired.
    if should_stop is not None and should_stop():
        # Preserve missing-trial semantics instead of returning an empty successful generation.
        raise CancellationRequested("Stopped before generation.")
    # Preserve the model's declared EOS convention, with tokenizer EOS as the fallback.
    eos = bundle.model.generation_config.eos_token_id
    # Use the tokenizer convention only when the model has no configured EOS token.
    if eos is None:
        # Keep the absence explicit if neither source defines EOS.
        eos = bundle.tokenizer.eos_token_id
    # Reuse EOS as generation padding only when no actual pad token exists.
    pad = bundle.tokenizer.pad_token_id
    # Supply an explicit padding token for common GPT-2 tokenizers without mutating them.
    if pad is None and eos is not None:
        # Respect the first stop token when EOS is represented by multiple IDs.
        pad = eos[0] if isinstance(eos, list) else eos
    # Create fresh decoding settings rather than inheriting hidden repository penalties.
    options: dict[str, Any] = {
        "max_new_tokens": settings.max_new_tokens,  # Bound only the generated suffix.
        "do_sample": settings.do_sample,  # Select greedy or multinomial decoding explicitly.
        "num_beams": 1,  # Disable beam-search-specific score transformations.
        "num_return_sequences": 1,  # Let the runner assign identities to independent repeats.
        "use_cache": settings.use_cache,  # Keep per-generation KV caching explicit.
        "bos_token_id": bundle.tokenizer.bos_token_id,  # Preserve the actual BOS convention.
        "eos_token_id": eos,  # Preserve configured stopping-token semantics.
        "pad_token_id": pad,  # Make padding behavior explicit without tokenizer mutation.
        "repetition_penalty": 1.0,  # Do not silently penalize repeated tokens.
        "return_dict_in_generate": False,  # Retain only raw tokens, not a vocabulary-sized cache.
    }
    # Supply sampling-only controls only when they can affect the requested strategy.
    if settings.do_sample:
        # Preserve the explicit temperature for multinomial sampling.
        options["temperature"] = settings.temperature
        # Preserve the nucleus probability mass threshold.
        options["top_p"] = settings.top_p
        # Preserve top-k filtering, including the explicit disabled value of zero.
        options["top_k"] = settings.top_k
    # Let the established HF config validate and expose all effective defaults.
    generation_config = GenerationConfig(**options)
    # Transformers >=4.50 can merge non-neutral checkpoint defaults even into a
    # supplied config. Explicitly disable that behavior where the API supports it.
    model_defaults = {"use_model_defaults": False} if "use_model_defaults" in inspect.signature(bundle.model.generate).parameters else {}

    class StopOnRequest(StoppingCriteria):
        """Bridge the ordinary cancellation predicate to HF's decode-step interface."""

        def __call__(self, tokens: Any, scores: Any, **kwargs: Any) -> bool:
            """Stop after a decode step only when the caller's predicate requests it."""
            # Avoid model, filesystem, or upload side effects in a stopping criterion.
            return should_stop is not None and should_stop()

    # Construct the exact unpadded prompt tensor on the declared device.
    tokens = torch.tensor([input_ids], dtype=torch.long, device=bundle.device)
    # Keep all prompt positions real, even if special-token IDs overlap.
    mask = torch.ones_like(tokens)
    # Scope randomness and inference independently of other notebook model calls.
    with isolated_seed(seed, bundle.device), inference_scope(bundle.model):
        # Execute the standard generator; KV state never survives this individual call.
        output = bundle.model.generate(input_ids=tokens, attention_mask=mask, generation_config=generation_config, stopping_criteria=StoppingCriteriaList([StopOnRequest()]), **model_defaults)
    # Retain only newly generated token IDs, including any actual EOS token.
    suffix = tuple(int(value) for value in output[0, len(input_ids) :].detach().cpu().tolist())
    # Normalize the possible scalar/list EOS configuration for finish-reason inspection.
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    # Give a real EOS token priority over an independently expiring time budget.
    if suffix and suffix[-1] in eos_ids:
        # Preserve ordinary model termination.
        reason = "eos"
    # Reaching the complete declared token budget is a valid bounded observation.
    elif len(suffix) == settings.max_new_tokens:
        # Preserve length termination rather than pretending the model chose to stop.
        reason = "length"
    # An early non-EOS stop comes only from our explicit cooperative criterion.
    else:
        # Mark the output partial so the runner will rerun this logical trial on resume.
        reason = "cancelled"
    # Save raw text without stripping specials or normalizing tokenization spaces.
    text = bundle.tokenizer.decode(suffix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    # Return token/text/config provenance without writing any artifact here.
    return GenerationRecord(token_ids=suffix, text=text, finish_reason=reason, effective_config=generation_config.to_dict())


def behavioral_trial(
    bundle: ModelBundle,
    example: Example,
    spec: RunSpec,
    seed: int,
    should_stop: Callable[[], bool],
) -> TrialResult:
    """Run the first ordinary recipe: paired-candidate scores and optional generation.

    The primary metric is the target's summed log probability minus the other
    candidate's summed log probability. Unequal candidate lengths are intentional;
    mean log probabilities are saved separately, not silently substituted.
    """
    # Tokenize once by the same explicit policy used in all subsequent model calls.
    prefix = encode_prompt(bundle, example.text, spec.prompt)
    # Preserve both exact teacher-forced candidate likelihoods.
    scores = score_continuations(bundle, example.text, example.candidates, spec.prompt, should_stop=should_stop)
    # Index candidate results by their dataset labels, not their ranking or token count.
    by_id = {score.candidate_id: score for score in scores}
    # Locate the non-target candidate without guessing semantic labels such as correct/incorrect.
    other = next(candidate.id for candidate in example.candidates if candidate.id != example.target_candidate)
    # Calculate the predeclared unnormalized sequence-log-likelihood margin.
    margin = by_id[example.target_candidate].sum_logprob - by_id[other].sum_logprob
    # Execute generation only when the scientific specification actually asks for it.
    generated = generate(bundle, prefix, spec.generation, spec.prompt, seed=seed, should_stop=should_stop) if spec.generation.enabled else None
    # Build one validated result; the runner separately assigns attempt/completion bookkeeping.
    return TrialResult(input_ids=prefix, attention_mask=(1,) * len(prefix), candidate_scores=scores, target_logprob_margin=margin, generation=generated)