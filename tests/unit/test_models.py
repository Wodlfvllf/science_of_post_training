# Verify scoring with independent next-token calculations, not just replaying our own reductions.
"""Offline CPU model tests covering exact boundaries, padding, sampling, and state cleanup."""

from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest

from interp.models import continuation_logprobs, encode_prompt, generate, score_continuations
from interp.runtime import inference_scope
from interp.schemas import Candidate, GenerationSpec, PromptSpec


def test_multitoken_scores_match_manual_next_token_calculation(bundle):
    """Score each requested candidate token at the preceding causal logit position."""
    # Import numerical operations for this explicitly model-dependent test.
    import torch

    # Include unequal lengths, leading spaces, and UTF-8 bytes in actual candidates.
    candidates = (Candidate(id="short", text=" 42"), Candidate(id="unicode", text=" Café ☕"))
    # Use the exact raw-text/BOS reference convention.
    settings = PromptSpec()
    # Execute the actual teacher-forced scoring path on the offline HF model.
    scores = score_continuations(bundle, "Answer:", candidates, settings)
    # Independently recompute each suffix probability from direct model outputs.
    for score in scores:
        # Materialize the saved actual model input, not an independently encoded answer.
        tokens = torch.tensor([score.input_ids], dtype=torch.long)
        # Obtain a plain numerical reference without retaining a gradient graph.
        with torch.no_grad():
            # Call the reference module normally with the explicit attention mask.
            logits = bundle.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
        # Compute one likelihood per target token with a visibly explicit causal shift.
        manual = [float(torch.log_softmax(logits[0, position - 1].float(), dim=-1)[tokens[0, position]]) for position in range(score.continuation_start, tokens.shape[1])]
        # Compare the implementation's retained token scores to the independent calculation.
        assert score.token_logprobs == pytest.approx(manual, abs=1e-6)
        # Independently check the unnormalized primary score.
        assert score.sum_logprob == pytest.approx(math.fsum(manual), abs=1e-6)
        # Verify the recorded continuation is actually multi-token for this fixture.
        assert len(score.token_logprobs) > 1
        # Confirm exact byte-token decoding recovers the requested Unicode/whitespace suffix.
        assert bundle.tokenizer.decode(score.input_ids[score.continuation_start :], clean_up_tokenization_spaces=False) == score.text


def test_left_right_padding_and_unequal_suffixes_match_manual_logits(bundle):
    """Unpadded prefix lengths must map correctly onto left/right padded tensor columns."""
    # Import Torch only in the numerical test body.
    import torch

    # Create deterministic, position-dependent logits without changing global RNG state.
    logits = torch.randn(2, 8, 13, generator=torch.Generator().manual_seed(9))
    # Make one left-padded and one right-padded input with unequal real-token counts.
    tokens = torch.tensor([[0, 0, 3, 4, 5, 6, 0, 0], [1, 2, 3, 4, 5, 6, 0, 0]])
    # Define contiguous real spans independently of padding token IDs.
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]])
    # Use prefix lengths in unpadded, not tensor-column, coordinates.
    starts = [2, 1]
    # Extract both variable-length continuations through the scoring utility.
    values = continuation_logprobs(logits, tokens, mask, starts)
    # Independently index the actual real-token positions for each row.
    for row, start in enumerate(starts):
        # Locate contiguous real coordinates from the explicit attention mask.
        positions = torch.nonzero(mask[row]).flatten().tolist()
        # Apply the preceding-real-token causal shift explicitly.
        manual = [float(torch.log_softmax(logits[row, positions[index - 1]], dim=-1)[tokens[row, positions[index]]]) for index in range(start, len(positions))]
        # Check every retained suffix token, excluding all padding.
        assert values[row] == pytest.approx(manual, abs=1e-6)
    # Ensure the utility did not force equal continuation lengths.
    assert [len(row) for row in values] == [2, 5]


def test_invalid_scoring_masks_logits_and_boundaries_fail(bundle):
    """Do not broadcast holes, empty suffixes, or non-finite probabilities into valid scores."""
    # Import numerical tensors only for this explicit scoring validation test.
    import torch

    # Construct a simple valid shape with deterministic finite logits.
    logits = torch.zeros(1, 4, 10)
    # Provide real integer token IDs.
    tokens = torch.tensor([[1, 2, 3, 4]])
    # Reject an internal masked hole with ambiguous causal sequence semantics.
    with pytest.raises(ValueError, match="contiguous"):
        # Keep the token sequence shape unchanged while invalidating its logical mask.
        continuation_logprobs(logits, tokens, torch.tensor([[1, 0, 1, 1]]), [1])
    # Reject a boundary that leaves no requested continuation tokens.
    with pytest.raises(ValueError, match="prefix"):
        # Set the prefix length equal to the full real sequence length.
        continuation_logprobs(logits, tokens, torch.ones_like(tokens), [4])
    # Introduce a NaN in a selected next-token distribution deliberately.
    logits[0, 0, 0] = float("nan")
    # Reject rather than sanitize undefined candidate probabilities.
    with pytest.raises(ValueError, match="finite"):
        # Score a suffix that would consume that invalid distribution.
        continuation_logprobs(logits, tokens, torch.ones_like(tokens), [1])


def test_joint_tokenization_boundary_changes_are_refused(bundle):
    """A real BPE boundary merge must fail before scoring against a changed prompt."""
    # Reuse established offline tokenizer classes rather than mocking tokenization semantics.
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    # Make a tiny real BPE vocabulary whose merge crosses the a/b prompt boundary.
    backend = Tokenizer(models.BPE(vocab={"<pad>": 0, "<bos>": 1, "<eos>": 2, "a": 3, "b": 4, "ab": 5}, merges=[("a", "b")]))
    # Expose the same tokenizer interface as the model execution path.
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>")
    # Reuse the actual model but substitute this explicitly boundary-merging tokenizer.
    merging = replace(bundle, tokenizer=tokenizer)
    # Reject the changed conditioning prefix instead of scoring token ab after token a.
    with pytest.raises(ValueError, match="changed the prompt prefix"):
        # Joint encoding of a+b differs from standalone a, even after the same BOS insertion.
        score_continuations(merging, "a", (Candidate(id="merged", text="b"),), PromptSpec())


def test_context_and_bos_are_explicit_without_truncation(bundle):
    """Missing conditioning or excessive prompt/generation length cannot be silently repaired."""
    # Require a real prefix if BOS insertion is disabled.
    with pytest.raises(ValueError, match="At least one"):
        # An empty raw text with no BOS has no next-token conditioning distribution.
        encode_prompt(bundle, "", PromptSpec(add_bos=False))
    # Reject an oversized prompt under a deliberately short declared limit.
    with pytest.raises(ValueError, match="context limit"):
        # Do not slice this original text down to fit.
        encode_prompt(bundle, "too long", PromptSpec(max_sequence_tokens=2))
    # Reject a generation budget that cannot fit rather than reducing its requested length.
    with pytest.raises(ValueError, match="generation budget"):
        # Give generation less context space than prompt plus complete token budget.
        generate(bundle, (1,), GenerationSpec(max_new_tokens=3), PromptSpec(max_sequence_tokens=2), seed=1)


def test_sampling_replays_per_trial_and_restores_surrounding_rng(bundle):
    """Intervening trials must not change a repeat's token sample or surrounding global RNG."""
    # Inspect Torch's actual CPU random state for restoration assertions.
    import torch

    # Select genuine multinomial decoding with explicitly neutral truncation filters.
    settings = GenerationSpec(do_sample=True, temperature=0.8, max_new_tokens=6)
    # Use the actual exact-text/BOS prefix from the offline tokenizer.
    prefix = encode_prompt(bundle, "A:", PromptSpec())
    # Preserve unrelated notebook RNG state before the explicit generation action.
    torch_before, python_before = torch.get_rng_state().clone(), random.getstate()
    # Generate one trial from its isolated seed.
    first = generate(bundle, prefix, settings, PromptSpec(), seed=123)
    # Consume a different trial's randomness between replications.
    generate(bundle, prefix, settings, PromptSpec(), seed=456)
    # Replay the original logical trial independently of preceding execution order.
    repeated = generate(bundle, prefix, settings, PromptSpec(), seed=123)
    # Compare actual raw output tokens, not just displayed text.
    assert first.token_ids == repeated.token_ids
    # Ensure Torch's caller-owned CPU randomness was restored after every call.
    assert torch.equal(torch_before, torch.get_rng_state())
    # Ensure Python's unrelated randomness was restored as well.
    assert random.getstate() == python_before
    # Preserve explicit full-sequence/no-cache generation in saved provenance.
    assert first.effective_config["use_cache"] is False


def test_inference_modes_restore_on_exception(bundle, monkeypatch):
    """Behavioral inference cannot leave caller-owned mixed module modes altered after failure."""
    # Select intentionally mixed training/evaluation flags under a restoration fixture.
    with monkeypatch.context() as scoped:
        # Simulate a caller retaining the parent in train mode.
        scoped.setattr(bundle.model, "training", True)
        # Keep a child explicitly in eval mode to catch recursive train(flag) restoration bugs.
        scoped.setattr(bundle.model.transformer.h[0], "training", False)
        # Preserve every module's original caller-owned mode.
        original = [module.training for module in bundle.model.modules()]
        # Force an exception inside the scope, not just a normal return.
        with pytest.raises(RuntimeError):
            # Scope evaluation/autograd policies independently.
            with inference_scope(bundle.model):
                # Confirm evaluation is actually selected before the operation.
                assert not any(module.training for module in bundle.model.modules())
                # Simulate failure during the model operation.
                raise RuntimeError("Injected model failure.")
        # Restore each mode exactly, including heterogeneous child states.
        assert [module.training for module in bundle.model.modules()] == original


def test_generation_cancellation_remains_partial(bundle):
    """An early cooperative stop must not be reported as a model-selected EOS or full length."""
    # Track preflight/decode-step stop checks without creating a timer thread.
    checks = 0

    def stop():
        """Request cancellation only after generation has started producing tokens."""
        # Use a deterministic test predicate independent of execution speed.
        nonlocal checks
        # Count each actual cancellation check.
        checks += 1
        # Allow preflight and one decode step before explicitly stopping.
        return checks >= 3

    # Generate from the actual fixture with a cooperative cancellation predicate.
    result = generate(bundle, (1,), GenerationSpec(max_new_tokens=8), PromptSpec(), seed=1, should_stop=stop)
    # Preserve the exact partial-output termination category.
    assert result.finish_reason == "cancelled"
    # Do not count the shorter suffix as if the full declared budget completed.
    assert len(result.token_ids) < 8


@pytest.mark.parametrize("left_padding", [True, False])
def test_real_padded_forward_matches_unpadded_reference(bundle, left_padding):
    """Padding extraction agrees with real GPT-2 only under the same logical position IDs."""
    # Import Torch only for this explicit real-model parity test.
    import torch

    # Use distinct unequal sequence lengths with suffixes longer than one token.
    sequences = ((1, 10, 11, 12), (1, 10, 20, 21, 22, 23))
    # Use the same logical prefix boundary for both independent unpadded sequences.
    starts = [2, 2]
    # Allocate a rectangular batch with a real padding ID defined by the fixture.
    tokens = torch.zeros(2, 6, dtype=torch.long)
    # Keep padding semantics explicit, not inferred from possibly overlapping special IDs.
    mask = torch.zeros_like(tokens)
    # Place each complete real-token span at the selected left/right alignment.
    for row, sequence in enumerate(sequences):
        # Compute physical offset without modifying logical prompt/suffix coordinates.
        offset = tokens.shape[1] - len(sequence) if left_padding else 0
        # Copy all actual tokens, including the full continuation.
        tokens[row, offset : offset + len(sequence)] = torch.tensor(sequence)
        # Mark only the contiguous real span as visible to causal attention.
        mask[row, offset : offset + len(sequence)] = 1
    # GPT-2's learned absolute positions must start at zero for every unpadded sequence.
    position_ids = (mask.cumsum(dim=-1) - 1).masked_fill(mask == 0, 0)
    # Run actual padded/unpadded forwards under the same explicit evaluation policy.
    with inference_scope(bundle.model):
        # Produce a reference full-batch forward with padding-aware positions and no KV cache.
        padded = bundle.model(input_ids=tokens, attention_mask=mask, position_ids=position_ids, use_cache=False).logits
        # Extract exact suffix probabilities in logical unpadded coordinates.
        batched_scores = continuation_logprobs(padded, tokens, mask, starts)
        # Compare against a separately evaluated unpadded reference for each row.
        for row, sequence in enumerate(sequences):
            # Materialize only one real sequence with no physical padding.
            unpadded = torch.tensor([sequence], dtype=torch.long)
            # Preserve an all-real attention mask in the independent baseline.
            real_mask = torch.ones_like(unpadded)
            # Let ordinary unpadded GPT-2 positions supply the independent reference.
            logits = bundle.model(input_ids=unpadded, attention_mask=real_mask, use_cache=False).logits
            # Extract the same multi-token continuation from the independent baseline.
            expected = continuation_logprobs(logits, unpadded, real_mask, [starts[row]])[0]
            # Permit only negligible float32 batching round-off, not a position offset.
            assert batched_scores[row] == pytest.approx(expected, abs=1e-6)


def test_model_failure_restores_sampling_rng_and_modes(bundle, monkeypatch):
    """An exception inside generate must restore both random streams and module modes."""
    # Inspect the caller-owned CPU Torch state before injecting a model exception.
    import torch

    # Snapshot unrelated surrounding randomness, not just the trial-local seed.
    python_before, torch_before = random.getstate(), torch.get_rng_state().clone()
    # Preserve every module's existing evaluation/training mode.
    modes = [module.training for module in bundle.model.modules()]

    def fail(**kwargs):
        """Consume real RNG values inside the protected scope, then fail deliberately."""
        # Consume Python randomness so restoration is not a vacuous assertion.
        random.random()
        # Consume Torch randomness for the same protected-state test.
        torch.rand(5)
        # Simulate an execution failure during the actual HF generation boundary.
        raise RuntimeError("Controlled generation failure.")

    # Inject a failing model operation rather than failing before scopes are entered.
    monkeypatch.setattr(bundle.model, "generate", fail)
    # Propagate the actual execution error rather than sanitizing numerical state silently.
    with pytest.raises(RuntimeError, match="Controlled"):
        # Enter isolated RNG/evaluation scopes through the actual generation wrapper.
        generate(bundle, (1,), GenerationSpec(do_sample=True), PromptSpec(), seed=321)
    # Restore Python's unrelated random stream even after model failure.
    assert random.getstate() == python_before
    # Restore Torch's unrelated CPU random stream even after model failure.
    assert torch.equal(torch.get_rng_state(), torch_before)
    # Restore heterogeneous module modes, not only the top-level training flag.
    assert [module.training for module in bundle.model.modules()] == modes


def test_offline_fixture_never_calls_checkpoint_download_api(monkeypatch):
    """Constructing the engineering fixture must not merely rely on network-cache availability."""
    # Inspect actual established model-loading APIs rather than only setting offline variables.
    import transformers

    from interp.models import load_model
    from interp.schemas import ModelSpec

    def forbidden(*args, **kwargs):
        """Fail on any attempt to resolve a pretrained checkpoint during fixture creation."""
        # Local random configuration must be sufficient for the whole fixture.
        raise AssertionError("Offline fixture cannot call from_pretrained.")

    # Guard all pretrained config/model/tokenizer download/cache entry points used by load_model.
    for factory in (transformers.AutoConfig, transformers.AutoModelForCausalLM, transformers.AutoTokenizer):
        # Reject cached checkpoint lookup as well as actual network downloads.
        monkeypatch.setattr(factory, "from_pretrained", forbidden)
    # Build a genuine tiny random model solely from local config and byte vocabulary.
    local = load_model(ModelSpec())
    # Preserve the fixture's explicit identity in numerical provenance.
    assert local.profile["backend"] == "fixture" and str(local.device) == "cpu"