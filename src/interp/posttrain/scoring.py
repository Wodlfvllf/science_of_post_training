# Grade declared final answers here; never infer an answer from a reasoning trace.
"""Bounded exact-rational scoring and observable arithmetic-restatement labels.

This is a conservative parser, not a symbolic algebra system or an LLM judge.
Unsupported notation and ambiguous suffixes abstain rather than earning a lucky
pass. Equation claims describe generated text, not internal error detection.
"""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction
from typing import Literal

from .schemas import Grade, Problem

# Bound both string scanning and integer arithmetic before constructing numbers.
_MAX_NUMBER_CHARS = 512
_MAX_DIGITS = 256
_MAX_OUTPUT_CHARS = 1024 * 1024
# Commas are thousands separators only; decimal commas and partial groups are invalid.
_INTEGER = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
_SCALAR = rf"[+-]?(?:{_INTEGER}(?:\.[0-9]*)?|\.[0-9]+)"
_FINAL = re.compile(r"\bFinal\s+answer\s*:", re.IGNORECASE)
_BOX = re.compile(r"\\boxed\s*\{")
_BOX_MARKER = re.compile(r"\\boxed\b")
# Remove only recognized terminal control strings, never arbitrary angle-bracket text.
_END_TOKENS = (
    "</s>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>",
    "<|im_end|>", "<|end|>", "<|return|>",
)


def _math_wrapper(text: str) -> str:
    """Remove one balanced presentation wrapper without repairing malformed dollars."""
    # A broken display-math wrapper must not become a valid currency-prefixed answer.
    if text.startswith("$$") and (not text.endswith("$$") or len(text) <= 4):
        raise ValueError("Unbalanced display-math delimiter.")
    for opening, closing in (("$$", "$$"), (r"\(", r"\)"), (r"\[", r"\]"), ("$", "$")):
        if text.startswith(opening) and text.endswith(closing) and len(text) >= 2 * len(opening):
            return text[len(opening):-len(closing)].strip()
    return text


def _strip_control_tokens(text: str) -> str:
    """Strip only trailing whitespace and known model controls in linear space.

    Move an end offset instead of allocating one progressively smaller string per
    token, so a long repeated-control suffix cannot cause quadratic copying.
    """
    # Keep internal tokens intact: only a contiguous terminal suffix is presentation noise.
    end = len(text)
    while True:
        while end and text[end - 1].isspace():
            end -= 1
        token = next((token for token in _END_TOKENS if text.endswith(token, 0, end)), None)
        if token is None:
            return text[:end]
        end -= len(token)


def numeric_answer(text: str) -> str:
    """Return one exact rational in canonical ``integer`` or ``numerator/denominator`` form.

    Accept signed integers, finite decimals, simple numeric fractions, and simple
    ``\\frac{numerator}{denominator}`` notation. One currency prefix ($/£/€/¥)
    and one pair of math delimiters ($, $$, \\(, \\[) are optional. Whitespace
    around fraction operands is allowed; arithmetic, exponents, units, mixed
    numbers, nested LaTeX, and multiple alternatives are deliberately unsupported.

    Raises:
        ValueError: Non-string, empty, ambiguous, zero-denominator, unsupported,
            or over-limit input (512 characters or 256 total decimal digits).
    """
    # Check limits before stripping or constructing potentially huge integers.
    if not isinstance(text, str) or len(text) > _MAX_NUMBER_CHARS:
        raise ValueError("Numeric answer exceeds the supported text bound.")
    if sum(character in "0123456789" for character in text) > _MAX_DIGITS:
        raise ValueError("Numeric answer exceeds the supported digit bound.")
    # Balanced delimiters wrap the whole answer; unmatched dollar prefixes remain currency.
    value = _math_wrapper(text.strip())
    # A sign may precede the currency symbol; a second sign is not silently canceled.
    currency = re.fullmatch(r"([+-]?)\s*[$£€¥]\s*(.+)", value, re.DOTALL)
    if currency is not None:
        sign, value = currency.groups()
        if sign and value.startswith(("+", "-")):
            raise ValueError("Numeric answer contains ambiguous signs.")
        value = sign + value
    # Convert only a whitelisted numeric grammar, never arbitrary expression text.
    latex = re.fullmatch(r"([+-]?)\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", value)
    if latex is not None:
        sign, numerator, denominator = latex.groups()
        parts = (numerator.strip(), denominator.strip())
    else:
        sign = ""
        parts = tuple(part.strip() for part in value.split("/"))
    if len(parts) not in (1, 2) or any(re.fullmatch(_SCALAR, part) is None for part in parts):
        raise ValueError("Expected exactly one supported numeric answer.")
    # Decimal-to-Fraction conversion is exact and independent of decimal context precision.
    numbers = tuple(Fraction(Decimal(part.replace(",", ""))) for part in parts)
    if len(numbers) == 2 and numbers[1] == 0:
        raise ValueError("Numeric answer has a zero denominator.")
    result = numbers[0] if len(numbers) == 1 else numbers[0] / numbers[1]
    if sign == "-":
        result = -result
    # Fraction normalizes signs, reduces fractions, and emits integers without '/1'.
    return str(result)


def _unbox(text: str, start: int) -> tuple[str, int]:
    """Read one balanced box, returning its content and exclusive closing offset.

    ``start`` must follow the opening brace. Nested braces are retained so simple
    fractions can be parsed afterwards; a truncated box is invalid, not missing.
    """
    # Count braces rather than using a regex that truncates at a fraction's first brace.
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index], index + 1
    raise ValueError("Unclosed boxed answer.")


def _explicit_value(text: str) -> str:
    """Parse an entire marker suffix, optionally containing exactly one boxed value.

    A marker does not authorize ignoring explanatory prose, a second answer, or a
    malformed expression after its number. A balanced outer math wrapper is allowed.
    """
    # Normalize only whole-value presentation delimiters before checking for a box.
    value = _math_wrapper(text.strip())
    box = _BOX.match(value)
    if box is not None:
        content, end = _unbox(value, box.end())
        if value[end:].strip():
            raise ValueError("Unexpected text after boxed final answer.")
        return numeric_answer(content)
    return numeric_answer(value)


def _step_claim(text: str, problem: Problem) -> Literal["correct", "incorrect", "mixed", "not_observed"]:
    """Classify explicit restatements of the declared ordered multiplication only.

    Inspect all generated text, including any thinking segment, but never the
    supplied draft. Missing or unsupported equations are unobserved. Matching a
    claim is descriptive: quotations and self-corrections are not an LLM judgment.
    """
    # Ordinary math has no predeclared multiplication whose claims can be audited.
    if problem.task != "repair" or problem.operands is None:
        return "not_observed"
    a, b, _ = problem.operands
    # Capture a whole numeric-looking RHS so malformed commas cannot match a lucky prefix.
    atom = r"[+-]?(?:[0-9][0-9,.]*|\.[0-9]+)"
    rhs = rf"(?:[+-]?\\frac\s*\{{[^{{}}\n]+\}}\s*\{{[^{{}}\n]+\}}|{atom}(?:\s*/\s*{atom})?)"
    equation = re.compile(
        rf"(?<![\w.,+*/-]){re.escape(str(a))}\s*(?:\*|×|\\times)\s*"
        rf"{re.escape(str(b))}\s*=\s*({rhs})(?![\w,/]|\.[0-9.])"
        r"(?!\s*(?:[/+*=×^-]|\\times))"
    )
    observed: set[bool] = set()
    for match in equation.finditer(text):
        try:
            value = numeric_answer(match.group(1))
        except ValueError:
            # Unsupported restatements do not establish either a correct or wrong claim.
            continue
        observed.add(value == str(a * b))
    if len(observed) == 2:
        return "mixed"
    if observed:
        return "correct" if True in observed else "incorrect"
    return "not_observed"


def grade_answer(text: str, problem: Problem, *, initial_thinking: bool = False) -> Grade:
    """Grade one final answer exactly, independently of formatting and step claims.

    Use the last explicit ``Final answer:`` marker, otherwise the last box,
    otherwise the last ``####``. An invalid higher-priority marker blocks all
    fallback. Only a valid explicit final marker earns ``format_ok``. With a
    closing ``</think>``, final extraction sees only the suffix after its last
    occurrence. An unclosed thinking region (including a template-supplied opening
    passed as initial_thinking) cannot earn final-answer credit. Known trailing
    model-control strings are removed repeatedly.

    Missing/invalid output has no answer and is incorrect. ``extraction`` retains
    the successful marker type or ``missing``/``invalid``; it is not a detection
    rate. ``propagated_error`` is a numerical match for error-prefix trials only.
    Corrupt gold labels raise ValueError rather than being counted as model errors.
    """
    # Labels must already be canonical, even when this API is used without a data loader.
    gold = numeric_answer(problem.answer)
    if gold != problem.answer:
        raise ValueError("Problem gold answer must be canonical.")
    if not isinstance(text, str) or len(text) > _MAX_OUTPUT_CHARS:
        return Grade(answer=None, correct=False, format_ok=False, extraction="invalid")
    clean = _strip_control_tokens(text)
    # Claims use generated text; final answers never use a hidden/reasoning-only marker.
    claim = _step_claim(clean, problem)
    # Track explicit delimiters instead of crediting a truncated reasoning-only answer.
    thinking = initial_thinking
    for marker in re.finditer(r"</?think>", clean):
        thinking = marker.group() == "<think>"
    if thinking:
        return Grade(answer=None, correct=False, format_ok=False, extraction="missing",
                     unfinished_thinking=True, step_claim=claim)
    final_text = clean.rsplit("</think>", 1)[-1]
    finals = list(_FINAL.finditer(final_text))
    boxes = list(_BOX_MARKER.finditer(final_text))
    extraction: Literal["final", "boxed", "hash", "missing", "invalid"] = "missing"
    answer: str | None = None
    try:
        if finals:
            extraction = "final"
            answer = _explicit_value(final_text[finals[-1].end():])
        elif boxes:
            extraction = "boxed"
            # A later truncated box declaration blocks an earlier valid box or hash answer.
            opening = _BOX.match(final_text, boxes[-1].start())
            if opening is None:
                raise ValueError("Boxed answer has no opening brace.")
            content, end = _unbox(final_text, opening.end())
            # A boxed term inside a larger expression or later contradiction is not
            # an answer. Permit only terminal presentation punctuation/delimiters.
            if final_text[end:].strip() not in ("", ".", "$", "$$", r"\)", r"\]", "$.", "$$.", r"\).", r"\]."):
                raise ValueError("Substantive text follows the last boxed answer.")
            answer = numeric_answer(content)
        elif "####" in final_text:
            extraction = "hash"
            answer = _explicit_value(final_text.rsplit("####", 1)[1])
    except ValueError:
        # Do not rescue a malformed declaration using any earlier correct-looking number.
        extraction = "invalid"
    propagated = (
        problem.task == "repair" and problem.prefix == "error"
        and problem.operands is not None and problem.proposed_product is not None
        and answer == str(problem.proposed_product + problem.operands[2])
    )
    return Grade(
        answer=answer, correct=answer == gold, format_ok=extraction == "final",
        extraction=extraction, step_claim=claim, propagated_error=propagated,
    )