# Exercise dataset/scoring contracts offline, with fake modules at every HF boundary.
"""Exact-answer, split-isolation, matched-draft, and immutable-JSONL regression tests.

These tests never acquire data or instantiate a model. Dataset acquisition is
tested against in-memory stand-ins, including opt-in and immutable revision checks.
"""

from __future__ import annotations

import builtins
import importlib
import json
import random
import sys
from decimal import localcontext
from types import ModuleType, SimpleNamespace

import pytest

from interp.artifacts import ArtifactError
from interp.posttrain import data as posttrain_data
from interp.posttrain.data import (
    gsm8k_problems,
    load_problems,
    make_repair_problems,
    validate_problems,
    write_problems,
)
from interp.posttrain.schemas import Problem
from interp.posttrain.scoring import grade_answer, numeric_answer


def _math(**changes) -> Problem:
    """Construct one canonical, unassisted fixture; allow explicit contract variations."""
    # Keep reference labels separate from a question containing no reference solution.
    values = dict(
        id="math-1", source_id="source-1", group_id="group-1", split="pilot",
        task="math", question="What is six times seven?", answer="42",
        provenance="offline-fixture",
    )
    values.update(changes)
    return Problem(**values)


def _repair() -> tuple[Problem, Problem]:
    """Build a small hand-verified pair independent of the synthetic generator."""
    # This source's correct product, gold, and propagated error are 56, 65, and 66.
    common = dict(
        source_id="repair-source", group_id="repair-group", split="pilot", task="repair",
        question="Seven boxes hold eight items each, with nine loose items. How many items?",
        answer="65", provenance="offline-fixture", operands=(7, 8, 9),
    )
    correct = Problem(
        **common, id="repair-correct", prefix="correct", proposed_product=56,
        draft="First calculate the items in the boxes: 7 * 8 = 56.",
    )
    error = Problem(
        **common, id="repair-error", prefix="error", proposed_product=57,
        draft="First calculate the items in the boxes: 7 * 8 = 57.",
    )
    return correct, error


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", "42"), (" +0042 ", "42"), ("-0.00", "0"),
        (".5", "1/2"), ("-.125", "-1/8"), ("42.", "42"),
        ("2/4", "1/2"), ("-2 / -4", "1/2"), ("1/-2", "-1/2"),
        ("0.5 / 2", "1/4"), ("1,234,567.50", "2469135/2"),
        ("$1,234.50", "2469/2"), ("-$42", "-42"), ("£-42", "-42"),
        ("€42", "42"), ("¥42", "42"), ("$42$", "42"), ("$$42$$", "42"),
        (r"\(42\)", "42"), (r"\[42\]", "42"),
        (r"\frac{2}{4}", "1/2"), (r"-\frac{-2}{4}", "1/2"),
        (r"$\frac{1}{2}$", "1/2"), ("0/7", "0"),
    ],
)
def test_numeric_answer_exact_canonical_forms(text, expected):
    """All supported display forms reduce to exact, signed canonical rationals."""
    # Check both reduction and canonical idempotence, without a floating-point comparison.
    assert numeric_answer(text) == expected
    assert numeric_answer(expected) == expected


@pytest.mark.parametrize(
    "text",
    [
        "", " ", "forty-two", "42 or 43", "answer = 42", "42 items", "1 2",
        "1,2", "12,34", "1234,567", "1,,000", "1,0000", "1,000,", "1.2,3",
        "NaN", "Infinity", "-inf", "1e9", "1e999999999", "0x2a", "1_000",
        "1/0", "1/-0.0", r"\frac{1}{0}", "1/2/3", "2 + 40", "6*7", "2**999999",
        "1 1/2", "42.5.", "$$42$", "$42$$", "$", "+$-42", "--42",
        r"\sqrt{1764}", r"\frac{\frac{1}{2}}{3}", r"\frac{1}{2} garbage",
        "__import__('os').system('echo unsafe')", "9" * 257, " " * 513 + "42", None,
    ],
)
def test_numeric_answer_rejects_ambiguous_unsupported_or_unbounded_input(text):
    """Malformed syntax cannot be repaired into a lucky numeric prefix or suffix."""
    # ValueError is the declared parser failure, not eval execution or arbitrary exceptions.
    with pytest.raises(ValueError):
        numeric_answer(text)


def test_numeric_parser_ignores_decimal_context_rounding():
    """Exact conversion must not depend on a caller's global Decimal precision."""
    # Deliberately choose insufficient arithmetic precision for this decimal's digits.
    with localcontext() as context:
        context.prec = 2
        assert numeric_answer("0.123456789") == "123456789/1000000000"


@pytest.mark.parametrize(
    ("text", "answer", "correct", "format_ok", "extraction"),
    [
        ("Final answer: 42", "42", True, True, "final"),
        ("fInAl AnSwEr: 84/2", "42", True, True, "final"),
        ("Final answer: 41", "41", False, True, "final"),
        (r"Final answer: \boxed{42}", "42", True, True, "final"),
        (r"Some work. \boxed{42}", "42", True, False, "boxed"),
        ("Work = 99\n#### 42", "42", True, False, "hash"),
        ("42", None, False, False, "missing"),
        ("I computed 6 * 7 = 42.", None, False, False, "missing"),
        ("", None, False, False, "missing"),
        ("Final answer: 42\nFinal answer: 41", "41", False, True, "final"),
        ("Final answer: 42\nFinal answer:", None, False, False, "invalid"),
        (r"\boxed{42} Final answer: unknown", None, False, False, "invalid"),
        ("#### 42\nFinal answer: 4,2", None, False, False, "invalid"),
        ("Final answer: 42 or 41", None, False, False, "invalid"),
        ("Final answer: 42\nActually 41", None, False, False, "invalid"),
        ("Final answer: 1/0", None, False, False, "invalid"),
        ("Final answer: 42e999999", None, False, False, "invalid"),
        ("Final answer: $$42$", None, False, False, "invalid"),
        (r"\boxed{42} then \boxed{unknown}", None, False, False, "invalid"),
        (r"\boxed{42} then \boxed{", None, False, False, "invalid"),
        (r"\boxed{42} then \boxed", None, False, False, "invalid"),
        (r"#### 42 then \boxed 41", None, False, False, "invalid"),
        ("#### 42\n#### nope", None, False, False, "invalid"),
        (r"#### 42 \boxed{41}", "41", False, False, "boxed"),
        ("Final answer: 42</s><|endoftext|>\n", "42", True, True, "final"),
        ("Final answer: 42<|im_end|><|eot_id|>", "42", True, True, "final"),
        ("Final answer: 42<|unknown|>", None, False, False, "invalid"),
        ("<think>Final answer: 42</think>41", None, False, False, "missing"),
        ("<think>Final answer: 41</think>Final answer: 42", "42", True, True, "final"),
        ("</think>Final answer: 42</think>no declaration", None, False, False, "missing"),
    ],
)
def test_final_answer_semantics_and_marker_precedence(text, answer, correct, format_ok, extraction):
    """Final-answer correctness is independent of formatting; invalid markers block fallback."""
    # A hand-authored gold label avoids depending on synthetic-generation correctness here.
    grade = grade_answer(text, _math())
    assert (grade.answer, grade.correct, grade.format_ok, grade.extraction) == (
        answer, correct, format_ok, extraction,
    )
    assert grade.step_claim == "not_observed"
    assert grade.propagated_error is False


def test_box_balancing_and_priority_are_not_flat_regex_extraction():
    """Nested fraction braces and a later lower-priority marker cannot corrupt extraction."""
    # A boxed rational remains a supported exact numeric answer despite its inner braces.
    problem = _math(answer="1/2")
    grade = grade_answer(r"\boxed{99} then \boxed{\frac{1}{2}}", problem)
    assert grade.correct and grade.extraction == "boxed" and not grade.format_ok
    # The explicit final marker is authoritative even when a correct box occurs after it.
    invalid = grade_answer(r"Final answer: nope. \boxed{\frac{1}{2}}", problem)
    assert invalid.extraction == "invalid" and not invalid.correct


@pytest.mark.parametrize(
    ("text", "claim"),
    [
        ("Final answer: 65", "not_observed"),
        ("I checked the draft. Final answer: 65", "not_observed"),
        ("7 * 8 = 56.\nFinal answer: 65", "correct"),
        ("7 × 8 = 57.\nFinal answer: 65", "incorrect"),
        (r"7 \times 8 = 56. Final answer: 65", "correct"),
        ("7*8=57\n7*8=56\nFinal answer: 65", "mixed"),
        ("7*8=112/2\nFinal answer: 65", "correct"),
        (r"7*8=\frac{112}{2}. Final answer: 65", "correct"),
        (r"7*8=\frac{112}{2} Final answer: 65", "correct"),
        ("17*8=56\nFinal answer: 65", "not_observed"),
        ("7*80=56\nFinal answer: 65", "not_observed"),
        ("8*7=56\nFinal answer: 65", "not_observed"),
        ("7*8=5,6\nFinal answer: 65", "not_observed"),
        ("7*8=56/0\nFinal answer: 65", "not_observed"),
        ("7*8=56foo\nFinal answer: 65", "not_observed"),
        ("7*8=56 - 1\nFinal answer: 65", "not_observed"),
        ("<think>7*8=56</think>Final answer: 65", "correct"),
    ],
)
def test_repair_claims_are_explicit_generated_equations_only(text, claim):
    """Correct finals can coexist with wrong/missing claims; prompts are never inferred."""
    # The supplied wrong draft alone must not become an observed generated equation.
    _, error = _repair()
    grade = grade_answer(text, error)
    assert grade.correct
    assert grade.step_claim == claim
    assert not grade.propagated_error


def test_propagated_error_is_numerical_and_error_prefix_only():
    """Propagation is descriptive, and neither step claims nor prefix labels define accuracy."""
    # A wrong final following a correct restatement is still an incorrect final answer.
    correct, error = _repair()
    grade = grade_answer("7*8=56\nFinal answer: 66", error)
    assert grade.propagated_error and not grade.correct and grade.step_claim == "correct"
    # A bare matching number is not a parsed final answer and cannot imply propagation.
    assert not grade_answer("66", error).propagated_error
    assert not grade_answer("Final answer: 66 or 65", error).propagated_error
    assert not grade_answer("Final answer: 65", correct).propagated_error
    assert not grade_answer("Final answer: 66", correct).propagated_error


def test_scoring_bounds_and_corrupt_gold_fail_closed():
    """Unbounded outputs abstain; corrupted reference labels fail explicitly."""
    # The scorer may receive malformed outputs directly, outside a rollout schema.
    assert grade_answer("x" * (1024 * 1024 + 1), _math()).extraction == "invalid"
    assert grade_answer(None, _math()).extraction == "invalid"
    with pytest.raises(ValueError, match="canonical"):
        grade_answer("Final answer: 42", _math(answer="84/2"))


def test_generation_is_deterministic_isolated_and_complete():
    """Source count means pairs; generator randomness and scientific identities are stable."""
    # Read global random state without seeding or otherwise modifying another test's RNG.
    before = random.getstate()
    problems = make_repair_problems(count=30, seed=19)
    assert random.getstate() == before
    assert problems == make_repair_problems(count=30, seed=19)
    assert problems[:20] == make_repair_problems(count=10, seed=19)
    assert len(problems) == 60 and len({row.source_id for row in problems}) == 30
    assert len({row.id for row in problems}) == 60
    assert len({row.draft for row in problems}) == 60
    for correct, error in zip(problems[::2], problems[1::2], strict=True):
        a, b, c = correct.operands
        assert (correct.prefix, error.prefix) == ("correct", "error")
        assert correct.source_id == error.source_id == correct.group_id == error.group_id
        assert correct.question == error.question and correct.provenance == error.provenance
        assert correct.answer == error.answer == str(a * b + c)
        assert correct.proposed_product == a * b != error.proposed_product
        assert correct.draft == f"First calculate the items in the boxes: {a} * {b} = {a * b}."
        assert error.draft == f"First calculate the items in the boxes: {a} * {b} = {error.proposed_product}."
        assert "Final answer" not in correct.question and "Final answer" not in correct.draft
    validate_problems(problems)


def test_generation_roles_are_disjoint_even_with_different_seeds():
    """Role membership must derive from numerical identity, not sampling RNG or ordinal."""
    # Union multiple seed samples per role before looking for any cross-role overlap.
    numerical = {}
    identities = {}
    for role in ("development", "pilot", "test"):
        rows = tuple(
            row for seed in (1, 2, 2026)
            for row in make_repair_problems(count=40, split=role, seed=seed)
        )
        numerical[role] = {row.operands for row in rows}
        identities[role] = {row.source_id for row in rows}
    for left, right in (("development", "pilot"), ("pilot", "test"), ("development", "test")):
        assert numerical[left].isdisjoint(numerical[right])
        assert identities[left].isdisjoint(identities[right])
    # Fully paired samples from separate roles also pass combined-dataset validation.
    validate_problems(tuple(
        row for role in ("development", "pilot", "test")
        for row in make_repair_problems(count=5, split=role)
    ))


@pytest.mark.parametrize("factory", [make_repair_problems, gsm8k_problems])
@pytest.mark.parametrize(
    "options",
    [
        {"count": 0}, {"count": -1}, {"count": True}, {"count": 1.5}, {"count": 10_001},
        {"split": "train"}, {"split": "invalid"}, {"seed": -1}, {"seed": True},
        {"seed": 2**32},
    ],
)
def test_acquisition_arguments_fail_before_work(factory, options):
    """Impossible counts/roles/seeds fail before any optional import or download request."""
    # No opt-in is supplied, and argument errors should precede the network gate itself.
    with pytest.raises(ValueError, match="count|split|seed"):
        factory(**options)


def test_validation_rejects_empty_duplicate_and_noncanonical_math():
    """Changing identifiers cannot legitimize duplicates or noncanonical reference answers."""
    # An empty dataset cannot form a scientific trial schedule.
    with pytest.raises(ValueError, match="contain"):
        validate_problems(())
    with pytest.raises(ValueError, match="Duplicate observation"):
        validate_problems((_math(), _math()))
    for split in ("pilot", "test"):
        duplicate = _math(
            id="other", source_id="other-source", group_id="other-group", split=split,
            question="  WHAT is six TIMES seven?\n",
        )
        with pytest.raises(ValueError, match="Duplicate question"):
            validate_problems((_math(), duplicate))
    for answer in ("84/2", "42.0", "NaN", "1,2"):
        with pytest.raises(ValueError):
            validate_problems((_math(answer=answer),))


@pytest.mark.parametrize("axis", ["group_id", "source_id"])
def test_validation_rejects_each_lineage_axis_crossing_roles(axis):
    """Neither source nor resampling-group leakage may hide behind distinct questions."""
    # Give the second observation a different question to isolate the lineage check.
    first = _math()
    values = dict(
        id="second", source_id="source-2", group_id="group-2", split="test",
        question="What is twenty-one plus twenty-one?",
    )
    values[axis] = getattr(first, axis)
    with pytest.raises(ValueError, match="crosses"):
        validate_problems((first, _math(**values)))


def test_repair_validation_requires_complete_unique_matched_pairs():
    """Missing/extra variants and silently altered scientific pair fields are invalid."""
    # The independently constructed fixture first demonstrates the positive contract.
    correct, error = _repair()
    validate_problems((correct, error))
    with pytest.raises(ValueError, match="exactly one"):
        validate_problems((correct,))
    extra = error.model_copy(update={
        "id": "error-extra", "proposed_product": 58,
        "draft": "First calculate the items in the boxes: 7 * 8 = 58.",
    })
    with pytest.raises(ValueError, match="exactly one"):
        validate_problems((correct, error, extra))
    for changes in (
        {"question": "A different word problem?"}, {"group_id": "different-group"},
        {"provenance": "other-source-revision"},
    ):
        with pytest.raises(ValueError, match="mismatched"):
            validate_problems((correct, error.model_copy(update=changes)))
    # These unchecked model copies must be revalidated rather than trusted as frozen records.
    for changes in (
        {"answer": "66"}, {"operands": (7, 8, 10)}, {"prefix": "correct"},
        {"proposed_product": None}, {"draft": error.draft + " Final answer: 65"},
    ):
        with pytest.raises(ValueError):
            validate_problems((correct, error.model_copy(update=changes)))


def test_repair_relabeling_cannot_duplicate_questions_tuples_or_drafts():
    """New source IDs do not create new independent problems from the same arithmetic."""
    # Vary the question to force the stronger numerical/draft identity checks to run.
    pair = _repair()
    relabeled = tuple(row.model_copy(update={
        "id": row.id + "-other", "source_id": "other-source", "group_id": "other-group",
        "question": "Reworded but numerically identical question?",
    }) for row in pair)
    with pytest.raises(ValueError, match="unique|Duplicate"):
        validate_problems((*pair, *relabeled))


def test_jsonl_roundtrip_immutable_publication_and_failed_validation(tmp_path):
    """Valid files round-trip; identical writes are idempotent and conflicts preserve bytes."""
    # Publish only disposable local fixture data through the real atomic writer.
    path = tmp_path / "nested" / "problems.jsonl"
    problems = make_repair_problems(count=3)
    write_problems(path, problems)
    before = path.read_bytes()
    assert before.endswith(b"\n") and load_problems(path) == problems
    write_problems(path, problems)
    assert path.read_bytes() == before
    with pytest.raises(ArtifactError, match="different content"):
        write_problems(path, tuple(reversed(problems)))
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".pending-*"))
    invalid = tmp_path / "must-not-exist" / "invalid.jsonl"
    with pytest.raises(ValueError):
        write_problems(invalid, problems[:-1])
    assert not invalid.parent.exists()


@pytest.mark.parametrize(
    "payload",
    [b"\xff\n", b"\n", b"{}\n", b"[1,2]\n", b"{broken\n", b'{"id":"a","id":"b"}\n'],
)
def test_jsonl_rejects_corruption_with_row_location(tmp_path, payload):
    """UTF-8, JSON syntax, duplicate keys, blank rows, and schema corruption fail clearly."""
    # Feed exact malformed bytes rather than letting a serializer repair the fixture.
    path = tmp_path / "corrupt.jsonl"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="row 1"):
        load_problems(path)


def test_jsonl_size_and_row_bounds_are_enforced(tmp_path, monkeypatch):
    """Reduced test limits exercise read/write bounds without large fixture allocation."""
    # A complete small math record provides a known-good starting file.
    path = tmp_path / "bounded.jsonl"
    write_problems(path, (_math(),))
    size = path.stat().st_size
    monkeypatch.setattr(posttrain_data, "_MAX_FILE_BYTES", size - 1)
    with pytest.raises(ValueError, match="bound"):
        load_problems(path)
    with pytest.raises(ValueError, match="bound"):
        write_problems(tmp_path / "too-large.jsonl", (_math(),))
    monkeypatch.setattr(posttrain_data, "_MAX_FILE_BYTES", size * 4)
    monkeypatch.setattr(posttrain_data, "_MAX_LINE_BYTES", size - 1)
    with pytest.raises(ValueError, match="bound"):
        load_problems(path)
    monkeypatch.setattr(posttrain_data, "_MAX_LINE_BYTES", size * 2)
    monkeypatch.setattr(posttrain_data, "_MAX_ROWS", 1)
    path.write_bytes(path.read_bytes() * 2)
    with pytest.raises(ValueError, match="bound"):
        load_problems(path)


def test_jsonl_rejects_symlinks_without_touching_targets(tmp_path):
    """Local immutable dataset boundaries reject symlink reads and writes."""
    # A real file makes it possible to prove the linked destination was not modified.
    target = tmp_path / "target.jsonl"
    write_problems(target, (_math(),))
    before = target.read_bytes()
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular"):
        load_problems(link)
    with pytest.raises(ArtifactError, match="symlink"):
        write_problems(link, (_math(),))
    assert target.read_bytes() == before


@pytest.fixture
def fake_gsm8k(monkeypatch):
    """Install in-memory HF stand-ins and record all metadata/data acquisition calls."""
    # The reference trace contains a conspicuous sentinel that must never enter a Problem.
    sha = "a" * 40
    calls = []
    datasets = {
        split: [
            {"question": f"{split} fixture question {index}?", "answer": f"SECRET_SOLUTION {index}\n#### {index + 1}"}
            for index in range(80 if split == "train" else 20)
        ]
        for split in ("train", "test")
    }

    class FakeApi:
        """Return only a controlled immutable dataset revision; never contact a service."""

        def dataset_info(self, repo_id, *, revision):
            """Record the exact floating-ref lookup and return its fake full SHA."""
            # Tests inspect these calls instead of relying on a real cache or network.
            calls.append(("info", repo_id, revision))
            return SimpleNamespace(sha=sha)

    def fake_load(repo_id, config, *, revision, split):
        """Record a pinned acquisition and return the requested in-memory upstream split."""
        # Refuse accidental branch-based loading even inside the fake implementation.
        assert revision == sha
        calls.append(("load", repo_id, config, revision, split))
        return datasets[split]

    hub = ModuleType("huggingface_hub")
    hub.HfApi = FakeApi
    module = ModuleType("datasets")
    module.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "datasets", module)
    return SimpleNamespace(sha=sha, calls=calls, datasets=datasets, hub=hub)


def test_gsm8k_opt_in_precedes_all_optional_imports(monkeypatch):
    """Even a full SHA is not an implicit permission to acquire remote or cached data."""
    # Refuse optional imports outright, proving the negative boundary without package installs.
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        """Fail immediately if module import or an unapproved call reaches optional HF code."""
        # Ordinary local dependencies retain their normal import semantics.
        if name.split(".")[0] in {"datasets", "huggingface_hub"}:
            raise AssertionError("Unexpected optional acquisition import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(posttrain_data)
    for revision in ("main", "a" * 40):
        with pytest.raises(ValueError, match="allow_downloads=True"):
            gsm8k_problems(count=1, revision=revision)


def test_gsm8k_floating_ref_is_resolved_once_and_solutions_are_not_prompts(fake_gsm8k):
    """Resolve metadata before loading, then persist row-level identity and verified labels only."""
    # This opt-in reaches only fake modules installed by the fixture, never the real Hub.
    rows = gsm8k_problems(count=5, revision="release-tag", allow_downloads=True)
    assert fake_gsm8k.calls == [
        ("info", "openai/gsm8k", "release-tag"),
        ("load", "openai/gsm8k", "main", fake_gsm8k.sha, "train"),
    ]
    assert len({row.source_id for row in rows}) == 5
    for row in rows:
        provenance = json.loads(row.provenance)
        assert provenance["revision"] == fake_gsm8k.sha
        assert provenance["role"] == "pilot" and provenance["source_split"] == "train"
        assert row.answer == str(provenance["source_row"] + 1)
        assert row.question == fake_gsm8k.datasets["train"][provenance["source_row"]]["question"]
        assert row.draft is None and row.prefix == "none"
        assert "SECRET_SOLUTION" not in row.model_dump_json()


def test_gsm8k_pinned_acquisition_disjoint_pools_and_local_rng(fake_gsm8k):
    """Pinned refs bypass metadata lookup; roles remain disjoint across sampling seeds."""
    # Each role receives two independently seeded samples without changing global RNG state.
    before = random.getstate()
    observed = {}
    for role in ("development", "pilot", "test"):
        rows = tuple(
            row for seed in (1, 7)
            for row in gsm8k_problems(
                count=10, split=role, seed=seed, revision=fake_gsm8k.sha, allow_downloads=True,
            )
        )
        observed[role] = {row.source_id for row in rows}
    assert random.getstate() == before
    assert all(call[0] == "load" for call in fake_gsm8k.calls)
    assert observed["development"].isdisjoint(observed["pilot"])
    assert observed["test"].isdisjoint(observed["development"] | observed["pilot"])
    options = dict(count=5, revision=fake_gsm8k.sha, allow_downloads=True)
    assert gsm8k_problems(**options) == gsm8k_problems(**options)


@pytest.mark.parametrize("answer", ["no marker 42", "#### 42 or 43", "#### 1/0", "#### 4,2", "#### 42\n#### nope"])
def test_gsm8k_corrupt_selected_labels_are_not_replaced(fake_gsm8k, answer):
    """An invalid selected source aborts acquisition rather than drawing an easier replacement."""
    # Corrupt all candidate labels so any preselected sample exercises the same failure.
    for row in fake_gsm8k.datasets["test"]:
        row["answer"] = answer
    with pytest.raises(ValueError, match="GSM8K gold"):
        gsm8k_problems(count=1, split="test", revision=fake_gsm8k.sha, allow_downloads=True)
    assert len(fake_gsm8k.calls) == 1


def test_gsm8k_rejects_impossible_pool_and_unresolved_revision(fake_gsm8k):
    """A source count or metadata error cannot silently change the requested dataset."""
    # The fake test source has only twenty rows and cannot supply twenty-one unique problems.
    with pytest.raises(ValueError, match="source pool"):
        gsm8k_problems(count=21, split="test", revision=fake_gsm8k.sha, allow_downloads=True)
    fake_gsm8k.hub.HfApi = lambda: SimpleNamespace(
        dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="abbreviated"),
    )
    before = len(fake_gsm8k.calls)
    with pytest.raises(ValueError, match="full immutable SHA"):
        gsm8k_problems(count=1, revision="main", allow_downloads=True)
    assert len(fake_gsm8k.calls) == before