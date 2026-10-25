# Prepare immutable numerical datasets here, before scheduling any model execution.
"""Offline paired arithmetic data and explicitly opted-in, SHA-pinned GSM8K import.

Questions, reference labels, and provenance remain separate. Synthetic evaluation
roles own disjoint numerical tuples; GSM8K development/pilot own disjoint training
rows and final evaluation uses test rows. No reference solution becomes a prompt.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ..artifacts import atomic_write
from ..data import canonical_bytes
from .schemas import Problem, StudySplit
from .scoring import numeric_answer

# Keep the initial study format bounded in both memory and independently scheduled rows.
_MAX_SOURCE_COUNT = 10_000
_MAX_ROWS = 2 * _MAX_SOURCE_COUNT
_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_LINE_BYTES = 1024 * 1024
_ROLES: tuple[StudySplit, ...] = ("development", "pilot", "test")
_REPAIR_POOL_SIZE = 500 * 500


def _request(count: int, split: StudySplit, seed: int) -> None:
    """Validate bounded source counts, explicit evaluation roles, and isolated RNG seeds.

    Reject booleans and non-integers rather than allowing Python's integer coercion.
    These checks run before optional imports, network access, or pool allocation.
    """
    # Count means source questions, not the number of repair-condition observations.
    if type(count) is not int or not 1 <= count <= _MAX_SOURCE_COUNT:
        raise ValueError(f"count must be an integer from 1 to {_MAX_SOURCE_COUNT}.")
    if split not in _ROLES:
        raise ValueError("split must be development, pilot, or test.")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a nonnegative 32-bit integer.")


def _digest(text: str) -> str:
    """Return stable source/pool identity without Python's randomized process hash."""
    # SHA-256 membership is independent of sampling seed and interpreter invocation.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_problems(problems: tuple[Problem, ...]) -> None:
    """Validate canonical labels, observation uniqueness, split isolation, and complete pairs.

    Math question duplicates are rejected after case/whitespace normalization even
    under different source IDs. Repair sources must contain exactly one correct
    and one single-error draft, with identical question, gold, group, split,
    operands, and provenance. Numerical tuples and drafts cannot be relabeled as
    new repair sources. Schema validation is reapplied to catch unchecked copies.

    Raises:
        ValueError: Empty/oversized data, malformed records, noncanonical gold,
            duplicate observations/questions/drafts, leakage, or incomplete pairs.
    """
    # Refuse vacuous or unbounded study schedules before allocating validation indexes.
    if not problems or len(problems) > _MAX_ROWS:
        raise ValueError(f"Dataset must contain 1 to {_MAX_ROWS} observations.")
    ids: set[str] = set()
    lineage: dict[tuple[str, str], str] = {}
    sources: dict[str, list[Problem]] = defaultdict(list)
    questions: dict[str, Problem] = {}
    tuples: dict[tuple[int, int, int], str] = {}
    drafts: set[str] = set()
    for original in problems:
        # Frozen Pydantic model_copy/model_construct can bypass validators; do not trust them.
        if not isinstance(original, Problem):
            raise ValueError("Dataset rows must be Problem records.")
        row = Problem.model_validate(original.model_dump(mode="python"))
        if row.id in ids:
            raise ValueError(f"Duplicate observation ID: {row.id}")
        ids.add(row.id)
        if numeric_answer(row.answer) != row.answer:
            raise ValueError(f"Gold answer must be canonical: {row.id}")
        if not row.question.strip() or not row.provenance.strip():
            raise ValueError("Question and provenance must contain non-whitespace text.")
        # Both correlated lineage axes are independently forbidden from crossing roles.
        for axis in ("source_id", "group_id"):
            key = (axis, getattr(row, axis))
            if lineage.setdefault(key, row.split) != row.split:
                raise ValueError(f"{axis} crosses dataset splits.")
        normalized = " ".join(row.question.split()).casefold()
        prior = questions.setdefault(normalized, row)
        if prior is not row and (
            row.task == "math" or prior.task == "math" or prior.source_id != row.source_id
        ):
            raise ValueError("Duplicate question under different observations/sources.")
        sources[row.source_id].append(row)
        if row.task == "repair":
            # Independently demand actual integers, not a schema-bypassing None product.
            if row.operands is None or type(row.proposed_product) is not int:
                raise ValueError("Repair requires operands and an integer proposed product.")
            a, b, c = row.operands
            if row.answer != str(a * b + c):
                raise ValueError("Repair gold disagrees with operands.")
            if (row.proposed_product == a * b) != (row.prefix == "correct"):
                raise ValueError("Repair prefix disagrees with its numerical product.")
            expected = f"First calculate the items in the boxes: {a} * {b} = {row.proposed_product}."
            if row.draft != expected or row.draft in drafts:
                raise ValueError("Repair requires a unique, exact single-step draft.")
            drafts.add(row.draft)
            if tuples.setdefault(row.operands, row.source_id) != row.source_id:
                raise ValueError("Duplicate repair numerical tuple under different sources.")
    # Source-level checks prohibit mixing unrelated tasks or selectively retaining variants.
    for rows in sources.values():
        if any(row.task == "repair" for row in rows):
            if len(rows) != 2 or {row.prefix for row in rows} != {"correct", "error"}:
                raise ValueError("Repair source needs exactly one correct and one error variant.")
            left, right = rows
            matched = ("task", "question", "answer", "group_id", "split", "operands", "provenance")
            if any(getattr(left, field) != getattr(right, field) for field in matched):
                raise ValueError("Repair pair has mismatched question, gold, lineage, or provenance.")
        # After split/pair validation, keep every correlated source in one resampling/GPU group.
        if len({row.group_id for row in rows}) != 1:
            raise ValueError("Each source_id must map to exactly one group_id.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently accepting a changed label."""
    # Check each object before handing it to the schema, including nested future metadata.
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    """Refuse nonstandard JSON constants, even before strict schema validation."""
    # Python's JSON reader otherwise accepts NaN/Infinity outside the JSON standard.
    raise ValueError("Non-finite JSON constants are unsupported.")


def load_problems(path: Path) -> tuple[Problem, ...]:
    """Read bounded UTF-8 JSONL and apply record-level and cross-source validation.

    Only regular, non-symlink local files are accepted. Limit total bytes to 32 MiB,
    each row to 1 MiB, and observations to 20,000. Blank lines, duplicate JSON keys,
    invalid UTF-8, and malformed labels are errors, with a one-based row location.
    A complete final JSON row need not have a terminal newline.
    """
    # Refuse special files before opening a stream that could block or never terminate.
    if path.is_symlink() or not path.is_file():
        raise ValueError("Problem input must be a regular, non-symlink file.")
    rows: list[Problem] = []
    consumed = 0
    with path.open("rb") as handle:
        while True:
            # Bound actual reads, not only stat size, to cover concurrently growing files.
            line = handle.readline(_MAX_LINE_BYTES + 1)
            if not line:
                break
            consumed += len(line)
            if len(line) > _MAX_LINE_BYTES or consumed > _MAX_FILE_BYTES or len(rows) >= _MAX_ROWS:
                raise ValueError("Problem JSONL exceeds its supported size/row bound.")
            try:
                raw = json.loads(
                    line.decode("utf-8"), object_pairs_hook=_unique_object,
                    parse_constant=_invalid_constant,
                )
                rows.append(Problem.model_validate(raw))
            except (ValueError, RecursionError) as error:
                # Preserve location without including solution-bearing source contents.
                raise ValueError(f"Invalid problem JSONL row {len(rows) + 1}.") from error
    problems = tuple(rows)
    validate_problems(problems)
    return problems


def write_problems(path: Path, problems: tuple[Problem, ...]) -> None:
    """Validate and atomically publish immutable canonical UTF-8 JSONL.

    Parent directories are created only after validation/serialization succeeds.
    Repeating exactly the same bytes is idempotent; different existing bytes or a
    destination symlink raise an artifact error. Row order is intentionally retained.
    The writer enforces the reader's bounds so every emitted artifact is loadable.
    """
    # Never create a destination for corrupt labels or incomplete matched pairs.
    validate_problems(problems)
    lines: list[bytes] = []
    size = 0
    for problem in problems:
        line = canonical_bytes(problem.model_dump(mode="json")) + b"\n"
        size += len(line)
        if len(line) > _MAX_LINE_BYTES or size > _MAX_FILE_BYTES:
            raise ValueError("Problem JSONL exceeds its supported size bound.")
        lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reuse the project's fsync and atomic no-overwrite publication boundary.
    atomic_write(path, b"".join(lines), immutable=True)


def _shuffled_indices(size: int, rng: random.Random) -> Iterator[int]:
    """Yield a finite random permutation lazily, without replacement or retry loops.

    A sparse Fisher-Yates swap map costs memory proportional to consumed indices,
    not the full candidate pool. Exhaustion is explicit rather than probabilistic.
    """
    # Use only the caller-owned RNG; dataset creation cannot perturb global random state.
    swaps: dict[int, int] = {}
    for remaining in range(size, 0, -1):
        chosen = rng.randrange(remaining)
        value = swaps.get(chosen, chosen)
        replacement = swaps.pop(remaining - 1, remaining - 1)
        if chosen != remaining - 1:
            swaps[chosen] = replacement
        yield value


def make_repair_problems(
    *, count: int = 50, split: StudySplit = "pilot", seed: int = 2026,
) -> tuple[Problem, ...]:
    """Generate ``count`` sources, each with both correct/error first-step variants.

    Versioned candidates use boxes/items in [2, 501] and loose items in [1, 99].
    Each ordered multiplication owns one deterministic loose-item count and role;
    SHA-256 role membership is independent of seed. This guarantees distinct
    drafts and disjoint numerical tuples across roles even under different seeds.
    Nonzero ±1..3 errors are source-stable, not a changed original solution trace.
    Counts are bounded to 1..10,000 sources, with finite pool-exhaustion checks.
    """
    # Validate the request before sampling or assembling any observations.
    _request(count, split, seed)
    rows: list[Problem] = []
    for index in _shuffled_indices(_REPAIR_POOL_SIZE, random.Random(seed)):
        a, b = 2 + index // 500, 2 + index % 500
        c = 1 + int(_digest(f"repair-v1-loose:{a}:{b}"), 16) % 99
        digest = _digest(f"repair-v1:{a}:{b}:{c}")
        if _ROLES[int(digest, 16) % len(_ROLES)] != split:
            continue
        source = "repair-" + digest[:32]
        offset = (-3, -2, -1, 1, 2, 3)[int(digest[:16], 16) % 6]
        question = (
            f"There are {a} boxes with {b} items in each box and {c} loose items. "
            "How many items are there in total?"
        )
        provenance = canonical_bytes({
            "generator": "repair-v1", "operands": [a, b, c], "role": split,
        }).decode("utf-8")
        # Both variants use exactly the schema's declared single multiplication step.
        for prefix, product in (("correct", a * b), ("error", a * b + offset)):
            rows.append(Problem(
                id=f"{source}-{prefix}", source_id=source, group_id=source,
                split=split, task="repair", question=question, answer=str(a * b + c),
                provenance=provenance, prefix=prefix, operands=(a, b, c),
                proposed_product=product,
                draft=f"First calculate the items in the boxes: {a} * {b} = {product}.",
            ))
        if len(rows) == 2 * count:
            break
    if len(rows) != 2 * count:
        raise ValueError("Requested count exceeds the selected repair source pool.")
    problems = tuple(rows)
    validate_problems(problems)
    return problems


def gsm8k_problems(
    *, count: int = 50, split: StudySplit = "pilot", seed: int = 2026,
    revision: str = "main", allow_downloads: bool = False,
) -> tuple[Problem, ...]:
    """Acquire and freeze a source-row sample from a full-SHA-pinned GSM8K snapshot.

    Acquisition always requires ``allow_downloads=True``, including pinned/cache
    requests; use local JSONL for guaranteed offline reuse. A floating ref is
    resolved through ``HfApi.dataset_info`` only after opt-in; full 40-hex SHAs
    bypass that metadata lookup. Optional dependencies are imported here only.

    Development/pilot partition train rows by a seed-independent stable hash;
    test uses the upstream test split. Sample without replacement before parsing
    labels, never replacing corrupt selected rows. Verify only the final ####
    label with the exact parser; save SHA, source row, upstream split, and role in
    provenance, never the reference solution. Missing packages, pool exhaustion,
    invalid labels/revisions, and upstream acquisition failures are explicit errors.
    """
    # Refuse invalid requests and implicit network access before importing either HF package.
    _request(count, split, seed)
    if not isinstance(revision, str) or not revision.strip() or len(revision) > 256 or revision != revision.strip():
        raise ValueError("revision must be a nonempty bounded dataset ref or full SHA.")
    if allow_downloads is not True:
        raise ValueError("GSM8K acquisition requires explicit allow_downloads=True.")
    sha = revision.lower()
    if re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise ValueError("GSM8K ref resolution requires huggingface_hub.") from error
        info = HfApi().dataset_info("openai/gsm8k", revision=revision)
        sha = info.sha
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
            raise ValueError("Dataset ref did not resolve to a full immutable SHA.")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ValueError("GSM8K acquisition requires the optional datasets package.") from error
    upstream = "test" if split == "test" else "train"
    dataset = load_dataset("openai/gsm8k", "main", revision=sha, split=upstream)
    # Bound source-row traversal even if an unexpected dataset ref has a different shape.
    if not 1 <= len(dataset) <= 100_000:
        raise ValueError("GSM8K source has an unsupported row count.")
    pool = [
        row for row in range(len(dataset))
        if split == "test" or _ROLES[int(_digest(f"gsm8k-v1:train:{row}"), 16) % 2] == split
    ]
    if count > len(pool):
        raise ValueError("Requested count exceeds the selected GSM8K source pool.")
    selected = random.Random(seed).sample(pool, count)
    rows: list[Problem] = []
    for index in selected:
        raw = dataset[index]
        # Unexpected source shapes are dataset errors, not incidental AttributeErrors.
        if not isinstance(raw, Mapping):
            raise ValueError(f"Invalid GSM8K record at source row {index}.")
        question, reference = raw.get("question"), raw.get("answer")
        if not isinstance(question, str) or not question.strip() or len(question) > _MAX_LINE_BYTES:
            raise ValueError(f"Invalid GSM8K question at source row {index}.")
        if not isinstance(reference, str) or len(reference) > _MAX_LINE_BYTES or "####" not in reference:
            raise ValueError(f"Missing/bounded GSM8K gold marker at source row {index}.")
        try:
            gold = numeric_answer(reference.rsplit("####", 1)[1])
        except ValueError as error:
            raise ValueError(f"Unsupported GSM8K gold at source row {index}.") from error
        source = f"gsm8k-{sha}-{upstream}-{index}"
        provenance = canonical_bytes({
            "dataset": "openai/gsm8k", "config": "main", "revision": sha,
            "source_split": upstream, "source_row": index, "role": split,
        }).decode("utf-8")
        rows.append(Problem(
            id=source, source_id=source, group_id=source, split=split, task="math",
            question=question, answer=gold, provenance=provenance,
        ))
    problems = tuple(rows)
    validate_problems(problems)
    return problems