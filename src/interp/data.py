# Load and inspect datasets here; no model execution or network access belongs here.
"""Exact-text dataset loading, stable content hashes, and conservative pairing checks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schemas import Example


def canonical_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically for hashing and persistence.

    Args:
        value: JSON-compatible primitives; convert Pydantic records with
            ``model_dump(mode='json')`` first.

    Returns:
        UTF-8 JSON with sorted keys, no formatting whitespace, and no NaN/inf.

    Raises:
        TypeError: An input is not JSON serializable.
        ValueError: An input contains a non-finite number.
    """
    # Preserve Unicode and reject nonstandard JSON floats instead of silently encoding them.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    """Return a complete SHA-256 digest for exactly the supplied bytes."""
    # Use a standard cryptographic content hash, not Python's randomized hash().
    return hashlib.sha256(value).hexdigest()


def dataset_digest(examples: Iterable[Example]) -> str:
    """Hash validated semantic dataset content in stable example-ID order.

    JSON indentation/key order and input row order do not change this digest.
    Prompt/candidate whitespace, labels, group assignments, and provenance do.
    """
    # Sort by immutable observation identity rather than file row order.
    ordered = sorted(examples, key=lambda example: example.id)
    # Include every validated field so label/provenance changes invalidate identity.
    return digest_bytes(canonical_bytes([example.model_dump(mode="json") for example in ordered]))


def validate_examples(examples: Iterable[Example], conditions: tuple[str, str]) -> tuple[Example, ...]:
    """Validate uniqueness, complete pairs, matched candidates, and grouped splits.

    Args:
        examples: Validated observations, possibly in arbitrary input order.
        conditions: Exactly the two predeclared experiment conditions.

    Returns:
        A tuple sorted by pair and declared condition order.

    Raises:
        ValueError: Empty data, duplicate IDs, incomplete/unmatched pairs, or any
            related source/group/pair appearing in more than one split.
    """
    # Freeze an iterable once so validation cannot consume it differently on later passes.
    rows = tuple(examples)
    # Refuse a run that would misleadingly report success without observations.
    if not rows:
        # Require an actual engineering or research dataset.
        raise ValueError("The dataset is empty.")
    # Track observation uniqueness explicitly.
    ids: set[str] = set()
    # Group variants by their declared pair identity.
    pairs: dict[str, list[Example]] = defaultdict(list)
    # Track splits for every lineage axis, not just example IDs.
    split_by_lineage: dict[tuple[str, str], str] = {}
    # Validate all rows before scheduling any model work.
    for row in rows:
        # Reject duplicate example IDs even when their text happens to match.
        if row.id in ids:
            # Stable trials require unique observation identities.
            raise ValueError(f"Duplicate example ID: {row.id}")
        # Record the now-validated observation identity.
        ids.add(row.id)
        # Reject undeclared conditions before they can create unmatched comparisons.
        if row.condition not in conditions:
            # Keep pair semantics exactly as configured.
            raise ValueError(f"Undeclared condition for example {row.id}.")
        # Collect this condition under its independent pair label.
        pairs[row.pair_id].append(row)
        # Check all declared related-observation labels against held-out splits.
        for axis in ("source_id", "group_id", "pair_id"):
            # Namespace labels by axis so identical unrelated strings do not collide.
            key = (axis, getattr(row, axis))
            # Remember the first split assignment, if this label is new.
            previous = split_by_lineage.setdefault(key, row.split)
            # Reject all later assignments to a different split.
            if previous != row.split:
                # Surface leakage instead of producing inflated held-out performance.
                raise ValueError(f"{axis} {key[1]} crosses dataset splits.")
    # Require a single comparable observation from each condition in every pair.
    for pair_id, pair in pairs.items():
        # Enforce complete two-condition coverage without duplicate conditions.
        if len(pair) != 2 or {row.condition for row in pair} != set(conditions):
            # Do not drop incomplete pairs silently during analysis.
            raise ValueError(f"Pair {pair_id} needs exactly one example per condition.")
        # Extract the two matched observations for lineage and target checks.
        left, right = pair
        # Match lineage, candidate texts/order, target, and split within each pair.
        if (left.source_id, left.group_id, left.split, left.candidates, left.target_candidate) != (
            right.source_id, right.group_id, right.split, right.candidates, right.target_candidate
        ):
            # Prevent changes to answer definitions from masquerading as prompt effects.
            raise ValueError(f"Pair {pair_id} has mismatched lineage, splits, or candidates.")
    # Build a stable map for condition order rather than alphabetically renaming baseline.
    condition_order = {condition: index for index, condition in enumerate(conditions)}
    # Return a predictable schedule that is invariant to JSONL row order.
    return tuple(sorted(rows, key=lambda row: (row.pair_id, condition_order[row.condition])))


def load_examples(path: Path, conditions: tuple[str, str]) -> tuple[Example, ...]:
    """Read UTF-8 JSONL, validate every nonempty row, then apply dataset-wide checks.

    Blank lines are ignored; invalid JSON/schema errors include their row number.
    No model or tokenizer is loaded by dataset inspection.
    """
    # Collect validated observations before the cross-row checks.
    examples: list[Example] = []
    # Open only the explicit local data file; remote loading is deliberately separate.
    with path.open("r", encoding="utf-8") as handle:
        # Track one-based row numbers for understandable validation errors.
        for number, line in enumerate(handle, start=1):
            # Ignore formatting-only blank lines without changing any prompt text.
            if not line.strip():
                # Continue to the next JSONL row.
                continue
            # Add row context without echoing potentially sensitive dataset content.
            try:
                # Parse and validate exact strings and all required lineage fields.
                examples.append(Example.model_validate_json(line))
            # Convert parser/schema failures into a location-only error.
            except ValueError as error:
                # Do not include raw prompts or exception representations in logs.
                raise ValueError(f"Invalid dataset row {number} in {path.name}.") from error
    # Apply pairing and leakage checks to the complete local dataset.
    return validate_examples(examples, conditions)
