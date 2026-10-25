# Inspect saved observations and incomplete pairs here; no model work belongs in analysis.
"""Read-only descriptive tables for paired likelihood effects and failed attempts.

This release does not infer causality or estimate uncertainty from correlated tiny
fixture observations. Incomplete pairs are shown explicitly instead of dropped.
"""

from __future__ import annotations

from typing import Any

from .artifacts import RunSnapshot


def paired_effects(snapshot: RunSnapshot) -> list[dict[str, Any]]:
    """Return one row per planned pair/replicate, including incomplete comparisons.

    ``delta`` is condition-two margin minus condition-one margin. The conditions
    are read from the frozen spec; the function does not guess names or rank them.
    These are descriptive paired changes, not independent uncertainty estimates.
    """
    # Retrieve the predeclared order of the two experimental conditions.
    first, second = snapshot.metadata.spec.conditions
    # Index only validated completed observations by their logical pairing labels.
    complete = {(record.pair_id, record.replicate, record.condition): record for record in snapshot.records if record.status == "completed"}
    # Preserve one representative lineage/split snapshot per planned pair.
    pairs = {example.pair_id: example for example in snapshot.metadata.examples}
    # Collect all planned comparisons rather than only pairs that happened to finish.
    rows: list[dict[str, Any]] = []
    # Present pairs in stable order for reproducible inspection and plotting.
    for pair_id, example in sorted(pairs.items()):
        # Preserve independent replicate identity instead of averaging it away silently.
        for replicate in range(snapshot.metadata.spec.replicates):
            # Look up the first condition's successful output, if published.
            left = complete.get((pair_id, replicate, first))
            # Look up the second condition's successful output, if published.
            right = complete.get((pair_id, replicate, second))
            # Extract the retained raw sequence-likelihood margin when available.
            left_margin = left.result.target_logprob_margin if left is not None else None
            # Extract the matched changed-condition margin when available.
            right_margin = right.result.target_logprob_margin if right is not None else None
            # Mark missing evidence explicitly rather than substituting zero or excluding it.
            missing = [condition for condition, record in ((first, left), (second, right)) if record is None]
            # Add a descriptive row with its correlation/split labels retained.
            rows.append({"pair_id": pair_id, "source_id": example.source_id, "group_id": example.group_id, "split": example.split, "replicate": replicate, "first_condition": first, "second_condition": second, "first_margin": left_margin, "second_margin": right_margin, "delta": right_margin - left_margin if not missing else None, "status": "incomplete" if missing else "complete", "missing_conditions": missing})
    # Return ordinary rows without writing, importing plotting libraries, or loading models.
    return rows


def attempt_rows(snapshot: RunSnapshot) -> list[dict[str, Any]]:
    """Return a CPU-readable table of retained attempts and safe error categories."""
    # Preserve attempt status instead of hiding retries and interruptions behind final success.
    return [{"example_id": record.example_id, "condition": record.condition, "replicate": record.replicate, "attempt_id": str(record.attempt_id), "status": record.status, "seed": record.seed, "error_type": record.error_type, "finish_reason": record.result.generation.finish_reason if record.result and record.result.generation else None, "generated_text": record.result.generation.text if record.result and record.result.generation else None} for record in snapshot.records]
