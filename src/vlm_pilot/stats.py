"""Paired-only whole-family contrasts: filter task/split/condition/site first.
Unseen plan coverage is unknown; incomplete families cannot establish equivalence.
"""
from __future__ import annotations

import math
from statistics import NormalDist, mean

_KEY = ("family_id", "case_id", "task", "phase", "condition", "operation", "site")


def _finite(value):
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Expected a finite number")
        return float(value)
    except OverflowError as exc:
        raise ValueError("Nonfinite value") from exc


def _validate(row):
    scores = row.get("scores", {})
    if type(scores) is not dict or any(type(k) is not str for k in scores):
        raise ValueError("scores must be a string-to-number mapping")
    numbers = [v for v in row.values() if isinstance(v, (int, float)) and type(v) is not bool]
    for value in (*numbers, *scores.values()):
        _finite(value)
    if "target_hit" in row and type(row["target_hit"]) is not bool:
        raise ValueError("target_hit must be boolean")
    if "elapsed_seconds" in row and _finite(row["elapsed_seconds"]) < 0:
        raise ValueError("Negative elapsed time")


def _interval(differences, bounded, samples, seed):
    if not differences:
        return None, "unavailable"
    center, count = mean(differences), len(differences)
    if len(set(differences)) > 1:
        import numpy as np  # Lazy and local RNG: importing this module is model-free.

        values, rng = np.asarray(differences, dtype=float), np.random.default_rng(seed)
        # O(families + samples) memory; divide first to avoid finite-score sum overflow.
        means = [_finite(float((rng.choice(values, count) / count).sum())) for _ in range(samples)]
        low, high = (_finite(float(v)) for v in np.quantile(means, [.025, .975]))
        if low < high:
            return [low, high], "family_percentile_bootstrap"
    if not bounded:
        return None, "unbounded_degenerate"
    # Hoeffding for independent family differences in [-1,1], at alpha=.05.
    epsilon = math.sqrt(2 * math.log(40) / count)
    return [max(-1., center-epsilon), min(1., center+epsilon)], "conservative_hoeffding"


def paired_summary(rows: list[dict], before: str, after: str, *, metric: str = "target_hit",
                   bootstrap_samples: int = 2000, seed: int = 0,
                   equivalence_margin: float = .05) -> dict:
    """Average all within-family rows, then bootstrap families (Boolean constants: Hoeffding).
    Optional labels default to ''; site=None; constant unbounded metrics get no CI.
    """
    if type(rows) is not list or not all(type(r) is dict for r in rows):
        raise ValueError("rows must be a list of dictionaries")
    valid_labels = all(type(c) is str and c for c in (before, after, metric))
    if not valid_labels or before == after:
        raise ValueError("Distinct checkpoints and a metric are required")
    for value, low, high in ((seed, 0, 2**63-1), (bootstrap_samples, 2, 100000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid seed/bootstrap_samples")
    margin = _finite(equivalence_margin)
    if not 0 <= margin <= 1:
        raise ValueError("Invalid equivalence margin")
    sides, splits, bounded = {before: {}, after: {}}, {}, True
    for row in rows:
        checkpoint = row.get("checkpoint")
        if type(checkpoint) is not str:
            raise ValueError("Invalid checkpoint")
        if checkpoint not in sides:
            continue
        _validate(row)
        key = tuple(row.get(k, None if k == "site" else "") for k in _KEY)
        if (any(type(v) is not str for v in key[:-1]) or not key[0] or not key[1]
                or key[-1] is not None and type(key[-1]) is not str):
            raise ValueError("Invalid pairing identity")
        split = row.get("split", "")
        if type(split) is not str or splits.setdefault(key[0], split) != split:
            raise ValueError("Family crosses splits")
        if key in sides[checkpoint]:
            raise ValueError("Duplicate pairing key")
        status = row.get("status", "completed")
        if type(status) is not str or not status:
            raise ValueError("Invalid status")
        value = None
        if status == "completed":
            raw = row.get(metric, row.get("scores", {}).get(metric))
            if metric == "target_hit" and type(raw) is not bool:
                raise ValueError("Completed target_hit must be boolean")
            value = float(raw) if type(raw) is bool else _finite(raw)
            bounded = bounded and type(raw) is bool
        sides[checkpoint][key] = value
    all_keys = set(sides[before]) | set(sides[after])
    completed = {c: {k for k, v in side.items() if v is not None} for c, side in sides.items()}
    matched = completed[before] & completed[after]
    groups = {}
    for key in all_keys:
        groups.setdefault(key[0], set()).add(key)
    families = sorted(f for f, keys in groups.items() if keys <= matched)
    incomplete = sorted(set(groups) - set(families))
    averages = {c: [mean(sides[c][k] for k in sorted(groups[f], key=repr))
                    for f in families] for c in sides}
    differences = [_finite(a-b) for a, b in zip(averages[after], averages[before], strict=True)]
    interval, method = _interval(differences, bounded, bootstrap_samples, seed)
    state = ("insufficient_data" if len(families) < 2 or interval is None
             else "incomplete" if incomplete else "complete")
    paired_rows = sum(len(groups[f]) for f in families)
    missing = {c: sorted(all_keys - completed[c], key=repr) for c in sides}
    return {"state": state, "before": before, "after": after, "metric": metric,
            "before_mean": mean(averages[before]) if families else None,
            "after_mean": mean(averages[after]) if families else None,
            "difference": mean(differences) if families else None, "ci95": interval,
            "interval_method": method, "equivalence_margin": margin,
            "equivalent": state == "complete" and -margin < interval[0] and interval[1] < margin,
            "paired_families": len(families), "paired_rows": paired_rows,
            "paired_only": {"families": len(families), "rows": paired_rows},
            "family_differences": dict(zip(families, differences, strict=True)),
            "matched_rows": len(matched), "excluded_matched_rows": len(matched)-paired_rows,
            "incomplete_families": incomplete,
            "missing_before": len(missing[before]), "missing_after": len(missing[after]),
            "missing_keys_before": [list(k) for k in missing[before]],
            "missing_keys_after": [list(k) for k in missing[after]],
            "coverage": {"observed_families": len(groups), "observed_pair_keys": len(all_keys),
                         "before_rows": len(sides[before]), "after_rows": len(sides[after]),
                         "family_fraction": len(families)/len(groups) if groups else 0.,
                         "paired_row_fraction": paired_rows/len(all_keys) if all_keys else 0.,
                         "failed_before": len(sides[before])-len(completed[before]),
                         "failed_after": len(sides[after])-len(completed[after]),
                         "unseen_plan_coverage": "unknown"}}


def family_difference_interval(differences: dict[str, float], *, bound: float = 2.,
                               samples: int = 2000, seed: int = 0, alpha: float = .05) -> dict:
    """CI for a preregistered family-level difference-of-contrasts in [-bound,bound]."""
    if not differences:
        return {"difference": None, "ci95": None, "families": 0, "state": "insufficient_data"}
    if type(samples) is not int or not 2 <= samples <= 100000:
        raise ValueError("Invalid bootstrap sample count")
    bound = _finite(bound)
    if not 0 < alpha < 1:
        raise ValueError("Invalid contrast interval alpha")
    values = [_finite(value) for value in differences.values()]
    if bound <= 0 or any(abs(value) > bound for value in values):
        raise ValueError("Family contrast exceeds its declared bound")
    scaled = [value / bound for value in values]
    interval, method = _interval(scaled, True, samples, seed)
    if alpha != .05:
        if len(set(values)) <= 1:
            epsilon = bound * math.sqrt(2 * math.log(2 / alpha) / len(values))
            interval = [(max(-bound, mean(values)-epsilon))/bound,
                        (min(bound, mean(values)+epsilon))/bound]
        else:
            import numpy as np

            rng, array = np.random.default_rng(seed), np.asarray(scaled)
            draws = [float(rng.choice(array, len(array)).mean()) for _ in range(samples)]
            interval = list(np.quantile(draws, [alpha/2, 1-alpha/2]))
    return {"difference": mean(values), "ci95": [value*bound for value in interval],
            "families": len(values), "interval_method": method,
            "state": "complete" if len(values) >= 2 else "insufficient_data",
            "family_differences": differences, "alpha": alpha}


def simulate_power(differences: dict[str, float], *, confirmation_families: int = 200,
                   effect: float = .15, samples: int = 2000, alpha: float = .05,
                   seed: int = 0) -> dict:
    """Planning simulation from centered discovery-family variation; not trial evidence.

    Resample family differences and assess a normal-approximation two-sided test.
    Degenerate discovery data cannot justify a zero-variance power guarantee.
    """
    import numpy as np

    if (type(confirmation_families) is not int or not 2 <= confirmation_families <= 10000
            or type(samples) is not int or not 2 <= samples <= 100000 or not 0 < alpha < 1):
        raise ValueError("Invalid power simulation settings")
    values = np.asarray([_finite(value) for value in differences.values()])
    effect = _finite(effect)
    if len(values) < 2 or float(values.std(ddof=1)) == 0:
        return {"state": "insufficient_discovery_variance", "power": None,
                "discovery_families": len(values)}
    centered = values - values.mean()
    rng, critical = np.random.default_rng(seed), NormalDist().inv_cdf(1-alpha/2)
    rejected = 0
    for _ in range(samples):
        draw = rng.choice(centered, confirmation_families) + effect
        standard_error = draw.std(ddof=1) / math.sqrt(confirmation_families)
        rejected += standard_error > 0 and abs(draw.mean()) > critical * standard_error
    return {"state": "planning_only", "power": rejected/samples,
            "discovery_families": len(values), "confirmation_families": confirmation_families,
            "assumed_effect": effect, "alpha": alpha, "samples": samples,
            "approximate_mde80": (critical + NormalDist().inv_cdf(.8))
            * float(values.std(ddof=1)) / math.sqrt(confirmation_families),
            "assumptions": "IID source families, discovery variance representative, normal test approximation"}