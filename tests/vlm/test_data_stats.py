"""Offline synthetic invariants, not scientific model results."""
from __future__ import annotations

import hashlib
import json
import random

import pytest

from vlm_pilot import data
from vlm_pilot.stats import paired_summary


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    raw = data.generate_dataset(root, engineering=1, discovery=0, pilot_validation=0, confirmation=0)
    return root, raw


def test_determinism_factorial_and_disjoint_donors(tmp_path):
    settings = dict(engineering=8, discovery=0, pilot_validation=0, confirmation=0)
    state = random.getstate()
    first = data.generate_dataset(tmp_path / "one", **settings)
    assert random.getstate() == state
    assert first == data.generate_dataset(tmp_path / "two", **settings)
    assert first == data.load_dataset(tmp_path / "one")
    representatives = [data.representative_case(f) for f in first["families"]]
    assert len({(c["edge"], c["direction"], c["palette"]) for c in representatives}) == 8
    for family in first["families"]:
        assert len(family["images"]) == 4 and len(family["cases"]) == 12
        assert len({item["sha256"] for item in family["images"]}) == 4
        assert set(family["palettes"][0]).isdisjoint(family["palettes"][1])
        for item in family["images"]:
            png = (tmp_path / "one" / item["relative_path"]).read_bytes()
            assert png == (tmp_path / "two" / item["relative_path"]).read_bytes()
            assert hashlib.sha256(png).hexdigest() == item["sha256"]
        for case in family["cases"][:8]:
            assert case["target"] == (case["anchor"] + case["edge"]*case["direction"]) % 3
            assert case["gold"] == family["palettes"][case["palette"]][case["target"]]


@pytest.mark.parametrize("path,value", [
    (("families", 0, "nodes", 0, "x"), float("nan")),
    (("families", 0, "cases", 0, "gold"), "invalid"),
    (("families", 0, "cases", 1, "case_id"), lambda r: r["families"][0]["cases"][0]["case_id"]),
    (("families", 0, "images", 0, "relative_path"), "../escape.png"),
    (("families", 0, "images", 0, "sha256"), "0" * 64),
])
def test_metadata_tampering(dataset, path, value):
    root, raw = dataset
    container = raw
    for part in path[:-1]:
        container = container[part]
    container[path[-1]] = value(raw) if callable(value) else value
    (root / "dataset.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        data.load_dataset(root)


def _row(checkpoint, family, case, hit, **extra):
    return {"checkpoint": checkpoint, "family_id": str(family), "case_id": str(case),
            "target_hit": hit, **extra}


def test_known_pairs_family_correlation_missing_and_integrity():
    rows = [_row(c, f, 0, c == "after" and f % 2 == 0) for f in range(20)
            for c in ("before", "after")]
    result = paired_summary(rows, "before", "after", bootstrap_samples=200, seed=17)
    replicated = [dict(r, case_id=str(i)) for r in rows for i in range(8)]
    other = paired_summary(replicated, "before", "after", bootstrap_samples=200, seed=17)
    assert result["difference"] == .5 and result["paired_families"] == 20
    assert result["ci95"] == other["ci95"] and other["paired_rows"] == 160
    rows += [_row("before", 0, 1, True)]
    result = paired_summary(rows, "before", "after")
    assert result["state"] == "incomplete" and result["paired_families"] == 19
    assert result["missing_after"] == 1 and result["excluded_matched_rows"] == 1
    assert result["coverage"]["family_fraction"] == 19/20 and not result["equivalent"]
    with pytest.raises(ValueError):
        paired_summary(rows + [rows[0]], "before", "after")
    with pytest.raises(ValueError):
        paired_summary([dict(rows[0], scores={"red": float("inf")})], "before", "after")
    failed = paired_summary([_row("before", 0, 0, False),
                             _row("after", 0, 0, False, status="failed")], "before", "after")
    assert failed["state"] == "insufficient_data" and failed["coverage"]["failed_after"] == 1


def test_saturated_controls_require_conservative_equivalence():
    rows = [_row(c, f, 0, True) for f in range(200) for c in ("before", "after")]
    result = paired_summary(rows, "before", "after")
    assert result["difference"] == 0 and result["interval_method"] == "conservative_hoeffding"
    assert result["ci95"][0] < -.05 and result["ci95"][1] > .05
    assert not result["equivalent"]


def test_cued_edge_control_is_not_a_relation_following_question(dataset):
    _, raw = dataset
    family = raw["families"][0]
    for edge in (1, -1):
        case = data.make_case(family, edge=edge, direction=-1, palette=0, task="edge_readout")
        target = (family["anchor"] + 1) % 3 if edge == 1 else family["anchor"]
        assert case["target"] == target and "Look only at the arrow between" in case["question"]


def test_degenerate_difference_of_contrasts_is_not_an_exact_zero_interval():
    from vlm_pilot.stats import family_difference_interval

    result = family_difference_interval({str(i): 0. for i in range(60)})
    assert result["ci95"][0] < 0 < result["ci95"][1]
    assert result["interval_method"] == "conservative_hoeffding"