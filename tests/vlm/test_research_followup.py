"""Model-free prompt/gold and response-alias engineering diagnostic contracts."""
import importlib.util
import math
from pathlib import Path

from vlm_pilot.data import _family

SPEC = importlib.util.spec_from_file_location("research_followup_tests",
    Path(__file__).resolve().parents[2] / "scripts/vlm_research_followup.py")
followup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(followup)


def test_decomposition_keeps_images_and_correct_nodes():
    family = _family(17, "engineering", 0)
    for case in family["cases"][:8]:
        cases = followup.variants(family, case)
        assert len(cases) == 6
        shapes = [n["shape"] for n in family["nodes"]]
        for item in cases:
            assert item["image_id"] == case["image_id"]
            assert item["gold"] in item["candidates"]
            assert item["case_id"] != case["case_id"]
            if item["diagnostic"].endswith("shape"):
                assert item["gold"] == shapes[(case["anchor"]+case["edge"]*case["direction"]) % 3]
            elif item["diagnostic"] == "anchor_readout":
                assert item["gold"] == family["palettes"][case["palette"]][case["anchor"]]
            else:
                assert item["gold"] == case["gold"]


def test_response_alias_dedup_ties_and_logsumexp():
    result = {"scores": {"red": -1001., "Red": -1001., "blue": -1001., "Blue": -1001.},
              "candidate_token_ids": {"red": [1], "Red": [2], "blue": [3], "Blue": [3]}}
    grading = followup.classify_scores(result, ["red", "blue"])
    assert grading["lower"] is grading["title"] is None
    assert grading["combined"] == "red"
    assert grading["combined_scores"]["red"] == -1001. + math.log(2)
    assert grading["combined_scores"]["blue"] == -1001.