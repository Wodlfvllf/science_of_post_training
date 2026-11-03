"""Stdlib, model-free contracts for the FIXED prospective cue/context schedule.

Unittest spies use tiny temporary checksummed image bytes, not rendered PNGs or
models. No torch, GPU, network, subprocess or shell is needed. The renderer's
_family/make_case and atomic JSON writer are the only project plumbing used;
their schemas may import Pydantic. CLI tests replace all loading/P0 with spies.
These tests are provided as source; writing this file does not execute them.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vlm_pilot.data import COLORS, SHAPES, _family, make_case

PATH = Path(__file__).absolute().parents[2] / "scripts/vlm_cue_context.py"
PROTOCOL = PATH.parents[1] / "configs/vlm/cue-context-development.json"


def load_module():
    spec = importlib.util.spec_from_file_location("cue_context_contracts", PATH)
    module = importlib.util.module_from_spec(spec)
    # No bytecode/cache output is needed even when these tests are later run.
    exec(compile(PATH.read_bytes(), str(PATH), "exec"), module.__dict__)
    return module


cue = load_module()


def generation(text, *, finish="eos"):
    return {"text": text, "token_ids": [11, 2] if finish == "eos" else [11] * 32,
            "finish_reason": finish, "parsed_answer": "deliberately_not_the_raw_answer",
            "correct": True, "valid": True,
            "settings": {"max_new_tokens": 32, "do_sample": False, "use_cache": False,
                         "eos_token_id": 2, "profile": "unrestricted_greedy_audit"}}


def passed_p0():
    check = {"passed": True, "finite": True, "argmax_match": True,
             "mismatches": 0, "max_abs": 0.0}
    return {"passed": True, "checks": {k: deepcopy(check) for k in
            ("native_embedding", "capture_only", "self_qk")},
            "candidate_suffixes": {c: [20 + i] for i, c in enumerate(COLORS + SHAPES)},
            "multi_token_candidate_checks": {}}


class TinyEngine:
    """Spy native protocol. Answers NEVER come from score winners/Engine parsing."""

    def __init__(self, *, answer=None, failure=None, malformed=None):
        self.answer, self.failure, self.malformed = answer, failure, malformed
        self.prepares, self.scores, self.generations, self.probes = [], [], [], []
        self._features, self._family = {}, None
        self.tokenizer = SimpleNamespace(eos_token_id=2)
        self.model_profile = {"test_double": True, "native_eos": 2}
        self.runtime_profile = {"test_double": True, "device": "cuda:0"}
        self.settings = SimpleNamespace(answer_prefix="Answer:", system_message="original system")
        self.peak_cached_images = 0

    def _prompt(self, question, has_image, profile):
        assert has_image is True and profile == "common"
        return "original system\n<image>\n" + question + "\nassistant\nAnswer:"

    def _fail(self, operation):
        if self.failure and self.failure[0] == operation:
            if self.failure[1] == len(getattr(self, {"prepare": "prepares", "score": "scores",
                                                   "generate": "generations"}[operation])):
                error = self.failure[2] if len(self.failure) > 2 else RuntimeError("first operation boom")
                raise error

    def prepare(self, root, family, case, *, profile):
        self.prepares.append(deepcopy(case))
        self._fail("prepare")
        if self._family != family["family_id"]:
            self._features.clear()
            self._family = family["family_id"]
        self._features[case["image_id"]] = "native image only"
        self.peak_cached_images = max(self.peak_cached_images, len(self._features))
        ids = (1, 5, sum(case["question"].encode()))
        return SimpleNamespace(case=case, prompt=self._prompt(case["question"], True, profile),
                               input_ids=ids, row=2, image_columns=(1,))

    def score(self, prepared):
        self.scores.append(prepared)
        self._fail("score")
        candidates = prepared.case["candidates"]
        result = {"scores": {c: -float(i) for i, c in enumerate(candidates)},
                  "prediction": candidates[0], "tied": False,
                  "candidate_token_ids": {c: [20 + i] for i, c in enumerate(candidates)},
                  "rendered_prompt": prepared.prompt, "input_ids": list(prepared.input_ids),
                  "row": prepared.row, "image_columns": list(prepared.image_columns), "routing": {}}
        if self.malformed:
            self.malformed("score", result)
        return result, {}

    def generate(self, prepared, budget):
        self.generations.append((prepared, budget))
        self._fail("generate")
        assert budget == 32
        result = (self.answer(prepared) if self.answer else
                  generation("triangle" if prepared.case["candidates"] == list(SHAPES) else "red"))
        if self.malformed:
            self.malformed("generate", result)
        return deepcopy(result)

    def preflight(self, root, family, case):
        self.probes.append(deepcopy(case))
        return passed_p0()


class CueContextContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dataset = {"families": [_family(31, "engineering", i) for i in range(24)]}
        # Original scenes/complete factors; checksummed arbitrary bytes avoid Pillow.
        for family in self.dataset["families"][8:16]:
            for item in family["images"]:
                payload = ("image-" + item["image_id"]).encode()
                path = self.root / item["relative_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                item["sha256"] = hashlib.sha256(payload).hexdigest()
        self.family = self.dataset["families"][8]
        self.protocol = deepcopy(cue.FIXED_PROTOCOL)
        self.plan = SimpleNamespace(plan_id=self.protocol["plan_id"],
                                    dataset_sha256=self.protocol["dataset_sha256"])

    def row(self, *, nonoracle=False, kind="cue_context"):
        rows = cue.make_schedule(self.family)
        return next(r for r in rows if r["kind"] == kind and
                    (not nonoracle or r["cue_node"] != r["destination_node"]))

    def test_import_is_stdlib_inert_even_with_models_and_filesystem_reads_blocked(self):
        payload = PATH.read_bytes()
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "transformers", "numpy", "PIL", "filelock", "vlm_pilot"}:
                raise AssertionError("Inert import attempted dependency: " + name)
            return original_import(name, *args, **kwargs)

        namespace = {"__file__": str(PATH), "__name__": "inert_cue_import"}
        with patch("builtins.__import__", side_effect=guarded), patch.object(Path, "open", side_effect=AssertionError), \
                patch.object(Path, "stat", side_effect=AssertionError):
            exec(compile(payload, str(PATH), "exec"), namespace)
            with self.assertRaises(SystemExit) as exit_info:
                namespace["_parser"]().parse_args(["--help"])
            self.assertEqual(exit_info.exception.code, 0)
        self.assertIsNone(namespace["_HELPER"])

    def test_committed_protocol_exact_concrete_pins_and_counts(self):
        protocol = cue.read_protocol(PROTOCOL)
        self.assertEqual(protocol, self.protocol)
        self.assertEqual(cue.validate_protocol(protocol, self.plan), protocol)
        self.assertEqual(protocol["kind"], "cue_context_development_v1")
        self.assertEqual(protocol["prior_use"], "prior_engineering_used_not_confirmation")
        counts = cue.stage_counts()
        self.assertEqual((counts["cue_context"], counts["task"], counts["total"]), (576, 256, 832))
        self.assertEqual(cue.stage_counts(1)["total"], 104)
        self.assertEqual(protocol["generations_both_models"], 1664)

    def test_strict_protocol_rejects_every_override_extra_vocab_cap_split_and_pin(self):
        patches = [{"schema_version": True}, {"family_count": 8.0}, {"family_offset": 0},
                   {"family_indices": list(range(4, 12))}, {"split": "confirmation"},
                   {"split": "pilot_validation"}, {"cues": ["triangle"]},
                   {"cues": ["square", "circle", "triangle"]}, {"contexts": ["lookup_only"]},
                   {"vocabularies": {"COLORS": ["red"], "SHAPES": list(SHAPES)}},
                   {"seed": 123}, {"resume": True}, {"probe_only": True}, {"allow_downloads": True},
                   {"plan_id": "a" * 64}, {"dataset_sha256": "b" * 64},
                   {"primary": "target_accuracy"}, {"generation": {"max_new_tokens": 2}},
                   {"generation": {**self.protocol["generation"], "max_new_tokens": 64}},
                   {"generation": {**self.protocol["generation"], "max_new_tokens": 32.0}},
                   {"generation": {**self.protocol["generation"], "do_sample": 0}},
                   {"competing_anchor_rule": "gold_plus_one"}, {"checkpoints": ["8b-cpt"]}]
        for change in patches:
            with self.subTest(change=change), self.assertRaises(ValueError):
                cue.validate_protocol({**self.protocol, **change}, self.plan)
        for key in ("plan_id", "dataset_sha256"):
            plan = deepcopy(self.plan)
            setattr(plan, key, "c" * 64)
            with self.subTest(key=key), self.assertRaises(ValueError):
                cue.validate_protocol(self.protocol, plan)
        for count in (True, 0, 9, 8.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                cue.stage_counts(count)

    def test_duplicate_nested_json_nonfinite_and_protocol_size_fail_closed(self):
        raw = PROTOCOL.read_text()
        malformed = [raw.replace('"family_count": 8', '"family_count": 8, "family_count": 8'),
                     raw.replace('"max_new_tokens": 32', '"max_new_tokens": 32, "max_new_tokens": 32'),
                     raw.replace('"family_count": 8', '"family_count": NaN'),
                     raw.replace('"family_count": 8', '"family_count": 1e999'), " ", "x" * (1024**2 + 1)]
        path = self.root / "protocol.json"
        for text in malformed:
            path.write_text(text)
            with self.subTest(text=text[:50]), self.assertRaises(ValueError):
                cue.read_protocol(path)

    def test_select_fixed_eight_arbitrary_prior_engineering_no_outcome_filters(self):
        original = deepcopy(self.dataset)
        dataset = deepcopy(original)
        for index, split in enumerate(("discovery", "pilot_validation", "confirmation"), 24):
            dataset["families"].append(_family(31, split, index))
        selected = cue.select_families(self.protocol, dataset)
        self.assertEqual([f["index"] for f in selected], list(range(8, 16)))
        self.assertEqual(len({c["case_id"] for f in selected for c in cue.selected_cases(f)}), 64)
        self.assertEqual(self.dataset, original)
        for families in (original["families"][1:], original["families"][::-1],
                         [*original["families"][:-1], original["families"][0]]):
            with self.subTest(families=len(families)), self.assertRaises(ValueError):
                cue.select_families(self.protocol, {"families": families})

    def test_scene_bounds_and_complete_canonical_eight_cases(self):
        cases = cue.selected_cases(self.family)
        self.assertEqual([(c["edge"], c["direction"], c["palette"]) for c in cases], list(cue.FACTORS))
        self.assertEqual(len({c["image_id"] for c in cases}), 4)
        for change in ({"index": 7}, {"index": 16}, {"index": True}, {"anchor": -1},
                       {"anchor": 3}, {"anchor": False}, {"split": "confirmation"},
                       {"palettes": [["red"] * 3, ["blue"] * 3]}, {"nodes": []}):
            family = {**deepcopy(self.family), **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                cue.selected_cases(family)
        family = deepcopy(self.family)
        family["cases"][1] = deepcopy(family["cases"][0])
        with self.assertRaises(ValueError):
            cue.selected_cases(family)
        family = deepcopy(self.family)
        family["cases"][0]["direction"] = True
        with self.assertRaises(ValueError):
            cue.selected_cases(family)

    def test_fixed_vocabulary_not_family_node_order_and_never_gold_chosen(self):
        family = deepcopy(self.family)
        for node, shape in zip(family["nodes"], reversed(SHAPES), strict=True):
            node["shape"] = shape
        family["cases"] = [make_case(family, edge=e, direction=d, palette=p) for e, d, p in cue.FACTORS]
        rows = cue.make_schedule(family)
        self.assertEqual(len(rows), 104)
        for factor in cue.FACTORS:
            context = [r for r in rows if r["kind"] == "cue_context" and
                       (r["edge"], r["direction"], r["palette"]) == factor]
            self.assertEqual([r["cue"] for r in context], [s for s in SHAPES for _ in range(3)])
            self.assertEqual(set(r["cue_node"] for r in context), {0, 1, 2})
            self.assertEqual(sum(r["cue_node"] == r["destination_node"] for r in context), 3)
            self.assertTrue(all(r["oracle_cue_after_test"] is None for r in context))
        with patch.object(cue._helper(), "_vocabulary", return_value=(COLORS[:-1], SHAPES)):
            with self.assertRaises(ValueError):
                cue.make_schedule(family)

    def test_all_104_stages_exact_templates_same_final_and_same_images(self):
        rows = cue.make_schedule(self.family)
        self.assertEqual(len({r["case_id"] for r in rows}), 104)
        self.assertEqual(sum(r["kind"] == "cue_context" for r in rows), 72)
        self.assertEqual(sum(r["kind"] == "task" for r in rows), 32)
        for original in cue.selected_cases(self.family):
            factor = (original["edge"], original["direction"], original["palette"])
            items = [r for r in rows if (r["edge"], r["direction"], r["palette"]) == factor]
            self.assertEqual(len(items), 13)
            self.assertTrue(all(r["image_id"] == original["image_id"] for r in items))
            for shape in SHAPES:
                context = [r for r in items if r["cue"] == shape]
                final = f"Now inspect the {shape} only. What color is its chip? Reply with one color name."
                self.assertEqual(context[0]["case"]["question"], final)
                for row in context:
                    expected = final
                    if row["cue_context"] != "lookup_only":
                        anchor_shape = self.family["nodes"][row["context_anchor"]]["shape"]
                        traversal = "forward" if row["direction"] == 1 else "backward"
                        expected = f"Starting at the {anchor_shape}, follow one arrow {traversal}. " + final
                    self.assertEqual(row["case"]["question"], expected)
                    self.assertEqual(row["case"]["candidates"], list(COLORS))
                    self.assertEqual(row["case"]["gold"], original["gold"])
                    self.assertTrue(all(color not in expected for color in COLORS))
            for row in (r for r in items if r["kind"] == "task"):
                answer = row["task_answer"]
                anchor = self.family["nodes"][original["anchor"]]["shape"]
                traversal = "forward" if original["direction"] == 1 else "backward"
                expected = (f"Starting at the {anchor}, follow one arrow {traversal}. What is the destination {answer}? Reply with one {answer} name."
                            if row["task_frame"] == 1 else
                            f"From the {anchor}, move one arrow {traversal}. Name the {answer} at the destination. Reply with one {answer} name.")
                self.assertEqual(row["case"]["question"], expected)
                expected_gold = self.family["nodes"][original["target"]]["shape"] if answer == "shape" else original["gold"]
                self.assertEqual(row["original_gold"], expected_gold)
                self.assertEqual(row["case"]["candidates"], list(SHAPES if answer == "shape" else COLORS))

    def test_competing_anchor_fixed_independent_of_direction_edge_palette_or_destination(self):
        rows = [r for r in cue.make_schedule(self.family) if r["cue_context"] == "competing_anchor"]
        competing = (self.family["anchor"] + 1) % 3
        self.assertEqual({r["context_anchor"] for r in rows}, {competing})
        self.assertEqual({r["direction"] for r in rows}, {1, -1})
        # Do NOT reject coincidences: competing anchor can equal original destination.
        self.assertTrue(any(r["context_anchor"] == r["destination_node"] for r in rows))
        self.assertTrue(any(r["context_anchor"] != r["destination_node"] for r in rows))
        for r in rows:
            self.assertEqual(r["anchor_color"], self.family["palettes"][r["palette"]][competing])
            self.assertEqual(r["original_true_anchor_color"], self.family["palettes"][r["palette"]][self.family["anchor"]])
        # Pure prompt function has no gold/destination input or answer-dependent branch.
        for d in (1, -1):
            self.assertEqual(cue.cue_question("triangle", d, "circle", "lookup_only"),
                             cue.FINAL_INSTRUCTION.format(cue="circle"))
        for invalid in (("hexagon", 1, "circle", "lookup_only"), ("triangle", True, "circle", "true_relation"),
                        ("triangle", 1, "red", "lookup_only")):
            with self.assertRaises(ValueError):
                cue.cue_question(*invalid)

    def test_exact_adjacent_helper_run_stage_all_vocab_32_fresh_prepares_one_image_cache(self):
        engine = TinyEngine()
        progress = []
        h = cue._helper()
        self.assertEqual(Path(h.__file__), PATH.with_name("vlm_generated_decomposition.py"))
        self.assertIs(h._image_reference, cue._image_reference)
        with patch.object(h, "_run_stage", wraps=h._run_stage) as run_stage, \
                patch.object(h, "selected_cases", side_effect=AssertionError("Hardcoded e1p0 selection forbidden")):
            record = cue.evaluate_family(engine, self.root, self.family, self.root / "family.json",
                progress=lambda stage, **details: progress.append((stage, details)))
        self.assertEqual(run_stage.call_count, 104)
        self.assertTrue(all(call.args[5] == 32 for call in run_stage.call_args_list))
        self.assertEqual((len(engine.prepares), len(engine.scores), len(engine.generations)), (104, 104, 104))
        self.assertEqual(len({c["case_id"] for c in engine.prepares}), 104)
        self.assertEqual(engine.peak_cached_images, 1)
        self.assertEqual(engine._features, {})
        self.assertEqual(record["status"], "completed")
        self.assertTrue(record["sealed"])
        self.assertEqual([v["operation"] for _, v in progress], ["prepare", "score", "generate"] * 104)
        for row, scored, (generated, budget) in zip(record["rows"], engine.scores, engine.generations, strict=True):
            self.assertIs(scored, generated)
            self.assertEqual(budget, 32)
            self.assertTrue(row["input"]["rendered_prompt"].endswith("Answer:"))
            self.assertEqual(row["score"]["input_ids"], row["input"]["input_ids"])
            self.assertEqual(row["input"]["image"]["image_id"], row["image_id"])

    def test_dual_grade_cue_following_wrong_original_and_original_hit_wrong_cue(self):
        for which in ("cue_gold", "original_gold"):
            row = self.row(nonoracle=True)
            engine = TinyEngine(answer=lambda prepared, row=row, which=which: generation(row[which].upper() + "."))
            cue.evaluate_stage(engine, self.root, self.family, row)
            self.assertTrue(row["answer_valid"])
            self.assertEqual(row["original_correct"], which == "original_gold")
            self.assertEqual(row["cue_correct"], which == "cue_gold")
            self.assertFalse(row["oracle_cue_after_test"])
            self.assertEqual(row["cue_intervention_label"], "all_node_intervention_not_autonomous_rescue")
        row = next(r for r in cue.make_schedule(self.family) if r["cue_node"] == r["destination_node"])
        engine = TinyEngine(answer=lambda prepared: generation(row["cue_gold"]))
        cue.evaluate_stage(engine, self.root, self.family, row)
        self.assertTrue(row["oracle_cue_after_test"])
        self.assertTrue(row["cue_correct"] and row["original_correct"])

    def test_invalid_and_truncated_strict_delivery_no_parser_score_or_oracle_fallback(self):
        outputs = [generation("The color is red."), generation("<think>red</think>red"),
                   generation("red.."), generation("red!"), generation("red blue"),
                   generation("Answer: red"), generation(""), generation("red", finish="length")]
        for output in outputs:
            row = self.row()
            engine = TinyEngine(answer=lambda prepared, output=output: output)
            cue.evaluate_stage(engine, self.root, self.family, row)
            with self.subTest(output=output["text"]):
                self.assertEqual(row["status"], "invalid_answer")
                self.assertEqual(row["errors"], [])
                self.assertIsNone(row["parsed_answer"])
                self.assertFalse(row["cue_correct"] or row["original_correct"])
                self.assertEqual(row["generation"], output)
        # Invalid stages do not stop or gate the fixed next cue/task.
        engine = TinyEngine(answer=lambda prepared: generation("red", finish="length"))
        record = cue.evaluate_family(engine, self.root, self.family, self.root / "invalid-family.json")
        self.assertEqual(len(engine.generations), 104)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["counts"]["counts"]["invalid_answer"], 104)

    def test_first_operation_error_stops_immediately_seals_104_no_retry(self):
        for operation, expected in (("prepare", (1, 0, 0)), ("score", (1, 1, 0)), ("generate", (1, 1, 1))):
            engine = TinyEngine(failure=(operation, 1))
            path = self.root / f"failed-{operation}.json"
            with self.subTest(operation=operation), self.assertRaises(RuntimeError):
                cue.evaluate_family(engine, self.root, self.family, path)
            record = json.loads(path.read_text())
            self.assertEqual((len(engine.prepares), len(engine.scores), len(engine.generations)), expected)
            self.assertEqual(len(record["rows"]), 104)
            self.assertEqual(record["rows"][0]["status"], "failed")
            self.assertEqual(record["rows"][0]["errors"][0]["operation"], operation)
            self.assertEqual(sum(r["status"] == "not_started" for r in record["rows"]), 103)
            self.assertEqual(record["status"], "unfinished")
            self.assertEqual(engine._features, {})
            with self.assertRaises(FileExistsError):
                cue.evaluate_family(engine, self.root, self.family, path)
            self.assertEqual((len(engine.prepares), len(engine.scores), len(engine.generations)), expected)

    def test_malformed_scores_and_native_eos_settings_fail_before_later_operations(self):
        mutations = [("score", lambda r: r["scores"].pop("orange")),
                     ("score", lambda r: r["scores"].update(red=float("nan"))),
                     ("score", lambda r: r.update(prediction="orange")),
                     ("score", lambda r: r.update(routing={"hidden_hook": True})),
                     ("generate", lambda r: r["settings"].update(max_new_tokens=64)),
                     ("generate", lambda r: r["settings"].update(max_new_tokens=32.0)),
                     ("generate", lambda r: r["settings"].update(use_cache=True)),
                     ("generate", lambda r: r["settings"].update(do_sample=True)),
                     ("generate", lambda r: r["settings"].update(eos_token_id=99)),
                     ("generate", lambda r: r["settings"].update(eos_token_id=2.0)),
                     ("generate", lambda r: r["settings"].update(profile="custom_decoder")),
                     ("generate", lambda r: r.update(token_ids=[11], finish_reason="eos")),
                     ("generate", lambda r: r.update(token_ids=[2, 11, 2])),
                     ("generate", lambda r: r.update(token_ids=[11], finish_reason="length"))]
        for operation, mutate in mutations:
            row = self.row()
            engine = TinyEngine(malformed=lambda op, result, operation=operation, mutate=mutate:
                                mutate(result) if op == operation else None)
            with self.subTest(operation=operation), self.assertRaises(RuntimeError):
                cue.evaluate_stage(engine, self.root, self.family, row)
            self.assertEqual(len(engine.generations), 0 if operation == "score" else 1)
            self.assertEqual(row["status"], "failed")
            self.assertFalse(row["original_correct"] or row["cue_correct"])
            self.assertIsNotNone(row["score"] if operation == "score" else row["generation"])

    def test_progress_failure_at_score_stops_before_score_or_generate_and_publishes(self):
        engine = TinyEngine()

        def progress(stage, **details):
            if details["operation"] == "score":
                raise OSError("progress sink closed")

        path = self.root / "failed-progress.json"
        with self.assertRaises(RuntimeError):
            cue.evaluate_family(engine, self.root, self.family, path, progress=progress)
        self.assertEqual((len(engine.prepares), len(engine.scores), len(engine.generations)), (1, 0, 0))
        record = json.loads(path.read_text())
        self.assertEqual(record["rows"][0]["errors"][0]["operation"], "progress")
        self.assertEqual(record["counts"]["counts"]["not_started"], 103)

    def test_interrupt_publishes_partial_family_and_does_not_continue(self):
        engine = TinyEngine(failure=("generate", 2, KeyboardInterrupt("operator stopped")))
        path = self.root / "interrupted-family.json"
        with self.assertRaises(KeyboardInterrupt):
            cue.evaluate_family(engine, self.root, self.family, path)
        record = json.loads(path.read_text())
        self.assertEqual(len(engine.generations), 2)
        self.assertEqual(record["failure"]["type"], "KeyboardInterrupt")
        self.assertEqual(record["counts"]["counts"]["not_started"], 102)
        self.assertTrue(record["sealed"])

    def test_every_factor_image_checked_and_cached_steps_recheck_sha_and_bound(self):
        for case in cue.selected_cases(self.family):
            image = cue._image_reference(self.root, self.family, case)
            self.assertEqual(image["image_id"], case["image_id"])
            self.assertTrue(image["verified_before_prepare"])
        row = self.row()
        item = next(i for i in self.family["images"] if i["image_id"] == row["image_id"])
        image_path = self.root / item["relative_path"]
        engine = TinyEngine()
        cue.evaluate_stage(engine, self.root, self.family, row)
        image_path.write_bytes(b"changed while features cached")
        second = cue.make_schedule(self.family)[1]
        with self.assertRaises(RuntimeError):
            cue.evaluate_stage(engine, self.root, self.family, second)
        self.assertEqual(len(engine.prepares), 1)
        self.assertIn("checksum", second["errors"][0]["message"])
        image_path.write_bytes(b"x" * (8 * 1024**2 + 1))
        with self.assertRaises(ValueError):
            cue._image_reference(self.root, self.family, row["case"])
        for change in ({"direction": True}, {"palette": True}, {"edge": True}):
            with self.assertRaises(ValueError):
                cue._image_reference(self.root, self.family, {**row["case"], **change})
        family = deepcopy(self.family)
        matched = next(i for i in family["images"] if i["image_id"] == row["image_id"])
        matched["relative_path"] = "../outside.png"
        with self.assertRaises(ValueError):
            cue._image_reference(self.root, family, row["case"])

    def records(self, *, complete=False, interference=False):
        records = [cue._family_record(f) for f in self.dataset["families"][8:16]]
        if complete:
            for record in records:
                for row in record["rows"]:
                    answer = row["cue_gold"] if row["kind"] == "cue_context" else row["original_gold"]
                    if interference and row["cue_context"] in cue.CONTEXTS[1:]:
                        answer = next(c for c in COLORS if c != row["cue_gold"])
                    row.update(status="completed", attempted=True, generation=generation(answer))
        return records

    def test_full_partial_root_denominators_and_reconstructed_raw_flags(self):
        records = self.records()
        # Forged old flags must not become delivered answers or oracle rescue.
        records[0]["rows"][0].update(original_correct=True, cue_correct=True, parsed_answer="red")
        report = cue.summarize(records, checkpoint="8b-gspo", p0_passed=True)
        self.assertEqual((report["n_family"], report["n"]), (8, 832))
        self.assertEqual(report["counts"]["not_started"], 832)
        self.assertEqual(report["counts"]["target_hit"], 0)
        self.assertEqual(report["counts"]["cue_following"], 0)
        self.assertTrue(report["unfinished"])
        self.assertFalse(report["primary_interpretable_as_complete_schedule"])
        self.assertTrue(all(c["n"] == 192 for c in report["contexts"].values()))
        self.assertTrue(all(c["n"] == 64 for c in report["tasks"].values()))
        self.assertTrue(all(c["n"] == 64 for c in report["cue_context_groups"]))
        self.assertEqual(len(report["factor_cue_context_groups"]), 72)
        self.assertTrue(all(c["n"] == 8 for c in report["factor_cue_context_groups"]))
        with self.assertRaises(ValueError):
            cue.summarize(records[:7], checkpoint="8b-gspo")
        records[0]["rows"].pop()
        with self.assertRaises(ValueError):
            cue.summarize(records, checkpoint="8b-gspo")

    def test_primary_is_equal_family_context_minus_lookup_not_rescue_or_stage_claim(self):
        records = self.records(complete=True, interference=True)
        report = cue.summarize(records, checkpoint="8b-mpo", p0_passed=True)
        self.assertEqual(report["primary_value"], -1)
        self.assertEqual(report["cue_compliance_change"], {"true_relation": -1, "competing_anchor": -1})
        self.assertEqual(report["contexts"]["lookup_only"]["cue_following"], 192)
        self.assertEqual(report["contexts"]["true_relation"]["cue_following"], 0)
        self.assertTrue(all(f["primary_value"] == -1 for f in report["paired_families"]))
        self.assertTrue(report["execution_complete"])
        self.assertEqual(report["hypothesis_decision"], "not_automated")
        self.assertEqual(report["confirmation"], "untouched")
        self.assertIn("context_interference", report["primary_interpretation"])

    def test_relational_shape_exact_all_eight_reversal_palette_and_two_frames(self):
        records = self.records(complete=True)
        report = cue.summarize(records, checkpoint="8b-gspo", p0_passed=True)
        self.assertEqual(report["primary_value"], 0)
        for family in report["paired_families"]:
            shape = family["shape_relational"]
            self.assertEqual(shape["paired_frames_n"], 8)
            self.assertEqual(shape["paired_frames_both_exact"], 8)
            for frame in shape["frames"]:
                self.assertEqual(frame["n_outcomes"], 8)
                self.assertTrue(frame["all_eight_exact"])
                for kind in ("direction_reversal", "edge_reversal", "palette"):
                    self.assertEqual(frame["groups"][kind]["n"], 4)
                    self.assertEqual(frame["groups"][kind]["all_exact"], 4)
                for kind in ("joint_direction_reversal_palette", "joint_edge_reversal_palette"):
                    self.assertEqual(frame["groups"][kind]["n"], 2)
                    self.assertEqual(frame["groups"][kind]["all_exact"], 2)
        wrong = next(r for r in records[0]["rows"] if r["task_answer"] == "shape" and r["task_frame"] == 1)
        wrong["generation"] = generation(next(s for s in SHAPES if s != wrong["original_gold"]))
        family = cue.summarize(records, checkpoint="8b-gspo", p0_passed=True)["paired_families"][0]
        self.assertFalse(family["shape_relational"]["frames"][0]["all_eight_exact"])
        self.assertEqual(family["shape_relational"]["frames"][0]["exact"], 7)
        self.assertEqual(family["shape_relational"]["paired_frames_both_exact"], 7)

    def test_summary_rejects_eval_gold_schedule_or_family_identity_tampering(self):
        for change in ("gold", "case", "family", "schedule"):
            records = self.records()
            if change == "gold":
                records[0]["rows"][0]["cue_gold"] = "NOT_A_COLOR"
            elif change == "case":
                records[0]["rows"][0]["case"]["question"] += " supplied answer red"
            elif change == "family":
                records[0]["family"]["anchor"] = 0
            else:
                records[0]["schedule_sha256"] = "0" * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                cue.summarize(records, checkpoint="8b-gspo")

    def test_cli_requires_protocol_cached_only_cuda_and_rejects_resume_budget_probe(self):
        args = ["--protocol", "protocol", "--plan", "plan", "--checkpoint", "8b-mpo",
                "--output", "output", "--device", "cuda:0", "--cached-only"]
        parsed = cue._parser().parse_args(args)
        self.assertTrue(parsed.cached_only)
        for bad in (args[:-1], args + ["--resume"], args + ["--probe-only"], args + ["--families", "1"],
                    args + ["--budget", "2"], [*args[:9], "cpu", "--cached-only"]):
            with self.subTest(args=bad), self.assertRaises(SystemExit):
                cue._parser().parse_args(bad)
        for device in ("cpu", "cuda", "auto", "cuda:-1", "cuda:00"):
            with self.assertRaises(argparse.ArgumentTypeError):
                cue._device(device)

    def cli_environment(self, engine, *, load_error=None):
        """Mock execution boundaries only; retain real schedule/evaluator/publication."""
        from vlm_pilot import data as data_module
        from vlm_pilot import io as io_module
        from vlm_pilot import plan as plan_module

        stack = ExitStack()
        self.addCleanup(stack.close)
        inference = SimpleNamespace(answer_prefix="Answer:", model_dump=lambda **kw: {
            "answer_prefix": "Answer:", "dtype": "float32", "system_message": "original system"})
        self.plan.spec = SimpleNamespace(dataset=self.root, inference=inference)
        self.plan.source_sha256 = "d" * 64
        self.plan.model_dump = lambda **kw: {"plan_id": self.plan.plan_id,
            "dataset_sha256": self.plan.dataset_sha256, "source_sha256": self.plan.source_sha256}
        pair = {name: SimpleNamespace(name=name, family="8b", backend="internvl", stage=name[3:], **pin)
                for name, pin in cue.CHECKPOINT_PINS.items()}
        stack.enter_context(patch.object(cue, "REPO", self.root))
        read_plan = stack.enter_context(patch.object(plan_module, "read_plan", return_value=self.plan))
        dataset_load = stack.enter_context(patch.object(data_module, "load_dataset", return_value=self.dataset))
        stack.enter_context(patch.object(io_module, "identity", return_value=self.plan.dataset_sha256))
        stack.enter_context(patch.object(plan_module, "checkpoint_for", side_effect=lambda plan, name: pair[name]))
        lock = Mock()
        lock.return_value.__enter__ = Mock(return_value=None)
        lock.return_value.__exit__ = Mock(return_value=False)
        stack.enter_context(patch.dict(sys.modules, {"filelock": SimpleNamespace(FileLock=lock)}))
        constructor = Mock(side_effect=load_error) if load_error else Mock(return_value=engine)
        stack.enter_context(patch.object(cue._helper(), "cached_engine_class", return_value=constructor))
        stack.enter_context(patch("builtins.print"))
        protocol = self.root / "protocol.json"
        protocol.write_bytes(PROTOCOL.read_bytes())
        plan = self.root / "plan.json"
        plan.write_text('{"original_plan_test_double": true}')
        output = self.root / "runs/vlm-pilot/report-cue-test"
        args = ["--protocol", str(protocol), "--plan", str(plan), "--checkpoint", "8b-gspo",
                "--output", str(output), "--device", "cuda:0", "--cached-only"]
        return SimpleNamespace(args=args, output=output, lock=lock, constructor=constructor,
                               inference=inference, pair=pair, read_plan=read_plan, dataset_load=dataset_load)

    def test_cli_all_832_common_native_calls_p0_both_pins_actual_sources_and_sealed_json(self):
        engine = TinyEngine()
        env = self.cli_environment(engine)
        self.assertEqual(cue.main(env.args), 0)
        self.assertEqual((len(engine.prepares), len(engine.scores), len(engine.generations)), (832, 832, 832))
        self.assertEqual(len(engine.probes), 1)
        self.assertEqual(engine.probes[0]["candidates"], list(COLORS + SHAPES))
        self.assertEqual(engine.probes[0]["task"], "binding")
        self.assertEqual(engine.probes[0]["question"], cue.selected_cases(self.family)[0]["question"])
        env.read_plan.assert_called_once_with(self.root / "plan.json", check_source=True)
        env.dataset_load.assert_called_once_with(self.root, verify_images=False)
        self.assertIs(env.constructor.call_args.args[1], env.inference)
        self.assertEqual(env.constructor.call_args.kwargs, {"device": "cuda:0", "allow_downloads": False})
        self.assertEqual(env.lock.call_args.args[0], str(Path(tempfile.gettempdir()) / "interp-vlm-cuda-0.lock"))
        self.assertEqual(env.lock.call_args.kwargs, {"timeout": 0})
        metadata = json.loads((env.output / "metadata.json").read_text())
        report = json.loads((env.output / "report.json").read_text())
        progress = json.loads((env.output / "progress.json").read_text())
        self.assertEqual(metadata["protocol_file_sha256"], hashlib.sha256(PROTOCOL.read_bytes()).hexdigest())
        self.assertEqual(metadata["original_inference"], metadata["effective_inference"])
        self.assertEqual(metadata["model"], engine.model_profile)
        self.assertEqual(metadata["runtime"], engine.runtime_profile)
        self.assertEqual(report["metadata"], metadata)
        self.assertEqual(report["metadata_sha256"], cue._sha(cue._canonical(metadata)))
        for relative, source in metadata["source_files"].items():
            actual = (PATH.parents[1] / relative).read_bytes()
            self.assertEqual(source["text"].encode(), actual)
            self.assertEqual(source["sha256"], hashlib.sha256(actual).hexdigest())
            self.assertEqual(source["size_bytes"], len(actual))
        self.assertEqual((progress["completed"], progress["total"]), (832, 832))
        self.assertTrue(progress["sealed"])
        self.assertFalse(progress["unfinished"])
        for index in range(8, 16):
            family = json.loads((env.output / f"family-{index:04d}.json").read_text())
            self.assertEqual(len(family["rows"]), 104)
            self.assertTrue(family["sealed"])
            self.assertIn("source_files", family["metadata"])
        self.assertEqual(len(report["family_artifacts"]), 8)
        for receipt in report["family_artifacts"]:
            payload = (env.output / receipt["path"]).read_bytes()
            self.assertEqual(receipt["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(receipt["size_bytes"], len(payload))
        with self.assertRaises(FileExistsError):
            cue.main(env.args)
        self.assertEqual(env.constructor.call_count, 1)

    def test_cli_load_failure_seals_all_eight_families_full_832_denominator(self):
        env = self.cli_environment(TinyEngine(), load_error=RuntimeError("cached checkpoint unavailable"))
        with self.assertRaises(RuntimeError):
            cue.main(env.args)
        report = json.loads((env.output / "report.json").read_text())
        self.assertEqual(report["counts"]["not_started"], 832)
        self.assertTrue(report["unfinished"])
        self.assertIsNone(report["metadata"]["model"])
        self.assertFalse(report["p0_passed"])
        self.assertEqual(len(list(env.output.glob("family-*.json"))), 8)

    def test_cli_p0_failure_no_stages_seals_unfinished_and_requires_all_candidates(self):
        engine = TinyEngine()
        p0 = passed_p0()
        p0["candidate_suffixes"].pop("square")
        engine.preflight = lambda *args: p0
        env = self.cli_environment(engine)
        with self.assertRaises(RuntimeError):
            cue.main(env.args)
        self.assertEqual(len(engine.prepares), 0)
        report = json.loads((env.output / "report.json").read_text())
        self.assertEqual(report["counts"]["not_started"], 832)
        self.assertTrue(report["unfinished"])
        self.assertTrue((env.output / "preflight.json").is_file())

    def test_cli_operation_failure_stops_before_generation_next_family_and_reports_832(self):
        engine = TinyEngine(failure=("score", 2))
        env = self.cli_environment(engine)
        with self.assertRaises(RuntimeError):
            cue.main(env.args)
        self.assertEqual((len(engine.prepares), len(engine.scores), len(engine.generations)), (2, 2, 1))
        report = json.loads((env.output / "report.json").read_text())
        self.assertEqual(report["n"], 832)
        self.assertEqual(report["counts"]["failed"], 1)
        self.assertEqual(report["counts"]["not_started"], 830)
        self.assertTrue(report["unfinished"])
        progress = json.loads((env.output / "progress.json").read_text())
        self.assertTrue(progress["sealed"] and progress["unfinished"])
        self.assertEqual(progress["total"], 832)

    def test_cli_rejects_mismatched_pair_and_nonreport_roots_before_model(self):
        env = self.cli_environment(TinyEngine())
        env.pair["8b-mpo"].revision = "a" * 40
        with self.assertRaises(ValueError):
            cue.main(env.args)
        self.assertEqual(env.constructor.call_count, 0)
        self.assertFalse(env.output.exists())
        args = list(env.args)
        args[args.index("--output") + 1] = str(self.root / "non-report")
        with self.assertRaises(ValueError):
            cue.main(args)
        self.assertEqual(env.constructor.call_count, 0)


if __name__ == "__main__":
    unittest.main()