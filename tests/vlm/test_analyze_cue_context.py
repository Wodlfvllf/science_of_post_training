"""Fabricated consistency fixtures ONLY; no torch, models, GPU, network or commands.

The smallest enrolled schedule still has ALL eight families and 832 stages per
endpoint. Original data._family geometry supplies 24 engineering families; all
their fake image bytes are checksummed, never decoded. Scores, generations, P0,
native profile labels and token IDs are explicitly FABRICATED, not execution
evidence. Only test-local protocol plan/data hashes are replaced: production's
trusted frozen protocol, native pin and runtime checks are never weakened.

The fixture writer uses cue._family_record and its real raw grader/summaries.
Tests run no inference/Engine or subprocess. Merely adding this file does not
execute tests, import NumPy or configure an environment.
"""
from __future__ import annotations

import builtins
import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).absolute().parents[2] / "scripts/analyze_vlm_cue_context.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_cue_context_contracts", PATH)
    module = importlib.util.module_from_spec(spec)
    exec(compile(PATH.read_bytes(), str(PATH), "exec"), module.__dict__)
    return module


audit = _load()


def _write(path, value):
    path.write_bytes(audit._canonical(value))


class FabricatedPair:
    """No native computation: tiny complete JSON/file consistency evidence only."""

    def __init__(self, root):
        self.root = root
        self.snapshot = audit._Snapshot()
        self.tools = audit._tools(self.snapshot)
        self.cue, self.helper, data, sources, source_hash, _ = self.tools
        self.dataset_root = root / "FABRICATED_DATA_NOT_PIXELS"
        (self.dataset_root / "images").mkdir(parents=True)
        dataset = {"schema_version": 1, "name": "reading-or-routing-v1", "seed": 31, "image_size": 448,
                   "split_counts": {"engineering": 24, "discovery": 0, "pilot_validation": 0, "confirmation": 0},
                   "families": [data._family(31, "engineering", i) for i in range(24)]}
        for family in dataset["families"]:
            for image in family["images"]:
                payload = ("FABRICATED IMAGE BYTES NOT PNG: " + image["image_id"]).encode()
                (self.dataset_root / image["relative_path"]).write_bytes(payload)
                image["sha256"] = audit._sha(payload)
        _write(self.dataset_root / "dataset.json", dataset)
        self.inference = {"dtype": "float32", "device": "cuda:0", "image_size": 448,
            "max_context_tokens": 1024, "system_message": "FABRICATED TEST ONLY", "answer_prefix": "Answer:",
            "atol": .0001, "rtol": .0001}
        pins = [{"name": name, "family": "8b", "stage": name[3:], "backend": "internvl", "fixture_seed": 17, **pin}
                for name, pin in self.cue.CHECKPOINT_PINS.items()]
        spec = {"dataset": str(self.dataset_root), "name": "FABRICATED TEST ONLY", "seed": 31,
                "checkpoints": pins, "inference": self.inference, "max_new_tokens": 64}
        data_hash = audit._sha(audit._canonical(dataset))
        plan_id = audit._sha(audit._canonical({"spec": {k: v for k, v in spec.items() if k != "dataset"},
                                              "dataset": data_hash, "source": source_hash}))
        self.plan = {"schema_version": 1, "plan_id": plan_id, "created_at": "FABRICATED TEST ONLY",
                     "dataset_sha256": data_hash, "source_sha256": source_hash, "spec": spec, "dataset": dataset}
        # Test-local module only: impossible to satisfy the REAL pinned hashes
        # with fake PNG bytes. All other frozen fields/selection stay unchanged.
        self.protocol = {**deepcopy(self.cue.FIXED_PROTOCOL), "plan_id": plan_id, "dataset_sha256": data_hash}
        self.cue.FIXED_PROTOCOL = self.protocol
        self.runtime = {"python": "FABRICATED TEST ONLY", "packages": {"torch": "NOT IMPORTED"}, "device": "cuda:0",
                        "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555", "extra_dynamic_field": {"pid": 71},
                        "threads": 4}
        self.roots = {checkpoint: root / checkpoint for checkpoint in self.cue.CHECKPOINT_PINS}
        self.out = root / "analysis"
        with audit._local_data(data):
            for checkpoint, pin in zip(self.cue.CHECKPOINT_PINS, pins, strict=True):
                output = self.roots[checkpoint]
                output.mkdir()
                model = {"checkpoint": pin, "scientific_fixture": False, "dtype": "torch.float32",
                         "common_inference": self.inference, "image_tokens": 256, "tile_grid": [16, 16],
                         "evidence_note": "FABRICATED LABELS, NOT A REAL CHECKPOINT RUN"}
                metadata = {"kind": self.protocol["kind"], "created_at": "FABRICATED TEST ONLY",
                    "protocol": self.protocol, "protocol_file_sha256": audit._sha(audit._canonical(self.protocol)),
                    "protocol_canonical_sha256": audit._sha(audit._canonical(self.protocol)), "plan": self.plan,
                    "plan_file_sha256": audit._sha(audit._canonical(self.plan)), "source_check": True,
                    "pilot_source_sha256": source_hash, "source_files": {k: {"sha256": audit._sha(v),
                        "size_bytes": len(v), "text": v.decode()} for k, v in sources.items()},
                    "source_sha256": audit._sha(audit._canonical({k: audit._sha(v) for k, v in sources.items()})),
                    "checkpoint": checkpoint, "matched_checkpoint_pins": self.cue.CHECKPOINT_PINS,
                    "model": model, "runtime": self.runtime, "original_inference": self.inference,
                    "effective_inference": self.inference, "device_override": "cuda:0", "allow_downloads": False,
                    "cached_only": True, "conditioning": "original_common_answer_prefix", "max_new_tokens": 32,
                    "do_sample": False, "use_cache": False, "eos_override": False, "sites": [], "case_hooks": False,
                    "full_captures": False, "image_cache": "at_most_one_native_image_no_decoder_states",
                    "p0": "usual_native_embedding_capture_self_ALL_COLORS_plus_SHAPES",
                    "expected_schedule": self.cue.stage_counts(), "family_indices": list(audit.INDICES),
                    "family_ids": [f["family_id"] for f in dataset["families"][8:16]], "selection_filter": "none",
                    "prior_use": self.protocol["prior_use"], "confirmation": "untouched", "p0_passed": True}
                records = []
                for family in dataset["families"][8:16]:
                    record = self.cue._family_record(family)
                    for row in record["rows"]:
                        self.stage(row, family, checkpoint)
                    record.update(status="completed", failure=None, sealed=True, metadata=deepcopy(metadata))
                    record["counts"] = self.cue._family_summary(record)
                    _write(output / f"family-{family['index']:04d}.json", record)
                    records.append(record)
                metadata.update(elapsed_seconds=2.0, failure=None, sealed=True)
                check = {"passed": True, "finite": True, "argmax_match": True, "mismatches": 0, "max_abs": 0.0}
                p0 = {"passed": True, "fixture_only": False, "model": model, "runtime": self.runtime,
                      "checks": {k: deepcopy(check) for k in ("native_embedding", "capture_only", "self_qk")},
                      "candidate_suffixes": {c: [20 + i] for i, c in enumerate(self.cue.COLORS + self.cue.SHAPES)},
                      "multi_token_candidate_checks": {}}
                report = self.cue.summarize(records, checkpoint=checkpoint, p0_passed=True)
                report.update(metadata=metadata, metadata_sha256=audit._sha(audit._canonical(metadata)),
                              family_artifacts=[self.receipt(output, r) for r in records])
                progress = {"stage": "sealed_complete", "device": "cuda:0", "completed": 832, "attempted": 832,
                            "total": 832, "elapsed_seconds": 2.1, "eta_seconds": 0, "sealed": True,
                            "unfinished": False, "failure": None, "expected_schedule": self.cue.stage_counts(),
                            "counts": report["counts"]}
                for name, value in (("metadata.json", metadata), ("preflight.json", p0),
                                    ("progress.json", progress), ("report.json", report)):
                    _write(output / name, value)

    def stage(self, row, family, checkpoint):
        case = row["case"]
        prompt = (self.inference["system_message"] + "\n<img>" + "<IMG_CONTEXT>" * 256
                  + "</img>\n" + case["question"] + "\nassistant\nAnswer:")
        ids = [1] + [9] * 256 + [sum(case["question"].encode()) + 100, 10]
        image = next(i for i in family["images"] if i["image_id"] == row["image_id"])
        inputs = {"case": deepcopy(case), "question": case["question"], "profile": "common",
                  "rendered_prompt": prompt, "input_ids": ids, "row": len(ids) - 1, "image_columns": list(range(1, 257)),
                  "image": {**{k: image[k] for k in ("image_id", "relative_path", "sha256")}, "verified_before_prepare": True}}
        candidates = case["candidates"]
        score = {"scores": {c: -float(i + 1) for i, c in enumerate(candidates)},
                 "candidate_token_ids": {c: [20 + i] for i, c in enumerate(candidates)},
                 "candidate_token_logprobs": {c: [-float(i + 1)] for i, c in enumerate(candidates)},
                 "prediction": candidates[0], "tied": False, "routing": {}, "score_kind": "complete_candidate_sequence_logprob",
                 **{k: inputs[k] for k in ("rendered_prompt", "input_ids", "row", "image_columns")}}
        answer = row["cue_gold"] if row["kind"] == "cue_context" else row["original_gold"]
        # Known heterogeneous family contrasts and opposing GSPO effects expose
        # primary averaging/cancellation without selecting families by outcomes.
        if row["cue_context"] == "true_relation" and family["index"] % 2 == 0:
            answer = row["original_true_anchor_color"]
        if row["cue_context"] == "competing_anchor" and checkpoint == "8b-mpo" and family["index"] % 2 == 0:
            answer = row["original_true_anchor_color"]
        if row["task_answer"] == "shape" and row["task_frame"] == 1 and family["index"] == 8 and row["edge"] == 1:
            answer = "NOT A BARE SHAPE"
        truncated = row["task_answer"] == "color" and row["task_frame"] == 2 and family["index"] == 9 and row["palette"] == 1
        generation = {"text": answer, "token_ids": [11] * 32 if truncated else [11, 2],
            "finish_reason": "length" if truncated else "eos", "parsed_answer": "UNTRUSTED ENGINE FALLBACK",
            "valid": True, "correct": True, "settings": {"max_new_tokens": 32, "eos_token_id": 2,
                "do_sample": False, "use_cache": False, "profile": "unrestricted_greedy_audit"}}
        valid = self.helper.strict_token_answer(generation, candidates) is not None
        row.update(status="completed" if valid else "invalid_answer", attempted=True, input=inputs, score=score,
                   score_checked=True, score_exact=score["prediction"] == case["gold"], generation=generation, elapsed_seconds=.01)
        if not valid:
            row["answer_failure"] = "invalid_or_truncated_bare_answer"
        self.cue.grade_row(row)

    @staticmethod
    def receipt(root, record):
        name = f"family-{record['family_index']:04d}.json"
        payload = (root / name).read_bytes()
        return {"path": name, "family_index": record["family_index"], "sha256": audit._sha(payload),
                "size_bytes": len(payload), **{k: record[k] for k in ("family_canonical_sha256", "schedule_sha256")}}

    def edit(self, checkpoint, name, mutate, *, rereceipt=False):
        path = self.roots[checkpoint] / name
        value = json.loads(path.read_bytes())
        mutate(value)
        _write(path, value)
        if rereceipt:
            report_path = self.roots[checkpoint] / "report.json"
            report = json.loads(report_path.read_bytes())
            index = value["family_index"] - 8
            report["family_artifacts"][index] = self.receipt(self.roots[checkpoint], value)
            _write(report_path, report)

    def coherent_metadata_edit(self, checkpoint, mutate):
        root = self.roots[checkpoint]
        metadata = json.loads((root / "metadata.json").read_bytes())
        mutate(metadata)
        _write(root / "metadata.json", metadata)
        report = json.loads((root / "report.json").read_bytes())
        report["metadata"] = metadata
        report["metadata_sha256"] = audit._sha(audit._canonical(metadata))
        _write(root / "report.json", report)
        p0 = json.loads((root / "preflight.json").read_bytes())
        p0.update(runtime=metadata["runtime"], model=metadata["model"])
        _write(root / "preflight.json", p0)
        for index in audit.INDICES:
            self.edit(checkpoint, f"family-{index:04d}.json",
                      lambda r: r.update(metadata={k: v for k, v in metadata.items() if k not in audit.FINAL_METADATA_FIELDS}),
                      rereceipt=True)

    def analyze(self):
        # Re-register every trusted file in the CURRENT audit snapshot. Only the
        # test-local loaded cue module's frozen plan/data hashes are substituted.
        def tools(snapshot):
            for path in self.snapshot.files:
                snapshot.read(path)
            return self.tools
        with patch.object(audit, "_tools", side_effect=tools):
            return audit.analyze(self.roots["8b-mpo"], self.roots["8b-gspo"], self.out)


class CueContextAnalysisContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def pair(self):
        return FabricatedPair(self.root)

    def test_complete_all_rows_tasks_shapes_roles_intervals_and_integrity(self):
        pair = self.pair()
        before = {p: p.read_bytes() for root in pair.roots.values() for p in root.iterdir()}
        result = pair.analyze()
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(result["kind"], "cue_context_complete_experiment_model_free_audit_v1")
        self.assertEqual(result["primary"], pair.protocol["primary"])
        self.assertTrue(result["execution_complete"])
        self.assertFalse(result["paper_ready"])
        self.assertEqual(result["hypothesis_decision"], "not_automated")
        self.assertEqual(result["n_stages"], 1664)
        for checkpoint, summary in result["endpoints"].items():
            self.assertEqual(summary["counts"]["counts"]["invalid_answer"], 8)
            self.assertEqual(summary["delivery"]["counts"]["truncated_length"], 4)
            self.assertEqual(summary["tasks"]["shape_frame_1"]["original_correct"], 60)
            self.assertEqual(summary["tasks"]["color_frame_2"]["original_correct"], 60)
            self.assertEqual(summary["shape_relational"]["frames"]["1"]["families_all_eight_exact"], 7)
            self.assertEqual(summary["shape_relational"]["frames"]["2"]["families_all_eight_exact"], 8)
            self.assertEqual(summary["shape_relational"]["paired_frames_both_exact"], 60)
            self.assertEqual(summary["shape_relational"]["families_all_eight_exact_both_frames"], 7)
            shape_groups = summary["shape_relational"]["frames"]["1"]["groups"]
            self.assertEqual(shape_groups["edge_reversal"], {"n": 32, "all_exact": 28})
            self.assertEqual(shape_groups["joint_direction_reversal_palette"], {"n": 16, "all_exact": 15})
            self.assertEqual(shape_groups["joint_edge_reversal_palette"], {"n": 16, "all_exact": 14})
            for task in summary["tasks"].values():
                self.assertEqual(task["n"], 64)
                self.assertEqual(task["strict_exact_rate"], task["original_correct"] / 64)
            for cell in summary["cue_role_coincidences_after_test"].values():
                self.assertEqual(cell["n"], 192)
                self.assertEqual(cell["oracle_cue_after_test"], 64)
                self.assertEqual(cell["cue_equals_original_anchor_after_test"], 64)
                self.assertEqual(cell["denominator_filter"], "none")
            self.assertEqual(summary["cue_role_coincidences_after_test"]["competing_anchor"]["context_anchor_equals_destination_after_test"], 96)
            for ci in (*summary["context_minus_lookup"].values(), summary["primary_average_contrast"]):
                self.assertEqual((ci["families"], ci["bound"], ci["seed"], ci["samples"], ci["alpha"]), (8, 1, 20261103, 5000, .05))
                self.assertEqual(len(ci["family_differences"]), 8)
            expected = (-1 / 3) if checkpoint == "8b-mpo" else (-1 / 6)
            self.assertAlmostEqual(summary["primary_average_contrast"]["difference"], expected)
        interaction = result["gspo_minus_mpo_contrast_interactions"]["primary_average_contrast"]
        self.assertEqual(interaction["bound"], 2)
        self.assertAlmostEqual(interaction["difference"], 1 / 6)
        with (pair.out / "stages.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1664)
        self.assertEqual(len({(r["checkpoint"], r["case_id"]) for r in rows}), 1664)
        for row in rows:
            if row["cue"]:
                self.assertEqual(row["cue_correct"] == "True", bool(row["parsed_answer"]) and row["parsed_answer"] == row["cue_gold"])
                if row["oracle_cue_after_test"] == "True":
                    self.assertEqual(row["original_correct"], row["cue_correct"])
            if row["original_anchor_copy"] == "True":
                self.assertEqual(row["parsed_answer"], row["original_true_anchor_color"])
        self.assertEqual(result["provenance"]["runtime_ignored_fields"], [])
        self.assertFalse(result["provenance"]["token_attestation"])
        inventory = result["provenance"]["inputs_and_sources"]
        self.assertEqual(sum("FABRICATED_DATA_NOT_PIXELS/images/" in p for p in inventory), 96)
        seal = json.loads((pair.out / "integrity.json").read_bytes())
        self.assertTrue(seal["before_after_equal"])
        self.assertEqual(seal["input_inventory_sha256"], result["provenance"]["inventory_sha256"])
        for name, receipt in seal["files"].items():
            payload = (pair.out / name).read_bytes()
            self.assertEqual(receipt, {"sha256": audit._sha(payload), "size_bytes": len(payload)})
        with self.assertRaises(FileExistsError):
            pair.analyze()

    def test_artifact_tamper_rejected_before_family_parse_or_source_execution(self):
        pair = self.pair()
        path = pair.roots["8b-gspo"] / "family-0008.json"
        path.write_bytes(b"NOT JSON: CHANGED ARTIFACT")
        original = audit._decode
        def decode(payload):
            self.assertNotEqual(payload, b"NOT JSON: CHANGED ARTIFACT")
            return original(payload)
        with patch.object(audit, "_decode", side_effect=decode), patch.object(audit, "_tools", side_effect=AssertionError("Source executed too early")):
            with self.assertRaisesRegex(ValueError, "before parsing"):
                audit.analyze(pair.roots["8b-mpo"], pair.roots["8b-gspo"], pair.out)
        self.assertFalse(pair.out.exists())

    def test_raw_flags_score_context_caps_and_errors_rejected_even_with_new_receipt(self):
        pair = self.pair()
        path = pair.roots["8b-gspo"] / "family-0008.json"
        original_family, original_report = path.read_bytes(), (path.parent / "report.json").read_bytes()
        mutations = [lambda r: r["rows"][0].update(cue_correct=not r["rows"][0]["cue_correct"]),
            lambda r: r["rows"][0].update(attempted=1),
            lambda r: r["rows"][0].update(errors=[{"operation": "generate"}]),
            lambda r: r["rows"][0].update(edge=True),
            lambda r: r["rows"][0]["input"].update(profile="native"),
            lambda r: r["rows"][0]["score"]["input_ids"].append(99),
            lambda r: r["rows"][0]["score"]["candidate_token_logprobs"].update(red=[-99.]),
            lambda r: r["rows"][0]["generation"]["settings"].update(max_new_tokens=32.0),
            lambda r: r["rows"][0]["generation"]["settings"].update(use_cache=True),
            lambda r: r["rows"][0]["generation"].update(token_ids=[11]),
            lambda r: r["rows"][0].update(cue_gold="NOT A COLOR"),
            lambda r: r["rows"].pop(), lambda r: r["rows"].reverse(),
            lambda r: r["counts"]["contexts"]["lookup_only"].update(n=23)]
        for mutation in mutations:
            path.write_bytes(original_family)
            (path.parent / "report.json").write_bytes(original_report)
            with self.subTest(mutation=mutation):
                pair.edit("8b-gspo", path.name, mutation, rereceipt=True)
                with self.assertRaises((ValueError, RuntimeError)):
                    pair.analyze()
                self.assertFalse(pair.out.exists())

    def test_complete_summary_not_just_counts_is_compared(self):
        pair = self.pair()
        pair.edit("8b-gspo", "report.json", lambda r: r.update(primary_interpretation="AUTOMATIC SUCCESS"))
        with self.assertRaisesRegex(ValueError, "Complete report"):
            pair.analyze()
        self.assertFalse(pair.out.exists())

    def test_metadata_seal_and_local_embedded_source_match(self):
        pair = self.pair()
        pair.coherent_metadata_edit("8b-gspo", lambda m: m["source_files"][audit.SOURCES[0]].update(text="raise AssertionError('ARCHIVE CODE EXECUTED')"))
        with self.assertRaisesRegex(ValueError, "local helper source bytes"):
            pair.analyze()
        self.assertFalse(pair.out.exists())

    def test_runtime_pair_ignores_no_dynamic_fields(self):
        pair = self.pair()
        pair.coherent_metadata_edit("8b-gspo", lambda m: m["runtime"]["extra_dynamic_field"].update(pid=999))
        with self.assertRaisesRegex(ValueError, "Endpoint mismatch: runtime"):
            pair.analyze()
        self.assertFalse(pair.out.exists())

    def test_full_recorded_gpu_uuid_required(self):
        pair = self.pair()
        pair.coherent_metadata_edit("8b-gspo", lambda m: m["runtime"].update(gpu_uuid="GPU-11111111"))
        with self.assertRaisesRegex(ValueError, "GPU UUID"):
            pair.analyze()
        self.assertFalse(pair.out.exists())

    def test_different_full_gpu_uuid_is_not_ignored_in_pairing(self):
        pair = self.pair()
        pair.coherent_metadata_edit("8b-gspo", lambda m: m["runtime"].update(
            gpu_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"))
        with self.assertRaisesRegex(ValueError, "Endpoint mismatch: runtime"):
            pair.analyze()
        self.assertFalse(pair.out.exists())

    def test_model_full_native_pin_cannot_be_relabelled(self):
        pair = self.pair()
        pair.coherent_metadata_edit("8b-gspo", lambda m: m["model"]["checkpoint"].update(revision="a" * 40))
        with self.assertRaises(ValueError):
            pair.analyze()
        self.assertFalse(pair.out.exists())

    def test_actual_image_required_even_for_unscheduled_original_family(self):
        pair = self.pair()
        image = pair.plan["dataset"]["families"][0]["images"][0]
        (pair.dataset_root / image["relative_path"]).write_bytes(b"CHANGED UNSCHEDULED IMAGE")
        with self.assertRaisesRegex(ValueError, "checksum"):
            pair.analyze()
        self.assertFalse(pair.out.exists())

    def test_invalid_truncated_answers_not_rescued_by_engine_parser_flags(self):
        pair = self.pair()
        pair.edit("8b-gspo", "family-0008.json", lambda r: next(row for row in r["rows"] if row["status"] == "invalid_answer").update(
            parsed_answer="triangle", answer_valid=True, original_correct=True), rereceipt=True)
        with self.assertRaisesRegex(ValueError, "reconstructed grade"):
            pair.analyze()
        self.assertFalse(pair.out.exists())

    def test_missing_extra_symlink_alias_and_duplicate_json_rejected(self):
        pair = self.pair()
        root = pair.roots["8b-gspo"]
        path, extra = root / "family-0008.json", root / "backup.zip"
        original = path.read_bytes()
        for mode in ("missing", "extra", "symlink", "hardlink", "duplicate", "nonfinite"):
            path.unlink(missing_ok=True)
            path.write_bytes(original)
            extra.unlink(missing_ok=True)
            if mode == "missing":
                path.unlink()
            elif mode == "extra":
                extra.write_bytes(b"NOT ALLOWED")
            elif mode in ("symlink", "hardlink"):
                path.unlink()
                other = pair.roots["8b-mpo"] / path.name
                path.symlink_to(other) if mode == "symlink" else os.link(other, path)
            elif mode == "duplicate":
                (root / "metadata.json").write_text('{"sealed":true,"sealed":true}')
            else:
                (root / "metadata.json").write_text('{"value":1e999}')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                pair.analyze()
            self.assertFalse(pair.out.exists())

    def test_mutation_during_statistics_fails_before_any_output(self):
        pair = self.pair()
        original = audit._statistics
        def changed(*args):
            result = original(*args)
            pair.edit("8b-gspo", "report.json", lambda r: r.update(n=831))
            return result
        with patch.object(audit, "_statistics", side_effect=changed), self.assertRaisesRegex(ValueError, "changed"):
            pair.analyze()
        self.assertFalse(pair.out.exists())

    def test_write_failure_never_publishes_integrity(self):
        pair = self.pair()
        original = Path.open
        def opened(path, mode="r", *args, **kwargs):
            if path == pair.out / "stages.csv" and mode == "xb":
                raise OSError("FABRICATED WRITE FAILURE")
            return original(path, mode, *args, **kwargs)
        with patch.object(Path, "open", opened), self.assertRaises(OSError):
            pair.analyze()
        self.assertFalse((pair.out / "integrity.json").exists())

    def test_constant_intervals_exact_bounds_conservative_and_no_success_gate(self):
        for bound in (1, 2):
            for value in (0., float(bound), -float(bound)):
                with self.subTest(bound=bound, value=value):
                    ci = audit._interval({str(i): value for i in range(8)}, bound=bound)
                    self.assertEqual(ci["interval_method"], "conservative_hoeffding")
                    self.assertTrue(-bound <= ci["ci95"][0] < ci["ci95"][1] <= bound)
                    self.assertEqual(ci["difference"], value)
                    self.assertNotIn("success", ci)
        with self.assertRaises(ValueError):
            audit._interval({str(i): 1.1 for i in range(8)}, bound=1)

    def test_import_help_cli_validation_no_models_or_eager_numpy_and_binding_restored(self):
        pair = self.pair()
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "transformers", "numpy", "PIL", "pydantic", "yaml", "filelock"}:
                raise AssertionError("Forbidden import: " + name)
            return original_import(name, *args, **kwargs)
        saved = {name: sys.modules.get(name) for name in ("vlm_pilot", "vlm_pilot.data")}
        with patch("builtins.__import__", side_effect=guarded):
            module = _load()
            with self.assertRaises(SystemExit) as info:
                module.main(["--help"])
            self.assertEqual(info.exception.code, 0)
            for args in ([], ["--mpo", "x", "--gspo", "y"], ["--mpo", "x", "--gspo", "y", "--output", "z", "--seed", "1"]):
                with self.assertRaises(SystemExit):
                    module.main(args)
            # Full valid input validation, not merely an early missing-file gate.
            snapshot = audit._Snapshot()
            with audit._local_data(pair.tools[2]):
                for checkpoint, root in pair.roots.items():
                    docs = audit._documents(root, snapshot)
                    audit._endpoint(docs, checkpoint, *pair.tools[:5], snapshot)
        self.assertEqual(saved, {name: sys.modules.get(name) for name in saved})

    def test_overlap_no_clobber_and_real_frozen_protocol_refused_for_fake_data(self):
        pair = self.pair()
        mpo, gspo = pair.roots.values()
        for a, b, out in ((mpo, mpo, pair.out), (mpo, gspo, mpo / "analysis"), (mpo, gspo, self.root)):
            with self.subTest(out=out), self.assertRaisesRegex(ValueError, "Overlapping"):
                audit.analyze(a, b, out)
        # Unpatched production loads the true protocol from LOCAL source and
        # rejects this fixture's substituted full plan/data hashes.
        with self.assertRaisesRegex(ValueError, "fixed cue/context"):
            audit.analyze(mpo, gspo, pair.out)
        self.assertFalse(pair.out.exists())


if __name__ == "__main__":
    unittest.main()