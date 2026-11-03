"""Model-free audit of TWO COMPLETE fixed cue/context development endpoints.

API: analyze(mpo, gspo, output) returns the report after publishing a NEW output
directory. CLI requires --mpo, --gspo and --output, with no analysis overrides.
Import/help are inert and stdlib-only. Validation executes only trusted adjacent
local runner/helper/data bytes, never archived source text, an Engine or torch.
Statistics alone lazily execute the local stats file and may import NumPy.

All 832 stages/endpoint, the embedded plan, actual dataset/image files, package
source, helper source, P0 and exact recorded runtime are required. Image paths
are the original plan's paths: there is no relocation/search/repair fallback.
Hashes establish consistency, NOT execution, decoded pixels, token decoding or
native EOS attestation. Invalid/truncated strict delivery stays in denominators;
it is not semantic-zero reasoning evidence. Previously used engineering indices
8..15 are development, not confirmation, and the frozen primary is unchanged.

Validation/statistics finish before ANY write. Inputs/source are rechecked before
publication and after output readback; integrity.json is written LAST. A late
mutation/write failure may leave an unsealed output, never a successful return.
Temporary import bindings are restored; no locks, pyc, caches or models are made.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import stat
import sys
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import UUID

REPO = Path(__file__).absolute().parents[1]
SEED, SAMPLES, ALPHA = 20261103, 5000, .05
SOURCES = ("scripts/vlm_cue_context.py", "scripts/vlm_generated_decomposition.py")
INDICES = tuple(range(8, 16))
DOCUMENTS = {"metadata.json", "report.json", "preflight.json", "progress.json"} | {
    f"family-{i:04d}.json" for i in INDICES}
FINAL_METADATA_FIELDS = {"elapsed_seconds", "failure", "sealed"}
__all__ = ["analyze", "main"]


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _equal(actual, expected, message):
    # Canonical equality also rejects bool/int and float/int substitutions.
    _require(_canonical(actual) == _canonical(expected), message)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _safe(path):
    path = Path(path).expanduser().absolute()
    _require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
             "Symlink/traversal paths are forbidden")
    return path


def _overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def _pairs(items):
    result = dict(items)
    _require(len(result) == len(items), "Duplicate JSON key")
    return result


def _decode(payload):
    result = json.loads(payload, object_pairs_hook=_pairs)
    _canonical(result)  # Reject NaN/Infinity, including overflowing float literals.
    return result


def _signature(info):
    # atime is deliberately omitted: ordinary read-only access may change it.
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


class _Snapshot:
    """Bounded bytes plus inode/timestamps, including detection of hardlink aliases."""

    def __init__(self):
        self.files, self.directories, self._inodes = {}, {}, {}

    def read(self, path, maximum=32 * 1024**2):
        path = _safe(path)
        try:
            before = path.stat()
            _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum,
                     "Missing, empty or oversized input")
            with path.open("rb") as stream:
                _equal(_signature(os.fstat(stream.fileno())), _signature(before),
                       "Input changed before reading")
                payload = stream.read(maximum + 1)
                _equal(_signature(os.fstat(stream.fileno())), _signature(before),
                       "Input changed while reading")
            after = _safe(path).stat()
        except FileNotFoundError as error:
            raise ValueError("Missing input: " + str(path)) from error
        _require(len(payload) == before.st_size and len(payload) <= maximum,
                 "Input changed size while reading")
        _equal(_signature(after), _signature(before), "Input changed while reading")
        key, inode = str(path), (before.st_dev, before.st_ino)
        _require(self._inodes.setdefault(inode, key) == key, "Aliased input/source file")
        receipt = {"sha256": _sha(payload), "size_bytes": len(payload), "stat": _signature(before)}
        _equal(self.files.setdefault(key, receipt), receipt, "Input/source changed between reads")
        return payload

    def directory(self, path, expected):
        path = _safe(path)
        _require(path.is_dir(), "Missing input directory")
        names = sorted(p.name for p in path.iterdir())
        _equal(names, sorted(expected), "Input directory inventory differs")
        _equal(self.directories.setdefault(str(path), names), names, "Input file list changed")

    def unchanged(self):
        for path, names in self.directories.items():
            self.directory(path, names)
        checked = _Snapshot()
        for path in self.files:
            checked.read(path)
        _equal(checked.files, self.files, "Input/source changed during analysis")


def _documents(root, snapshot):
    """Verify ALL family artifact sizes/hashes BEFORE parsing ANY family JSON."""
    snapshot.directory(root, DOCUMENTS)
    docs = {name: _decode(snapshot.read(root / name)) for name in
            ("metadata.json", "report.json", "preflight.json", "progress.json")}
    metadata, report = docs["metadata.json"], docs["report.json"]
    _require(type(metadata) is dict and type(report) is dict, "Metadata/report objects required")
    _equal(report["metadata"], metadata, "Report/separate metadata differ")
    _equal(report["metadata_sha256"], _sha(_canonical(metadata)), "Sealed metadata hash differs")
    _require(metadata["sealed"] is True and metadata["failure"] is None,
             "Require sealed, operation-complete metadata")
    receipts = report["family_artifacts"]
    _require(type(receipts) is list and len(receipts) == 8, "All eight family artifact receipts required")
    raw = {}
    for index, receipt in zip(INDICES, receipts, strict=True):
        name = f"family-{index:04d}.json"
        _require(type(receipt) is dict and set(receipt) == {
            "path", "sha256", "size_bytes", "family_index", "family_canonical_sha256", "schedule_sha256"},
            "Family artifact receipt schema differs")
        _equal([receipt["path"], receipt["family_index"]], [name, index], "Family artifact schedule differs")
        _require(type(receipt["size_bytes"]) is int and 0 < receipt["size_bytes"] <= 32 * 1024**2
                 and all(_digest(receipt[k]) for k in
                         ("sha256", "family_canonical_sha256", "schedule_sha256")),
                 "Family artifact full hashes/size required")
        payload = snapshot.read(root / name)
        _require(len(payload) == receipt["size_bytes"] and _sha(payload) == receipt["sha256"],
                 "Family artifact checksum/size differs before parsing")
        raw[name] = payload
    docs.update({name: _decode(payload) for name, payload in raw.items()})
    return docs


def _load(path, payload, name):
    # No loader.exec_module: compile the already checked LOCAL bytes, without pyc.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(payload, str(path), "exec"), module.__dict__)
    return module


@contextmanager
def _local_data(data):
    """Bind the lightweight exact local data module, not an installed model package.

    Only pure schedule/vocabulary imports use these temporary bindings. They are
    restored even on error; callers must not run concurrent project imports in
    the same interpreter during this audit.
    """
    package = ModuleType("vlm_pilot")
    package.__path__ = [str(REPO / "src/vlm_pilot")]
    package.data = data
    replacements = {"vlm_pilot": package, "vlm_pilot.data": data}
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _source_files():
    root = _safe(REPO / "src/vlm_pilot")
    _require(root.is_dir(), "Local package source is missing")
    _require(not any(p.is_symlink() for p in root.rglob("*")), "Symlink package source")
    return sorted(root.rglob("*.py"))


def _tools(snapshot):
    """Load only trusted file paths. Embedded source text is NEVER executed."""
    snapshots = {name: snapshot.read(REPO / name, 1024**2) for name in SOURCES}
    files = _source_files()
    payloads = {path: snapshot.read(path, 1024**2) for path in files}
    source_hash = _sha(_canonical([(p.relative_to(REPO / "src/vlm_pilot").as_posix(), _sha(payloads[p]))
                                   for p in files]))
    cue, helper = [_load(REPO / name, snapshots[name], "_cue_audit_" + Path(name).stem)
                   for name in SOURCES]
    path = REPO / "src/vlm_pilot/data.py"
    data = _load(path, payloads[path], "vlm_pilot.data")
    cue._helper = lambda: helper  # Dependency wiring only; no parser/scorer/grade override.
    with _local_data(data):
        cue._vocabulary_check()  # Its helper really imports the lightweight local data.
    return cue, helper, data, snapshots, source_hash, files


def _plan_and_dataset(metadata, cue, data, source_hash, snapshot):
    plan, protocol = metadata["plan"], metadata["protocol"]
    _require(type(plan) is dict and type(plan["spec"]) is dict, "Embedded complete plan required")
    cue.validate_protocol(protocol, SimpleNamespace(**plan))
    _equal(metadata["protocol_canonical_sha256"], _sha(_canonical(protocol)), "Protocol canonical hash differs")
    for key in ("protocol_file_sha256", "plan_file_sha256"):
        _require(_digest(metadata[key]), "Full recorded file hash required: " + key)
    _equal(_sha(_canonical(plan["dataset"])), plan["dataset_sha256"], "Embedded dataset full hash differs")
    spec = {k: v for k, v in plan["spec"].items() if k != "dataset"}
    _equal(_sha(_canonical({"spec": spec, "dataset": plan["dataset_sha256"],
                           "source": plan["source_sha256"]})), plan["plan_id"], "Full plan identity differs")
    _equal(plan["source_sha256"], source_hash, "Locked local package source differs")
    _equal(metadata["pilot_source_sha256"], source_hash, "Pilot source hash differs")
    _require(type(plan["spec"]["dataset"]) is str and Path(plan["spec"]["dataset"]).is_absolute(),
             "Original absolute dataset path required; no relocation fallback")
    root = _safe(plan["spec"]["dataset"])
    _require(root.is_dir() and (root / "images").is_dir(), "Original dataset/image directories required")
    # Retain existing directory contents without outlawing unrelated previews;
    # additions/removals during analysis still invalidate the final seal.
    for directory in (root, root / "images"):
        snapshot.directory(directory, [p.name for p in directory.iterdir()])
    actual = _decode(snapshot.read(root / "dataset.json", 16 * 1024**2))
    _equal(actual, plan["dataset"], "Actual/embedded dataset differ")
    # This local stdlib-only path checks the original scene schema and EVERY
    # family/image checksum, but deliberately does not import Pillow/decode pixels.
    _equal(data.load_dataset(root, verify_images=False), actual, "Dataset validation differs")
    for family in actual["families"]:
        for image in family["images"]:
            relative = Path(image["relative_path"])
            _require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe image reference")
            payload = snapshot.read(root / relative, 8 * 1024**2)
            _equal(_sha(payload), image["sha256"], "Actual original image checksum differs")
    families = cue.select_families(protocol, actual)
    _equal(metadata["family_ids"], [f["family_id"] for f in families], "Metadata family IDs differ")
    return families, root


def _metadata(docs, checkpoint, cue, helper, data, sources, source_hash, snapshot):
    m, p0 = docs["metadata.json"], docs["preflight.json"]
    families, root = _plan_and_dataset(m, cue, data, source_hash, snapshot)
    expected = {"kind": cue.FIXED_PROTOCOL["kind"], "checkpoint": checkpoint,
        "source_check": True, "matched_checkpoint_pins": cue.CHECKPOINT_PINS,
        "allow_downloads": False, "cached_only": True, "conditioning": "original_common_answer_prefix",
        "max_new_tokens": 32, "do_sample": False, "use_cache": False, "eos_override": False,
        "sites": [], "case_hooks": False, "full_captures": False,
        "image_cache": "at_most_one_native_image_no_decoder_states",
        "p0": "usual_native_embedding_capture_self_ALL_COLORS_plus_SHAPES",
        "expected_schedule": cue.stage_counts(), "family_indices": list(INDICES),
        "selection_filter": "none", "prior_use": cue.FIXED_PROTOCOL["prior_use"],
        "confirmation": "untouched", "p0_passed": True, "failure": None, "sealed": True}
    for key, value in expected.items():
        _equal(m[key], value, "Metadata differs: " + key)
    _require(type(m["created_at"]) is str and m["created_at"]
             and _finite(m["elapsed_seconds"]) and m["elapsed_seconds"] >= 0, "Metadata time evidence")
    _require(type(m["source_files"]) is dict and set(m["source_files"]) == set(sources), "Helper source inventory differs")
    for name, payload in sources.items():
        _equal(m["source_files"][name], {"sha256": _sha(payload), "size_bytes": len(payload),
                                        "text": payload.decode("utf-8")}, "Actual local helper source bytes differ")
    _equal(m["source_sha256"], _sha(_canonical({k: _sha(v) for k, v in sources.items()})),
           "Helper source aggregate full hash differs")
    inference = m["plan"]["spec"]["inference"]
    _equal(m["original_inference"], inference, "Original inference differs")
    _equal(m["effective_inference"], inference, "Effective inference differs")
    _require(inference["dtype"] == "float32" and type(inference["image_size"]) is int
             and inference["image_size"] == 448 and inference["answer_prefix"] == "Answer:"
             and type(inference["system_message"]) is str and bool(inference["system_message"])
             and type(inference["max_context_tokens"]) is int
             and 384 <= inference["max_context_tokens"] <= 4096, "Original FP32/448/Answer: inference required")
    pins = m["plan"]["spec"]["checkpoints"]
    _require(type(pins) is list and all(type(p) is dict for p in pins)
             and len({p["name"] for p in pins}) == len(pins), "Unique plan checkpoint definitions required")
    pair = {}
    for name, pin in cue.CHECKPOINT_PINS.items():
        matches = [p for p in pins if p["name"] == name]
        _require(len(matches) == 1, "Both matched native checkpoints required")
        cp = matches[0]
        for key, value in {"name": name, "family": "8b", "stage": name[3:], "backend": "internvl", **pin}.items():
            _equal(cp[key], value, "Native full checkpoint pin differs: " + key)
        pair[name] = cp
    model, runtime = m["model"], m["runtime"]
    _require(type(model) is dict and type(runtime) is dict and bool(runtime), "Model/runtime profile evidence required")
    _equal(model["checkpoint"], pair[checkpoint], "Model profile checkpoint differs from native full plan pin")
    _equal(model["common_inference"], inference, "Model profile inference differs")
    for key, value in {"scientific_fixture": False, "dtype": "torch.float32",
                       "image_tokens": 256, "tile_grid": [16, 16]}.items():
        _equal(model[key], value, "Native model profile differs: " + key)
    # Weight/tokenizer manifests, when present, are preserved as profile evidence;
    # no weight reads or fabricated requirement for a giant model manifest here.
    device = m["device_override"]
    _require(type(device) is str and re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", device) is not None,
             "Recorded explicit CUDA device required")
    _equal(runtime["device"], device, "Runtime device differs")
    _require(type(runtime.get("python")) is str and bool(runtime["python"])
             and type(runtime.get("packages")) is dict and bool(runtime["packages"]), "Recorded runtime provenance required")
    uuid = runtime.get("gpu_uuid")
    _require(type(uuid) is str and re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid) is not None
             and UUID(uuid[4:]).int != 0, "Full nonzero recorded GPU UUID required; no guessing")
    helper._require_preflight(p0, cue.COLORS + cue.SHAPES)
    _require(p0["fixture_only"] is False, "Native, not fixture, P0 required")
    _equal(p0["model"], model, "P0/model profile differs")
    _equal(p0["runtime"], runtime, "P0/runtime profile differs")
    return m, families, root


def _stage(row, expected, family, inference, helper, cue, state):
    """Check its OWN fresh input/score, native recorded settings and raw dual grade."""
    mutable = {"status", "attempted", "input", "score", "generation", "parsed_answer", "answer_valid",
               "exact", "score_exact", "errors", "original_correct", "cue_correct", "target_hit",
               "cue_following", "score_original_correct", "score_cue_correct", "score_checked",
               "oracle_cue_after_test"}
    _require(type(row) is dict and set(expected) <= set(row), "Full raw stage record required")
    for key in set(expected) - mutable:
        _equal(row[key], expected[key], "Canonical scheduled row differs: " + key)
    _require(row["attempted"] is True and row["errors"] == [] and row["score_checked"] is True
             and row["status"] in ("completed", "invalid_answer"), "Unattempted/incomplete/operation-error stage")
    inputs, score, generation, case = (row[k] for k in ("input", "score", "generation", "case"))
    _require(all(type(v) is dict for v in (inputs, score, generation)), "Raw input/score/generation required")
    _equal(inputs["case"], case, "Fresh prepared case differs")
    _equal([inputs["question"], inputs["profile"]], [case["question"], "common"], "Original input question/profile differs")
    prompt, ids, columns = (inputs[k] for k in ("rendered_prompt", "input_ids", "image_columns"))
    _require(type(prompt) is str and prompt.endswith("Answer:") and prompt.count(case["question"]) == 1
             and inference["system_message"] in prompt and prompt.count("<IMG_CONTEXT>") == 256,
             "Original common prompt/image tokens/answer prefix required")
    skeleton = prompt.replace(case["question"], "{QUESTION}")
    _equal(skeleton, state.setdefault("prompt_skeleton", skeleton), "Native prompt template changed")
    _require(type(ids) is list and bool(ids) and all(type(t) is int and t >= 0 for t in ids)
             and type(inputs["row"]) is int and inputs["row"] == len(ids) - 1
             and len(ids) + 32 <= inference["max_context_tokens"], "Original integer token IDs/context required")
    _require(type(columns) is list and len(columns) == 256 and all(type(i) is int for i in columns)
             and columns == sorted(set(columns)) and all(0 <= i < len(ids) for i in columns)
             and len({ids[i] for i in columns}) == 1, "Native 256 image columns required")
    image_token = ids[columns[0]]
    _equal([i for i, t in enumerate(ids) if t == image_token], columns, "Image token column lookup differs")
    _equal(image_token, state.setdefault("image_token", image_token), "Image token changed")
    matches = [i for i in family["images"] if i["image_id"] == case["image_id"]]
    _require(len(matches) == 1, "Original per-row image lookup required")
    _equal(inputs["image"], {**{k: matches[0][k] for k in ("image_id", "relative_path", "sha256")},
                            "verified_before_prepare": True}, "Original per-row image provenance differs")
    helper._check_score(score, {}, inputs, case["candidates"])
    _equal(score["score_kind"], "complete_candidate_sequence_logprob", "Native full-candidate scoring required")
    logs = score["candidate_token_logprobs"]
    _require(type(logs) is dict and set(logs) == set(case["candidates"]), "All candidate logprobs required")
    for candidate, values in logs.items():
        tokens = score["candidate_token_ids"][candidate]
        _require(type(values) is list and len(values) == len(tokens)
                 and all(_finite(v) and v <= 0 for v in values)
                 and math.isclose(sum(values), score["scores"][candidate], rel_tol=1e-12, abs_tol=1e-12)
                 and len(ids) + len(tokens) - 1 <= inference["max_context_tokens"], "Candidate logprob sum/context differs")
    settings = generation["settings"]
    _require(type(settings) is dict and type(settings.get("eos_token_id")) is int
             and settings["eos_token_id"] >= 0 and type(settings.get("max_new_tokens")) is int
             and settings.get("profile") == "unrestricted_greedy_audit", "Recorded native cap/EOS/profile required")
    eos = settings["eos_token_id"]
    _equal(eos, state.setdefault("eos", eos), "Recorded native EOS changed")
    helper._check_generation(SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=eos)), generation, 32)
    parsed = helper.strict_token_answer(generation, case["candidates"])
    graded = cue.grade_row(deepcopy(row))
    for key in ("parsed_answer", "answer_valid", "exact", "original_correct", "cue_correct",
                "target_hit", "cue_following", "score_original_correct", "score_cue_correct", "oracle_cue_after_test"):
        _equal(row[key], graded[key], "Raw reconstructed grade differs: " + key)
    _equal(row["status"], "completed" if parsed is not None else "invalid_answer", "Raw delivery status differs")
    _equal(row["score_exact"], score["prediction"] == case["gold"], "Raw score exact differs")
    _require(_finite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0, "Stage timing evidence required")
    if parsed is None:
        _equal(row["answer_failure"], "invalid_or_truncated_bare_answer", "Invalid delivery reason differs")
    else:
        _require("answer_failure" not in row, "Completed answer has an invalid-delivery flag")
    # Equal prompts must retain the same recorded tokenization; paired endpoints
    # must also match these complete input assertions, not just a template name.
    tokenization = {k: inputs[k] for k in ("input_ids", "row", "image_columns")}
    tokenization["candidate_token_ids"] = score["candidate_token_ids"]
    _equal(tokenization, state.setdefault("tokenizations", {}).setdefault(prompt, tokenization),
           "Recorded tokenization changed for the same prompt")


def _endpoint(docs, checkpoint, cue, helper, data, sources, source_hash, snapshot):
    metadata, families, dataset_root = _metadata(docs, checkpoint, cue, helper, data, sources, source_hash, snapshot)
    records, state = [], {}
    family_metadata = {k: v for k, v in metadata.items() if k not in FINAL_METADATA_FIELDS}
    for family, receipt in zip(families, docs["report.json"]["family_artifacts"], strict=True):
        record = docs[f"family-{family['index']:04d}.json"]
        _require(type(record) is dict and record["sealed"] is True and record["failure"] is None
                 and record["status"] == "completed", "Sealed complete family required")
        _equal(record["metadata"], family_metadata, "Family pre-finalization metadata differs")
        _equal([record["family_id"], record["family_index"], record["family"]],
               [family["family_id"], family["index"], family], "Actual original family identity differs")
        for key in ("family_canonical_sha256", "schedule_sha256"):
            _equal(record[key], receipt[key], "Family artifact canonical receipt differs")
        rows, expected = record["rows"], cue.make_schedule(family)
        _require(type(rows) is list and len(rows) == 104, "All 104 scheduled raw rows required")
        for row, scheduled in zip(rows, expected, strict=True):
            _stage(row, scheduled, family, metadata["original_inference"], helper, cue, state)
        _equal(record["counts"], cue._family_summary(record), "Complete raw family summary differs")
        records.append(record)
    summary = cue.summarize(records, checkpoint=checkpoint, p0_passed=True, failure=None)
    reconstructed_report = {k: v for k, v in docs["report.json"].items()
                            if k not in {"metadata", "metadata_sha256", "progress", "family_artifacts"}}
    _equal(reconstructed_report, summary, "Complete report differs from raw reconstruction")
    _require(summary["execution_complete"] is True and summary["counts"]["attempted"] == 832,
             "All 832 stages must be operation-complete")
    _require(all(c["n"] == 192 for c in summary["contexts"].values())
             and all(c["n"] == 64 for c in summary["tasks"].values())
             and len(summary["cue_context_groups"]) == 9
             and all(c["n"] == 64 for c in summary["cue_context_groups"])
             and len(summary["factor_cue_context_groups"]) == 72
             and all(c["n"] == 8 for c in summary["factor_cue_context_groups"]), "Unfiltered context/cue/factor/task counts differ")
    progress = docs["progress.json"]
    for key, value in {"stage": "sealed_complete", "device": metadata["device_override"], "completed": 832,
                       "attempted": 832, "total": 832, "sealed": True, "unfinished": False, "failure": None,
                       "expected_schedule": cue.stage_counts(), "counts": summary["counts"]}.items():
        _equal(progress[key], value, "Sealed progress differs: " + key)
    if "progress" in docs["report.json"]:
        _equal(docs["report.json"]["progress"], progress, "Embedded progress differs")
    return {"metadata": metadata, "records": records, "summary": summary,
            "state": state, "dataset_root": dataset_root}


def _stats(snapshot):
    path = _safe(REPO / "src/vlm_pilot/stats.py")
    return _load(path, snapshot.read(path, 1024**2), "_cue_audit_local_stats")


def _interval(differences, *, bound=1, stats=None):
    """Fixed separate 95% family intervals; NOT simultaneous/significance gates."""
    _require(type(differences) is dict and len(differences) == 8
             and all(type(k) is str and bool(k) and _finite(v) for k, v in differences.items()),
             "Exactly eight independent family values required")
    _require(type(bound) is int and bound in (1, 2), "Contrast bound must be exactly 1 or 2")
    stats = _stats(_Snapshot()) if stats is None else stats
    # The trusted helper rescales the EXACT bound and uses its _interval's
    # conservative Hoeffding fallback, never a zero-width bootstrap 'proof'.
    result = stats.family_difference_interval(differences, bound=bound, samples=SAMPLES, seed=SEED, alpha=ALPHA)
    return {**result, "bound": bound, "samples": SAMPLES, "seed": SEED}


def _csv_row(checkpoint, row):
    parsed, color_task = row["parsed_answer"], row["task_answer"] != "shape"
    cue_node, anchor = row["cue_node"], row["context_anchor"]
    return {"checkpoint": checkpoint, "family_id": row["family_id"], "family_index": row["family_index"],
        "case_id": row["case_id"], "kind": row["kind"], "edge": row["edge"], "direction": row["direction"],
        "palette": row["palette"], "image_id": row["image_id"], "cue": row["cue"], "context": row["cue_context"],
        "task_frame": row["task_frame"], "task_answer": row["task_answer"], "question": row["case"]["question"],
        "status": row["status"], "attempted": row["attempted"], "parsed_answer": parsed, "answer_valid": row["answer_valid"],
        "original_gold": row["original_gold"], "original_color_gold": row["original_color_gold"], "cue_gold": row["cue_gold"],
        "original_correct": row["original_correct"], "cue_correct": row["cue_correct"],
        "score_prediction": row["score"]["prediction"], "score_original_correct": row["score_original_correct"],
        "score_cue_correct": row["score_cue_correct"], "original_anchor": row["original_anchor"],
        "context_anchor": anchor, "destination_node": row["destination_node"], "cue_node": cue_node,
        "original_true_anchor_color": row["original_true_anchor_color"], "context_anchor_color": row["anchor_color"],
        "original_anchor_copy": parsed is not None and parsed == row["original_true_anchor_color"] if color_task else None,
        "context_anchor_copy": parsed is not None and parsed == row["anchor_color"] if anchor is not None else None,
        "oracle_cue_after_test": row["oracle_cue_after_test"],
        "cue_equals_original_anchor_after_test": cue_node == row["original_anchor"] if cue_node is not None else None,
        "cue_equals_context_anchor_after_test": cue_node == anchor if cue_node is not None and anchor is not None else None,
        "context_anchor_equals_destination_after_test": anchor == row["destination_node"] if anchor is not None else None,
        "finish_reason": row["generation"]["finish_reason"], "generated_tokens": len(row["generation"]["token_ids"]),
        "input_tokens": len(row["input"]["input_ids"])}


def _shape_summary(summary):
    families = summary["paired_families"]
    frames = {}
    for frame in (1, 2):
        items = [f["shape_relational"]["frames"][frame - 1] for f in families]
        frames[str(frame)] = {"n_outcomes": sum(i["n_outcomes"] for i in items),
            "exact": sum(i["exact"] for i in items), "n_families": 8,
            "families_all_eight_exact": sum(i["all_eight_exact"] for i in items),
            "groups": {k: {"n": sum(i["groups"][k]["n"] for i in items),
                            "all_exact": sum(i["groups"][k]["all_exact"] for i in items)} for k in items[0]["groups"]}}
    return {"frames": frames, "paired_frames_n": 64,
        "paired_frames_both_exact": sum(f["shape_relational"]["paired_frames_both_exact"] for f in families),
        "families_all_eight_exact_both_frames": sum(all(i["all_eight_exact"] for i in f["shape_relational"]["frames"]) for f in families),
        "definition": "Every paired/quad outcome must equal its own relation gold, not merely differ"}


def _statistics(endpoint, cue, stats):
    summary, checkpoint = endpoint["summary"], endpoint["metadata"]["checkpoint"]
    families = summary["paired_families"]
    contrasts = {context: _interval({f["family_id"]: f["cue_compliance_change"][context] for f in families}, stats=stats)
                 for context in cue.CONTEXTS[1:]}
    primary = _interval({f["family_id"]: f["primary_value"] for f in families}, stats=stats)
    rows = [_csv_row(checkpoint, cue.grade_row(deepcopy(row))) for record in endpoint["records"] for row in record["rows"]]
    roles = {}
    for context in cue.CONTEXTS:
        items = [r for r in rows if r["context"] == context]
        keys = ("oracle_cue_after_test", "cue_equals_original_anchor_after_test", "cue_equals_context_anchor_after_test",
                "context_anchor_equals_destination_after_test", "original_anchor_copy", "context_anchor_copy")
        roles[context] = {"n": len(items), "context_anchor_present": sum(r["context_anchor"] is not None for r in items),
                         **{key: sum(r[key] is True for r in items) for key in keys}, "denominator_filter": "none"}
    tasks = {key: {**cell, "strict_exact_rate": cell["original_correct"] / cell["n"],
                   "strict_valid_rate": cell["answer_valid"] / cell["n"]} for key, cell in summary["tasks"].items()}
    delivery = Counter("truncated_length" if r["finish_reason"] == "length" else
                       "valid_bare_eos" if r["answer_valid"] else "invalid_bare_eos" for r in rows)
    return {"counts": summary, "context_minus_lookup": contrasts, "primary_average_contrast": primary,
        "tasks": tasks, "shape_relational": _shape_summary(summary), "cue_role_coincidences_after_test": roles,
        "delivery": {"n": 832, "counts": dict(delivery), "generated_tokens": sum(r["generated_tokens"] for r in rows),
                     "input_tokens": sum(r["input_tokens"] for r in rows), "semantic_reasoning_accuracy": "not_measured"}}, rows


def _markdown(report):
    text = "# Complete cue/context development experiment audit\n\nNot paper-ready; no automated scientific decision.\n\n"
    text += "| Endpoint | True relation − lookup | Competing anchor − lookup | Frozen primary average (95% CI) |\n|---|---:|---:|---|\n"
    for checkpoint, item in report["endpoints"].items():
        p, c = item["primary_average_contrast"], item["context_minus_lookup"]
        text += (f"| {checkpoint} | {c['true_relation']['difference']:+.4f} | {c['competing_anchor']['difference']:+.4f} "
                 f"| {p['difference']:+.4f} {p['ci95']} |\n")
    text += "\nAll intervals use eight whole families, not 832 independent stages. Separate 95% intervals are not simultaneous coverage.\n\n"
    text += "| Endpoint/task | Strict exact / 64 | Valid / 64 |\n|---|---:|---:|\n"
    for checkpoint, item in report["endpoints"].items():
        for task, cell in item["tasks"].items():
            text += f"| {checkpoint}/{task} | {cell['original_correct']} / 64 | {cell['answer_valid']} / 64 |\n"
    return text + "\n" + "\n".join("- " + c for c in report["caveats"]) + "\n"


def analyze(mpo, gspo, output):
    """Validate both full endpoints; publish report/1664-row CSV/Markdown, seal last."""
    roots, output = [_safe(p) for p in (mpo, gspo)], _safe(output)
    _require(not _overlap(*roots) and not any(_overlap(r, output) for r in roots), "Overlapping input/output paths")
    if output.exists():
        raise FileExistsError("Output exists; no overwrite/resume")
    snapshot = _Snapshot()
    # Receipt failures, including malformed tampered family text, precede imports
    # and ALL family parsing/statistics. No archive-embedded program is trusted.
    docs = [_documents(root, snapshot) for root in roots]
    cue, helper, data, sources, source_hash, source_files = _tools(snapshot)
    snapshot.read(Path(__file__), 1024**2)
    with _local_data(data):
        endpoints = [_endpoint(d, checkpoint, cue, helper, data, sources, source_hash, snapshot)
                     for d, checkpoint in zip(docs, cue.CHECKPOINT_PINS, strict=True)]
    before, after = [e["metadata"] for e in endpoints]
    for key in ("plan", "plan_file_sha256", "protocol", "protocol_file_sha256", "protocol_canonical_sha256",
                "pilot_source_sha256", "source_files", "source_sha256", "original_inference", "effective_inference",
                "runtime", "device_override", "matched_checkpoint_pins", "family_ids", "family_indices"):
        _equal(before[key], after[key], "Endpoint mismatch: " + key)
    _equal(endpoints[0]["state"], endpoints[1]["state"], "Endpoint prompt/token/EOS assertions differ")
    _require(not any(_overlap(output, p) for p in
                     (endpoints[0]["dataset_root"], endpoints[1]["dataset_root"], REPO / "src", REPO / "scripts")),
             "Output overlaps dataset/source")
    stats, summaries, rows = _stats(snapshot), {}, []
    for endpoint in endpoints:
        checkpoint = endpoint["metadata"]["checkpoint"]
        summaries[checkpoint], csv_rows = _statistics(endpoint, cue, stats)
        rows.extend(csv_rows)
    _require(len(rows) == 1664 and len({(r["checkpoint"], r["case_id"]) for r in rows}) == 1664,
             "All 1664 unique stage CSV records required")
    interactions = {}
    for name in ("primary_average_contrast", *cue.CONTEXTS[1:]):
        values = [summaries[k][name if name == "primary_average_contrast" else "context_minus_lookup"]
                  for k in cue.CHECKPOINT_PINS]
        if name != "primary_average_contrast":
            values = [v[name] for v in values]
        a, b = [v["family_differences"] for v in values]
        _equal(sorted(a), sorted(b), "Stage contrast family pairing differs")
        interactions[name] = _interval({k: b[k] - a[k] for k in a}, bound=2, stats=stats)
    inventory = deepcopy(snapshot.files)
    report = {"kind": "cue_context_complete_experiment_model_free_audit_v1", "execution_complete": True,
        "paper_ready": False, "hypothesis_decision": "not_automated", "n_families": 8, "n_stages": 1664,
        "primary": cue.FIXED_PROTOCOL["primary"], "endpoints": summaries,
        "gspo_minus_mpo_contrast_interactions": interactions, "confirmation": "untouched",
        "multiplicity": "Separate descriptive 95% family CIs, alpha .05 each; NOT simultaneous or automated success tests",
        "caveats": ["Engineering indices 8..15 were used previously: development, not confirmation or globally fresh families.",
            "Frozen primary remains the equal average of the two family context-minus-lookup cue-compliance contrasts.",
            "Negative contrasts describe interference in this external prompt assay, not a native internal mediator.",
            "All three cues are tested without outcome filters; oracle/anchor coincidences are labelled only afterward, not autonomous rescue.",
            "Invalid/truncated strict delivery remains in every denominator; strict exactness is not semantic reasoning accuracy.",
            "Shape/color vocabularies have three/six answers; matched wording does not establish a common internal selector code.",
            "Eight whole families are uncertainty units. Constant samples use bounded conservative intervals, not certainty.",
            "Equal 32-token caps do not equalize FLOPs or account for all candidate-scoring cost.",
            "Stage interaction is not isolated optimizer causality, a weight-mediated mechanism, novelty or publication readiness.",
            "Hashes/profile assertions are consistency evidence, not independent execution, decoded-pixel, tokenizer or native-EOS attestation.",
            "Original plan/protocol FILE hashes are fully compared as recorded; their original files cannot be independently located from this schema.",
            "Actual dataset and EVERY original image file are required and hash-checked; no pixel decoding or path relocation occurs."],
        "provenance": {"plan_id": before["plan"]["plan_id"], "dataset_sha256": before["plan"]["dataset_sha256"],
            "protocol": before["protocol"], "source_sha256": source_hash, "runtime": before["runtime"],
            "runtime_ignored_fields": [], "inference": before["original_inference"],
            "models": {e["metadata"]["checkpoint"]: e["metadata"]["model"] for e in endpoints},
            "inputs_and_sources": inventory, "inventory_sha256": _sha(_canonical(inventory)),
            "images": "All original bytes checked, pixels NOT decoded", "token_attestation": False}}
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    artifacts = {"report.json": _canonical(report), "stages.csv": text.getvalue().encode("utf-8"),
                 "REPORT.md": _markdown(report).encode("utf-8")}

    def unchanged():
        _equal([str(p) for p in _source_files()], [str(p) for p in source_files], "Package source inventory changed")
        snapshot.unchanged()

    unchanged()  # Full validation/statistics/source stability BEFORE ANY write.
    _safe(output).mkdir(parents=True, exist_ok=False)
    for name, payload in artifacts.items():
        with (output / name).open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    checked = _Snapshot()
    checked.directory(output, artifacts)
    for name, payload in artifacts.items():
        _require(checked.read(output / name) == payload, "Output bytes changed before sealing")
    unchanged()
    seal = {"input_inventory_sha256": report["provenance"]["inventory_sha256"], "before_after_equal": True,
            "files": {name: {"sha256": _sha(payload), "size_bytes": len(payload)} for name, payload in artifacts.items()}}
    with (_safe(output) / "integrity.json").open("xb") as stream:
        stream.write(_canonical(seal))  # LAST publication; failed analyses have no success seal.
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("mpo", "gspo", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    analyze(args.mpo, args.gspo, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())