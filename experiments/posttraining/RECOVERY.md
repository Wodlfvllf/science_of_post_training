# Recover after a Linux GPU container reset

Use the trusted clean checkout at `/marimo/science_of_post_training_recovered`,
detached at `d846061a58cc9159d3685260ca03bd27049005cc`. Keep its experiment package
and locked configs unchanged. Add the recovery shell script and adjacent Python
helper from this change to that checkout; do **not** extract scripts/code from the
backup, run a general unzip over the repo, or pull a different scientific recipe.
The existing `scripts/run_posttraining.sh` must also be present for `--run`.

Required package-source digest:
`f9f24e86e8e979e4b59056c6f8cd286a197722cf1aeb9b8b0a2ae0c0e67a827f`.
Recovery compares this with the checkout, archived source bytes, each saved plan,
and the installed editable package. It does not check out Git revisions or edit
`src/interp`, configs, saved metadata, or model settings.

## 1. Get the archive into the terminal filesystem

Notebook authentication is **not** terminal authentication. An empty `incoming`
directory means the files still need downloading. In the notebook process with
the already authenticated `GoogleDriveFileSystem` object named `fs`, use the
same remote directory/path syntax used by `DriveBackup`. This selects the newest
timestamp-named archive with a receipt, not merely the newest unacknowledged ZIP:

```python
from pathlib import Path

remote_dir = "YOUR_EXISTING_BACKUP_FOLDER"  # Same value used by DriveBackup.
receipts = sorted(p for p in fs.ls(remote_dir, detail=False) if p.endswith(".zip.sha256"))
if not receipts:
    raise ValueError("No acknowledged backups found")
remote_receipt = receipts[-1]
remote_zip = remote_receipt.removesuffix(".sha256")
incoming = Path("/marimo/science_of_post_training_recovered/incoming")
incoming.mkdir(parents=True, exist_ok=True)
local_zip = incoming / Path(remote_zip).name
fs.get_file(remote_receipt, str(local_zip) + ".sha256")
fs.get_file(remote_zip, str(local_zip))
print(local_zip)  # Only the local path, never tokens or filesystem credentials.
```

If the filesystem listing returns IDs rather than the timestamp names used by the
backup worker, select the matching ZIP/receipt pair in the Drive UI and use its
filesystem paths explicitly. Do not invent a receipt for a ZIP lacking one.
The shell independently verifies the receipt and archive; download completion
alone is not validation. Notebook and terminal must share this local filesystem.

## 2. One command

Replace `BACKUP.zip` with the filename printed above. Its adjacent receipt must be
named `BACKUP.zip.sha256`, containing exactly the lowercase hash, two spaces, and
the ZIP basename (optional final newline), as written by `DriveBackup`.

```sh
bash /marimo/science_of_post_training_recovered/scripts/recover_posttraining.sh /marimo/science_of_post_training_recovered/incoming/BACKUP.zip --run
```

- **No flag:** verify, prepare the private environment, validate and restore,
  then check GPU runtime; **do not launch experiments**.
- **`--run`:** all of the above, then run the existing orchestration script from
  the repo root using this checkout's `.venv`. It resumes incomplete jobs, skips
  completed jobs, and subsequently starts missing checkpoints/studies, including
  repair if it was never started. Its normal preflights/tests still apply and it
  may download weights/data. It is not an RL-only or time-budgeted command.
- **`--check-only`:** offline receipt, ZIP, source, conflict and full snapshot
  validation; no installs, restored artifacts, Torch imports, or GPU checks.
  Uses the existing `.venv` if present, otherwise the bootstrap interpreter;
  that interpreter needs the trusted project installed with its core dependencies.
  Lock directories/files are still created. This is not resume approval.
- **`RECOVERY_PYTHON=/path/to/python3.13`:** choose the bootstrap interpreter
  when `python3` is unsuitable. No Python version manager is installed for you.
  `DEVICE` defaults to `cuda:0`. Cache environment settings are left unchanged.

Python >=3.11 is required for the helper; this archive records **3.13.11 exactly**.
Normal preparation rejects a Python-version mismatch **before installing large
dependencies**. A valid existing isolated `.venv` is reused, never deleted; a
broken or system-site-packages venv is refused for manual inspection. If dependency
installation was interrupted, rerun the same command. Do not repair global Python.

The private venv installs the local editable project with `posttrain,dev,notebooks`
and these recorded direct pins from PyPI (including Torch, not a custom CUDA index):
Torch 2.11.0, transformers 4.57.6, tokenizers 0.22.2, safetensors 0.8.0,
huggingface-hub 0.36.2, datasets 4.8.5, fsspec 2026.2.0, pydantic 2.13.4,
PyYAML 6.0.3, filelock 3.31.0. This is **not** a complete transitive lockfile.
A satisfied installation is not repeated. `pip check` must pass.

## Safety and failure behavior

- Linux `flock` and `filelock.FileLock` use the same BSD OS locks. Recovery holds
  the pipeline lock **before staging or installing**, plus existing run writer
  locks and planned pipeline destinations. Active pipeline/individual writers
  fail immediately. Lock files are never deleted. Local POSIX storage is required;
  network-filesystem locking/atomicity is not validated. Stop other sessions that
  could create unrelated new jobs or modify source during recovery.
- A fresh private staging directory receives only direct `data/posttrain` JSON/
  JSONL and actual manifests, metadata and referenced shards under `runs/posttrain`.
  Archived code/configs/logs/reports/orphan shards/locks are never restored.
  Every ZIP member is checked for traversal, duplicate/colliding names, links,
  special/encrypted files, unsupported compression, CRC and bounded expansion.
  Caps: 512 MiB ZIP, 64 MiB/member, 2 GiB expanded total, 100,000 entries,
  compression ratio <=1,000. Full project validation applies stricter shard limits.
- Staged snapshots pass the installed project's `read_snapshot` before publication.
  All destination conflicts are checked first. Different existing bytes cause a
  refusal; even an older archived manifest cannot replace newer live progress.
  Use a separate clean checkout to inspect a conflicting historical archive.
  Identical files are skipped; atomic no-clobber files are published before
  manifests. Interrupted setup/restoration can be rerun without destructive rollback.
- After restoration, all live snapshots are validated again. Incomplete jobs must
  exactly match the current runtime profile, including Python/packages, GPU/CUDA,
  precision flags, **threads and interop threads**. No weights are loaded for this
  comparison. The runner still validates actual model/tokenizer fingerprints and
  numerical preflight later; a preparation success does not guarantee generation.
- The prior runtime recorded **20/20 threads, CUDA 13.0, Blackwell 96 GB**. The new
  host reportedly has **4 CPUs and 32 GiB RAM**. Expect possible thread mismatches;
  recovery reports and blocks them instead of calling thread setters or editing
  metadata. Recover a genuinely compatible host/runtime or design a separate run;
  do not bypass provenance checks to continue this one.
- **32 GiB host RAM may be insufficient** for loading 7.3B FP32 weights (~27 GiB
  weights alone) with CPU buffers and system overhead, even with 96 GB VRAM.
  No automatic RAM sizing, swap changes, fit guarantee, throughput/ETA promise,
  or automatic fallback is provided. Allow sufficient disk for staging, venv,
  downloaded wheels and model cache too.
- Runtime mismatches leave restored artifacts and the venv intact, return failure,
  and never launch the pipeline. Successful runtime probes, including mismatch
  details and archive/local/installed source digests, are written to unique
  `runs/posttrain/logs/recovery-*.json` receipts. Errors before that point remain
  visible in terminal output.
- `--run` releases recovery's locks before calling the unchanged old script, which
  reacquires its own pipeline/writer locks. A competing pipeline can win the small
  handoff window; the losing command fails safely. No edits occur in that gap.
- A SHA-256 receipt establishes byte consistency, **not authenticity**. Use backups
  and local checkout/scripts from trusted storage. Recovery neither executes
  archived code nor restores caches, weights, secrets, notebook auth or backups.
  Restart the notebook backup worker explicitly after recovery and verify uploads.