# VLM sandbox recovery

The independent VLM backup schema is not the text-study recovery schema. Use the
stdlib-only [recovery helper](../../scripts/recover_vlm.py), not the posttraining
restorer. It executes no archived source, model code or notebook.

## Explicit procedure

1. Authenticate Drive directly in the new notebook. Inspect only the designated
   `science-of-post-training-backups/vlm-posttraining` folder. Choose the latest
   ZIP **with its checksum receipt**; an unacknowledged newer ZIP is not verified.
2. Download ZIP and receipt to new local scratch paths. Run
   `inspect_archive(archive_path, receipt_path)`. This checks the whole ZIP SHA,
   exact inventory membership, every member's size/hash/CRC, bounded sizes and
   safe paths. Receipt hashes establish consistency, not independent authenticity.
3. Keep all GPU workers and artifact writers idle. Call
   `restore_archive(archive, receipt, new_staging, checkout)`. The stage must not
   exist. All destination conflicts are checked before copying. Existing identical
   artifacts are skipped; differing bytes or symlinks stop recovery. Source,
   notebooks, scripts, configs and docs remain in staging and are never published
   over checkout files. Only data, VLM results and root plan files are published.
   Dataset indexes and run manifests are last; every published file is rechecked.
4. Rebuild the isolated environment explicitly. Inspect the restored source-pinned
   plan and all published records with `validate_records()`/`inspect_run()`.
   Completed count is not correctness. Partial engineering audits retain their
   original full denominators and are not failed restorations.
5. Reestablish a Drive worker and require a verified upload/readback boundary.
   Inspect whether the intended next diagnostic already has complete/partial
   output. Never blindly rerun a budget: it limits **new** trials, not total work.
6. Require fresh checkpoint P0 on the actual device/runtime. A changed runtime
   must not silently resume historical trials. New diagnostic outputs can use
   their own recorded profile, with both compared endpoints sharing it. Do not
   bypass source/runtime checks, mutate old metadata or relock solely for recovery.
7. Launch each unfinished diagnostic through the detached explicit notebook
   controller, with a new output directory and durable status/log receipts.
   A detached process survives a tab disconnect, not sandbox deletion. Back up
   new reports, metadata and original images and verify the resulting snapshot.

## Verified recovery on 2 November 2026

- Latest receipt-paired archive: `20261101T164159Z-5bb277ef15cb42d49da4ef358c8e9811.zip`.
  SHA256: `cc24cc873fd85f297c96c81d63c60f71ba07d96a31663ec44f073a6a9f8d0aa8`.
  Size 41,456,637 bytes; 6,722 members, all verified before staging.
- Published 6,685 missing artifact files, including 1,296 original PNGs, 12 run
  manifests and the dataset index. Model-free semantic validation found 5,128
  completed trials, zero failed attempts. Original plan/source/data hashes match.
- Both complete C1 text-state reports survived. No C2 generated-decomposition
  report existed; two fresh C2 jobs subsequently completed without baseline reruns.
- New isolated runtime: Python 3.13.11, Torch 2.15.0, CUDA 13.2,
  Transformers 4.57.6, FP32/eager. Both 8B MPO/GSPO P0 passed: zero tolerance
  mismatches, native embedding maxabs 3.34e−5 / 4.20e−5, capture/self differences
  zero and matching argmax. Historical Torch 2.14.1 trials were inspected only.
- The active authenticated notebook was preserved. A tested detached controller
  was loaded from an uploaded file only after byte comparison against the exact
  published Git object. No archived notebook was opened or executed to recover
  credentials. The normal read-only inspection form was restored.

Full progress/results and limitations are in [counter-hypotheses](COUNTER_HYPOTHESES.md).
Recovery is operational success, not completion of the paper's causal evidence.