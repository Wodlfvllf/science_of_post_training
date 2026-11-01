# Real-model pilot findings · 2026-11-01

**Decision: hold mechanism confirmation.** Execution succeeded, but the current
assay does not establish a material MPO→GSPO visual-binding improvement. This is
preliminary discovery evidence, not a circuit finding or an optimizer-level claim.

## Execution and provenance

- All four 2B/8B MPO/GSPO baselines completed: **1,920/1,920 trials, zero failed
  attempts**, on the same **40 independent discovery families**. Each checkpoint
  has 320 binding, 40 cued-color, 40 cued-edge and 80 text-control cases.
- Native/common GSPO generation audits saved **72 additional published trials**:
  12 per profile at 2B and 24 per profile at 8B. These cover one/two engineering
  families respectively, not the full 288-case engineering schedules.
- Four separate response-case diagnostics saved **96 diagnostic rows** on two
  engineering families. They are not new locked trials or confirmation evidence.
- GPU: RTX PRO 6000 Blackwell Server Edition; Python 3.13.11, Torch 2.14.1+cu130,
  Transformers 4.57.6; FP32/eager, one 448px tile, uncached candidate scoring.
  Native/capture/self P0 passed before each research job. Generation budget: 64.
- Plan: `44f4c5dbae51af518f044c735dc96ae8f6b043cb3c7ca70bd0512fb3cf9fc050`.
  Pilot source: `9fd0792188e28885836c22f53ccf3686ee9077c0611eb714ece5c27b16b978b1`.
  Dataset: `cb93e6e3e4034750252ebce2d1d995e59c537f8a6bf6de1fc963b69b0e2e3de5`.
  Original images, plans, scorers and checkpoint pins were not replaced.

## Locked common-prefix baseline

| Size/stage | Binding | Cued color | Cued edge | Text binding |
| --- | ---: | ---: | ---: | ---: |
| 2B MPO | 29/320 (9.06%) | 40/40 | 31/40 | 20/80 |
| 2B GSPO | 33/320 (10.31%) | 39/40 | 28/40 | 18/80 |
| 8B MPO | 44/320 (13.75%) | 40/40 | 36/40 | 57/80 |
| 8B GSPO | 50/320 (15.63%) | 40/40 | 38/40 | 66/80 |

Whole-family paired bootstrap binding changes:

- **2B:** +1.25 percentage points, 95% interval **[−1.875, +4.375]**.
- **8B:** +1.875 points, interval **[−1.875, +5.625]**.
- **8B text control:** +11.25 points, interval **[+5, +18.75]**.

Both binding intervals include zero; this is not proof of equivalence. A separate
post-hoc paired binding-change-minus-text-change estimate at 8B is −9.375 points,
interval [−17.1875, −2.5]. It is discovery analysis, not a new preregistered endpoint.
Near-ceiling cued controls do not prove preservation of every visual operation.

## Error and measurement checks

Anchor-color selections account for 269/320 and 259/320 binding answers at 2B
MPO/GSPO, and 242/320 and 231/320 at 8B. No binding prediction selected a color
absent from its recipient palette. This suggests a referent/traversal shortcut;
it does not identify a head or Q/K operation.

Common generation often agrees with the wrong candidate answer. Native generation
often explains the same wrong color. Native strict-format validity is **not**
semantic accuracy; correct explanations and one truncated 8B text-control answer
were observed. Capitalized generation can disagree with lowercase-only ranking.

The separate lowercase/title-case/class-combined audit did not repair binding on
its two engineering families: 8B MPO and GSPO both score 2/16 lowercase, 3/16
title-case and 2/16 after combining distinct lower/title prefix probabilities.
At 2B the corresponding counts are MPO 5/5/5 and GSPO 4/5/4 out of 16. These small,
post-hoc subsets do not clear all response-policy concerns. Prefix-event scoring
is not unrestricted response accuracy, and no original results were regraded.

## Actual component weights

The **340 ViT tensors are bit-exact equal** across MPO→GSPO at both sizes.
However, all **six projector tensor names** were audited and the projector is
**not frozen**: 7,662,591 parameter entries differ at 2B, and 21,802,852 at 8B.
Maximum absolute differences are 0.000244140625 and 0.00027877092361450195.
Thus an unchanged ViT does **not** establish unchanged projected image features.
Any subsequent decoder-weight attribution must control the projector background.

## Artifacts, durability and next gate

Fresh remote baseline/engineering reports, completeness checks, raw answers,
component tensor hashes and response-case diagnostics are retained. The
[independent backup helper](../../scripts/backup_vlm.py) captured published
boundaries plus original data/images/source and verified Google Drive ZIP bytes
and checksum receipts by readback. Model weights, environments and credentials
are excluded. Backup is not automatic container restoration.

The helper and regressions were published; **720 local tests passed**, with zero
failures, errors or skips. This does not independently attest remote execution.

No training, real-model H2/H3 hypothesis jobs, pilot-validation or confirmation
were run during this takeover. The 60 pilot-validation and 200 confirmation
families remain untouched. Before opening them, establish a credible relational
assay and a selective, explained stage-linked operation on development data;
resolve response policy and projector changes. Do not train merely to force a
positive endpoint difference, or present this sweep as a mechanistic contribution.