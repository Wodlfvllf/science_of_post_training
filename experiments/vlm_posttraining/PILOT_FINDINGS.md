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

## Follow-up evidence · same day, explicitly developmental

The user requested continued paper-oriented investigation after the initial stop
gate. None of the following changes the original scorer, data, plan or primary
endpoint. New diagnostics are separately sourced and never counted as locked
confirmation trials.

- **Qualification:** independent one-tile torchvision RGB/bicubic/ImageNet pixel
  preprocessing matches exactly on six engineering images; native separator and
  tokenizer EOS both equal 151645. No preprocessing/EOS mismatch found.
- **Decomposition:** eight engineering families, 64 variants per 8B endpoint.
  Destination-shape answers are 37/64 at both stages; original destination-color
  answers are 9/64 MPO and 10/64 GSPO under separate case-combined scoring.
  Direct anchor and destination-color readout are 64/64 each. Explicit wording
  yields 17/64 and 16/64 color answers. This identifies a behavioral composition
  gap, not a latent accurate selector or a new mechanistic contribution.
- **Projector factorial:** eight engineering families, native/self/opposite-stage/
  permuted-delta sham. MPO binding 9/64→11/64 with GSPO projector; GSPO binding
  stays 10/64 with MPO projector. Self/sham accuracies unchanged, raw self score
  differences zero, every swap restored bit-exactly. No material bidirectional
  projector sufficiency established. Small effects are not proof of equivalence.
- **Original H2 screen:** four engineering families selected by a stable shard,
  sixteen tested layer/KV sites, seventeen conditions: 1,088 trials per endpoint.
  Joint six-condition flip/cancellation success is zero at every site. Fifteen
  exploratory stage routing-minus-permutation effects are zero and one is −0.125.
  No site earns confirmation. Discovery families have baseline measurements but
  no intervention measurements yet; a new operation needs a prespecified screen,
  not repeated site fishing.
- **Task-query handoff:** four engineering families, two query directions,
  four independent groups and seven conditions: 224 diagnostic rows per stage.
  No role-specific rescue. Conditions use an explicitly recorded open assistant
  turn rather than the original answer prefix; donors have no answer response.
- **Earlier rows:** four families, traversal/destination/final-prefix rows,
  two groups and nine conditions: 432 rows per stage. Actual word-end token
  prefixes and self-score parity validated. Scores change slightly but no answer
  prediction changes; this is a bounded null, not proof of absence everywhere.
- **Distributed handoff:** all eight KV groups jointly at prespecified layer 24,
  four families, two directions and seven conditions: 56 rows per stage plus
  seven patched greedy audits. No rescue or wrong-role color recombination.
  Candidate scores change, but every audited generation returns the same wrong
  blue destination answer. Joint patching does not establish synergy or necessity.
- **Earlier-stage alternative:** new SFT discovery baselines complete at both
  sizes, 480 trials each. 2B SFT binding 27/320; 8B SFT 39/320. SFT→MPO binding
  changes +0.625 points [−0.9375, +2.1875] and +1.5625 [−0.3125, +3.4375].
  This exploratory extension also does not establish a material binding gain.

Total manifest-published work is now **5,128 trials** (six discovery baselines,
generation audits, two original routing screens). Separately there are **3,024
new diagnostic rows** plus the earlier 96 answer-case rows, and 14 patched greedy
audits. Neither those counts nor a failed explanation is a publication guarantee.
All 60 pilot-validation and 200 confirmation families remain untouched.

The [working manuscript](PAPER_DRAFT.md) includes the verified evidence and missing
causal ledger. It is **not submission-ready**: no supported selective weight →
operation → answer mechanism, untouched confirmation, independent transfer or
cleared methods-level novelty exists yet. The next defensible research direction
is selection-to-attribute composition/anchor interference, with response-policy
and donor-copy explanations controlled—not a claim that GSPO improved grounding.

Follow-up toolkit regression receipt: **933 tests passed**, zero failures/errors/
skips. Drive snapshots include diagnostic source, exact images, published records,
metadata and reports; verification is at each acknowledged snapshot boundary.