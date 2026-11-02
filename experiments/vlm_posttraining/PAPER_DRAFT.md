# Reading or Routing? Preliminary Diagnostics of Visual Referent Selection Across MPO→GSPO

**Provisional title · internal manuscript skeleton · evidence cutoff: 1 November 2026.**
**Not submission-ready.** Mechanism confirmation is on hold. Novelty is not
cleared; a weak baseline is not a new contribution, and a negative result does
not guarantee publication. This draft neither promises acceptance nor supplies
missing evidence with prospective language disguised as results.

Evidence below is transcribed from the recorded pilot findings, with methods
checked against the proposal, sprint and current diagnostic scripts. No new
execution, artifact reanalysis or literature retrieval was performed for this
draft. The early proposal/sprint statements that no real models had run are
historical; the later pilot findings govern the present results. New follow-up
diagnostic results remain **PENDING**, with explicit blanks in Section 5.

## Abstract

**Preliminary.** We investigate whether an adjacent released post-training
transition changes visual referent selection, rather than assuming improved
grounding or a particular neural mechanism. A reversible single-image relation
assay independently varies arrow direction, query direction and visible colors.
Four InternVL3.5 MPO/GSPO checkpoints completed 1,920 candidate-scored cases on
40 independent discovery families. Binding accuracy was 29/320 versus 33/320 at
2B and 44/320 versus 50/320 at 8B; paired changes were +1.25 percentage points
(95% family interval [−1.875, +4.375]) and +1.875 points
([−1.875, +5.625]). These estimates do not establish material visual-binding
improvement or equivalence. Frequent anchor-color answers motivate competing
shortcut and routing explanations. Released ViT tensors are identical within
each stage pair, but projector weights differ, precluding a frozen-interface
assumption. Small generation and capitalization audits do not resolve every
measurement concern. Projector compatibility, anchor lookup and earlier
query-routing hypotheses remain untested causal alternatives. **This preliminary
study does not assert a mechanism:** selective intervention, weight mediation,
untouched confirmation and independent transfer evidence are missing.

## 1. Introduction and related-work boundary

Answer accuracy alone does not specify which visual computation post-training
changes. The prospective question is narrower: can an identified released weight
difference be connected to a query-dependent operation and a referent-specific
answer, while ruling out perception and answer-policy alternatives? This is an
unresolved objective here, not an established research gap that prior work has
necessarily left open.

Close priors already occupy much of the proposed territory:

| Prior | Overlap and required boundary |
| --- | --- |
| [CARVE / Does RL Post-Training Ground Vision-Language Models?](https://openreview.net/forum?id=xLUhlfoeKH) | Its reviewed abstract covers certified edits, label binding, regional mediation, size/stage comparisons and decoder localization. Full methods remain unread. No methods-level novelty clearance is possible yet. |
| [Inside VLM Chart Reading, v2](https://arxiv.org/html/2609.13745v2) | Includes InternVL3.5-8B, factor patches, prompt-series mediation, downstream resets and separate-donor recombination. Hybrid answers and rescue/blocking are not new methods here. |
| [Frankenstein-Style Analysis](https://arxiv.org/html/2602.12395v1) | Studies parameter differences, mid/late localization, checkpoint merging and freezing. A projector swap or smaller merge alone is not a new explanation. |
| [Same Reward, Different Skills](https://arxiv.org/html/2610.01908v1) | Covers cued readout, find-and-bind, relational discovery and grounding/accuracy dissociation. Another weak-binding sweep is insufficient. |

These descriptions concern the cited authors' findings, not our reproductions.
A potential contribution would require a distinct, explained operation linked
selectively to released weights and surviving untouched tests. The current
baseline, error pattern and component audit do **not** meet that standard.

## 2. Methods and scope

### 2.1 Released stages, roles and immutable pins

The native OpenGVLab InternVL3.5 quartet is multimodal continual pretraining
(CPT) → supervised fine-tuning (SFT) → mixed preference optimization (MPO) →
additional online GSPO. “Pretrained” is a released multimodal CPT endpoint whose
Qwen3 initializer already has post-training ancestry, not a raw unaligned base.
MPO combines DPO, BCO and LM/SFT losses; it is **not pure DPO**. The unsuffixed
release is the GSPO endpoint, not merely another name for SFT.

All repository IDs below are under `OpenGVLab/`; revisions are full commits.
Six SFT/MPO/GSPO endpoints have now executed discovery baselines. CPT pins
provide planned lineage context, not measured results. The SFT extension was
chosen after the MPO→GSPO null and is explicitly exploratory.

| Stage / role | 2B native release and revision | 8B native release and revision |
| --- | --- | --- |
| CPT: prospective pre-SFT context | `InternVL3_5-2B-Pretrained` · `ce387f38cd21603625cdeb6c37f34a2bb89a771a` | `InternVL3_5-8B-Pretrained` · `e21d3bab4089ed923c08072a1eefc8b5faf41cd8` |
| SFT: prospective intermediate context | `InternVL3_5-2B-Instruct` · `a1e50c526faea805730148e7b1f2af18abe9e4b2` | `InternVL3_5-8B-Instruct` · `27506812aa9914804018996329e895977ee2d0c8` |
| MPO: measured adjacent reference | `InternVL3_5-2B-MPO` · `bf5ffd82a66a10df7de9fe520239a837c8808798` | `InternVL3_5-8B-MPO` · `bb33f08275aa566f3442c1f5002512ffb66d7952` |
| GSPO: measured adjacent endpoint | `InternVL3_5-2B` · `7d7bd7bcc35b77b391883ebd7f9686cfa0e25cc0` | `InternVL3_5-8B` · `9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79` |

The 8B transition is the planned primary scientific comparison; 2B supplies
engineering and cross-size replication. Total stored parameters are
2,348,347,392 and 8,528,318,464, respectively. Use native pinned code/tokenizers,
not HF/Flash/MoE conversions. Endpoint differences confound objective, rewards,
data mixture, training duration and optimizer; weight surgery cannot identify
the causal effect of the GSPO optimizer. Cross-size comparison is not a
randomized parameter-count experiment. No new training has been performed.

Original MiMo-VL-7B SFT/RL is prospective independent-lineage confirmation:
`XiaomiMiMo/MiMo-VL-7B-SFT` at `8f10075987cb64cd18751217865fe94542ddd548`
and `XiaomiMiMo/MiMo-VL-7B-RL` at `c64b51cbea3d815a027c7862736411a7b6a345ae`.
It has not been executed here; its adapter, freeze mask and numerical validity
remain to be established. It is not an independent randomized optimizer trial.

### 2.2 Reversible assay and statistical unit

Each family contains one three-node directed-cycle scene program, four image
variants and eight binding questions crossing edge direction, query traversal
direction and two color palettes. With anchor $a$, orientation $e$ and traversal
direction $d$, the certified answer is

$$
S=(a+ed)\bmod 3,\qquad Y=c_S.
$$

Reversing either direction changes the selected neighbor; reversing both cancels.
Shapes identify nodes, and colors are visible chips. This is a diagram assay,
not OCR, charts, natural photographs or cross-image reasoning. Donor and recipient
images, when interventions are eventually run, are processed separately.
Programs certify answers and constrained edits; checksums do not establish human
readability or absence of all visual artifacts.

The executed discovery set comprises **40 independent source families**, each
with eight binding, one cued-color, one cued-edge and two text-control cases.
Thus each checkpoint has 480 cases, including 320 binding cases. The four
checkpoints share these same families: 1,920 completed cases are neither 1,920
independent observations nor 1,920 correct answers. The generated split also
contains 24 engineering, 60 pilot-validation and 200 confirmation families;
the latter two splits remain entirely untouched.

Stage contrasts average within family and resample paired whole families, not
individual variants, prompts or checkpoints. Reported 95% intervals are the
recorded discovery family-bootstrap intervals. They do not capture training-seed
uncertainty. Constant/zero-discordance outcomes require conservative bounded
checks rather than a zero-width bootstrap claim. Near-ceiling controls and
nonsignificance cannot establish equivalence or selective preservation.

### 2.3 Inference, candidate scoring and P0

Recorded runtime: RTX PRO 6000 Blackwell Server Edition, Python 3.13.11,
Torch 2.14.1+cu130 and Transformers 4.57.6. Inference is **FP32, eager attention,
one 448px tile, no KV cache**, with explicit common prompting and `Answer:`
prefix. Native slow-tokenizer assets and checkpoint code are pinned; different
release template/default files are not assumed equivalent. Complete candidate
suffixes are scored; verified single-token suffixes permit one prefill, while
multi-token candidates require sequence scoring. Ties count as misses.
Candidate ranking is not unrestricted response accuracy.

Native-embedding, capture-only and self-Q/K P0 checks passed before each recorded
research job. They check finite logits, mismatches under fixed
$\mathrm{atol}=\mathrm{rtol}=10^{-4}$, argmax agreement, routing reconstruction
and multi-token parity where applicable. A passed P0 validates the tested
profile/sites, not all layers, contexts, tiles or future interventions. The new
independent pixel-transform qualification subsequently passed on six engineering
images: the published one-tile torchvision RGB/bicubic/ImageNet transform matched
every supplied pixel bit-exactly. The native conversation separator and tokenizer
EOS also matched (ID 151645). This does not validate dynamic tiling or unrestricted
semantic response scoring.

## 3. Verified preliminary results

### 3.1 Locked common-prefix baseline

All four schedules completed **480/480 cases each**, with zero failed attempts:
**1,920/1,920 total**. Completion denotes published execution, not correctness.

| Size / stage | Binding | Cued color | Cued edge | Text binding |
| --- | ---: | ---: | ---: | ---: |
| 2B SFT (exploratory extension) | 27/320 (8.44%) | 40/40 | 32/40 | 18/80 |
| 2B MPO | 29/320 (9.06%) | 40/40 | 31/40 | 20/80 |
| 2B GSPO | 33/320 (10.31%) | 39/40 | 28/40 | 18/80 |
| 8B SFT (exploratory extension) | 39/320 (12.19%) | 40/40 | 35/40 | 57/80 |
| 8B MPO | 44/320 (13.75%) | 40/40 | 36/40 | 57/80 |
| 8B GSPO | 50/320 (15.63%) | 40/40 | 38/40 | 66/80 |

The exploratory SFT→MPO binding changes also span zero: 2B +0.625 points
([−0.9375, +2.1875]) and 8B +1.5625 points ([−0.3125, +3.4375]). There is no
demonstrated earlier-stage binding improvement to substitute for the original
primary contrast. Six discovery schedules now total 2,880 complete cases.

| Paired discovery contrast, GSPO minus MPO | Change (percentage points) | Recorded 95% family interval |
| --- | ---: | ---: |
| 2B binding | +1.25 | [−1.875, +4.375] |
| 8B binding | +1.875 | [−1.875, +5.625] |
| 8B text binding | +11.25 | [+5, +18.75] |
| 8B binding change minus text change (**post hoc**) | −9.375 | [−17.1875, −2.5] |

Per-checkpoint accuracy intervals and other contrasts are not supplied in the
recorded findings and are not invented here. Both binding intervals include
zero. The results do not establish material visual-stage improvement, equivalence,
or a stage-linked routing mechanism. The post-hoc interaction is descriptive
development analysis, not a substituted primary endpoint.

### 3.2 Errors and answer-policy validity

Anchor-color selections were 269/320 → 259/320 at 2B and 242/320 → 231/320
at 8B. No binding prediction selected a color absent from its recipient palette.
This pattern is compatible with an anchor/traversal shortcut, but does not
identify its implementation or exclude an interface/perception explanation.

Separate GSPO generation audits published **72 additional trials**: common/native
profiles each covered 12 cases at 2B and 24 at 8B, on one/two engineering families,
respectively. Generation was greedy, uncached and bounded to 64 new tokens;
these were partial audits, not complete 288-case engineering schedules.
Common/native answers often reiterated the wrong color. Correct explanatory
answers and one truncated 8B text-control response were also observed. **Strict
format invalidity is not semantic error**, and no native long-CoT claim follows.

Four separate capitalization diagnostics produced **96 rows** on two engineering
families. Binding counts under lowercase/title-case/combined distinct-prefix
scoring were 5/16, 5/16, 5/16 for 2B MPO; 4/16, 5/16, 4/16 for 2B GSPO; and
2/16, 3/16, 2/16 for both 8B stages. This small post-hoc audit did not repair
binding. Combining distinct surface-prefix probabilities is not full semantic
answer scoring and does not clear every output-policy confound. Original locked
results were **not regraded**, and these rows are not confirmation trials.

### 3.3 Actual component changes

The **340 ViT tensors are bit-exact equal** within both MPO→GSPO pairs. However,
the audited six-tensor projector differs: **7,662,591 entries** at 2B and
**21,802,852** at 8B, with maximum absolute differences **0.000244140625** and
**0.00027877092361450195**. Frozen ViT weights do not imply unchanged projected
image features or a decoder-only explanation. Any decoder-delta attribution must
control the projector background. Changed-entry counts alone reveal no function.

## 4. Three alternative, falsifiable hypotheses

These alternatives may coexist and are not claimed findings. They replace no
frozen endpoint; any new diagnostic wording remains development work.

| Hypothesis | Discriminating test and prediction | Evidence against the proposed explanation |
| --- | --- | --- |
| **A. Projector–decoder interface compatibility** | Compare native, self, opposite-stage projector endpoint and matched delta sham in both decoder backgrounds. Re-extract features. A reproducible, selective background-by-projector interaction could implicate compatibility; identical-input projector-output comparisons would distinguish changed mapping from upstream drift. | No material selective endpoint effect, equally strong sham effects, or general cued/text damage rather than a relation-specific effect. A null full-projector swap does not exclude joint/distributed changes. |
| **B. Anchor lookup shortcut** | Compare original color, destination-shape, explicit outgoing/incoming shape/color, anchor readout and destination readout. Counterbalance anchors and directions. A stable anchor-biased selector with usable cued destination readout predicts failure before color handoff. | Reliable destination selection with color-only failure, errors tracking unreadable chips/edges, or no consistent anchor preference under controlled wording. Behavior alone cannot prove a shortcut circuit. |
| **C. Earlier query-conditioned routing failure/change** | After behavioral qualification, localize pre-answer query/selector states. Within checkpoint, Q-only and image-K-only substitutions should predict recipient-palette neighbor flips; joint edge/query reversal should cancel. Preserve recipient V, text contributions and total image mass; compare V-only, wrong-anchor and matched shams. | Donor-color copying, nonspecific disruption, absent cancellation/role specificity, or sufficient explanation by value/interface/output changes. A null tested site does not exclude redundant or distributed routing. |

An initial original-prompt routing screen subsequently completed 1,088 conditions
per endpoint: four engineering families × sixteen layer/KV groups × seventeen
conditions. All six required single-flip/cancellation predictions were jointly
correct on **zero** families at every tested site. Fifteen of sixteen exploratory
stage routing-minus-permutation contrasts were exactly zero; the remaining
contrast was −0.125, not the proposed positive signature. These are small,
development-only results and do not exclude a distributed or earlier-token
mechanism. They do not justify choosing a sensitive head as an H2 result.

Only successful, selective causal tests could distinguish an operation from a
behavioral description. The sprint's H1–H3 remain prospective: stage-linked
dependence, conditional Q/K routing, then bounded weight contribution with
necessity/rescue/blocking and same-input replay. None has passed on real models.

## 5. CURRENT follow-up diagnostics: results pending

The current follow-up script defines `qualify`, `decompose` and `projector` modes.
It uses 1–24 engineering families only (default eight), a new output directory,
cached pinned models, diagnostic P0 and separate source/runtime metadata; it
does not emit locked study trials. Decomposition adds the six wordings in
Hypothesis B to binding cases. New wording is not a baseline correction.

The projector helper loads only native `mlp1.{0,1,3}.{weight,bias}` tensors.
`ProjectorSwap` is an eval-only context for exact converted endpoint copies,
self controls and within-tensor permuted delta shams. It reports hashes,
perturbation/rounding diagnostics and bit-exact restoration; feature caches are
cleared before rescoring and after restoration. A whole-projector transplant is
an interface diagnostic, **not** a localized circuit or optimizer experiment.
Implemented checks are not evidence that a remote run succeeded.

**Fill the following only from saved, inspected diagnostic artifacts. Blanks
mean unavailable, not zero, null, failed or statistically nonsignificant.**

| Required new diagnostic evidence | CURRENT verified result |
| --- | --- |
| Qualification checkpoint/revision, source hash, runtime, families | 8B GSPO, original pinned plan/runtime; six engineering families; separate diagnostic source metadata retained. |
| Independent RGB/bicubic/ImageNet pixel parity and native separator/EOS agreement | All six images bit-exact (maxabs 0); native separator/tokenizer EOS both 151645. |
| Decomposition counts per wording, stage, family and capitalization policy | Eight engineering families, 64 binding variants/stage; distinct lower/title-prefix combined diagnostic: destination shape 37/64 both stages, original color MPO 9/64 and GSPO 10/64; direct anchor and destination color cues 64/64 each. Explicit relation wording shape 36/64→31/64 and color 17/64→16/64. These development wordings do not repair the original locked baseline. |
| Bidirectional projector endpoint/self/sham contrasts and family intervals | Eight engineering families: MPO native/self/sham 9/64 binding, GSPO-projector replay 11/64; GSPO native/self/sham/rollback all 10/64. Cued color/edge remain 8/8; text remains 12/16 MPO and 14/16 GSPO. This is not a material bidirectional effect; paired intervals and score-level checks still required. |
| Applied endpoint checksums, sham scale/rounding and post-swap feature re-extraction | Recorded for each of the six tensors; each condition clears image-feature caches and prepares fresh embeddings. Failed pre-fix diagnostic retained separately, not counted. |
| Self/no-op score parity and bit-exact restoration for actual diagnostic runs | All six contexts across both decoder backgrounds report bit-exact restoration. Accuracy self parity observed; raw score self parity audit pending. |
| Task-query single-group handoff | Four engineering families, both query directions, four independently edited groups and seven conditions: 224 rows per stage. No recipient-color rescue or wrong-role counterfactual signature; native/capture/self parity passed. Open-assistant task wording is distinct from the original baseline. |
| Earlier-query-row routing | Four engineering families, two groups, three semantic rows and nine conditions: 432 rows per stage. Actual traversal/destination token-prefix maps and self parity passed; no prediction changed. Maximum candidate-score changes were 0.02158/0.02674/0.06552 for MPO traversal/destination/final-prefix, and 0.01767/0.01971/0.07249 for GSPO. These bounded nulls do not exclude distributed routing. |
| Joint all-group handoff | All eight KV groups at externally prespecified layer 24, four families, both directions, seven conditions: 56 rows per stage. Candidate scores changed (maximum 1.7811 MPO, 2.0498 GSPO), but all conditions had identical aggregate hits; no wrong-role color transfer. Seven patched greedy audits per stage on the first family all returned the same incorrect blue destination answer with EOS, not donor shape answers. No claim of necessity, synergy or weight mediation follows. |
| Text-residual counter-hypothesis C1 | Two complete reports recovered from a receipt/member-verified Drive snapshot: 384 engineering rows per 8B endpoint plus six patched audits each. Post-block task-noun/final-assistant substitutions at four prespecified layers had no selective accuracy rescue; several reduced hits like wrong-role controls. Self and pre-task traversal structural gates passed. Raw margin/response interpretation remains a separate requirement. |
| Generated-selector counter-hypothesis C2 | New-runtime four-family screen after restoring the absent experiment: MPO direct 2/8, generated shape correct 5/8, generated-shape color chain 5/8, shifted-prediction cue 2/8; GSPO corresponding 2/8, 5/8, 4/8, 1/8, with one invalid shape delivery included. Own-prediction cue construction was audited; no oracle or best-candidate fallback. This small, repeatedly used engineering subset suggests a changed computation trajectory, not the native mechanism, held-out replication or a GSPO-acquired ability. |
| Mechanistic interpretation justified by these controls | The tested selective routing/handoff explanations are unsupported. No real-model H3 mediator was identified; bounded weight-to-operation mediation and untouched confirmation remain missing. |

## 6. Missing causal-evidence ledger and frozen confirmation

| Evidence required to finish the proposed paper | Present status / remaining requirement |
| --- | --- |
| Credible relational assay and response-policy qualification | Weak discovery baseline; complete new qualification/decomposition and readability audits before causal selection. |
| Selective stage-linked operation | Bounded original H2 engineering screen has no joint flip/cancellation signature; earlier/token-task routing alternatives remain under investigation. H3 weight mediation remains absent. |
| Recipient-specific hybrid answers and joint cancellation | Missing; rule out donor copying with disjoint/same palettes, wrong roles and matched shams. |
| Necessity, rescue and downstream blocking | Missing; correct/wrong-role mediator controls must avoid generic destructive clamps and floor artifacts. |
| Weight → operation → answer link | Missing; bounded opposing rollback/replay, same-input module comparison, matched real/control deltas and mediator tests, with projector background controlled. |
| Selective retention | Not established by ceiling controls; freeze preservation margins and use informative cued/text controls with uncertainty. |
| Untouched validation and confirmation | All 60 pilot-validation and 200 confirmation families untouched; no outcomes may guide current decisions. |
| Scope beyond one synthetic task/lineage | No MiMo or independently audited photographic/OCR transfer results or validated adapters. |
| Novelty and final claims/artifact audit | CARVE methods unread; operation-level overlap and independent human audit remain unresolved. |

Before opening **any** pilot-validation outcomes, freeze the screen's sites
(at most two per size/checkpoint), tensor/norm scope, projector backgrounds,
donor construction, prompts, scoring, exclusions, primary 8B contrast,
material-effect thresholds, retention margins and multiplicity. Discovery-based
power must use family-level variance of the actual contrast. The sprint's
approximately 15-point routing and 10-point weight screens are proposed filters,
not demonstrated effects or power guarantees; final values remain **____**.

The 60 pilot-validation families are development for the final study. Any changes
after their use require a documented new freeze before the **200 untouched
confirmation families** are opened. Include all scheduled families, unpatched
failures and completeness accounting; jointly-correct subsets are secondary with
an explicit selection funnel. Do not reuse confirmation for tuning, redefine a
failed primary endpoint, or pool repeated artifacts as independent evidence.
Freeze independent-lineage/transfer protocols before their held-out outcomes too.
At this cutoff, **all validation/confirmation outcomes remain untouched and blank**.

## 7. Limitations, ethics and reproducibility

The low binding baseline limits intervention identifiability and may reflect
diagram readability, instruction interpretation, anchor heuristics or interface
limitations. Two engineering families do not resolve response policy. Counterfactual
hooks/swaps can be off manifold; successful interchange does not establish a
unique circuit, and local nulls do not prove absent information. No training-seed,
long-CoT, dynamic-tiling, photographic or broad-VLM generality is established.

Audit model/code licenses and separate image/photograph rights before release.
Fresh renders do not certify absence of pretraining task familiarity. Future
photos/OCR require permission, privacy/identifiable-content checks and blinded
edit/readability audits. Do not train merely to force a positive difference.

Recorded evidence identities are plan
`44f4c5dbae51af518f044c735dc96ae8f6b043cb3c7ca70bd0512fb3cf9fc050`,
pilot source `9fd0792188e28885836c22f53ccf3686ee9077c0611eb714ece5c27b16b978b1`
and dataset `cb93e6e3e4034750252ebce2d1d995e59c537f8a6bf6de1fc963b69b0e2e3de5`.
Reports, raw answers, component hashes and original image/plan bytes are recorded
as retained, with a Drive snapshot verified by byte/checksum readback. This is
durability evidence, not independent remote-execution attestation or automatic
restoration. Offline test success is engineering evidence, not a hypothesis pass.

An eventual public artifact should expose exact pins, source/data hashes, scene
programs, family splits, scores/raw outputs, P0, failures, intervention definitions
and reproducible family statistics, subject to licenses. Exclude credentials,
private cloud paths and unrelated studies. A reviewed submission must be anonymous
and self-contained: remove identifying repository/account links and metadata;
do not publicly advertise this venue-specific internal draft as a submission.
The existing identifiable repository is not an anonymous contributed artifact.

## 8. LLM Usage Statement — actual assistance, draft for human review

**GitHub Copilot was used for research ideation, literature synthesis and
comparison, experimental planning, implementation/debugging of code and notebooks,
assistance with execution through the user-authorized hosted notebook/browser,
artifact inspection and preliminary analysis, and drafting this manuscript.**
Execution assistance included managing recorded baseline/engineering jobs and
backups; it does not mean the pending follow-up diagnostics or H2/H3 ran.
Copilot helped interpret counts, uncertainty, errors and component audits.
This statement is not limited to grammar editing. LLM output is not evidence or
independent verification. Human authors must audit influential sources, code,
raw artifacts, statistics and every final claim, and remain responsible for
accuracy, attribution, ethics and disclosure. Final human-review record: **____**.

## 9. Discussion, deadlines and paper decision

Current conclusion: hold mechanism confirmation and resolve development-stage
qualification. The verified sweep does not support the proposed weight-linked
routing explanation, and it is not reframed as a novel negative contribution.
Proceed only if selective causal evidence and a methods-level novelty review
justify it; otherwise defer or stop. A negative result carries **no publication
guarantee**, and no manuscript-confidence or acceptance promise is made.

The project records an internal **4–5 November 2026 evidence gate** and the
[CVPR 2027 dates](https://cvpr.thecvf.com/Conferences/2027/Dates) /
[author requirements](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
as checked on 31 October, not rechecked for this draft: **10 November, 23:59 AoE**
enrollment/title/abstract/all authors and complete profiles; **16 November,
23:59 AoE** full paper; **23 November, 23:59 AoE** supplement. Title/abstract may
be revised until the paper deadline; authors cannot be added/removed after
enrollment. Profile moderation may take two weeks. The recorded main limit is
eight pages excluding references and the required LLM Usage Statement.
Missing central evidence cannot be deferred to the supplement. A deadline is
not a reason to fabricate an abstract, open holdouts prematurely or submit this
skeleton as a completed manuscript.

### Local source record

- [experiments/vlm_posttraining/PROPOSAL.md](experiments/vlm_posttraining/PROPOSAL.md)
- [experiments/vlm_posttraining/SPRINT.md](experiments/vlm_posttraining/SPRINT.md)
- [experiments/vlm_posttraining/PILOT_FINDINGS.md](experiments/vlm_posttraining/PILOT_FINDINGS.md)
- [experiments/vlm_posttraining/README.md](experiments/vlm_posttraining/README.md)
- [configs/vlm/pilot.yaml](configs/vlm/pilot.yaml)
- [scripts/vlm_research_followup.py](scripts/vlm_research_followup.py)
- [scripts/vlm_projector_diagnostic.py](scripts/vlm_projector_diagnostic.py)
- [src/vlm_pilot/runtime.py](src/vlm_pilot/runtime.py)
- [src/vlm_pilot/stats.py](src/vlm_pilot/stats.py)