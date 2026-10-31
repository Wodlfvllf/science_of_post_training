# Reading or Routing? A Deadline-Gated VLM Mechanism Study

**Decision memo, 31 October 2026. The [pilot implementation](README.md) now exists;
no downloaded-checkpoint/GPU experiment or hypothesis result is claimed.**
This narrows the broader [proposal](PROPOSAL.md) after checking the readable CARVE
abstract. It does not assert findings or cleared novelty. Do not change active
maths/repair workers or their pinned source/configuration for this study.

## 1. Core paper objective

**Identify whether an adjacent post-training transition changes query-conditioned
selection of visible evidence, explain the changed computation, and test whether
specific released weight differences can selectively reverse and replay it.**

Working title: **Reading or Routing? Tracing Post-Training Changes in Visual
Referent Selection.**

The potential contribution is not that RL improves accuracy, changes attention,
or sometimes fails to bind labels. It is the connection

**released weight change → identified routing operation → referent-specific answer**,

with direct perception, value transformation and output-policy explanations tested.
An attention head being important, a successful patch, or a smaller layer merge
does not by itself establish a new explanation. If no such connection survives
controls, do not force this into a main-track paper for the deadline.

## 2. What the closest papers already establish

| Verified primary source | Already covered | Boundary for this paper |
| --- | --- | --- |
| [Does RL Post-Training Ground Vision-Language Models? A Causal Measurement / CARVE](https://openreview.net/forum?id=xLUhlfoeKH) | Solver-certified edits, value-versus-label binding, regional activation patching, decoder localization, 3B/7B differences and attention–causality disagreement. | Repeating these analyses on InternVL or adding CPT/SFT columns is too incremental. |
| [Inside VLM Chart Reading, v2](https://arxiv.org/html/2609.13745v2) | Includes InternVL3.5-8B; factor-wise patches, prompt-series mediation, downstream resets and separate-donor geometry/scale recombination. | Hybrid answers and rescue/blocking are existing methods, not novelty by themselves. |
| [Frankenstein-Style Analysis](https://arxiv.org/html/2602.12395v1) | Mid/late localization, parameter-update geometry, coarse checkpoint merging and training-time freezing. | A coarse merge, or merely reducing its size, is insufficient. Identify the operation changed. |
| [Same Reward, Different Skills](https://arxiv.org/html/2610.01908v1) | Cued readout, find-and-bind, relational discovery, counterfactual consistency and controlled visual-access training. | A new behavioral hierarchy or another grounding-versus-accuracy gap is insufficient. |

CARVE's abstract is verified through the shared browser page. Its PDF retrieval
still failed; stop retrying and request the PDF from the user before the final
methods-level overlap decision. The chart/Frankenstein HTMLs were checked directly;
Same Reward was checked in the preceding research review. Reported findings are
the authors' results, not reproductions. Do not claim that the unread CARVE methods
exclude the proposed operation-level analysis.

## 3. Three linked, falsifiable hypotheses

These are one prospective paper, not three parallel paper projects.

| Hypothesis | Evidence needed | Result that rejects the proposed explanation |
| --- | --- | --- |
| **H1: Stage-linked binding/routing change is selective.** MPO→additional GSPO changes query-dependent selection or its tested causal dependence beyond direct readout. | Matched families show a material change in binding and/or the preregistered routing endpoint, under common prompts; color/edge-readout and text-fact controls do not account for it. | Effects track direct readout, color priors, invalid answers, termination or prompt mode; no material routing difference. |
| **H2: The changed operation is conditional Q/K routing, not merely value features or global image-attention mass.** | Q-only, K-only and joint interventions predict recipient-specific route flips/cancellation, across independently changed donor colors, with recipient V and image-attention mass held fixed. Compare V-only interventions and matched shams. | Donor-color copying, indiscriminate disruption, failure of cancellation/role specificity, or V/other feature changes accounting for the effect. |
| **H3: Bounded adjacent-stage weight differences contribute to H2.** | Selected projection-group rollback/replay have opposing selective effects; identical-input module replay and mediator clamp/rescue connect the weight effect to the tested route. | Only whole-layer merges work, control deltas work equally well, cued/text controls deteriorate similarly, or the routing mediator does not account for surgery effects. |

H1 alone is prior-covered. H2 alone remains an ordinary mechanistic result unless
the computation is genuinely explained. H2 + H3 + untouched confirmation is the
candidate contribution. The hypotheses do not assume improvement, a universal
head, a low-rank update, or that all models use the same route.

Failure at a tested site does not prove absence of a distributed mechanism.
Failed replay rejects portable sufficiency of that group, not every possible
background-dependent mechanism. Those outcomes still stop this deadline path.

## 4. Cheap decisive assay: reversible visual relation

Use a single 448px diagram with three shape-identified sockets and a directed
cycle. Each socket has an independently assigned color chip. No OCR, chart axis,
printed answer, Blender pipeline or second image is needed.

Example: triangle → circle → square → triangle; chips red / blue / green.

| Image edges | Query from triangle | Required color |
| --- | --- | --- |
| Original | Follow one arrow | blue |
| Original | Follow one arrow in reverse | green |
| All arrows reversed | Follow one arrow | green |
| All arrows reversed | Follow one arrow in reverse | blue |

Cross these four cases with a second color assignment, such as yellow / purple /
orange, to make **eight image–question cases per source family**. Counterbalance
anchors, colors, shape positions, arrow directions, layouts and paraphrases.
Maintain node positions while reversing arrowheads within a family. A deterministic
scene program verifies each answer; human inspection checks actual readability.

The high-level computation is

$$
S=(a+ed)\bmod 3,\qquad Y=c_S,
$$

where $a$ is the anchor, $e$ the visual edge orientation and $d$ the query traversal
direction relative to that orientation. Flipping $e$ or $d$ changes the selected
neighbor; flipping both cancels. This algebra does not assume that neural Q/K
states implement it.

For a red/blue/green recipient, use donors with disjoint colors:

- Query-direction/Q-only substitution predicts **green**, not a donor color.
- Edge-direction/K-only substitution also predicts **green**.
- Joint substitution predicts cancellation back to **blue**.
- V-only donors change colors without changing the selected referent, if a
  separable value representation exists at the tested site.

Repeat across color assignments and intervention directions. Same-answer/wrong-role
donors and joint cancellation rule out simple donor copying and generic answer
flipping. Bidirectional ViT contextualization means ROI states need not be pure
edge/value variables; specificity is an empirical requirement.

If the diagram is unreadable, simplify only on the engineering set. Do not search
many tasks on validation results. A predeclared simpler left/right adjacency assay
is allowed as an engineering fallback, with a new locked protocol and split.

### Intervention implementation requirements

- Start with native pinned InternVL releases and the slow tokenizer. Align prompt
  lengths, token positions and rotary coordinates; use only pre-answer states.
- Interchanges are within checkpoint. No raw cross-checkpoint or cross-size state
  transplantation; matching architectures do not imply aligned representations.
- For selected attention rows, replace only conditional routing over allowed image
  columns. Preserve recipient values, text-side contributions and total image mass:

$$
\widetilde A_{ij}=m_i^{(r)}
\frac{\exp(\widetilde z_{ij})}{\sum_{k\in\mathcal J}\exp(\widetilde z_{ik})}
\quad(j\in\mathcal J),\qquad
m_i^{(r)}=\sum_{j\in\mathcal J}A_{ij}^{(r)}.
$$

  Outside $\mathcal J$, keep the recipient attention row unchanged. Compute
  substituted logits from Q-only, K-only or joint donors after correctly handling
  normalization/rotary positions. Do not retain all heads' dense attention tensors.
- Confirm capture-only, no-op and self-patch parity against native inference;
  disable intervention KV caching initially and recompute downstream states.
- GQA shares K/V heads. Respect the actual query-to-KV group mapping; do not
  advertise independent edits to a shared K head as isolated query-head edits.
- Compare matched real control deltas and perturbation/entropy/location shams.
  A negative intervention can reflect off-manifold damage, not missing information.

## 5. Preliminary experiments and internal gate

Initial scientific models: **2B and 8B MPO/final pairs**, four checkpoints loaded
sequentially. The 8B transition is the primary scientific comparison; 2B is cheaper
engineering and cross-size replication. Do not let a small-model null stand in
for the larger model. Defer CPT/SFT, MiMo and 14B until the pilot earns expansion.

Use 24 engineering families, plus 40 discovery and 60 pilot-validation families.
The 100 pilot families are development data in the final study, not final
confirmation. After this screen, use **200 new untouched confirmation families**.
Opening the pilot-validation set fixes sites and thresholds for that screen;
any later change is documented development and evaluated on the fresh set.

| Date | Experiment | Required output / stop condition |
| --- | --- | --- |
| Oct 31–Nov 1 | **P0: Engineering and provenance.** Validate image preprocessing, token mapping, native/capture/self-patch parity, common answer prefix, tensor differences and peak memory. | Working bounded loader/hook path, data audit and timing. Unresolved parity by Nov 1 is a no-go for the sprint. |
| Nov 1–2 | **P1: Behavioral qualification.** Run the four checkpoints on discovery families and direct color/edge-readout plus matched text-fact controls. | Identify floor/ceiling, parsing and prompt confounds. No readable/usable larger-model assay means stop rather than expand benchmarks. |
| Nov 2–3 | **P2: Routing screen.** Bounded layer/group search only on discovery families; freeze at most two candidate groups per size. | Recipient-color predictions, role specificity and joint cancellation beyond matched shams. No selective signature means stop H2. |
| Nov 3–4 | **P3: Weight and mediator test.** Rollback/replay, identical-input module comparison, correct mediator rescue/clamp and matched control resets. Evaluate the frozen screen on pilot-validation families. | Weight → operation → answer connection, not merely a sensitive head or a layer merge. |
| Nov 4–5 | **Decision, not abstract fabrication.** Inspect family-paired intervals, controls, failures and source overlap. | Conditional continuation only if the material mechanistic signature survives; otherwise defer or reframe outside this deadline. |

For group $G$, define $\Delta W_G=W_{\mathrm{GSPO},G}-W_{\mathrm{MPO},G}$.
Replay inserts this group difference into MPO; rollback removes it from GSPO.
All other weights remain the corresponding recipient checkpoint's weights.
Replay both parameter versions on identical module inputs to distinguish changed
operations from upstream state drift. Keep selected tensors in memory; do not
save or repeatedly reload full hybrid checkpoints.

If discovery implicates associated Q/K normalization parameters, include and
identify them explicitly in the selected operation rather than silently treating
projection matrices as the entire route. Clamp/rescue distributions must come from
native same-input runs in the corresponding recipient background before the
selected weight edit, not gold-designed attention or donor answer-bearing states.
Compare wrong-role rescues and control resets, and check for floor effects: a
generally destructive clamp cannot establish specific mediation.

### Minimum pilot signal — proposed filters, not power guarantees

Before opening validation, freeze one primary 8B stage contrast in hybrid-target
accuracy relative to matched shams. A practical large-effect screen is approximately
**15 percentage points** in this stage-linked routing contrast, with a family-paired
interval excluding zero. Opposing rollback/replay effects of approximately
**10 points** are a practical weight-contribution screen. Record raw effects, not
unstable per-item fractions of a possibly tiny baseline gain.

These magnitudes are deadline filters, not theoretically privileged thresholds.
They must be frozen or revised from discovery variance before validation. A
behavioral gain is not required if a material change in tested routing dependence
is established, but that result must be described as changed dependence, not
improved grounding. Do not substitute a post-hoc alternative endpoint to pass.

Correct recipient-payload/cancellation predictions and weight–mediator dependence
are mandatory; a significant accuracy difference cannot replace them. Perception
and text controls need preregistered preservation margins, such as ±5 points.
Non-significance is not equivalence, and saturated controls are not decisive.
Pilot imprecision permits further measurement, not a claim of selective preservation.

### Metrics, units and cost

- Score complete candidate color sequences. If every candidate is verified to be
  exactly one token under the frozen answer prefix, a single prefill supplies all
  candidate scores; otherwise budget separate sequence scoring explicitly.
- Candidate ranking is not unrestricted generation accuracy. Audit free answers,
  native prompt/thinking settings, termination and truncation separately. No claim
  about long-CoT reasoning follows from direct-answer interventions alone.
- Report all valid generated families, including unpatched failures. Jointly-correct
  subsets are secondary with their selection funnel, not the primary denominator.
- Bootstrap paired whole families and simulate power using discovery variance.
  Account for testing multiple candidate effects; do not treat layers, queries or
  donor variants as independent observations. Degenerate zero-discordance intervals
  require appropriate conservative/exact checks, not a zero-width bootstrap claim.
- For illustration, a binary paired outcome with 20% discordance has approximate
  MDE $2.8\sqrt{0.2/G}$: **16.2 points at 60 families**, **8.9 at 200**.
  The routing difference-of-contrasts needs its own variance/power simulation.

Planning counts: four checkpoints × 100 families × (eight task cases + four
diagnostics) = **4,800 baseline scored conditions**. Twelve primary intervention/
control conditions per validation family would add **2,880**, plus a bounded
discovery search, engineering and native-generation audits. Finalize the actual
matrix from measured costs without dropping required controls. Scored conditions
are not necessarily forward passes.

Implementation refinement: the stronger same-palette/wrong-anchor routing controls
increase H2 to **17 conditions per family/site/checkpoint**. H3 has **17** including
its task controls. The immutable schedule and recorded actual forward counts,
rather than this earlier illustrative arithmetic, determine execution budgets.

$$
T=\sum_m(T_{\mathrm{download/load},m}+N_{s,m}t_{s,m}
       +N_{i,m}t_{i,m}+N_{g,m}t_{g,m}).
$$

Measure these terms on P0. A suggested pilot budget is **24 GPU-hours**, not a
throughput promise. Include downloads, host RAM, disk, activation memory and
human auditing. Use an actually free remote GPU; do not compete with an active
pinned inference worker. Do not begin a sweep whose measured cost cannot meet
the internal gate. No new large-model training is part of this sprint.

## 6. Conditional path to the full paper

| Date | Expansion, only after the pilot passes |
| --- | --- |
| Nov 5–8 | Freeze the final protocol. Begin 200 fresh confirmation families at both sizes; audit the original MiMo SFT→RL pair and replicate operation/weight tests using its own discovery sites. Add CPT/SFT behavioral context sequentially. |
| Nov 6–10 | Audit a separate photographic/visual-domain transfer set. Printed-board photographs test rendering robustness, not broad natural-image generality. A second relational object task strengthens the claim but cannot be tuned on its test outcomes. |
| Nov 9–10 | Enrollment decision: credible primary causal evidence, a usable 8B result, clear contribution beyond priors and a coherent draft. Complete all author profiles/enrollment; use only factual preliminary claims. |
| Nov 11–14 | Finish frozen confirmation, required controls, uncertainty and writing. If MiMo or natural transfer fails, narrow claims and assess whether the remaining contribution is still substantial. No automatic publishable-negative fallback. |
| Nov 15–16 | Independent artifact/source/claim audit, anonymity, mandatory disclosure and submission. Central evidence must already be in the paper. |
| Nov 17–23 | Supplement: code, full condition tables, reproducibility, detailed controls and additional analysis. Not a plan to obtain the missing central result after Nov 16. |

The broader target remains the two InternVL quartets plus MiMo pair (ten
checkpoints), but exhaustive causal sweeps are needed only for justified adjacent
comparisons. Illustrative full baseline cost is ten × 300 × (eight + four)
= 36,000 scored conditions; three endpoint pairs × 200 × twelve adds 14,400.
Actual cost depends on candidate scoring, sites, controls and native audits.
14B, training-time freeze experiments, broad probe/attention atlases and multiple
benchmark sweeps are excluded from the deadline-critical path.

Released endpoint surgery identifies contributions of differences between trained
systems. It does not isolate the causal effect of the GSPO optimizer, its rewards,
data mixture or compute budget. Cross-size replication is not a randomized size
experiment; shared head indices or identical effect directions are not assumed.

## 7. Core abstract — deliberately result-free for now

**Provisional enrollment wording, not a results abstract:**

Multimodal post-training can improve answers without revealing which internal
operation changed. This study asks whether stage-linked changes in visual referent
selection arise from query-conditioned evidence routing rather than improved
attribute perception or answer policy. We design a single-image assay that
independently varies visual relation direction, query direction and visible
attributes; reversing both directions preserves the required answer. The planned
analysis compares released InternVL3.5 checkpoints at two sizes under common
inference conditions. Within each checkpoint, routing interventions preserve
recipient value states and image-attention mass, testing recipient-specific
counterfactual answers rather than donor-response copying. Targeted rollback and
replay of adjacent-stage projection differences then test whether the identified
computation contributes to the checkpoint change. Direct attribute reading, edge
recognition, text-fact reasoning and native-answer audits test competing
explanations. Confirmation uses untouched source families and a separately audited
visual transfer setting. The objective is a bounded, falsifiable explanation of
one post-training transition, not a universal algorithm-level grounding claim.

Before using a results-based abstract, fill only verified statements:

- Transition/models, primary endpoint difference and family-level interval.
- Explained operation and specific weight intervention, with uncertainty.
- Control preservation actually demonstrated, not merely nonsignificant.
- Confirmation/transfer scope and important negative boundaries.

Never write “we identify,” “we show” or invented effect sizes for an unrun analysis.
If only behavioral results exist at enrollment, say only that; do not imply that
future causal experiments have already succeeded.

## 8. Official deadline and administrative checks

The [CVPR 2027 dates](https://cvpr.thecvf.com/Conferences/2027/Dates) and
[author guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
were checked on 31 October:

- **Nov 10, 23:59 AoE:** enrollment/title/abstract, all co-authors, completed
  profiles and enrollment. The requested four–five-day horizon is our internal
  Nov 4–5 decision, not the official enrollment date.
- **Nov 16, 23:59 AoE:** full paper. Title/abstract may be revised until this
  deadline; authors cannot be added/removed after Nov 10.
- **Nov 23, 23:59 AoE:** supplementary material; reviewers need not read it.
- Check OpenReview profiles now: non-institutional account moderation can take
  up to two weeks. Check reciprocal-reviewer eligibility/quotas if submitting
  more than one paper without an eligible reviewer.
- Eight-page main-paper limit excluding references and the required **LLM Usage
  Statement**. Disclose actual GitHub Copilot assistance in ideation, literature
  synthesis, planning and any later code/writing; all claims remain human-reviewed.
- Cite influential/contemporaneous work. The contemporaneous-work policy does
  not excuse close duplication or missing attribution.
- Keep submission artifacts anonymous and self-contained. Do not link the
  author's identifiable repository as contributed submission content. Do not
  publicly post this internal venue-specific memo as a submission announcement.

**Current decision: GO for P0/P1; conditional GO for the mechanism pilot; no
commitment to a Nov 16 submission until the evidence and novelty gates pass.**