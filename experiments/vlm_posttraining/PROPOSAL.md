# Whose Evidence? How Post-Training Changes Visual Referent–Value Binding

## Status and recommendation

**Research proposal with a separately implemented [pilot](README.md), not a
validated research result.** Literature and public checkpoint provenance were
reviewed on 31 October 2026. Offline random-Qwen3 tests check the new pilot;
no actual VLM checkpoints were loaded or GPU measurements made. The active
text-study Python source is unchanged. Published results cited below are authors'
findings, not reproductions.

**Deadline refinement:** the readable CARVE abstract now establishes substantial
overlap with generic stage/size grounding and binding analyses. The narrower
[objective, preliminary hypotheses and deadline gates](SPRINT.md) supersede this
document as the execution priority. A task/scale sweep alone is not the planned
contribution; require an explained stage-linked computation and selective weight
intervention. No new VLM results are asserted.

**Conditional go for a small mechanism-discovery pilot; not yet a paper commitment.**
A collection of benchmark scores or attention heatmaps is insufficient novelty.
The proposed contribution must be a specific, reproducible explanation of how a
post-training transition changes query-dependent visual binding, with competing
perception and answer-policy explanations tested. Acceptance cannot be guaranteed.

Every prompt contains one image. Counterfactual donor and recipient images are
processed in separate executions. This studies selection and attachment of values
to referents within an image, not cross-image interaction or information sharing.

## 1. Research question and competing explanations

**Does multimodal post-training change how the model selects the queried object
and attaches the correct visible value to it?**

Use the high-level abstraction

$$
S=f(I,q),\qquad V=(v_1,\ldots,v_m),\qquad Y=V_S,
$$

where $S$ is the query-selected referent, $V$ the table of visible attributes, and
$Y$ the required answer. This is a hypothesis to test, not an assumed neural
architecture or exhaustive decomposition of model behavior.

| Explanation | Distinguishing prediction |
| --- | --- |
| Better perception | Direct, explicitly cued attribute readout improves; effects follow value-bearing visual states. |
| Better or different binding | Query-dependent referent selection and selector–value interchange improve or change selectively, beyond cued readout. |
| Different answer policy | Parsing, termination, output priors or confidence explain the apparent gain; predicted hybrid answers are not selectively recovered. |

These explanations can coexist. Neither a probe's accuracy nor failure on a blank
image alone establishes the correct use of a particular object or attribute.

## 2. Closest literature and the novelty boundary

| Primary source | Relevant finding or method | What the proposal must add |
| --- | --- | --- |
| [Does RL Post-Training Ground Vision-Language Models? A Causal Measurement / CARVE](https://openreview.net/forum?id=xLUhlfoeKH) | Readable abstract reports solver-certified counterfactual edits, label-binding failures, regional patches, size-dependent gain mediation, frozen-ViT decoder localization and attention–causality disagreement. | This already occupies the generic grounding/binding/stage/size claim. Require a distinct explained operation and selective weight contribution, not another model-family replication. Full methods remain unread. |
| [Same Reward, Different Skills: When Multimodal RL Learns to Look](https://arxiv.org/abs/2610.01908) | Matched visual-access training, counterfactual scenes, cued readout, named-target binding and relational discovery. Benchmark gains can conceal grounding loss under the studied recipe. | Not another accuracy-versus-grounding dissociation or behavioral find-and-bind hierarchy. Identify a specific internal binding mechanism and its stage-linked change. |
| [What does RL improve for Visual Reasoning? A Frankenstein-Style Analysis](https://arxiv.org/abs/2602.12395) | Visual-token interventions, parameter comparisons, coarse checkpoint merging and training-time freezing. Reports mid/late changes across studied Qwen-based lineages. | Not merely mid/late localization or layer merging. Connect query-role-specific states to selective errors and targeted weight interventions. |
| [Inside VLM Chart Reading](https://arxiv.org/abs/2609.13745) | Chart-factor patches, prompt-series mediator resets and factorial geometry/scale recombination; v2 revised 28 October 2026. | Binding, downstream blocking and compositional donor tests are already established here. New objects or post-training columns alone are weak novelty. A distinct stage-dependent mechanism and natural-image confirmation must carry the contribution. |
| [Evidence-RL: Towards Evidence-intensive Visual Reasoning](https://arxiv.org/abs/2608.08021) | Evidence-token neutralization versus matched non-evidence regions; uses evidence intensity in GRPO. | Target-versus-control erasure is a baseline, not the new method. Test which referent's value is used and why. |
| [RL makes MLLMs see better than SFT](https://arxiv.org/abs/2510.16333) | Encoder evaluations, probes and optimization localization, prominently involving DPO/preference optimization. | Separate encoder improvement from binding; do not generalize one preference recipe to all online RL. |
| [Hidden in plain sight: VLMs overlook their visual representations](https://arxiv.org/abs/2506.08008) | Distinguishes available visual representations from their effective use. | “Encoded but not used” alone is not new. Establish a selective causal handoff rather than just probe–behavior disagreement. |
| [What's in the Image?](https://arxiv.org/abs/2411.17491) | Query-text states can retain visual information; studies cross-modal information flow. | Removing later image tokens is not equivalent to removing all visual evidence. Include possible text-side binding states. |
| [Attention Sinks in VLMs Are Borrowed, Not Seen](https://arxiv.org/abs/2610.15855) | Attention redistribution can leave answers unchanged. | Attention is descriptive support, not the primary evidence-use metric. |

Additional boundaries: [Counting Circuits](https://arxiv.org/abs/2603.18523) already
studies visual counting interventions; [Inducing Causal Structure for Interpretable
Neural Networks](https://arxiv.org/abs/2112.00826) provides causal-abstraction and
referent/pointer-value precedents; [activation-patching best practices](https://arxiv.org/abs/2309.16042)
motivate metric and corruption controls.

**CARVE access update:** its full abstract was read through the shared browser
page on 31 October. It is a close prior, not an unknown contribution. The PDF
retrieval still failed; stop repeated attempts and request the PDF from the user
before a final methods-level overlap assessment. No novelty clearance is claimed.

### Potential paper contributions — only if observed

1. A falsifiable, query-role-specific causal binding abstraction on fresh visual
   counterfactuals, including hybrid answers that rule out simple donor-answer copying.
2. A selective post-training change in a tested selector→evidence pathway,
   connected to identified weight changes rather than a coarse layer heatmap.
3. Held-out natural-image/OCR and second-lineage confirmation under a frozen
   analysis protocol, including failure cases and uncertainty.

The experimental techniques themselves are not claimed to be new. If the closest
prior already explains the same mechanism, reframe or stop.

## 3. Open checkpoint lineages

### Primary: InternVL3.5

The [official release card](https://huggingface.co/OpenGVLab/InternVL3_5-8B) and
training description provide
multimodal continual pretraining → SFT → MPO → online GSPO endpoints.

| Released stage | Pinned 2B-labelled scale comparison | Pinned 8B-labelled primary model |
| --- | --- | --- |
| Multimodal CPT endpoint | [OpenGVLab/InternVL3_5-2B-Pretrained](https://huggingface.co/OpenGVLab/InternVL3_5-2B-Pretrained/tree/ce387f38cd21603625cdeb6c37f34a2bb89a771a) | [OpenGVLab/InternVL3_5-8B-Pretrained](https://huggingface.co/OpenGVLab/InternVL3_5-8B-Pretrained/tree/e21d3bab4089ed923c08072a1eefc8b5faf41cd8) |
| SFT | [OpenGVLab/InternVL3_5-2B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-2B-Instruct/tree/a1e50c526faea805730148e7b1f2af18abe9e4b2) | [OpenGVLab/InternVL3_5-8B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-8B-Instruct/tree/27506812aa9914804018996329e895977ee2d0c8) |
| MPO | [OpenGVLab/InternVL3_5-2B-MPO](https://huggingface.co/OpenGVLab/InternVL3_5-2B-MPO/tree/bf5ffd82a66a10df7de9fe520239a837c8808798) | [OpenGVLab/InternVL3_5-8B-MPO](https://huggingface.co/OpenGVLab/InternVL3_5-8B-MPO/tree/bb33f08275aa566f3442c1f5002512ffb66d7952) |
| Additional online GSPO | [OpenGVLab/InternVL3_5-2B](https://huggingface.co/OpenGVLab/InternVL3_5-2B/tree/7d7bd7bcc35b77b391883ebd7f9686cfa0e25cc0) | [OpenGVLab/InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B/tree/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79) |

- **The larger quartet is part of the primary scientific design**, not a fallback
  used only if small models fail. All four 8B releases are public, ungated,
  full BF16 checkpoints with Apache-2.0 card metadata and explicit adjacent parents.
  Each has **8,528,318,464 total stored parameters**, including vision/projector
  components; the 8B label is not an exact total multimodal parameter count.
- The 2B-labelled model has approximately **2.348B total parameters**. Use the
  matched 1B-labelled quartet for hook engineering if useful, not as the only
  scientific model when behavior is at a floor.
- “Pretrained” means the VLM's released multimodal CPT endpoint before its
  subsequent SFT; its Qwen3 language initializer already has post-training ancestry.
  It is not a completely unaligned foundation model.
- **MPO is DPO + BCO + LM/SFT loss**, not pure DPO.
- The final release adds online **GSPO**. Make **MPO→GSPO** the primary adjacent
  online-stage comparison; SFT→final conflates two stages.
- Published SFT/MPO recipes train vision encoder, projector and language model.
  The audited online code freezes the vision model, leaving projector/LM trainable.
  **Actual released weight equality still requires a component audit.**
- Native architecture/tokenizer mappings match; templates and length defaults
  differ. Freeze common prompting/preprocessing for the primary assay and audit
  native prompting separately. Do not mix native releases, -HF conversions or
  -Flash models; Flash changes compression and adds training.
- Native cards use `AutoModel`/`AutoTokenizer` with custom remote code. Pin and
  inspect that code. Baseline preprocessing uses 448px tiles, bicubic resizing,
  ImageNet normalization and dynamic tiling; preserve a pixel→tile→merged-token map.
- The inspected online scorer rewards answer and format, not verified intermediate
  reasoning or visual faithfulness. Endpoint comparisons do not isolate the reward,
  optimizer, data mixture or training duration.

#### Larger-model metadata audit and scale design

The pinned 8B quartet shares `InternVLChatModel` custom code and Qwen3 LM dimensions:
36 layers, hidden size 4096, 32 query heads, 8 KV heads, head dimension 128 and
151,936 embedding rows. Its InternViT has 24 layers, hidden size 1024, 448px images
and 14px patches. The indexed tensor names match across all four releases (745
keys), as do stored weight byte totals. These checks do **not** establish tensor
equality, effective loader defaults or numerical inference parity.

The native slow-tokenizer vocabularies contain the same 151,643 token→ID entries;
the final vocabulary has different serialization but zero mapping differences.
Merge rules, added-token assets/decoder and special-token maps also match. However,
the final release alone includes a fast-tokenizer asset and omits the tokenizer's
chat template. Declared tokenizer maximum length changes from 32,768 at CPT/SFT to
14,588 at MPO/final, and raw configs differ in attention/default fields. Use the
pinned native slow tokenizer and explicit prompt, stop-token, attention, precision
and length settings; test effective parity rather than treating these raw files
as identical. No remote model code or weight tensors were executed in this audit.

Run the frozen behavioral/causal endpoints on the **same source families at both
2B and 8B**. Discover any size-specific sites only on discovery families; keep the
causal abstraction and confirmation protocol shared. Do not transplant raw states
between sizes or assume identical layer/head indices. Report paired stage changes
within each size and their uncertainty. This is cross-size replication, not a
controlled causal estimate of changing parameter count alone: training recipes and
model architecture also differ between sizes.

#### Optional larger dense replication: InternVL3.5-14B

All four releases are public, ungated full BF16 checkpoints with explicit serial
parent metadata and Apache-2.0 cards. Each has **15,119,523,840 total parameters**,
despite the 14B label. Availability/revisions/counts were checked; the detailed
processor/config/tensor and GPU audits remain pending for this size.

| Released stage | Pinned checkpoint |
| --- | --- |
| Multimodal CPT endpoint | [OpenGVLab/InternVL3_5-14B-Pretrained](https://huggingface.co/OpenGVLab/InternVL3_5-14B-Pretrained/tree/98672894a9bb471d8e5899ce94096f6a028d0f28) |
| SFT | [OpenGVLab/InternVL3_5-14B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-14B-Instruct/tree/72c82460a2c05a3bc2b47be230c44edc88e86ed4) |
| MPO | [OpenGVLab/InternVL3_5-14B-MPO](https://huggingface.co/OpenGVLab/InternVL3_5-14B-MPO/tree/0ddec3e5709d390fa66b7ca72a1cabea08df302c) |
| Additional online GSPO | [OpenGVLab/InternVL3_5-14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B/tree/a1e37197b393ce9eec9df700fef65c11f4a6ffbd) |

Choose whether to include this size before opening its confirmation outcomes.
It is a resource-gated extension, not a replacement for independent-family
confirmation and not a model under a strict 14B-total-parameter cap.

#### Alternative for a language-base-initialized CPT→SFT comparison

The older InternVL3-8B triplet uses the released **Qwen2.5-7B base** initializer,
unlike InternVL3.5's post-trained Qwen3 initializer. Public full checkpoints and
explicit serial parent metadata were checked; total parameters are **7,944,373,760**:

- CPT: [OpenGVLab/InternVL3-8B-Pretrained](https://huggingface.co/OpenGVLab/InternVL3-8B-Pretrained/tree/1774db636867e4a7321330f5c10192aa32033f40)
- SFT: [OpenGVLab/InternVL3-8B-Instruct](https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct/tree/ddb3a169d5582e5c76e0809a128e55ab63686ada)
- MPO: [OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B/tree/853e3a797a661694b1b8ece0cb72dc2b23e3dac9)

This is a three-stage option, **not an online-RL quartet** and not a serial parent
of InternVL3.5. The CPT checkpoint is still a multimodal pretraining endpoint,
not a model trained from scratch with no prior language/vision ancestry.
Separate compatibility and license-file audits are required before execution;
its cards' Apache-2.0 frontmatter and MIT license prose are inconsistent.

### Independent confirmation: original MiMo-VL pair

- [XiaomiMiMo/MiMo-VL-7B-SFT](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-SFT/tree/8f10075987cb64cd18751217865fe94542ddd548)
- [XiaomiMiMo/MiMo-VL-7B-RL](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL/tree/c64b51cbea3d815a027c7862736411a7b6a345ae)
- Approximately **8.306B total parameters**, Qwen-ViT + MiMo LM. The
  [report](https://arxiv.org/html/2506.03569v1) describes mixed on-policy rewards,
  including reasoning, grounding/counting and learned preference rewards.
- Parsed configurations, tokenizer/processor assets and templates match in the
  reviewed pair. Its RL freeze mask is not established; audit weights rather than
  assuming a frozen encoder. Do not mix original releases with 2508 editions.
- Both original releases are public, ungated full BF16 checkpoints with MIT card
  metadata; revisions/counts were rechecked during the larger-model audit. The
  cards' narrative establishes SFT→RL; their Hub `base_model` fields self-reference
  and cannot independently prove lineage. No pre-SFT VLM endpoint is released.
- This is a different architecture/training project, not an independent randomized
  replication of one algorithm. No pure pretrained VLM stage is inferred.

Optional pure-DPO auxiliary: LLaVA-1.5-7B → RLAIF-V-7B. This adds loader/licensing
complexity and checkpoint-selection overlap with hallucination benchmarks; defer
it unless the primary result specifically needs a DPO replication.

**Recommended core: 2B + 8B InternVL3.5 quartets and the original MiMo SFT/RL
pair — ten scientific checkpoints, loaded sequentially.** The 1B model is optional
engineering scaffolding; 14B and the older InternVL3 triplet are scoped extensions.

## 4. Data: a decisive factorial binding assay

Ask: **“What color is the cube that matches the sphere's size?”**
Keep two cubes at stable locations. Independently change the sphere's size
(selector A versus B) and the two cubes' colors (visible value table).

| Scene | Selector | Values at A / B | Correct answer |
| --- | --- | --- | --- |
| $I_{00}$ | A | red / blue | red |
| $I_{10}$ | B | red / blue | blue |
| $I_{01}$ | A | green / yellow | green |
| $I_{11}$ | B | green / yellow | yellow |

Use $I_{00}$ as recipient and $I_{11}$ as donor:

- Selector-only interchange predicts **blue**, not donor answer yellow.
- Value-table-only interchange predicts **green**, not donor answer yellow.
- Coherent joint interchange predicts **yellow**.
- Taking selector states from $I_{10}$ and value states from $I_{01}$ also predicts
  **yellow**, which is neither single donor's answer.

Balance directions, selector states, candidate colors and positions. Add nuisance
edits, same-answer/wrong-role donors and held-out scene/template compositions.
Early ViT states are contextualized, so spatial tokens are not guaranteed to carry
only their nominal region's content. Later selector states might also encode answer
preferences: role specificity must be demonstrated, not assumed.

### Source families and transfer data

1. **Fresh CLEVR-like renders:** exact scene programs, object masks, independently
   crossed selectors/values and fixed-camera counterfactuals. Fresh images avoid
   exact-image reuse, not familiarity with CLEVR-like tasks. Validate renderer
   dependencies, visibility, uniqueness and collateral pixel changes.
2. **Audited TextVQA/OCR referents:** image + question, without supplied OCR or gold
   facts. Select images with multiple text-bearing referents. Re-annotate edited
   answers, include same-text sham renders and matched distractor edits. Test
   perspective, font, spacing, background and human readability.
3. **GQA is a reserve**, not another mandatory sweep. Scene graphs support some
   attribute/reference edits but cannot certify every absence/count/collateral change.
4. Prefer a small set of **new photographed arrangements/signs with randomized
   values** as a stronger exact-image holdout if natural-image editing proves fragile.

Sources: [CLEVR data](https://cs.stanford.edu/people/jcjohns/clevr/),
[CLEVR generator](https://github.com/facebookresearch/clevr-dataset-gen),
[TextVQA](https://textvqa.org/dataset/),
[GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html).
Check data/code licences and underlying photograph rights separately. Broad
pretraining corpora prevent comprehensive contamination certification.

**No benchmark supplies 300 ready-made, valid factorial counterfactual families.**
Editing and annotation quality, not just GPU capacity, are major feasibility gates.
Do not lead with charts: the chart-binding prior is particularly close.

## 5. Experiment matrix

| ID | Experiment | Scientific purpose and required output |
| --- | --- | --- |
| E0 | Lineage, component and numerical audit | Pins, weight differences, template/processor parity, capture-only and self-patch checks. Prerequisite, not a scientific finding. |
| E1 | Behavioral binding versus cued readout | All-endpoint and pair correctness, selector changes, value changes and nuisance invariance. Establish whether the task is learnable and which stage changes behavior. |
| E2 | Perception and answer-policy controls | Explicitly cued attribute/OCR readout, image-removed/blank/mismatched controls, verified-facts diagnostic, formatting and termination audits. Test competing explanations. |
| E3 | Region-specific causal effects | Selector-anchor and value-region patches/erasures versus matched non-evidence regions. Save signed answer-score changes; generic erasure is only a baseline. |
| E4 | Selector/value activation interchange | Within-checkpoint selector-only, value-only, joint-cross and joint-coherent interventions. Test recipient-specific hybrid answers rather than donor copying. |
| E5 | Necessity, restoration and downstream blocking | Rescue an earlier selector intervention, then reset a later candidate state. Compare role/location-matched controls on untouched families. Establish dependence on tested sites. |
| E6 | Selective post-training weight interventions | After E4/E5 succeed, roll back identified stage deltas and insert them in the earlier endpoint. Compare size/location/norm-matched deltas and unrelated-task retention. Connect the mechanism to weights, not optimizer-level causality. |
| E7 | Generalization and robustness | Frozen protocol on OCR/new photographs and MiMo; hold out layouts/templates; verify prompt, output-budget, token-mapping and precision sensitivity. |

E1–E3 are diagnostics, not the paper by themselves. E4–E6 must produce the distinct
mechanistic contribution. E7 tests its scope. Do not select sites by maximizing the
held-out stage difference or pool repeated prompts as independent examples.

## 6. Interventions, metrics and controls

### Primary metric

For checkpoint $\theta$, a declared donor/recipient distribution $\mathcal D$ and
interchanged role $J$, measure interchange intervention accuracy:

$$
\mathrm{IIA}_{\theta,\mathcal D}
=\Pr_{(b,d,J)\sim\mathcal D}
\left[\hat y_{\theta}^{\mathrm{patch}_J}(b,d)
=y_H(b\leftarrow_J d)\right].
$$

Report **candidate-scored IIA separately from free-generation accuracy**. Score
complete candidate-answer sequences, not just their first tokens. Save individual
candidate scores and signed margin changes. Use balanced answers and a frozen
answer-scoring policy; a candidate list does not measure unrestricted output quality.

Report unconditional outcomes and a separately labelled jointly-correct subset,
with the complete selection funnel. Normalized recovery is secondary: disclose
small denominators and retain negative/overshooting effects without clipping.
High IIA supports the specified abstraction on the tested distribution, not a
unique circuit, a complete decomposition or universal abstraction.

### Required validity controls

- **Same-checkpoint interventions first.** Raw cross-stage activation transfer is
  not the primary assay; matching architectures do not guarantee aligned coordinates.
- Capture-only, no-op and self-patching must preserve outputs/logits under declared
  tolerances. Verify hooks are removed even after errors.
- Keep image shapes, tiling and token positions matched. Freeze the primary tile
  budget; audit alternate valid preprocessing separately.
- Capture only appropriate image/question prefill states, not gold-answer-bearing
  donor states. Initially disable KV caching and recompute all downstream states;
  validate any cached intervention path independently before use.
- Match random/control sites by location, token count and perturbation scale where
  relevant. Include both directions and coherent-image reference runs.
- For erasure, compare target and selector-anchor evidence against the same edit
  process on matched non-evidence regions. If evidence removal makes the question
  unanswerable, evaluate support withdrawal/abstention rather than an invalid gold.
- OCR sham edits, distractor edits and blinded human audits address editing artifacts.
  Failure of an artifact classifier does not establish artifact absence.
- Inspect unrelated candidates/tasks to distinguish selective binding from general
  confidence suppression or catastrophic disruption.
- Probes and attention are secondary. A null local patch can miss distributed or
  redundant pathways; successful interchanges can create off-manifold states.
- Direct-answer findings do not automatically describe native long-CoT reasoning.
  Compare prompting/budget profiles explicitly; count truncation and invalid outputs.

## 7. What “300 samples” means

Start with **300 independent source-question families in the controlled assay**:
100 discovery, 200 untouched confirmation. A family contains all selector/value
variants, nuisance edits, donors and prompt variants. Prefer distinct source scenes
and one anchor question per source. Split and resample at the family level.

Use a separate, independently audited OCR transfer set only after the controlled
pilot succeeds; 300 families there is a target, not an assumed ready dataset.
Do not split 300 across many benchmarks and claim each is well powered.

Freeze sites, masks, templates, metrics and exclusions before confirmation. Group
bootstrap entire source families, preserving checkpoint pairing. Estimate variance
on discovery data and simulate power for the final grouped design.

For a family-level paired difference $D_g$, a rough two-sided 5%-level, 80%-power
minimum detectable effect is

$$
\mathrm{MDE}\approx 2.80\,s_D/\sqrt G.
$$

Illustration only: a binary paired endpoint with 20% checkpoint disagreement and
200 independent confirmatory families has an MDE near **8.9 percentage points**.
Detecting 5 points would require roughly **628 confirmatory families**, before
multiplicity or filtering. Repeated generations/edits do not supply new independent
families. A nonsignificant difference is not evidence of equivalence; choose margins
in advance. Item uncertainty does not estimate training-seed variability.

## 8. Hardware, software and durable evidence

One RTX PRO 6000 Blackwell 96GB is a credible starting point for bounded inference
and selective interventions. Approximate **weights-only**, decimal-GB footprints:

| Model | BF16 | FP32 |
| --- | ---: | ---: |
| InternVL3.5, 2.348B total | 4.70GB | 9.39GB |
| InternVL3.5-8B, 8.528B total | 17.06GB | 34.11GB |
| InternVL3.5-14B, 15.120B total; optional | 30.24GB | 60.48GB |
| MiMo-VL, 8.306B total | 16.61GB | 33.22GB |

Actual peak GPU/host memory and throughput are unmeasured. KV cache, prefill,
tiles, activations, attention workspaces and loading peaks add memory. Load models
sequentially and retain selected states, not every head's full attention matrix.
The 8B/MiMo models are reasonable bounded single-GPU inference candidates on 96GB;
the 14B extension requires its own peak-memory gate. These are not validated fit
or throughput claims, and full-parameter training is outside the initial scope.
Validate precision and Blackwell compatibility against an FP32/eager reference;
do not assume the text-only model's parity results transfer to a VLM.

Use existing libraries: Transformers/native pinned model code and scoped PyTorch
hooks; safetensors for component audits; a validated renderer/Pillow for data;
NumPy/pandas and grouped bootstrap for analysis; ordinary plotting for figures.
Do not assume TransformerLens or another adapter supports the native multimodal
model without loader, preprocessing and numerical parity tests.

Store enough evidence to reproduce the conclusions without model reruns:

- Checkpoint/code/runtime/template/processor identities and checksums.
- Source image, counterfactual images, scene programs, rights, human audit and gold answers.
- Family split, donor/recipient IDs, region masks and pixel→tile→token mappings.
- Every intervention's checkpoint, site, layer/head/token support and control type.
- Raw outputs, complete candidate scores, validity/termination flags and timing.
- Small selected activation captures or reproducible capture recipes, not a full dump.
- Published per-trial results, completeness manifests, grouped statistics and figures.

Future VLM configuration/study/run identities must be separate from the active
text-only maths/repair plans. Reuse durable artifact practices after compatibility
tests; do not assume the existing storage/backup schema already handles VLM images
or large activation artifacts. No pinned scientific source should change underneath
an active worker.

## 9. Stop-gated execution roadmap

1. **Engineering gate:** audit releases; validate image/token mapping and bounded
   capture/self-patch parity on a small engineering set. Stop on unresolved mismatch.
2. **Behavior gate:** generate discovery families; test readout/binding and parsing
  across both the 2B and 8B quartets. Larger-model evidence is required even when
  the small model succeeds. Decide any optional 14B/older-InternVL3 extension
  before opening its confirmation data; do not silently exclude weak stages.
3. **Mechanism gate:** identify candidate selector/value sites on discovery data;
   require predicted hybrid answers and control specificity. Freeze the protocol.
4. **Confirmation gate:** evaluate untouched families; test restoration and selective
   downstream blocking with family-level uncertainty. Do not tune on these results.
5. **Stage-to-weight gate:** test identified weight deltas with matched controls.
   A broad mid/late merge without role-specific evidence does not clear novelty.
6. **Generality gate:** replicate on valid natural/OCR families and the original
   MiMo pair. Report boundaries rather than implying all VLMs share one circuit.
7. **Paper decision:** proceed only with a material contribution beyond the closest
   papers. Stop/reframe if the result reduces to attention, global image sensitivity,
   formatting, donor-answer copying or a known binding explanation.

Development can start with an engineering batch of a few dozen scenes. Measure
per-intervention cost there before budgeting the full sweep. No wall-clock estimate
is justified yet; image editing and analysis may dominate inference time.

### Optional controlled training — not required to start

If an algorithm-level claim becomes necessary, use a common small SFT parent,
identical fresh training data, explicit shared freeze/LoRA masks, matched training
budgets and multiple independent seeds. SFT versus GRPO objectives then become
controlled experimental factors. This is conditional additional work, not a promise
to reproduce large distributed training recipes on one GPU.

Released endpoints and weight interventions can explain differences between trained
systems; they do not isolate “the causal effect of RL versus SFT” as algorithms.

## 10. Venue and timing

A strong CVPR submission should offer a vision-specific mechanism, credible causal
tests and meaningful generalization. It need not propose a new training method,
but benchmark breadth cannot substitute for a distinct explanation.

The [official CVPR 2027 dates](https://cvpr.thecvf.com/Conferences/2027/Dates) list
registration on **10 November 2026**, submission on **16 November 2026** and
supplementary materials on **23 November 2026**, all AoE. The
[author guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
allow title/abstract revision until the paper deadline but require all authors'
profiles/enrollment by 10 November. Starting on 31 October without validated VLM
instrumentation or curated counterfactuals is a high-risk sprint, not a credible
unconditional paper commitment. Use the [4–5-day evidence gate](SPRINT.md) first
and select the submission cycle after the mechanism and novelty gates succeed.