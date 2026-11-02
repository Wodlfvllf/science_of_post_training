# Critical audit: identify the computation before searching for its location

2 November 2026. This is a revision of the research logic, **not a new results
claim or a replacement for the frozen C2 development protocol**. No additional
inference is authorized merely by writing this memo. Existing nulls remain in
the record. A larger run is not a substitute for an identifiable question.

## 1. The central inference we have not earned

The certified task is `destination = (anchor + edge × direction) mod 3`, followed
by lookup of the destination chip's color. This defines the external task. It
does **not** show that the model has an internal, task-independent destination
variable, or that its decoder implements that variable in Q/K routing.

The experiments repeatedly moved from this task algebra to a guessed neural
interchange. Self-parity checked the numerical implementation, not the semantic
validity of that interchange. Failed interchanges therefore often had several
explanations besides a false scientific hypothesis.

Likewise, a generated shape followed by a cued-color question is a **different
program** from a direct-color forward. Its success cannot establish that the
direct forward computed the right destination and lost it during color binding.

## 2. Three incompatible computations fit the observed results

Let `S_shape` and `S_color` denote the referents selected under the two tasks,
`L(image, cue)` the cued color lookup, and `D` the delivered-answer policy.

| Competing model | Direct-color computation | Shape then color | Why present observations do not distinguish it |
| --- | --- | --- | --- |
| **Task-dependent selection** | Color wording retrieves the anchor; shape wording invokes another selection procedure. | A generated shape explicitly addresses a different node for lookup. | Shape/color questions differ in task, wording and answer vocabulary. |
| **Shared selection, downstream interference** | A useful selected referent exists, but an anchor/default or output-type policy overrides its color readout. | Explicit cue bypasses the competing policy. | Cued lookup and chain gains do not measure the native selector in the direct-color pass. |
| **Heuristic shape guessing plus genuine lookup** | Anchor-color heuristic. | Shape/layout/direction priors sometimes guess correctly; lookup faithfully reads the guessed node. | Five correct shape answers out of eight, without full reversal tests, do not establish relational selection. |

These are not interchangeable descriptions of one mechanism. Each predicts a
different failure location and requires different causal evidence. Small null
patches do not choose among them.

## 3. Specific weaknesses in our earlier reasoning

1. **Marginals were overinterpreted.** Shape-correct 5/8 and chain-correct 5/8
   allow anywhere from two to five jointly correct cases. Case-level joint
   outcomes, including successful lookup of incorrectly selected nodes, are
   required. The documented four-family cue audit checks construction; it does
   not justify inferring a native internal mediator from equal aggregate counts.
2. **Shape and color were not symmetric measurements.** Three shape candidates
   versus six color candidates, different question syntax, and the ability to
   rule out the named anchor change difficulty and priors. Colors visible in an
   image supply a different effective choice set again. A uniform chance rate
   cannot establish a computation. Matched wording and a separately qualified
   balanced common answer-code scheme would address different parts of this
   problem; neither should silently replace the existing scorer.
3. **Some interventions were causally mismatched.** In this causal decoder, image
   tokens precede the question. Their K/V states cannot depend on later anchor,
   traversal or task words for a fixed image. Changing a later query is thus a
   structural image-K/V no-op, not a wrong-anchor K control. Color/edge changes
   can still change contextual image K. This must not be conflated with query
   dependence of image K.
4. **Tensor alignment was mistaken for semantic alignment.** A shape-task state
   may address a shape boundary while a color-task receiver needs its chip. Same
   checkpoint, token positions and RoPE coordinates do not guarantee a common
   selector code. A donor's certified task target is not evidence that the donor
   state actually represents that target. Same-palette donor targets can equal
   the desired hybrid answer, making answer copying a competing explanation.
5. **Intervention potency and outcome margins were secondary.** A failed argmax
   change with a small downstream perturbation is not comparable to an equally
   accurate null under a large perturbation. Conversely, large arbitrary damage
   does not establish semantic necessity. Native-target propensity, signed
   target/anchor margins and matched shams belong beside every semantic result.
6. **A broader sweep is not a fresh hypothesis.** The one-layer joint test was
   fixed locally, but it followed a sixteen-site screen and other row/wording
   tests on overlapping families. That history is exploration. The 20 C2-new
   engineering families are not globally untouched confirmation.
7. **Controls have narrower meanings than their names.** Direct64 is a token-cap
   control, not equal compute: uncached repeated prefixes and extra decoder calls
   differ, while vision features are cached. A discarded-call repeat is primarily
   a contamination/determinism check. Shifted prediction can accidentally become
   the true destination after an incorrect first prediction. Fixed selectorless
   cues are outcome-blind, not randomized or guaranteed balanced by role.

## 4. First extract the missing decomposition, without a GPU

For every existing original case, retain one row containing image/family ID,
edge/query/palette, true destination, generated shape, actual cued shape, direct
answer, chain answer, raw tokens/finish reason, and both of these grades:

- **Task success:** does the chain return the original destination color?
- **Lookup faithfulness:** does it return the color at the node it was actually
  told to inspect, even if the generated selector was wrong?

Partition all scheduled cases into invalid selection, correct selection/faithful
lookup, correct selection/unfaithful lookup, wrong selection/faithful lookup,
and wrong selection/unfaithful lookup. Preserve direct/chain rescues and
regressions jointly within family. Invalid format is a delivery failure, not an
automatic semantic-reasoning error. Do not select only native-correct cases as
the primary denominator.

For unchanged image/prompt/token IDs/checkpoint/runtime under deterministic
generation, record repeated-input consistency. Build a coverage table of all
three possible node cues and mark missing measurements as **unobserved**, not
failed. This reveals which questions actually need new inference.

The replication analyzer reconstructs raw grading and joint outcomes. This audit
adds an exhaustive invalid-selector/selector-correctness/lookup-faithfulness
partition and per-case cue identity, cue color and target coincidence. These are
descriptive analyses, not changed primary endpoints. Synthetic integrity tests
do not establish real replication execution, which remains unverified. Its token
counts are not a complete compute ledger. Its +15-point/positive-interval rule
is a documented development resource filter, not evidence that the true effect
exceeds 15 points, a mechanism test, or a publication criterion.

## 5. The decisive behavioral intervention is on cue content and context

Before another neural patch search, specify a small **exhaustive identity ×
context matrix**, rather than only correct-versus-wrong aggregate accuracy:

| Factor | Required levels and purpose |
| --- | --- |
| Supplied cue identity | All three shapes, in a fixed schedule independent of target correctness. Test whether output follows the indicated node. |
| Original relation context | Remove relation text; retain the true original anchor/query; retain a counterbalanced different anchor/query. Hold cue identity and the final color instruction fixed. |
| Direction algebra | Original/reversed image edges × forward/backward query. Single reversals change the referent; double reversal cancels. |
| Palette | Independent recipient colors, with expected answers recomputed by the scene program. Distinguish node selection from color priors. |
| Matched syntax | Shape and color tasks in both predeclared wording frames. If using a common answer code, provide all alternatives and counterbalance code assignments; no correct option is singled out. |

Not every factor needs a full Cartesian product immediately. Start with the
cue/context matrix using existing images, fixed wording and all cases. Stage
comparisons, reversal qualification and reference-frame checks are subsequent
requirements, not claimed controlled by that smaller matrix. Keep the existing
20-family C2 protocol unchanged; any added conditions get a distinct prospective
protocol and output identity.

**Operational chains still use model-generated identities only.** Exhaustive
cues are labelled interventions, not a model's inferred answer. Choosing the
gold cue afterward is an **oracle upper bound**, never an autonomous rescue.
Report target-color and cued-node-color outcomes separately under every cue.

The interpretable contrasts are:

- **Anchor interference:** with image/cue fixed, does reintroducing a competing
  anchor/query shift answers away from the cued-node color toward anchor color?
- **Cue compliance:** with context fixed, does changing only the cue move the
  output to the newly addressed recipient node, rather than a fixed default?
- **Relational selection:** do shape selections obey the edge/query reversal
  algebra and remain consistent under palette changes, rather than layout priors?
- **Stage interaction:** does the same prespecified interference/selection
  contrast differ between matched released endpoints? Shared behavior is not a
  GSPO-acquired ability and endpoint differences do not isolate an optimizer.

If context has no effect after an explicit cue, that is not proof of no native
interference: a dominant explicit instruction can bypass the original route.
If shape selection fails reversal/palette tests, abandon the accurate-selector
story before trying to transplant its supposed representation.

## 6. Only a qualified behavioral model earns a neural intervention

Suppose a donor selects node `s`. A proposed selector transfer must predict the
color at the *corresponding recipient node*, not the donor color, shape name or
an unspecified accuracy increase. The correspondence must be declared: spatial
socket, shape identity, or relational role. Independently exchange shape labels,
positions and palettes to distinguish these coordinate systems; existing aligned
render variants do not distinguish them all.

For an interchange, report the change in signed contrast
`log p(recipient counterfactual color) − log p(recipient native-target color)`,
relative to the self/sham intervention, alongside argmax, delivery audits, net
perturbation and joint reversal/cancellation results. Per-group RMS values cannot
be added to infer a net vector. Single-token and sequence/prefix estimands must
remain explicitly distinguished from unrestricted answer probability.

A credible progression is:

1. Demonstrate the predicted native operation on development data, including
   reversal/reference-frame failures that could falsify the model.
2. Identify one mediator whose intervention yields **role-specific recipient
   recombination**, not generic task switching, donor copying or destruction.
3. Test native necessity/blocking and correct versus wrong-role rescue, including
   informative readout/policy controls and the changed projector background.
4. If retaining a post-training mechanism claim, show bounded stage-weight
   rollback/replay changes **that same operation** on identical module inputs.
5. Freeze endpoints, exclusions, sites and margins before untouched confirmation.

Do not perform step 4 merely because a sensitive state or a tiny baseline stage
effect exists. A shared decomposition result supports neither the original H3
nor a replacement optimizer-specific mechanism.

## 7. Paper decision and current operational truth

The original stage-linked routing mechanism is **unsupported by the tested
evidence**, not globally disproved. The external chain result is a behavioral
lead, not a novel prompting method or native mechanism. Generic decomposition,
grounding gaps, regional mediation and checkpoint swaps overlap the priors already
listed in the sprint. CARVE's full methods still need review; no novelty clearance
is asserted. A defensible paper needs a distinctive explanation and evidence, not
just this experiment list or a statistically positive prompting comparison.

Read-only inspection of the newly shared sandbox during this audit found no
dataset, no isolated VLM environment, no replication report-start files and no
notebook job receipts. The controller import fails in that incomplete checkout;
its displayed notebook outputs are not launch/completion evidence. The twenty-
family batch is **not verified running or complete**. Drive was not usable from
the scratch kernel (`fs` undefined); no authentication was attempted. Historical
checksum-verified backup boundaries remain the recovery source. No new GPU
experiment was launched during this audit.

**Next priority:** obtain the existing raw C2 contingency/readout report, then
specify the content/context identification test. Do not spend another GPU session
on a broader location sweep while these causal alternatives remain conflated.