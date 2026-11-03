# Cue content and relation context · fixed developmental identification test

3 November 2026. The [protocol](../../configs/vlm/cue-context-development.json)
and [runner](../../scripts/vlm_cue_context.py) fix the schedule before outcomes.
This is separate from the 20-family C2 replication; it does not change its
primary outcome. No pilot-validation or confirmation family is enrolled.

## Competing explanations

The generated chain may help because (i) shape and color wording invoke different
selection procedures, (ii) an explicit node cue bypasses anchor/relation
interference, or (iii) shape predictions are heuristic while explicitly cued
lookup is reliable. Existing marginal accuracies and failed state swaps do not
identify which explanation is true.

Test **all three cue identities** in a fixed order, not a gold-selected cue.
Keep the image and final instruction fixed across three contexts:

- Explicit chip cue only.
- The original anchor and relation request, then the same chip cue.
- A counterbalanced different anchor (original anchor +1 modulo3), then the
  same chip cue. The different anchor is not chosen from model outcomes.

Grade both the **cued-node color** and the **original relation-target color**.
Changing a cue is an explicit content intervention, not an autonomous inference.
An oracle cue occurs only because all cues are tested, and is labelled afterward.

The primary descriptive contrast is the average family-level change in strict
cue compliance from cue-only to the two relation-context conditions. A negative
contrast is interference in this particular prompt construction; it does not
prove native internal anchor interference. Separate contrasts and cue/anchor
coincidences must also be reported, so averaging cannot hide opposing effects.
No automatic material-effect or paper-readiness gate is applied.

## Relational qualification and counts

Eight previously used engineering families (indices8–15), both edge orientations,
both query directions and both palettes. These are development data, not globally
untouched replication. Each family has:

- 8 variants ×3 cue identities ×3 contexts =72 cue/context stages.
- 8 variants ×2 matched question frames ×shape/color outputs =32 task stages.

Total **104 stages/family, 832/checkpoint, 1,664 across both 8B endpoints**.
Every stage scores all fixed candidates and generates under the unchanged
common answer prefix, FP32/eager, uncached greedy, 32-token cap. Invalid/truncated
delivery remains in the denominator; it is not automatic semantic error.
Shape/color vocabularies still differ (3 versus6); matched syntax alone does not
prove a shared answer or selector representation.

Report shape correctness, joint single-reversal and palette consistency, and
double-reversal cancellation across both wording frames. Isolated correct
guesses are not relational competence. Whole families, not stages or cues, are
uncertainty units. Equal generation caps are not equal compute.

## Interpretation and scientific next gate

Reliable explicit lookup with context sensitivity is a behavioral lead for
interference; reliable shape reversal with unreliable color reversal is a lead
for task-dependent composition. Neither alone is a novel mechanistic result.
If shape selection does not obey reversal/palette predictions, reject the
portable accurate-selector abstraction before transplanting its supposed state.
If an explicit cue suppresses context effects, that can be a dominant instruction
bypassing the original computation, not proof of no native interference.

Only a qualified operation with declared counterfactual/reference-frame
predictions earns further neural intervention, necessity/rescue tests and a
weight-to-operation claim. Shared behavior at two endpoints is not a
GSPO-acquired skill; released endpoints do not isolate optimizer effects.