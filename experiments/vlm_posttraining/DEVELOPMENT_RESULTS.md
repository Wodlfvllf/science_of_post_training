# Controlled development results · 3 November 2026

**Execution and raw-result audits completed. These are development findings, not
held-out confirmation or a demonstrated post-training circuit.** Protocols were
fixed before these outcomes; the earlier failed/null tests remain in the record.

## Execution boundary

The detached seven-step queue completed with exit code 0 at 07:20:27 UTC.
Both endpoints passed native numerical P0 and used the same FP32/eager,
uncached, original common answer-prefix profile: Python 3.13.11, Torch 2.15.0,
CUDA 13.2, Transformers 4.57.6, RTX PRO 6000 Blackwell Server Edition.

Historical plan/source/data pins were preserved. All 5,128 prior published
trials and 1,296 images were restored and verified, not rerun. The new work is
**640 planned C2 stage slots plus 1,664 cue/context stages**; skipped stages after
invalid selectors retain denominators. These are not 2,304 independent sources
or additional manifest-published baseline trials.

The original notebook connection displayed a stale blocking waiter after the
queue finished. Terminal status receipt confirmed exit 0 and no GPU processes;
reconnecting restored accurate read-only notebook status. The experiments were
not restarted to repair the display.

## 1. Generated-selection replication with competing controls

[Frozen C2 protocol](C2_REPLICATION.md): 20 engineering families, indices4–23,
40 cases/checkpoint at fixed edge +1/palette0 with both query directions.
Families exclude the first C2 screen but are not globally untouched; some have
other prior engineering measurements. Neither validation nor confirmation opened.

| Delivered-answer condition | MPO exact /40 | GSPO exact /40 |
| --- | ---: | ---: |
| Original direct32 | 7 | 7 |
| Original direct64 | 7 | 7 |
| One-turn decomposition instruction64 | 8 | 8 |
| Own generated shape → cued color32 | 17 | 19 |
| Shifted generated-shape cue32 | 12 | 12 |
| Outcome-blind fixed-node cue32 | 11 | 11 |
| Fresh direct32 after discarding shape output | 7 | 7 |
| Correct delivered destination shape32 | 17 | 19 |

**Frozen GSPO primary:** chain − direct = **+30 percentage points**, paired-family
95% interval **[+17.5, +42.5]**. It passes the documented +15-point/positive-
interval resource screen. This does not prove a true effect greater than15
points, a native mechanism, an optimizer effect, or publication readiness.
MPO descriptive gain is +25 points. Five GSPO exploratory secondary comparisons
use 99% family intervals:

- Chain − direct64: +30 points, [+12.5, +45].
- Chain − one-turn instruction: +27.5 points, [+12.5, +42.5].
- Chain − fixed-node cue: +20 points, [−2.5, +42.5].
- Chain − shifted cue: +17.5 points, [−2.5, +37.5].
- Chain − wasted-call repeat: +30 points, [+12.5, +45].

The latter cue comparisons remain imprecise. The repeat is a deterministic
contamination check, not useful extra reasoning. Direct64 controls maximum
generation budget, not FLOPs; vision features are reused while repeated decoder
prefixes differ. No full compute-equivalence claim is made.

**GSPO − MPO interaction in the chain/direct effect:** +5 points, 95% interval
[−5, +17.5]. No supported stage-specific acquisition claim follows.

### Error decomposition, not equal marginals

Raw records were reconstructed, and each cue checked against its own generated
selector. Among all40 scheduled cases:

| Selector/lookup class | MPO | GSPO |
| --- | ---: | ---: |
| Invalid selector; chain skipped | 1 | 1 |
| Correct selector, faithful cued lookup | 17 | 19 |
| Wrong selector, faithful cued lookup | 22 | 20 |
| Valid selector, unfaithful cued lookup | 0 | 0 |

Thus the chain faithfully reads **every valid selected node** (39/40 overall),
including the wrongly selected ones. This exposes selection as the error
bottleneck of this **external pipeline**. It does not measure the selector in
the original direct-color forward. Shifted cues sometimes accidentally indicate
the original destination; no control is called uniformly wrong. GSPO shifted-cue
delivery has two invalid answers; all denominators remain40.

## 2. Exhaustive cue/content and relation-context identification

[Fixed cue/context protocol](CUE_CONTEXT.md): eight previously used engineering
families, indices8–15; all edge/query/palette variants. All three shape cues are
tested under cue-only, original relation context, and a competing anchor context.
The final explicit cue instruction is identical. Two shape/color wording frames
are also tested. Each endpoint has832 stages, not832 independent observations.

| Context | MPO follows cue /192 | GSPO follows cue /192 |
| --- | ---: | ---: |
| Cue only | 192 | 192 |
| Original anchor/relation, then cue | 192 | 192 |
| Competing anchor/relation, then cue | 191 | 192 |

Original relation-target accuracy is64/192 in every context at both endpoints:
the exhaustive cue schedule includes that target one-third of the time. This
is explicit instruction following, **not model-inferred destination rescue**.

The frozen average context-minus-cue-only compliance estimate is −0.2604 points
for MPO and zero for GSPO. Eight-family all-zero effects use broad conservative
intervals; **do not claim equivalence or universal absence of interference**.
A dominant explicit cue could bypass interference in the original question.
The sole MPO mismatch was a non-target cue in a competing-anchor condition,
returning red rather than the cued square's purple chip. It was retained.

### Relation qualification rejects the reliable-selector assumption

| Task/frame | MPO exact /64 | GSPO exact /64 | MPO/GSPO invalid delivery |
| --- | ---: | ---: | ---: |
| Destination shape, frame1 | 29 | 27 | 4 /11 |
| Destination shape, frame2 | 28 | 24 | 8 /19 |
| Destination color, frame1 | 16 | 12 | 0 /5 |
| Destination color, frame2 | 10 | 12 | 0 /0 |

No family has all eight relational shape outcomes correct in either frame at
either endpoint. In frame1, **0/32 edge-reversal pairs** have both answers correct
at both endpoints; frame2 has1/32. Only2/32 or3/32 direction-reversal pairs have
both correct answers. Paired palette consistency is higher, but does not repair
edge/query failures. Shape/color answer vocabularies differ, so the gap alone
cannot isolate an internal handoff failure.

The consistent inference is narrower: explicit node lookup is highly reliable,
while generated relational selection is unreliable and does not robustly satisfy
the reversible task algebra. That favors investigating selection heuristics or
task-dependent procedures over assuming an accurate shared native selector.
It does not uniquely choose between those alternatives without further tests.

## Integrity, uncertainty and paper gate

Both pairs were analyzed from raw family artifacts, not summary flags alone.
The C2 audit reconstructs cases, strict raw grading, protocol/source/runtime
matching, repeat traces, full denominators and cue faithfulness. The cue audit
also verifies every original image checksum and family artifact hash, reconstructs
all1,664 rows, and publishes an integrity receipt last. Statistical units are
paired source families. Repeated engineering use and campaign exploration limit
claims; selective subsets are not primary evidence.

These experiments support an externally structured selection/lookup benefit
beyond the tested longer direct budget and one-turn instruction. **They do not
establish a novel method, native selector mediator, GSPO-specific operation,
weight-to-operation causal link, or held-out/general-domain confirmation.**
The 60 pilot-validation and200 confirmation families remain untouched. The next
scientific step must qualify relation/reference-frame competence and identify a
distinctive causal operation, not open confirmation merely to certify a prompting
gain or present a sensitive-state patch as a circuit.

Full local regression receipt for the implemented tools: **1,418 tests passed**,
zero failures/errors/skips. Tests establish mechanics, not experimental findings.
Periodic Google Drive snapshots ran during inference; the final published
boundary must include both raw pairs and both audit integrity receipts and be
checked by readback before container deletion.