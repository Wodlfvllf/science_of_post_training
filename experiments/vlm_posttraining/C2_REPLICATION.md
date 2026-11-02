# C2 development replication · fixed before new outcomes, 2 November 2026

## Claim being tested

Model-generated destination identity followed by an independently cued color
lookup improves strict delivered color accuracy over the original direct prompt.
This is an external computation trajectory, **not** the native mechanism, a novel
prompting method or a GSPO-specific improvement. Decomposition/self-ask and visual
programming are established priors. A mechanistic contribution remains to be earned.

## Fixed schedule and statistics

The [machine-readable protocol](../../configs/vlm/c2-development.json) binds the
original plan/data hashes and selects engineering family indices **4–23**, two
query directions on edge +1/palette 0. Indices 0–3 from C2 hypothesis development
are excluded. These twenty families are **C2-new development data**, not globally
untouched: some have earlier decomposition/projector measurements. No pilot-
validation or confirmation outcomes are opened.

Both pinned 8B MPO/GSPO checkpoints use the same new runtime, original common
system/answer prefix, FP32/eager/no cache. Each case retains invalid/truncated
answers and skipped chains in the full 40-case denominator. No cue can use a
gold selector; only an EOS-terminated model-generated bare shape authorizes it.

**Primary:** GSPO chain-minus-direct delivered accuracy, average each family's
two cases, then estimate a paired-family 95% bootstrap interval (10,000 draws,
seed 20261102; constant data use conservative bounded intervals). An exploratory
material lead requires **at least +15 percentage points with interval excluding
zero**. This filter is a resource gate, not a significance/acceptance guarantee.
MPO is a descriptive replication, not a second primary to choose after the fact.

Report paired rescues/regressions, shape/chain contingency and every invalid
reason. No post-hoc semantic extraction substitutes for strict primary grading;
raw semantic issues are reported separately.

## Fixed competing controls

- Original direct generation, maximum 32 tokens.
- Original direct generation, maximum 64 tokens. This is a budget control,
  **not** FLOP matching: early EOS means actual compute differs.
- One-turn composed instruction, maximum 64 tokens, no supplied selector.
- Fresh direct 32-token call after discarding the shape call. This tests state/
  cache contamination; deterministic repetition is not useful extra reasoning.
- Fixed selectorless shape cue from family index/query sign, independent of
  target correctness. It can coincide with target/anchor; count both coincidences
  and whether lookup follows the actual cue's color.
- Original prediction-shifted cue rule and model-generated shape→color chain.

If the chain benefit disappears against the composed prompt or becomes a
formatting artifact, reject an interface-specific explanation. If it persists,
the next discriminating test holds a supplied **model-predicted** identity fixed
while retaining/removing the original anchor/relation wording, and evaluates all
three cue identities against both original-target and cued-node colors. That test
must be specified before its outcomes, not interpreted as an internal circuit.

## Durable evidence

The runner stores the exact protocol text/hash, diagnostic source text/hashes,
plan, runtime, raw generation, actual inputs, all family files and all planned
denominators. It is separate from the locked science package. No failed/partial
report earns a complete result. Checkpoint jobs are sequential and use the native
GPU lock/P0. Verified Drive snapshots capture report directories; imports never
start inference. No full model weight is uploaded to Drive.

## Publication gate

Do not enroll this as a completed mechanism result. Needed after a successful
development replication: selective content/anchor controls, native necessity or
a narrowly stated external-interface claim, frozen held-out replication, transfer
appropriate to scope, novelty review including CARVE's full methods, independent
claim audit, and actual LLM-use disclosure. Nulls remain in the manuscript.