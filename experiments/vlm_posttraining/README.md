# Reading-or-Routing pilot: executable runbook

Implementation of the [hypotheses](SPRINT.md) in the separate
[vlm_pilot package](../../src/vlm_pilot/). **Offline random-model tests establish
engineering correctness, not H1/H2/H3 or real-model GPU results.**

The original maths/repair Python source and fingerprints are unchanged. This
package has its own plan/source identity and standalone `interp-vlm` /
`python -m vlm_pilot` command. Do not update source beneath an active locked run.

## Implemented scope

| Hypothesis | Executable evidence |
| --- | --- |
| H1 | Eight edge×query-direction×palette cases, cued color and arrow recognition, text-fact controls; separate optional native/common generation audits. |
| H2 | Q-only/K-only flips, joint cancellation, value substitution, same/disjoint-palette and wrong-anchor donors, color-only shams, route/value permutations, uniform routing, capture/self controls. |
| H3 | Selected projection rollback/replay, matched weight-delta sham, identical-input Q/K audit, conditional-route clamp/rescue and wrong/sham clamps, route-only replay, perception/text controls. |

First adapter: **native dense InternVL3.5-2B/8B, Qwen3, Transformers 4.57.6**.
No MiMo/Qwen2.5-VL, HF/Flash/MoE conversions, quantization, dynamic tiling, training,
automatic circuit discovery or long-CoT attribution is claimed. Each prompt has
one 448px tile. Scoped attention hooks edit one pre-answer row of an entire shared
KV group, retaining recipient V, text contribution and total image-attention mass.
All downstream states are recomputed without KV caching.

## Run from marimo notebooks

Open [notebooks/vlm_pilot.py](../../notebooks/vlm_pilot.py) in your remote marimo
editor. This is marimo's Python notebook format, not a Jupyter JSON wrapper. The
existing notebook kernel needs marimo only; it does not import Torch or install
VLM dependencies on opening.

1. Select **setup**, confirm the action, submit settings, then click **Launch once**.
	This creates the isolated VLM environment without changing the notebook kernel.
2. Select **smoke** for the CPU engineering test, or **pilot + prepare** for the
	research data and locked plan. Inspect the scene preview before GPU work.
3. Select **preflight** or **run**, a checkpoint matching the mode, and a free
	logical GPU (`cuda:N`). Check model downloads explicitly when needed. Start
	with the default 12-trial invocation; uncheck the budget only after inspection.
4. Supply discovery sites/weight counterpart for routing or weight runs. Use
	**freeze** before validation/confirmation and pass its protocol path.
5. Select **audit-weights** for actual vision/projector equality, or **analyze**
	with saved trial-run directories (one per line) for the paired CPU report.
6. Refresh status/logs or submit a trial-run path in the independent read-only
	inspection form. In lazy mode, evaluate the cells marimo marks stale after
	submissions/clicks; ordinary recomputation cannot repeat a consumed launch.

The [notebook helper](../../scripts/vlm_notebook.py) launches the same CLI in a
detached subprocess and saves logs plus actual completion/exit-code receipts
under the notebook job directory. Refreshing, reopening the notebook or restarting
the kernel does not launch/retry jobs. Closing a tab does not deliberately stop
workers; **container deletion/platform resets still can**, and receipts from a
dead worker are marked unacknowledged rather than successful. Resume published
trials explicitly. Only one notebook-owned action runs at a time; the separate
GPU occupancy guard still protects other inference workers. No Drive backup is
implied. The notebook and CLI below use the same analysis; inspection cannot execute models.

## 1. Separate remote environment

From the repository root on the remote machine:

```bash
bash scripts/run_vlm_pilot.sh setup
bash scripts/run_vlm_pilot.sh smoke
```

Setup creates **.venv-vlm**, installs `vlm,notebooks,dev` and checks dependencies.
The notebook extra is required because smoke imports the entry notebook using
this isolated interpreter, not the already running marimo kernel. Setup does not
modify the text environment, download models or start GPU workers. Use
`PYTHON_BOOTSTRAP` / `VLM_ENV_DIR` if necessary. Set `TORCH_INDEX_URL` to the
platform-supported public CUDA wheel index if required. Installation alone is not
Blackwell validation; actual Torch/CUDA/precision/hardware are recorded.

Smoke runs offline tests and the actual random-Qwen3 baseline/routing/rollback
workflow. No pretrained weights are downloaded. Its fixture stage labels and
answers have no scientific meaning.

GPU runs require a **free** `cuda:N`. Occupancy is checked using the logical
device's UUID, and a process lock prevents competing pilot workers. Nothing kills
or shares a GPU with the active maths/repair worker. Credentials belong in the
supported secret store/HF authentication, never CLI arguments.

## 2. Generate, audit and lock scenes

```bash
bash scripts/run_vlm_pilot.sh data data/vlm-pilot --contact-sheet runs/vlm-pilot-scenes.png
bash scripts/run_vlm_pilot.sh preview configs/vlm/pilot.yaml
bash scripts/run_vlm_pilot.sh lock configs/vlm/pilot.yaml --output runs/vlm-pilot/plan.json
```

Defaults: **24 engineering + 40 discovery + 60 pilot-validation + 200 confirmation
families**. A family has four images, eight binding questions and four diagnostics.
Saved programs certify gold answers and the arrowhead/color-only pixel edits;
checksums certify actual image bytes. Verify shape/chip/arrow readability yourself
using the contact sheet before P0. Preview/lock import no Torch and fetch no models.

The [native pilot config](../../configs/vlm/pilot.yaml) pins both quartets. Initially
execute only MPO/final at 2B/8B; CPT/SFT provide optional later context. The
[smoke config](../../configs/vlm/smoke.yaml) is explicitly random and CPU-only.
Plans bind images, prompts, precision, full checkpoint/code pins and pilot source.
Outputs are no-clobber. Changed science requires a new plan, never edited metadata.
`--data-root` permits checksummed relocation. Generated data/results are git-ignored.

## 3. P0 and H1 baseline

### Retry after the CUDA UUID compatibility error

If P0 stopped with **“Cannot identify the selected logical GPU UUID”**, it failed
before model loading. PyTorch can expose UUIDs as `_CUuuid` objects rather than
strings. The runtime now normalizes their actual sixteen-byte UUID and still
queries that GPU's compute processes. It never guesses an ordinal or bypasses
the occupied-GPU guard.

Because this runtime fix changes the pilot source hash, the original plan cannot
be used for new model work. **Do not edit its hash, delete it, regenerate images
or reinstall CUDA packages.** After the failed worker is no longer active:

1. Update the remote checkout with `git pull --ff-only`.
2. In marimo select **prepare**, mode **pilot**, set **Plan** to
	`runs/vlm-pilot/plan-gpu-uuid.json`, leave **Output** blank, and leave model
	downloads unchecked. Submit and launch once. Existing data are verified and
	reused; a new source-pinned plan is written without replacing the old plan.
3. Select **preflight**, checkpoint **2b-mpo**, a free **cuda:N**, enable model
	downloads, and **keep the new Plan field unchanged**. Submit and launch once.
4. Wait for `passed: true` and worker exit code `0`. Do not run a long baseline
	while GPU identity or numerical preflight is unverified.

Keep using that new plan for subsequent actions. If that plan path already exists
from an earlier source version, choose another new plan path; never overwrite it.
Archived old fixture results remain readable, but source changes prevent executing
new trials under their old plans. The original maths/repair source is unaffected.

### Retry after the single-file checkpoint asset error

At the pinned revisions, **all four 2B checkpoints contain one safetensors file**;
the four 8B checkpoints use a shard index. The provenance/tensor reader now
supports both, reading a bounded single-file header without loading a second
model. It still requires full revisions, checksums, bounded files/tensors and
safe shard paths. Transport/authentication failures or corrupt indices cannot
silently select another layout or revision.

If the old reader produced `EntryNotFoundError` immediately after model loading:

1. Let the failed worker exit, then update the remote checkout with
	`git pull --ff-only`.
2. Submit **prepare**, mode **pilot**, **Plan** set to
	`runs/vlm-pilot/plan-loader-v2.json`, **Output** blank, model downloads unchecked.
	This verifies/reuses the existing images and creates a new source-pinned plan.
	Keep old plans and cached weights; do not edit their hashes or reinstall setup.
3. Submit **preflight** for **2b-mpo** on a free **cuda:N**, keeping that new Plan
	path, enabling model downloads and leaving Output blank. Cached weights are
	reused when available at the pinned revision. No baseline starts automatically.

The `torch_dtype` deprecation warning is not a numerical failure. Native code may
print **“FlashAttention2 is not installed.”** That is not a requirement to install
it: this pilot explicitly selects **eager attention** with `use_flash_attn=False`.
Require the actual P0 `passed: true` / worker exit code `0` before research trials.
An asset-load failure establishes neither numerical parity nor a hypothesis result.

Start bounded, inspect timing/raw results, then repeat without the trial budget:

```bash
bash scripts/run_vlm_pilot.sh preflight runs/vlm-pilot/plan.json --checkpoint 2b-mpo --device cuda:0 --allow-downloads --output runs/vlm-pilot/p0-2b-mpo.json
bash scripts/run_vlm_pilot.sh run runs/vlm-pilot/plan.json --checkpoint 2b-mpo --phase baseline --split discovery --device cuda:0 --allow-downloads --max-new-trials 12 --output runs/vlm-pilot/2b-mpo-baseline-discovery
```

Every run repeats and saves P0: native vision/embedding equivalence, capture-only,
self-Q/K parity, finite logits, argmax equality, declared tolerances, selected-site
reconstruction and multi-token candidate-prefix checks. Failure blocks trials;
do not silently loosen tolerances. Real GPU peak fit and throughput remain untested.

One-command sequential discovery baselines for both MPO/final pairs:

```bash
DEVICE=cuda:0 bash scripts/run_vlm_pilot.sh baseline
```

This explicit action permits pinned model downloads. It uses the common prefix
and scores complete candidate sequences, not only the first token. Candidates
that rewrite conditioning tokens are rejected; ties count as misses. No
correctness-based source filtering or implicit sample pooling is used.

Answer-policy audits use separate directories/settings, for example:

```bash
bash scripts/run_vlm_pilot.sh run runs/vlm-pilot/plan.json --checkpoint 8b-mpo --phase baseline --split discovery --generate --prompt-profile native --device cuda:0 --allow-downloads --output runs/vlm-pilot/8b-mpo-native-audit
```

This is unrestricted **greedy, uncached**, bounded generation, not a sufficient
native long-thinking budget or pass@k experiment. Raw tokens/text, EOS/truncation,
exact parsing and invalid answers persist. Reports separate native/common and
generation/non-generation jobs. Candidate ranking is not free-generation accuracy.

## 4. H2 routing and H3 weight discovery

Select zero-based **layer:KV-group** sites explicitly. The following sites are
engineering examples, not discovered scientific locations:

```bash
bash scripts/run_vlm_pilot.sh run runs/vlm-pilot/plan.json --checkpoint 8b-mpo --phase routing --split discovery --sites 8:0,16:1 --device cuda:0 --allow-downloads --output runs/vlm-pilot/8b-mpo-routing-discovery
bash scripts/run_vlm_pilot.sh run runs/vlm-pilot/plan.json --checkpoint 8b-gspo --phase routing --split discovery --sites 8:0,16:1 --device cuda:0 --allow-downloads --output runs/vlm-pilot/8b-gspo-routing-discovery
```

Each site has **17 routing conditions per family/checkpoint**. Donor query lengths,
image columns, masks and actual rotary coordinates must align. Image-only donors
must preserve prompt tokens exactly. A mismatch fails closed and is saved; do not
pad, drop or silently realign confirmation items. `--save-captures` optionally saves
bounded CPU pre-answer Q/K/V, hidden rows and rotary coordinates as safetensors;
no gold-answer-bearing donor states or full attention dumps are captured.

Actual saved-tensor equality audit and selected-group rollback:

```bash
bash scripts/run_vlm_pilot.sh audit-weights runs/vlm-pilot/plan.json --before 8b-mpo --after 8b-gspo --allow-downloads --output runs/vlm-pilot/components-8b.json
bash scripts/run_vlm_pilot.sh run runs/vlm-pilot/plan.json --checkpoint 8b-gspo --phase weights --split discovery --sites 8:0 --source-checkpoint 8b-mpo --projections q_proj,k_proj --device cuda:0 --allow-downloads --output runs/vlm-pilot/8b-rollback-discovery
```

Replay reverses checkpoint/source. Sources load selected tensors on CPU, not a
second GPU model. The **17-condition H3 schedule** includes task controls and
native/swap/sham/clamp/replay comparisons. Selected slices restore bit-exactly on
normal exit or failure. Shams permute the delta with the same seed for the
sham-plus-clamp comparison. Conditional clamps preserve recipient mass/V and do
not isolate every possible weight-induced upstream change.

Use `--projections v_proj,o_proj` or explicit real-control sites in **new discovery
outputs**. `--include-norms` changes layer-wide shared Q/K norms and is labelled as
such; it is not group-specific. Identical-input audit reconstruction error is
reported separately from the weight effect. A selective operation is a hypothesis
to validate, not an implication of a successful hook or merge.

## 5. Freeze before validation

```bash
bash scripts/run_vlm_pilot.sh freeze runs/vlm-pilot/plan.json --sites-2b 8:0 --sites-8b 16:1 --note "Replace example sites with findings selected on discovery; record rationale." --output runs/vlm-pilot/protocol.json
bash scripts/run_vlm_pilot.sh run runs/vlm-pilot/plan.json --checkpoint 8b-gspo --phase weights --split pilot_validation --sites 16:1 --source-checkpoint 8b-mpo --protocol runs/vlm-pilot/protocol.json --device cuda:0 --allow-downloads --output runs/vlm-pilot/8b-rollback-validation
```

The immutable protocol freezes at most two sites per checkpoint, projections/norm
scope, source, primary endpoint, material-effect screens, preservation margin and
primary-site multiplicity. Optional screen/margin flags are set **before** freeze.
All confirmation and intervention pilot-validation jobs require matching protocol.
The code cannot prove that a human has never inspected test data; maintain split
discipline and record changes. Pilot data are development; final confirmation is
the **200 fresh families**, not reused pilot validation.

## 6. Inspect, analyze, resume

These names match the baseline launcher's outputs:

```bash
bash scripts/run_vlm_pilot.sh inspect runs/vlm-pilot/2b-mpo-baseline-discovery
bash scripts/run_vlm_pilot.sh analyze runs/vlm-pilot/2b-mpo-baseline-discovery runs/vlm-pilot/2b-gspo-baseline-discovery --output runs/vlm-pilot/baseline-report
```

Reports include JSON/CSV/Markdown, explicit completeness, family-paired stage and
clamp/surgery contrasts, H1/H3 binding-minus-control selectivity, the primary
routing stage interaction, joint flip/cancellation/non-copy signatures and
discovery-based power planning. Primary-site intervals use the frozen multiplicity
rule; other estimates are descriptive. Non-significant controls are not
equivalence. Saturated data cannot justify zero-variance power or zero-width CIs.

Incomplete jobs/comparisons require `--allow-incomplete`. Duplicate snapshots,
different operations, prompt profiles and conflicting frozen protocols cannot be
pooled. No automatic “hypothesis passed” or publication decision is emitted.

Repeat the identical run command to resume. Completed resume or `--max-new-trials 0`
inspection is model-free. Cooperative `--max-seconds` cannot interrupt a synchronous
download/forward. Whole-family `--num-shards` / `--shard-index` jobs use distinct
outputs and free GPUs. Only sealed checksummed **manifest-referenced successes**
advance completion; failed/cancelled attempts remain inspectable.

**Remote backup is not automatic.** Preserve plan, exact images/metadata, protocol,
run metadata/manifest/referenced records and referenced captures externally. The
existing text-study Drive backup does not certify these VLM artifacts. Local
completion is not verified remote durability. Keep submission artifacts anonymous.

## Local verification versus remaining research

Verified on macOS arm64, Python 3.11.14, Torch 2.10.0/CPU and Transformers 4.57.6:
**610 full-suite tests passed**, zero failures/errors/skips; Ruff and shell syntax
passed; the wheel includes all 14 pilot modules and the standalone CLI. Actual CLI
scene generation, locking, P0, durable trial inspection and protocol freezing were
exercised without model downloads. These are engineering checks, not research results.

The offline tests exercise scene certification, native tiny-Qwen3 normalization,
RoPE/GQA/mass preservation, weight slicing/sham/restoration, aligned teacher
forcing, frozen schedules, corruption/failure recovery and model-free resume.
Real InternVL dependency/code loading, Blackwell precision, actual memory/runtime,
native donor alignment, visual readability, effect estimation, photographic/MiMo
adapter generalization, novelty and remote backup still require validation.