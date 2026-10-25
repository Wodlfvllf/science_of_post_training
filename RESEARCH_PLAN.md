# Personal interpretability research: repository architecture

Prepared 2026-10-02. This is an implementation blueprint, not an implemented or GPU-tested toolkit. No research project is selected yet.

## 1. Objective and constraints

Build a personal research workbench that makes the loop **observe → form competing hypotheses → test → inspect failures → replicate** fast and trustworthy. It is not a public framework or an attempt to implement every interpretability method.

- Develop and analyze on the Mac; execute larger runs in molab or on an available office GPU.
- The user reports a 12-hour molab allowance that renews daily. Filesystem persistence and actual GPU allocation remain unverified.
- Hugging Face remote storage is connected in molab. Python authentication and upload permissions must still be checked independently.
- Keep notebooks thin, experiment logic importable, model execution explicit, and results recoverable without the original machine.
- Prioritize behavioral baselines, selective activations, linear probes, and controlled interventions. Add advanced methods when a question needs them.
- Success means a new hypothesis needs a small project recipe, not changes throughout the toolkit.

## 2. Research workflows and reusable tools

| Direction in the research brief | Experiments to enable | Reusable tools | Main validity risk |
| --- | --- | --- | --- |
| Model forensics and unusual behavior | Reproduce a transcript; change one hypothesized cause; resample continuations | Transcript browser, paired prompts, branching records, outcome scoring | Inferring motives from one output; failing to reproduce on the original model |
| CoT faithfulness and thought anchors | Hint/no-hint comparisons; prefix continuation; sentence resampling and controlled edits | Token/span mapping, continuation runner, matched edit controls | Off-distribution edits; treating readable reasoning as a complete account of computation |
| Truth, uncertainty, evaluation awareness, user models | Layer/position probe sweeps; held-out contexts; rare-event monitoring | Selective capture, grouped splits, linear probes, lexical baselines, calibration | Surface cues, answer leakage, excessive false positives |
| Steering and causal investigation | Direction addition/removal; clean-to-corrupt patching; strength/site sweeps | Scoped interventions, donor/recipient alignment, no-op and random-direction controls | Changed behavior does not establish the full proposed mechanism |
| Latent knowledge and objective interpretability tasks | Compare prompting, readouts, and probes on tasks with known answers | Candidate scoring, retrieval metrics, budget accounting | Weak black-box baselines; testing on the words used to develop the method |
| Model diffing and character | Compare base/fine-tuned checkpoints on identical inputs; test behavioral differences | Paired model runs, disagreement tables, compatible activation comparisons | Tokenizer/precision differences; assuming coordinates align across unrelated models |
| Prompt-injection robustness and monitoring | Controlled role/context changes; conditional steering; benign capability checks | Role-aware spans, detection metrics, causal gating, side-effect evaluations | Confusing suspicious-looking text with successful instruction override |
| SAE, J-Lens, NLA, attribution methods | Inspect matched published artifacts; test reconstruction/readout predictions by intervention | Optional library integration, artifact validation, baseline comparisons | Attractive descriptions without causal or downstream validation |
| Post-training and generalization | Later: small adapter experiments, checkpoint comparisons, multiple OOD evaluations | Training-library integration, dataset lineage, checkpoint metadata | Training-set contamination; one benchmark standing in for alignment |

These are capabilities to support, not projects to pursue simultaneously. Automated auditing, filler-token investigations, and training-time concept ablation should compose these primitives rather than introduce separate infrastructure.

## 3. Proposed repository structure

The following is a future layout. Create modules only when their implementation phase starts; do not scaffold empty projects or speculative adapters.

```text
Interpretability_research/
├── pyproject.toml
├── uv.lock
├── .gitignore
├── src/
│   ├── interp/
│   │   ├── __init__.py
│   │   ├── schemas.py       # Experiment inputs, configuration, and result records
│   │   ├── runtime.py       # Environment checks, provenance, and RNG policy
│   │   ├── data.py          # Examples, prompt rendering, token alignment, and splits
│   │   ├── models.py        # Model lifecycle, generation, and continuation scoring
│   │   ├── runner.py        # Trial scheduling, budgets, cancellation, and resume
│   │   ├── artifacts.py     # Local shards, manifests, and explicit Hub persistence
│   │   ├── metrics.py       # Paired effects, monitoring metrics, and uncertainty
│   │   ├── capture.py       # Selected activation sites and retained token positions
│   │   ├── probes.py        # Linear baselines, validation, and probe inference
│   │   ├── interventions.py # Scoped patching, direction edits, and controls
│   │   ├── plots.py         # Token views, heatmaps, curves, and failure tables
│   │   └── cli.py           # Thin command-line entry point
│   └── research_projects/
│       └── <project_slug>/  # Add only after choosing the research question
│           ├── __init__.py
│           ├── dataset.py
│           ├── recipe.py
│           └── evaluation.py
├── configs/
│   ├── smoke.yaml
│   └── <project_slug>/
├── notebooks/               # Thin marimo Python notebooks
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/            # Tiny offline models/tokenizers and controlled data
├── data/                    # Generated/downloaded data; ignored by git
└── runs/                    # Local working artifacts; ignored by git
```

### Dependency boundaries

- Project recipes import `interp`; the shared package does not import project-specific logic.
- The notebook and CLI invoke the same recipe and runner. No separate notebook-only experiment implementation.
- The runner receives an ordinary trial function; no plugin registry, workflow engine, or large inheritance hierarchy.
- Core modules never depend on marimo. Plot functions return figures/tables rather than controlling notebook execution.
- Package installation makes both packages importable; no permanent `sys.path` modifications or copied cells.
- Code, configs, and small fixtures go in git. Model weights and research outputs do not.

## 4. Small contracts, explicit scientific meaning

Use plain functions and a few validated records. Avoid a universal abstraction that hides important model differences.

| Contract | Required information |
| --- | --- |
| `Example` | Stable ID; source/group/pair IDs; messages or text; condition; label and label provenance |
| `RunSpec` | Hypothesis and alternative; dataset identity; model/tokenizer revisions; rendering and decoding settings; primary metric; splits; controls; budgets |
| `SiteSpec` | Library hook name or exact module path; input/output boundary; layer; tensor axes; normalization convention; selected positions; execution phase |
| `TrialRecord` | Trial/attempt IDs; condition and replicate; exact input/output tokens and text; score; termination status; artifact references; errors |
| `CheckpointManifest` | Schema version; specification fingerprint; completed trial IDs; artifact checksums; remote revision; execution provenance |

Suggested functions are `load_model()`, `generate()`, `score_continuations()`, `capture()`, `fit_probe()`, `evaluate_probe()`, `run_intervention()`, `run_trials()`, `sync()`, and `restore()`. Retain access to the underlying model and tokenizer for novel experiments.

Separate experimental identity from execution attempts. Changing a dataset, model revision, prompt, method, or scientific setting creates a new identity. Changing an output directory or upload interval does not. Record hardware/backend details even when they are not part of the scientific specification; do not silently mix incompatible numerical execution profiles during resume.

## 5. Reuse libraries rather than rebuild them

| Layer | Initial choice | Extension policy |
| --- | --- | --- |
| Reference model execution | PyTorch + Hugging Face Transformers | One verified dense decoder model family first; explicit capability errors elsewhere |
| Instrumentation | Local HF-backed TransformerLens `TransformerBridge` | Introduce in the activation phase after parity tests; reuse its hooks/adapters, not a custom tracing engine |
| Alternate tracing | NNsight | Only for a concrete unsupported operation or remote-execution requirement; not a second mandatory backend |
| Probes and baselines | scikit-learn pipelines + NumPy | Start with logistic regression and mean-difference directions; add more complex probes only against these baselines |
| Configuration | Pydantic + safely loaded YAML | A single resolved config per run; no Hydra-style configuration hierarchy initially |
| Artifacts | JSONL records, JSON manifests, safetensors arrays, `huggingface_hub` | Add Parquet for large analysis tables if needed; avoid custom tensor formats and untrusted pickle |
| Interactive analysis | marimo + Plotly | No bespoke web dashboard; CPU analysis should work from saved results |
| Quality checks | pytest + Ruff | Offline CPU CI first; GPU and real-Hub integration tests explicitly opt in |
| Advanced methods/training | Published artifacts, SAELens, PEFT/TRL or suitable existing libraries | Project-specific optional dependencies; do not build an SAE or distributed trainer |

Keep installation groups separate: base execution/storage, instrumentation/probes, notebooks, development, and later training. Resolve and lock tested versions, not guessed pins. GPU-specific Torch wheels must match the actual driver/hardware; a lock alone does not prove a Blackwell environment works.

Current TransformerLens documentation describes 4.0's HF-backed bridge and removal of legacy `HookedTransformer` classes. Older tutorial code needs migration or its own pinned environment. Raw HF numerical conventions are the default: compatibility modes, weight processing, quantization, and alternate drivers are explicit settings. Never reuse an activation cache after changing these settings. Library model support is not proof that every analysis supports every hook or architecture.

## 6. Scientific correctness safeguards

### Prompts, tokens, and scores

- Save exact rendered prompts, token IDs, attention masks, chat-template identity, special-token choices, role spans, and any thinking-mode setting.
- Distinguish unpadded sequence positions from padded batch indices. Define the last non-padding position rather than assuming the last tensor column is valid.
- Continuation scoring uses the correct next-token shift and scores only the requested continuation. Handle multi-token answers explicitly; do not treat a word as one token without checking.
- Preserve raw outputs and parser versions. Record truncated generations, invalid formats, and failed trials instead of silently excluding them.
- Re-tokenize after text edits. Patching requires explicit donor/recipient alignment; matching tensor shapes alone is insufficient.

### Activations and interventions

- Start with selected residual/block boundaries and token positions. Name embedding and final-normalization sites separately from zero-based transformer layers.
- Specify whether a captured tensor is before or after normalization. Returned hidden-state tuples are not a universal residual-site convention.
- Capture only requested slices, detach retained observations, and move them off GPU. Preserve capture dtype by default; any conversion is recorded and checked for finite values.
- Hooks exist only inside a context that cleans up on success, exception, and notebook interruption. Observation-only hooks must not alter outputs.
- Do not permanently edit weights or mutate the shared model for routine experiments. Prevent simultaneous runs on the same instrumented model.
- Initially support teacher-forced, full-sequence patching with KV caching disabled. Add generation-time edits separately, distinguishing prefill from decode and absolute positions from decode steps.
- Editing an earlier activation requires recomputing affected downstream/KV states. Do not reuse baseline KV caches after an intervention.
- Direction methods declare normalization and scale. Probe coefficients from standardized features must be converted to raw activation coordinates before use as intervention directions; causal meaning still requires testing.
- Compare unmodified, no-op/self-patched, targeted, and suitable random/norm-matched controls. Report benign-task side effects as well as the target behavior.
- Raw effect sizes are primary. A normalized patch-recovery score is undefined or unstable when the clean/corrupt denominator is near zero; mark it accordingly.
- Evaluation mode and autograd mode are separate. Inference defaults must not disable gradients in later attribution or training recipes.

### Probes and evaluation

- Split by source example, conversation, entity, or prompt family before making paraphrases/resamples. Paired variants and derived samples stay in the same split.
- Fit scalers, directions, dimensionality reduction, and other learned preprocessing on training data only. Choose layer, position, regularization, and threshold on validation data; reserve test data for final evaluation.
- Define when features become available. A monitor that claims to predict behavior must not consume the answer or future tokens.
- Include majority, shuffled-label, lexical/nuisance-feature, and practical prompting/monitor baselines where applicable. A probe decoding a concept is not evidence that the model uses it.
- Track AUROC and precision-recall metrics, but also false-positive rates, recall at a declared threshold, and precision under relevant prevalence. For long-context monitoring, evaluate sequence-level false alarms rather than treating token decisions as independent.
- Aggregate paired comparisons and uncertainty at the independent source-example/group level, not across every token or correlated resample.
- Keep exploration and confirmatory evaluation distinguishable. Record the sweep space and hold out new prompts before making general claims.
- Treat CoT text and automated judges as evidence sources, not ground truth. Log judge prompts/model revisions and manually audit a sample of labels.

## 7. Durable runs for short-lived sessions

A successful local write is not a durable checkpoint. Connected Hub storage is not assumed to sync local artifacts automatically.

1. Freeze the resolved specification and source fingerprint. Record the git commit and dirty-source fingerprint; require clean committed code for confirmatory runs, or explicitly archive relevant source changes without secrets.
2. Assign stable trial IDs from example, condition, intervention, and replicate. Use unique execution-attempt IDs and only one writer per run in the initial implementation.
3. Finalize immutable result/activation shards through temporary files, atomic local rename, and checksums. Never count a partial write as a completed trial.
4. Upload sealed artifacts explicitly, then publish a checkpoint manifest referencing acknowledged artifacts. A small bounded `create_commit()` can publish files and a manifest together; do not assume a large `upload_folder()` is one atomic commit.
5. Advance the durable-completion boundary only after upload/manifest acknowledgement. Keep unsynced local shards, expose upload errors, and await any background uploads before controlled shutdown.
6. At restart, resolve a remote revision, restore the latest valid manifest and required artifacts at that revision, verify checksums/specification, and schedule only missing trials. Unpublished/orphan uploads do not automatically count as completed work.

Resume at completed trial boundaries, not halfway through a generation. Rerun an interrupted trial. For stochastic replay, use isolated trial seeds or checkpoint the exact batch membership and RNG state; one global seed is insufficient after reordering or OOM rebatching. Deterministic reference tests have declared numerical tolerances; cross-device bitwise equality is not promised.

Configure checkpoint frequency by elapsed time and shard size, not one Hub commit per example. A reset can lose work since the last acknowledged checkpoint; do not promise zero loss. Stop scheduling new work before the session deadline and leave a configurable upload margin. Authentication uses provider-supported secrets or existing token configuration, never tokens in cells, configs, logs, or manifests. Verify a small private-repo upload/download round trip before a large run.

## 8. Fast execution and the marimo workflow

- A notebook contains controls, one explicit model-loading action, a frozen experiment submission, progress/results views, and plots. The long-running function lives in the package.
- Prefer lazy execution for expensive research notebooks. Gate work explicitly with submitted settings/run actions; test behavior in the selected runtime mode rather than assuming a run button always resets identically.
- Imports, notebook startup, and plot-cell reruns must not launch experiments, download models, register persistent hooks, or upload data.
- marimo does not track mutations to model objects. Scoping hooks and interventions in library calls is essential; rerunning an import is not model-state reset.
- After pulling changed code, restart or use verified module reload behavior. Do not pull new code in the middle of a run and label it with the old revision.
- Offer dry-run and smoke-run modes that validate data, tokens, sites, expected artifact size, and selected runtime before a full sweep.
- Estimate retained activation bytes from examples × retained positions × hidden size × sites × dtype bytes. Budget KV caches/logits separately; fitting weights in VRAM does not imply the experiment fits.
- Reuse immutable captures across compatible probe fits; invalidate on model, prompt, tokenizer, site, numerical profile, or method changes.
- Batch conservatively by token length, stream outputs, and avoid all-layer/all-token caches by default. Log peak memory, tokens processed, wall time, and errors.
- CPU fixtures support Mac development. MPS and CUDA are opt-in verified profiles, not assumed equivalent. Inspect the actual GPU and run a numerical smoke test before choosing precision or optimized kernels.
- Add two-GPU support only when one tested GPU is insufficient. Compilation, quantization, FlashAttention, and vLLM are performance projects with equivalence checks, not initial defaults.

## 9. Implementation order and release gates

No time estimates: progress is defined by working end-to-end slices.

| Phase | Concrete deliverable | Gate before continuing |
| --- | --- | --- |
| A1: usable local loop | Package/configuration; offline fixture; one HF model path; paired prompts; continuation scoring; trial records and CPU plots | Token/scoring and deterministic replay tests pass; CLI and notebook call the same runner |
| A2: survive molab resets | Immutable shards; specification/provenance; explicit Hub sync/restore; budgets/cancellation; smoke notebook | Interrupted-run recovery passes offline; small real Hub round trip and actual GPU smoke pass before large remote runs |
| B1: inspect and probe | Verified Bridge sites; selective capture; grouped splits; logistic-regression/mean-difference baselines; layer/position plots | Capture-only parity, padding/position, train-only preprocessing, and leakage checks pass |
| B2: test causes | Full-sequence teacher-forced patching; direction edits; self-patch/random controls; side-effect comparison | Hooks clean up on errors; self-patch/no-op match baseline; actual target slices are verified |
| C: first chosen question | One project package with data, recipe, evaluation, controls, and held-out inputs | Reproduction and simplest useful baseline work before advanced methods are added |
| D: earned extensions | Only the needed CoT branching, conditional monitoring, model diffing, published lenses/SAEs, or training integration | Adds measurable insight or usefulness beyond the established baseline, without breaking earlier gates |

The initial smoke recipe is an engineering fixture, not a commitment to a toy-model research direction. Start real research after A2; add B capabilities alongside the question rather than finishing every tool first.

## 10. Acceptance tests that protect research results

1. **Replay and notebook safety:** an offline tiny decoder/tokenizer runs paired trials through the shared entry points. Scores/tokens reproduce within tolerance. Importing code or recomputing displays does not load models, run new trials, write artifacts, or upload anything.
2. **Recovery from a clean machine:** inject failures during a shard write, after artifact upload but before manifest publication, and after publication before local acknowledgement. With an empty local directory, restore only verified completed work, reject corrupt/incompatible artifacts, and finish missing trials without duplicate logical results. Use a fake remote store offline, then separately validate real Hub authentication.
3. **Capture and causal controls:** compare baseline, capture-only, self-patch, and cross-patch runs. The first three agree; cross-patching replaces exactly the declared slices. A forced exception/interruption leaves no hooks or modified weights and a subsequent baseline still agrees.
4. **Scoring alignment:** short, multi-token, padded, and unequal-length continuations agree with an explicit manual next-token calculation. Edited text has fresh token/span alignment.
5. **Leakage resistance:** derived prompt groups never cross splits; changing test features does not change training preprocessing; shuffled-label performance is evaluated across repeated shuffles and sample-aware uncertainty, not required to equal chance on a single tiny dataset.
6. **Capability and artifact compatibility:** wrong site names, tensor shapes, revisions, tokenizer conventions, and numerical profiles fail clearly instead of silently producing plots.

All are proposed tests. None has been executed by preparing this document.

## 11. Keep the build lean

Do not initially build a plugin system, agent orchestrator, custom SAE trainer, distributed scheduler, exhaustive model registry, website, or automatic paper-writing system. Do not create all potential research projects in advance.

Promote project-local code into the shared package after a second genuine use demonstrates the common interface. Keep project-specific labels, prompt heuristics, judges, and success criteria local. Prefer an understandable function with explicit inputs over a universal experiment class.

Before implementation, the only unresolved operational choices are the first research question/checkpoint, actual GPU/runtime profile, and authenticated results-repository destination. None prevents planning or offline tests; confirm them when the relevant phase begins.

## 12. Documentation consulted

- [marimo reactivity and side effects](https://docs.marimo.io/guides/reactivity/)
- [marimo run-button semantics](https://docs.marimo.io/api/inputs/run_button/)
- [Hub uploads, commit behavior, and acknowledgement](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Revision-pinned Hub downloads](https://huggingface.co/docs/huggingface_hub/guides/download)
- [TransformerLens 4.0 architecture and migration](https://transformerlensorg.github.io/TransformerLens/content/news/release-4.0.html)
- [Selecting analysis tools and checking scientific interpretation](https://transformerlensorg.github.io/TransformerLens/content/analysis_tools.html)
- [NNsight tracing/instrumentation reference](https://nnsight.net/documentation/)
- [scikit-learn leakage and preprocessing pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [PyTorch module hook lifecycle](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)