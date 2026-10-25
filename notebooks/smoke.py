# This is a thin marimo UI for the packaged workflow, not a second experiment implementation.
# Freeze a local configuration, click Run deliberately, then inspect saved results separately.
# Importing/opening this notebook does not load weights, run trials, or upload results.
"""Explicit offline smoke execution and read-only paired-result inspection.

Install the package's model/notebook extras before opening this file. The default
configuration is an engineering fixture. Real-model downloads require both a
configured pinned HF checkpoint and the separately submitted opt-in checkbox.
No upload control is present; sync is an independent explicit CLI/library action.
"""

import marimo

# Record the marimo notebook format validated for this checked-in Python app.
__generated_with = "0.24.2"

# Define notebook presentation without executing any model-related cell.
app = marimo.App(width="medium")


@app.cell
def _():
    """Import lightweight entry points without creating a separately resettable click guard."""
    # Locate configuration defaults without relying on permanent sys.path modifications.
    from pathlib import Path

    # Assign new immutable submission tickets only when settings are frozen explicitly.
    from uuid import uuid4

    # Import marimo UI primitives independently of model/remote libraries.
    import marimo as mo

    # Import only read-only checkpoint and descriptive-table functions for analysis.
    from interp.artifacts import read_run

    # Share exactly the same model-free config preparation used by the CLI.
    from interp.config import prepare_run, with_execution

    # Keep incomplete comparisons and failed attempts visible.
    from interp.metrics import attempt_rows, paired_effects

    # Import a lazy plot wrapper, not model computation.
    from interp.plots import paired_effect_figure

    # Share the ordinary runner and explicit click guard rather than duplicating experiments.
    from interp.runner import ExplicitRunGate, run_experiment, summarize_run

    # Define a stable repository anchor for this checked-in notebook's defaults.
    repo_root = Path(__file__).resolve().parent.parent
    # Export unique names once to marimo's reactive dependency graph.
    return (ExplicitRunGate, Path, attempt_rows, mo, paired_effect_figure, paired_effects, prepare_run, read_run, repo_root, run_experiment, summarize_run, uuid4, with_execution)


@app.cell
def _(mo):
    """Explain explicit execution and the limits of the engineering model."""
    # Display research safeguards without running or writing an experiment.
    mo.md("""
    # Research workflow · first slice

    **Default: tiny random GPT-2, offline CPU.** Its scores are an engineering check,
    not evidence about a trained model or a causal mechanism.

    1. Freeze settings below (validation only).
    2. Click **Run / resume** once for intentional model work.
    3. Submit the saved-results path in the separate read-only inspection form.

    Model downloads are opt-in. Local writes are **not** remote checkpoints.
    Lazy execution is recommended; click tickets prevent accidental repeat work.
    """)
    # This display cell defines no experiment state.
    return


@app.cell
def _(mo, repo_root):
    """Collect settings without applying them until the form is submitted."""
    # Gate paths/download policy behind one immutable settings submission.
    settings_form = mo.ui.dictionary({"config": mo.ui.text(value=str(repo_root / "configs" / "smoke.yaml"), label="Local configuration"), "output_dir": mo.ui.text(value=str(repo_root / "runs" / "smoke"), label="Run directory"), "allow_downloads": mo.ui.checkbox(value=False, label="Explicitly allow pinned HF model downloads")}).form(submit_button_label="Freeze settings — no model work")
    # Display the unsubmitted form without validating/loading a model automatically.
    settings_form
    # Export one unique UI object for model-free frozen preparation.
    return (settings_form,)


@app.cell
def _(ExplicitRunGate, Path, mo, prepare_run, settings_form, uuid4, with_execution):
    """Create frozen inputs, their guard, and a zero-click button in the same cell."""
    # Stop initial execution until settings have actually been submitted.
    mo.stop(settings_form.value is None)
    # Read/validate only local config and paired data; this does not load any weights.
    prepared_run = with_execution(prepare_run(Path(settings_form.value["config"])), output_dir=Path(settings_form.value["output_dir"]))
    # Bind action tickets to these exact frozen settings, not later unsaved field edits.
    submission_id = uuid4().hex
    # Freeze the separate model-download opt-in along with the submitted settings.
    allow_downloads = bool(settings_form.value["allow_downloads"])
    # Recreate the guard only together with a new submission and an unclicked button.
    run_gate = ExplicitRunGate()
    # Use a monotonic click counter rather than relying on run_button reset semantics in lazy mode.
    run_button = mo.ui.button(value=0, on_click=lambda count: count + 1, label="Run / resume frozen settings")
    # Show the actual content identity and full planned observation count before execution.
    mo.vstack([mo.md(f"**Run identity:** {prepared_run.run_id}\n\n**Planned trials:** {len(prepared_run.examples) * prepared_run.spec.replicates}"), run_button])
    # Export frozen inputs and their explicit action widget without starting a trial.
    return (allow_downloads, prepared_run, run_button, run_gate, submission_id)


@app.cell
def _(allow_downloads, mo, prepared_run, run_button, run_experiment, run_gate, submission_id):
    """Perform one explicitly clicked shared run; recomputation reuses its summary."""
    # Prevent initial/submission-only execution from triggering expensive work.
    mo.stop(run_button.value == 0)
    # Consume the click before loading models so errors cannot cause implicit retry loops.
    run_summary = run_gate.perform((submission_id, run_button.value), lambda: run_experiment(prepared_run, allow_downloads=allow_downloads))
    # Display local versus remotely acknowledged completion separately.
    mo.ui.table([run_summary.to_dict()])
    # Retain only a small shared-runner summary, not a hidden model object.
    return (run_summary,)


@app.cell
def _(mo, repo_root):
    """Define independent read-only inspection, including existing/restored runs."""
    # Require an explicit inspection submission without depending on the model action cell.
    inspection_form = mo.ui.text(value=str(repo_root / "runs" / "smoke"), label="Saved run directory").form(submit_button_label="Inspect saved results — read only")
    # Show the CPU-only analysis action independently of run/download settings.
    inspection_form
    # Export only this independent inspection control.
    return (inspection_form,)


@app.cell
def _(Path, attempt_rows, inspection_form, mo, paired_effects, read_run, summarize_run):
    """Read verified records without model imports, trial execution, writes, or uploads."""
    # Do not read a result folder until its inspection path is explicitly submitted.
    mo.stop(inspection_form.value is None)
    # Resolve the actual saved-result destination for read-only verification.
    _directory = Path(inspection_form.value).expanduser().resolve()
    # Verify every declared checksum and logical success before plotting any result.
    saved_snapshot = read_run(_directory)
    # Keep incomplete pairs visible in the descriptive table.
    effect_rows = paired_effects(saved_snapshot)
    # Display acknowledgement counts, paired comparisons, and safe failed-attempt categories.
    mo.vstack([mo.ui.table([summarize_run(_directory).to_dict()]), mo.ui.table(effect_rows), mo.ui.table(attempt_rows(saved_snapshot))])
    # Export only saved rows for independent display/plot recomputation.
    return (effect_rows,)


@app.cell
def _(effect_rows, paired_effect_figure):
    """Plot saved paired changes without depending on a model-loading/run action."""
    # Build a purely descriptive figure from already verified CPU-readable rows.
    paired_effect_figure(effect_rows)
    # Define no mutable experiment state in this display-only cell.
    return


# Start the frontend only when this notebook is executed directly, not when imported.
if __name__ == "__main__":
    # Let marimo manage the reactive UI; all expensive work remains behind explicit clicks.
    app.run()
