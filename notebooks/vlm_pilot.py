"""Marimo entry notebook: explicit pilot actions and independent read-only monitoring.

The kernel needs marimo only. VLM packages are installed in .venv-vlm by an
explicit setup action; heavy jobs execute via the existing standalone CLI.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib.util
    import sys
    from pathlib import Path
    from uuid import uuid4

    import marimo as mo

    repo_root = Path(__file__).resolve().parent.parent
    _spec = importlib.util.spec_from_file_location(
        "vlm_notebook_controls", repo_root / "scripts" / "vlm_notebook.py")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
    ClickGate = _module.ClickGate
    controller = _module.NotebookPilot(repo_root)
    return ClickGate, controller, mo, repo_root, uuid4


@app.cell
def _(mo):
    mo.md("""
    # Reading or Routing · VLM pilot

    **No automatic installs, downloads or inference on opening this notebook.**
    Choose an action, submit settings, then click **Launch once**.

    1. **setup** installs a separate VLM environment; the notebook kernel stays unchanged.
    2. **smoke** exercises meaningless random-model plumbing, or **prepare** creates the selected data/plan.
    3. **preflight** / **run** start one checkpoint on an unused GPU. Begin with a 12-trial budget.
    4. Discover sites before **freeze** and validation/confirmation; site numbers are not supplied as findings.

    Jobs use detached subprocesses and disk logs. Refreshing/recomputing cells cannot
    relaunch a consumed click. A tab disconnect does not deliberately kill work, but
    container deletion or a platform reset still can. Published successes resume;
    local completion is **not** verified Drive backup. GPU/hypothesis results are unvalidated.
    """)
    return


@app.cell
def _(mo):
    settings_form = mo.ui.dictionary({
        "action": mo.ui.dropdown(["setup", "smoke", "prepare", "preflight", "run", "freeze",
                      "baseline-all", "audit-weights", "analyze"],
                                  value="setup", label="Explicit action"),
        "mode": mo.ui.dropdown(["smoke", "pilot"], value="smoke", label="Data/model configuration"),
        "torch_index": mo.ui.dropdown(["", "https://download.pytorch.org/whl/cu130",
                                       "https://download.pytorch.org/whl/cu128"], value="",
                                      label="Setup: optional public CUDA wheel index"),
        "checkpoint": mo.ui.dropdown(["fixture-mpo", "fixture-gspo", "2b-mpo", "2b-gspo",
                                      "8b-mpo", "8b-gspo", "2b-cpt", "2b-sft", "8b-cpt", "8b-sft"],
                                     value="fixture-mpo", label="Checkpoint (must match mode)"),
        "phase": mo.ui.dropdown(["baseline", "routing", "weights"], value="baseline", label="Hypothesis phase"),
        "split": mo.ui.dropdown(["engineering", "discovery", "pilot_validation", "confirmation"],
                                value="discovery", label="Source-family split"),
        "device": mo.ui.text(value="cpu", label="Explicit device: cpu for fixture; free cuda:N for pilot"),
        "allow_downloads": mo.ui.checkbox(value=False, label="Explicitly permit pinned HF model downloads"),
        "sites": mo.ui.text(value="", label="Run/P0 sites: layer:KV-group pairs (discovery findings)"),
        "source_checkpoint": mo.ui.text(value="", label="Weight source / audit's second checkpoint"),
        "projections": mo.ui.dropdown(["q_proj,k_proj", "q_proj", "k_proj", "v_proj,o_proj"],
                                      value="q_proj,k_proj", label="Weight operation"),
        "include_norms": mo.ui.checkbox(value=False, label="Also change layer-wide shared Q/K norms"),
        "plan": mo.ui.text(value="", label="Plan path (blank uses the mode default)"),
        "protocol": mo.ui.text(value="", label="Frozen protocol path for validation/confirmation"),
        "output": mo.ui.text(value="", label="Output path (blank selects a distinct operation directory)"),
        "bounded": mo.ui.checkbox(value=True, label="Limit this invocation to the trial budget below"),
        "max_new_trials": mo.ui.number(start=0, stop=100000, value=12, step=1, label="New-trial budget"),
        "generate": mo.ui.checkbox(value=False, label="Baseline only: unrestricted greedy answer audit"),
        "prompt_profile": mo.ui.dropdown(["common", "native"], value="common", label="Prompt profile"),
        "save_captures": mo.ui.checkbox(value=False, label="Save bounded activation captures"),
        "sites_fixture": mo.ui.text(value="", label="Freeze: fixture sites (engineering only)"),
        "sites_2b": mo.ui.text(value="", label="Freeze: discovered 2B sites"),
        "sites_8b": mo.ui.text(value="", label="Freeze: discovered 8B sites"),
        "note": mo.ui.text(value="", label="Freeze: discovery selection rationale"),
        "analysis_runs": mo.ui.text_area(value="", label="Analyze: saved trial-run directories, one per line"),
        "allow_incomplete": mo.ui.checkbox(value=False, label="Analyze: label incomplete descriptive results explicitly"),
        "confirmed": mo.ui.checkbox(value=False, label="I intend this explicit install/data/model action"),
    }).form(submit_button_label="Freeze action settings — no execution")
    settings_form
    return (settings_form,)


@app.cell
def _(ClickGate, controller, mo, settings_form, uuid4):
    mo.stop(settings_form.value is None)
    frozen_settings = dict(settings_form.value)
    if frozen_settings["bounded"]:
        frozen_settings["max_new_trials"] = int(frozen_settings["max_new_trials"])
    else:
        frozen_settings["max_new_trials"] = None
    submission = uuid4().hex
    gate = ClickGate()
    launch_button = mo.ui.button(value=0, on_click=lambda count: count + 1, label="Launch once / resume")
    try:
        _commands, _environment, _output = controller.commands(frozen_settings)
        _preview = mo.plain_text("\n".join(" ".join(argv) for argv in _commands))
    except (ValueError, RuntimeError) as _error:
        _preview = mo.callout(str(_error), kind="warn")
    mo.vstack([_preview, launch_button])
    return frozen_settings, gate, launch_button, submission


@app.cell
def _(controller, frozen_settings, gate, launch_button, mo, submission):
    mo.stop(launch_button.value == 0)
    try:
        launch_receipt = gate.perform((submission, launch_button.value),
                                      lambda: controller.launch(frozen_settings))
        _view = mo.ui.table([launch_receipt])
    except (ValueError, RuntimeError, OSError) as _error:
        _view = mo.callout(str(_error), kind="danger")
    _view
    return


@app.cell
def _(mo):
    refresh_button = mo.ui.button(value=0, on_click=lambda count: count + 1, label="Refresh status/log — read only")
    refresh_button
    return (refresh_button,)


@app.cell
def _(controller, mo, refresh_button):
    _refresh_ticket = refresh_button.value
    _jobs = controller.monitor()
    if _jobs:
        _view = mo.vstack([mo.ui.table(_jobs), mo.md("### Latest worker log"),
                          mo.plain_text(controller.log_tail(_jobs[0]["job_id"]))])
    else:
        _view = mo.md("No notebook-owned jobs. Submit an explicit action above to start one.")
    _view
    return


@app.cell
def _(mo):
    inspect_form = mo.ui.text(value="", label="Saved trial-run directory (not a job-log directory)").form(
        submit_button_label="Verify published trials — read only")
    inspect_form
    return (inspect_form,)


@app.cell
def _(controller, inspect_form, mo):
    mo.stop(inspect_form.value is None or not inspect_form.value.strip())
    try:
        _summary = controller.inspect_saved(inspect_form.value.strip())
        _view = mo.ui.table([_summary])
    except (ValueError, RuntimeError, OSError) as _error:
        _view = mo.callout(str(_error), kind="danger")
    _view
    return


if __name__ == "__main__":
    app.run()