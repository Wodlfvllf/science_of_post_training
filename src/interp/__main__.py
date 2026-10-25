# Support explicit `python -m interp` without executing anything when imported.
"""Module-mode command-line entry point for the personal research workbench."""

from .cli import main

# Only a direct module execution may dispatch a CLI operation.
if __name__ == "__main__":
    # Propagate the shared CLI's conventional success/failure/interrupt exit code.
    raise SystemExit(main())
