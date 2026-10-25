"""Safe, actionable study failures without logging dependency messages or prompts.

Only workbench-authored messages and numerical measurements belong in StudyError.
Use study_stage around optional-library calls to identify the failing operation
while retaining the original exception as a Python cause for local debugging.
The CLI prints these controlled messages, not arbitrary exception strings.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


class StudyError(ValueError):
    """A workbench-controlled study error whose message is safe to show in the CLI."""


@contextmanager
def study_stage(name: str) -> Iterator[None]:
    """Label a fixed operation without exposing raw model, HTTP, or tokenizer errors.

    Callers must supply a literal stage label, never prompt text or credentials.
    Controlled errors retain their explanation; interrupts retain their semantics.
    The original exception remains chained for deliberate in-process debugging.
    """
    try:
        yield
    except StudyError:
        raise
    except Exception as error:
        # Dependency exceptions can contain request headers, local paths, or text inputs.
        raise StudyError(
            f"{name} failed ({type(error).__name__}); raw dependency details are hidden."
        ) from error