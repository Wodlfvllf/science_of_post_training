"""Personal post-training studies: freeze inputs, generate, verify, then compare.

This package contains two project recipes, not a general experiment framework.
Importing it does not load Torch, download data, or start an experiment. Use the
``interp posttrain`` CLI or the ordinary functions in ``plan``, ``runner``, and
``metrics`` from a script or a marimo cell with an explicit run action.
"""