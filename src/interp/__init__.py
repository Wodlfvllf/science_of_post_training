# Importing this package must not load models, touch results, or contact the network.
"""Personal research primitives with explicit execution and persistence boundaries.

Import individual modules for their functions. In particular, model libraries are
loaded only inside explicit model-execution calls, not by importing ``interp``.
"""

# Keep the package version available without importing any optional dependencies.
__version__ = "0.1.0"
