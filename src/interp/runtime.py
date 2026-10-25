# Use these helpers only during explicit execution, never to trigger work on import.
"""Execution provenance, stable trial seeds, and scoped model/RNG state.

No environment variables or authentication secrets are captured. RNG scopes restore
the caller's Python and selected Torch states even if a trial raises an exception.
They are intended for the initial single-threaded, single-model runner, not concurrent
threads sharing global RNG state.
"""

from __future__ import annotations

import platform
import random
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from .data import canonical_bytes, digest_bytes

if TYPE_CHECKING:
    import torch


class CancellationRequested(RuntimeError):
    """A cooperative stop before a trial can produce a complete result."""


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for persisted execution records."""
    # Avoid locale-dependent or naive timestamps when comparing remote sessions.
    return datetime.now(UTC)


def trial_id(run_id: str, example_id: str, replicate: int) -> str:
    """Identify a logical trial independently of attempts, batching, and run directory."""
    # Use structured serialization so concatenated identifiers cannot collide ambiguously.
    return digest_bytes(canonical_bytes([run_id, example_id, replicate]))


def trial_seed(base_seed: int, pair_id: str, replicate: int) -> int:
    """Derive an order-independent 32-bit seed shared by the two paired conditions.

    Sharing a seed is a matched-randomness control, not a claim that different prompts
    produce identical sampling paths. Unrelated pairs and repeats receive new seeds.
    """
    # Hash the declared base seed and independent paired-observation identity.
    digest = digest_bytes(canonical_bytes([base_seed, pair_id, replicate]))
    # Convert the first eight hex characters to a portable 32-bit integer.
    return int(digest[:8], 16)


def runtime_profile(device: torch.device) -> dict[str, Any]:
    """Capture conservative numerical compatibility information without dumping secrets.

    Call after explicit model loading. Resume requires exact equality of this profile;
    cross-device or changed-library reproducibility is deliberately not promised.
    """
    # Load the numerical library only when the caller explicitly starts model work.
    import torch

    # Record relevant package versions without importing UI or remote-storage libraries.
    packages: dict[str, str] = {}
    # Keep the compatibility list limited to code that affects computation/persistence.
    for name in ("interp-research", "torch", "transformers", "tokenizers", "safetensors", "pydantic", "PyYAML", "filelock"):
        # Handle a source checkout that has not been installed as a distribution yet.
        try:
            # Preserve the actually installed version, not a guessed dependency pin.
            packages[name] = version(name)
        # Make a missing distribution visible rather than inferring a version.
        except PackageNotFoundError:
            # Record an explicit absence for compatibility checking.
            packages[name] = "not-installed"
    # Capture settings that can change floating-point results on the same hardware.
    profile: dict[str, Any] = {
        "python": sys.version.split()[0],  # Pin the executed Python version.
        "system": platform.system(),  # Distinguish macOS and Linux numerical profiles.
        "machine": platform.machine(),  # Distinguish ARM and x86 execution.
        "packages": packages,  # Preserve the tested calculation/serialization stack.
        "device": str(device),  # Record the actual selected model device.
        "threads": torch.get_num_threads(),  # Refuse silently changed CPU reductions.
        "interop_threads": torch.get_num_interop_threads(),  # Preserve thread-pool policy.
        "matmul_precision": torch.get_float32_matmul_precision(),  # Record matmul policy.
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),  # Record RNG/kernel policy.
        "default_dtype": str(torch.get_default_dtype()),  # Capture surrounding Torch defaults.
        "cuda_version": torch.version.cuda,  # Distinguish runtime wheel profiles.
    }
    # Capture CUDA-only settings without selecting or initializing a GPU on CPU runs.
    if device.type == "cuda":
        # Preserve the actual GPU name rather than assuming the reported grant hardware.
        profile["gpu"] = torch.cuda.get_device_name(device)
        # Record TF32 matmul permission separately from parameter dtype.
        profile["allow_tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        # Record cuDNN TF32 permission for completeness.
        profile["allow_tf32_cudnn"] = torch.backends.cudnn.allow_tf32
        # Preserve automatic kernel-selection policy.
        profile["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        # Preserve deterministic cuDNN policy.
        profile["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    # Return data only; do not modify backend flags or process environment variables.
    return profile


@contextmanager
def isolated_seed(seed: int, device: torch.device) -> Iterator[None]:
    """Temporarily seed Python and CPU/selected-CUDA Torch RNGs, restoring on exit.

    NumPy is not used by this trial recipe; recipes using its RNG should use their
    own local ``numpy.random.Generator`` rather than process-global state.
    """
    # Import Torch only inside the explicitly requested execution scope.
    import torch

    # Preserve the caller's Python random state before changing it.
    python_state = random.getstate()
    # Snapshot only the selected CUDA device, never all available GPUs.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Guarantee restoration of the Python state as well as Torch's scoped state.
    try:
        # Fork CPU and selected GPU state without affecting unrelated device RNGs.
        with torch.random.fork_rng(devices=devices):
            # Seed Python for any ordinary trial-local random choices.
            random.seed(seed)
            # Seed CPU Torch directly; torch.manual_seed() would also seed other devices.
            torch.random.default_generator.manual_seed(seed)
            # Seed the selected GPU only if the model actually executes there.
            if devices:
                # Select the model's device temporarily without changing global selection.
                with torch.cuda.device(devices[0]):
                    # Set that GPU's trial-local RNG state.
                    torch.cuda.manual_seed(seed)
            # Execute the caller's ordinary function inside the isolated state.
            yield
    # Restore Python state even for model errors or notebook interrupts.
    finally:
        # Leave the surrounding notebook's unrelated randomness unchanged.
        random.setstate(python_state)


@contextmanager
def inference_scope(model: Any) -> Iterator[None]:
    """Temporarily select evaluation and inference modes, restoring every module flag.

    Restoring each module separately preserves deliberately mixed train/eval states.
    This scope is for behavioral inference, not future attribution/training functions.
    """
    # Import Torch only for an explicit forward/generation request.
    import torch

    # Snapshot every module flag rather than assuming all submodules share one mode.
    training_flags = [(module, module.training) for module in model.modules()]
    # Disable dropout while preserving a reliable cleanup boundary.
    try:
        # Evaluation mode and autograd control are independent settings.
        model.eval()
        # Disable autograd only for this bounded behavioral calculation.
        with torch.inference_mode():
            # Execute the requested model operation without retaining a graph.
            yield
    # Restore caller-owned module state even if execution fails or is interrupted.
    finally:
        # Restore flags directly so heterogeneous modes are not overwritten recursively.
        for module, was_training in training_flags:
            # Leave the model exactly as it was before this inference call.
            module.training = was_training
