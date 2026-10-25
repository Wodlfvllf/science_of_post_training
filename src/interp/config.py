# Safely load a single YAML configuration and preview its dataset before execution.
"""Configuration resolution and model-free experiment preparation.

Relative dataset/output paths resolve against the YAML file's directory. Paths,
budgets, and shard sizes are excluded from run identity; actual dataset content,
scientific settings, and installed package source are included.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .data import canonical_bytes, dataset_digest, digest_bytes, load_examples
from .schemas import Example, RunSpec


@dataclass(frozen=True)
class PreparedRun:
    """Validated local inputs ready for explicit model loading, with no side effects."""

    # Preserve resolved settings for the CLI and notebook to share.
    spec: RunSpec
    # Freeze all validated dataset observations in stable scheduling order.
    examples: tuple[Example, ...]
    # Pin semantic dataset content rather than input-file formatting.
    dataset_sha256: str
    # Pin the exact installed package code that will execute the trial.
    source_sha256: str
    # Identify logical trials independently of output location and invocation budgets.
    run_id: str


def load_config(path: Path) -> RunSpec:
    """Load one YAML mapping using SafeLoader and resolve its local paths.

    Unknown keys and ambiguous scientific settings fail Pydantic validation.
    Duplicate YAML keys are rejected: a later value must not silently override an
    earlier scientific setting. This function reads files but writes nothing.
    """
    # Convert an explicit user path to a stable config-directory anchor.
    config_path = path.expanduser().resolve()
    # Read only this local configuration file.
    with config_path.open("r", encoding="utf-8") as handle:
        # Safely parse the YAML node tree without allowing Python object construction.
        node = yaml.compose(handle, Loader=yaml.SafeLoader)
    # Reject empty configurations and scalar/list roots early.
    if not isinstance(node, yaml.MappingNode):
        # Require the flat top-level experiment configuration contract.
        raise ValueError("The configuration must be a YAML mapping.")
    # Check duplicate keys recursively before SafeLoader could overwrite them.
    _check_yaml_keys(node, set())
    # Construct primitives only after duplicate/cyclic mapping checks have passed.
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Validate every supplied field before accessing or resolving its paths.
    spec = RunSpec.model_validate(raw)
    # Resolve dataset paths independently of the process working directory.
    dataset_path = (config_path.parent / spec.dataset.path.expanduser()).resolve()
    # Resolve operational output paths by the same explicit rule.
    output_path = (config_path.parent / spec.execution.output_dir.expanduser()).resolve()
    # Revalidate the resolved record instead of bypassing validators with model_copy().
    resolved = spec.model_dump(mode="python")
    # Replace only the local dataset location, not its scientific identity.
    resolved["dataset"]["path"] = dataset_path
    # Replace only the local artifact destination.
    resolved["execution"]["output_dir"] = output_path
    # Return a fully validated resolved configuration.
    return RunSpec.model_validate(resolved)


def _check_yaml_keys(node: yaml.Node, ancestors: set[int]) -> None:
    """Reject duplicate/non-string mapping keys, YAML merges, and cyclic aliases."""
    # Detect recursive aliases without treating harmless reused nodes as cycles.
    if id(node) in ancestors:
        # Avoid recursion errors or ambiguous recursively defined settings.
        raise ValueError("Cyclic YAML aliases are not supported.")
    # Track ancestors only along this branch of the node tree.
    branch = ancestors | {id(node)}
    # Inspect each mapping's actual key nodes before values are constructed.
    if isinstance(node, yaml.MappingNode):
        # Track exact string keys within this one mapping.
        seen: set[str] = set()
        # Visit every key/value pair without applying implicit YAML merge semantics.
        for key, value in node.value:
            # Restrict keys to ordinary strings and disallow the special merge tag.
            if not isinstance(key, yaml.ScalarNode) or key.tag != "tag:yaml.org,2002:str":
                # Make the accepted YAML subset explicit and predictable.
                raise ValueError("Configuration keys must be strings; YAML merges are unsupported.")
            # Compare the literal key to previous keys in this mapping.
            if key.value in seen:
                # Do not silently choose whichever duplicate happened to appear last.
                raise ValueError(f"Duplicate YAML configuration key: {key.value}")
            # Mark the accepted key as used.
            seen.add(key.value)
            # Apply the same conservative rules to nested settings.
            _check_yaml_keys(value, branch)
    # Inspect list elements for nested mappings or alias cycles as well.
    elif isinstance(node, yaml.SequenceNode):
        # Visit each item of the YAML list.
        for value in node.value:
            # Recurse using the current ancestor branch.
            _check_yaml_keys(value, branch)


def source_digest() -> str:
    """Fingerprint package Python sources only, excluding personal files and credentials.

    This is conservative: even a package-source comment change invalidates resume.
    It is not a replacement for committing/archiving a research recipe in git.
    """
    # Anchor the fingerprint to the actual installed package, including editable installs.
    package_dir = Path(__file__).resolve().parent
    # Collect filenames and individual source digests in deterministic order.
    entries = [(path.relative_to(package_dir).as_posix(), digest_bytes(path.read_bytes())) for path in sorted(package_dir.rglob("*.py"))]
    # Bind filenames and contents so moving/renaming a module changes identity too.
    return digest_bytes(canonical_bytes(entries))


def compute_run_id(spec: RunSpec, data_sha: str, code_sha: str) -> str:
    """Compute experimental identity, excluding machine paths and invocation budgets."""
    # Include every scientific setting but not output locations or expected-hash duplication.
    science = spec.model_dump(mode="json", exclude={"execution": True, "dataset": {"path", "expected_sha256"}})
    # Bind the resolved science to actual dataset content and executing package source.
    return digest_bytes(canonical_bytes({"spec": science, "dataset": data_sha, "source": code_sha}))


def with_execution(prepared: PreparedRun, **overrides: Any) -> PreparedRun:
    """Apply validated operational overrides without changing scientific trial identity.

    Only ExecutionSpec fields may be overridden. An explicit new output path uses
    the caller's cwd; original YAML paths were already resolved by load_config().
    """
    # Preserve all scientific fields in ordinary primitives for revalidation.
    raw = prepared.spec.model_dump(mode="python")
    # Apply only operational overrides under the validated execution schema.
    raw["execution"].update(overrides)
    # Resolve a supplied output path without changing the archived dataset identity.
    if "output_dir" in overrides:
        # Make the caller's explicit operational destination unambiguous.
        raw["execution"]["output_dir"] = Path(overrides["output_dir"]).expanduser().resolve()
    # Revalidate instead of using unvalidated Pydantic model_copy updates.
    spec = RunSpec.model_validate(raw)
    # Return a new preparation object while keeping content and logical trial identities.
    return replace(prepared, spec=spec)


def prepare_run(path: Path) -> PreparedRun:
    """Validate settings and paired data and compute identity without loading a model.

    Returns:
        Resolved inputs reusable by the explicit execution entry points.

    Raises:
        ValueError: Invalid configuration/data or a mismatched pinned dataset digest.
        OSError: A required local file is unavailable.
    """
    # Load the single validated configuration and resolve its paths.
    spec = load_config(path)
    # Validate all pairs and lineage before any optional heavyweight imports.
    examples = load_examples(spec.dataset.path, spec.conditions)
    # Hash exact validated dataset semantics in stable observation order.
    data_sha = dataset_digest(examples)
    # Verify an externally predeclared dataset identity, when provided.
    if spec.dataset.expected_sha256 is not None and spec.dataset.expected_sha256 != data_sha:
        # Stop before a changed dataset can be mislabeled as a replication.
        raise ValueError("The validated dataset does not match expected_sha256.")
    # Pin the package source that implements the scientific calculation.
    code_sha = source_digest()
    # Bind scientific settings, validated data, and actual executing source together.
    run_id = compute_run_id(spec, data_sha, code_sha)
    # Return a frozen, model-free preparation result for shared UI/CLI execution.
    return PreparedRun(spec, examples, data_sha, code_sha, run_id)
