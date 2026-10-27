#!/usr/bin/env bash
# Run from any directory. Only the trusted checkout helper is executed, never ZIP code.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BOOTSTRAP="${RECOVERY_PYTHON:-python3}"
command -v "$PYTHON_BOOTSTRAP" >/dev/null || {
  echo 'Python >=3.11 is required; set RECOVERY_PYTHON to an available interpreter.' >&2
  exit 1
}
# The helper takes OS locks before staging, restoration, or environment setup.
exec "$PYTHON_BOOTSTRAP" -I "$ROOT/scripts/recover_posttraining.py" "$@"