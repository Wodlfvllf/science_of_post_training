#!/usr/bin/env bash
# A separate environment/worker. Never activate, update or kill the text-study worker.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_DIR="${VLM_ENV_DIR:-$ROOT/.venv-vlm}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"

if [[ "${1:-}" == "setup" ]]; then
  if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$PYTHON_BOOTSTRAP" -m venv "$ENV_DIR"
  fi
  "$ENV_DIR/bin/python" -m pip install --upgrade pip
  # Supply a public PyTorch wheel index explicitly if the remote CUDA stack needs it.
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    "$ENV_DIR/bin/python" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
  fi
  "$ENV_DIR/bin/python" -m pip install -e '.[vlm,dev]'
  "$ENV_DIR/bin/python" -m pip check
  printf '%s\n' 'Separate VLM environment installed. No models downloaded or workers started.'
  exit 0
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  printf '%s\n' 'Run: bash scripts/run_vlm_pilot.sh setup' >&2
  exit 1
fi
PYTHON="$ENV_DIR/bin/python"

if [[ "${1:-}" == "smoke" ]]; then
  # Restrict only these random CPU-fixture subprocesses; never change scientific GPU policy.
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  "$PYTHON" -m pytest tests/vlm -q
  if [[ ! -e data/vlm-smoke/dataset.json ]]; then
    "$PYTHON" -m vlm_pilot data data/vlm-smoke --engineering 1 --discovery 1 \
      --pilot-validation 0 --confirmation 0 --contact-sheet runs/vlm-smoke-scenes.png
  fi
  mkdir -p runs/vlm-smoke
  if [[ ! -e runs/vlm-smoke/plan.json ]]; then
    "$PYTHON" -m vlm_pilot lock configs/vlm/smoke.yaml --output runs/vlm-smoke/plan.json
  fi
  for checkpoint in fixture-mpo fixture-gspo; do
    "$PYTHON" -m vlm_pilot run runs/vlm-smoke/plan.json --checkpoint "$checkpoint" \
      --phase baseline --split discovery --output "runs/vlm-smoke/$checkpoint-baseline"
    "$PYTHON" -m vlm_pilot run runs/vlm-smoke/plan.json --checkpoint "$checkpoint" \
      --phase routing --split discovery --sites 0:0 --output "runs/vlm-smoke/$checkpoint-routing"
  done
  "$PYTHON" -m vlm_pilot run runs/vlm-smoke/plan.json --checkpoint fixture-gspo \
    --phase weights --split discovery --sites 0:0 --source-checkpoint fixture-mpo \
    --output runs/vlm-smoke/fixture-gspo-weights
  printf '%s\n' 'Offline random-model plumbing complete; not hypothesis evidence.'
  exit 0
fi

if [[ "${1:-}" == "baseline" ]]; then
  # This explicit action permits pinned native model downloads. DEVICE must identify a FREE GPU.
  DEVICE="${DEVICE:-cuda:0}"
  PLAN="${VLM_PLAN:-runs/vlm-pilot/plan.json}"
  if [[ ! -e data/vlm-pilot/dataset.json ]]; then
    "$PYTHON" -m vlm_pilot data data/vlm-pilot --contact-sheet runs/vlm-pilot-scenes.png
  fi
  mkdir -p "$(dirname "$PLAN")"
  if [[ ! -e "$PLAN" ]]; then
    "$PYTHON" -m vlm_pilot lock configs/vlm/pilot.yaml --output "$PLAN"
  fi
  for checkpoint in 2b-mpo 2b-gspo 8b-mpo 8b-gspo; do
    "$PYTHON" -m vlm_pilot run "$PLAN" --checkpoint "$checkpoint" --split discovery \
      --phase baseline --device "$DEVICE" --allow-downloads \
      --output "runs/vlm-pilot/$checkpoint-baseline-discovery"
  done
  exit 0
fi

exec "$PYTHON" -m vlm_pilot "$@"