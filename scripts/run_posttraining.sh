#!/usr/bin/env bash
# Run from the repository root on Linux, with the tested environment activated.
# No package upgrades, source edits, automatic fallbacks, or destructive resets.
set -Eeuo pipefail
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
test -f pyproject.toml || { echo 'Run from the repository root.' >&2; exit 1; }
command -v flock >/dev/null || { echo 'Linux flock is required.' >&2; exit 1; }
mkdir -p runs/posttrain/logs data/posttrain
# Hold one orchestration lock; individual runs retain their own writer locks.
exec 9>runs/posttrain/.pipeline.lock
flock -n 9 || { echo 'Another pipeline is already running.' >&2; exit 1; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
exec > >(tee "runs/posttrain/logs/pipeline-$STAMP.log") 2>&1
trap 'echo "Stopped: line $LINENO. Fix the error and rerun; saved trials are preserved." >&2' ERR
trap 'echo "Interrupted; rerun to resume saved trials."; exit 130' INT
trap 'echo "Terminated; rerun to resume saved trials."; exit 143' TERM

# Keep the existing cache location rather than forcing a second weight download.
echo "Using $DEVICE; Hugging Face cache: ${HF_HOME:-existing Hugging Face default}"
"$PYTHON" -m pip check
"$PYTHON" -c 'import sys, torch; assert torch.cuda.is_available(), "CUDA unavailable"; d=torch.device(sys.argv[1]); assert d.type == "cuda", "Select a CUDA device"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(d))' "$DEVICE"
"$PYTHON" -m pytest -q
"$PYTHON" -m pip freeze > "runs/posttrain/logs/environment-$STAMP.txt"

cli() { "$PYTHON" -m interp posttrain "$@"; }

for study in math repair; do
  plan="data/posttrain/$study-fp32-plan.json"
  dataset="data/posttrain/repair-pilot.jsonl"
  kind=repair
  if [[ "$study" == math ]]; then
    dataset="data/posttrain/gsm8k-pilot.jsonl"
    kind=gsm8k
  fi
  if [[ ! -e "$plan" ]]; then
    # Never silently regenerate a lost plan when an archived run already exists.
    for checkpoint in base rl sft dpo; do
      if [[ -e "runs/posttrain/$study-fp32-$checkpoint/metadata.json" ]]; then
        echo "Missing $plan but archived results exist. Restore the original plan before continuing." >&2
        exit 1
      fi
    done
    if [[ ! -e "$dataset" ]]; then
      cli data "$kind" --split pilot --count 50 --seed 2026 \
        --allow-downloads --output "$dataset"
    fi
    cli lock "configs/posttrain/$study-pilot.yaml" --output "$plan"
  fi

  for checkpoint in base rl sft dpo; do
    directory="runs/posttrain/$study-fp32-$checkpoint"
    echo "=== $study / $checkpoint ==="
    # Validate plan/job identity without loading a model. This also handles
    # already-complete runs without repeating downloads or numerical preflight.
    state="$(cli run "$plan" --checkpoint "$checkpoint" --device "$DEVICE" \
      --output-dir "$directory" --max-new-trials 0)"
    reason="$(printf '%s\n' "$state" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["stop_reason"])')"
    if [[ "$reason" != complete ]]; then
      cli preflight "$plan" --checkpoint "$checkpoint" --device "$DEVICE" --allow-downloads
      # run resumes matching existing jobs and skips only published successes.
      cli run "$plan" --checkpoint "$checkpoint" --device "$DEVICE" \
        --output-dir "$directory" --allow-downloads
    fi
    cli inspect "$directory" | "$PYTHON" -c '
import json, sys
result = json.load(sys.stdin)
print(json.dumps(result, indent=2))
if result["missing_trials"]:
    sys.exit("Run stopped incomplete; rerun the script to continue. No report generated.")
'
  done
  cli analyze "runs/posttrain/$study-fp32-base" "runs/posttrain/$study-fp32-rl" \
    "runs/posttrain/$study-fp32-sft" "runs/posttrain/$study-fp32-dpo" \
    --output "runs/posttrain/$study-fp32-report-$STAMP"
done
echo 'Both experiments completed across all four checkpoints; reports are under runs/posttrain/.'
echo 'Copy data/posttrain and runs/posttrain to verified persistent storage before stopping the server.'