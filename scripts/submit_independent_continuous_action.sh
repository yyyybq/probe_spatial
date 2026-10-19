#!/usr/bin/env bash
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
tag=independent_goal_conditioned_continuous_action_p8_v1_20260921
entry="$root/scripts/run_independent_continuous_action_job.sh"
aec2_name=${AEC2_NAME:-h800}

for model in dino f3r vjepa2 wan; do
  output="$root/results/$tag/raw_results/$model/training_and_integrity.json"
  if [[ -s "$output" ]]; then
    echo "[skip] completed model=$model"
    continue
  fi
  job_name="indp8-cont-action-${model}-v1"
  EXTRA_ENV_EXPORTS="MODEL=$model" bash /mnt/umm/users/yinbaiqiao/submit.sh "$aec2_name" 1 1 "$job_name" "$entry"
done
