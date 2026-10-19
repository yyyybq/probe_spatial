#!/usr/bin/env bash
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
python_bin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
source_tag=independent_actionability_algebra_p8_final_v1_20260913
tag=independent_goal_conditioned_continuous_action_p8_v1_20260921
package="$root/results/$tag"
output_dir=${OUTPUT_DIR:-$package}

for model in dino f3r vjepa2 wan; do
  test -s "$package/raw_results/$model/continuous_action_per_sample.csv"
  test -s "$package/raw_results/$model/training_and_integrity.json"
done

mkdir -p "$output_dir"

"$python_bin" "$root/scripts/summarize_independent_continuous_action.py" \
  --run-root "$package/raw_results" \
  --output-dir "$output_dir" \
  --train-manifest "$root/artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/manifests/algebra_p8_independent_v1_train.jsonl" \
  --old-actionability-table "$root/results/$source_tag/paper/tables/TABLE_1_ACTIONABILITY.csv"

chmod a+r "$output_dir"/continuous_action_*.csv "$output_dir"/CONTINUOUS_ACTION_INTEGRITY.json \
  "$output_dir"/CONTINUOUS_ACTION_RESULTS.md "$output_dir"/TABLE_CONTINUOUS_ACTION_*.csv \
  "$output_dir"/TABLE_CONTINUOUS_ACTION_*.tex
