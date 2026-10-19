#!/usr/bin/env bash
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
python_bin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
tag=independent_goal_conditioned_continuous_action_p8_relational_v2_20260923
package="$root/results/$tag"
output_dir=${OUTPUT_DIR:-$package}

"$python_bin" "$root/scripts/summarize_independent_continuous_action_relational_v2.py" \
  --run-root "$package/raw_results" \
  --output-dir "$output_dir" \
  --train-manifest "$root/artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/manifests/algebra_p8_independent_v1_train.jsonl"
