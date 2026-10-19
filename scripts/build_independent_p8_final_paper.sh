#!/usr/bin/env bash
# Rebuild every aggregate table, figure and result-dependent report from raw CSVs.
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
python_bin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
tag=independent_actionability_algebra_p8_final_v1_20260913
package="$root/results/$tag"
paper=${PAPER_OUTPUT_DIR:-"$package/paper"}

test ! -e "$paper"
for model in dino f3r vjepa2 wan; do
  test -s "$package/raw_results/$model/actionability_per_sample.csv"
  test -s "$package/raw_results/$model/dynamics_per_sample.csv"
  test -s "$package/raw_results/$model/algebra_per_sample.csv"
  test -s "$package/raw_results/$model/training_and_integrity.json"
done

"$python_bin" "$root/scripts/summarize_independent_p8_final.py" \
  --run-root "$package/raw_results" --output-dir "$paper"
"$python_bin" "$root/scripts/render_independent_p8_paper_reports.py" \
  --paper-dir "$paper" --package-root "$package"
