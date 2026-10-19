#!/usr/bin/env bash
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
python_bin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
tag=independent_goal_conditioned_continuous_action_p8_relational_v2_20260923
protocol_sha=005c9adfe2686b29deec8c21ef555a6c22108441fc3e33779e94cd3b583ea0d2
science_tag=independent_actionability_algebra_p8_continuous_relational_v2_20260923
receipt="$root/results/$tag/coverage/wan_l29_coverage_receipt.json"

test ! -e "$receipt"
"$python_bin" "$root/scripts/verify_independent_p8_final_features.py" \
  --science-tag "$science_tag" \
  --protocol-sha256 "$protocol_sha" \
  --request-manifest "$root/results/$tag/wan_l29_feature_request.jsonl" \
  --feature-root "$root/features/$tag" \
  --model wan \
  --artifact-output "$receipt"
