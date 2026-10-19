#!/usr/bin/env bash
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
python_bin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
source_tag=independent_actionability_algebra_p8_final_v1_20260913
tag=independent_goal_conditioned_continuous_action_p8_relational_v2_20260923
model=${MODEL:?MODEL is required}
base_request="$root/artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/manifests/feature_request_manifest_v2.jsonl"
base_feature_root="$root/features/$source_tag"
wan_request="$root/results/$tag/wan_l29_feature_request.jsonl"
wan_feature_root="$root/features/$tag"
manifest_dir="$root/results/$source_tag/manifests"
source_manifest_dir="$root/artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/manifests"
package="$root/results/$tag"
protocol="$package/CONTINUOUS_ACTION_PROTOCOL.md"
output_dir="$package/raw_results/$model"

test -x "$python_bin"
test ! -e "$output_dir"
test "$(sha256sum "$protocol" | cut -d' ' -f1)" = 005c9adfe2686b29deec8c21ef555a6c22108441fc3e33779e94cd3b583ea0d2
test "$(sha256sum "$base_request" | cut -d' ' -f1)" = fb1dbd2d71878edae7eae9034a16ee9809fc0c0cfb67352e0a301ad0edbed369
test "$(sha256sum "$wan_request" | cut -d' ' -f1)" = fdd80a9b84f9050c2a4c10e27d94dece74f1f6fde24471231e243f0082713818
test "$(sha256sum "$root/scripts/run_independent_continuous_action_relational_v2.py" | cut -d' ' -f1)" = 490eb4a2ca25fa1b97961e74a59a26a92fc1a45d7432c05eecf8f0f5553ec67d

case "$model" in
  dino) receipt="$root/results/$source_tag/execution/full_dino_v1/coverage_receipt.json" ;;
  f3r) receipt="$root/results/$source_tag/coverage/f3r_coverage_receipt.json" ;;
  vjepa2) receipt="$root/results/$source_tag/coverage/vjepa2_coverage_receipt.json" ;;
  wan) receipt="$package/coverage/wan_l29_coverage_receipt.json" ;;
  *) echo "unsupported MODEL=$model" >&2; exit 2 ;;
esac

"$python_bin" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "COMPLETE_VALID" and p["requested"] == p["verified"] and p["missing_or_invalid"] == p["duplicates"] == p["fallback"] == p["raw_id_or_contract_mismatch"] == 0' "$receipt"
nvidia-smi

"$python_bin" "$root/scripts/run_independent_continuous_action_relational_v2.py" \
  --model "$model" \
  --base-request "$base_request" \
  --base-feature-root "$base_feature_root" \
  --wan-request "$wan_request" \
  --wan-feature-root "$wan_feature_root" \
  --manifest-dir "$manifest_dir" \
  --source-manifest-dir "$source_manifest_dir" \
  --protocol "$protocol" \
  --output-dir "$output_dir" \
  --device cuda

chmod -R a+rX "$output_dir"
