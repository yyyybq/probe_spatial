#!/usr/bin/env bash
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
python_bin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
source_tag=independent_actionability_algebra_p8_final_v1_20260913
tag=independent_goal_conditioned_continuous_action_p8_v1_20260921
model=${MODEL:?MODEL is required}
request="$root/artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/manifests/feature_request_manifest_v2.jsonl"
feature_root="$root/features/$source_tag"
manifest_dir="$root/results/$source_tag/manifests"
source_manifest_dir="$root/artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/manifests"
package="$root/results/$tag"
protocol="$package/CONTINUOUS_ACTION_PROTOCOL.md"
output_dir="$package/raw_results/$model"

case "$model" in
  dino) receipt="$root/results/$source_tag/execution/full_dino_v1/coverage_receipt.json" ;;
  f3r) receipt="$root/results/$source_tag/coverage/f3r_coverage_receipt.json" ;;
  vjepa2) receipt="$root/results/$source_tag/coverage/vjepa2_coverage_receipt.json" ;;
  wan) receipt="$root/results/$source_tag/execution/full_wan_v1/coverage_receipt.json" ;;
  *) echo "unsupported MODEL=$model" >&2; exit 2 ;;
esac

test -x "$python_bin"
test -f "$protocol"
test "$(sha256sum "$protocol" | cut -d' ' -f1)" = db73875cd1d31afe9f41bcb14c2f8cb8e1233594c810b86ad426f96141e3d54c
nvidia-smi

"$python_bin" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "COMPLETE_VALID" and p["requested"] == p["verified"] == 12358 and p["missing_or_invalid"] == p["duplicates"] == p["fallback"] == p["raw_id_or_contract_mismatch"] == 0' "$receipt"

"$python_bin" "$root/scripts/run_independent_continuous_action.py" \
  --model "$model" \
  --request-manifest "$request" \
  --feature-root "$feature_root" \
  --manifest-dir "$manifest_dir" \
  --source-manifest-dir "$source_manifest_dir" \
  --protocol "$protocol" \
  --output-dir "$output_dir" \
  --device cuda

chmod -R a+rX "$output_dir"

