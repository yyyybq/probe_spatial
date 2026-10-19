#!/usr/bin/env bash
# Verify exact full-model coverage, then run the frozen final probe/evaluator.
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
python_bin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
tag=independent_actionability_algebra_p8_final_v1_20260913
protocol_sha=6bf96b1d30cb2d9e55126fdbc2ecca1a62a25d13d1123c2066a614d2455d6fb4
request="$root/artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/manifests/feature_request_manifest_v2.jsonl"
model=${MODEL:?MODEL is required}
feature_root="$root/features/$tag"
manifest_dir="$root/results/$tag/manifests"
output_dir="$root/results/$tag/raw_results/$model"
package="$root/results/$tag"
join_audit="$package/coverage/FEATURE_LABEL_JOIN_AUDIT.json"

case "$model" in
  dino) receipt="$package/execution/full_dino_v1/coverage_receipt.json" ;;
  f3r) receipt="$package/coverage/f3r_coverage_receipt.json" ;;
  vjepa2) receipt="$package/coverage/vjepa2_coverage_receipt.json" ;;
  wan) receipt="$package/execution/full_wan_v1/coverage_receipt.json" ;;
  *) echo "unsupported MODEL=$model" >&2; exit 2 ;;
esac
test -x "$python_bin"
test -f "$request"
test -f "$root/results/$tag/FINAL_PROTOCOL.md"
nvidia-smi

"$python_bin" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "COMPLETE_VALID" and p["requested"] == p["verified"] == 12358 and p["missing_or_invalid"] == p["duplicates"] == p["fallback"] == p["raw_id_or_contract_mismatch"] == 0' "$receipt"
"$python_bin" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "COMPLETE_VALID" and p["scene_boundaries"]["overlap"] == 0 and all(v["request_units"] == 12358 for v in p["per_model"].values())' "$join_audit"

"$python_bin" "$root/scripts/run_independent_p8_final_analysis.py" \
  --model "$model" --request-manifest "$request" \
  --feature-root "$feature_root" --manifest-dir "$manifest_dir" \
  --output-dir "$output_dir" --device cuda

# tempfile.mkdtemp is deliberately private while results are incomplete.  Once
# the evaluator atomically promotes the directory, make the immutable outputs
# readable to the submit-side summarizer across SCO's squashed NFS identity.
chmod -R a+rX "$output_dir"
