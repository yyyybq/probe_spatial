#!/usr/bin/env bash
# One node, one disjoint deterministic shard per visible GPU, then full verify.
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
python_bin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
tag=independent_actionability_algebra_p8_final_v1_20260913
protocol_sha=6bf96b1d30cb2d9e55126fdbc2ecca1a62a25d13d1123c2066a614d2455d6fb4
request="$root/artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/manifests/feature_request_manifest_v2.jsonl"
feature_root="$root/features/$tag"
model=${MODEL:?MODEL is required}
gpu_count=${GPUS_PER_NODE:?GPUS_PER_NODE is required}
execution="$root/results/$tag/execution/full_${model}_v1"
receipt="$execution/coverage_receipt.json"

case "$model" in dino|f3r|vjepa2|wan) ;; *) echo "unsupported MODEL=$model" >&2; exit 2 ;; esac
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq "$gpu_count"
test ! -e "$receipt"
mkdir -p "$execution"

pids=()
for ((shard=0; shard<gpu_count; shard++)); do
  (
    export CUDA_VISIBLE_DEVICES="$shard"
    export MODEL="$model" SCOPE=final NUM_SHARDS="$gpu_count" SHARD_INDEX="$shard"
    if [[ "$model" == vjepa2 ]]; then
      export VJEPA2_REVISION=b3c1679b7c34d3255ef3547f27c7b226aefab26f
    fi
    bash "$root/scripts/run_independent_p8_final_feature_job.sh"
  ) >"$execution/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more $model shards failed; preserving exact completed units for resume" >&2
  exit 1
fi

"$python_bin" "$root/scripts/verify_independent_p8_final_features.py" \
  --science-tag "$tag" --protocol-sha256 "$protocol_sha" \
  --request-manifest "$request" --feature-root "$feature_root" \
  --model "$model" --artifact-output "$receipt"
