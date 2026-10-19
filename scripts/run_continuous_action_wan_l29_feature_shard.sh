#!/usr/bin/env bash
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
python_bin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
tag=independent_goal_conditioned_continuous_action_p8_relational_v2_20260923
science_tag=independent_actionability_algebra_p8_continuous_relational_v2_20260923
protocol_sha=005c9adfe2686b29deec8c21ef555a6c22108441fc3e33779e94cd3b583ea0d2
request_sha=fdd80a9b84f9050c2a4c10e27d94dece74f1f6fde24471231e243f0082713818
extractor_sha=c10892ceaeeb31e7626388018ec123706bf7587ac85624cfbe239af82f29e486
request="$root/results/$tag/wan_l29_feature_request.jsonl"
output="$root/features/$tag"
data_root=/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2
model_id=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
revision=0fad780a534b6463e45facd96134c9f345acfa5b
model_source=/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b
num_shards=${NUM_SHARDS:-8}
shard_index=${SHARD_INDEX:?SHARD_INDEX is required}

test -x "$python_bin"
test -d "$model_source"
test "$(sha256sum "$root/results/$tag/CONTINUOUS_ACTION_PROTOCOL.md" | cut -d' ' -f1)" = "$protocol_sha"
test "$(sha256sum "$request" | cut -d' ' -f1)" = "$request_sha"
test "$(sha256sum "$root/scripts/extract_continuous_action_wan_l29.py" | cut -d' ' -f1)" = "$extractor_sha"
test "$num_shards" -eq 8
test "$shard_index" -ge 0
test "$shard_index" -lt "$num_shards"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WAN_T2V_DTYPE=bf16
nvidia-smi

"$python_bin" "$root/scripts/extract_continuous_action_wan_l29.py" \
  --science-tag "$science_tag" \
  --protocol-sha256 "$protocol_sha" \
  --request-manifest "$request" \
  --data-root "$data_root" \
  --output-root "$output" \
  --model wan \
  --model-id "$model_id" \
  --model-revision "$revision" \
  --model-source "$model_source" \
  --device cuda \
  --num-shards "$num_shards" \
  --shard-index "$shard_index"

chmod -R a+rX "$output/wan"
