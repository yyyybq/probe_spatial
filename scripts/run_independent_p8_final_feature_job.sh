#!/usr/bin/env bash
# Parameterized SCO worker for frozen independent-P8 pooled feature units.
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
python_bin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
tag=independent_actionability_algebra_p8_final_v1_20260913
protocol_sha=6bf96b1d30cb2d9e55126fdbc2ecca1a62a25d13d1123c2066a614d2455d6fb4
request="$root/artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/manifests/feature_request_manifest_v2.jsonl"
data_root=/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2
model=${MODEL:?MODEL is required}
scope=${SCOPE:-smoke}
num_shards=${NUM_SHARDS:-1}
shard_index=${SHARD_INDEX:-0}

test -x "$python_bin"
test -f "$request"
test -f "$root/results/$tag/FINAL_PROTOCOL.md"

case "$model" in
  dino)
    model_id=facebook/dinov2-large
    revision=47b73eefe95e8d44ec3623f8890bd894b6ea2d6c
    model_source="$root/model_cache/models--facebook--dinov2-large/snapshots/$revision"
    ;;
  f3r)
    model_id=jedyang97/Fast3R_ViT_Large_512
    revision=a2c770b768ceb3a53c36c4f7a3619db0413dc3a1
    model_source=/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--jedyang97--Fast3R_ViT_Large_512/snapshots/a2c770b768ceb3a53c36c4f7a3619db0413dc3a1
    ;;
  vjepa2)
    model_id=facebook/vjepa2-vitl-fpc64-256
    revision=${VJEPA2_REVISION:?VJEPA2_REVISION is required}
    model_source="$root/model_cache/models--facebook--vjepa2-vitl-fpc64-256/snapshots/$revision"
    ;;
  wan)
    model_id=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    revision=0fad780a534b6463e45facd96134c9f345acfa5b
    model_source=/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b
    export WAN_T2V_DTYPE=bf16
    ;;
  *)
    echo "unsupported MODEL=$model" >&2
    exit 2
    ;;
esac

test -d "$model_source"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi

common=(
  --science-tag "$tag"
  --protocol-sha256 "$protocol_sha"
  --request-manifest "$request"
  --data-root "$data_root"
  --model "$model"
  --model-id "$model_id"
  --model-revision "$revision"
  --model-source "$model_source"
  --device cuda
)

if [[ "$scope" == "smoke" ]]; then
  output="$root/features/${tag}_smoke_v1"
  receipt="$root/results/$tag/execution/smoke_${model}_v1/coverage_receipt.json"
  "$python_bin" "$root/scripts/extract_independent_p8_final_features.py" \
    "${common[@]}" --output-root "$output" --smoke-balanced
  "$python_bin" "$root/scripts/verify_independent_p8_final_features.py" \
    --science-tag "$tag" \
    --protocol-sha256 "$protocol_sha" \
    --request-manifest "$request" \
    --feature-root "$output" \
    --model "$model" \
    --smoke-balanced \
    --artifact-output "$receipt"
elif [[ "$scope" == "final" ]]; then
  output="$root/features/$tag"
  extra=()
  if [[ "$model" == "dino" ]]; then
    extra+=(--reuse-dino-root "$root/features/independent_actionability_algebra_p8/independent_actionability_algebra_p8_20260831T091224Z_28bda217/dino_l0_science_pilot_v1")
  fi
  "$python_bin" "$root/scripts/extract_independent_p8_final_features.py" \
    "${common[@]}" --output-root "$output" \
    --num-shards "$num_shards" --shard-index "$shard_index" "${extra[@]}"
else
  echo "unsupported SCOPE=$scope" >&2
  exit 2
fi
