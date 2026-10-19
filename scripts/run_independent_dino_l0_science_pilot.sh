#!/usr/bin/env bash
# Shared-pilot DINO L0 extraction, with fixed local model provenance and raw IDs.
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
tag=independent_actionability_algebra_p8_20260831T091224Z_28bda217
python_bin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
request="$root/artifacts/$tag/manifests/science_pilot_subset_v1_feature_request_v2_dino_l0.jsonl"
feature_root="$root/features/independent_actionability_algebra_p8/$tag/dino_l0_science_pilot_v1"
receipt_dir="$root/artifacts/$tag/execution/dino_l0_science_pilot_v1"
data_root=/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2
model_source=/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c

test -x "$python_bin"
test -f "$request"
test -d "$data_root"
test -f "$model_source/config.json"
test -f "$model_source/preprocessor_config.json"
test -f "$model_source/model.safetensors"
mkdir -p "$receipt_dir"
nvidia-smi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$python_bin" "$root/scripts/extract_independent_dino_features.py" \
  --science-tag "$tag" \
  --request-manifest "$request" \
  --data-root "$data_root" \
  --output-root "$feature_root" \
  --device cuda \
  --batch-size 8 \
  --model-revision 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c \
  --model-source "$model_source" \
  --local-files-only
"$python_bin" "$root/scripts/verify_independent_feature_store.py" \
  --science-tag "$tag" \
  --request-manifest "$request" \
  --feature-root "$feature_root" \
  --model dino \
  --artifact-output "$receipt_dir/coverage_receipt.json"
