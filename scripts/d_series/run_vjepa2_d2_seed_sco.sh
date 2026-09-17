#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
SEED=${SEED:?SEED is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
FEATURE_AUDIT=${FEATURE_AUDIT:?FEATURE_AUDIT is required}
D1_CHECKPOINT=${D1_CHECKPOINT:?D1_CHECKPOINT is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
SCENE_ROOT=${SCENE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2}
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/seed${SEED}_d2
PROVENANCE=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher

[[ "${SEED}" == 0 || "${SEED}" == 1 || "${SEED}" == 2 ]] || { echo "SEED must be 0, 1, or 2" >&2; exit 2; }
mkdir -p "${REPORT}"
date -u +%FT%TZ > "${REPORT}/started_utc.txt"
nvidia-smi > "${REPORT}/nvidia_smi.txt"
"${PY}" - <<'PY'
import torch
assert torch.cuda.device_count() == 1, f"expected exactly one visible GPU, got {torch.cuda.device_count()}"
print(torch.cuda.get_device_name(0))
PY
for name in base_commit.txt git_dirty.txt working_tree.patch source_snapshot.tar.gz source_hashes.sha256; do
  test -f "${PROVENANCE}/${name}"
  cp "${PROVENANCE}/${name}" "${REPORT}/${name}"
done
export D_SERIES_BASE_COMMIT=${D_SERIES_BASE_COMMIT:-$(cat "${REPORT}/base_commit.txt")}
export D_SERIES_GIT_DIRTY=${D_SERIES_GIT_DIRTY:-$(cat "${REPORT}/git_dirty.txt")}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
FROZEN_ROOT=$(mktemp -d "/tmp/dseries_${RUN_TAG}_source.XXXXXX")
cleanup() { rm -rf -- "${FROZEN_ROOT}"; }
trap cleanup EXIT
tar -xzf "${REPORT}/source_snapshot.tar.gz" -C "${FROZEN_ROOT}"
printf '%s\n' "${FROZEN_ROOT}" > "${REPORT}/executed_source_root.txt"
sha256sum "${REPORT}/source_snapshot.tar.gz" > "${REPORT}/executed_source_snapshot.sha256"
cd "${FROZEN_ROOT}"
export PYTHONPATH="${FROZEN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

validate_d1() {
  "${PY}" - "${D1_CHECKPOINT}" "${SEED}" <<'PY'
import json, pathlib, sys
checkpoint, seed = pathlib.Path(sys.argv[1]), int(sys.argv[2])
receipt_path = checkpoint.parent / "run_receipt.json"
assert checkpoint.is_file() and checkpoint.stat().st_size > 0, checkpoint
receipt = json.loads(receipt_path.read_text())
expected = {"status": "complete_development_evaluation", "experiment": "D1", "model": "vjepa2_vitl256",
            "reference": "isolated_target", "branch": "pooled", "seed": seed,
            "manifest_sha256": "781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9"}
assert all(receipt.get(key) == value for key, value in expected.items()), receipt
PY
}

validate_output() {
  "${PY}" - "${OUTPUT_DIR}" "${SEED}" <<'PY'
import csv, json, pathlib, sys
root, seed = pathlib.Path(sys.argv[1]), int(sys.argv[2])
required = [root / "checkpoint.pt", root / "per_sample_metrics.csv", root / "run_receipt.json"]
assert all(path.is_file() and path.stat().st_size > 0 for path in required), required
receipt = json.loads(required[2].read_text())
expected = {"status": "complete_development_evaluation", "experiment": "D2", "model": "vjepa2_vitl256",
            "branch": "pooled", "seed": seed, "n_dev": 552, "n_dev_scenes": 69, "test_scenes": 0,
            "manifest_sha256": "781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9"}
assert all(receipt.get(key) == value for key, value in expected.items()), receipt
with required[1].open(newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 3864, len(rows)
assert {int(row["seed"]) for row in rows} == {seed}
assert {row["condition"] for row in rows} == {"true_motion", "identity_motion", "matched_horizon_shuffled_motion",
    "independently_trained_context_only", "last_context", "true_pooled_target_reference", "train_layout_prior"}
PY
}

validate_d1
if [[ -e "${OUTPUT_DIR}" ]]; then
  if validate_output; then
    printf '%s\tSKIPPED_COMPLETE\td2_vjepa2_pooled_s%s\n' "$(date -u +%FT%TZ)" "${SEED}" | tee "${REPORT}/stage_events.tsv"
    printf '{"stage":"seed%s_D2","seed":%s,"failed":0,"resume_skipped":true,"ended_utc":"%s"}\n' \
      "${SEED}" "${SEED}" "$(date -u +%FT%TZ)" > "${REPORT}/status.json"
    exit 0
  fi
  printf '%s\tREFUSED_PARTIAL_OR_INVALID\td2_vjepa2_pooled_s%s\n' "$(date -u +%FT%TZ)" "${SEED}" | tee "${REPORT}/stage_events.tsv" >&2
  exit 1
fi
printf '%s\tSTARTED\td2_vjepa2_pooled_s%s\n' "$(date -u +%FT%TZ)" "${SEED}" | tee "${REPORT}/stage_events.tsv"
if ! "${PY}" scripts/d_series/run_d2.py --model vjepa2_vitl256 --seed "${SEED}" \
  --scene-root "${SCENE_ROOT}" --feature-root "${FEATURE_ROOT}" --feature-audit-receipt "${FEATURE_AUDIT}" \
  --d1-checkpoint "${D1_CHECKPOINT}" --output-dir "${OUTPUT_DIR}" > "${REPORT}/d2_vjepa2_pooled_s${SEED}.log" 2>&1; then
  printf '%s\tFAILED\td2_vjepa2_pooled_s%s\n' "$(date -u +%FT%TZ)" "${SEED}" | tee -a "${REPORT}/stage_events.tsv" >&2
  printf '{"stage":"seed%s_D2","seed":%s,"failed":1,"ended_utc":"%s"}\n' \
    "${SEED}" "${SEED}" "$(date -u +%FT%TZ)" > "${REPORT}/status.json"
  exit 1
fi
validate_output
printf '%s\tCOMPLETE\td2_vjepa2_pooled_s%s\n' "$(date -u +%FT%TZ)" "${SEED}" | tee -a "${REPORT}/stage_events.tsv"
printf '{"stage":"seed%s_D2","seed":%s,"failed":0,"ended_utc":"%s"}\n' \
  "${SEED}" "${SEED}" "$(date -u +%FT%TZ)" > "${REPORT}/status.json"

