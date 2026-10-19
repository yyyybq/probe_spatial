#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
SEED=${SEED:?SEED is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
FEATURE_AUDIT=${FEATURE_AUDIT:?FEATURE_AUDIT is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
SCENE_ROOT=${SCENE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2}
MANIFEST_SHA256=781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/seed${SEED}_serial
PROVENANCE=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher

[[ "${SEED}" == 0 || "${SEED}" == 1 || "${SEED}" == 2 ]] || { echo "SEED must be 0, 1, or 2" >&2; exit 2; }
mkdir -p "${REPORT}" "${OUTPUT_ROOT}"
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

FROZEN_ROOT=$(mktemp -d "/tmp/dseries_${RUN_TAG}_s${SEED}_source.XXXXXX")
cleanup() { rm -rf -- "${FROZEN_ROOT}"; }
trap cleanup EXIT
tar -xzf "${REPORT}/source_snapshot.tar.gz" -C "${FROZEN_ROOT}"
printf '%s\n' "${FROZEN_ROOT}" > "${REPORT}/executed_source_root.txt"
sha256sum "${REPORT}/source_snapshot.tar.gz" > "${REPORT}/executed_source_snapshot.sha256"
cd "${FROZEN_ROOT}"
export PYTHONPATH="${FROZEN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PY}" - "${FEATURE_AUDIT}" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert audit["status"] == "COMPLETE_VALID", audit
assert audit["manifest_sha256"] == "781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9", audit
assert audit["model"] == "wan21_t2v_1p3b", audit
assert audit["roles"] == {"prefix": {"expected": 2721, "complete": 2721},
                           "isolated_target": {"expected": 5442, "complete": 5442}}, audit
PY

write_status() {
  local stage=$1 failed=$2
  "${PY}" - "${REPORT}/status.json" "${stage}" "${SEED}" "${failed}" <<'PY'
import json, pathlib, sys
path, stage, seed, failed = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({
    "stage": stage, "seed": int(seed), "failed": int(failed),
    "ended_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}, indent=2, sort_keys=True) + "\n")
PY
}

stage_state() {
  local output_dir=$1 experiment=$2 branch=$3 expected_rows=$4
  if [[ ! -e "${output_dir}" ]]; then return 1; fi
  if ! "${PY}" - "${output_dir}" "${experiment}" "${branch}" "${SEED}" "${expected_rows}" "${MANIFEST_SHA256}" <<'PY'
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
experiment, branch, seed, expected_rows, manifest = sys.argv[2:]
seed, expected_rows = int(seed), int(expected_rows)
required = [root / "checkpoint.pt", root / "per_sample_metrics.csv", root / "run_receipt.json"]
assert root.is_dir() and all(p.is_file() and p.stat().st_size > 0 for p in required), required
receipt = json.loads(required[2].read_text())
assert receipt["status"] == "complete_development_evaluation", receipt
assert receipt["experiment"] == experiment and receipt["model"] == "wan21_t2v_1p3b", receipt
assert receipt["branch"] == branch and receipt["seed"] == seed, receipt
assert receipt["manifest_sha256"] == manifest and receipt["n_dev"] == 552, receipt
assert receipt["n_dev_scenes"] == 69 and receipt["test_scenes"] == 0, receipt
if experiment == "D1":
    assert receipt["reference"] == "isolated_target" and receipt["depth_only"] is True, receipt
else:
    assert (root.parent / f"d1_wan_pooled_s{seed}" / "checkpoint.pt").is_file(), receipt
with required[1].open(newline="") as handle: rows = list(csv.DictReader(handle))
assert len(rows) == expected_rows, (root, len(rows), expected_rows)
assert {int(row["seed"]) for row in rows} == {seed}, root
print(f"resume-valid: {root}")
PY
  then return 2; fi
}

run_stage() {
  local name=$1 experiment=$2 branch=$3 expected_rows=$4
  shift 4
  local output_dir="${OUTPUT_ROOT}/${name}"
  set +e; stage_state "${output_dir}" "${experiment}" "${branch}" "${expected_rows}"; local state=$?; set -e
  if (( state == 0 )); then
    printf '%s\tSKIPPED_COMPLETE\t%s\n' "$(date -u +%FT%TZ)" "${name}" | tee -a "${REPORT}/stage_events.tsv"
    return 0
  fi
  if (( state != 1 )); then
    printf '%s\tREFUSED_PARTIAL_OR_INVALID\t%s\n' "$(date -u +%FT%TZ)" "${name}" | tee -a "${REPORT}/stage_events.tsv" >&2
    write_status "${name}" 1; exit 1
  fi
  printf '%s\tSTARTED\t%s\n' "$(date -u +%FT%TZ)" "${name}" | tee -a "${REPORT}/stage_events.tsv"
  if ! "$@" --output-dir "${output_dir}" > "${REPORT}/${name}.log" 2>&1; then
    printf '%s\tFAILED\t%s\n' "$(date -u +%FT%TZ)" "${name}" | tee -a "${REPORT}/stage_events.tsv" >&2
    write_status "${name}" 1; exit 1
  fi
  stage_state "${output_dir}" "${experiment}" "${branch}" "${expected_rows}"
  printf '%s\tCOMPLETE\t%s\n' "$(date -u +%FT%TZ)" "${name}" | tee -a "${REPORT}/stage_events.tsv"
}

d1_common=("${PY}" scripts/d_series/run_d1.py --model wan21_t2v_1p3b --seed "${SEED}" --depth-only
  --scene-root "${SCENE_ROOT}" --feature-root "${FEATURE_ROOT}" --feature-audit-receipt "${FEATURE_AUDIT}"
  --reference isolated_target)
run_stage "d1_wan_spatial_s${SEED}" D1 spatial 4416 "${d1_common[@]}" --branch spatial
run_stage "d1_wan_pooled_s${SEED}" D1 pooled 4416 "${d1_common[@]}" --branch pooled
run_stage "d2_wan_pooled_s${SEED}" D2 pooled 3864 \
  "${PY}" scripts/d_series/run_d2.py --model wan21_t2v_1p3b --seed "${SEED}" \
  --scene-root "${SCENE_ROOT}" --feature-root "${FEATURE_ROOT}" --feature-audit-receipt "${FEATURE_AUDIT}" \
  --d1-checkpoint "${OUTPUT_ROOT}/d1_wan_pooled_s${SEED}/checkpoint.pt"

write_status "seed${SEED}_Wan_D1_D2" 0
