#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
FEATURE_AUDIT=${FEATURE_AUDIT:?FEATURE_AUDIT is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
SCENE_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/seed0_d1_d3ab
PROVENANCE=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher

test ! -e "${OUTPUT_ROOT}" || { echo "refusing existing diagnostic output ${OUTPUT_ROOT}" >&2; exit 2; }
mkdir -p "${REPORT}" "${OUTPUT_ROOT}"
date -u +%FT%TZ > "${REPORT}/started_utc.txt"
nvidia-smi > "${REPORT}/nvidia_smi.txt"
for name in base_commit.txt git_dirty.txt working_tree.patch source_snapshot.tar.gz source_hashes.sha256; do
  test -f "${PROVENANCE}/${name}"; cp "${PROVENANCE}/${name}" "${REPORT}/${name}"
done
export D_SERIES_BASE_COMMIT=${D_SERIES_BASE_COMMIT:-$(cat "${REPORT}/base_commit.txt")}
export D_SERIES_GIT_DIRTY=${D_SERIES_GIT_DIRTY:-$(cat "${REPORT}/git_dirty.txt")}
FROZEN_ROOT=$(mktemp -d "/tmp/dseries_${RUN_TAG}_source.XXXXXX")
tar -xzf "${REPORT}/source_snapshot.tar.gz" -C "${FROZEN_ROOT}"
printf '%s\n' "${FROZEN_ROOT}" > "${REPORT}/executed_source_root.txt"
sha256sum "${REPORT}/source_snapshot.tar.gz" > "${REPORT}/executed_source_snapshot.sha256"
cd "${FROZEN_ROOT}"

pids=(); names=()
launch_d1() {
  local gpu=$1 reference=$2 branch=$3 name=$4
  local extra=()
  if [[ "${reference}" == isolated_target && "${branch}" == pooled ]]; then
    extra+=(--v2-mapping experiments/d_series_scannetpp_v1/v2_reuse_mapping.csv)
  fi
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" scripts/d_series/run_d1.py \
    --model vjepa2_vitl256 --reference "${reference}" --branch "${branch}" --seed 0 --depth-only \
    --scene-root "${SCENE_ROOT}" --feature-root "${FEATURE_ROOT}" --feature-audit-receipt "${FEATURE_AUDIT}" \
    --output-dir "${OUTPUT_ROOT}/${name}" "${extra[@]}" > "${REPORT}/${name}.log" 2>&1 &
  pids+=("$!"); names+=("${name}")
}
launch_d1 0 isolated_target spatial d1_isolated_spatial_s0
launch_d1 1 isolated_target pooled d1_isolated_pooled_s0
launch_d1 2 native_target spatial d1_native_spatial_s0
launch_d1 3 native_target pooled d1_native_pooled_s0
set +e; d1_failed=0
for pid in "${pids[@]}"; do wait "${pid}" || d1_failed=$((d1_failed + 1)); done
set -e
if (( d1_failed > 0 )); then
  printf '{"stage":"D1","failed":%d,"ended_utc":"%s"}\n' "${d1_failed}" "$(date -u +%FT%TZ)" > "${REPORT}/status.json"
  exit 1
fi

pids=()
for spec in "0 spatial" "1 pooled"; do
  read -r gpu branch <<<"${spec}"
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" scripts/d_series/run_d3_ab.py --branch "${branch}" --seed 0 \
    --scene-root "${SCENE_ROOT}" --feature-root "${FEATURE_ROOT}" --feature-audit-receipt "${FEATURE_AUDIT}" \
    --native-d1-checkpoint "${OUTPUT_ROOT}/d1_native_${branch}_s0/checkpoint.pt" \
    --output-dir "${OUTPUT_ROOT}/d3_ab_${branch}_s0" > "${REPORT}/d3_ab_${branch}_s0.log" 2>&1 &
  pids+=("$!")
done
set +e; d3_failed=0
for pid in "${pids[@]}"; do wait "${pid}" || d3_failed=$((d3_failed + 1)); done
set -e
printf '{"stage":"seed0_D1_D3_AB","d1_failed":0,"d3_failed":%d,"ended_utc":"%s"}\n' \
  "${d3_failed}" "$(date -u +%FT%TZ)" > "${REPORT}/status.json"
(( d3_failed == 0 ))
