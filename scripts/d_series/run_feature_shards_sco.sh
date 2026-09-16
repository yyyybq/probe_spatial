#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
MODEL=${MODEL:?MODEL is required}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
NPROC=${NPROC:-8}
TOTAL_WINDOWS=2721
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
SCENE_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/${MODEL}
PROVENANCE=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher
mkdir -p "${REPORT}"
date -u +%FT%TZ > "${REPORT}/started_utc.txt"
nvidia-smi > "${REPORT}/nvidia_smi.txt"
for name in base_commit.txt git_dirty.txt working_tree.patch source_snapshot.tar.gz source_hashes.sha256; do
  test -f "${PROVENANCE}/${name}"
  cp "${PROVENANCE}/${name}" "${REPORT}/${name}"
done
test -s "${REPORT}/base_commit.txt"
test -s "${REPORT}/source_snapshot.tar.gz"
export D_SERIES_BASE_COMMIT=${D_SERIES_BASE_COMMIT:-$(cat "${REPORT}/base_commit.txt")}
export D_SERIES_GIT_DIRTY=${D_SERIES_GIT_DIRTY:-$(cat "${REPORT}/git_dirty.txt")}
cd "${PROJECT_ROOT}"

case "${MODEL}" in
  vjepa2_vitl256) extractor=scripts/d_series/extract_vjepa2.py ;;
  dinov2_vitl14) extractor=scripts/d_series/extract_dinov2.py ;;
  wan21_t2v_1p3b) extractor=scripts/d_series/extract_wan.py ;;
  *) echo "unsupported MODEL=${MODEL}" >&2; exit 2 ;;
esac

chunk=$(( (TOTAL_WINDOWS + NPROC - 1) / NPROC ))
pids=(); gpu=0
while (( gpu < NPROC )); do
  start=$(( gpu * chunk )); count=${chunk}
  if (( start >= TOTAL_WINDOWS )); then break; fi
  if (( start + count > TOTAL_WINDOWS )); then count=$(( TOTAL_WINDOWS - start )); fi
  if [[ "${MODEL}" == "wan21_t2v_1p3b" ]]; then
    CUDA_VISIBLE_DEVICES=${gpu} WAN_T2V_DTYPE=bf16 "${PY}" "${extractor}" --scene-root "${SCENE_ROOT}" \
      --output-root "${FEATURE_ROOT}" --window-start "${start}" --max-windows "${count}" \
      > "${REPORT}/gpu${gpu}_${start}_${count}.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${extractor}" --scene-root "${SCENE_ROOT}" \
      --output-root "${FEATURE_ROOT}" --window-start "${start}" --max-windows "${count}" \
      > "${REPORT}/gpu${gpu}_${start}_${count}.log" 2>&1 &
  fi
  pids+=("$!"); gpu=$((gpu + 1))
done

set +e; failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=$((failed + 1)); done
set -e
printf '{"model":"%s","processes":%d,"failed":%d,"ended_utc":"%s"}\n' \
  "${MODEL}" "${#pids[@]}" "${failed}" "$(date -u +%FT%TZ)" > "${REPORT}/status.json"
(( failed == 0 ))
