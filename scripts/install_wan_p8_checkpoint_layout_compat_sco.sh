#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
RUNS_ROOT=${PROJECT_ROOT}/logs/inscene15k_streaming/runs/${RUN_TAG}
REPORT_DIR=${PROJECT_ROOT}/reports/wan_p8_recovery/${RUN_TAG}
LAYERS=(0 5 10 15 25 29)

cd "${PROJECT_ROOT}"
mkdir -p "${REPORT_DIR}/provenance/training"
record="${REPORT_DIR}/provenance/training/checkpoint_layout_compat.tsv"
[[ ! -e "${record}" ]] || { echo "refusing existing record: ${record}" >&2; exit 2; }
printf 'layer\trun_dir\tcheckpoint_dir\tcompat_link\texpected_target\n' > "${record}"

for layer in "${LAYERS[@]}"; do
  name="${RUN_TAG}_canonical_clean_action_dynamics_wan_p8_layer${layer}"
  run_dir="${RUNS_ROOT}/${name}"
  [[ -d "${run_dir}" ]] || { echo "missing run dir: ${run_dir}" >&2; exit 2; }
  mapfile -t checkpoint_dirs < <(find "${run_dir}/VideoProbe3D" -mindepth 2 -maxdepth 2 -type d -name checkpoints | sort)
  [[ ${#checkpoint_dirs[@]} -eq 1 ]] || {
    echo "expected one checkpoint dir for layer ${layer}, got ${#checkpoint_dirs[@]}" >&2
    exit 2
  }
  checkpoint_dir=${checkpoint_dirs[0]}
  [[ -f "${checkpoint_dir}/epoch=0-step=306.ckpt" ]] || {
    echo "missing current-run epoch-0 checkpoint: ${checkpoint_dir}" >&2
    exit 2
  }
  compat_dir="${run_dir}/checkpoints"
  compat_link="${compat_dir}/last.ckpt"
  expected_target="${checkpoint_dir}/epoch=49-step=15300.ckpt"
  [[ ! -e "${compat_dir}" ]] || { echo "refusing existing compat dir: ${compat_dir}" >&2; exit 2; }
  mkdir "${compat_dir}"
  ln -s "${expected_target}" "${compat_link}"
  [[ "$(readlink "${compat_link}")" == "${expected_target}" ]] || {
    echo "compat link verification failed: ${compat_link}" >&2
    exit 2
  }
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${layer}" "${run_dir}" "${checkpoint_dir}" "${compat_link}" "${expected_target}" >> "${record}"
done

echo "CHECKPOINT LAYOUT COMPAT PASS"
