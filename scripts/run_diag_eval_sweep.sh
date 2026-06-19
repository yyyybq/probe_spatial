#!/usr/bin/env bash
# Run eval_diag.py for every inscene15k_ext run that has a checkpoint and a matching config.
#   bash scripts/run_diag_eval_sweep.sh
set -euo pipefail

LOGS_ROOT=${LOGS_ROOT:-logs}
SPLIT=${SPLIT:-val}
PYTHON=${PYTHON:-python}
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

mapfile -t ckpts < <(
    find "${LOGS_ROOT}" -path '*/inscene15k_ext_*/checkpoints/*.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn \
        | cut -d' ' -f2-
)

if (( ${#ckpts[@]} == 0 )); then
    echo "[skip] no inscene15k_ext checkpoint files found under ${LOGS_ROOT}"
    exit 0
fi

declare -A seen_runs=()
evaluated_run_dirs=()
for ckpt in "${ckpts[@]}"; do
    run=$(grep -o 'inscene15k_ext_[^/]*' <<< "${ckpt}" | tail -n1 || true)
    if [[ -z "${run}" ]]; then
        echo "[skip] cannot infer run name from ${ckpt}"
        continue
    fi
    if [[ -n "${seen_runs[${run}]:-}" ]]; then
        continue
    fi
    seen_runs[${run}]=1

    job=${run#inscene15k_ext_}
    cfg_file="configs/experiment/inscene15k_ext/${job}.yaml"
    if [[ ! -f "${cfg_file}" ]]; then
        echo "[skip] no config for ${run}: ${cfg_file}"
        continue
    fi

    echo "==== eval ${run} (${ckpt}) ===="
    "${PYTHON}" vidfm3d/eval_diag.py \
        experiment="inscene15k_ext/${job}" \
        "ckpt_path='${ckpt}'" \
        +eval_split="${SPLIT}" \
        train=false test=false
    evaluated_run_dirs+=("${LOGS_ROOT}/inscene15k_ext/runs/${run}")
done

if (( ${#evaluated_run_dirs[@]} == 0 )); then
    echo "[skip] no evaluated run directories"
    exit 0
fi

"${PYTHON}" vidfm3d/eval_diag_compare.py \
    --runs "${evaluated_run_dirs[@]}" \
    --split "${SPLIT}" \
    --output "comparison_${SPLIT}.csv"
