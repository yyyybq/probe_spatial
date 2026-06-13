#!/usr/bin/env bash
# Run eval_diag.py for every inscene15k_ext run that has a last.ckpt and a matching config.
#   bash scripts/run_diag_eval_sweep.sh
set -euo pipefail

LOGS_ROOT=${LOGS_ROOT:-logs}
SPLIT=${SPLIT:-val}

mapfile -t ckpts < <(
    find "${LOGS_ROOT}" -path '*/inscene15k_ext_*/checkpoints/last.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn \
        | cut -d' ' -f2-
)

if (( ${#ckpts[@]} == 0 )); then
    echo "[skip] no inscene15k_ext last.ckpt files found under ${LOGS_ROOT}"
    exit 0
fi

declare -A seen_runs=()
for ckpt in "${ckpts[@]}"; do
    run_dir=$(dirname "$(dirname "${ckpt}")")
    run=$(basename "${run_dir}")
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

    echo "==== eval ${run} ===="
    python vidfm3d/eval_diag.py \
        experiment="inscene15k_ext/${job}" \
        ckpt_path="${ckpt}" \
        eval_split="${SPLIT}" \
        train=false test=false
done

mapfile -t run_dirs < <(find "${LOGS_ROOT}" -type d -name 'inscene15k_ext_*' 2>/dev/null | sort)
if (( ${#run_dirs[@]} == 0 )); then
    echo "[skip] no evaluated run directories found under ${LOGS_ROOT}"
    exit 0
fi

python vidfm3d/eval_diag_compare.py \
    --runs "${run_dirs[@]}" \
    --split "${SPLIT}" \
    --output "comparison_${SPLIT}.csv"
