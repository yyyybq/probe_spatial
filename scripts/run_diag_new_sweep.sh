#!/usr/bin/env bash
# Sweep training for newly added C2/C3 diagnostic probes.
#   DRY_RUN=1 bash scripts/run_diag_new_sweep.sh
#   bash scripts/run_diag_new_sweep.sh
#   VFMS="wan vjepa2" PROBES="path_integration" bash scripts/run_diag_new_sweep.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
DEV=${DEV:-0}
DRY_RUN=${DRY_RUN:-0}
VFMS=${VFMS:-"wan vjepa2 cogvideox"}
PROBES=${PROBES:-"path_integration counterfactual"}
EXTRA=${EXTRA:-}

export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

for probe in ${PROBES}; do
    for vfm in ${VFMS}; do
        cfg="inscene15k_ext/${probe}_${vfm}_v1"
        cfg_file="configs/experiment/${cfg}.yaml"
        if [[ ! -f "${cfg_file}" ]]; then
            echo "[skip] missing config: ${cfg_file}"
            continue
        fi
        echo "==== train ${cfg} ===="
        cmd=(env CUDA_VISIBLE_DEVICES="${DEV}" "${PYTHON}" vidfm3d/train.py "experiment=${cfg}")
        if [[ -n "${EXTRA}" ]]; then
            # EXTRA is intentionally shell-split to match the old sweep script.
            read -r -a extra_args <<< "${EXTRA}"
            cmd+=("${extra_args[@]}")
        fi
        if [[ "${DRY_RUN}" == "1" ]]; then
            printf '[dry-run]'
            printf ' %q' "${cmd[@]}"
            printf '\n'
        else
            "${cmd[@]}"
        fi
    done
done
