#!/usr/bin/env bash
# Legacy non-streaming sweep training for C2/C3 diagnostic probes.
#   ALLOW_NON_STREAMING=1 DRY_RUN=1 bash scripts/run_diag_new_sweep.sh
#   ALLOW_NON_STREAMING=1 bash scripts/run_diag_new_sweep.sh
#   ALLOW_NON_STREAMING=1 VFMS="wan vjepa2" PROBES="path_integration" bash scripts/run_diag_new_sweep.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
ALLOW_NON_STREAMING=${ALLOW_NON_STREAMING:-0}
DEV=${DEV:-0}
DRY_RUN=${DRY_RUN:-0}
VFMS=${VFMS:-"wan vjepa2 cogvideox"}
PROBES=${PROBES:-"path_integration counterfactual"}
EXTRA=${EXTRA:-}

if [[ "${ALLOW_NON_STREAMING}" != "1" ]]; then
    cat >&2 <<'EOF'
[err] run_diag_new_sweep.sh is a legacy non-streaming sweep.
      Streaming is the project default. Use:
        bash scripts/run_streaming_probe_sweep.sh
      To intentionally run this old path, set ALLOW_NON_STREAMING=1.
EOF
    exit 2
fi

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
