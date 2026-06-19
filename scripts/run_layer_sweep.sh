#!/usr/bin/env bash
# Train one diagnostic probe over multiple cached feature layers.
#
# Examples:
#   DRY_RUN=1 VFM=wan PROBE=ego_belief LAYERS="0 5 10 15 20 25 29" bash scripts/run_layer_sweep.sh
#   VFM=vjepa2 PROBE=action_dynamics LAYERS="0 5 11 17 23" bash scripts/run_layer_sweep.sh
#   VFM=cogvideox PROBE=abnormal LAYERS="last 20" bash scripts/run_layer_sweep.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
DEV=${DEV:-0}
DRY_RUN=${DRY_RUN:-0}
VFM=${VFM:-wan}
PROBE=${PROBE:-view_consistency}
T=${T:-749}
LAYERS=${LAYERS:-}
EXTRA=${EXTRA:-}

export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

cfg="inscene15k_ext/${PROBE}_${VFM}_v1"
cfg_file="configs/experiment/${cfg}.yaml"
if [[ ! -f "${cfg_file}" ]]; then
    echo "[err] missing config: ${cfg_file}" >&2
    exit 1
fi

if [[ -z "${LAYERS}" ]]; then
    LAYERS=$("${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --format list)
fi

declare -A seen_layers=()
for layer in ${LAYERS}; do
    if [[ "${layer}" == "default" ]]; then
        layer=$("${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --field default_layer)
    elif [[ "${layer}" == "last" ]]; then
        layer=$("${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --field last_layer)
    fi
    if [[ -n "${seen_layers[${layer}]:-}" ]]; then
        continue
    fi
    seen_layers["${layer}"]=1

    if [[ "${VFM}" == "wan" || "${VFM}" == "cogvideox" ]]; then
        postfix="_t${T}_layer${layer}"
    else
        postfix="_layer${layer}"
    fi

    job="${PROBE}_${VFM}_layer${layer}"
    echo "==== train ${cfg} ${postfix} ===="
    cmd=(env CUDA_VISIBLE_DEVICES="${DEV}" "${PYTHON}" vidfm3d/train.py
        "experiment=${cfg}"
        "feat_postfix=${postfix}"
        "feature_layer=${layer}"
        "feature_timestep=${T}"
        "job_name=${job}"
        "paths.run_folder_name=inscene15k_ext_${job}"
        "logger.wandb.name=inscene15k_ext_${job}"
    )
    if [[ -n "${EXTRA}" ]]; then
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
