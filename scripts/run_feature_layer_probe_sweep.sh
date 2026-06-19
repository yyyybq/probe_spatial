#!/usr/bin/env bash
# End-to-end layer sweep:
#   1) extract feature caches for many layers,
#   2) train one probe per layer,
#   3) evaluate each checkpoint,
#   4) summarize best/default/last layer scores.
#
# Examples:
#   DRY_RUN=1 VFM=wan PROBE=view_consistency LAYERS="0 5 10 15 20 25 29" \
#     bash scripts/run_feature_layer_probe_sweep.sh
#
#   VFM=vjepa2 PROBE=action_dynamics LAYERS="0 5 11 17 23" DEV=0 \
#     EXTRA_TRAIN="trainer.max_epochs=10 logger.wandb.offline=true" \
#     bash scripts/run_feature_layer_probe_sweep.sh
#
#   CFG=inscene15k_ext/sae_qwen2_5vl_v1 VFM=qwen2_5_vl PROBE=sae \
#     SUMMARY_PROBE=sae_spatial \
#     LAYERS="-1 8 16 24" bash scripts/run_feature_layer_probe_sweep.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
VFM=${VFM:-wan}
PROBE=${PROBE:-view_consistency}
LAYERS=${LAYERS:-}
T=${T:-749}
DEV=${DEV:-0}
SPLIT=${SPLIT:-val}
DRY_RUN=${DRY_RUN:-0}

DATA_ROOT=${DATA_ROOT:-${INSCENE_DATA_ROOT:?set DATA_ROOT or INSCENE_DATA_ROOT}}
case "${VFM}" in
    qwen2_5_vl*|bagel) DEFAULT_FEAT_ROOT=${INSCENE_MLLM_FEAT_ROOT:?set INSCENE_MLLM_FEAT_ROOT} ;;
    *) DEFAULT_FEAT_ROOT=${INSCENE_FEAT_ROOT:?set INSCENE_FEAT_ROOT} ;;
esac
FEAT_ROOT=${FEAT_ROOT:-${DEFAULT_FEAT_ROOT}}
SHUFFLED_FEAT_ROOT=${SHUFFLED_FEAT_ROOT:-${INSCENE_SHUFFLED_FEAT_ROOT:?set INSCENE_SHUFFLED_FEAT_ROOT}}
TARGET_FEAT_ROOT=${TARGET_FEAT_ROOT:-${INSCENE_TARGET_FEAT_ROOT:?set INSCENE_TARGET_FEAT_ROOT}}
LOGS_ROOT=${LOGS_ROOT:-logs/inscene15k_ext/runs}

EXTRACT=${EXTRACT:-1}
TRAIN=${TRAIN:-1}
EVAL=${EVAL:-1}
SUMMARIZE=${SUMMARIZE:-1}

EXTRA_EXTRACT=${EXTRA_EXTRACT:-}
EXTRA_TRAIN=${EXTRA_TRAIN:-}
EXTRA_EVAL=${EXTRA_EVAL:-}
SUMMARY_PROBE=${SUMMARY_PROBE:-${PROBE}}
MLLM=${MLLM:-auto}
BACKEND=${BACKEND:-}
MODEL_ID=${MODEL_ID:-}
VFM_NAME=${VFM_NAME:-${VFM}}
NUM_FRAMES=${NUM_FRAMES:-}
RESIZE=${RESIZE:-}

export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

cfg="${CFG:-inscene15k_ext/${PROBE}_${VFM}_v1}"
cfg_file="configs/experiment/${cfg}.yaml"
if [[ ! -f "${cfg_file}" ]]; then
    echo "[err] missing config: ${cfg_file}" >&2
    echo "      pass CFG=inscene15k_ext/<config_name> for nonstandard names such as sae_qwen2_5vl_v1" >&2
    exit 1
fi

if [[ "${MLLM}" == "auto" ]]; then
    case "${VFM}" in
        qwen2_5_vl*|bagel) MLLM=1 ;;
        *) MLLM=0 ;;
    esac
fi
if [[ "${MLLM}" == "1" && -z "${BACKEND}" ]]; then
    case "${VFM}" in
        qwen2_5_vl*) BACKEND=qwen2_5_vl ;;
        bagel) BACKEND=bagel_hf ;;
        *) BACKEND=generic_hf ;;
    esac
fi

if [[ -z "${LAYERS}" ]]; then
    LAYERS=$("${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --format list 2>/dev/null \
        || "${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --field default_layer)
fi

resolve_layer() {
    local layer="$1"
    if [[ "${layer}" == "default" ]]; then
        "${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --field default_layer
    elif [[ "${layer}" == "last" ]]; then
        "${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --field last_layer
    else
        printf '%s\n' "${layer}"
    fi
}

run_cmd() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

split_extra() {
    local src="$1"
    local -n dst="$2"
    dst=()
    if [[ -n "${src}" ]]; then
        read -r -a dst <<< "${src}"
    fi
}

needs_mode() {
    local mode="$1"
    case "${mode}" in
        normal)
            return 0
            ;;
        shuffled)
            [[ "${PROBE}" == "abnormal" ]]
            ;;
        target_isolated)
            [[ "${PROBE}" == "action_dynamics" || "${PROBE}" == "path_integration" || "${PROBE}" == "counterfactual" ]]
            ;;
        *)
            return 1
            ;;
    esac
}

layers_resolved=()
declare -A seen_layers=()
for layer_raw in ${LAYERS}; do
    layer="$(resolve_layer "${layer_raw}")"
    if [[ -n "${seen_layers[${layer}]:-}" ]]; then
        continue
    fi
    seen_layers["${layer}"]=1
    layers_resolved+=("${layer}")
done
layers_joined="${layers_resolved[*]}"

split_extra "${EXTRA_EXTRACT}" extra_extract_args
split_extra "${EXTRA_TRAIN}" extra_train_args
split_extra "${EXTRA_EVAL}" extra_eval_args

if [[ "${EXTRACT}" == "1" ]]; then
    for mode in normal shuffled target_isolated; do
        if ! needs_mode "${mode}"; then
            continue
        fi
        if [[ "${MLLM}" == "1" && "${mode}" != "normal" ]]; then
            echo "[warn] MLLM extractor currently supports normal caches only; skip mode=${mode}" >&2
            continue
        fi
        case "${mode}" in
            normal) out_root="${FEAT_ROOT}" ;;
            shuffled) out_root="${SHUFFLED_FEAT_ROOT}" ;;
            target_isolated) out_root="${TARGET_FEAT_ROOT}" ;;
        esac
        echo "==== extract ${VFM} mode=${mode} layers=${layers_joined} ===="
        if [[ "${MLLM}" == "1" ]]; then
            cmd=("${PYTHON}" -m features.run_inscene15k_mllm
                --backend "${BACKEND}"
                --vfm-name "${VFM_NAME}"
                --data-root "${DATA_ROOT}"
                --out-root "${out_root}"
                --output-layers "${layers_resolved[@]}"
            )
            if [[ -n "${MODEL_ID}" ]]; then
                cmd+=(--model-id "${MODEL_ID}")
            fi
            if [[ -n "${NUM_FRAMES}" ]]; then
                cmd+=(--num-frames "${NUM_FRAMES}")
            fi
            if [[ -n "${RESIZE}" ]]; then
                read -r -a resize_args <<< "${RESIZE}"
                cmd+=(--resize "${resize_args[@]}")
            fi
        else
            cmd=("${PYTHON}" -m features.run_inscene15k
                --vfm "${VFM}"
                --mode "${mode}"
                --data-root "${DATA_ROOT}"
                --out-root "${out_root}"
                --t "${T}"
                --output-layers "${layers_resolved[@]}"
            )
        fi
        cmd+=("${extra_extract_args[@]}")
        run_cmd "${cmd[@]}"
    done
fi

evaluated_runs=()
for layer in "${layers_resolved[@]}"; do
    job="${PROBE}_${VFM}_layer${layer}"
    run_name="inscene15k_ext_${job}"
    run_dir="${LOGS_ROOT}/${run_name}"
    ckpt="${run_dir}/checkpoints/last.ckpt"

    if [[ "${TRAIN}" == "1" ]]; then
        echo "==== train ${cfg} layer=${layer} ===="
        cmd=(env CUDA_VISIBLE_DEVICES="${DEV}" "${PYTHON}" vidfm3d/train.py
            "experiment=${cfg}"
            "feature_layer=${layer}"
            "feature_timestep=${T}"
            "job_name=${job}"
            "paths.run_folder_name=${run_name}"
            "logger.wandb.name=${run_name}"
        )
        cmd+=("${extra_train_args[@]}")
        run_cmd "${cmd[@]}"
    fi

    if [[ "${EVAL}" == "1" ]]; then
        if [[ "${DRY_RUN}" != "1" && ! -f "${ckpt}" ]]; then
            echo "[warn] missing checkpoint, skip eval: ${ckpt}" >&2
            continue
        fi
        echo "==== eval ${cfg} layer=${layer} ===="
        cmd=("${PYTHON}" vidfm3d/eval_diag.py
            "experiment=${cfg}"
            "feature_layer=${layer}"
            "feature_timestep=${T}"
            "job_name=${job}"
            "paths.run_folder_name=${run_name}"
            "ckpt_path=${ckpt}"
            "+eval_split=${SPLIT}"
            "train=false"
            "test=false"
        )
        cmd+=("${extra_eval_args[@]}")
        run_cmd "${cmd[@]}"
        evaluated_runs+=("${run_dir}")
    fi
done

if [[ "${SUMMARIZE}" == "1" ]]; then
    last_layer=$("${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --field last_layer 2>/dev/null || true)
    out_csv="layer_sweep_${PROBE}_${VFM}_${SPLIT}.csv"
    echo "==== summarize ${PROBE} ${VFM} ===="
    cmd=("${PYTHON}" scripts/summarize_layer_sweep.py
        --runs-root "${LOGS_ROOT}"
        --pattern "inscene15k_ext_${PROBE}_${VFM}_layer*"
        --split "${SPLIT}"
        --vfm "${VFM}"
        --probe "${SUMMARY_PROBE}"
        --output "${out_csv}"
    )
    run_cmd "${cmd[@]}"
    if [[ -n "${last_layer}" ]]; then
        echo "[info] registered last layer for ${VFM}: ${last_layer}"
    fi
fi
