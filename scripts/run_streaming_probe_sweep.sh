#!/usr/bin/env bash
# Shared streaming-prefix sweep for A1/A2/B1/B2/C1/C2/C3.
#
# The streaming cache is probe-agnostic: each selected prefix [I_0..I_t] is
# forwarded once and stored under prefix_<tail>. Training then sweeps one fixed
# prefix length per job, which avoids variable-length batches while preserving
# the online-history protocol.
#
# Examples:
#   DRY_RUN=1 PROBES="ego_belief ego_belief_v2" PREFIX_LENGTHS="4 8 16 32 64" \
#     bash scripts/run_streaming_probe_sweep.sh
#
#   PYTHON=/data/baiqiao/miniconda3/envs/vidfm3d/bin/python DEV=0 \
#   PROBES="view_consistency ego_belief action_dynamics path_integration counterfactual" \
#   LAYERS="default" EXTRA_TRAIN="trainer.max_epochs=20 logger.wandb.offline=true" \
#     bash scripts/run_streaming_probe_sweep.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
VFM=${VFM:-wan}
T=${T:-749}
DEV=${DEV:-0}
DRY_RUN=${DRY_RUN:-0}
SPLIT=${SPLIT:-val}

PROBES=${PROBES:-"streaming_depth view_consistency ego_belief ego_belief_v2 action_dynamics path_integration counterfactual"}
PREFIX_LENGTHS=${PREFIX_LENGTHS:-"4 8 16 32 64"}
LAYERS=${LAYERS:-default}

DATA_ROOT=${DATA_ROOT:-${INSCENE_DATA_ROOT:?set DATA_ROOT or INSCENE_DATA_ROOT}}
STREAM_ROOT_BASE=${STREAM_ROOT_BASE:-${INSCENE_STREAMING_FEAT_ROOT:?set STREAM_ROOT_BASE or INSCENE_STREAMING_FEAT_ROOT}}
TARGET_FEAT_ROOT=${TARGET_FEAT_ROOT:-${INSCENE_TARGET_FEAT_ROOT:-}}
LOGS_ROOT=${LOGS_ROOT:-logs/inscene15k_streaming/runs}

EXTRACT_STREAMING=${EXTRACT_STREAMING:-1}
EXTRACT_TARGETS=${EXTRACT_TARGETS:-1}
TRAIN=${TRAIN:-1}
EVAL=${EVAL:-1}

EXTRA_EXTRACT=${EXTRA_EXTRACT:-}
EXTRA_TARGET_EXTRACT=${EXTRA_TARGET_EXTRACT:-}
EXTRA_TRAIN=${EXTRA_TRAIN:-}
EXTRA_EVAL=${EXTRA_EVAL:-}

export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

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

resolve_layer_token() {
    local token="$1"
    case "${token}" in
        default)
            "${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --field default_layer
            ;;
        last)
            "${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --field last_layer
            ;;
        all)
            "${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --format list | tr ' ' '\n'
            ;;
        *)
            printf '%s\n' "${token}"
            ;;
    esac
}

safe_name() {
    local value="$1"
    value="${value// /_}"
    value="${value//,/}"
    printf '%s' "${value}"
}

probe_cfg() {
    local probe="$1"
    printf 'inscene15k_streaming/%s_%s_v1' "${probe}" "${VFM}"
}

probe_needs_target_cache() {
    local probe="$1"
    [[ "${probe}" == "action_dynamics" || "${probe}" == "path_integration" || "${probe}" == "counterfactual" ]]
}

probe_uses_shared_hidden_object() {
    local probe="$1"
    [[ "${probe}" == "ego_belief" || "${probe}" == "ego_belief_v2" ]]
}

layers_resolved=()
declare -A seen_layers=()
for token in ${LAYERS}; do
    while read -r layer; do
        [[ -z "${layer}" ]] && continue
        if [[ -n "${seen_layers[${layer}]:-}" ]]; then
            continue
        fi
        seen_layers["${layer}"]=1
        layers_resolved+=("${layer}")
    done < <(resolve_layer_token "${token}")
done
if (( ${#layers_resolved[@]} == 0 )); then
    echo "[err] no layers resolved from LAYERS='${LAYERS}'" >&2
    exit 1
fi

video_channels=$("${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --field in_channels 2>/dev/null || true)
if [[ -z "${video_channels}" ]]; then
    case "${VFM}" in
        cogvideox) video_channels=3072 ;;
        vjepa2|vjepa) video_channels=1024 ;;
        *) video_channels=1536 ;;
    esac
fi

max_prefix=0
for prefix_len in ${PREFIX_LENGTHS}; do
    if ! [[ "${prefix_len}" =~ ^[0-9]+$ ]]; then
        echo "[err] bad PREFIX_LENGTHS value: ${prefix_len}" >&2
        exit 1
    fi
    (( prefix_len > max_prefix )) && max_prefix=${prefix_len}
done
if (( max_prefix < 1 )); then
    echo "[err] PREFIX_LENGTHS must contain at least one positive integer" >&2
    exit 1
fi

prefix_name="prefix_$(safe_name "${PREFIX_LENGTHS}")"
stream_root="${STREAM_ROOT_BASE}/${VFM}_${prefix_name}"
hidden_prefix_list="[${PREFIX_LENGTHS// /,}]"

split_extra "${EXTRA_EXTRACT}" extra_extract_args
split_extra "${EXTRA_TARGET_EXTRACT}" extra_target_extract_args
split_extra "${EXTRA_TRAIN}" extra_train_args
split_extra "${EXTRA_EVAL}" extra_eval_args

echo "[info] vfm=${VFM} probes=${PROBES}"
echo "[info] prefix_lengths=${PREFIX_LENGTHS} layers=${layers_resolved[*]}"
echo "[info] streaming_feat_root=${stream_root}"

if [[ "${EXTRACT_STREAMING}" == "1" ]]; then
    echo "==== extract shared streaming-prefix cache ===="
    cmd=("${PYTHON}" -m features.run_inscene15k
        --vfm "${VFM}"
        --mode streaming_prefix
        --data-root "${DATA_ROOT}"
        --out-root "${stream_root}"
        --t "${T}"
        --output-layers "${layers_resolved[@]}"
        --prefix-min-len 1
        --prefix-max-len "${max_prefix}"
        --prefix-lengths "${PREFIX_LENGTHS}"
    )
    cmd+=("${extra_extract_args[@]}")
    run_cmd "${cmd[@]}"
fi

needs_targets=0
for probe in ${PROBES}; do
    if probe_needs_target_cache "${probe}"; then
        needs_targets=1
    fi
done
if [[ "${needs_targets}" == "1" && "${EXTRACT_TARGETS}" == "1" ]]; then
    if [[ -z "${TARGET_FEAT_ROOT}" ]]; then
        echo "[err] C probes need TARGET_FEAT_ROOT or INSCENE_TARGET_FEAT_ROOT" >&2
        exit 1
    fi
    echo "==== extract target-isolated cache for C probes ===="
    cmd=("${PYTHON}" -m features.run_inscene15k
        --vfm "${VFM}"
        --mode target_isolated
        --data-root "${DATA_ROOT}"
        --out-root "${TARGET_FEAT_ROOT}"
        --t "${T}"
        --output-layers "${layers_resolved[@]}"
        --num-targets 0
    )
    cmd+=("${extra_target_extract_args[@]}")
    run_cmd "${cmd[@]}"
fi

for probe in ${PROBES}; do
    cfg="$(probe_cfg "${probe}")"
    cfg_file="configs/experiment/${cfg}.yaml"
    if [[ ! -f "${cfg_file}" ]]; then
        echo "[err] missing config for probe=${probe}: ${cfg_file}" >&2
        exit 1
    fi

    for prefix_len in ${PREFIX_LENGTHS}; do
        for layer in "${layers_resolved[@]}"; do
            layer_name="$(safe_name "${layer}")"
            job="${probe}_${VFM}_p${prefix_len}_layer${layer_name}"
            run_name="inscene15k_streaming_${job}"
            run_dir="${LOGS_ROOT}/${run_name}"
            ckpt="${run_dir}/checkpoints/last.ckpt"

            if [[ "${TRAIN}" == "1" ]]; then
                echo "==== train ${probe} prefix=${prefix_len} layer=${layer} ===="
                cmd=(env CUDA_VISIBLE_DEVICES="${DEV}" "${PYTHON}" vidfm3d/train.py
                    "experiment=${cfg}"
                    "vfm_name=${VFM}"
                    "video_channels=${video_channels}"
                    "streaming_feat_root=${stream_root}"
                    "feature_layer=${layer}"
                    "feature_timestep=${T}"
                    "prefix_len=${prefix_len}"
                    "job_name=${job}"
                    "paths.run_folder_name=${run_name}"
                    "logger.wandb.name=${run_name}"
                )
                if probe_uses_shared_hidden_object "${probe}"; then
                    cmd+=("streaming_hidden_prefix_lengths=${hidden_prefix_list}")
                fi
                if probe_needs_target_cache "${probe}"; then
                    cmd+=("target_feat_root=${TARGET_FEAT_ROOT}")
                fi
                cmd+=("${extra_train_args[@]}")
                run_cmd "${cmd[@]}"
            fi

            if [[ "${EVAL}" == "1" && "${probe}" != "streaming_depth" ]]; then
                if [[ "${DRY_RUN}" != "1" && ! -f "${ckpt}" ]]; then
                    echo "[warn] missing checkpoint, skip eval: ${ckpt}" >&2
                    continue
                fi
                echo "==== eval ${probe} prefix=${prefix_len} layer=${layer} ===="
                cmd=("${PYTHON}" vidfm3d/eval_diag.py
                    "experiment=${cfg}"
                    "vfm_name=${VFM}"
                    "video_channels=${video_channels}"
                    "streaming_feat_root=${stream_root}"
                    "feature_layer=${layer}"
                    "feature_timestep=${T}"
                    "prefix_len=${prefix_len}"
                    "job_name=${job}"
                    "paths.run_folder_name=${run_name}"
                    "ckpt_path=${ckpt}"
                    "+eval_split=${SPLIT}"
                    "train=false"
                    "test=false"
                )
                if probe_uses_shared_hidden_object "${probe}"; then
                    cmd+=("streaming_hidden_prefix_lengths=${hidden_prefix_list}")
                fi
                if probe_needs_target_cache "${probe}"; then
                    cmd+=("target_feat_root=${TARGET_FEAT_ROOT}")
                fi
                cmd+=("${extra_eval_args[@]}")
                run_cmd "${cmd[@]}"
            fi
        done
    done
done
