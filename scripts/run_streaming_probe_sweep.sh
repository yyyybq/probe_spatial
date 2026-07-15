#!/usr/bin/env bash
# Shared streaming-prefix sweep for A1/A2/B1/B2/C1/C2/C3.
#
# The streaming cache is probe-agnostic: each selected prefix [I_0..I_t] is
# forwarded once and stored under prefix_<tail>. Training then sweeps one fixed
# prefix length per job, which avoids variable-length batches while preserving
# the online-history protocol.
#
# Default model matrix:
#   wan cogvideox vjepa2 dino aether f3r qwen2_5_vl_3b bagel
#
# Examples:
#   # Full default model matrix.
#   DRY_RUN=1 PROBES="ego_belief ego_belief_v2" PREFIX_LENGTHS="8 12 16 24" \
#     bash scripts/run_streaming_probe_sweep.sh
#
#   # Single-model override, preserving the old usage pattern.
#   DRY_RUN=1 VFM=wan PROBES="ego_belief ego_belief_v2" PREFIX_LENGTHS="8 12 16 24" \
#     bash scripts/run_streaming_probe_sweep.sh
#
#   # B2 sanity: current-view visible object localization.
#   DEV=0 VFM=wan PROBES="visible_ego_belief_v2" PREFIX_LENGTHS="8 12 16 24" \
#   LAYERS="default" bash scripts/run_streaming_probe_sweep.sh
#
#   PYTHON=/data/baiqiao/miniconda3/envs/vidfm3d/bin/python DEV=0 \
#   PROBES="view_consistency ego_belief action_dynamics path_integration counterfactual" \
#   LAYERS="default" EXTRA_TRAIN="trainer.max_epochs=20 logger.wandb.offline=true" \
#     bash scripts/run_streaming_probe_sweep.sh
set -euo pipefail

DEFAULT_VFMS=${DEFAULT_VFMS:-"wan cogvideox vjepa2 dino aether f3r qwen2_5_vl_3b bagel"}
if [[ -z "${STREAMING_SWEEP_SINGLE_VFM:-}" ]]; then
    if [[ -n "${VFMS:-}" ]]; then
        sweep_vfms="${VFMS}"
    elif [[ -n "${VFM:-}" ]]; then
        sweep_vfms="${VFM}"
    else
        sweep_vfms="${DEFAULT_VFMS}"
    fi
    echo "[info] streaming sweep vfms=${sweep_vfms}"
    for sweep_vfm in ${sweep_vfms}; do
        echo "==== model sweep: ${sweep_vfm} ===="
        env STREAMING_SWEEP_SINGLE_VFM=1 VFM="${sweep_vfm}" VFM_NAME="${sweep_vfm}" bash "${BASH_SOURCE[0]}"
    done
    exit 0
fi

PYTHON=${PYTHON:-python}
VFM=${VFM:-wan}
VFM_NAME=${VFM_NAME:-${VFM}}
T=${T:-749}
DEV=${DEV:-0}
DRY_RUN=${DRY_RUN:-0}
SPLIT=${SPLIT:-val}

PROBES=${PROBES:-"streaming_depth view_consistency ego_belief ego_belief_v2 action_dynamics path_integration counterfactual"}
PREFIX_LENGTHS=${PREFIX_LENGTHS:-"8 12 16 24"}
B_HIDDEN_PREFIX_LENGTHS=${B_HIDDEN_PREFIX_LENGTHS:-"8 12 16 24"}
C_PREFIX_LENGTHS=${C_PREFIX_LENGTHS:-${PREFIX_LENGTHS}}
C_TARGET_HORIZONS=${C_TARGET_HORIZONS:-"1 2 4"}
STREAMING_MOTION_STEP=${STREAMING_MOTION_STEP:-0.35}
STREAMING_ROTATION_WEIGHT=${STREAMING_ROTATION_WEIGHT:-0.5}
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
MLLM=${MLLM:-auto}
BACKEND=${BACKEND:-}
MODEL_ID=${MODEL_ID:-}
NUM_FRAMES=${NUM_FRAMES:-}
RESIZE=${RESIZE:-}
VIDEO_CHANNELS=${VIDEO_CHANNELS:-auto}

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
    if [[ "${MLLM}" == "1" ]]; then
        printf 'inscene15k_streaming/direct_vlm_%s_v1' "${probe}"
    else
        printf 'inscene15k_streaming/%s_%s_v1' "${probe}" "${VFM}"
    fi
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
if [[ "${MLLM}" == "1" && -z "${MODEL_ID}" ]]; then
    case "${VFM}" in
        qwen2_5_vl_3b) MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct ;;
        qwen2_5_vl) MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct ;;
        bagel) MODEL_ID=ByteDance-Seed/BAGEL-7B-MoT ;;
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
motion_tag="${STREAMING_MOTION_STEP//./p}"
rotation_tag="${STREAMING_ROTATION_WEIGHT//./p}"
stream_root="${STREAM_ROOT_BASE}/${VFM}_${prefix_name}_m${motion_tag}_r${rotation_tag}"
hidden_prefix_list="[${PREFIX_LENGTHS// /,}]"
hidden_train_prefix_list="[${B_HIDDEN_PREFIX_LENGTHS// /,}]"

split_extra "${EXTRA_EXTRACT}" extra_extract_args
split_extra "${EXTRA_TARGET_EXTRACT}" extra_target_extract_args
split_extra "${EXTRA_TRAIN}" extra_train_args
split_extra "${EXTRA_EVAL}" extra_eval_args

echo "[info] vfm=${VFM} probes=${PROBES}"
echo "[info] prefix_lengths=${PREFIX_LENGTHS} layers=${layers_resolved[*]}"
echo "[info] streaming_feat_root=${stream_root}"

resolve_channels() {
    local layer="$1"
    if [[ "${VIDEO_CHANNELS}" != "auto" ]]; then
        printf '%s\n' "${VIDEO_CHANNELS}"
        return 0
    fi
    "${PYTHON}" - "$stream_root" "$VFM" "$layer" "$T" <<'PY'
import glob
import os
import sys

from safetensors.torch import load_file
from vidfm3d.utils.feature_layers import default_feature_channels, feature_filename

root, vfm, layer, timestep = sys.argv[1:5]
fname = feature_filename(vfm, feature_layer=int(layer), feature_timestep=int(timestep))
patterns = [
    os.path.join(root, vfm, "*", "*", "*", "*", fname),
    os.path.join(root, vfm, "*", "*", "*", fname),
    os.path.join(root, vfm, "*", "*", fname),
]
matches = []
for pattern in patterns:
    matches.extend(glob.glob(pattern))
matches = sorted(set(matches))
if matches:
    feat = load_file(matches[0])["feat"]
    print(int(feat.shape[-1]))
else:
    print(int(default_feature_channels(vfm)))
PY
}

if [[ "${EXTRACT_STREAMING}" == "1" ]]; then
    echo "==== extract shared streaming-prefix cache ===="
    if [[ "${MLLM}" == "1" ]]; then
        cmd=("${PYTHON}" -m features.run_inscene15k_mllm
            --backend "${BACKEND}"
            --vfm-name "${VFM_NAME}"
            --mode streaming_prefix
            --data-root "${DATA_ROOT}"
            --out-root "${stream_root}"
            --output-layers "${layers_resolved[@]}"
            --source scannetpp
            --prefix-min-len 1
            --prefix-max-len "${max_prefix}"
            --prefix-lengths "${PREFIX_LENGTHS}"
            --streaming-motion-step "${STREAMING_MOTION_STEP}"
            --streaming-rotation-weight "${STREAMING_ROTATION_WEIGHT}"
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
            --mode streaming_prefix
            --data-root "${DATA_ROOT}"
            --out-root "${stream_root}"
            --t "${T}"
            --output-layers "${layers_resolved[@]}"
            --source scannetpp
            --prefix-min-len 1
            --prefix-max-len "${max_prefix}"
            --prefix-lengths "${PREFIX_LENGTHS}"
            --streaming-motion-step "${STREAMING_MOTION_STEP}"
            --streaming-rotation-weight "${STREAMING_ROTATION_WEIGHT}"
        )
    fi
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
    if [[ "${MLLM}" == "1" ]]; then
        cmd=("${PYTHON}" -m features.run_inscene15k_mllm
            --backend "${BACKEND}"
            --vfm-name "${VFM_NAME}"
            --mode target_isolated
            --data-root "${DATA_ROOT}"
            --out-root "${TARGET_FEAT_ROOT}"
            --output-layers "${layers_resolved[@]}"
            --source scannetpp
            --prefix-max-len "${max_prefix}"
            --prefix-lengths "${PREFIX_LENGTHS}"
            --streaming-motion-step "${STREAMING_MOTION_STEP}"
            --streaming-rotation-weight "${STREAMING_ROTATION_WEIGHT}"
            --num-targets 0
            --target-from-streaming-windows
            --target-prefix-lengths "${C_PREFIX_LENGTHS}"
            --target-horizons "${C_TARGET_HORIZONS}"
        )
        if [[ -n "${MODEL_ID}" ]]; then
            cmd+=(--model-id "${MODEL_ID}")
        fi
        if [[ -n "${RESIZE}" ]]; then
            read -r -a resize_args <<< "${RESIZE}"
            cmd+=(--resize "${resize_args[@]}")
        fi
    else
        cmd=("${PYTHON}" -m features.run_inscene15k
            --vfm "${VFM}"
            --mode target_isolated
            --data-root "${DATA_ROOT}"
            --out-root "${TARGET_FEAT_ROOT}"
            --t "${T}"
            --output-layers "${layers_resolved[@]}"
            --source scannetpp
            --prefix-max-len "${max_prefix}"
            --prefix-lengths "${PREFIX_LENGTHS}"
            --streaming-motion-step "${STREAMING_MOTION_STEP}"
            --streaming-rotation-weight "${STREAMING_ROTATION_WEIGHT}"
            --num-targets 0
            --target-from-streaming-windows
            --target-prefix-lengths "${C_PREFIX_LENGTHS}"
            --target-horizons "${C_TARGET_HORIZONS}"
        )
    fi
    cmd+=("${extra_target_extract_args[@]}")
    run_cmd "${cmd[@]}"
fi

if [[ "${needs_targets}" == "1" && -z "${TARGET_FEAT_ROOT}" ]]; then
    echo "[err] C probes need TARGET_FEAT_ROOT or INSCENE_TARGET_FEAT_ROOT" >&2
    exit 1
fi

for probe in ${PROBES}; do
    cfg="$(probe_cfg "${probe}")"
    cfg_file="configs/experiment/${cfg}.yaml"
    if [[ ! -f "${cfg_file}" ]]; then
        echo "[err] missing config for probe=${probe}: ${cfg_file}" >&2
        exit 1
    fi

    probe_prefix_lengths="${PREFIX_LENGTHS}"
    if probe_uses_shared_hidden_object "${probe}"; then
        probe_prefix_lengths="${B_HIDDEN_PREFIX_LENGTHS}"
    elif probe_needs_target_cache "${probe}"; then
        probe_prefix_lengths="${C_PREFIX_LENGTHS}"
    fi

    for prefix_len in ${probe_prefix_lengths}; do
        for layer in "${layers_resolved[@]}"; do
            layer_name="$(safe_name "${layer}")"
            job="${probe}_${VFM}_p${prefix_len}_layer${layer_name}"
            run_name="inscene15k_streaming_${job}"
            run_dir="${LOGS_ROOT}/${run_name}"
            ckpt="${run_dir}/checkpoints/last.ckpt"

            if [[ "${TRAIN}" == "1" ]]; then
                video_channels="$(resolve_channels "${layer}")"
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
                    cmd+=("streaming_hidden_prefix_lengths=${hidden_train_prefix_list}")
                fi
                if probe_needs_target_cache "${probe}"; then
                    cmd+=("target_feat_root=${TARGET_FEAT_ROOT}")
                fi
                cmd+=("${extra_train_args[@]}")
                run_cmd "${cmd[@]}"
            fi

            if [[ "${EVAL}" == "1" && "${probe}" != "streaming_depth" ]]; then
                video_channels="$(resolve_channels "${layer}")"
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
                    cmd+=("streaming_hidden_prefix_lengths=${hidden_train_prefix_list}")
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
