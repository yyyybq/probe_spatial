#!/usr/bin/env bash
# 8-GPU orchestration for Streaming Prefix Depth.
#
# For each prefix spec:
#   1) shard feature extraction across GPU_IDS using --start/--end,
#   2) train all requested layers with a bounded GPU job queue,
#   3) continue to the next prefix spec.
#
# The script is resume-friendly because features/run_inscene15k.py skips
# completed scenes and training uses the configured autoresume behavior.
set -euo pipefail

PYTHON=${PYTHON:-/data/baiqiao/miniconda3/envs/vidfm3d/bin/python}
VFM=${VFM:-wan}
CFG=${CFG:-inscene15k_streaming/streaming_depth_wan_v1}
T=${T:-749}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
LAYERS=${LAYERS:-"0 5 10 15 20 25 29"}
PREFIX_SPECS=${PREFIX_SPECS:-"1:16:1 1:32:1 1:81:1"}
TOTAL_SCENES=${TOTAL_SCENES:-2370}

DATA_ROOT=${DATA_ROOT:-${INSCENE_DATA_ROOT:?set DATA_ROOT or INSCENE_DATA_ROOT}}
STREAM_ROOT_BASE=${STREAM_ROOT_BASE:-${INSCENE_STREAMING_FEAT_ROOT:?set STREAM_ROOT_BASE or INSCENE_STREAMING_FEAT_ROOT}}
LOGS_ROOT=${LOGS_ROOT:-logs/inscene15k_streaming/runs}
RUN_LOG_DIR=${RUN_LOG_DIR:-logs/streaming_prefix_depth_8gpu}
SPLIT=${SPLIT:-val}

EXTRACT=${EXTRACT:-1}
TRAIN=${TRAIN:-1}
SUMMARIZE=${SUMMARIZE:-1}
DRY_RUN=${DRY_RUN:-0}

EXTRA_EXTRACT=${EXTRA_EXTRACT:-}
EXTRA_TRAIN=${EXTRA_TRAIN:-logger.wandb.offline=true}
EXTRA_SUMMARY=${EXTRA_SUMMARY:-}

export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}" "${RUN_LOG_DIR}"

run_or_print() {
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

parse_prefix_spec() {
    local spec="$1"
    local normalized
    normalized="$(sed -E 's/^min//; s/_?max/:/; s/_?stride/:/; s/[,_-]+/:/g; s/^p//; s/^://; s/:$//' <<< "${spec}")"
    IFS=':' read -r pmin pmax pstride <<< "${normalized}"
    pstride="${pstride:-1}"
    if ! [[ "${pmin:-}" =~ ^[0-9]+$ && "${pmax:-}" =~ ^[0-9]+$ && "${pstride:-}" =~ ^[0-9]+$ ]]; then
        echo "[err] bad prefix spec '${spec}'. Use min:max:stride, e.g. 1:81:1" >&2
        exit 1
    fi
    if (( pmin < 1 || pmax < pmin || pstride < 1 )); then
        echo "[err] invalid prefix spec '${spec}'" >&2
        exit 1
    fi
}

safe_layer_name() {
    local layer="$1"
    if [[ "${layer}" == -* ]]; then
        printf 'neg%s' "${layer#-}"
    else
        printf '%s' "${layer}"
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

gpu_arr=()
for gpu in ${GPU_IDS}; do
    gpu_arr+=("${gpu}")
done
if (( ${#gpu_arr[@]} == 0 )); then
    echo "[err] GPU_IDS is empty" >&2
    exit 1
fi

split_extra "${EXTRA_EXTRACT}" extra_extract_args
split_extra "${EXTRA_TRAIN}" extra_train_args
split_extra "${EXTRA_SUMMARY}" extra_summary_args

video_channels=$("${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --field in_channels 2>/dev/null || true)
if [[ -z "${video_channels}" ]]; then
    video_channels=1536
fi

echo "[info] started at $(date)"
echo "[info] gpus=${GPU_IDS}"
echo "[info] total_scenes=${TOTAL_SCENES}"
echo "[info] prefix_specs=${PREFIX_SPECS}"
echo "[info] layers=${layers_resolved[*]}"
echo "[info] logs=${RUN_LOG_DIR}"

for spec in ${PREFIX_SPECS}; do
    parse_prefix_spec "${spec}"
    prefix_name="pmin${pmin}_pmax${pmax}_s${pstride}"
    feat_root="${STREAM_ROOT_BASE}/${VFM}_${prefix_name}"

    if [[ "${EXTRACT}" == "1" ]]; then
        echo "==== extract ${prefix_name} across ${#gpu_arr[@]} GPUs ===="
        pids=()
        shard_count=${#gpu_arr[@]}
        shard_size=$(( (TOTAL_SCENES + shard_count - 1) / shard_count ))
        for shard_idx in "${!gpu_arr[@]}"; do
            gpu="${gpu_arr[${shard_idx}]}"
            start=$(( shard_idx * shard_size ))
            end=$(( start + shard_size ))
            if (( start >= TOTAL_SCENES )); then
                continue
            fi
            if (( end > TOTAL_SCENES )); then
                end=${TOTAL_SCENES}
            fi
            log_file="${RUN_LOG_DIR}/extract_${prefix_name}_gpu${gpu}_s${start}_${end}.log"
            cmd=(env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m features.run_inscene15k
                --vfm "${VFM}"
                --mode streaming_prefix
                --data-root "${DATA_ROOT}"
                --out-root "${feat_root}"
                --t "${T}"
                --output-layers "${layers_resolved[@]}"
                --prefix-min-len "${pmin}"
                --prefix-max-len "${pmax}"
                --prefix-stride "${pstride}"
                --start "${start}"
                --end "${end}"
            )
            cmd+=("${extra_extract_args[@]}")
            if [[ "${DRY_RUN}" == "1" ]]; then
                run_or_print "${cmd[@]}"
            else
                echo "[launch] gpu=${gpu} scenes=[${start},${end}) log=${log_file}"
                "${cmd[@]}" >"${log_file}" 2>&1 &
                pids+=("$!")
            fi
        done
        if [[ "${DRY_RUN}" != "1" ]]; then
            for pid in "${pids[@]}"; do
                wait "${pid}"
            done
        fi
    fi

    if [[ "${TRAIN}" == "1" ]]; then
        echo "==== train ${prefix_name} layers=${layers_resolved[*]} ===="
        train_pids=()
        for idx in "${!layers_resolved[@]}"; do
            layer="${layers_resolved[${idx}]}"
            gpu="${gpu_arr[$(( idx % ${#gpu_arr[@]} ))]}"
            layer_name="$(safe_layer_name "${layer}")"
            job="streaming_depth_${VFM}_${prefix_name}_layer${layer_name}"
            run_name="inscene15k_streaming_${job}"
            log_file="${RUN_LOG_DIR}/train_${job}_gpu${gpu}.log"
            cmd=(env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" vidfm3d/train.py
                "experiment=${CFG}"
                "vfm_name=${VFM}"
                "video_channels=${video_channels}"
                "streaming_feat_root=${feat_root}"
                "feature_layer=${layer}"
                "feature_timestep=${T}"
                "prefix_min_len=${pmin}"
                "prefix_max_len=${pmax}"
                "prefix_stride=${pstride}"
                "job_name=${job}"
                "paths.run_folder_name=${run_name}"
                "logger.wandb.name=${run_name}"
            )
            cmd+=("${extra_train_args[@]}")
            if [[ "${DRY_RUN}" == "1" ]]; then
                run_or_print "${cmd[@]}"
            else
                echo "[launch] gpu=${gpu} layer=${layer} log=${log_file}"
                "${cmd[@]}" >"${log_file}" 2>&1 &
                train_pids+=("$!")
            fi
        done
        if [[ "${DRY_RUN}" != "1" ]]; then
            for pid in "${train_pids[@]}"; do
                wait "${pid}"
            done
        fi
    fi
done

if [[ "${SUMMARIZE}" == "1" ]]; then
    echo "==== summarize ===="
    cmd=("${PYTHON}" scripts/summarize_streaming_prefix_sweep.py
        --runs-root "${LOGS_ROOT}"
        --pattern "inscene15k_streaming_streaming_depth_${VFM}_*"
        --split "${SPLIT}"
        --output "streaming_prefix_depth_${VFM}_${SPLIT}.csv"
    )
    cmd+=("${extra_summary_args[@]}")
    run_or_print "${cmd[@]}"
fi

echo "[info] finished at $(date)"
