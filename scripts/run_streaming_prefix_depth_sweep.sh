#!/usr/bin/env bash
# End-to-end Streaming Prefix Depth sweep.
#
# Pipeline:
#   1) extract streaming-prefix feature caches for each prefix spec and layer set,
#   2) train one depth+camera probe per (prefix spec, layer),
#   3) optionally run eval_diag.py when the probe type is supported,
#   4) summarize run metadata / eval summaries into one CSV.
#
# Examples:
#   DRY_RUN=1 LAYERS="default last" PREFIX_SPECS="1:16:1 1:81:1" \
#     bash scripts/run_streaming_prefix_depth_sweep.sh
#
#   PYTHON=/data/baiqiao/miniconda3/envs/vidfm3d/bin/python DEV=0 \
#   LAYERS="10 20 29" PREFIX_SPECS="1:16:1 1:32:1 1:81:1" \
#   EXTRA_TRAIN="trainer.max_epochs=20 logger.wandb.offline=true" \
#     bash scripts/run_streaming_prefix_depth_sweep.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
VFM=${VFM:-wan}
CFG=${CFG:-inscene15k_streaming/streaming_depth_wan_v1}
T=${T:-749}
DEV=${DEV:-0}
DRY_RUN=${DRY_RUN:-0}
SPLIT=${SPLIT:-val}

DATA_ROOT=${DATA_ROOT:-${INSCENE_DATA_ROOT:?set DATA_ROOT or INSCENE_DATA_ROOT}}
STREAM_ROOT_BASE=${STREAM_ROOT_BASE:-${INSCENE_STREAMING_FEAT_ROOT:?set STREAM_ROOT_BASE or INSCENE_STREAMING_FEAT_ROOT}}
LOGS_ROOT=${LOGS_ROOT:-logs/inscene15k_streaming/runs}

# Keep the default modest: full all-layer x all-prefix extraction is expensive.
LAYERS=${LAYERS:-default}
PREFIX_SPECS=${PREFIX_SPECS:-"1:81:1"}

EXTRACT=${EXTRACT:-1}
TRAIN=${TRAIN:-1}
EVAL=${EVAL:-1}
SUMMARIZE=${SUMMARIZE:-1}

EXTRA_EXTRACT=${EXTRA_EXTRACT:-}
EXTRA_TRAIN=${EXTRA_TRAIN:-}
EXTRA_EVAL=${EXTRA_EVAL:-}
EXTRA_SUMMARY=${EXTRA_SUMMARY:-}

export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

cfg_file="configs/experiment/${CFG}.yaml"
if [[ ! -f "${cfg_file}" ]]; then
    echo "[err] missing config: ${cfg_file}" >&2
    exit 1
fi

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

parse_prefix_spec() {
    local spec="$1"
    local normalized
    normalized="$(sed -E 's/^min//; s/_?max/:/; s/_?stride/:/; s/[,_-]+/:/g; s/^p//; s/^://; s/:$//' <<< "${spec}")"
    IFS=':' read -r pmin pmax pstride <<< "${normalized}"
    if [[ -z "${pmin:-}" || -z "${pmax:-}" ]]; then
        echo "[err] bad prefix spec '${spec}'. Use min:max:stride, e.g. 1:81:1" >&2
        exit 1
    fi
    pstride="${pstride:-1}"
    if ! [[ "${pmin}" =~ ^[0-9]+$ && "${pmax}" =~ ^[0-9]+$ && "${pstride}" =~ ^[0-9]+$ ]]; then
        echo "[err] bad prefix spec '${spec}'. Values must be positive integers." >&2
        exit 1
    fi
    if (( pmin < 1 || pmax < pmin || pstride < 1 )); then
        echo "[err] invalid prefix spec '${spec}': require 1 <= min <= max and stride >= 1" >&2
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

split_extra "${EXTRA_EXTRACT}" extra_extract_args
split_extra "${EXTRA_TRAIN}" extra_train_args
split_extra "${EXTRA_EVAL}" extra_eval_args
split_extra "${EXTRA_SUMMARY}" extra_summary_args

echo "[info] cfg=${CFG}"
echo "[info] vfm=${VFM} layers=${layers_resolved[*]} video_channels=${video_channels}"
echo "[info] prefix_specs=${PREFIX_SPECS}"

for spec in ${PREFIX_SPECS}; do
    parse_prefix_spec "${spec}"
    prefix_name="pmin${pmin}_pmax${pmax}_s${pstride}"
    feat_root="${STREAM_ROOT_BASE}/${VFM}_${prefix_name}"

    if [[ "${EXTRACT}" == "1" ]]; then
        echo "==== extract streaming_prefix ${VFM} ${prefix_name} layers=${layers_resolved[*]} ===="
        cmd=("${PYTHON}" -m features.run_inscene15k
            --vfm "${VFM}"
            --mode streaming_prefix
            --data-root "${DATA_ROOT}"
            --out-root "${feat_root}"
            --t "${T}"
            --output-layers "${layers_resolved[@]}"
            --prefix-min-len "${pmin}"
            --prefix-max-len "${pmax}"
            --prefix-stride "${pstride}"
        )
        cmd+=("${extra_extract_args[@]}")
        run_cmd "${cmd[@]}"
    fi

    for layer in "${layers_resolved[@]}"; do
        layer_name="$(safe_layer_name "${layer}")"
        job="streaming_depth_${VFM}_${prefix_name}_layer${layer_name}"
        run_name="inscene15k_streaming_${job}"
        run_dir="${LOGS_ROOT}/${run_name}"
        ckpt="${run_dir}/checkpoints/last.ckpt"

        if [[ "${TRAIN}" == "1" ]]; then
            echo "==== train ${CFG} ${prefix_name} layer=${layer} ===="
            cmd=(env CUDA_VISIBLE_DEVICES="${DEV}" "${PYTHON}" vidfm3d/train.py
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
            run_cmd "${cmd[@]}"
        fi

        if [[ "${EVAL}" == "1" ]]; then
            if [[ "${DRY_RUN}" != "1" && ! -f "${ckpt}" ]]; then
                echo "[warn] missing checkpoint, skip eval: ${ckpt}" >&2
                continue
            fi
            echo "==== eval-if-supported ${CFG} ${prefix_name} layer=${layer} ===="
            cmd=("${PYTHON}" vidfm3d/eval_diag.py
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
                "ckpt_path=${ckpt}"
                "+eval_split=${SPLIT}"
                "train=false"
                "test=false"
            )
            cmd+=("${extra_eval_args[@]}")
            if [[ "${DRY_RUN}" == "1" ]]; then
                run_cmd "${cmd[@]}"
            elif ! "${cmd[@]}"; then
                echo "[warn] eval_diag failed or unsupported for ${job}; summary will still include run metadata." >&2
            fi
        fi
    done
done

if [[ "${SUMMARIZE}" == "1" ]]; then
    out_csv="streaming_prefix_depth_${VFM}_${SPLIT}.csv"
    echo "==== summarize streaming prefix sweep ===="
    cmd=("${PYTHON}" scripts/summarize_streaming_prefix_sweep.py
        --runs-root "${LOGS_ROOT}"
        --pattern "inscene15k_streaming_streaming_depth_${VFM}_*"
        --split "${SPLIT}"
        --output "${out_csv}"
    )
    cmd+=("${extra_summary_args[@]}")
    run_cmd "${cmd[@]}"
fi
