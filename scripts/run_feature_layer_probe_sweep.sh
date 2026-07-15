#!/usr/bin/env bash
# Legacy non-streaming end-to-end layer sweep:
#   1) extract feature caches for many layers,
#   2) train one probe per layer,
#   3) evaluate each checkpoint,
#   4) summarize best/default/last layer scores.
#
# Examples:
#   ALLOW_NON_STREAMING=1 DRY_RUN=1 VFM=wan PROBE=view_consistency LAYERS="0 5 10 15 20 25 29" \
#     bash scripts/run_feature_layer_probe_sweep.sh
#
#   ALLOW_NON_STREAMING=1 VFM=vjepa2 PROBE=action_dynamics LAYERS="0 5 11 17 23" DEV=0 \
#     EXTRA_TRAIN="trainer.max_epochs=10 logger.wandb.offline=true" \
#     bash scripts/run_feature_layer_probe_sweep.sh
#
#   ALLOW_NON_STREAMING=1 CFG=inscene15k_ext/sae_qwen2_5vl_v1 VFM=qwen2_5_vl PROBE=sae \
#     SUMMARY_PROBE=sae_spatial \
#     LAYERS="-1 8 16 24" bash scripts/run_feature_layer_probe_sweep.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
ALLOW_NON_STREAMING=${ALLOW_NON_STREAMING:-0}
VFM=${VFM:-wan}
PROBE=${PROBE:-view_consistency}
LAYERS=${LAYERS:-}
T=${T:-749}
DEV=${DEV:-0}
SPLIT=${SPLIT:-val}
DRY_RUN=${DRY_RUN:-0}

if [[ "${ALLOW_NON_STREAMING}" != "1" ]]; then
    cat >&2 <<'EOF'
[err] run_feature_layer_probe_sweep.sh is the legacy non-streaming cache/probe path.
      Streaming is the project default. Use:
        bash scripts/run_streaming_probe_sweep.sh
      To intentionally run the legacy non-streaming layer sweep, set:
        ALLOW_NON_STREAMING=1 bash scripts/run_feature_layer_probe_sweep.sh ...
EOF
    exit 2
fi

DATA_ROOT=${DATA_ROOT:-${INSCENE_DATA_ROOT:?set DATA_ROOT or INSCENE_DATA_ROOT}}
case "${VFM}" in
    qwen2_5_vl*|bagel) DEFAULT_FEAT_ROOT=${INSCENE_MLLM_FEAT_ROOT:-} ;;
    *) DEFAULT_FEAT_ROOT=${INSCENE_FEAT_ROOT:-} ;;
esac
FEAT_ROOT=${FEAT_ROOT:-${DEFAULT_FEAT_ROOT:?set FEAT_ROOT or the matching INSCENE_*_FEAT_ROOT}}
SHUFFLED_FEAT_ROOT=${SHUFFLED_FEAT_ROOT:-${INSCENE_SHUFFLED_FEAT_ROOT:-}}
TARGET_FEAT_ROOT=${TARGET_FEAT_ROOT:-${INSCENE_TARGET_FEAT_ROOT:-}}
CONTEXT_FEAT_ROOT=${CONTEXT_FEAT_ROOT:-${INSCENE_CONTEXT_FEAT_ROOT:-}}
TARGET_NUM_TARGETS=${TARGET_NUM_TARGETS:-0}
CONTEXT_LEN=${CONTEXT_LEN:-76}
CONTEXT_STRIDE=${CONTEXT_STRIDE:-1}
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
VIDEO_CHANNELS=${VIDEO_CHANNELS:-auto}

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
if [[ "${MLLM}" == "1" && -z "${MODEL_ID}" ]]; then
    case "${VFM}" in
        qwen2_5_vl_3b) MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct ;;
        qwen2_5_vl) MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct ;;
        bagel) MODEL_ID=ByteDance-Seed/BAGEL-7B-MoT ;;
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
        context_segment)
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

resolve_channels() {
    local layer="$1"
    if [[ "${VIDEO_CHANNELS}" != "auto" ]]; then
        printf '%s\n' "${VIDEO_CHANNELS}"
        return 0
    fi
    "${PYTHON}" - "$FEAT_ROOT" "$VFM" "$layer" "$T" <<'PY'
import glob
import os
import sys

from safetensors.torch import load_file
from vidfm3d.utils.feature_layers import default_feature_channels, feature_filename

root, vfm, layer, timestep = sys.argv[1:5]
try:
    fname = feature_filename(vfm, feature_layer=int(layer), feature_timestep=int(timestep))
except Exception:
    fname = f"feature_layer{layer}.sft"
pattern = os.path.join(root, vfm, "*", "*", fname)
matches = sorted(glob.glob(pattern))
if matches:
    feat = load_file(matches[0])["feat"]
    print(int(feat.shape[-1]))
else:
    print(int(default_feature_channels(vfm)))
PY
}

if [[ "${EXTRACT}" == "1" ]]; then
    for mode in normal shuffled target_isolated context_segment; do
        if ! needs_mode "${mode}"; then
            continue
        fi
        case "${mode}" in
            normal) out_root="${FEAT_ROOT}" ;;
            shuffled)
                if [[ -z "${SHUFFLED_FEAT_ROOT}" ]]; then
                    echo "[err] set SHUFFLED_FEAT_ROOT or INSCENE_SHUFFLED_FEAT_ROOT for shuffled extraction" >&2
                    exit 1
                fi
                out_root="${SHUFFLED_FEAT_ROOT}"
                ;;
            target_isolated)
                if [[ -z "${TARGET_FEAT_ROOT}" ]]; then
                    echo "[err] set TARGET_FEAT_ROOT or INSCENE_TARGET_FEAT_ROOT for target_isolated extraction" >&2
                    exit 1
                fi
                out_root="${TARGET_FEAT_ROOT}"
                ;;
            context_segment)
                if [[ -z "${CONTEXT_FEAT_ROOT}" ]]; then
                    echo "[err] set CONTEXT_FEAT_ROOT or INSCENE_CONTEXT_FEAT_ROOT for context_segment extraction" >&2
                    exit 1
                fi
                out_root="${CONTEXT_FEAT_ROOT}"
                ;;
        esac
        echo "==== extract ${VFM} mode=${mode} layers=${layers_joined} ===="
        if [[ "${MLLM}" == "1" ]]; then
            cmd=("${PYTHON}" -m features.run_inscene15k_mllm
                --backend "${BACKEND}"
                --vfm-name "${VFM_NAME}"
                --mode "${mode}"
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
            if [[ "${mode}" == "shuffled" ]]; then
                cmd+=(--shuffle-seed 42)
            elif [[ "${mode}" == "target_isolated" ]]; then
                cmd+=(--num-targets "${TARGET_NUM_TARGETS}")
            elif [[ "${mode}" == "context_segment" ]]; then
                cmd+=(--context-len "${CONTEXT_LEN}" --context-stride "${CONTEXT_STRIDE}")
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
            if [[ "${mode}" == "target_isolated" ]]; then
                # C1/C2/C3 targets are exact isolated frame features. Cache every
                # frame by default; sparse target caches make many sampled horizons
                # scientifically unusable.
                cmd+=(--num-targets "${TARGET_NUM_TARGETS}")
            elif [[ "${mode}" == "context_segment" ]]; then
                # C1/C2/C3 inputs are causal video-segment forwards, e.g.
                # [I_1..I_48], which must not contain future target frames.
                cmd+=(--context-len "${CONTEXT_LEN}" --context-stride "${CONTEXT_STRIDE}")
            fi
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
    video_channels="$(resolve_channels "${layer}")"

    if [[ "${TRAIN}" == "1" ]]; then
        echo "==== train ${cfg} layer=${layer} ===="
        cmd=(env CUDA_VISIBLE_DEVICES="${DEV}" "${PYTHON}" vidfm3d/train.py
            "experiment=${cfg}"
            "feature_layer=${layer}"
            "feature_timestep=${T}"
            "video_channels=${video_channels}"
            "job_name=${job}"
            "paths.run_folder_name=${run_name}"
            "logger.wandb.name=${run_name}"
        )
        if [[ "${MLLM}" == "1" ]]; then
            cmd+=("vfm_name=${VFM_NAME}" "vlm_feat_root=${FEAT_ROOT}")
            if [[ "${PROBE}" == "abnormal" && -n "${SHUFFLED_FEAT_ROOT}" ]]; then
                cmd+=("shuffled_feat_root=${SHUFFLED_FEAT_ROOT}")
            fi
            if [[ "${PROBE}" == "action_dynamics" || "${PROBE}" == "path_integration" || "${PROBE}" == "counterfactual" ]]; then
                cmd+=("target_feat_root=${TARGET_FEAT_ROOT}" "context_feat_root=${CONTEXT_FEAT_ROOT}")
            fi
        fi
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
            "video_channels=${video_channels}"
            "job_name=${job}"
            "paths.run_folder_name=${run_name}"
            "ckpt_path=${ckpt}"
            "+eval_split=${SPLIT}"
            "train=false"
            "test=false"
        )
        if [[ "${MLLM}" == "1" ]]; then
            cmd+=("vfm_name=${VFM_NAME}" "vlm_feat_root=${FEAT_ROOT}")
            if [[ "${PROBE}" == "abnormal" && -n "${SHUFFLED_FEAT_ROOT}" ]]; then
                cmd+=("shuffled_feat_root=${SHUFFLED_FEAT_ROOT}")
            fi
            if [[ "${PROBE}" == "action_dynamics" || "${PROBE}" == "path_integration" || "${PROBE}" == "counterfactual" ]]; then
                cmd+=("target_feat_root=${TARGET_FEAT_ROOT}" "context_feat_root=${CONTEXT_FEAT_ROOT}")
            fi
        fi
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
