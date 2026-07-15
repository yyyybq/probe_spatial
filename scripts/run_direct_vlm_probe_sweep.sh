#!/usr/bin/env bash
# Direct VLM layer probing without SAE.
#
# Default behavior is the current streaming protocol. Set STREAMING=0 and
# ALLOW_NON_STREAMING=1 only for legacy full-clip / normal-cache experiments.
#
# Examples:
#   DRY_RUN=1 VFMS="qwen2_5_vl qwen2_5_vl_3b" PREFIX_LENGTHS="8 12 16 24" \
#     LAYERS="-1 8 16 24 31" bash scripts/run_direct_vlm_probe_sweep.sh
#
#   STREAMING=0 ALLOW_NON_STREAMING=1 VFM=qwen2_5_vl PROBES="ego_belief_v2" LAYERS="-1 16 31" DEV=0 \
#     EXTRA_TRAIN="trainer.max_epochs=20 logger.wandb.offline=true" \
#     bash scripts/run_direct_vlm_probe_sweep.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
STREAMING=${STREAMING:-1}
ALLOW_NON_STREAMING=${ALLOW_NON_STREAMING:-0}
VFMS=${VFMS:-${VFM:-qwen2_5_vl}}
PROBES=${PROBES:-}
LAYERS=${LAYERS:-"-1 8 16 24 31"}
DEV=${DEV:-0}
DRY_RUN=${DRY_RUN:-0}
EXTRACT=${EXTRACT:-1}
TRAIN=${TRAIN:-1}
EVAL=${EVAL:-1}
SUMMARIZE=${SUMMARIZE:-1}

DATA_ROOT=${DATA_ROOT:-${INSCENE_DATA_ROOT:?set DATA_ROOT or INSCENE_DATA_ROOT}}
FEAT_ROOT=${FEAT_ROOT:-${INSCENE_MLLM_FEAT_ROOT:-}}
LOGS_ROOT=${LOGS_ROOT:-logs/inscene15k_ext/runs}
SPLIT=${SPLIT:-val}
NUM_FRAMES=${NUM_FRAMES:-8}
RESIZE=${RESIZE:-}
EXTRA_EXTRACT=${EXTRA_EXTRACT:-}
EXTRA_TRAIN=${EXTRA_TRAIN:-}
EXTRA_EVAL=${EXTRA_EVAL:-}

if [[ "${STREAMING}" == "1" ]]; then
    STREAMING_PROBES=${PROBES:-"streaming_depth view_consistency ego_belief ego_belief_v2 action_dynamics path_integration counterfactual"}
    for vfm in ${VFMS}; do
        echo "==== direct VLM streaming vfm=${vfm} layers=${LAYERS} prefixes=${PREFIX_LENGTHS:-8 12 16 24} ===="
        cmd=(env
            PYTHON="${PYTHON}"
            VFM="${vfm}"
            VFM_NAME="${vfm}"
            PROBES="${STREAMING_PROBES}"
            LAYERS="${LAYERS}"
            DEV="${DEV}"
            DRY_RUN="${DRY_RUN}"
            EXTRACT_STREAMING="${EXTRACT}"
            EXTRACT_TARGETS="${EXTRACT}"
            TRAIN="${TRAIN}"
            EVAL="${EVAL}"
            MLLM=1
            DATA_ROOT="${DATA_ROOT}"
            STREAM_ROOT_BASE="${STREAM_ROOT_BASE:-${INSCENE_STREAMING_FEAT_ROOT:?set STREAM_ROOT_BASE or INSCENE_STREAMING_FEAT_ROOT}}"
            TARGET_FEAT_ROOT="${TARGET_FEAT_ROOT:-${INSCENE_TARGET_FEAT_ROOT:-}}"
            LOGS_ROOT="${LOGS_ROOT:-logs/inscene15k_streaming/runs}"
            SPLIT="${SPLIT}"
            EXTRA_EXTRACT="${EXTRA_EXTRACT}"
            EXTRA_TARGET_EXTRACT="${EXTRA_TARGET_EXTRACT:-}"
            EXTRA_TRAIN="${EXTRA_TRAIN}"
            EXTRA_EVAL="${EXTRA_EVAL}"
        )
        if [[ -n "${PREFIX_LENGTHS:-}" ]]; then
            cmd+=(PREFIX_LENGTHS="${PREFIX_LENGTHS}")
        fi
        if [[ -n "${MODEL_ID:-}" ]]; then
            cmd+=(MODEL_ID="${MODEL_ID}")
        fi
        if [[ -n "${RESIZE}" ]]; then
            cmd+=(RESIZE="${RESIZE}")
        fi
        cmd+=(bash scripts/run_streaming_probe_sweep.sh)
        "${cmd[@]}"
    done
    exit 0
fi

if [[ "${ALLOW_NON_STREAMING}" != "1" ]]; then
    cat >&2 <<'EOF'
[err] STREAMING=0 requested, but non-streaming direct VLM is legacy.
      Streaming is the project default. Either omit STREAMING=0, or explicitly set:
        STREAMING=0 ALLOW_NON_STREAMING=1 bash scripts/run_direct_vlm_probe_sweep.sh ...
EOF
    exit 2
fi

PROBES=${PROBES:-"view_consistency abnormal ego_belief ego_belief_v2 action_dynamics path_integration counterfactual"}
FEAT_ROOT=${FEAT_ROOT:?set FEAT_ROOT or INSCENE_MLLM_FEAT_ROOT for legacy non-streaming direct VLM}

backend_for_vfm() {
    case "$1" in
        qwen2_5_vl*) printf 'qwen2_5_vl' ;;
        bagel) printf 'bagel_hf' ;;
        *) printf 'generic_hf' ;;
    esac
}

cfg_for_probe_vfm() {
    local probe="$1"
    local vfm="$2"
    local specific="inscene15k_ext/direct_vlm_${probe}_${vfm}_v1"
    local generic="inscene15k_ext/direct_vlm_${probe}_v1"
    if [[ -f "configs/experiment/${specific}.yaml" ]]; then
        printf '%s' "${specific}"
    else
        printf '%s' "${generic}"
    fi
}

model_id_for_vfm() {
    case "$1" in
        qwen2_5_vl_3b) printf 'Qwen/Qwen2.5-VL-3B-Instruct' ;;
        qwen2_5_vl) printf 'Qwen/Qwen2.5-VL-7B-Instruct' ;;
        bagel) printf 'ByteDance-Seed/BAGEL-7B-MoT' ;;
        *) printf '' ;;
    esac
}

for vfm in ${VFMS}; do
    backend="$(backend_for_vfm "${vfm}")"
    model_id="$(model_id_for_vfm "${vfm}")"
    for probe in ${PROBES}; do
        cfg="$(cfg_for_probe_vfm "${probe}" "${vfm}")"
        cfg_file="configs/experiment/${cfg}.yaml"
        if [[ ! -f "${cfg_file}" ]]; then
            echo "[err] missing direct VLM config: ${cfg_file}" >&2
            exit 1
        fi
        echo "==== direct VLM probe=${probe} vfm=${vfm} layers=${LAYERS} ===="
        cmd=(env
            PYTHON="${PYTHON}"
            VFM="${vfm}"
            VFM_NAME="${vfm}"
            PROBE="${probe}"
            CFG="${cfg}"
            LAYERS="${LAYERS}"
            DEV="${DEV}"
            DRY_RUN="${DRY_RUN}"
            EXTRACT="${EXTRACT}"
            TRAIN="${TRAIN}"
            EVAL="${EVAL}"
            SUMMARIZE="${SUMMARIZE}"
            MLLM=1
            ALLOW_NON_STREAMING=1
            BACKEND="${backend}"
            DATA_ROOT="${DATA_ROOT}"
            FEAT_ROOT="${FEAT_ROOT}"
            LOGS_ROOT="${LOGS_ROOT}"
            SPLIT="${SPLIT}"
            NUM_FRAMES="${NUM_FRAMES}"
            EXTRA_EXTRACT="${EXTRA_EXTRACT}"
            EXTRA_TRAIN="${EXTRA_TRAIN}"
            EXTRA_EVAL="${EXTRA_EVAL}"
        )
        if [[ -n "${model_id}" ]]; then
            cmd+=(MODEL_ID="${model_id}")
        fi
        if [[ -n "${RESIZE}" ]]; then
            cmd+=(RESIZE="${RESIZE}")
        fi
        cmd+=(bash scripts/run_feature_layer_probe_sweep.sh)
        "${cmd[@]}"
    done
done
