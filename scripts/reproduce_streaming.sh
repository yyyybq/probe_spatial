#!/usr/bin/env bash
# ==============================================================================
# reproduce_streaming.sh
# Full reproduction pipeline: data download → feature extraction → probing
# for the streaming setting (A1/A2/B2/C1/C2/C3) on InsScene-15K.
#
# Usage (minimal, all defaults):
#   INSCENE_BASE=/data/InsScene-15K bash scripts/reproduce_streaming.sh
#
# Usage (selective steps):
#   INSCENE_BASE=/data/InsScene-15K DOWNLOAD=0 EXTRACT=1 TRAIN=1 EVAL=1 \
#     VFMS="wan vjepa2" PROBES="view_consistency ego_belief_v2" \
#     bash scripts/reproduce_streaming.sh
#
# Dry-run (preview all commands):
#   DRY_RUN=1 INSCENE_BASE=/data/InsScene-15K bash scripts/reproduce_streaming.sh
#
# ==============================================================================
# REQUIRED environment variable:
#   INSCENE_BASE   Root directory for InsScene-15K
#                  Expected layout after download:
#                    ${INSCENE_BASE}/data/processed_infinigen/
#                    ${INSCENE_BASE}/data/processed_scannetpp_v2/
#
# OPTIONAL environment variables:
#   VFMS           Space-separated VFM names           (default: "wan vjepa2")
#   PROBES         Space-separated probe names         (default: see below)
#                  Probe → Paper name mapping:
#                    streaming_depth   → A1 Depth & Camera
#                    view_consistency  → A2 View Consistency
#                    ego_belief_v2     → B2 Object-Query Localization
#                    action_dynamics   → C1 Latent Dynamics
#                    path_integration  → C2 Path Integration
#                    counterfactual    → C3 Counterfactual Action
#   PREFIX_LENGTHS Space-separated prefix lengths      (default: "4 8 16 32 64")
#   GPU_WAN        CUDA device for wan                 (default: 0)
#   GPU_VJEPA2     CUDA device for vjepa2              (default: 1)
#   DOWNLOAD       1=download dataset, 0=skip          (default: 1)
#   EXTRACT        1=extract features, 0=skip          (default: 1)
#   TRAIN          1=train probes, 0=skip              (default: 1)
#   EVAL           1=evaluate checkpoints, 0=skip      (default: 1)
#   PARALLEL       1=run VFMs in parallel, 0=sequential(default: 0)
#   DRY_RUN        1=print commands only, 0=execute    (default: 0)
#   PYTHON         Python executable                   (default: python)
#   SEEDS          Seeds for multi-seed runs           (default: "42")
#   EXTRA_TRAIN    Extra hydra overrides for training  (default: "")
# ==============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

INSCENE_BASE="${INSCENE_BASE:?Please set INSCENE_BASE=/path/to/InsScene-15K}"

VFMS="${VFMS:-wan vjepa2}"
PROBES="${PROBES:-streaming_depth view_consistency ego_belief_v2 action_dynamics path_integration counterfactual}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-4 8 16 32 64}"

GPU_WAN="${GPU_WAN:-0}"
GPU_VJEPA2="${GPU_VJEPA2:-1}"

DOWNLOAD="${DOWNLOAD:-1}"
EXTRACT="${EXTRACT:-1}"
TRAIN="${TRAIN:-1}"
EVAL="${EVAL:-1}"
PARALLEL="${PARALLEL:-0}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-42}"
EXTRA_TRAIN="${EXTRA_TRAIN:-}"

# Derived paths
DATA_ROOT="${INSCENE_BASE}/data"
STREAM_ROOT_BASE="${INSCENE_BASE}/FEAT_STREAMING"
TARGET_ROOT_BASE="${INSCENE_BASE}/FEAT_TARGET"

# Project root (this script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

# ── VFM metadata ──────────────────────────────────────────────────────────────

vfm_model_id() {
    case "$1" in
        wan)        printf 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers' ;;
        vjepa2)     printf 'facebook/vjepa2-vitl-fpc64-256' ;;
        cogvideox)  printf 'THUDM/CogVideoX-5b-I2V' ;;
        *) echo "[err] unknown VFM: $1" >&2; exit 1 ;;
    esac
}

vfm_default_layer() {
    "${PYTHON}" scripts/resolve_feature_layers.py --vfm "$1" --field default_layer 2>/dev/null \
        || case "$1" in wan|cogvideox) printf '20';; vjepa2) printf '23';; esac
}

vfm_in_channels() {
    "${PYTHON}" scripts/resolve_feature_layers.py --vfm "$1" --field in_channels 2>/dev/null \
        || case "$1" in wan) printf '1536';; cogvideox) printf '3072';; vjepa2) printf '1024';; esac
}

vfm_timestep() {
    # vjepa2 is an encoder and ignores T, but we keep 749 to match yaml defaults
    printf '749'
}

vfm_gpu() {
    case "$1" in
        wan)       printf '%s' "${GPU_WAN}" ;;
        vjepa2)    printf '%s' "${GPU_VJEPA2}" ;;
        cogvideox) printf '%s' "${GPU_WAN}" ;;  # override GPU_COGVIDEOX if needed
        *)         printf '0' ;;
    esac
}

# ── Helpers ───────────────────────────────────────────────────────────────────

run_cmd() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

safe_name() {
    local v="$1"; v="${v// /_}"; v="${v//,/}"; printf '%s' "${v}"
}

probe_needs_target_cache() {
    [[ "$1" == "action_dynamics" || "$1" == "path_integration" || "$1" == "counterfactual" ]]
}

probe_uses_shared_hidden_object() {
    [[ "$1" == "ego_belief" || "$1" == "ego_belief_v2" ]]
}

# ── Step 1: Data download ─────────────────────────────────────────────────────

step_download() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo " STEP 1: Download InsScene-15K dataset"
    echo "         Output: ${INSCENE_BASE}/data/"
    echo "════════════════════════════════════════════════════════"

    if [[ -d "${DATA_ROOT}/processed_infinigen" && -d "${DATA_ROOT}/processed_scannetpp_v2" ]]; then
        echo "[skip] Data already present at ${DATA_ROOT}"
        return
    fi

    run_cmd "${PYTHON}" data/download_inscene15k.py --step all --base-dir "${INSCENE_BASE}"
}

# ── Step 2: Feature extraction ────────────────────────────────────────────────

step_extract() {
    local vfm="$1"
    local model_id layer T
    model_id="$(vfm_model_id "${vfm}")"
    layer="$(vfm_default_layer "${vfm}")"
    T="$(vfm_timestep "${vfm}")"
    local gpu
    gpu="$(vfm_gpu "${vfm}")"

    # Derive streaming feat root: separate dir per VFM
    local prefix_name
    prefix_name="prefix_$(safe_name "${PREFIX_LENGTHS}")"
    local stream_root="${STREAM_ROOT_BASE}/${vfm}_${prefix_name}"
    local target_root="${TARGET_ROOT_BASE}/${vfm}"

    local max_prefix=0
    for p in ${PREFIX_LENGTHS}; do
        (( p > max_prefix )) && max_prefix="${p}"
    done

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo " STEP 2a [${vfm}] Extract streaming-prefix features"
    echo "         GPU ${gpu} | layer ${layer} | max_prefix ${max_prefix}"
    echo "         Output: ${stream_root}"
    echo "════════════════════════════════════════════════════════"
    run_cmd env CUDA_VISIBLE_DEVICES="${gpu}" \
        "${PYTHON}" -m features.run_inscene15k \
        --vfm "${vfm}" \
        --model-id "${model_id}" \
        --mode streaming_prefix \
        --data-root "${DATA_ROOT}" \
        --out-root "${stream_root}" \
        --t "${T}" \
        --output-layers "${layer}" \
        --prefix-min-len 1 \
        --prefix-max-len "${max_prefix}" \
        --prefix-lengths "$(printf '%s' "${PREFIX_LENGTHS}" | tr ' ' ',')"

    # Only extract target cache if C probes are in the probe list
    local needs_targets=0
    for probe in ${PROBES}; do
        if probe_needs_target_cache "${probe}"; then needs_targets=1; fi
    done

    if (( needs_targets )); then
        echo ""
        echo "════════════════════════════════════════════════════════"
        echo " STEP 2b [${vfm}] Extract target-isolated features (C probes)"
        echo "         GPU ${gpu} | layer ${layer}"
        echo "         Output: ${target_root}"
        echo "════════════════════════════════════════════════════════"
        run_cmd env CUDA_VISIBLE_DEVICES="${gpu}" \
            "${PYTHON}" -m features.run_inscene15k \
            --vfm "${vfm}" \
            --model-id "${model_id}" \
            --mode target_isolated \
            --data-root "${DATA_ROOT}" \
            --out-root "${target_root}" \
            --t "${T}" \
            --output-layers "${layer}" \
            --num-targets 0
    fi
}

# ── Step 3/4: Train + Eval ────────────────────────────────────────────────────

step_train_eval() {
    local vfm="$1"
    local layer T video_channels gpu
    layer="$(vfm_default_layer "${vfm}")"
    T="$(vfm_timestep "${vfm}")"
    video_channels="$(vfm_in_channels "${vfm}")"
    gpu="$(vfm_gpu "${vfm}")"

    local prefix_name
    prefix_name="prefix_$(safe_name "${PREFIX_LENGTHS}")"
    local stream_root="${STREAM_ROOT_BASE}/${vfm}_${prefix_name}"
    local target_root="${TARGET_ROOT_BASE}/${vfm}"
    local hidden_prefix_list="[$(printf '%s' "${PREFIX_LENGTHS}" | tr ' ' ',')]"

    local layer_name
    layer_name="$(safe_name "${layer}")"

    local extra_train_args=()
    if [[ -n "${EXTRA_TRAIN}" ]]; then
        read -r -a extra_train_args <<< "${EXTRA_TRAIN}"
    fi

    for probe in ${PROBES}; do
        local cfg="inscene15k_streaming/${probe}_${vfm}_v1"
        local cfg_file="${PROJECT_ROOT}/configs/experiment/${cfg}.yaml"
        if [[ ! -f "${cfg_file}" ]]; then
            echo "[warn] missing config, skip probe=${probe} vfm=${vfm}: ${cfg_file}" >&2
            continue
        fi

        for prefix_len in ${PREFIX_LENGTHS}; do
            for seed in ${SEEDS}; do
                local job="${probe}_${vfm}_p${prefix_len}_layer${layer_name}_s${seed}"
                local run_name="inscene15k_streaming_${job}"
                local run_dir="${PROJECT_ROOT}/logs/inscene15k_streaming/runs/${run_name}"
                local ckpt="${run_dir}/checkpoints/last.ckpt"

                # ── Train ──
                if [[ "${TRAIN}" == "1" ]]; then
                    echo ""
                    echo "── train  probe=${probe}  vfm=${vfm}  prefix=${prefix_len}  layer=${layer}  seed=${seed} ──"
                    local train_cmd=(
                        env CUDA_VISIBLE_DEVICES="${gpu}"
                        "${PYTHON}" vidfm3d/train.py
                        "experiment=${cfg}"
                        "vfm_name=${vfm}"
                        "video_channels=${video_channels}"
                        "streaming_feat_root=${stream_root}"
                        "feature_layer=${layer}"
                        "feature_timestep=${T}"
                        "prefix_len=${prefix_len}"
                        "job_name=${job}"
                        "paths.run_folder_name=${run_name}"
                        "logger.wandb.name=${run_name}"
                        "seed=${seed}"
                    )
                    if probe_uses_shared_hidden_object "${probe}"; then
                        train_cmd+=("streaming_hidden_prefix_lengths=${hidden_prefix_list}")
                    fi
                    if probe_needs_target_cache "${probe}"; then
                        train_cmd+=("target_feat_root=${target_root}")
                    fi
                    train_cmd+=("${extra_train_args[@]}")
                    run_cmd "${train_cmd[@]}"
                fi

                # ── Eval ──
                if [[ "${EVAL}" == "1" && "${probe}" != "streaming_depth" ]]; then
                    if [[ "${DRY_RUN}" != "1" && ! -f "${ckpt}" ]]; then
                        echo "[warn] checkpoint not found, skip eval: ${ckpt}" >&2
                        continue
                    fi
                    echo ""
                    echo "── eval   probe=${probe}  vfm=${vfm}  prefix=${prefix_len}  layer=${layer}  seed=${seed} ──"
                    local eval_cmd=(
                        "${PYTHON}" vidfm3d/eval_diag.py
                        "experiment=${cfg}"
                        "vfm_name=${vfm}"
                        "video_channels=${video_channels}"
                        "streaming_feat_root=${stream_root}"
                        "feature_layer=${layer}"
                        "feature_timestep=${T}"
                        "prefix_len=${prefix_len}"
                        "job_name=${job}"
                        "paths.run_folder_name=${run_name}"
                        "ckpt_path=${ckpt}"
                        "+eval_split=val"
                        "train=false"
                        "test=false"
                        "seed=${seed}"
                    )
                    if probe_uses_shared_hidden_object "${probe}"; then
                        eval_cmd+=("streaming_hidden_prefix_lengths=${hidden_prefix_list}")
                    fi
                    if probe_needs_target_cache "${probe}"; then
                        eval_cmd+=("target_feat_root=${target_root}")
                    fi
                    run_cmd "${eval_cmd[@]}"
                fi

            done  # seeds
        done  # prefix_lengths
    done  # probes
}

# ── Per-VFM pipeline ──────────────────────────────────────────────────────────

run_vfm() {
    local vfm="$1"
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "  VFM: ${vfm}   GPU: $(vfm_gpu "${vfm}")   layer: $(vfm_default_layer "${vfm}")"
    echo "╚══════════════════════════════════════════════════════╝"

    cd "${PROJECT_ROOT}"

    if [[ "${EXTRACT}" == "1" ]]; then
        step_extract "${vfm}"
    fi

    if [[ "${TRAIN}" == "1" || "${EVAL}" == "1" ]]; then
        step_train_eval "${vfm}"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo "══════════════════════════════════════════════════════════"
echo " probe_spatial — Streaming Reproduction Pipeline"
echo "══════════════════════════════════════════════════════════"
echo " INSCENE_BASE   : ${INSCENE_BASE}"
echo " VFMS           : ${VFMS}"
echo " PROBES         : ${PROBES}"
echo " PREFIX_LENGTHS : ${PREFIX_LENGTHS}"
echo " SEEDS          : ${SEEDS}"
echo " DOWNLOAD       : ${DOWNLOAD}"
echo " EXTRACT        : ${EXTRACT}"
echo " TRAIN          : ${TRAIN}"
echo " EVAL           : ${EVAL}"
echo " PARALLEL       : ${PARALLEL}"
echo " DRY_RUN        : ${DRY_RUN}"
echo "══════════════════════════════════════════════════════════"

# Step 1: Data download (once, not per-VFM)
if [[ "${DOWNLOAD}" == "1" ]]; then
    cd "${PROJECT_ROOT}"
    step_download
fi

# Step 2-4: Feature extraction + training + eval, per VFM
if [[ "${PARALLEL}" == "1" ]]; then
    pids=()
    for vfm in ${VFMS}; do
        run_vfm "${vfm}" &
        pids+=($!)
        echo "[parallel] launched ${vfm} (PID $!)"
    done
    all_ok=1
    for pid in "${pids[@]}"; do
        wait "${pid}" || { echo "[FAIL] background job PID=${pid}"; all_ok=0; }
    done
    [[ "${all_ok}" == "1" ]] || exit 1
else
    for vfm in ${VFMS}; do
        run_vfm "${vfm}"
    done
fi

echo ""
echo "══════════════════════════════════════════════════════════"
echo " All done."
echo "══════════════════════════════════════════════════════════"
