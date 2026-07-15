#!/usr/bin/env bash
# ==============================================================================
# reproduce_streaming.sh
# Full reproduction pipeline: data → features (multi-layer) → probing
# Sweep dimensions: VFM × layer × probe × prefix_len
#
# ── Quick start ───────────────────────────────────────────────────────────────
# Default (wan + vjepa2, default layer each, all streaming probes):
#   INSCENE_BASE=/data/InsScene-15K bash scripts/reproduce_streaming.sh
#
# Layer sweep — same layers for all VFMs:
#   INSCENE_BASE=/data/InsScene-15K LAYERS="0 5 10 15 20 25 29" \
#     VFMS="wan" bash scripts/reproduce_streaming.sh
#
# Layer sweep — per-VFM layer lists:
#   INSCENE_BASE=/data/InsScene-15K \
#     LAYERS_WAN="0 5 10 15 20 25 29" \
#     LAYERS_VJEPA2="0 5 11 17 23" \
#     bash scripts/reproduce_streaming.sh
#
# Dry-run to preview all commands:
#   DRY_RUN=1 INSCENE_BASE=/data/InsScene-15K bash scripts/reproduce_streaming.sh
#
# ── Required ──────────────────────────────────────────────────────────────────
#   INSCENE_BASE      Root of InsScene-15K. Temporal streaming reproduction is
#                     ScanNet++ only:
#                       ${INSCENE_BASE}/data/processed_scannetpp_v2/
#
# ── Step flags ────────────────────────────────────────────────────────────────
#   DOWNLOAD   1=download dataset (default: 1)
#   EXTRACT    1=extract features (default: 1)
#   TRAIN      1=train probes     (default: 1)
#   EVAL       1=eval checkpoints (default: 1)
#
# ── Model + layer config ──────────────────────────────────────────────────────
#   VFMS          Space-separated VFM names      (default: "wan vjepa2")
#                 Supported: wan | vjepa2 | cogvideox
#
#   LAYERS        Layers for all VFMs            (default: "default")
#                 "default"  → registered default for each VFM
#                              wan=20, vjepa2=23, cogvideox=20
#                 "0 5 10 20"→ these exact layers for all VFMs
#                 Per-model overrides always take priority (see below).
#
#   LAYERS_WAN         Layer list for wan        (overrides LAYERS for wan)
#   LAYERS_VJEPA2      Layer list for vjepa2     (overrides LAYERS for vjepa2)
#   LAYERS_COGVIDEOX   Layer list for cogvideox  (overrides LAYERS for cogvideox)
#
#   Example: extract every 5th wan layer + vjepa2 default
#     LAYERS_WAN="0 5 10 15 20 25 29" LAYERS_VJEPA2="default" ...
#
# ── GPU config ────────────────────────────────────────────────────────────────
#   GPU_DEFAULT   Fallback GPU for all VFMs      (default: 0)
#   GPU_WAN       GPU for wan                    (default: GPU_DEFAULT)
#   GPU_VJEPA2    GPU for vjepa2                 (default: GPU_DEFAULT)
#   GPU_COGVIDEOX GPU for cogvideox              (default: GPU_DEFAULT)
#
# ── Probe config ──────────────────────────────────────────────────────────────
#   PROBES          Space-separated probe names  (default: all 6 streaming probes)
#                     streaming_depth → A1 Depth & Camera
#                     view_consistency → A2 View Consistency
#                     ego_belief_v2   → B2 Object-Query Localization
#                     action_dynamics → C1 Latent Dynamics
#                     path_integration→ C2 Path Integration
#                     counterfactual  → C3 Counterfactual Action
#   PREFIX_LENGTHS  Space-separated prefix lengths (default: "8 12 16 24")
#   C_TARGET_HORIZONS
#                   Space-separated C target horizons (default: "1 2 4")
#
# ── Other ─────────────────────────────────────────────────────────────────────
#   PARALLEL     1=run VFMs in parallel, 0=sequential (default: 0)
#   DRY_RUN      1=print commands only                (default: 0)
#   PYTHON       Python executable                    (default: python)
#   EXTRA_TRAIN  Extra hydra overrides for training   (default: "")
# ==============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

INSCENE_BASE="${INSCENE_BASE:?Please set INSCENE_BASE=/path/to/InsScene-15K}"

VFMS="${VFMS:-wan vjepa2 cogvideox}"
PROBES="${PROBES:-streaming_depth view_consistency ego_belief_v2 action_dynamics path_integration counterfactual}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-8 12 16 24}"
C_TARGET_HORIZONS="${C_TARGET_HORIZONS:-1 2 4}"
STREAMING_MOTION_STEP="${STREAMING_MOTION_STEP:-0.35}"
STREAMING_ROTATION_WEIGHT="${STREAMING_ROTATION_WEIGHT:-0.5}"

# Global layer default; per-model variables (LAYERS_WAN etc.) override this.
LAYERS="${LAYERS:-default}"

# GPU assignments
GPU_DEFAULT="${GPU_DEFAULT:-0}"
GPU_WAN="${GPU_WAN:-${GPU_DEFAULT}}"
GPU_VJEPA2="${GPU_VJEPA2:-${GPU_DEFAULT}}"
GPU_COGVIDEOX="${GPU_COGVIDEOX:-${GPU_DEFAULT}}"

DOWNLOAD="${DOWNLOAD:-1}"
EXTRACT="${EXTRACT:-1}"
TRAIN="${TRAIN:-1}"
EVAL="${EVAL:-1}"
PARALLEL="${PARALLEL:-0}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON="${PYTHON:-python}"
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

vfm_in_channels() {
    "${PYTHON}" scripts/resolve_feature_layers.py --vfm "$1" --field in_channels 2>/dev/null \
        || case "$1" in wan) printf '1536';; cogvideox) printf '3072';; vjepa2) printf '1024';; esac
}

vfm_timestep() {
    # vjepa2 is an encoder and ignores T; keep 749 to match yaml defaults for all
    printf '749'
}

vfm_gpu() {
    local upper
    upper="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]' | tr '-' '_')"
    # Evaluate GPU_<UPPER_VFM> if set, else GPU_DEFAULT
    local var="GPU_${upper}"
    printf '%s' "${!var:-${GPU_DEFAULT}}"
}

# Resolve layer token: "default" → registered default for the VFM,
# a number → that number, space-separated list → each entry resolved.
# Returns space-separated list of integer layer ids.
resolve_layers_for_vfm() {
    local vfm="$1"
    # Per-model override: LAYERS_WAN, LAYERS_VJEPA2, LAYERS_COGVIDEOX
    local upper
    upper="$(printf '%s' "${vfm}" | tr '[:lower:]' '[:upper:]' | tr '-' '_')"
    local per_model_var="LAYERS_${upper}"
    local raw="${!per_model_var:-${LAYERS}}"

    local result=()
    for token in ${raw}; do
        case "${token}" in
            default)
                local dl
                dl="$("${PYTHON}" scripts/resolve_feature_layers.py --vfm "${vfm}" \
                        --field default_layer 2>/dev/null)" \
                    || dl="$(case "${vfm}" in wan|cogvideox) printf '20';; vjepa2) printf '23';; *) printf '0';; esac)"
                result+=("${dl}")
                ;;
            last)
                local ll
                ll="$("${PYTHON}" scripts/resolve_feature_layers.py --vfm "${vfm}" \
                        --field last_layer 2>/dev/null)" || ll="default"
                result+=("${ll}")
                ;;
            *)
                # Numeric literal
                result+=("${token}")
                ;;
        esac
    done

    # Deduplicate while preserving order
    local seen=()
    local unique=()
    for l in "${result[@]}"; do
        local found=0
        for s in "${seen[@]+"${seen[@]}"}"; do [[ "${s}" == "${l}" ]] && found=1; done
        if (( ! found )); then unique+=("${l}"); seen+=("${l}"); fi
    done

    printf '%s' "${unique[*]}"
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

    if [[ -d "${DATA_ROOT}/processed_scannetpp_v2" ]]; then
        echo "[skip] Data already present at ${DATA_ROOT}"
        return
    fi

    run_cmd "${PYTHON}" data/download_inscene15k.py --step download_scannetpp --base-dir "${INSCENE_BASE}"
    run_cmd "${PYTHON}" data/download_inscene15k.py --step extract_scannetpp --base-dir "${INSCENE_BASE}"
}

# ── Step 2: Feature extraction ────────────────────────────────────────────────
# All requested layers are extracted in a single model-forward pass per scene.
# streaming_prefix and target_isolated share the same layer list.

step_extract() {
    local vfm="$1"
    local model_id T gpu
    model_id="$(vfm_model_id "${vfm}")"
    T="$(vfm_timestep "${vfm}")"
    gpu="$(vfm_gpu "${vfm}")"

    # Resolve all layers for this VFM (space-separated)
    local layers_str
    layers_str="$(resolve_layers_for_vfm "${vfm}")"

    local prefix_name
    prefix_name="prefix_$(safe_name "${PREFIX_LENGTHS}")"
    local motion_tag rotation_tag
    motion_tag="${STREAMING_MOTION_STEP//./p}"
    rotation_tag="${STREAMING_ROTATION_WEIGHT//./p}"
    local stream_root="${STREAM_ROOT_BASE}/${vfm}_${prefix_name}_m${motion_tag}_r${rotation_tag}"
    local target_root="${TARGET_ROOT_BASE}/${vfm}"

    local max_prefix=0
    for p in ${PREFIX_LENGTHS}; do (( p > max_prefix )) && max_prefix="${p}"; done

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo " STEP 2a [${vfm}] Extract streaming-prefix features"
    echo "         GPU ${gpu} | layers: ${layers_str} | max_prefix ${max_prefix}"
    echo "         Output: ${stream_root}"
    echo "         (all layers extracted in one pass)"
    echo "════════════════════════════════════════════════════════"
    # shellcheck disable=SC2086
    run_cmd env CUDA_VISIBLE_DEVICES="${gpu}" \
        "${PYTHON}" -m features.run_inscene15k \
        --vfm "${vfm}" \
        --model-id "${model_id}" \
        --mode streaming_prefix \
        --data-root "${DATA_ROOT}" \
        --out-root "${stream_root}" \
        --t "${T}" \
        --output-layers ${layers_str} \
        --source scannetpp \
        --prefix-min-len 1 \
        --prefix-max-len "${max_prefix}" \
        --prefix-lengths "$(printf '%s' "${PREFIX_LENGTHS}" | tr ' ' ',')" \
        --streaming-motion-step "${STREAMING_MOTION_STEP}" \
        --streaming-rotation-weight "${STREAMING_ROTATION_WEIGHT}"

    # Extract target cache only if C probes are requested
    local needs_targets=0
    for probe in ${PROBES}; do
        if probe_needs_target_cache "${probe}"; then needs_targets=1; fi
    done

    if (( needs_targets )); then
        echo ""
        echo "════════════════════════════════════════════════════════"
        echo " STEP 2b [${vfm}] Extract target-isolated features (C probes)"
        echo "         GPU ${gpu} | layers: ${layers_str}"
        echo "         Output: ${target_root}"
        echo "════════════════════════════════════════════════════════"
        # shellcheck disable=SC2086
        run_cmd env CUDA_VISIBLE_DEVICES="${gpu}" \
            "${PYTHON}" -m features.run_inscene15k \
            --vfm "${vfm}" \
            --model-id "${model_id}" \
            --mode target_isolated \
            --data-root "${DATA_ROOT}" \
            --out-root "${target_root}" \
            --t "${T}" \
            --output-layers ${layers_str} \
            --source scannetpp \
            --prefix-max-len "${max_prefix}" \
            --prefix-lengths "$(printf '%s' "${PREFIX_LENGTHS}" | tr ' ' ',')" \
            --streaming-motion-step "${STREAMING_MOTION_STEP}" \
            --streaming-rotation-weight "${STREAMING_ROTATION_WEIGHT}" \
            --num-targets 0 \
            --target-from-streaming-windows \
            --target-prefix-lengths "$(printf '%s' "${PREFIX_LENGTHS}" | tr ' ' ',')" \
            --target-horizons "$(printf '%s' "${C_TARGET_HORIZONS}" | tr ' ' ',')"
    fi
}

# ── Step 3/4: Train + Eval — sweep (probe × prefix_len × layer) ───────────────

step_train_eval() {
    local vfm="$1"
    local T video_channels gpu
    T="$(vfm_timestep "${vfm}")"
    video_channels="$(vfm_in_channels "${vfm}")"
    gpu="$(vfm_gpu "${vfm}")"

    local prefix_name
    prefix_name="prefix_$(safe_name "${PREFIX_LENGTHS}")"
    local motion_tag rotation_tag
    motion_tag="${STREAMING_MOTION_STEP//./p}"
    rotation_tag="${STREAMING_ROTATION_WEIGHT//./p}"
    local stream_root="${STREAM_ROOT_BASE}/${vfm}_${prefix_name}_m${motion_tag}_r${rotation_tag}"
    local target_root="${TARGET_ROOT_BASE}/${vfm}"
    local hidden_prefix_list="[$(printf '%s' "${PREFIX_LENGTHS}" | tr ' ' ',')]"

    # Resolve layer list for this VFM
    local layers_str
    layers_str="$(resolve_layers_for_vfm "${vfm}")"

    local extra_train_args=()
    if [[ -n "${EXTRA_TRAIN}" ]]; then
        read -r -a extra_train_args <<< "${EXTRA_TRAIN}"
    fi

    for probe in ${PROBES}; do
        local cfg="inscene15k_streaming/${probe}_${vfm}_v1"
        local cfg_file="${PROJECT_ROOT}/configs/experiment/${cfg}.yaml"
        if [[ ! -f "${cfg_file}" ]]; then
            echo "[warn] missing config, skipping probe=${probe} vfm=${vfm}: ${cfg_file}" >&2
            continue
        fi

        for layer in ${layers_str}; do
            local layer_name
            layer_name="$(safe_name "${layer}")"

            for prefix_len in ${PREFIX_LENGTHS}; do
                local job="${probe}_${vfm}_p${prefix_len}_layer${layer_name}"
                local run_name="inscene15k_streaming_${job}"
                local run_dir="${PROJECT_ROOT}/logs/inscene15k_streaming/runs/${run_name}"
                local ckpt="${run_dir}/checkpoints/last.ckpt"

                # ── Train ──
                if [[ "${TRAIN}" == "1" ]]; then
                    echo ""
                    echo "── train  probe=${probe}  vfm=${vfm}  layer=${layer}  prefix=${prefix_len} ──"
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
                    echo "── eval   probe=${probe}  vfm=${vfm}  layer=${layer}  prefix=${prefix_len} ──"
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
                    )
                    if probe_uses_shared_hidden_object "${probe}"; then
                        eval_cmd+=("streaming_hidden_prefix_lengths=${hidden_prefix_list}")
                    fi
                    if probe_needs_target_cache "${probe}"; then
                        eval_cmd+=("target_feat_root=${target_root}")
                    fi
                    run_cmd "${eval_cmd[@]}"
                fi

            done  # prefix_lengths
        done  # layers
    done  # probes
}

# ── Per-VFM pipeline ──────────────────────────────────────────────────────────

run_vfm() {
    local vfm="$1"
    local layers_str gpu
    layers_str="$(resolve_layers_for_vfm "${vfm}")"
    gpu="$(vfm_gpu "${vfm}")"
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    printf "  VFM: %-10s  GPU: %s  layers: %s\n" "${vfm}" "${gpu}" "${layers_str}"
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
echo " C_TARGET_HORIZONS: ${C_TARGET_HORIZONS}"
echo " MOTION_STEP    : ${STREAMING_MOTION_STEP}"
echo " ROTATION_WEIGHT: ${STREAMING_ROTATION_WEIGHT}"
echo " LAYERS         : ${LAYERS}"
for vfm in ${VFMS}; do
    upper="$(printf '%s' "${vfm}" | tr '[:lower:]' '[:upper:]' | tr '-' '_')"
    var="LAYERS_${upper}"
    [[ -n "${!var:-}" ]] && echo "   ${var}      : ${!var}"
done
echo " GPU_DEFAULT    : ${GPU_DEFAULT}"
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

# Steps 2-4: Feature extraction + training + eval, per VFM
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
