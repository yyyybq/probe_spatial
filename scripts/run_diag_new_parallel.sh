#!/usr/bin/env bash
# Legacy non-streaming C2/C3 probes in parallel across available GPUs.
#   PYTHON=/path/to/python bash scripts/run_diag_new_parallel.sh
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/run_diag_new_parallel.sh
#   GPU_GROUPS="0,1 2,3 4,5 6,7" JOBS="path_integration_wan_v1 ..." bash scripts/run_diag_new_parallel.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
ALLOW_NON_STREAMING=${ALLOW_NON_STREAMING:-0}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
GPU_GROUPS=${GPU_GROUPS:-}
LOG_ROOT=${LOG_ROOT:-logs/parallel_new_probes}
EXTRA=${EXTRA:-"logger.wandb.offline=true"}
JOBS=${JOBS:-}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29550}

if [[ "${ALLOW_NON_STREAMING}" != "1" ]]; then
    cat >&2 <<'EOF'
[err] run_diag_new_parallel.sh is a legacy non-streaming sweep.
      Streaming is the project default. Use:
        bash scripts/run_streaming_probe_sweep.sh
      To intentionally run this old path, set ALLOW_NON_STREAMING=1.
EOF
    exit 2
fi

export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}" "${LOG_ROOT}"
: > "${LOG_ROOT}/pids.txt"

if [[ -n "${JOBS}" ]]; then
    read -r -a jobs <<< "${JOBS}"
else
    jobs=(
        path_integration_wan_v1
        path_integration_vjepa2_v1
        path_integration_cogvideox_v1
        counterfactual_wan_v1
        counterfactual_vjepa2_v1
        counterfactual_cogvideox_v1
    )
fi

if [[ -n "${GPU_GROUPS}" ]]; then
    read -r -a gpu_list <<< "${GPU_GROUPS}"
else
    read -r -a gpu_list <<< "${GPUS}"
fi
if (( ${#gpu_list[@]} == 0 )); then
    echo "No GPUs listed in GPUS/GPU_GROUPS"
    exit 1
fi

pids=()
for i in "${!jobs[@]}"; do
    job="${jobs[$i]}"
    gpu="${gpu_list[$((i % ${#gpu_list[@]}))]}"
    cfg="inscene15k_ext/${job}"
    log="${LOG_ROOT}/${job}.log"
    cfg_file="configs/experiment/${cfg}.yaml"
    if [[ ! -f "${cfg_file}" ]]; then
        echo "[skip] missing config: ${cfg_file}"
        continue
    fi

    echo "==== launch ${cfg} on GPU(s) ${gpu}; log=${log} ===="
    (
        set -x
        export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
        export MASTER_PORT="$((MASTER_PORT_BASE + i))"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" vidfm3d/train.py \
            "experiment=${cfg}" \
            ${EXTRA}
    ) >"${log}" 2>&1 &
    pids+=("$!")
    echo "${pids[-1]} ${job} ${gpu} ${log}" >> "${LOG_ROOT}/pids.txt"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"
