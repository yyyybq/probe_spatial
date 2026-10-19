#!/usr/bin/env bash
# Benchmark C-series streaming training against AOSS-backed features.
#
# This wrapper intentionally keeps metrics collection non-fatal: the process
# exit code is the training exit code, so scheduler state reflects training.
set -euo pipefail

PYTHON=${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
MANIFEST=${INSCENE_MANIFEST:?set INSCENE_MANIFEST to a local JSONL manifest}
OUT_ROOT=${OUT_ROOT:-/mnt/local_cache/inscene/benchmarks}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
PROCESSES=${PROCESSES:-1}
GPU_IDS=${GPU_IDS:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
BATCHES=${BATCHES:-500}
VFM=${VFM:-dino}
LAYER=${LAYER:-0}
PREFIX_LEN=${PREFIX_LEN:-8}
PROBE=${PROBE:-action_dynamics}
STREAM_ROOT_BASE=${STREAM_ROOT_BASE:-${INSCENE_STREAMING_FEAT_ROOT:?set STREAM_ROOT_BASE or INSCENE_STREAMING_FEAT_ROOT}}
TARGET_FEAT_ROOT=${TARGET_FEAT_ROOT:-${INSCENE_TARGET_FEAT_ROOT:?set TARGET_FEAT_ROOT or INSCENE_TARGET_FEAT_ROOT}}
DATA_ROOT=${DATA_ROOT:-${INSCENE_DATA_ROOT:?set DATA_ROOT or INSCENE_DATA_ROOT}}
STREAMING_MOTION_STEP=${STREAMING_MOTION_STEP:-0.35}
STREAMING_ROTATION_WEIGHT=${STREAMING_ROTATION_WEIGHT:-0.5}
motion_tag="${STREAMING_MOTION_STEP//./p}"
rotation_tag="${STREAMING_ROTATION_WEIGHT//./p}"
STREAMING_FEAT_ROOT="${STREAM_ROOT_BASE}/${VFM}_prefix_${PREFIX_LEN}_m${motion_tag}_r${rotation_tag}"

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export INSCENE_MANIFEST="${MANIFEST}"
export INSCENE_DATA_ROOT="${DATA_ROOT}"
export INSCENE_STREAMING_FEAT_ROOT="${STREAMING_FEAT_ROOT}"
export INSCENE_TARGET_FEAT_ROOT="${TARGET_FEAT_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"

mkdir -p "${OUT_ROOT}" "${MPLCONFIGDIR}"
run_dir="${OUT_ROOT}/${RUN_TAG}"
mkdir -p "${run_dir}"

IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"

manifest_sha="missing"
if [[ -f "${MANIFEST}" ]]; then
    manifest_sha="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
fi

cat > "${run_dir}/conditions.json" <<JSON
{
  "run_tag": "${RUN_TAG}",
  "started_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "project_root": "${PROJECT_ROOT}",
  "manifest": "${MANIFEST}",
  "manifest_sha256": "${manifest_sha}",
  "processes": ${PROCESSES},
  "gpu_ids": "${GPU_IDS}",
  "num_workers_per_process": ${NUM_WORKERS},
  "batches": ${BATCHES},
  "vfm": "${VFM}",
  "layer": "${LAYER}",
  "prefix_len": ${PREFIX_LEN},
  "probe": "${PROBE}",
  "streaming_feat_root": "${STREAMING_FEAT_ROOT}",
  "target_feat_root": "${TARGET_FEAT_ROOT}",
  "cache_note": "${CACHE_NOTE:-unspecified}",
  "sample_seed": "${SAMPLE_SEED:-default}",
  "other_aoss_jobs": "${OTHER_AOSS_JOBS:-unknown}"
}
JSON

monitor_gpu() {
    local out="$1"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found" > "${out}"
        return 0
    fi
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits \
        -l 2 > "${out}" 2>&1
}

summarize_run() {
    local train_rc="$1"
    local stats_rc=0
    "${PYTHON}" - "${run_dir}" "${train_rc}" <<'PY' || stats_rc=$?
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
train_rc = int(sys.argv[2])
it_pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)it/s")
bad_pattern = re.compile(r"(timeout|retry|traceback|worker.*exit|dataloader worker|error)", re.I)
progress_pattern = re.compile(
    r"\[(?P<ts>[0-9-]+ [0-9:,]+)\].*event=train_batch_end .*global_step=(?P<step>[0-9]+)"
)
ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")

def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac

process_summaries = []
total_batches = 0
for log_path in sorted(run_dir.glob("proc_*.log")):
    text = log_path.read_text(errors="replace")
    plain_text = ansi_pattern.sub("", text)
    rates = [float(m.group(1)) for m in it_pattern.finditer(text)]
    limit_match = re.search(r"limit_train_batches[= ]([0-9]+)", text)
    total_batches += int(limit_match.group(1)) if limit_match else 0
    progress = []
    for match in progress_pattern.finditer(plain_text):
        ts = dt.datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
        progress.append((int(match.group("step")), ts))
    step_seconds = []
    for (prev_step, prev_ts), (step, ts) in zip(progress, progress[1:]):
        delta_steps = step - prev_step
        if delta_steps > 0:
            step_seconds.append((ts - prev_ts).total_seconds() / delta_steps)
    process_summaries.append({
        "log": log_path.name,
        "it_s_count": len(rates),
        "it_s_last": rates[-1] if rates else None,
        "it_s_mean": sum(rates) / len(rates) if rates else None,
        "step_seconds_samples": len(step_seconds),
        "step_seconds_p50": percentile(step_seconds, 0.50),
        "step_seconds_p95": percentile(step_seconds, 0.95),
        "step_seconds_p99": percentile(step_seconds, 0.99),
        "keywords": sorted(set(m.group(1).lower() for m in bad_pattern.finditer(text))),
    })

peak_mem = {}
gpu_csv = run_dir / "gpu.csv"
if gpu_csv.exists():
    with gpu_csv.open(errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                gpu = row[1].strip()
                mem_used = float(row[3])
            except ValueError:
                continue
            peak_mem[gpu] = max(peak_mem.get(gpu, 0.0), mem_used)

summary = {
    "train_exit_code": train_rc,
    "processes": process_summaries,
    "peak_gpu_memory_mb": peak_mem,
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
    if [[ "${stats_rc}" != "0" ]]; then
        echo "Warning: benchmark statistics collection failed with rc=${stats_rc}" >&2
    fi
}

pids=()
rcs=()
monitor_gpu "${run_dir}/gpu.csv" &
gpu_monitor_pid=$!

for ((i = 0; i < PROCESSES; i++)); do
    gpu="${gpu_ids[$((i % ${#gpu_ids[@]}))]}"
    proc_dir="${run_dir}/proc_${i}"
    mkdir -p "${proc_dir}/aoss_cache"
    log="${run_dir}/proc_${i}.log"
    run_name="bench_${RUN_TAG}_p${i}_${PROBE}_${VFM}_gpu${gpu}"
    extra_train="autoresume=false trainer.max_epochs=1 +trainer.limit_train_batches=${BATCHES} +trainer.limit_val_batches=0 +trainer.num_sanity_val_steps=0 data.data_module.num_workers=${NUM_WORKERS} data.data_module.num_workers_val=0 logger.wandb.offline=true job_name=${run_name} paths.run_folder_name=${run_name} logger.wandb.name=${run_name}"
    (
        set +e
        export AOSS_CACHE_DIR="${proc_dir}/aoss_cache"
        echo "BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) proc=${i} gpu=${gpu} limit_train_batches=${BATCHES}"
        env \
            PYTHON="${PYTHON}" \
            DEV="${gpu}" \
            VFM="${VFM}" \
            VFM_NAME="${VFM}" \
            PROBES="${PROBE}" \
            PREFIX_LENGTHS="${PREFIX_LEN}" \
            C_PREFIX_LENGTHS="${PREFIX_LEN}" \
            LAYERS="${LAYER}" \
            TRAIN=1 \
            EVAL=0 \
            EXTRACT_STREAMING=0 \
            EXTRACT_TARGETS=0 \
            TARGET_FEAT_ROOT="${TARGET_FEAT_ROOT}" \
            STREAM_ROOT_BASE="${STREAM_ROOT_BASE}" \
            EXTRA_TRAIN="${extra_train}" \
            bash scripts/run_streaming_probe_sweep.sh
        rc=$?
        echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ) proc=${i} rc=${rc}"
        exit "${rc}"
    ) > "${log}" 2>&1 &
    pids+=("$!")
done

train_rc=0
for pid in "${pids[@]}"; do
    if wait "${pid}"; then
        rcs+=("0")
    else
        rc=$?
        rcs+=("${rc}")
        train_rc="${rc}"
    fi
done

kill "${gpu_monitor_pid}" >/dev/null 2>&1 || true
wait "${gpu_monitor_pid}" >/dev/null 2>&1 || true

printf '%s\n' "${rcs[@]}" > "${run_dir}/process_exit_codes.txt"
ps -eo pid,ppid,stat,comm,args | grep -E 'vidfm3d/train.py|DataLoader|aoss' | grep -v grep > "${run_dir}/residual_processes.txt" || true
summarize_run "${train_rc}" || true
exit "${train_rc}"
