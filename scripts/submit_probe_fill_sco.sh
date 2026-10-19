#!/usr/bin/env bash
# Prepare missing ABC/A1 jobs and submit SCO shards via ~/submit.sh.
#
# Usage:
#   bash scripts/submit_probe_fill_sco.sh                  # campaign=fill
#   CAMPAIGN=lingbot bash scripts/submit_probe_fill_sco.sh zoetrope
#   CAMPAIGN=all NUM_SHARDS=8 DRY_SUBMIT=1 bash scripts/submit_probe_fill_sco.sh
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
SUBMIT_SH=${SUBMIT_SH:-/mnt/umm/users/yinbaiqiao/submit.sh}
PYTHON=${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
cd "${PROJECT_ROOT}"

aec2_name=${1:-"zoetrope"}
# Optional: comma-separated pair for 50/50 shard split, e.g. zoetrope,h800
AEC2_SPLIT=${AEC2_SPLIT:-}
CAMPAIGN=${CAMPAIGN:-fill}
NUM_SHARDS=${NUM_SHARDS:-0}   # 0 = auto from job count
NPROC=${NPROC:-8}
SLOTS=${SLOTS:-16}
DRY_SUBMIT=${DRY_SUBMIT:-0}
WORKER_NODES=${WORKER_NODES:-1}
JOB_PREFIX=${JOB_PREFIX:-bq_probe_fill}
RUN_ID=${RUN_ID:-probe_fill_${CAMPAIGN}_$(date -u +%Y%m%d_%H%M)}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/logs/probing_jobs_${RUN_ID}}
AUTO_MONITOR=${AUTO_MONITOR:-1}
MONITOR_INTERVAL_SEC=${MONITOR_INTERVAL_SEC:-60}
MONITOR_MAX_HOURS=${MONITOR_MAX_HOURS:-48}

if [[ -n "${AEC2_SPLIT}" ]]; then
  IFS=',' read -r -a AEC2_LIST <<< "${AEC2_SPLIT}"
else
  AEC2_LIST=("${aec2_name}")
fi
if (( ${#AEC2_LIST[@]} < 1 )); then
  echo "[err] empty AEC2 list" >&2
  exit 1
fi

echo "=============================================="
echo " prepare + submit probe fill via submit.sh"
echo " aec2s      : ${AEC2_LIST[*]}"
echo " campaign   : ${CAMPAIGN}"
echo " RUN_ID     : ${RUN_ID}"
echo " OUT_DIR    : ${OUT_DIR}"
echo " DRY_SUBMIT : ${DRY_SUBMIT}"
echo " SUBMIT_SH  : ${SUBMIT_SH}"
echo "=============================================="

if [[ ! -x "${SUBMIT_SH}" ]]; then
  echo "[err] submit helper missing/not executable: ${SUBMIT_SH}" >&2
  exit 2
fi

# 1) Build missing-job TSV + CHECKLIST
"${PYTHON}" scripts/build_probe_fill_jobs.py \
  --campaign "${CAMPAIGN}" \
  --out-dir "${OUT_DIR}"

JOBS_TSV="${OUT_DIR}/jobs/all.tsv"
# subtract header
n_jobs=$(( $(wc -l < "${JOBS_TSV}") - 1 ))
if (( n_jobs < 1 )); then
  echo "[ok] nothing missing for campaign=${CAMPAIGN}; refusing empty submit"
  exit 0
fi

# Auto shard count: ~40-60 jobs/shard on 8GPU*2 slots
if (( NUM_SHARDS < 1 )); then
  NUM_SHARDS=$(( (n_jobs + 49) / 50 ))
  if (( NUM_SHARDS < 1 )); then NUM_SHARDS=1; fi
  if (( NUM_SHARDS > 12 )); then NUM_SHARDS=12; fi
fi
# Keep even shard count when splitting across two clusters.
if (( ${#AEC2_LIST[@]} == 2 )) && (( NUM_SHARDS % 2 == 1 )); then
  NUM_SHARDS=$((NUM_SHARDS + 1))
fi

mkdir -p "${OUT_DIR}/jobs/shards" "${OUT_DIR}/launcher"
rm -f "${OUT_DIR}/jobs/shards"/shard_*.tsv

# Split body only (keep header out of shards); write headerless shard TSVs
# that launch_probe_fill can still read (it skips a header line if present).
body="${OUT_DIR}/jobs/body.tsv"
tail -n +2 "${JOBS_TSV}" > "${body}"
split -d -a 2 -n "l/${NUM_SHARDS}" "${body}" "${OUT_DIR}/jobs/shards/shard_"
for f in "${OUT_DIR}/jobs/shards"/shard_*; do
  [[ -f "${f}" ]] || continue
  [[ "${f}" == *.tsv ]] && continue
  # prepend header for clarity
  {
    head -n 1 "${JOBS_TSV}"
    cat "${f}"
  } > "${f}.tsv"
  rm -f "${f}"
done

mapfile -t SHARDS < <(ls "${OUT_DIR}/jobs/shards"/shard_*.tsv | sort)
echo "created ${#SHARDS[@]} shards for ${n_jobs} jobs:"
for s in "${SHARDS[@]}"; do
  echo "  $(basename "${s}"): $(( $(wc -l < "${s}") - 1 )) jobs"
done

# Per-shard entry wrappers (so submit.sh only needs an entry script path).
mkdir -p "${OUT_DIR}/launcher/entries"
chmod +x scripts/run_probe_fill_shard.sh scripts/launch_probe_fill.sh

idx=0
entry_list=()
for shard in "${SHARDS[@]}"; do
  idx=$((idx + 1))
  shard_abs="$(readlink -f "${shard}")"
  entry="${OUT_DIR}/launcher/entries/run_shard_$(printf '%02d' "${idx}").sh"
  cat > "${entry}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${PROJECT_ROOT}
export OUT_DIR=${OUT_DIR}
export RUN_ID=${RUN_ID}
export SHARD_TSV=${shard_abs}
export SLOTS=${SLOTS}
export GPU_IDS_CSV=0,1,2,3,4,5,6,7
export FEAT_BACKEND=aoss
export ALLOW_MISSING_STREAMING=1
bash ${PROJECT_ROOT}/scripts/run_probe_fill_shard.sh
EOF
  chmod +x "${entry}"
  entry_list+=("${entry}")
done

cat > "${OUT_DIR}/PLAN.txt" <<EOF
Probe fill campaign
===================
campaign : ${CAMPAIGN}
RUN_ID   : ${RUN_ID}
jobs     : ${n_jobs}
shards   : ${#SHARDS[@]} x ${NPROC}GPU (slots=${SLOTS})
aec2s    : ${AEC2_LIST[*]}  (shards split round-robin / half-half)

See CHECKLIST.txt for per-probe missing counts.

Submit:
  AEC2_SPLIT=zoetrope,h800 DRY_SUBMIT=0 bash scripts/submit_probe_fill_sco.sh

Monitor:
  python scripts/monitor_probe_fill.py --run-id ${RUN_ID} --once
  tail -f ${OUT_DIR}/launcher/PROGRESS.txt
EOF

if [[ "${DRY_SUBMIT}" == "1" ]]; then
  echo "DRY_SUBMIT=1: skipping sco create (PLAN/CHECKLIST ready)"
  exit 0
fi

submit_log="${OUT_DIR}/launcher/sco_submit.log"
: > "${submit_log}"
assign_log="${OUT_DIR}/launcher/shard_aec2_assignment.txt"
: > "${assign_log}"
n_aec2=${#AEC2_LIST[@]}
half=$(( (${#entry_list[@]} + n_aec2 - 1) / n_aec2 ))
idx=0
for entry in "${entry_list[@]}"; do
  idx=$((idx + 1))
  # Contiguous half-half when 2 clusters; otherwise round-robin.
  if (( n_aec2 == 2 )); then
    if (( idx <= half )); then
      aec2="${AEC2_LIST[0]}"
    else
      aec2="${AEC2_LIST[1]}"
    fi
  else
    aec2="${AEC2_LIST[$(( (idx - 1) % n_aec2 ))]}"
  fi
  job_name=$(printf '%s_%s_%s_s%02d' "${JOB_PREFIX}" "${CAMPAIGN}" "${aec2}" "${idx}")
  echo "[submit] ${job_name} aec2=${aec2} entry=$(basename "${entry}")" | tee -a "${submit_log}"
  echo "shard=$(printf '%02d' "${idx}") aec2=${aec2} entry=$(basename "${entry}")" >> "${assign_log}"
  DRY_SUBMIT=0 \
    bash "${SUBMIT_SH}" "${aec2}" "${WORKER_NODES}" "${NPROC}" "${job_name}" "${entry}" \
    | tee -a "${submit_log}"
done

job_ids_file="${OUT_DIR}/launcher/sco_job_ids.txt"
grep -oE 'pt-[a-z0-9]+' "${submit_log}" | awk '!seen[$0]++' > "${job_ids_file}" || true
echo "Submitted ${#entry_list[@]} SCO jobs; ids: $(tr '\n' ' ' < "${job_ids_file}")"

if [[ "${AUTO_MONITOR}" == "1" ]]; then
  mon_log="${OUT_DIR}/launcher/monitor.log"
  mon_pid_file="${OUT_DIR}/launcher/monitor.pid"
  nohup "${PYTHON}" scripts/monitor_probe_fill.py \
    --run-id "${RUN_ID}" \
    --out-dir "${OUT_DIR}" \
    --expected "${n_jobs}" \
    --interval-sec "${MONITOR_INTERVAL_SEC}" \
    --max-hours "${MONITOR_MAX_HOURS}" \
    > "${mon_log}" 2>&1 &
  echo $! > "${mon_pid_file}"
  echo "monitor pid=$(cat "${mon_pid_file}") log=${mon_log}"
fi

echo "Manual:"
echo "  sco acp jobs list --workspace-name=aigc --page-size 20"
echo "  python scripts/monitor_probe_fill.py --run-id ${RUN_ID} --once"
echo "  tail -f ${OUT_DIR}/launcher/PROGRESS.txt"
