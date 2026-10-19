#!/usr/bin/env bash
# Canary: sync-only one smallish VFM (vjepa2) under GAP_DELTA.
# Run as root: sudo -E bash scripts/windowfix_v2/rerun_gap_delta_aoss_sync_vjepa2_canary.sh
set -euo pipefail
cd /mnt/umm/users/yinbaiqiao/probe_spatial
export AOSS_API_HOST="${AOSS_API_HOST:-aoss-internal.cn-fz-01.fjscmsapi-oss.com}"
export NO_PROXY="localhost,127.0.0.1,::1,baiqiao.${AOSS_API_HOST},${AOSS_API_HOST}"
export no_proxy="${NO_PROXY}"
export AOSS_CONF="${AOSS_CONF:-/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf}"
export CONF="${CONF:-${AOSS_CONF}}"
OUT=/mnt/umm/users/yinbaiqiao/probe_spatial/logs/quota_reclaim_20260901/gap_delta_sync_only
mkdir -p "${OUT}"
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
BATCH=/mnt/umm/users/yinbaiqiao/probe_spatial/artifacts/windowfix_v2/sco_plans/canonical_v2_gap_modelunit_bounded_r7_20260901/batches/wf2-gap-batch-05-vjepa2.json
echo "whoami=$(whoami) uid=$(id -u)"
echo "writing under GAP tree as $(whoami)..."
touch /mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/.sync_only_canary_$(date +%s) \
  || { echo "ERROR: cannot write GAP root; need root/owner"; exit 1; }
"${PY}" scripts/windowfix_v2/run_canonical_gap_sync_only.py \
  --batch "${BATCH}" \
  --streaming-only \
  --trust-current-tools \
  2>&1 | tee "${OUT}/vjepa2_canary.log"
