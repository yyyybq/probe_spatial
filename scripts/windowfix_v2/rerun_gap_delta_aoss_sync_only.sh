#!/usr/bin/env bash
# Publish already-extracted canonical_v2 gap leftovers. No extract.
# Usage: bash rerun_gap_delta_aoss_sync_only.sh <cog|wan|videomae|lingbot>
set -u
cd /mnt/umm/users/yinbaiqiao/probe_spatial
export GAP_SYNC_COMPLETENESS_ONLY=1
export GAP_SYNC_MAX_SCENES="${GAP_SYNC_MAX_SCENES:-128}"
export SYNC_EVERY_SEC="${SYNC_EVERY_SEC:-15}"
export SYNC_EVERY_GIB="${SYNC_EVERY_GIB:-1}"
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
SYNC=scripts/windowfix_v2/run_canonical_gap_sync_only.py
vfm="${1:?usage: $0 cog|wan|videomae|lingbot}"
case "${vfm}" in
  cog)
    shard=/mnt/umm/users/yinbaiqiao/probe_spatial/artifacts/windowfix_v2/sco_plans/canonical_v2_gap_cogvideox_split8_h800_r2_20260905/shards/wf2-gap-001-cogvideox-749-streaming_prefix.json
    ;;
  wan)
    shard=/mnt/umm/users/yinbaiqiao/probe_spatial/artifacts/windowfix_v2/sco_plans/canonical_v2_gap_wan_resume_zoetrope_r3_20260905/shards/wf2-gap-001-wan-749-streaming_prefix.json
    ;;
  videomae)
    shard=/mnt/umm/users/yinbaiqiao/probe_spatial/artifacts/windowfix_v2/sco_plans/canonical_v2_gap_resume_cgx_wan_videomae_zoetrope_r1_20260905/shards/wf2-gap-003-videomae-749-streaming_prefix.json
    ;;
  lingbot)
    shard=/mnt/umm/users/yinbaiqiao/probe_spatial/artifacts/windowfix_v2/sco_plans/canonical_v2_gap_lingbot_stream_h800_r2_20260908/shards/wf2-gap-001-lingbot-250-streaming_prefix.json
    ;;
  *)
    echo "unknown vfm=${vfm}" >&2
    exit 2
    ;;
esac
echo "whoami=$(whoami) uid=$(id -u) start=$(date -u +%Y-%m-%dT%H:%M:%SZ) vfm=${vfm} max_scenes=${GAP_SYNC_MAX_SCENES}"
echo "START $(basename "${shard}")"
exec "${PY}" "${SYNC}" --shard "${shard}" --trust-current-tools
