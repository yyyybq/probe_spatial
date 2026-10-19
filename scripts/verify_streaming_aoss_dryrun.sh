#!/usr/bin/env bash
# Full per-scene ads-cli --dryrun for FEAT_STREAMING/{minimax_h3,videomae,vjepa2}.
# PASS only if every scene reports copied=0 failed=0 and rc=0.
set -u -o pipefail

ADS=/mnt/umm/users/yinbaiqiao/aoss/ads-cli
CONF=/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf
export NO_PROXY="localhost,127.0.0.1,::1,baiqiao.aoss-internal.cn-fz-01.fjscmsapi-oss.com,.aoss-internal.cn-fz-01.fjscmsapi-oss.com,aoss-internal.cn-fz-01.fjscmsapi-oss.com"
export no_proxy="$NO_PROXY"

AK=$(awk '/^\[fj\]/{in_fj=1;next}/^\[/{in_fj=0} in_fj&&/^access_key/{gsub(/^access_key[[:space:]]*=[[:space:]]*/,"");print;exit}' "$CONF")
SK=$(awk '/^\[fj\]/{in_fj=1;next}/^\[/{in_fj=0} in_fj&&/^secret_key/{gsub(/^secret_key[[:space:]]*=[[:space:]]*/,"");print;exit}' "$CONF")
HOST=aoss-internal.cn-fz-01.fjscmsapi-oss.com
BUCKET=baiqiao
OUT=${OUT:-/mnt/umm/users/yinbaiqiao/probe_spatial/logs/aoss_verify_streaming_20260810}
mkdir -p "$OUT/full_scene_dryrun2"
SUMMARY="$OUT/full_scene_summary2.tsv"
printf 'vfm\tscene\trc\tfound\tcopied\tcopiedSize\tfailed\n' > "$SUMMARY"

verify_scene() {
  local vfm="$1" scene="$2"
  local src="/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING/${vfm}/scannetpp/${scene}/"
  local dst="s3://${AK}:${SK}@${BUCKET}.${HOST}/InsScene-15K/FEAT_STREAMING/${vfm}/scannetpp/${scene}/"
  local log="${OUT}/full_scene_dryrun2/${vfm}_${scene}.log"
  timeout 240 "$ADS" --dryrun -p 2 -l 1 --conntimeout 120 --timeout 600 sync "$src" "$dst" >"$log" 2>&1
  local rc=$?
  local parsed
  parsed=$(python3 -c '
import re,sys
txt=open(sys.argv[1],"rb").read().decode("utf-8","ignore")
ms=list(re.finditer(r"Found: (\d+), copied: (\d+), copiedSize: ([^,]+), checked: [^,]+, deleted: \d+, Skipped: \d+, failed: (\d+)", txt))
print("NA NA NA NA" if not ms else "%s %s %s %s" % (ms[-1].group(1), ms[-1].group(2), ms[-1].group(3).replace(" ","_"), ms[-1].group(4)))
' "$log")
  read -r found copied csize failed <<<"$parsed"
  flock "$SUMMARY" printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$vfm" "$scene" "$rc" "$found" "$copied" "$csize" "$failed" >> "$SUMMARY"
  if [[ "$rc" != "0" || "$copied" != "0" || "$failed" != "0" || "$found" == "NA" ]]; then
    echo "ISSUE ${vfm} ${scene} rc=${rc} found=${found} copied=${copied} failed=${failed} size=${csize}"
  fi
}
export -f verify_scene
export ADS AK SK HOST BUCKET OUT SUMMARY

for vfm in minimax_h3 videomae vjepa2; do
  echo "[$(date -Is)] start full dryrun ${vfm}"
  find "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING/${vfm}/scannetpp" -mindepth 1 -maxdepth 1 -type d -printf "${vfm} %f\n" \
    | sort \
    | xargs -P 12 -n 2 bash -c 'verify_scene "$0" "$1"'
  echo "[$(date -Is)] finish ${vfm}"
done

python3 - <<'PY'
from pathlib import Path
from collections import Counter, defaultdict
p = Path('/mnt/umm/users/yinbaiqiao/probe_spatial/logs/aoss_verify_streaming_20260810/full_scene_summary2.tsv')
rows = []
for i, line in enumerate(p.read_text().splitlines()):
    if i == 0 or not line.strip():
        continue
    parts = line.split('\t')
    if len(parts) < 7:
        continue
    vfm, scene, rc, found, copied, csize, failed = parts[:7]
    rows.append(dict(vfm=vfm, scene=scene, rc=rc, found=found, copied=copied, failed=failed, csize=csize))
print('total_rows', len(rows))
by = Counter(r['vfm'] for r in rows)
print('by_vfm', dict(by))
issues = [r for r in rows if r['rc'] != '0' or r['copied'] != '0' or r['failed'] != '0' or r['found'] == 'NA']
print('issues', len(issues))
for r in issues[:80]:
    print(r)
ok = all(by.get(v, 0) == 681 for v in ['minimax_h3', 'videomae', 'vjepa2']) and len(issues) == 0
print('VERDICT', 'PASS_DELETE_OK' if ok else 'FAIL_DO_NOT_DELETE')
tot = defaultdict(int)
for r in rows:
    if r['found'].isdigit():
        tot[r['vfm']] += int(r['found'])
print('found_objects', dict(tot))
PY
