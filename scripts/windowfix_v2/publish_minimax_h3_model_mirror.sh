#!/usr/bin/env bash
# Publish a verified, content-addressed MiniMax-H3 model mirror to AOSS.
#
# This is intentionally model-specific: it never touches feature namespaces.
# It writes a hardlink staging tree, uses the cluster's existing ads-cli sync
# transport, verifies every remote object by size, and records a receipt.  The
# original local model directory is retained; cleanup is a separate explicit
# decision after a successful AOSS hydration smoke.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3}"
AOSS_DIR="${AOSS_DIR:-/mnt/umm/users/yinbaiqiao/aoss}"
ADS_CLI="${ADS_CLI:-${AOSS_DIR}/ads-cli}"
PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}"
HOST="${AOSS_API_HOST:-aoss-internal.cn-fz-01.fjscmsapi-oss.com}"
BUCKET="${BUCKET:-baiqiao}"
MIRROR_PREFIX="${MIRROR_PREFIX:-InsScene-15K/MODEL_MIRRORS/MiniMax-H3}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_ROOT}/artifacts/windowfix_v2/model_mirrors/minimax_h3}"

[[ -x "${ADS_CLI}" ]] || { echo "missing ads-cli: ${ADS_CLI}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "missing python: ${PYTHON}" >&2; exit 2; }
[[ -d "${MODEL_ROOT}" && ! -L "${MODEL_ROOT}" ]] || { echo "missing model root" >&2; exit 2; }

manifest_tmp="${WORK_ROOT}/model_mirror_manifest.json"
mkdir -p "${WORK_ROOT}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/windowfix_v2/build_minimax_h3_model_mirror_manifest.py" \
  --model-root "${MODEL_ROOT}" --output "${manifest_tmp}" >/dev/null

content_sha="$(${PYTHON} - "${manifest_tmp}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['content_sha256'])
PY
)"
immutable_manifest="${WORK_ROOT}/manifests/${content_sha}.json"
mkdir -p "$(dirname "${immutable_manifest}")"
if [[ -e "${immutable_manifest}" ]]; then
  cmp -s "${manifest_tmp}" "${immutable_manifest}" || {
    echo "refusing immutable manifest collision: ${immutable_manifest}" >&2
    exit 2
  }
else
  install -m 0444 "${manifest_tmp}" "${immutable_manifest}"
fi
stage="$(mktemp -d "${WORK_ROOT}/.stage-${content_sha:0:16}-XXXXXX")"
cleanup() { rm -rf "${stage}"; }
trap cleanup EXIT

"${PYTHON}" - "${manifest_tmp}" "${MODEL_ROOT}" "${stage}" <<'PY'
import json, os, shutil, sys
from pathlib import Path
manifest=Path(sys.argv[1]); root=Path(sys.argv[2]); stage=Path(sys.argv[3])
payload=stage/'payload'; payload.mkdir(parents=True)
data=json.loads(manifest.read_text())
for item in data['assets']:
    src=root/item['path']; dst=payload/item['path']; dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_symlink() or not src.is_file(): raise SystemExit(f'unsafe source {src}')
    os.link(src, dst)
shutil.copy2(manifest, stage/'model_mirror_manifest.json')
PY

conf="${AOSS_DIR}/petreloss.conf"
ak="$(awk '/^\[fj\]/{f=1;next}/^\[/{f=0} f&&/^access_key/{sub(/^[^=]*=[[:space:]]*/,"");print;exit}' "${conf}")"
sk="$(awk '/^\[fj\]/{f=1;next}/^\[/{f=0} f&&/^secret_key/{sub(/^[^=]*=[[:space:]]*/,"");print;exit}' "${conf}")"
[[ -n "${ak}" && -n "${sk}" ]] || { echo "missing AOSS fj credentials" >&2; exit 2; }
remote="s3://${ak}:${sk}@${BUCKET}.${HOST}/${MIRROR_PREFIX}/${content_sha}/"
export NO_PROXY="localhost,127.0.0.1,::1,${BUCKET}.${HOST},.${HOST},${HOST}"
export no_proxy="${NO_PROXY}"
"${ADS_CLI}" -p "${THREADS:-8}" sync "${stage}/" "${remote}"

# Verify remote presence and exact byte sizes using the same configured AOSS
# client that dataset readers use.  Payload hashes are already bound in the
# manifest; a full remote re-read is intentionally deferred to hydration.
"${PYTHON}" - "${manifest_tmp}" "${BUCKET}" "${MIRROR_PREFIX}" "${content_sha}" <<'PY'
import json, sys
from vidfm3d.utils import aoss_fs
manifest,bucket,prefix,content=sys.argv[1:]
data=json.load(open(manifest, encoding='utf-8'))
base=f'aoss://{bucket}/{prefix}/{content}'
client=aoss_fs.get_client()
checks=[('model_mirror_manifest.json', None), *[(f"payload/{x['path']}", int(x['size'])) for x in data['assets']]]
missing=[]
for rel,size in checks:
    remote=aoss_fs._client_path(f'{base}/{rel}')
    if not client.contains(remote): missing.append(rel); continue
    if size is not None and int(client.size(remote)) != size: missing.append(f'{rel}:size')
if missing: raise SystemExit('remote verification failed: '+','.join(missing[:20]))
print(json.dumps({'status':'PASS_REMOTE_SIZE_VERIFIED','content_sha256':content,'objects':len(checks),'remote_base':base},sort_keys=True))
PY

receipt="${WORK_ROOT}/receipts/${content_sha}.json"
mkdir -p "$(dirname "${receipt}")"
"${PYTHON}" - "${immutable_manifest}" "${receipt}" "${BUCKET}" "${MIRROR_PREFIX}" <<'PY'
import hashlib,json,os,sys,tempfile
manifest,receipt,bucket,prefix=sys.argv[1:]
data=json.load(open(manifest,encoding='utf-8'))
out={'contract':'MINIMAX_H3_AOSS_MODEL_MIRROR_RECEIPT_V1','status':'PASS_REMOTE_SIZE_VERIFIED','repository':data['repository'],'manifest_path':manifest,'manifest_file_sha256':hashlib.sha256(open(manifest,'rb').read()).hexdigest(),'manifest_content_sha256':data['content_sha256'],'asset_count':data['asset_count'],'total_bytes':data['total_bytes'],'remote_base':f'aoss://{bucket}/{prefix}/{data["content_sha256"]}'}
out['content_sha256']=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(',',':')).encode()).hexdigest()
fd,tmp=tempfile.mkstemp(prefix='.receipt-',dir=os.path.dirname(receipt))
with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(out,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp,receipt); os.chmod(receipt,0o444)
print(json.dumps(out,sort_keys=True))
PY
