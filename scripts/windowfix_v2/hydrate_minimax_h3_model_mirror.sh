#!/usr/bin/env bash
# Hydrate MiniMax-H3 from the immutable AOSS model mirror into node-local cache.
# Diffusers requires an ordinary local directory; this script makes AOSS the
# source of truth and only returns a verified local cache path.
set -euo pipefail

AOSS_DIR="${AOSS_DIR:-/mnt/umm/users/yinbaiqiao/aoss}"
ADS_CLI="${ADS_CLI:-${AOSS_DIR}/ads-cli}"
PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/minimax-h3/bin/python}"
HOST="${AOSS_API_HOST:-aoss-internal.cn-fz-01.fjscmsapi-oss.com}"
BUCKET="${BUCKET:-baiqiao}"
MIRROR_PREFIX="${MIRROR_PREFIX:-InsScene-15K/MODEL_MIRRORS/MiniMax-H3}"
MIRROR_SHA="${MINIMAX_H3_MIRROR_SHA256:?set MINIMAX_H3_MIRROR_SHA256}"
CACHE_ROOT="${MINIMAX_H3_MODEL_CACHE_ROOT:-/mnt/local_cache/windowfix_v2_models/minimax_h3}"

[[ -x "${ADS_CLI}" && -x "${PYTHON}" ]] || { echo "missing ads-cli/python" >&2; exit 2; }
mkdir -p "${CACHE_ROOT}"
lock="${CACHE_ROOT}/.${MIRROR_SHA}.lock"
exec 9>"${lock}"
flock 9
final="${CACHE_ROOT}/${MIRROR_SHA}"

verify() {
  "${PYTHON}" - "$1" "${MIRROR_SHA}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected=sys.argv[2]; manifest=root/'model_mirror_manifest.json'
if not manifest.is_file(): raise SystemExit(2)
data=json.loads(manifest.read_text())
stored=data.pop('content_sha256',None)
actual=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
if stored != expected or actual != expected: raise SystemExit(3)
for item in data['assets']:
    p=root/'payload'/item['path']
    if not p.is_file() or p.stat().st_size != int(item['size']): raise SystemExit(4)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    if h.hexdigest()!=item['sha256']: raise SystemExit(5)
print(root/'payload')
PY
}

if [[ -d "${final}" ]] && model_path="$(verify "${final}")"; then
  printf '%s\n' "${model_path}"
  exit 0
fi
stage="$(mktemp -d "${CACHE_ROOT}/.stage-${MIRROR_SHA:0:16}-XXXXXX")"
cleanup() { rm -rf "${stage}"; }
trap cleanup EXIT
conf="${AOSS_DIR}/petreloss.conf"
ak="$(awk '/^\[fj\]/{f=1;next}/^\[/{f=0} f&&/^access_key/{sub(/^[^=]*=[[:space:]]*/,"");print;exit}' "${conf}")"
sk="$(awk '/^\[fj\]/{f=1;next}/^\[/{f=0} f&&/^secret_key/{sub(/^[^=]*=[[:space:]]*/,"");print;exit}' "${conf}")"
[[ -n "${ak}" && -n "${sk}" ]] || { echo "missing AOSS fj credentials" >&2; exit 2; }
export NO_PROXY="localhost,127.0.0.1,::1,${BUCKET}.${HOST},.${HOST},${HOST}"
export no_proxy="${NO_PROXY}"
remote="s3://${ak}:${sk}@${BUCKET}.${HOST}/${MIRROR_PREFIX}/${MIRROR_SHA}/"
"${ADS_CLI}" -p "${THREADS:-8}" sync "${remote}" "${stage}/"
model_path="$(verify "${stage}")"
if [[ -e "${final}" ]]; then
  quarantine="${final}.invalid.$(date -u +%Y%m%dT%H%M%SZ)"
  mv "${final}" "${quarantine}"
  echo "quarantined invalid model cache: ${quarantine}" >&2
fi
mv "${stage}" "${final}"
trap - EXIT
printf '%s\n' "${final}/payload"
