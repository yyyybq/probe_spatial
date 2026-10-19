#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
EVAL_RUN_TAG=${EVAL_RUN_TAG:?EVAL_RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
LAYER=${LAYER:?LAYER is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
MANIFEST=${PROJECT_ROOT}/manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl
DATA_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data
STREAMING_ROOT=${FEATURE_ROOT}/streaming
TARGET_ROOT=${FEATURE_ROOT}/target
REPORT_DIR=${PROJECT_ROOT}/reports/wan_p8_recovery/${RUN_TAG}
TRAIN_AUDIT=${REPORT_DIR}/train_audit.json
NAME=${RUN_TAG}_canonical_clean_action_dynamics_wan_p8_layer${LAYER}
EVAL_DIR=${PROJECT_ROOT}/logs/inscene15k_streaming/runs/${EVAL_RUN_TAG}/${NAME}
LOG=${REPORT_DIR}/logs/eval_layer${LAYER}.log
STATUS=${REPORT_DIR}/logs/eval_layer${LAYER}.status.json

case " ${LAYER} " in
  " 0 "|" 5 "|" 10 "|" 15 "|" 25 "|" 29 ") ;;
  *) echo "invalid layer ${LAYER}" >&2; exit 2 ;;
esac
[[ -f "${TRAIN_AUDIT}" ]] || { echo "missing train audit" >&2; exit 2; }
[[ ! -e "${EVAL_DIR}" ]] || { echo "refusing existing eval dir: ${EVAL_DIR}" >&2; exit 2; }
[[ ! -e "${STATUS}" ]] || { echo "refusing existing eval status: ${STATUS}" >&2; exit 2; }

exec > >(tee -a "${LOG}") 2>&1
started=$(date -u +%FT%TZ)
finish() {
  rc=$?
  trap - EXIT
  ended=$(date -u +%FT%TZ)
  printf '{"run_tag":"%s","eval_run_tag":"%s","layer":%d,"started":"%s","ended":"%s","rc":%d}\n' \
    "${RUN_TAG}" "${EVAL_RUN_TAG}" "${LAYER}" "${started}" "${ended}" "${rc}" > "${STATUS}"
  exit "${rc}"
}
trap finish EXIT

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PROJECT_ROOT
export INSCENE_DATA_ROOT="${DATA_ROOT}"
export INSCENE_MANIFEST="${MANIFEST}"
export INSCENE_STREAMING_FEAT_ROOT="${STREAMING_ROOT}"
export INSCENE_TARGET_FEAT_ROOT="${TARGET_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/matplotlib-root
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}"
cd "${PROJECT_ROOT}"

CKPT_INFO=$("${PY}" - "${TRAIN_AUDIT}" "${LAYER}" "${MANIFEST}" "${STREAMING_ROOT}" "${TARGET_ROOT}" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

audit = json.load(open(sys.argv[1]))
layer = sys.argv[2]
manifest = Path(sys.argv[3]).resolve()
streaming = Path(sys.argv[4]).resolve()
target = Path(sys.argv[5]).resolve()
assert audit.get('ok') is True
entry = audit['layers'][layer]
assert entry['status'] == 'CLEAN_COMPLETE'
assert sha(manifest) == '772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c'
checks = {item['name']: item for item in entry['feature_root_checks']}
assert Path(checks['streaming']['actual']).resolve() == streaming and checks['streaming']['ok'] is True
assert Path(checks['target']['actual']).resolve() == target and checks['target']['ok'] is True
checkpoint_link = Path(entry['checkpoint']).absolute()
checkpoint = checkpoint_link.resolve()
assert checkpoint_link.name == 'last.ckpt' and checkpoint_link.is_symlink()
assert checkpoint.exists() and checkpoint.name == 'epoch=49-step=15300.ckpt'
assert checkpoint == Path(entry['checkpoint_realpath']).resolve()
assert entry['checkpoint_state']['epoch'] >= 49 and entry['checkpoint_state']['global_step'] >= 15300
assert sha(checkpoint) == entry['checkpoint_sha256']
print(json.dumps({
    'input_path': str(checkpoint_link),
    'realpath': str(checkpoint),
    'sha256': entry['checkpoint_sha256'],
}))
PY
)
CKPT_INPUT=$("${PY}" -c 'import json,sys; print(json.loads(sys.argv[1])["input_path"])' "${CKPT_INFO}")
CKPT=$("${PY}" -c 'import json,sys; print(json.loads(sys.argv[1])["realpath"])' "${CKPT_INFO}")
CKPT_SHA=$("${PY}" -c 'import json,sys; print(json.loads(sys.argv[1])["sha256"])' "${CKPT_INFO}")
echo "EVAL INPUT GATE PASS layer=${LAYER} ckpt_input=${CKPT_INPUT} ckpt_realpath=${CKPT}"

command=(
  "${PY}" vidfm3d/eval_diag.py
  experiment=inscene15k_streaming/action_dynamics_wan_v1
  seed=42 train=false test=false autoresume=false
  vfm_name=wan video_channels=1536
  streaming_feat_root="${STREAMING_ROOT}"
  target_feat_root="${TARGET_ROOT}"
  feature_layer="${LAYER}" feature_timestep=749 prefix_len=8
  job_name="${NAME}"
  paths.run_folder_name="${EVAL_RUN_TAG}/${NAME}"
  logger.wandb.offline=true
  data.data_module.batch_size_per_device_val=4
  data.data_module.num_workers_val=2
  ckpt_path="${CKPT_INPUT}"
  +eval_split=val
)
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"

"${PY}" - "${EVAL_DIR}" "${MANIFEST}" "${STREAMING_ROOT}" "${TARGET_ROOT}" "${CKPT}" "${CKPT_SHA}" "${LAYER}" <<'PY'
import hashlib, json, math, os, sys
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

eval_dir, manifest, streaming, target, checkpoint = map(Path, sys.argv[1:6])
expected_ckpt_sha, layer = sys.argv[6], int(sys.argv[7])
summary_path = eval_dir / 'eval' / 'val_summary.json'
summary = json.load(summary_path.open())
required = [
    'mean_cos', 'global_R@1', 'global_R@5', 'mean_rank',
    'no_action_global_R@1', 'shuffled_action_global_R@1',
    'last_observation_global_R@1', 'action_gain_cos',
]
assert int(summary.get('n', 0)) == 276
assert all(math.isfinite(float(summary[key])) for key in required)
assert summary.get('probe_type') == 'action_dynamics'
assert int(summary.get('feature_layer', -1)) == layer
assert int(summary.get('feature_timestep', -1)) == 749
assert summary.get('vfm_name') == 'wan'
assert Path(summary['ckpt_path']).resolve() == checkpoint.resolve()
assert Path(summary['streaming_feat_root']).resolve() == streaming.resolve()
assert sha(manifest) == '772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c'
assert sha(checkpoint) == expected_ckpt_sha
assert not streaming.is_symlink() and not target.is_symlink()
provenance = {
    'ok': True,
    'layer': layer,
    'manifest': str(manifest.resolve()),
    'manifest_sha256': sha(manifest),
    'streaming_root': str(streaming.resolve()),
    'target_root': str(target.resolve()),
    'checkpoint': str(checkpoint.resolve()),
    'checkpoint_sha256': expected_ckpt_sha,
    'summary': str(summary_path.resolve()),
    'summary_sha256': sha(summary_path),
    'n': int(summary['n']),
    'required_metrics': {key: summary[key] for key in required},
}
prov_dir = eval_dir / 'provenance'
prov_dir.mkdir(parents=True, exist_ok=False)
tmp = prov_dir / f'.eval_provenance.json.tmp-{os.getpid()}'
tmp.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
os.replace(tmp, prov_dir / 'eval_provenance.json')
print(json.dumps({'ok': True, 'layer': layer, 'n': summary['n']}))
PY

echo "EVAL STAGING PASS layer=${LAYER}"
