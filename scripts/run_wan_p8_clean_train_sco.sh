#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
LAYER=${LAYER:?LAYER is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
DATA_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data
MANIFEST=${PROJECT_ROOT}/manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl
STREAMING_ROOT=${FEATURE_ROOT}/streaming
TARGET_ROOT=${FEATURE_ROOT}/target
REPORT_DIR=${PROJECT_ROOT}/reports/wan_p8_recovery/${RUN_TAG}
TRAIN_PROVENANCE=${REPORT_DIR}/provenance/training
NAME=${RUN_TAG}_canonical_clean_action_dynamics_wan_p8_layer${LAYER}
RUN_DIR=${PROJECT_ROOT}/logs/inscene15k_streaming/runs/${RUN_TAG}/${NAME}
LOG=${REPORT_DIR}/logs/train_layer${LAYER}.log
STATUS=${REPORT_DIR}/logs/train_layer${LAYER}.status.json

case " ${LAYER} " in
  " 0 "|" 5 "|" 10 "|" 15 "|" 25 "|" 29 ") ;;
  *) echo "invalid layer ${LAYER}" >&2; exit 2 ;;
esac
[[ ! -e "${RUN_DIR}" ]] || { echo "refusing existing RUN_DIR=${RUN_DIR}" >&2; exit 2; }
[[ ! -e "${STATUS}" ]] || { echo "refusing existing status=${STATUS}" >&2; exit 2; }
[[ -f "${TRAIN_PROVENANCE}/historical_training_contract.json" ]] || { echo "missing training contract" >&2; exit 2; }

exec > >(tee -a "${LOG}") 2>&1
started=$(date -u +%FT%TZ)
finish() {
  rc=$?
  trap - EXIT
  ended=$(date -u +%FT%TZ)
  printf '{"run_tag":"%s","layer":%d,"started":"%s","ended":"%s","rc":%d}\n' \
    "${RUN_TAG}" "${LAYER}" "${started}" "${ended}" "${rc}" > "${STATUS}"
  exit "${rc}"
}
trap finish EXIT

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PROJECT_ROOT
export INSCENE_DATA_ROOT="${DATA_ROOT}"
export INSCENE_MANIFEST="${MANIFEST}"
export INSCENE_STREAMING_FEAT_ROOT="${STREAMING_ROOT}"
export INSCENE_TARGET_FEAT_ROOT="${TARGET_ROOT}"
export HF_HOME=${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/matplotlib-root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}" "${RUN_DIR}/provenance"

cd "${PROJECT_ROOT}"
echo "START ${started} layer=${LAYER} name=${NAME}"
nvidia-smi
"${PY}" - "${REPORT_DIR}/full_gate_summary.json" "${MANIFEST}" "${STREAMING_ROOT}" "${TARGET_ROOT}" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

gate=json.load(open(sys.argv[1]))
manifest=Path(sys.argv[2]).resolve()
streaming=Path(sys.argv[3]).resolve()
target=Path(sys.argv[4]).resolve()
assert gate.get('ok') is True
assert gate.get('canonical_manifest_sha256') == '772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c'
assert sha(manifest) == gate['canonical_manifest_sha256']
assert Path(gate['streaming_root']).resolve() == streaming
assert Path(gate['target_root']).resolve() == target
assert gate.get('streaming_exact_raw') == 2724 and gate.get('target_exact_raw') == 2724
assert gate.get('finite_files_checked') == 20430
for value in gate.get('dataset_layers', {}).values():
    assert value.get('ok') is True and value.get('checked') == 2724 and not value.get('problems')
print('TRAIN INPUT GATE PASS')
PY

cp "${MANIFEST}" "${RUN_DIR}/provenance/launcher_manifest_snapshot.jsonl"
cp "${REPORT_DIR}/full_gate_summary.json" "${RUN_DIR}/provenance/full_gate_summary.json"
cp "${TRAIN_PROVENANCE}/historical_training_contract.json" "${RUN_DIR}/provenance/historical_training_contract.json"
cp "${TRAIN_PROVENANCE}/git_commit.txt" "${RUN_DIR}/provenance/launcher_git_commit.txt"
cp "${TRAIN_PROVENANCE}/git_status_short.txt" "${RUN_DIR}/provenance/launcher_git_status_short.txt"
cp "${TRAIN_PROVENANCE}/git_dirty.diff" "${RUN_DIR}/provenance/launcher_git_dirty.diff"
printf 'streaming=%s\ntarget=%s\n' "$(readlink -f "${STREAMING_ROOT}")" "$(readlink -f "${TARGET_ROOT}")" \
  > "${RUN_DIR}/provenance/launcher_feature_roots.txt"

command=(
  "${PY}" vidfm3d/train.py
  experiment=inscene15k_streaming/action_dynamics_wan_v1
  seed=42 train=true test=false
  vfm_name=wan video_channels=1536
  streaming_feat_root="${STREAMING_ROOT}"
  target_feat_root="${TARGET_ROOT}"
  feature_layer="${LAYER}" feature_timestep=749 prefix_len=8
  job_name="${NAME}"
  paths.run_folder_name="${RUN_TAG}/${NAME}"
  logger.wandb.name="${NAME}"
  logger.wandb.group="${RUN_TAG}"
  logger.wandb.offline=true
  autoresume=false ckpt_path=null
  data.data_module.batch_size_per_device=8
  data.data_module.batch_size_per_device_val=8
  data.data_module.num_workers=4
  data.data_module.num_workers_val=2
  data.data_module.pin_memory=true
  model.optimizer.lr=0.0003
  'model.optimizer.betas=[0.9,0.95]'
  model.optimizer.weight_decay=0.05
  model.scheduler.warmup_epochs=5
  model.scheduler.eta_min=0.000001
  model.probe.hidden_dim=512
  model.probe.num_layers=2
  model.probe.num_heads=8
  model.probe.max_seq_len=64
  model.mse_loss_weight=1.0
  model.cosine_loss_weight=1.0
  callbacks.model_checkpoint.every_n_epochs=5
  callbacks.model_checkpoint.save_last=true
  callbacks.model_checkpoint.save_top_k=-1
  callbacks.early_stopping=null
  trainer.min_epochs=1
  trainer.max_epochs=50
  trainer.devices='[0]'
  trainer.strategy=auto
  trainer.num_nodes=1
  trainer.check_val_every_n_epoch=1
  trainer.precision=bf16-mixed
  trainer.sync_batchnorm=true
  trainer.use_distributed_sampler=false
  trainer.deterministic=false
)
printf '%q ' "${command[@]}" > "${RUN_DIR}/provenance/launcher_train_command.txt"
printf '\n' >> "${RUN_DIR}/provenance/launcher_train_command.txt"
"${command[@]}"

"${PY}" - "${RUN_DIR}" "${MANIFEST}" "${STREAMING_ROOT}" "${TARGET_ROOT}" "${LAYER}" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch, yaml

run=Path(sys.argv[1]); manifest=Path(sys.argv[2]); stream=Path(sys.argv[3]).resolve(); target=Path(sys.argv[4]).resolve(); layer=int(sys.argv[5])
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

assert sha(manifest) == '772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c'
assert sha(run/'provenance/manifest_snapshot.jsonl') == sha(manifest)
prov=json.load(open(run/'provenance/run_provenance.json'))
assert prov['data']['scene_manifest_sha256'] == sha(manifest)
roots={item['name']:item for item in prov['data']['feature_roots']}
assert Path(roots['streaming']['realpath']).resolve() == stream and roots['streaming']['is_symlink'] is False
assert Path(roots['target']['realpath']).resolve() == target and roots['target']['is_symlink'] is False
cfg=yaml.safe_load(open(run/'.hydra/config.yaml'))
assert cfg['ckpt_path'] is None and cfg['autoresume'] is False and cfg['seed'] == 42
assert cfg['feature_layer'] == layer and cfg['trainer']['max_epochs'] == 50
assert Path(cfg['streaming_feat_root']).resolve() == stream and Path(cfg['target_feat_root']).resolve() == target
ckpt=run/'checkpoints/last.ckpt'
state=torch.load(ckpt,map_location='cpu',weights_only=False)
assert int(state['epoch']) >= 49, state.get('epoch')
print(json.dumps({'clean_exit_precheck':True,'layer':layer,'epoch':state['epoch'],'checkpoint':str(ckpt)}))
PY

echo "TRAIN CLEAN EXIT layer=${LAYER}"
