#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
RESULT=${ROOT}/results/experimental/se3_wan_vjepa2_exact_v2_layers_20260916_r1
SCO=/mnt/umm/users/yinbaiqiao/.sco/bin/sco
mkdir -p "${RESULT}/launcher"
name=bq-se3-v2-meta-$(date -u +%m%d%H%M%S)
command="cd ${ROOT} && bash scripts/run_se3_wan_vjepa2_v2_sidecar_inspect.sh"
output=$("${SCO}" acp jobs create --workspace-name=aigc --aec2-name=zoetrope \
  --job-name="${name}" --priority=HIGHEST --quota-type=reserved \
  --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm \
  --training-framework=pytorch --worker-nodes=1 --worker-spec=N4lS.Iq.I80.1 \
  --command="${command}")
printf '%s\n' "${output}" | tee "${RESULT}/launcher/sidecar_inspect_submit.log"
