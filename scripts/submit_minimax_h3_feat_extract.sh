#!/usr/bin/env bash
# Submit MiniMax-H3 streaming+target feature extraction to SCO ACP (8 GPUs).
#
# Usage:
#   bash scripts/submit_minimax_h3_feat_extract.sh [aec2_name] [job_name]
#
# Examples:
#   bash scripts/submit_minimax_h3_feat_extract.sh zoetrope bq_mmh3_f8
#   bash scripts/submit_minimax_h3_feat_extract.sh h800 bq_mmh3_f8
#
# Check:
#   sco acp jobs list --workspace-name=aigc
#   sco acp jobs stream-logs --workspace-name=aigc <job_id>

set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"

repo_dir="/mnt/umm/users/yinbaiqiao/probe_spatial"

aec2_name=${1:-"zoetrope"}
num_nodes=1
nproc_per_node=8
job_name=${2:-"bq_mmh3_f8"}

container_image_url="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
storage_mount="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"

if [ "${aec2_name}" = "zoetrope" ] || [ "${aec2_name}" = "h800" ]; then
    worker_spec="N4lS.Iq.I80."
elif [ "${aec2_name}" = "vigen" ] || [ "${aec2_name}" = "si" ]; then
    worker_spec="N6lS.Iq.I10."
else
    echo "Error: Invalid aec2_name '${aec2_name}'. Supported: zoetrope | h800 | vigen | si"
    exit 1
fi

echo "--------------------------------"
echo "aec2_name: ${aec2_name}"
echo "num_nodes: ${num_nodes}"
echo "nproc_per_node: ${nproc_per_node}"
echo "job_name: ${job_name}"
echo "worker_spec: ${worker_spec}${nproc_per_node}"
echo "repo_dir: ${repo_dir}"
echo "--------------------------------"

# Keep --command short: long escaped command strings have returned HTTP 403.
sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name=${aec2_name} \
    --job-name=${job_name} \
    --priority=HIGHEST \
    --container-image-url=${container_image_url} \
    --storage-mount=${storage_mount} \
    --training-framework=pytorch \
    --worker-nodes=${num_nodes} \
    --worker-spec="${worker_spec}${nproc_per_node}" \
    --command="cd ${repo_dir} && bash scripts/run_minimax_h3_8gpu_extract.sh"
