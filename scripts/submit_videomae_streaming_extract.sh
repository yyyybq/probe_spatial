#!/usr/bin/env bash
# Submit VideoMAE-only streaming_prefix multilayer extraction to SCO ACP (4 GPUs).
#
# Usage:
#   bash scripts/submit_videomae_streaming_extract.sh [aec2_name] [job_name]
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"

repo_dir="/mnt/umm/users/yinbaiqiao/probe_spatial"
aec2_name=${1:-"zoetrope"}
num_nodes=1
nproc_per_node=4
job_name=${2:-"bq_videomae_stream4"}

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

chmod +x "${repo_dir}/scripts/run_vjepa2_videomae_streaming_8gpu_extract.sh"

log_dir="${repo_dir}/logs/videomae_streaming_extract_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${log_dir}"
echo "log_dir=${log_dir}"

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
    --command="cd ${repo_dir} && ONLY_VFMS=videomae N_GPUS=4 GPUS_PER_VFM=4 LOG_DIR=${log_dir} bash scripts/run_vjepa2_videomae_streaming_8gpu_extract.sh"
