#!/usr/bin/env bash
set -euo pipefail

export PATH=/mnt/umm/users/yinbaiqiao/.sco/bin:$PATH
root=/mnt/umm/users/yinbaiqiao/probe_spatial
tag=independent_goal_conditioned_continuous_action_p8_relational_v2_20260923
launch_dir="$root/results/$tag/execution/wan_l29_launcher"
log="$launch_dir/sco_submit.log"
ids="$launch_dir/job_ids.tsv"

test ! -e "$log"
mkdir -p "$launch_dir"
: > "$log"
: > "$ids"
for shard in 0 1 2 3 4 5 6 7; do
  name="bq-p8ca-wl29-s${shard}-v2"
  command="cd $root && NUM_SHARDS=8 SHARD_INDEX=$shard bash scripts/run_continuous_action_wan_l29_feature_shard.sh"
  output=$(sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name=zoetrope \
    --job-name="$name" \
    --priority=HIGHEST \
    --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
    --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec=N4lS.Iq.I80.1 \
    --command="$command")
  printf '%s\n' "$output" | tee -a "$log"
  job_id=$(printf '%s\n' "$output" | grep -oE 'pt-[a-z0-9]+' | head -1)
  test -n "$job_id"
  printf '%s\t%s\t%s\n' "$shard" "$name" "$job_id" | tee -a "$ids"
done
test "$(wc -l < "$ids")" -eq 8
