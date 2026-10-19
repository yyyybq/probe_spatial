#!/usr/bin/env bash
# Cleanup approved log categories under logs/.
# Usage:
#   bash scripts/cleanup_logs_safe.sh          # user-owned paths only
#   bash scripts/cleanup_logs_safe.sh --sudo   # also remove root-owned leftovers
set -euo pipefail

LOGS="$(cd "$(dirname "$0")/.." && pwd)/logs"
cd "$LOGS"
USE_SUDO=0
[[ "${1:-}" == "--sudo" ]] && USE_SUDO=1

echo "==> cwd: $LOGS"

echo "==> [3] one-shot sync log (idempotent)"
rm -fv aoss_sync_features_20260725_0355.log aoss_sync_features_20260725_0355.pid || true

echo "==> [4] top-level status snapshots / temp scripts"
rm -fv \
  build_after_recovery_status.py \
  current_probe_status_20260719.py \
  current_train_processes.tsv \
  ddp_candidate_audit.tsv \
  ddp_recovery_handoff_20260717.md \
  latest_rr_run.txt \
  node_usage_snapshot_20260718_2158.tsv \
  no_progress_after_recovery_20260718.tsv \
  no_progress_investigation_20260718.tsv \
  probe_current_progress_eta_latest.json \
  probe_current_progress_eta_latest.tsv \
  probe_current_status_20260719_1908.json \
  probe_current_status_20260719_1908_live.json \
  probe_current_status_20260719_1908.tsv \
  probe_ddp4_handoff_dino_f3r_vjepa2_videomae_20260717.md \
  probe_epoch_eta_20260716_0353.json \
  probe_epoch_eta_actual_20260716_0355.json \
  probe_epoch_progress_latest.tsv \
  probe_epoch_speed_by_model_task.tsv \
  probe_node_gpu_status_latest.json \
  probe_progress_status_20260716_0346.json \
  probe_status_processes_20260716_0343.json \
  probe_status_processes_detailed_20260716_0345.json \
  probe_training_status_after_recovery_20260718.tsv \
  probe_training_status_after_recovery_20260718_v2.tsv \
  probe_training_status_current_20260718_2158.tsv \
  probe_training_status_latest.tsv \
  probing_jobs_20260725_ab_watcher.nohup.log \
  probing_jobs_20260725_ab_watcher.pid \
  unfinished_tasks_current_20260718_2158.tsv

echo "==> [1/partial] already-deleted or user-owned telemetry"
# perf_baseline was user-owned and may already be gone
rm -rf probing_jobs_20260725_ab/recovery_20260730/perf_baseline || true

echo "==> [2] old launch / job log dirs (user-owned; excludes eval_results)"
rm -rf \
  cog_target_missing_20260724_1808 \
  cog_target_repair_20260723_1848 \
  c_retrain_20260720_105443 \
  eval_all_20260719_230316 \
  eval_all_20260719_231250 \
  eval_b2_accel_20260720_072842 \
  eval_latestckpt_20260720_093340 \
  eval_missing_20260720_072251 \
  eval_new_nodes_20260720_072720 \
  eval_qwen_20260720_075236 \
  materialize_by_prefix_20260724_1840 \
  probe_ddp4_a2dino_p24_20260718_185414 \
  probe_ddp4_accel_dino_f3r_20260717_195226 \
  probe_ddp4_final_20260719_214152 \
  probe_ddp4_wave2_20260719_020756 \
  probe_ddp4_wave3_20260719_193931 \
  probe_ddp4_wave3_20260719_195357 \
  probe_ddp4_wave3_20260719_210324 \
  probe_ddp4_wave3_20260719_210850 \
  probe_longtail_ddp4_race_20260718_1907 \
  probe_longtail_ddp4_race2_20260719_0205 \
  probe_qwen_b2_p16_ddp4_race_20260719_2208 \
  probe_recovery_after_retire_bug_20260717_1920 \
  probe_train_24gpu_20260715_190655 \
  probe_train_24gpu_20260715_190835 \
  probe_train_24gpu_20260715_191752 \
  probe_train_24gpu_20260715_192452 \
  probe_train_24gpu_20260715_193241 \
  probe_train_24gpu_20260715_194202 \
  probe_train_64gpu_heavy_20260716_021635 \
  probe_train_64gpu_heavy_migrate_20260716_025904 \
  probe_train_b2_cogvideox_retry_20260716_023854 \
  probe_train_ddp_accel_20260717_184053 \
  probe_train_moved48_newnodes_20260716_0325 \
  probe_train_parallel_tail_20260716_032426 \
  probe_train_pending32_correct_nodes_20260716_0335 \
  probe_train_queued_restart_20260717_140742 \
  probe_train_qwen_b2_retry_lowworkers_20260716_041310 \
  probe_train_qwen_remaining_20260716_033855 \
  probe_train_restart_full_logs_20260716_030856 \
  probe_train_restart_full_logs_20260716_0312_fix1 \
  probe_train_rr_20260715_214211 \
  probing_jobs_20260724_1808 \
  probing_jobs_20260725_ab_retry_streaming_depth \
  probing_jobs_20260730 \
  remote_feature_extract_20260715_114805 \
  remote_feature_extract_20260715_115225 \
  remote_feature_extract_20260715_121043 \
  remote_feature_extract_aether_fixed_20260715_154341 \
  remote_feature_extract_aether_p12_target_retry_20260715_165804 \
  remote_feature_extract_f3r_target_retry_20260715_160222 \
  remote_feature_extract_new32_tail_20260715_172441 \
  remote_feature_extract_old_idle_aether_fill_20260715_180709 \
  remote_feature_extract_parallel_targets_20260715_155644 \
  remote_feature_extract_restore_20260715_151947 \
  remote_feature_extract_retry_core_20260715_121731 \
  remote_feature_extract_retry_f3r_20260715_132220 \
  remote_feature_extract_videomae_20260715_152845 \
  remote_feature_extract_videomae_p24_streaming_fill_20260715_200516 \
  remote_feature_extract_videomae_p24_streaming_fill_20260715_200930 \
  streaming_gap_48da_20260724_2245 \
  target_fix_20260723_0450 \
  target_fix_20260723_0458 \
  target_fix_multi_20260723_0540 \
  target_fix_multi_20260723_0555

ROOT_LEFTOVERS=(
  probing_jobs_20260725_ab/recovery_20260730/monitor/aoss_metrics
  probing_jobs_20260725_ab/recovery_20260730/monitor/gpu_metrics.jsonl
  feat_layers_20260722_0408
  feat_layers_20260722_0410
  target_layers_20260722_0948
  # partially wiped; remaining files inside are root-owned
  cog_target_missing_20260724_1808
  cog_target_repair_20260723_1848
  target_fix_20260723_0450
  target_fix_20260723_0458
  target_fix_multi_20260723_0540
  target_fix_multi_20260723_0555
)

if [[ "$USE_SUDO" -eq 1 ]]; then
  echo "==> [1+root] root-owned leftovers via sudo"
  sudo rm -rf "${ROOT_LEFTOVERS[@]}"
else
  echo "==> skip root-owned (pass --sudo to remove):"
  for p in "${ROOT_LEFTOVERS[@]}"; do
    [ -e "$p" ] && ls -ld "$p"
  done
fi

echo "==> remaining top-level:"
ls -1
echo "DONE"
