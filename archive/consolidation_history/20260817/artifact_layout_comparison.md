# Artifact Layout Comparison

This file compares representative artifact layouts used by the reconstruction script. It intentionally does not promote any candidate into the authoritative registry.

| area | representative run | config/log/checkpoint | eval/metric surface | interpretation |
| --- | --- | --- | --- | --- |
| Default B2 | `logs/inscene15k_streaming/runs/inscene15k_streaming_b2_dino_p8_layer0` | run_dir, train_log, checkpoint, summary, aggregate_metric | `logs/inscene15k_streaming/runs/inscene15k_streaming_b2_dino_p8_layer0/eval/val_summary.json` | Has explicit `eval/val_summary.json` and `eval/val_predictions.pt`; angle metrics are recoverable from per-run summaries. |
| Wan/Cog Layer Sweep | `logs/inscene15k_streaming/runs/wan_cog_layer_probe_ab_20260725_0402/wan_cog_layer_probe_ab_20260725_0402_ego_belief_cogvideox_p8_layer0` | run_dir, train_log, checkpoint, aggregate_metric | `logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.tsv` | Training/checkpoint evidence is present, but historical aggregate rows are provisional unless matched to canonical registry. |
| VideoMAE/V-JEPA | `logs/inscene15k_streaming/runs/vjepa2_videomae_a1_mask_20260812_1909/vjepa2_videomae_a1_mask_20260812_1909_streaming_mask_vjepa2_p8_layer0` | run_dir, train_log, checkpoint | `` | Hydra config, Lightning log, checkpoint, and visualizations exist; separate eval summaries are generally missing. |
| MiniMax-H3 | `logs/inscene15k_streaming/runs/minimax_h3_a1_pose_mask_20260812_1905/minimax_h3_a1_pose_mask_20260812_1905_streaming_camera_minimax_h3_p8_layer10` | run_dir, train_log | `` | Run directories/logs are present, but representative runs show partial lifecycle without completion/checkpoint evidence. |
| LingBot Video | `logs/inscene15k_streaming/runs/probe_fill_lingbot_20260814_1928/probe_fill_lingbot_20260814_1928_view_consistency_lingbot_p8_layer10` | run_dir, train_log, checkpoint | `` | Training-complete runs and checkpoints exist; no canonical or eval-summary metric surface was found in this pass. |
