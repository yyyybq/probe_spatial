# Deep Artifact Archaeology

This pass performs lifecycle classification from shallow, fixed artifact locations: job manifests, run directories, Hydra config, Lightning logs, checkpoint folders, eval summaries, and aggregate metric tables.

## Lifecycle Counts

- COMPLETE_CANONICAL: 240
- COMPLETE_RAW_ONLY: 34
- EVAL_COMPLETE_METRIC_UNRESOLVED: 162
- TRAIN_COMPLETE_EVAL_MISSING: 499
- TRAIN_RUNNING: 44
- TRAIN_RUNNING_OR_PARTIAL: 54
- JOB_DECLARED: 4
- FAILED: 59

## Campaign Counts

- default8 / COMPLETE_CANONICAL: 192
- default8 / COMPLETE_RAW_ONLY: 31
- default8 / TRAIN_RUNNING_OR_PARTIAL: 1
- lingbot_video_layer_benchmark / FAILED: 20
- lingbot_video_layer_benchmark / TRAIN_COMPLETE_EVAL_MISSING: 70
- lingbot_video_layer_benchmark / TRAIN_RUNNING: 36
- lingbot_video_layer_benchmark / TRAIN_RUNNING_OR_PARTIAL: 14
- minimax_h3_layer_benchmark / JOB_DECLARED: 4
- minimax_h3_layer_benchmark / TRAIN_COMPLETE_EVAL_MISSING: 130
- minimax_h3_layer_benchmark / TRAIN_RUNNING_OR_PARTIAL: 26
- videomae_vjepa_layer_sweep / FAILED: 35
- videomae_vjepa_layer_sweep / TRAIN_COMPLETE_EVAL_MISSING: 240
- videomae_vjepa_layer_sweep / TRAIN_RUNNING_OR_PARTIAL: 5
- wan_cog_layer_sweep / COMPLETE_CANONICAL: 48
- wan_cog_layer_sweep / COMPLETE_RAW_ONLY: 3
- wan_cog_layer_sweep / EVAL_COMPLETE_METRIC_UNRESOLVED: 162
- wan_cog_layer_sweep / FAILED: 4
- wan_cog_layer_sweep / TRAIN_COMPLETE_EVAL_MISSING: 59
- wan_cog_layer_sweep / TRAIN_RUNNING: 8
- wan_cog_layer_sweep / TRAIN_RUNNING_OR_PARTIAL: 8

## Representative Findings

- VideoMAE/V-JEPA representative runs include `.hydra/config.yaml`, `inscene15k_streaming.log`, checkpoints, and visualizations. Logs show `Trainer.fit` reached max epochs, but no separate eval summary/metric artifact was found in the inspected layout.
- MiniMax-H3 representative runs include configs/logs and early visualizations, but the inspected sample did not show a completion marker or checkpoint, so those rows are treated as partial/running until stronger evidence appears.
- LingBot representative runs include training-complete logs and checkpoints, but no matched eval summary/canonical metric surface was found.
- Wan/Cog representative unresolved runs include training-complete logs and checkpoints. Historical aggregate rows marked provisional are now classified as `EVAL_COMPLETE_METRIC_UNRESOLVED` rather than formal best.
- Default B2 has recoverable per-run `eval/val_summary.json` angle metrics; these are written to `results/reconstructed/default8_b2_angle_candidate.csv` for protocol review.

## MiniMax-H3 Cardinality

- MiniMax-H3 reconstructed rows: 160.
- Distinct MiniMax-H3 probes: action_dynamics, counterfactual, ego_belief_v2, path_integration, streaming_camera, streaming_depth, streaming_mask, view_consistency.
- Distinct MiniMax-H3 layers: 10, 20, 30, 40, 50.
- Distinct MiniMax-H3 prefixes: 8, 12, 16, 24.
- The 160-row count comes from the actual MiniMax-H3 job/run evidence universe, not from forcing the full 7-probe matrix. The 140/280 figures belong to campaigns whose design matrix is naturally `model x layer x prefix x 7 probes` or a doubled variant of that matrix. MiniMax-H3 therefore should be reported as its own campaign cardinality until a manifest explicitly declares the missing probe-family combinations.

## Default B2

- Default B2 candidate rows inspected: 32.
- Rows with summary path: 31.
- These are evaluation artifacts, but they still appear to be validation summaries; they should not be called held-out test results without a frozen split/protocol manifest.

## Machine-Readable Outputs

- `archive/consolidation_history/20260817/unresolved_660_reclassified.csv`
- `experiments/supersession_candidate.csv`
- `results/reconstructed/recovered_raw_metrics.csv`
- `results/reconstructed/default8_b2_angle_candidate.csv`
- `archive/consolidation_history/20260817/artifact_layout_comparison.md`

Rows in unresolved reclassification output: 660.
