# Historical Experiment Reconstruction Report

This is a candidate reconstruction. It does not overwrite `experiments/registry.csv` or canonical result tables.

## Universe Size

- Expected configurations: 1096
- Candidate reconstructed rows: 1096
- Existing registry rows: 949
- Candidate loose configurations found in old registry: 437
- Candidate loose configurations not represented in old registry: 659
- Old-registry loose configurations not represented in candidate: 422

## Reconstructed Status Counts

- COMPLETE_CANONICAL: 240
- COMPLETE_RAW_ONLY: 34
- EVAL_COMPLETE_METRIC_UNRESOLVED: 162
- FAILED: 59
- JOB_DECLARED: 4
- TRAIN_COMPLETE_EVAL_MISSING: 499
- TRAIN_RUNNING: 44
- TRAIN_RUNNING_OR_PARTIAL: 54

## Campaign Coverage Counts

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

## Reclassification Of Previous INCOMPLETE Rows

- COMPLETE_RAW_ONLY: 31
- EVAL_COMPLETE_METRIC_UNRESOLVED: 162
- TRAIN_RUNNING_OR_PARTIAL: 1

## Missing Vs Aggregation

- `COMPLETE_RAW_ONLY` and `COMPLETE_SUMMARY_ONLY` are not rerun requirements by default; they require aggregation/canonicalization or provenance review.
- `TRAIN_COMPLETE_EVAL_MISSING` usually means evaluation/canonicalization is missing, not training.
- `TRAIN_RUNNING_OR_PARTIAL`, `JOB_DECLARED`, and `EXPECTED_ONLY` are rerun candidates only after manual review of the listed evidence.
- `EVAL_COMPLETE_METRIC_UNRESOLVED` means a metric-like value exists but must not be used as formal best until provenance/protocol is resolved.
- Candidate rows that appear to need aggregation/canonicalization: 34
- Candidate rows currently marked rerun_needed=True: 59
- Candidate rows requiring deeper review before any rerun decision: 763

## Direct Evidence Example

Example query: historical F3R default-layer C3 at prefix 8.

- experiment_id: `default8_counterfactual_f3r_default_layer_benchmark_l24_p8_hall_s42`
- config: `configs/experiment/inscene15k_streaming/counterfactual_f3r_v1.yaml`
- run artifact: `logs/inscene15k_streaming/runs/inscene15k_streaming_c3_f3r_p8_layer24`
- raw metric: `logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_raw_metrics_with_extra.csv`
- summary: `logs/inscene15k_streaming/runs/inscene15k_streaming_c3_f3r_p8_layer24/eval/val_summary.json`
- canonical status: `COMPLETE_CANONICAL`
- metric: `intervention_validity=0.48462664714494874`

## Canonical Completeness

- Current canonical tables are complete for default A1/A2 and canonicalized C1/C2/C3 rows.
- Current B2 canonical table is incomplete for the default 8-model benchmark angle metric; default B2 has raw/summary evidence but needs angle canonicalization review.
- VideoMAE/V-JEPA and LingBot multi-layer campaigns mostly have training/checkpoint evidence but no matched eval summary/canonical rows in this archaeology pass.
- MiniMax-H3 has job/run evidence but representative artifacts are partial; it should not be merged into the 7-probe matrix without an explicit manifest.
- Wan/Cog provisional rows are separated as metric-unresolved and should not be described as formal best.

Machine-readable files:

- `experiments/expected_experiments.csv`
- `experiments/registry_reconstructed_candidate.csv`
- `archive/consolidation_history/20260817/incomplete_reconciliation.csv`
- `archive/consolidation_history/20260817/unresolved_660_reclassified.csv`
- `archive/consolidation_history/20260817/aggregation_missing_candidate.csv`
- `archive/consolidation_history/20260817/rerun_needed_candidate.csv`
- `archive/consolidation_history/20260817/rerun_review_candidate.csv`
- `experiments/supersession_candidate.csv`
- `results/reconstructed/default8_raw_index.csv`
- `results/reconstructed/recovered_raw_metrics.csv`
- `results/reconstructed/default8_b2_angle_candidate.csv`
