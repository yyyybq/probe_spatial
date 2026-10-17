# Repository Cleanup Report

## Kept Active

- `PROJECT_STATUS.md`
- `docs/task_status_dashboard.md`
- `experiments/task_status_1096.csv`
- `results/paper_ready_results.csv`
- `docs/FILE_MAP.md`
- `results/canonical/*.csv`
- `results/atomic_probe_coverage.csv`
- `experiments/evaluation_campaign_index.csv`

## Active Candidates

- `experiments/expected_experiments.csv`
- `experiments/registry_reconstructed_candidate.csv`
- `experiments/supersession_candidate.csv`
- `experiments/wan_cog_lineage_resolution.csv`
- `results/reconstructed/*`
- `metadata/probe_taxonomy.csv`
- default B2 candidate inputs and current evaluation mapping inputs remain in place.

## Archived

- `docs/historical_reconstruction_report.md` -> `archive/consolidation_history/20260817/historical_reconstruction_report.md` -> replacement: docs/FILE_MAP.md; experiments/task_status_1096.csv
- `docs/deep_artifact_archaeology.md` -> `archive/consolidation_history/20260817/deep_artifact_archaeology.md` -> replacement: docs/FILE_MAP.md; experiments/task_status_1096.csv
- `docs/artifact_layout_comparison.md` -> `archive/consolidation_history/20260817/artifact_layout_comparison.md` -> replacement: docs/FILE_MAP.md
- `docs/coverage_default8.md` -> `archive/consolidation_history/20260817/coverage_default8.md` -> replacement: docs/task_status_dashboard.md
- `docs/coverage_wan_cog.md` -> `archive/consolidation_history/20260817/coverage_wan_cog.md` -> replacement: docs/task_status_dashboard.md; docs/wan_cog_lineage_resolution.md
- `docs/coverage_videomae_vjepa.md` -> `archive/consolidation_history/20260817/coverage_videomae_vjepa.md` -> replacement: docs/task_status_dashboard.md
- `docs/coverage_minimax_h3.md` -> `archive/consolidation_history/20260817/coverage_minimax_h3.md` -> replacement: docs/task_status_dashboard.md
- `docs/coverage_lingbot_video.md` -> `archive/consolidation_history/20260817/coverage_lingbot_video.md` -> replacement: docs/task_status_dashboard.md
- `experiments/unresolved_660_reclassified.csv` -> `archive/consolidation_history/20260817/unresolved_660_reclassified.csv` -> replacement: experiments/task_status_1096.csv
- `experiments/incomplete_reconciliation.csv` -> `archive/consolidation_history/20260817/incomplete_reconciliation.csv` -> replacement: experiments/task_status_1096.csv
- `experiments/aggregation_missing_candidate.csv` -> `archive/consolidation_history/20260817/aggregation_missing_candidate.csv` -> replacement: experiments/task_status_1096.csv
- `experiments/rerun_needed_candidate.csv` -> `archive/consolidation_history/20260817/rerun_needed_candidate.csv` -> replacement: experiments/task_status_1096.csv
- `experiments/rerun_review_candidate.csv` -> `archive/consolidation_history/20260817/rerun_review_candidate.csv` -> replacement: experiments/work_queue/
- `experiments/eval_missing_372_reconciliation.csv` -> `archive/consolidation_history/20260817/eval_missing_372_reconciliation.csv` -> replacement: experiments/evaluation_campaign_index.csv; experiments/task_status_1096.csv
- `experiments/evaluation_only_needed.csv` -> `archive/consolidation_history/20260817/evaluation_only_needed.csv` -> replacement: dated pre-full-eval snapshot; active campaign protected
- `metadata/project_inventory.csv` -> `archive/consolidation_history/20260817/project_inventory.csv` -> replacement: metadata/file_catalog.csv
- `docs/project_inventory_20260817.md` -> `archive/consolidation_history/20260817/project_inventory_20260817.md` -> replacement: metadata/file_catalog.csv; docs/FILE_MAP.md
- `docs/worktree_cleanup_report.md` -> `archive/consolidation_history/20260817/worktree_cleanup_report.md` -> replacement: docs/repository_cleanup_report.md

## Deleted

- Deleted 23 Python bytecode cache files from `scripts/__pycache__/`; these are generated cache artifacts and contain no unique provenance.

## Investigate

- `syncRecord.txt`: source/use unclear; retained.
- untracked backend implementations under `features/lingbot/`, `features/minimax_h3/`, and related configs/scripts: retained as potentially unique implementation/provenance.
- historical logs/manifests/checkpoints outside the archived snapshot set: retained unless explicitly categorized later.

## Protected Active Evaluation

- `logs/probing_jobs_full_missing_eval_20260817`
- `logs/probing_jobs_full_missing_a1_eval_20260817`
- `logs/inscene15k_streaming/runs/full_missing_eval_20260817`
- `logs/inscene15k_streaming/runs/full_missing_a1_eval_20260817`
- `logs/remote_eval_20260817`
- `logs/inscene15k_streaming/eval_jobs`
- `logs/inscene15k_streaming/eval_a1_jobs`

## User Entry Points

- `PROJECT_STATUS.md`
- `docs/task_status_dashboard.md`
- `experiments/task_status_1096.csv`
- `results/paper_ready_results.csv`
- `docs/FILE_MAP.md`
