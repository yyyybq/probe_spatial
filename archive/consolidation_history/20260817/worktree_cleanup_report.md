HISTORICAL SNAPSHOT - NOT AUTHORITATIVE

# Worktree Cleanup Report

No files were deleted in this consolidation pass.

Machine-readable inventory:

- `metadata/project_inventory.csv`

Human-readable inventory:

- `docs/project_inventory_20260817.md`

## Version-control Required

Recommended to commit:

- `PROJECT_STATUS.md`
- `protocols/C_PROBES.md`
- `experiments/registry.csv`
- `metadata/project_inventory.csv`
- `results/canonical/*.csv`
- `scripts/build_experiment_registry.py`
- `scripts/build_canonical_results.py`
- `scripts/check_project_consistency.py`
- `scripts/build_project_inventory.py`
- New consolidation docs under `docs/`

Also likely required, but marked `INVESTIGATE` where untracked or touched before this pass:

- New backend source/config/scripts under `features/lingbot/`, `features/minimax_h3/`, `configs/experiment/inscene15k_streaming/*lingbot*`, `*minimax_h3*`, and related launcher/monitor scripts.
- Generated C manifests under `manifests/c_retrain_20260722/`.

## Artifact / Archive

Do not commit large raw logs/checkpoints/features directly. Archive or retain externally with provenance:

- `logs/inscene15k_streaming/**`
- `logs/probing_jobs_*`
- `logs/current_audit_20260817/**`

Small raw metric tables that feed the registry should be archived with hashes or copied to a durable artifact store:

- `logs/eval_results/results_table.csv`
- `logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_raw_metrics_with_extra.csv`
- `logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.tsv`

## Ignore

Already ignored or safe to keep ignored:

- Python caches
- `*.pt`, `*.pth`, `*.ckpt`
- `logs/`
- feature tensors and dataset bundles

`.gitignore` continues to keep heavy logs and checkpoints out of git. Canonical small CSVs now live under `results/canonical/` so they are not hidden by the `logs/` ignore rule.

## Investigate

Do not delete these without owner review:

- Existing modified source files under `vidfm3d/`, `features/`, and `configs/`
- Untracked backend implementations/configs/scripts
- `features/run_inscene15k_mllm.py` deletion
- `syncRecord.txt`

## Delete Candidates

No delete candidates were actioned. Deletion should only happen after canonical replacement and provenance are confirmed.
