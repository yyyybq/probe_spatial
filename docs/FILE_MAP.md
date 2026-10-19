# File Map

Navigation reviewed **2026-10-15**. Start with [the 2026-10-15 audit](RELEASE_AUDIT_20261015.md),
[the 2026-10-15 experiment inventory](EXPERIMENT_INVENTORY_20261015.md), then the
2026-10-11 [release audit](RELEASE_AUDIT_20261011.md) and
[three-part plan](RELEASE_READINESS_20261011.md) for scope definitions. The older
ledger/dashboard/export are historical snapshots, not a live status feed.

# Start Here

- [2026-10-19 handoff](HANDOFF_20261019.md): current project purpose and latest progress. Read this first.
- `PROJECT_STATUS.md`: 2026-10-16 evidence snapshot; its counts are superseded by the handoff.
- [2026-10-15 inventory](EXPERIMENT_INVENTORY_20261015.md): per-group completion, checkpoint tensor validity, feature-cache overflow, remaining work by kind.
- [Release plan](RELEASE_READINESS_20261011.md): canonical V2, timestep, other analyses, and publication requirements (2026-10-11 counts superseded where the 10-15 inventory differs).
- [Per-experiment inventory](../metadata/release_inventory/README.md): three source CSVs (with `checkpoint_tensor_status`), combined CSV and hash-linked summary; the 10-11 tables are kept in `metadata/release_inventory/archive_20261011/`.
- `artifacts/release_audit_20261015/`: read-only evidence of the 10-15 audit (campaign verification, scheduler snapshot, checkpoint sweep, AOSS feature finiteness jobs) and the small tools that produced it.
- [MiniMax re-extraction and retrain plan](MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md): scope correction (timestep caches only), submission list by stage, GPU-hour and storage estimates; queue in `metadata/release_inventory/retrain_queue_20261016.csv` (`scripts/build_retrain_queue_20261016.py`).
- `docs/task_status_dashboard.md`: historical dashboard for the 1260 tasks (success / running / failed / pending-eval).
- `experiments/task_status_1096.csv`: row-level lifecycle ledger for the 1260 historical tasks.
- `results/paper_ready_results.csv`: historical paper-table eligibility snapshot; not final-claim-ready by itself.
- `docs/FILE_MAP.md`: navigation and provenance map when you need to trace beyond the historical status files.

# Scientific Protocols

- `protocols/C_PROBES.md`: C1/C2 definitions plus the current evaluation-only
  C3-from-C1 action-sensitivity contract, controls, and interpretation limits.
- `EXPERIMENT_PROTOCOL.md`: current experiment protocol and cache/split constraints.
- `TEMPORAL_DATA_VALIDITY.md`: temporal validity warning for old non-video Infinigen streaming caches.
- `TRAINING_LOGIC_AUDIT.md`: task input contracts and causal-cache audit notes.

# Experiment Ledger

- `experiments/expected_experiments.csv`: historical candidate universe of expected configurations (frozen to the same 1096 keys when reconstruct runs with default `--freeze-1096`).
- `experiments/task_status_1096.csv`: authoritative lifecycle ledger; 1260 task rows (legacy filename).
- `experiments/registry.csv`: older canonical-results registry, not the complete historical universe.
- `experiments/registry_reconstructed_candidate.csv`: historical candidate reconstruction, useful for P1/P3/P4, not authoritative.
- `experiments/evaluation_campaign_index.csv`: completed evaluation campaign index; historical full-missing campaigns were not folded into this index.
- `experiments/wan_cog_lineage_resolution.csv`: P2 provenance ledger for historical Wan/Cog rows.

# Results

- Raw results live in run directories, historical aggregate TSV/CSV files, and `results/reconstructed/*` candidate tables.
- Reconstructed candidate results in `results/reconstructed/*` are evidence for review, not canonical claims.
- Canonical results live in `results/canonical/*.csv` and are scoped to already accepted rows.
- Paper-ready results live in `results/paper_ready_results.csv` and mean table eligibility under current audit gates, not final scientific claim readiness.
- `raw != canonical != paper-ready != final-claim-ready`.

# Historical Work Queues

- `experiments/work_queue/training_failed_review.csv`: failed training rows for later manual review.
- `experiments/work_queue/training_partial_review.csv`: partial/interrupted rows for later manual review.
- `experiments/work_queue/not_started_or_declared_review.csv`: expected/declared but not verified training rows.
- `archive/consolidation_history/20260817/evaluation_only_needed.csv`: pre-full-evaluation snapshot only; the historical campaign was protected; this document does not establish its current runtime state.
- `experiments/supersession_candidate.csv`: Wan/Cog provisional source table used by lineage resolution.
- Fill/retry operational queues (not a second universe): `logs/probing_jobs_probe_fill_fill_20260814_1928/jobs/shards/retry_20260819/` and lingbot twin; excluded videomae p24 list under `.../jobs/EXCLUDED_videomae_p24_retry_20260819.txt`.

# Reproduction Scripts

- `scripts/reconstruct_historical_experiments.py`: rebuilds expected/candidate reconstruction; indexes nested `probe_fill_*` run dirs; default freezes outputs to the 1096 ledger keys.
- `scripts/build_task_status_ledger.py`: historical generator for the lifecycle ledger, dashboard, paper-ready snapshot, and work_queue outputs.
- `scripts/check_task_status_consistency.py`: read-only consistency check for the 1260-row historical ledger; `--require-final` adds final-claim readiness.
- `scripts/update_release_inventory_20261015.py`: apply the 10-15 local evidence (campaign verification, scheduler snapshot, checkpoint sweep) to `canonical_v2.csv`/`timestep.csv`; run before the summarizer.
- `scripts/summarize_release_inventory.py`: validate and combine the three dated inventories, preserving run/package/scope units.
- `scripts/resolve_feature_layers.py`: registered defaults per backend (`default_layer`, `last_layer`, `in_channels`, `default_timestep`) for shell entrypoints.
- `scripts/build_canonical_results.py`: canonical result builder; do not regenerate accepted tables from stale historical inputs.
- `scripts/build_full_missing_eval_manifests.py`, `scripts/run_missing_eval_shard.py`, and `scripts/run_missing_a1_eval_shard.py`: historical evaluation tooling; current execution status must come from dated receipts, not this file.

# Historical Archive

- `archive/consolidation_history/20260817/`: historical archaeology reports and intermediate snapshots retained for provenance.
- Files there are marked historical and are not current sources of truth.
- Use `archive/consolidation_history/20260817/README.md` for old path -> archived path -> replacement mapping.
