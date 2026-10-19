# Probe Spatial — Project Status

**Current handoff (2026-10-19):** [docs/HANDOFF_20261019.md](docs/HANDOFF_20261019.md).
**Independent audit and per-run release status (2026-10-19 10:00 UTC):**
[docs/RELEASE_AUDIT_20261019.md](docs/RELEASE_AUDIT_20261019.md) and
`metadata/release_inventory/release_status_20261019.csv`; the counts in the body
below predate the 10-17/18 evaluations.
The page below is the 2026-10-16 snapshot. Canonical V2 (804), timestep (316)
and C3-from-C1 (180) evaluations have since been accepted; MiniMax L20/L25
bf16 consolidation and full verification passed on 2026-10-19 (job pt-f4g9bk61;
0 problems), all 48 B2 retrain final checkpoints are finite, and the 16-group
stage-5 dry run is clean (errors=0, HEAD 1e35916); the stage-5 pilot is not yet
submitted ([execution record](docs/MINIMAX_REEXTRACT_RETRAIN_EXECUTION_20261016.md)).

Evidence reviewed **2026-10-15**, corrected **2026-10-16** (local artifacts, one
scheduler snapshot at 22:5x UTC, two read-only GPU feature audits). This is not a live scheduler
report. Start with the [2026-10-15 experiment inventory](docs/EXPERIMENT_INVENTORY_20261015.md),
the [2026-10-15 code audit](docs/RELEASE_AUDIT_20261015.md) and the
[per-experiment CSVs](metadata/release_inventory/README.md). The
[2026-10-11 plan](docs/RELEASE_READINESS_20261011.md) keeps the scope
definitions; its counts are superseded where this page differs.

## What changed since 2026-10-11

- **2026-10-16 execution of the re-extraction/retrain plan.** Stage 0 merged the
  bf16 MiniMax storage contract into the main extractor and fixed the sync loop
  for brand-new namespaces (CPU suite: 472 passed); frozen runtime
  `runtime/tstep_minimax_bf16_v1_extractor_r2`. The 1-scene extraction canary
  passed on its fourth attempt (bf16 values identical to fp16 wherever fp16 was
  finite); the 32 extraction shards were submitted at 14:04 UTC. At 15:14 UTC no
  job had failed: B2 retrains (`PROBE_FP16_PASSTHROUGH=0`) 2 finished / 46 running,
  and both finished final checkpoints are finite (0 of 18,532,225 parameters);
  extraction shards 4 running / 28 queued behind a full Zoetrope pool (streaming
  shards take ~16 h at the measured ~11 min/scene). Consolidation, full
  verification and stage-5 tooling are written and tested (CPU suite: 487 passed,
  3 skipped pending the staged `run_tstep_probe_group.sh.stage5`); the 128 stage-5
  configs match the replaced runs key-for-key. Stage 5 is not approved or
  submitted. Ledger: [execution record](docs/MINIMAX_REEXTRACT_RETRAIN_EXECUTION_20261016.md).

- **Checkpoint weights were deserialized for the first time.** Of 1,272
  checkpoints named by the canonical-V2 and timestep tables, 906 were loaded
  (all 880 below 600 MB plus 26 sampled large A1 files); **143 contain
  non-finite parameters**: 16 canonical B2 heads (CogVideoX layers 0/5/10 and
  Aether layer 1) and 127 timestep runs (all 48 CogVideoX/MiniMax B2 heads and
  79 MiniMax-H3 layer-20/25 heads). Those runs had been counted as complete;
  their weights cannot be rescued by re-evaluation.
- **MiniMax-H3 fp16 timestep caches overflow at layers 20/25** (inf entries,
  |x| at the fp16 ceiling) in `FEAT_STREAMING/FEAT_TARGET_CANONICAL_V2_SEEDED_V1`
  (t500/t1000); layer 15 is clean. All 128 timestep MiniMax runs at those layers
  are invalid until the caches are re-extracted in bf16. The canonical MiniMax
  runs read the bf16 `MINIMAX_BF16_V2` namespaces, whose sampled files are all
  finite (job `pt-ljbe3zu0`, 2026-10-16); a 10-15 note that blocked them was
  wrong and has been withdrawn.
- **LingBot C40 finished** (40/40 complete, finite weights); LingBot is 120/120.
- **MiniMax-H3 all-layer campaign** (132 runs submitted 2026-10-11, 95 migrated
  to the H800 pool on 2026-10-12): 129 complete, 3 running. 36 of the 132 are
  `streaming_camera`/`counterfactual` runs outside the six-probe protocol and
  are tracked in a separate scope.
- Training now aborts on NaN/inf feature batches; both evaluators flag
  non-finite predictions instead of scoring them (CPU suite: 463 passed).

## Canonical V2

Original explicit matrix: **736 runs** (main552 + Wan168 + MiniMax L30 A/B16),
now **736 complete**. Identity-and-canary verified: **692** (668 on 10-11, plus
the 40 LingBot C runs, minus 16 B2 runs whose weights are non-finite). Still
unresolved: 28 Wan A2 runs with conflicting canary evidence.

Appended scope `canonical_v2_minimax_all_layers_20261011`: 96 six-probe runs
(95 complete, 1 running) and 36 non-canonical runs (34 complete, 2 running);
no per-run protocol canary was located for them, so they stay NOT_VERIFIED.

No canonical-V2 official metric has been accepted or exported. See the
[model/layer/probe matrix](docs/CANONICAL_V2_INVENTORY_20261011.md) for the
10-11 cell layout and the 10-15 inventory for the corrections.

## Timestep

**576/576 trainings ended**, but **160 are now DEGENERATE**: 127 with
non-finite weights and 33 more MiniMax layer-20/25 runs whose inputs overflow.
The 216 matched official validation evaluations split into 136 candidates
(Wan 72, CogVideoX A1/A2 48, MiniMax layer-15 A1/A2 16) and 80 rows that are
invalid rather than "degenerate scores": every CogVideoX/MiniMax B2 row and
every MiniMax layer-20/25 row. Source commits remain `unknown` in all 576
manifests. See the [timestep matrix](docs/TIMESTEP_INVENTORY_20261011.md).

## Other analyses

Unchanged since 2026-10-11: six result packages (Independent P8, continuous-action
relational V2, SE3 final r3, probe-validity paper, D-series Wan D1/D2, D-series
V-JEPA2 D1/D3) are accepted within their val/dev scope; V-JEPA2 D2 needs the
three-seed aggregate and unified export; C3-from-C1 V1, B2 P16 extension,
Active representation MVP and SeeSE3 replication lack complete results. See the
[analysis inventory](docs/ANALYSIS_INVENTORY_20261011.md).

## Historical ledger

`experiments/task_status_1096.csv` (1,260 rows) and
`results/paper_ready_results.csv` are separate historical scopes; see the
[earlier audit](docs/RESULTS_STATUS_AUDIT_20261011.md). They were not changed.

## Release state

Code fixes and verification are in the [2026-10-15 audit](docs/RELEASE_AUDIT_20261015.md).
Remaining blockers, by kind, are listed in the
[2026-10-15 inventory](docs/EXPERIMENT_INVENTORY_20261015.md): bf16
re-extraction of the timestep MiniMax-H3 layer-20/25 caches and retraining of
those 128 runs plus the 48 NaN B2 heads (plan:
[MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md](docs/MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md),
queue: `metadata/release_inventory/retrain_queue_20261016.csv`), acceptance and
export of the finite canonical/timestep cells, vendored-license recovery,
removal of personal paths, and an explicit release commit that includes the
untracked source files.
