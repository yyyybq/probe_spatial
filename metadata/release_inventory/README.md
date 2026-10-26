# Release experiment inventory — 2026-10-11, updated 2026-10-15

**2026-10-19:** these CSVs are frozen at the 2026-10-16 state; their
`evaluation_acceptance_status` and `result_status` predate the accepted v2
evaluations (804 canonical, 316 timestep, 180 C3-from-C1 rows). The current
per-run status is `release_status_20261019.csv`, built from these files and the
accepted tables by `python3 scripts/build_release_status_20261019.py`
(`--check` verifies it is current). `retrain_queue_20261016.csv` and
`retrain_submissions_20261016.csv` list the 176 re-train runs; the 128 stage-5
rows have no job id because stage 5 is not submitted.

The three input CSVs separate **canonical V2**, **timestep**, and **other
analyses**. They are dated local evidence inventories, not scheduler queries.

Each row retains its evidence path and observation date. `COMPLETE` says the
recorded training ended; `VERIFIED` says the row's stated correctness evidence
was accepted; these are separate fields. Scheduler success alone does not
certify scientific correctness.

For canonical V2, `correctness_scope=TRAINING_IDENTITY_AND_PROTOCOL_GATE`
limits `VERIFIED` to the training identity and input canary. Its final metrics
remain unaudited, recorded separately in `evaluation_acceptance_status`.
The summary's `training_complete_and_scope_verified` is therefore not a count
of accepted scientific results.

`YES_WITH_SCOPE` identifies a scientific result candidate usable within its
declared validation/development scope. It does not certify that a public code
archive, artifact download, dependencies, or licenses are already packaged.
`NO` means identified required work remains; `UNKNOWN` means the available
evidence does not settle readiness. Low scores are not automatically invalid.

`entry_level` distinguishes `run`, `package`, and `scope`. A multi-head training
bundle, an aggregate metric row, and an entire result package are different
units. Do not sum them, and do not count validation jobs, retries, or different
scope snapshots as additional finished experiments.

## 2026-10-15 update

`canonical_v2.csv` and `timestep.csv` gained a `checkpoint_tensor_status`
column (`FINITE`, `NONFINITE_FRACTION_x.xxxx`, `NOT_CHECKED_LARGE_CHECKPOINT`,
`MISSING_FILE`, `NO_CHECKPOINT_PATH`) from a CPU sweep that deserialized every
checkpoint smaller than 600 MB named by the tables and one large A1 checkpoint
per model/probe group. Rows whose weights are non-finite are marked
`correctness_status=DEGENERATE`, `result_status=INVALID_NONFINITE_WEIGHTS`;
such a checkpoint cannot be repaired by re-evaluation.

The LingBot C rows were updated from the finished recovery jobs, and the
MiniMax-H3 all-layer campaign (132 runs, scope
`canonical_v2_minimax_all_layers_20261011`, with 36 non-canonical
`streaming_camera`/`counterfactual` runs in a `_noncanonical_probe` scope) was
appended. The 2026-10-11 tables are preserved unchanged in `archive_20261011/`.

Rebuild after new evidence (reads local files only):

```bash
python scripts/update_release_inventory_20261015.py   # needs artifacts/release_audit_20261015/*
python scripts/summarize_release_inventory.py
```

The first command's inputs are produced by the read-only tools in
`artifacts/release_audit_20261015/tools/` (campaign verification, scheduler
snapshot, checkpoint sweep). The summary records hashes of all three input
tables. Its generation time is not the observation time of the underlying
experiment status.

See [the 2026-10-15 experiment inventory](../../docs/EXPERIMENT_INVENTORY_20261015.md)
for the grouped conclusions and remaining release work, and
[the 2026-10-11 plan](../../docs/RELEASE_READINESS_20261011.md) for the
original scope definitions.
