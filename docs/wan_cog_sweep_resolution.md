# Wan/Cog Sweep Resolution

Source artifacts:

- Registry: `experiments/registry.csv`
- Raw sweep table: `logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.tsv`
- Canonical B2 table: `results/canonical/b2_results.csv`
- Noncanonical remainder: `logs/current_audit_20260817/wan_cog_remaining_nonformal_jobs.tsv`

## Lifecycle Reconciliation Rule

Historical labels such as `formal_best`, `latest_provisional`, `REPAIRED_BEST`, and `TRULY_MISSING` are not experiment lifecycle states. They are preserved only in `status_reason` and `notes`.

Registry mapping:

- Non-provisional rows with parseable metric: `COMPLETE`
- Provisional rows, nearby-only rows, missing-class rows, and rows without parseable canonical metric: `INCOMPLETE`
- Future reruns should either supersede incomplete rows with a new `COMPLETE` row or mark unrecoverable rows `FAILED`/`INVALID` with evidence.

## Current Counts

From the generated registry:

- Wan/Cog layer-sweep rows in registry: 213
- `COMPLETE`: 51
- `INCOMPLETE`: 162

By probe/model:

- `ego_belief_v2/cogvideox`: 26 complete, 6 incomplete
- `ego_belief_v2/wan`: 22 complete, 2 incomplete
- `ego_belief/wan`: 3 complete, 21 incomplete
- `ego_belief/cogvideox`: 25 incomplete
- `streaming_depth/cogvideox`: 32 incomplete
- `streaming_depth/wan`: 20 incomplete
- `view_consistency/cogvideox`: 32 incomplete
- `view_consistency/wan`: 24 incomplete

## Canonical Status

B2 is now canonicalized for formal Wan/Cog rows:

- `results/canonical/b2_results.csv`: 48 B2 rows
- `logs/current_audit_20260817/b2_canonical_angle_formal_only.csv`: previous audit table retained as supporting artifact

Other Wan/Cog layer-sweep probes remain mostly incomplete and should not be reported as formal best.

## Remaining Rerun / Reconciliation Set

The authoritative machine-readable list is:

- `logs/current_audit_20260817/wan_cog_remaining_nonformal_jobs.tsv`

Rows in that file need one of:

1. Exact/best checkpoint location and re-evaluation under the current protocol.
2. A new rerun producing a parseable metric.
3. An explicit `FAILED` or `INVALID` registry update if unrecoverable.

## Expected Follow-up

Do not edit the historical TSV in place. Add new rows to the registry or regenerate the registry from a new raw artifact that supersedes the old configuration. Canonical tables must be regenerated through `scripts/build_canonical_results.py`.
