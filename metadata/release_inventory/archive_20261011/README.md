# Release experiment inventory — 2026-10-11

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

Rebuild the combined CSV and per-group counts after updating input inventories:

```bash
python scripts/summarize_release_inventory.py
```

The summary records hashes of all three input tables. Its generation time is
not the observation time of the underlying experiment status.

See [the release readiness plan](../../docs/RELEASE_READINESS_20261011.md) for
the grouped conclusions, unresolved results, and remaining code-release work.
