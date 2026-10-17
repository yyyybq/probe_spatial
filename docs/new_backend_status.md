# New Backend Status

Source artifacts:

- `logs/current_audit_20260817/new_backend_artifact_audit.csv`
- Backend configs under `configs/experiment/inscene15k_streaming/`
- Backend feature implementations under `features/lingbot/` and `features/minimax_h3/`
- Launch/monitor scripts under `scripts/`

Capability checklist statuses are not registry lifecycle states. They describe implementation readiness.

## MiniMax-H3

- `IMPLEMENTED`: yes. `features/minimax_h3/` and MiniMax-H3 configs/scripts exist.
- `SMOKE_TESTED`: partial/unknown from current canonical evidence.
- `RUN_ATTEMPTED`: yes. Multiple `probing_jobs_minimax_h3_*` manifests and monitors exist.
- `RAW_RESULTS_AVAILABLE`: no stable eval summaries found by the current audit.
- `CANONICAL_RESULTS_AVAILABLE`: no.

Current registry/canonical action:

- Do not report MiniMax-H3 performance.
- Keep implementation/config/job artifacts.
- Generate a minimal rerun manifest from incomplete job manifests before relaunching anything.

## LingBot

- `IMPLEMENTED`: yes. `features/lingbot/` and LingBot configs/scripts exist.
- `SMOKE_TESTED`: partial/unknown from current canonical evidence.
- `RUN_ATTEMPTED`: yes. `probing_jobs_probe_fill_lingbot_20260814_1928` exists.
- `RAW_RESULTS_AVAILABLE`: no stable eval summaries found by the current audit.
- `CANONICAL_RESULTS_AVAILABLE`: no.

Current registry/canonical action:

- Do not report LingBot performance.
- Preserve manifests and monitor logs as provenance.
- Relaunch only failed/incomplete jobs after a minimal rerun manifest is generated.

## Other New Backend / Probe-fill Runs

Probe-fill campaigns exist but are incomplete or not canonicalized:

- `probing_jobs_probe_fill_fill_20260814_1822`
- `probing_jobs_probe_fill_fill_20260814_1928`

These are artifact/campaign states, not scientific results.

## Minimal Rerun Manifest Policy

Before any new GPU work:

1. Read the original `jobs/all.tsv`.
2. Exclude jobs with verified eval summaries and canonical metrics.
3. Include only jobs with missing raw metrics, failed monitor status, or incomplete run directories.
4. Write the rerun manifest under `experiments/rerun_manifests/`.
5. Register the rerun campaign in `experiments/registry.csv` after metrics exist.
