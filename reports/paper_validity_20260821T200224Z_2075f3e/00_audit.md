# Phase 0 audit

- Run tag: `paper_validity_20260821T200224Z_2075f3e`
- Git: `2075f3eb270cb0e8907c3927b6b079c64965b75a` on `main`; dirty before this task: yes.
- Frozen split: `configs/splits/inscene15k_v1.json` (`e01a2194143405ed8f08613ec0bc672c7c237fa9bb21d2b8da441e7ad56bf086`). Scene-level split code runs before window expansion.
- Current node: no CUDA device. `INSCENE_DATA_ROOT`, `INSCENE_STREAMING_FEAT_ROOT`, `INSCENE_TARGET_FEAT_ROOT`, `INSCENE_MANIFEST`, and `INSCENE_SPLIT_MANIFEST` were unset.
- Historical roots resolve to `fj:s3://.../FEAT_STREAMING_BY_PREFIX/...` and `FEAT_TARGET_BY_PREFIX/...`; read-only AOSS checks timed out after 45 s and `/tmp/inscene_aoss_cache` was empty.
- `features/run_inscene15k.py`, `scripts/build_inscene_streaming_manifest.py`, and `docs/c_retrain_session_report_20260724.md` are currently empty/one-byte user-tree files. This blocks safe feature extraction and invalidates assumptions based on their historical contents.
- Repository `manifests/` is owned by `nobody:nogroup` with mode `0755`; both normal and approved targeted `mkdir` failed. New common manifests therefore live under `reports/paper_validity_20260821T200224Z_2075f3e/manifests/` rather than modifying permissions or old manifests.

## B2 definition found

Standard B2 chooses an object visible at `>=200` depth-valid pixels in history and treats `<200` in the target as hidden. `compute_object_target_for_id` then averages all valid object points, including residual target pixels. The Transformer reads every target/current patch. No query-only control existed before this run. Therefore all existing B2 checkpoints are standard, not strict-zero-hidden.

This run adds a separate strict path: raw target count exactly zero, positive raw instance IDs, coordinate-wise median over finite past-only points, and model-independent fixed-target labels. It does not relabel old checkpoints or results.

## C1/C2 definition found

The safe 2026-07 retraining resolved configs use independently extracted `streaming_prefix` inputs and exact `target_isolated` roots. C1's stored `no_action` and `shuffled_action` are evaluation interventions (`zeros` and batch roll), not retrained visual-only/action-only or magnitude-matched shuffle. Stored copy is contextual last observation, not isolated anchor copy. C2 stores no baselines.

## Artifact conclusions

- Inventory rows: 96; selected C checkpoint provenance rows: 64.
- Legacy C1/C2 per-sample dumps have no canonical IDs. This run reconstructs IDs and verifies intersections, but cross-prefix pairing is only `legacy_scene_order` quality.
- Existing B2 dumps omit object/window/sample IDs and strict visibility metadata, so strict common intersection and strict-eval cannot be recovered offline.
- New full workload: 32 strict B2 trains + 32 query-only trains + 64 C1 visual/action-only trains + 64 C2 visual/action-only trains = 192 training jobs, followed by strict/full/matched evaluation. This was not launched without a passing data/GPU smoke gate.
