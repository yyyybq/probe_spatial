# Phase 0 audit — `actionability_algebra_p8_20260831T030110Z_2075f3eb`

## Outcome

The scientific pilots are **not runnable with valid inputs**. The DINO smoke stopped at preflight; no legacy checkpoint was loaded and no model prediction was produced. The independent 9D action-convention audit passed.

## Frozen scope

- Models/settings: DINO L0, F3R L24, V-JEPA2 L23, Wan L20; prefix 8 only.
- Pilot 1: goal-conditioned one-step actionability.
- Pilot 2: evaluation-only action algebra/cycle consistency.
- No backbone extraction, no C1/C2 retraining, no layer sweep, no unrelated science task.
- Science tag: `actionability_algebra_p8_20260831T030110Z_2075f3eb`; git identity: `2075f3eb270cb0e8907c3927b6b079c64965b75a`.

## Hard-gate evidence

1. Q01 reports all 99 evaluation-only configurations as `EVAL_BLOCKED_UNAUDITABLE`. The requested default p8 C1/C2 rows are `CANONICAL_UNAUDITABLE`; DINO/F3R/V-JEPA2 feature lineages are `MANIFEST_MISSING`, and Wan L20 lacks a proven-exact paired C1/C2 lineage.
2. The Wave-B raw-ID audit contains 2,208 associations: 1,206 exact and 1,002 window-ID collisions. At Wan p8 specifically it has 120 exact and 156 collisions. Window/tail equality is therefore not raw-frame identity.
3. The configured local `FEAT_*_BY_PREFIX` roots for all four requested settings are absent on this node. Remote provenance is not substituted for an exact local canary.
4. Wan's manifest-exact versioned cache covers C1 layers 0/5/10/15/25/29, not fixed layer 20, and has no clean C2. Its later h1/h2/h4 extraction is only an 8-sample/2-scene canary at those same non-L20 layers.
5. Project Python exists at `/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python3.11` with torch `2.6.0+cu124`, but this node reports CUDA unavailable and zero devices.

## Frozen data contract

- Canonical manifest: `manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl`; SHA256 `772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c`.
- Sample key: `sample_id`, shaped `<scene>_w<window>_p8`.
- Split inventory: train 2,448 samples / 612 scenes; val 276 / 69; no test rows.
- Required input roles remain `streaming_prefix` for the anchor and manifest-exact `target_isolated` for endpoints/goals. A historical final stream element is never accepted as a target substitute.

## Checkpoint decision

The eight requested-setting checkpoint files are readable and hashed in `CHECKPOINT_INVENTORY.csv`, but are quarantined rather than reused. Readability and an old result row do not establish raw-ID-exact training/evaluation lineage. The six clean Wan C1 alternatives are inventoried but not substituted because the fixed setting is L20 and layer sweeps are outside scope.

## Gate result

- DINO smoke: `BLOCKED_MISSING_ARTIFACT` before checkpoint load.
- Four-model actionability: `BLOCKED_MISSING_ARTIFACT`; direct ranking is additionally `BLOCKED_ENVIRONMENT` on this node.
- Four-model algebra: `BLOCKED_MISSING_ARTIFACT`.
- Action convention: `COMPLETE_VALID` (7/7 tests).

Primary evidence: `reports/q00_phase1_direct_node_execution_log.md`, `results/analysis/q01_feature_semantics/summary.json`, `results/analysis/q01_feature_semantics/universe_1096_feature_semantics.csv`, `reports/wan_p8_canonical_clean_rerun_wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e.md`.
