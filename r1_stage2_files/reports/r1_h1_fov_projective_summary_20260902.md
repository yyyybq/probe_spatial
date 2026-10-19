# R1 H1 Camera, FOV Audit, and Projective Repair Pilot

## Implemented

- Added shared `canonical_camera_h1_resize_v1` utility for H1 native-to-render resize camera semantics.
- Added deterministic camera regression tests.
- Added full H1 FOV audit script.
- Added versioned projective repair pilot generator and paired renderer validation script.
- Reused the official Active Spatial HTTP renderer entrypoint on `10.119.28.8`.

## Camera Utility and Tests

- H1 effective K for native 640x480 to render 256x256 is:
  `[[128.0, 0.0, 128.0], [0.0, 170.6666666667, 128.0], [0.0, 0.0, 1.0]]`.
- Camera convention is explicit: OpenCV/COLMAP style `X=right, Y=down, Z=forward`, with `c2w` and `w2c` recorded.
- Deterministic tests passed:
  K scaling, native resolution inference, c2w/w2c round trip, left/right relation, yaw reversal, behind-camera, partial out-of-frame, non-square native-to-square render, known overlay regression.
- Real renderer equivalence passed:
  native 640x480 render resized to 256x256 vs direct H1 256x256 render had MAE `4.2661`, RMSE `13.5886`.

## FOV Audit Result

- Dataset: `exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/train_filtered.jsonl`.
- Audited all `1503` `fov_inclusion` samples under H1.
- H1 sample-target bbox valid: `1503/1503`.
- H1 sample-target visible by bbox projection proxy: `1502/1503`.
- H1 sample-target success: `1502/1503`.
- H0 vs H1 sample-target success disagreement: `1159/1503`.
- H1 initial success: `1503/1503`.
- Decision: `FOV-B`.
- Reason: H1 target poses are mostly valid, but initial states are already successful for all samples under the current bbox-based FOV metric. Do not delete old data in this round; treat FOV as requiring a later filtering/regeneration decision before training use.
- True RGB overlay: `50` H1 overlays from real renderer, scene `0226_840298`. Visual review found no systematic H1 bbox offset, but several 3D bbox truncation/wide-box cases remain.

## Projective Pilot Result

- Generator version: `projective_canonical_h1_v1`.
- Camera model: `canonical_camera_h1_resize_v1`.
- Requested pilot count: `120`.
- Generated accepted pilot samples: `110`.
- Rejected candidate repairs: `10`, all due to insufficient margin/score.
- Old H1 target relation success on accepted pilot lineage: `0/110`.
- New H1 target relation success: `110/110`.
- Relation distribution: `55 left`, `55 right`.
- Scene distribution: `0014_841007: 22`, `0016_840873: 22`, `0226_840298: 24`, `0228_840300: 18`, `0300_840573: 24`.
- Real renderer paired validation saved as `old target | new repaired target` contact sheet. Spot checks show old target has negative canonical margin and new target has matched positive margin with bboxes still on the same object pair.

## Old-vs-New Difficulty Comparison

- Translation distance mean: old `4.6229`, new `5.1554`.
- Yaw delta mean: old `74.31 deg`, new `99.01 deg`.
- Relation margin mean: old `-44.92 px`, new `+44.83 px`.
- Min bbox area mean: old `1851.50 px`, new `1788.81 px`.
- Target score mean: old `0.0278`, new `0.9734`.
- New pilot did not collapse to extreme object area or tiny separation, but yaw distribution shifted upward. Full regeneration should preserve and report this distribution explicitly.

## Occlusion Feasibility

- Current official renderer path returns RGB images only through HTTP.
- Underlying `gsplat.rendering.rasterization` supports `render_mode` values including `D`, `ED`, `RGB+D`, and `RGB+ED`, and returns a second tensor that can expose alpha-like output.
- The current `GaussianRenderer.render_image_from_cam_param()` discards non-RGB outputs, and the HTTP protocol/worker encodes only RGB image bytes.
- There is no low-cost object-ID or instance-mask output in the current renderer path.
- R1-v1 occlusion should remain the existing projection/depth proxy unless a versioned renderer API extension is implemented and separately validated. Do not claim true visible-fraction oracle yet.

## Files Changed

- `vagen/envs/active_spatial/canonical_camera.py`
- `scripts/r1_camera_utility_tests.py`
- `scripts/r1_fov_h1_audit.py`
- `scripts/r1_projective_repair_pilot_h1.py`

Existing dirty worktree files unrelated to this task were not modified by this round.

## New Artifacts

- `exps/vagen_active_spatial/r1_observation_aligned_reward_audit/r1_h1_fov_projective_20260902/phase0_renderer_validation/`
- `exps/vagen_active_spatial/r1_observation_aligned_reward_audit/r1_fov_h1_audit_20260902/`
- `exps/vagen_active_spatial/r1_observation_aligned_reward_audit/r1_projective_repair_pilot_h1_v1/`

## Remaining Risks

- FOV target pose validity is strong, but FOV initial-success degeneracy is severe under the current metric.
- Projective pilot accepted `110/120`; full regeneration needs retry/failure accounting, not blind expansion.
- Projective yaw distribution shifted higher in the repair pilot.
- Overlay and visibility checks still use projected 3D bboxes, not instance masks.
- Some real GS renders have dark/blurred splat artifacts, so visual review should focus on object-pair consistency and systematic K offset.

## Gate Decision

`PASS - EXPAND PROJECTIVE REGENERATION`

This only authorizes the next projective data-regeneration step after review. It does not authorize RL/PPO/SFT training and does not authorize full FOV deletion/filtering.

## Exact Next Step

Review the pilot contact sheet and mapping, then run a full 2003-row projective regeneration using the same `projective_canonical_h1_v1` pipeline with failure/retry logging and the same H1 validation gates. Do not start training.
