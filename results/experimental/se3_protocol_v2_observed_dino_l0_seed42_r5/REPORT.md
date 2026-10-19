# Observed-video Grounded Order Protocol — DINO

Status: **experimental / non-canonical**.  This is the primary protocol-v2
track: all endpoint supervision uses the real observed C2 feature `z_C`; the
optional synthetic dual-image manifest is not consulted.

## Contract

- Pose convention: `E` is w2c, `T_AB=E_B@inv(E_A)`, and `T_AC=T_BC@T_AB`.
- Test tuple: real observed `(A,B,C)`; `T1=T_AB`, `T2=T_BC`.
- Training: only observed `A→B` and `B→C` one-step target loss.  No order,
  composition, cycle, path, or rollout loss is included.
- Main analysis: `z_correct=P(P(z_A,T1),T2)`,
  `z_reversed=P(P(z_A,T2),T1)`, `z_direct=P(z_A,T2@T1)`, all compared with
  the real `z_C`.
- Static/dynamic caveat: RGB capture has no per-frame static certificate; this
  is reported risk, not a renderer-derived claim.

## A. Observed-order feature grounding

| baseline | one_step_grounding_cosine | correct_endpoint_cosine | reversed_order_endpoint_cosine | grounded_order_margin_cosine | order_ranking_accuracy | direct_endpoint_cosine | recursive_direct_gap_cosine | copy_normalized_improvement |
|---|---|---|---|---|---|---|---|---|
| copy | 0.1707 | 0.2573 | 0.2573 | 0.0000 | 0.0000 | 0.2573 | 0.0000 | 0.0000 |
| without_condition | 0.1664 | 0.2318 | 0.2318 | 0.0000 | 0.0000 | 0.2310 | 0.0033 | 0.0991 |
| action_only | 0.2327 | 0.2328 | 0.2488 | 0.0160 | 0.5682 | 0.2372 | 0.0279 | 0.0952 |
| full | 0.1354 | 0.1978 | 0.2024 | 0.0047 | 0.4773 | 0.1939 | 0.0098 | 0.2315 |
| matched_wrong_pose | 0.1857 | 0.2826 | 0.2756 | -0.0069 | 0.3636 | 0.2809 | 0.0094 | -0.0980 |
| reversed_order | 0.1840 | 0.2024 | 0.1978 | -0.0047 | 0.5227 | 0.1940 | 0.0103 | 0.2133 |
| shuffled_pose | 0.1832 | 0.2679 | 0.2719 | 0.0040 | 0.5682 | 0.2744 | 0.0096 | -0.0409 |

`reversed_order` is an intentional evaluation control: its "correct" column is
the full predictor evaluated with `T2` then `T1`, so a negative margin is
expected.  `matched_wrong_pose` uses another scene's same-window action pair;
`shuffled_pose` uses independently seeded derangements.  See
`stratified_observed_order.csv` for κ/family/motion/overlap/scene rows.

`one_step_grounding_cosine` is the mean of real `A→B` and real `B→C`
endpoint distances.  It is reported beside composition so an algebraically
stable copy-like model cannot be mistaken for a grounded transition model.

## B. Two-sided pose grounding

The frozen M4 adapter is trained only on real observed one-step frame pairs and
is then used to decode each predicted two-step feature relative to real `z_A`.

| baseline | decoded_12_rotation_error_deg | decoded_12_translation_error_m | decoded_21_rotation_error_deg | decoded_21_translation_error_m | two_branch_pose_assignment_accuracy | decoded_commutator_rotation_error_deg | decoded_commutator_translation_error_m |
|---|---|---|---|---|---|---|---|
| action_only | 31.9323 | 0.2278 | 34.2922 | 0.2266 | 0.4545 | 18.5031 | 0.0858 |
| full | 30.4377 | 0.2206 | 30.7741 | 0.2209 | 0.5000 | 7.1996 | 0.0494 |
| copy | 40.4026 | 0.2377 | 40.4303 | 0.2350 | 0.0000 | 6.7646 | 0.0488 |
| matched_wrong_pose | 40.8479 | 0.2422 | 40.6067 | 0.2409 | 0.4773 | 7.2903 | 0.0496 |
| shuffled_pose | 40.4864 | 0.2407 | 40.4168 | 0.2385 | 0.3409 | 7.2375 | 0.0502 |

The observed-pair M4 upper bound and its held-out R² are in
`two_sided_pose_grounding.json`.  Interpret action-only vs full jointly with A:
pose decodability without real-`z_C` endpoint grounding is not sufficient.

## C. Optional dual-image grounding

N/A by design.  `branches.jsonl` nearest-observed synthetic endpoints remain a
residual diagnostic only and were not used here.

## Gates

```json
{
  "matched_wrong_pose": {
    "fixed_points": 0,
    "same_scene_donors": 0
  },
  "observed_rollout_coverage": {
    "available_real_feature_steps": 2,
    "required_for_1_to_4_step_curve": 4,
    "status": "insufficient_C2_tuple_artifact"
  },
  "observed_tuple_feature_coverage": {
    "complete": true,
    "test_tuples": 44,
    "tuples": 276
  },
  "optional_dual_image_grounding": "N/A_not_used",
  "pose_convention": {
    "branch": "E_12 = T2 @ T1 @ E_A; E_21 = T1 @ T2 @ E_A",
    "extrinsic": "E = inv(camera_to_world), world_to_camera, column-vector convention",
    "raw_pose": "ScanNet++ scene_iphone_metadata.npz trajectories are camera_to_world",
    "relative": "T_AB = E_B @ inv(E_A), maps camera-A coordinates to camera-B coordinates"
  },
  "representative_cross_model_pilot": {
    "authorized": false,
    "blockers": [
      "requested_motion_family_support_below_minimum",
      "observed_C2_artifact_has_only_two_real_transition_steps"
    ],
    "status": "blocked_pending_more_observed_tuple_coverage"
  },
  "scene_split": {
    "counts": {
      "test": {
        "scenes": 11,
        "tuples": 44
      },
      "train": {
        "scenes": 48,
        "tuples": 192
      },
      "val": {
        "scenes": 10,
        "tuples": 40
      }
    },
    "scene_overlap": {
      "train_test": false,
      "train_val": false,
      "val_test": false
    }
  },
  "shuffled_pose": {
    "step1_fixed_points": 0,
    "step2_fixed_points": 0
  },
  "static_scene_risk": "Observed ScanNet++ RGB has no per-frame dynamic/static certificate.",
  "stratification_support": {
    "counts": {
      "different_axis_rotation_rotation": 1,
      "rotation_translation": 4,
      "translation_translation_commuting_control": 1
    },
    "gate": false,
    "minimum_per_requested_family": 5,
    "reason_if_failed": "Natural C2 held-out tuples do not supply enough requested rotation-translation, different-axis rotation-rotation, and translation-translation controls."
  },
  "training_one_step_only": true,
  "tuple_alignment": {
    "action_alignment_max_abs": 2.4048485924854646e-07,
    "all_c1_rows_valid": true,
    "all_c2_horizons_valid": true,
    "normalized_pose_composition_max_abs": 4.440892098500626e-16,
    "normalized_vs_raw_rotation_max_abs": 1.5316153878508487e-07,
    "raw_pose_composition_max_abs": 3.3306690738754696e-15,
    "rotation_determinant_max_abs": 2.384185614490164e-07,
    "rotation_orthogonality_max_abs": 2.603420417024438e-07,
    "target_alignment_max_abs": 1.430511474609375e-06,
    "tuple_count": 276,
    "tuple_scene_count": 69
  }
}
```
