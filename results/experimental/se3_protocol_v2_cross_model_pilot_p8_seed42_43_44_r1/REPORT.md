# Preliminary cross-model observed-order pilot

Status: **exploratory / non-canonical**.  The only endpoint ground truth is
the real observed C2 `z_C`; no synthetic endpoint, renderer, order loss,
composition loss, cycle loss, rollout loss, or test supervision was used.

## Main comparison

| model | layer | prefix | one-step | correct 2-step | reversed 2-step | order margin | order accuracy | grounding gain vs copy |
|---|---|---|---|---|---|---|---|---|
| DINO | 0 | p8 | 0.1369 | 0.2002 | 0.2040 | 0.0037 | 0.5455 | 0.2219 |
| F3R/Fast3R | 24 | p8 | 0.0081 | 0.0084 | 0.0087 | 0.0003 | 0.6288 | 0.1570 |
| V-JEPA2 | 23 | p8 | 0.1414 | 0.1704 | 0.1732 | 0.0028 | 0.4697 | 0.1148 |
| CogVideoX | 20 | p8 | 0.0019 | 0.0028 | 0.0029 | 0.0001 | 0.5152 | 0.0744 |

All entries are the full predictor's mean over the fixed seeds.  Lower cosine
distance is better; `margin = reversed - correct`; order accuracy is strict
correct-better-than-reversed ranking.  The exact per-seed/all-baseline output
is `all_baseline_seed_metrics.csv`.

## Scene bootstrap (full predictor)

| model | grounding gain 95% CI | order margin 95% CI | order accuracy 95% CI |
|---|---:|---:|---:|
| DINO | [0.1418, 0.2926] | [0.0004, 0.0074] | [0.4316, 0.6591] |
| F3R/Fast3R | [0.0945, 0.2142] | [0.0000, 0.0008] | [0.4998, 0.7424] |
| V-JEPA2 | [0.0628, 0.1649] | [-0.0028, 0.0101] | [0.3561, 0.5833] |
| CogVideoX | [-0.2454, 0.2897] | [-0.0002, 0.0005] | [0.3864, 0.6439] |

Scene is the bootstrap unit; each resample averages fixed-seed runs.

## Coverage

| model | layer | prefix/context | raw tuples | common tuples | test tuples/scenes | seeds | lineage |
|---|---:|---|---:|---:|---:|---:|---|
| DINO | 0 | p8 / single-frame isolated endpoint; p8 fixes observed tuple sampling | 276 | 276 | 44/11 | 3 | audited C1/C2 lineage |
| F3R/Fast3R | 24 | p8 / single-frame geometric endpoint; p8 fixes observed tuple sampling | 276 | 276 | 44/11 | 3 | audited C1/C2 lineage |
| V-JEPA2 | 23 | p8 / temporal-model p8 endpoint role | 276 | 276 | 44/11 | 3 | audited C1/C2 lineage |
| CogVideoX | 20 | p8 / video-generation-model p8 endpoint role | 276 | 276 | 44/11 | 3 | audited C1/C2 lineage |

Unavailable/excluded candidates are in `availability.json`.  The identical
common tuple list is `common_tuple_manifest.csv`.

## Hard-gate audit

```json
{
  "all_models_identical_test_tuple_count": 44,
  "all_models_identical_test_tuple_order": true,
  "common_intersection": {
    "anchor": "dino",
    "keys_identical_for_all_models": true,
    "scene_count": 69,
    "tuple_count": 276
  },
  "dino_r6_regression": {
    "max_abs_delta": 0.0,
    "passed": true,
    "reference": "/mnt/umm/users/yinbaiqiao/probe_spatial/results/experimental/se3_protocol_v2_observed_dino_l0_seed42_r6"
  },
  "limitations_not_blocking_exploratory_pilot": [
    "small_test_tuple_count",
    "rare_motion_family_coverage",
    "no_3_to_4_step_C2_rollout",
    "no_DINO_middle_layer",
    "SeeSE3_replication_incomplete",
    "no_renderer"
  ],
  "pose_convention": {
    "branch": "E_12 = T2 @ T1 @ E_A; E_21 = T1 @ T2 @ E_A",
    "extrinsic": "E = inv(camera_to_world), world_to_camera, column-vector convention",
    "raw_pose": "ScanNet++ scene_iphone_metadata.npz trajectories are camera_to_world",
    "relative": "T_AB = E_B @ inv(E_A), maps camera-A coordinates to camera-B coordinates"
  },
  "scene_leakage": {
    "train_test": false,
    "train_val": false,
    "val_test": false
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
  "static_scene_risk": "Observed ScanNet++ RGB has no per-frame dynamic/static certificate.",
  "training_contract": {
    "batch_size": 64,
    "cosine_loss_weight": 0.25,
    "hidden_dim": 64,
    "learning_rate": 0.001,
    "max_epochs": 180,
    "one_step_target_objective_only": true,
    "patience": 30,
    "training_seeds": [
      42,
      43,
      44
    ],
    "weight_decay": 0.0001
  }
}
```

## Scope limitations

- The common held-out set has only 44 tuples from 11 scenes; this is a group-meeting pilot, not a final statistical benchmark.
- Rare motion-family coverage and 3–4 step rollout are recorded limitations, not blockers for this frozen-intersection comparison.
- `p8` is fixed for all C artifacts; it is not a prefix sweep.  For DINO/F3R it only fixes tuple sampling, not a claimed temporal-context effect.
- No per-frame static-scene certificate is available for raw observed RGB.
