# SE(3) latent-structure MVP report

Status: **experimental / non-canonical**. This run does not modify the registry,
canonical tables, checkpoints, or feature artifacts.

## Data and coordinate audit

- Selected artifact: `dino`, layer `0`, prefix `8`.
- Source: ScanNet++ only; 69 canonical-validation scenes and 276 complete tuples.
- Raw `trajectories` are camera-to-world matrices. The repository converts them
  to world-to-camera extrinsics `E`; `T_AB = E_B E_A^-1` maps camera-A
  coordinates into camera-B coordinates. Therefore `T_AC = T_BC T_AB`.
- Maximum normalized composition residual: `4.441e-16`.
- Maximum raw-pose composition residual: `3.331e-15`.
- C1/C2 target alignment maximum absolute difference: `1.431e-06`.
- C1 action re-encoding maximum absolute difference: `2.405e-07`.
- Pose-support conclusion: **empirical_SE3_6DoF**. All three world-axis rotation and translation components have substantial coverage; this is not an SE(2)-only trajectory set. Coverage is empirical and non-uniform, not a claim of uniform sampling over SE(3).
- Static-scene caveat: ScanNet++ capture scenes are treated as static indoor
  reconstructions, but these artifacts have no per-frame dynamic-object/static
  certification. This remains a documented risk rather than a verified label.

## Experimental split and training contract

The 69 canonical-validation scenes were re-split by scene solely for this MVP:
`{'train': {'scenes': 48, 'tuples': 192}, 'val': {'scenes': 10, 'tuples': 40}, 'test': {'scenes': 11, 'tuples': 44}}`. Scene overlap is `{'train_val': False, 'train_test': False, 'val_test': False}`.
Each tuple supplies three isolated target features `(A,B,C)`. Training uses only
the one-step pairs `A->B` and `B->C` with MSE plus cosine target loss. No
composition, cycle, order, or rollout loss is used.

## Results

All distance columns are lower-is-better. Order-swap separation is deliberately
higher-is-more-sensitive and must be read together with its endpoint error.

| baseline | one_step_cosine | composition_structure_cosine | composition_recursive_endpoint_cosine | composition_direct_endpoint_cosine | cycle_cosine | order_swap_separation_cosine | rollout_step4_cosine |
|---|---|---|---|---|---|---|---|
| copy | 0.170665 | 0.000000 | 0.257341 | 0.257341 | 0.000000 | 0.000000 | 0.000000 |
| without_condition | 0.166367 | 0.003299 | 0.231837 | 0.230999 | 0.036771 | 0.000000 | 0.056058 |
| action_only | 0.232693 | 0.027863 | 0.232836 | 0.237175 | 0.232169 | 0.078457 | 0.233005 |
| wrong_pose | 0.185681 | 0.009394 | 0.282567 | 0.280857 | 0.034390 | 0.006884 | 0.072193 |
| full | 0.135408 | 0.009752 | 0.197771 | 0.193883 | 0.032178 | 0.007828 | 0.064612 |

The copy predictor illustrates the degeneracy gate: it can have essentially
zero composition/cycle error while its real endpoint grounding remains worse.
See `minimal_results_table.csv` for cosine and normalized-RMSE variants and all
rollout steps.

## Rollout definition

The four-step curve uses real isolated-feature endpoints along
`A -> B -> C -> B -> A`: two observed forward transforms followed by their
exact inverses. This supplies real endpoint error at every step without
inventing an unavailable fourth future feature.

## Sanity checks

```json
{
  "all_tuple_targets_valid": true,
  "copy_composition_is_near_zero": true,
  "copy_cycle_is_near_zero": true,
  "copy_grounding_is_nontrivial": true,
  "feature_manifest_alignment_passed": true,
  "grounding_reported_with_structure": true,
  "normal_predictor_finite": true,
  "pose_convention_confirmed": true,
  "scene_disjoint_split_passed": true,
  "se3_log_exp_roundtrip_max_abs": 3.857641796001232e-07,
  "static_scene_certification": false,
  "static_scene_risk": "ScanNet++ is treated as a static reconstruction source, but no per-frame static/dynamic certification is present."
}
```

## Reproduction

```bash
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
MPLCONFIGDIR=/tmp/probe_spatial_mpl "$PY" -m pytest -q tests/test_se3_latent_mvp.py
MPLCONFIGDIR=/tmp/probe_spatial_mpl "$PY" -m analyses.se3_latent_mvp.run_mvp \
  --config analyses/se3_latent_mvp/config.json \
  --output-dir results/experimental/se3_latent_mvp_dino_l0_p8_seed42_v2
```

## Minimal sweep extension

No sweep is launched here. To extend, provide another frozen C1/C2 dump pair
and matching manifest through the same config fields, then change only
`model`, `feature_layer`, `prefix`, and the input paths. The alignment gates
run before training, so an incomplete layer/timestep/prefix cell stops rather
than silently shrinking the sample set.
