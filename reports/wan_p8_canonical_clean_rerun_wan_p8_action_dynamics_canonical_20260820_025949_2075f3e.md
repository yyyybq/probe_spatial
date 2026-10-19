# Wan p8 Action Dynamics Canonical Clean Rerun Report

## Summary

This run stopped at the feature inventory gate.  I did not launch clean probe
training, did not run eval, and did not update `experiments/registry.csv` or
`results/canonical/*.csv`.

The blocking issue is target-index provenance: the canonical manifest is the
`h124` manifest, whose motion windows were built with enough future context for
horizons 1/2/4.  The target extraction command for this run used only
`target_horizon=1`, which caused the extractor to recompute shorter motion
windows for some scenes.  The resulting target files exist and are readable, but
948 canonical samples do not find their required raw target frame in
`target_indices.npy`.

## Frozen Inputs

- Run tag: `wan_p8_action_dynamics_canonical_20260820_025949_2075f3e`
- Git commit: `2075f3eb270cb0e8907c3927b6b079c64965b75a`
- Canonical manifest:
  `manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl`
- Canonical manifest SHA256:
  `772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c`
- Manifest counts: `train=2448`, `val=276`
- Extractor config:
  `reports/wan_p8_clean_rerun/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/extractor_config.json`
- Extractor config SHA256:
  `70796d8f4fce1cf0d8dc208d7bde3f6b3d8d7a8851976b77224f44708b1b296d`
- Wan model snapshot:
  `/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b`
- Environment snapshot:
  `reports/wan_p8_clean_rerun/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/environment.txt`
- Initial git dirty diff:
  `reports/wan_p8_clean_rerun/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/git_dirty.diff`
- Post-freeze tooling diff:
  `reports/wan_p8_clean_rerun/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/post_freeze_tooling_and_repo.diff`

## Feature Roots

- Streaming root:
  `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/streaming`
- Target root:
  `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/target`
- Provenance root:
  `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/provenance`
- Final feature root size: `1.6T`

## Extraction Status

Streaming extraction:

- Node: `10.119.28.240`
- GPUs: `0,1,2,3,6,7`
- Shards: `6/6` returned `rc=0`
- Logs:
  `reports/wan_p8_clean_rerun/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/logs/streaming_shard*.log`
- End evidence:
  all six logs contain `Done. Processed ... Failed: 0` and `END ... rc=0`
- Layer0 streaming files found: `2726`

Target extraction:

- Node: `10.119.28.240`
- GPUs: `0,1,2,3,6,7`
- Shards: `6/6` returned `rc=0`
- Logs:
  `reports/wan_p8_clean_rerun/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/logs/target_shard*.log`
- End evidence:
  all six logs contain `Done. Processed ... Failed: 0` and `END ... rc=0`
- Layer0 target files found: `682`

## Feature Inventory Gate

Inventory file:

- `reports/wan_p8_clean_rerun/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/feature_inventory.json`
- SHA256:
  `4c9befcd30f405f253e37d5479b7da56f05b348abe135d56781ec7a3d01cf4b5`

Layer coverage for canonical rows:

```text
layer 0:  streaming train=2448/2448 val=276/276; target train=2448/2448 val=276/276
layer 5:  streaming train=2448/2448 val=276/276; target train=2448/2448 val=276/276
layer 10: streaming train=2448/2448 val=276/276; target train=2448/2448 val=276/276
layer 15: streaming train=2448/2448 val=276/276; target train=2448/2448 val=276/276
layer 25: streaming train=2448/2448 val=276/276; target train=2448/2448 val=276/276
layer 29: streaming train=2448/2448 val=276/276; target train=2448/2448 val=276/276
```

The gate still failed because target raw-frame coverage is not 100%:

- Bad canonical samples: `948`
- Missing by split: `train=849`, `val=99`
- Details:
  `reports/wan_p8_clean_rerun/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/target_raw_mismatch_summary.json`

Examples:

```text
00777c41d4_w003_p8: manifest target_raw=906, extracted target_indices=[48,332,638,918]
00a231a370_w003_p8: manifest target_raw=565, extracted target_indices=[39,228,414,576]
00dd871005_w002_p8: manifest target_raw=475, extracted target_indices=[29,221,503,743]
00dd871005_w003_p8: manifest target_raw=723, extracted target_indices=[29,221,503,743]
```

## Registry/Canonical State

No canonical update was run.

Timestamps remained:

```text
2026-08-19 05:02:36 +0000 experiments/registry.csv
2026-08-19 05:02:48 +0000 results/canonical/c1_results.csv
2026-08-19 05:03:00 +0000 results/canonical/all_results.csv
```

## Required Fields

```text
RUN_TAG: wan_p8_action_dynamics_canonical_20260820_025949_2075f3e
GIT_COMMIT: 2075f3eb270cb0e8907c3927b6b079c64965b75a
CANONICAL_MANIFEST_SHA256: 772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c
EXTRACTOR_CONFIG_HASH: 70796d8f4fce1cf0d8dc208d7bde3f6b3d8d7a8851976b77224f44708b1b296d
STREAMING_FEATURE_ROOT: /mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/streaming
TARGET_FEATURE_ROOT: /mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/target
FEATURE_COVERAGE: FAIL; file/header coverage is 100% for six layers, but target raw-frame coverage fails for 948 canonical samples
TRAIN_STATUS_BY_LAYER: NOT_STARTED for layers 0,5,10,15,25,29
CHECKPOINT_BY_LAYER: NONE
EVAL_STATUS_BY_LAYER: NOT_RUN for layers 0,5,10,15,25,29
REGISTRY_UPDATED: false
FINAL_EVAL_COUNT: 422/428
CANONICAL_RESULTS_VALID: false
REMAINING_RISKS: target extraction must be manifest-exact for the h124 canonical windows; recomputing windows with target_horizon=1 is not provenance-equivalent
```

## Next Safe Action

Do not train from the current target root.  The next attempt should create a new
versioned RUN_TAG and extract target features from the canonical manifest's
exact raw target ids:

- for each JSONL row, use `streaming_window_indices[streaming_tail + 1]`;
- group those ids per `scene_id`;
- write one flat scene target cache with all required raw ids in
  `target_indices.npy`;
- rerun inventory before training.

Equivalently, if using the generic extractor, the window generator must be
provenance-equivalent to the canonical `h124` manifest; the current
`target_horizon=1` window reconstruction is not equivalent.

## Final Run Status

The run was explicitly frozen as `FAILED_FEATURE_ALIGNMENT` at
`2026-08-21T18:45:51Z`. All feature files are retained, but its target root is
ineligible for training or canonical evaluation. The machine-readable marker is
`reports/wan_p8_clean_rerun/wan_p8_action_dynamics_canonical_20260820_025949_2075f3e/status.json`.

The subsequent raw-index audit also found that the streaming root is not
canonical-aligned: `1029/2724` rows used a different eight-frame prefix from the
frozen manifest. The root is therefore ineligible in its entirety, not only on
the target side. See
`reports/wan_p8_recovery/wan_p8_manifest_exact_recovery_20260821.md`.
