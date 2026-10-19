# Wan P8 Action Dynamics Manifest Lineage Audit

Date: 2026-08-19 UTC

Scope: missing eval for `wan / prefix=8 / action_dynamics`, layers `{0,5,10,15,25,29}`.

No feature extraction, eval rerun, or registry/canonical update was executed during this audit.

## Inputs

- Canonical eval manifest: `manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl`
  - SHA256: `772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c`
  - Size: 54,265,502 bytes
- Historical training manifest path from training logs:
  - `logs/probing_jobs_20260724_1808/manifests_filtered/inscene_c_h124_prefix8_no48da_w003.jsonl`
  - Current local status: missing (`ls` returns "No such file or directory")
  - `10.119.28.240` status: missing (`ls` returns "No such file or directory")
  - Git status: path is not tracked in the current repository history (`git log --all -- <path>` returned no commits)

## 1. Manifest Comparison

### Canonical manifest

The canonical manifest was parsed directly.

| split | samples | unique scenes | unique sample_ids |
| --- | ---: | ---: | ---: |
| train | 2448 | 612 | 2448 |
| val | 276 | 69 | 276 |

Canonical train/val overlap:

- sample_id overlap: `0`
- scene_id overlap: `0`

First canonical rows:

- line 1: train sample `00777c41d4_w000_p8`, scene `00777c41d4`, `streaming_window_id=0`, `streaming_tail=7`
- line 5: val sample `00a231a370_w000_p8`, scene `00a231a370`, `streaming_window_id=0`, `streaming_tail=7`

### Historical filtered manifest

The historical filtered manifest file itself is unavailable, so its exact hash, samples, scenes, and subset relation cannot be verified.

The six training logs do show that training used this historical path:

- `.../action_dynamics_wan_p8_layer0/inscene15k_streaming.log:18` loaded 276 records from `.../inscene_c_h124_prefix8_no48da_w003.jsonl` for `split=val`.
- `.../action_dynamics_wan_p8_layer0/inscene15k_streaming.log:21` loaded 2447 records from the same manifest for `split=train`.
- The same train/val counts and manifest path appear in the layer `{5,10,15,25,29}` training logs at lines 18 and 21.

Observed count difference against canonical:

- canonical train: 2448
- historical filtered train from logs: 2447
- canonical val: 276
- historical filtered val from logs: 276

Because the filtered manifest content is unavailable, the following cannot be proven:

- whether filtered is a strict subset of canonical;
- whether canonical val samples appear in historical train;
- whether any train/val split rewrite happened inside the filtered file.

The canonical manifest itself has no train/val sample or scene leakage.

## 2. Filtered Manifest Lineage

### Code paths that prefer filtered manifests

`scripts/build_probe_fill_jobs.py:206-214`:

- builds `logs/probing_jobs_20260724_1808/manifests_filtered/inscene_c_h124_prefix{prefix}_no48da_w003.jsonl`;
- uses it if present;
- falls back to `manifests/c_retrain_20260722/inscene_c_h124_prefix{prefix}.jsonl`.

`scripts/prepare_vjepa2_videomae_abc_probe.sh:53-78`, `scripts/prepare_lingbot_abc_probe.sh:46-69`, and `scripts/prepare_minimax_h3_abc_probe.sh:46-69`:

- set `MANIFEST_DIR` to `logs/probing_jobs_20260724_1808/manifests_filtered`;
- choose `inscene_c_h124_prefix${prefix}_no48da_w003.jsonl` when present;
- otherwise fall back to the canonical `c_retrain_20260722` manifest.

These scripts are currently untracked files in this working tree, so there is no git-blameable commit provenance for these exact lines.

### Evidence for `no48da_w003`

`scripts/gapfill_48da_window0003.sh:1-25` documents a later repair path:

- target scene: `48da625e8c`;
- target window: `window_0003`;
- intent: gapfill `FEAT_STREAMING window_0003` for scene `48da625e8c`;
- prefix lengths: `8,12,16`;
- VFMs include `wan`.

Earlier failed run evidence:

- `logs/inscene15k_streaming/runs/wan_cog_layer_probe_20260724_1808/wan_cog_layer_probe_20260724_1808_action_dynamics_wan_p12_layer0/progress.jsonl` ends with `FileNotFoundError` for:
  - `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING/wan/scannetpp/48da625e8c/window_0003/prefix_000011/feature_t749_layer0.sft`
  - event time: `2026-07-24T22:41:15+0000`
  - global_step: `103`

Interpretation, constrained by available evidence:

- `no48da_w003` very likely means "exclude scene `48da625e8c`, window `0003`", matching the documented gapfill target and earlier FileNotFoundError.
- This is not proven from the filtered manifest body because the file is unavailable.

### Why current eval used canonical

`scripts/audit_all_missing_eval.py:175-184` resolves manifests by:

1. using a raw manifest path if available and it exists;
2. otherwise falling back to `manifests/c_retrain_20260722/inscene_c_h124_prefix{prefix}.jsonl`;
3. then falling back to `inscene_c_prefix{prefix}.jsonl`.

`scripts/run_missing_eval_shard.py:36-49` repeats the same resolution logic for a job row; `scripts/run_missing_eval_shard.py:89-91` exports the resolved manifest as `INSCENE_MANIFEST`.

The generated eval manifests explicitly used canonical:

- `logs/probing_jobs_full_missing_eval_20260817/jobs/eval_jobs.tsv:256-261`
- `logs/probing_jobs_remaining_diag_4node_20260817/jobs/eval_jobs.tsv:75-80`

All six rows point `scene_manifest` to:

- `/mnt/umm/users/yinbaiqiao/probe_spatial/manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl`

Therefore current eval used canonical because the historical filtered manifest was not available to the audit/runner path.

## 3. Feature Path Audit

### Alias and realpath

Streaming alias:

- path: `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_PREFIX_ALIAS/wan_prefix_8_m0p35_r0p5`
- type: symlink
- symlink target: `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING`
- symlink mtime: `2026-07-24 19:47:21 +0000`

Target alias:

- path: `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET_PREFIX_ALIAS/wan_prefix_8_m0p35_r0p5`
- type: symlink
- symlink target: `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET`

Current real feature root status:

- `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING` exists.
- `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING/wan/scannetpp` exists.
- Only one scene directory is visible under Wan Scannet++ streaming features: `48da625e8c`.
- On `10.119.28.240`, the same root also only showed `48da625e8c`.

### Canonical val feature coverage

Expected path shape for canonical val samples:

`/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING/wan/scannetpp/<scene>/window_<streaming_window_id>/prefix_<streaming_tail>/feature_t749_layer<L>.sft`

For the canonical val split:

- samples checked: 276
- scenes checked: 69
- layers checked: `{0,5,10,15,25,29}`

Coverage:

| layer | streaming exists | streaming missing | target exists | target missing |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 276 | 0 | 276 |
| 5 | 0 | 276 | 0 | 276 |
| 10 | 0 | 276 | 0 | 276 |
| 15 | 0 | 276 | 0 | 276 |
| 25 | 0 | 276 | 0 | 276 |
| 29 | 0 | 276 | 0 | 276 |

First missing canonical val sample for every layer:

- manifest line 5
- sample_id: `00a231a370_w000_p8`
- scene: `00a231a370`
- expected paths include:
  - `.../FEAT_STREAMING/wan/scannetpp/00a231a370/window_0000/prefix_000007/feature_t749_layer0.sft`
  - analogous files for layers `{5,10,15,25,29}`

Bounded searches under visible feature roots found no `wan/scannetpp/00a231a370` under:

- `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT`
- `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_CONTEXT`
- `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_SHUFFLED`
- `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING`
- `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET`

Current evidence favors "not generated for canonical Wan p8 val / not present in visible stores." It does not support alias drift: the alias points to the expected raw `FEAT_STREAMING` tree, and the eval paths match the generated job rows.

## 4. Six Run Audit

All six runs used the historical filtered manifest path in training logs, not the current canonical manifest.

| layer | train manifest evidence | last completed validation | last completed train epoch | interruption evidence | checkpoint | checkpoint SHA256 | status |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | log lines 18, 21: filtered manifest, val=276/train=2447 | line 224: epoch 5, step 1830 | line 225: epoch 5, step 1830 | line 232; lines 285/323 killed | `.../layer0/VideoProbe3D/uun8pprj/checkpoints/epoch=5-step=1830.ckpt` | `bf9fc56dac74dff0bc0bcb714b9695016b4229e8465b01251910ea7016a80fe6` | `INTERRUPTED_WITH_VALID_CHECKPOINT` |
| 5 | log lines 18, 21: filtered manifest, val=276/train=2447 | line 224: epoch 5, step 1830 | line 225: epoch 5, step 1830 | lines 280/304 killed | `.../layer5/VideoProbe3D/3nuaqwei/checkpoints/epoch=5-step=1830.ckpt` | `3d65b3230d3f944c13038ffbd8e17c7dcf2c6f4ca273766c981b02c8c38ebfd6` | `INTERRUPTED_WITH_VALID_CHECKPOINT` |
| 10 | log lines 18, 21: filtered manifest, val=276/train=2447 | line 224: epoch 5, step 1830 | line 225: epoch 5, step 1830 | line 231; lines 284/322 killed | `.../layer10/VideoProbe3D/wy1dvt0a/checkpoints/epoch=5-step=1830.ckpt` | `b689d7bda2813eb5d7aa33b5822d4843483919d265828947a3d008fb0d9b7451` | `INTERRUPTED_WITH_VALID_CHECKPOINT` |
| 15 | log lines 18, 21: filtered manifest, val=276/train=2447 | line 190: epoch 4, step 1525 | line 191: epoch 4, step 1525 | line 209; lines 262/288 killed | `.../layer15/VideoProbe3D/odzwym4g/checkpoints/epoch=4-step=1525.ckpt` | `6dc5334d0e4774b21ed656a4b13bfa05025c117f729408f21a77205105d72619` | `INTERRUPTED_WITH_VALID_CHECKPOINT` |
| 25 | log lines 18, 21: filtered manifest, val=276/train=2447 | line 224: epoch 5, step 1830 | line 225: epoch 5, step 1830 | line 230; lines 283/317 killed | `.../layer25/VideoProbe3D/v3jc32qp/checkpoints/epoch=5-step=1830.ckpt` | `607236c78cd547bde49356475059718c87c20155dd922f201728318be7a27db7` | `INTERRUPTED_WITH_VALID_CHECKPOINT` |
| 29 | log lines 18, 21: filtered manifest, val=276/train=2447 | line 224: epoch 5, step 1830 | line 225: epoch 5, step 1830 | line 227; lines 240/321 killed | `.../layer29/VideoProbe3D/2jvasa0z/checkpoints/epoch=5-step=1830.ckpt` | `e1b04eea218566f01236ffdf713095029c0c401e9d23cfbd08c6e222a048e605` | `INTERRUPTED_WITH_VALID_CHECKPOINT` |

Checkpoint file sizes were all 101,438,576 bytes.

The status is not `CLEAN_COMPLETE` because no inspected training log contains a clean fit completion after the listed checkpoints; each inspected log ends with a DataLoader worker termination. The status is not `INVALID` because each run has a non-empty checkpoint after a completed validation epoch.

## 5. Current Eval Failures

The six attempted evals all failed on the first canonical val sample because the expected Wan p8 feature was absent.

Examples:

- `logs/inscene15k_streaming/eval_jobs/remaining_diag_4node_20260817_slot2_action_dynamics_wan_p8_layer0.log:152-153` loaded canonical val with 276 samples.
- `...layer0.log:205` failed on `00a231a370/window_0000/prefix_000007/feature_t749_layer0.sft`.
- `...layer0.log:209` ended with `rc=1`.

All layers failed analogously:

- layer 0: eval log line 205, `rc=1` at line 209
- layer 5: eval log line 205, `rc=1` at line 209
- layer 10: eval log line 205, `rc=1` at line 209
- layer 15: eval log line 205, `rc=1` at line 209
- layer 25: eval log line 205, `rc=1` at line 209
- layer 29: eval log line 205, `rc=1` at line 209

## Decision Block

```text
MANIFEST_RELATION:
  Canonical manifest is verified: train=2448, val=276, train/val sample overlap=0, train/val scene overlap=0, sha256=772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c.
  Historical filtered manifest is unavailable locally, unavailable on 10.119.28.240, and not tracked in git, so subset relation and cross-manifest train/val leakage cannot be proven.

FILTER_REASON:
  Likely exclusion of scene 48da625e8c window_0003, based on the name no48da_w003, gapfill script comments at scripts/gapfill_48da_window0003.sh:1-25, and 20260724 FileNotFoundError for 48da625e8c/window_0003/prefix_000011. This remains not fully proven because the filtered manifest body is missing.

SPLIT_LEAKAGE:
  Canonical manifest alone shows no leakage: sample overlap=0, scene overlap=0.
  Historical-vs-canonical leakage is UNKNOWN_NOT_AUDITABLE because the historical filtered manifest file is missing.

FEATURE_STATUS:
  BLOCKED_MISSING_FEATURES. The Wan p8 alias resolves to FEAT_STREAMING; no alias drift was found. Canonical val coverage is 0/276 for each layer {0,5,10,15,25,29} under current visible Wan Scannet++ streaming features. No matching 00a231a370 Wan feature directory was found in bounded visible FEAT roots.

CHECKPOINT_STATUS:
  INTERRUPTED_WITH_VALID_CHECKPOINT for all six runs. Layers {0,5,10,25,29} have epoch=5-step=1830 checkpoints after completed validation; layer 15 has epoch=4-step=1525 after completed validation. All inspected logs later show DataLoader worker termination, so none are CLEAN_COMPLETE.

DECISION:
  BLOCKED_PROVENANCE. Do not write these six evals into canonical results now. The historical manifest is missing, historical/canonical split relation cannot be verified, and the exact Wan canonical-val feature provenance cannot be confirmed from current files.

ACTIONS_EXECUTED:
  Read-only audit only. No feature extraction, no eval rerun, no registry update, no canonical result update.

FINAL_EVAL_COUNT:
  422/428 target eval jobs have valid summaries. The six missing jobs remain wan/action_dynamics/prefix=8/layers={0,5,10,15,25,29}.

CANONICAL_RESULTS_VALID:
  false for these six missing Wan p8 action_dynamics rows. Existing canonical files remain as previously updated for successful evals only.
```
