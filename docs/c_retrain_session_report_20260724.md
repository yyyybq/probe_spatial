# C-series Retraining Session Report

Date: 2026-07-24

This document summarizes the work performed in the current Cursor working session for the `probe_spatial` project. It covers the C-series data bug context, AOSS/manifest work, feature audits, benchmarking, formal retraining rollout, rescue scheduling after node loss, and the final completion state.

## 1. Project Goal And Current Background

The project probes spatial and temporal reasoning in frozen video foundation model (VFM) features on InsScene-15K. The probing suite trains lightweight diagnostic heads over precomputed VFM features for tasks such as depth, camera/mask, view consistency, ego belief, and C-series target/action reasoning.

The current phase focused on the C-series experiments:

- `action_dynamics` / C1
- `path_integration` / C2
- `counterfactual` / C3

At the start of this working session, the project background was:

- A bug in `InsScene15KDataset` had previously caused C-series target features to be missed, so old C checkpoints were trained against zero/fallback targets and were not scientifically valid.
- The dataset loader had already been fixed before this session context was compacted.
- AOSS data had been uploaded, and feature paths had been updated to point to the AOSS bucket.
- The main goal of this session was to verify that C-series data/features were complete, prevent the old data-loading failure modes, test AOSS throughput, and run formal C-series retraining across available nodes.

The session mainly revolved around:

- auditing the `682/684` scene discrepancy;
- adding manifest audit metadata and exclusion files;
- benchmarking AOSS streaming throughput under `num_workers=4` and single-card dual-process settings;
- building final manifests for all C prefixes;
- launching C-series retraining safely across available H800 nodes;
- recovering from an over-aggressive initial launch and later from node reclamation;
- confirming all 72 C probing jobs completed successfully.

## 2. Completed Code Changes

### 2.1 `scripts/build_inscene_streaming_manifest.py`

Status: new/untracked script in the current workspace.

Main areas changed:

- manifest audit output;
- feature-root checking;
- metadata/checksum output;
- sample-level manifest generation for C-series retraining.

Problem before the changes:

- The initial manifest builder only wrote sample JSONL records.
- It did not output `included_scenes.txt` or `excluded_scenes.jsonl`.
- It did not explain why scenes were excluded.
- It did not include manifest versioning, SHA256, split counts, duplicate sample checks, or feature-root metadata.
- Early feature checks incorrectly assumed streaming features lived under `prefix_000007/feature.sft`; actual layout is `window_0000/prefix_000007/feature.sft`.
- The first DINO prefix-8 manifest audit produced misleading exclusions until the streaming feature path logic was corrected.

Concrete changes:

- Added imports:
  - `datetime`
  - `hashlib`
  - `subprocess`
  - `feature_filename`
- Changed `collect_scannetpp_scene()` to return `(scene, reason)` instead of only `scene`.
- Added explicit exclusion reasons:
  - `missing_scene`
  - `missing_images_dir`
  - `missing_masks_dir`
  - `missing_metadata`
  - `not_enough_valid_frames`
  - `no_valid_prefix_sample`
  - `missing_streaming_prefix_index`
  - `missing_streaming_feature`
  - `missing_target_feature`
  - `missing_target_indices`
- Added feature-root audit helpers:
  - `streaming_feature_path()`
  - `target_feature_paths()`
  - `feature_audit_reason()`
  - `git_commit()`
  - `sha256_file()`
- Added CLI arguments:
  - `--vfm-name`
  - `--feature-layer`
  - `--feature-timestep`
  - `--skip-feature-checks`
  - `--audit-dir`
- Added audit side effects:
  - `included_scenes.txt`
  - `excluded_scenes.jsonl`
  - `<manifest>.meta.json`
- Added duplicate `sample_id` accounting.
- Added split counts in metadata.
- Added SHA256 checksum for the final manifest.

Key implementation detail:

The corrected streaming feature path is:

```python
root / vfm_name / "scannetpp" / scene_id / f"window_{window_id:04d}" / f"prefix_{tail:06d}" / fname
```

This matches the dataset loader path:

- `InsScene15KDataset._streaming_prefix_scene_dir()`
- `InsScene15KDataset._streaming_prefix_dir_name()`
- `InsScene15KDataset._load_streaming_prefix_feat()`

Why this implementation:

- Manifest generation is now the single offline scan/audit point.
- Training no longer needs to list AOSS directories at startup.
- Exclusion reasons are persisted for reproducibility.
- The manifest becomes a versioned training artifact instead of an opaque JSONL file.

Affected flows:

- Offline C-series manifest generation.
- C-series training startup.
- Data-integrity debugging.
- Future feature completeness audits.

Validation:

- `python -m py_compile scripts/build_inscene_streaming_manifest.py` passed.
- `ReadLints` reported no linter errors.
- Final `h1,2,4` manifests were generated successfully for prefixes `8/12/16/24`.

Final manifests:

- `manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl`
- `manifests/c_retrain_20260722/inscene_c_h124_prefix12.jsonl`
- `manifests/c_retrain_20260722/inscene_c_h124_prefix16.jsonl`
- `manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl`

### 2.2 `scripts/run_aoss_streaming_benchmark.sh`

Status: new/untracked script in the current workspace.

Problem before the change:

- Earlier benchmark commands were ad hoc.
- A statistics command failure could produce a non-zero shell status even if training succeeded.
- There was no standardized record of:
  - manifest SHA256;
  - cache condition;
  - process count;
  - workers;
  - GPU memory;
  - per-job logs;
  - residual processes.
- The previous `exit_code=127` issue showed that non-critical stats code could incorrectly mark training as failed.

Concrete implementation:

- Created a wrapper for AOSS streaming benchmarks.
- Preserves the actual training exit code.
- Runs statistics collection as non-fatal.
- Records:
  - `conditions.json`
  - `gpu.csv`
  - `process_exit_codes.txt`
  - `residual_processes.txt`
  - `summary.json`
- Supports:
  - `PROCESSES`
  - `GPU_IDS`
  - `NUM_WORKERS`
  - `BATCHES`
  - `VFM`
  - `PREFIX_LEN`
  - `PROBE`
  - `CACHE_NOTE`
- Fixes the environment-root issue by exporting concrete `INSCENE_STREAMING_FEAT_ROOT` for the selected prefix:

```bash
STREAMING_FEAT_ROOT="${STREAM_ROOT_BASE}/${VFM}_prefix_${PREFIX_LEN}_m${motion_tag}_r${rotation_tag}"
export INSCENE_STREAMING_FEAT_ROOT="${STREAMING_FEAT_ROOT}"
```

This matters because `InsScene15KDataset.__init__` gives priority to the environment variable `INSCENE_STREAMING_FEAT_ROOT` over the Hydra argument.

Validation:

- `bash -n scripts/run_aoss_streaming_benchmark.sh` passed.
- `ReadLints` reported no linter errors.
- Used successfully for:
  - single-process `num_workers=4`, `500 batch`;
  - single-card dual-process `2 x num_workers=4`, `500 batch`.

### 2.3 `scripts/launch_c_retrain_40gpu.sh`

Status: new/untracked script in the current workspace.

Problem before the change:

- Formal C-series retraining required dispatching 72 jobs across multiple nodes.
- Existing `run_streaming_probe_sweep.sh` is a local sweep helper, not a multi-node scheduler.
- The rollout needed:
  - node assignment;
  - per-node bounded concurrency;
  - new run folders to avoid old invalid checkpoints;
  - explicit manifest use;
  - controlled `batch_size` and `num_workers`;
  - rescue scheduling after node loss.

Concrete implementation:

- Added a multi-node launcher.
- Default nodes:
  - `10.119.17.135`
  - `10.119.16.151`
  - `10.119.31.225`
  - `10.119.20.37`
  - `10.119.19.88`
- Default experiment matrix:
  - VFMs: `wan`, `cogvideox`, `vjepa2`, `dino`, `aether`, `f3r`
  - probes: `action_dynamics`, `path_integration`, `counterfactual`
  - prefixes: `8`, `12`, `16`, `24`
  - total: `6 x 3 x 4 = 72`
- Uses default layers:
  - `wan`: 20
  - `cogvideox`: 20
  - `vjepa2`: 23
  - `dino`: 0
  - `aether`: 1
  - `f3r`: 24
- Launches one remote supervisor per node.
- Supervisors run bounded local queues.
- Sets:
  - `autoresume=false`
  - `logger.wandb.offline=true`
  - unique `paths.run_folder_name=${RUN_ID}/${tag}`
  - `INSCENE_MANIFEST` per prefix
  - concrete target root per VFM/prefix
  - `STREAM_ROOT_BASE` while unsetting `INSCENE_STREAMING_FEAT_ROOT`
- Added later support for:
  - `NODES_CSV`
  - `JOB_LIST`

The `JOB_LIST` support enabled rescue scheduling for the 10 jobs affected by reclaimed nodes.

Validation:

- `bash -n scripts/launch_c_retrain_40gpu.sh` passed.
- `ReadLints` reported no linter errors.
- Used for:
  - initial safe formal run `c_retrain_safe_20260722_1435`;
  - rescue run `c_retrain_rescue_20260723_1748`.

### 2.4 Existing dataset loader changes used but not newly edited in this final part

File:

- `vidfm3d/data/components/inscene15k_dataset.py`

Status:

- It was already modified before this summarized session state.
- This session used and verified the behavior, but the main code edits in the current post-summary work were the scripts above.

Relevant existing methods:

- `InsScene15KDataset._load_scene_manifest()`
- `InsScene15KDataset._normalize_scene_manifest_record()`
- `InsScene15KDataset._streaming_prefix_dir_name()`
- `InsScene15KDataset._streaming_prefix_scene_dir()`
- `InsScene15KDataset._load_streaming_prefix_feat()`
- target feature fallback logic for flat target layout.

Why it matters:

- The loader can read a local sample-level manifest through `INSCENE_MANIFEST`.
- This avoids AOSS directory listing during every training process startup.
- The flat target feature layout is supported, avoiding the previous C-series zero-target bug.

Validation in this session:

- Training logs repeatedly showed:
  - `Loaded ... records from scene manifest ...`
- Formal C retraining completed 72/72 jobs successfully.

### 2.5 `scripts/run_c_eval_slots.py`

Status: new script added after retraining completed.

Problem before the change:

- The retrained checkpoints were not stored in the top-level `checkpoints/` directory expected by simple glob patterns.
- Actual Lightning checkpoints lived under each run's W&B-style directory:
  - `VideoProbe3D/<wandb_id>/checkpoints/epoch=...ckpt`
- Safe and rescue runs had 9 overlapping jobs, so evaluation needed deduplication by `(probe, vfm, prefix, layer)` and had to prefer rescue checkpoints for interrupted jobs.
- Direct local eval into a remote-created run directory hit permissions issues.
- Hydra required checkpoint paths containing `epoch=...step=...` to escape `=` characters.

Concrete implementation:

- Added a slot-based C-series evaluation supervisor.
- Discovers checkpoints from:
  - `c_retrain_safe_20260722_1435`
  - `c_retrain_rescue_20260723_1748`
- Deduplicates to 72 unique jobs and prefers rescue checkpoints.
- Reads each job's `.hydra/overrides.yaml` to recover exact VFM, layer, prefix, feature roots, and target roots.
- Sets per-prefix manifests:
  - `manifests/c_retrain_20260722/inscene_c_h124_prefix{8,12,16,24}.jsonl`
- Runs `vidfm3d/eval_diag.py` with:
  - fixed `INSCENE_DATA_ROOT`
  - `INSCENE_MANIFEST`
  - concrete `INSCENE_STREAMING_FEAT_ROOT`
  - concrete `INSCENE_TARGET_FEAT_ROOT`
  - escaped `ckpt_path`
  - isolated `paths.run_folder_name`
- Supports distributed slot assignment with:
  - `RUN_TAG`
  - `SLOT_COUNT`
  - `SLOT_IDS`
  - `GPUS`

Validation:

- `python -m py_compile scripts/run_c_eval_slots.py` passed.
- `ReadLints` reported no linter errors.
- Used successfully for final eval run `c_eval_20260724_1903`.

## 3. Discovered Bugs And Problems

### 3.1 Confirmed bug: feature audit initially used the wrong streaming feature path

Type: code logic issue in new audit script.

Symptom:

- First full feature audit excluded hundreds of valid scenes.
- Early output showed many records like:

```json
{"scene_id": "00777c41d4", "reason": "missing_streaming_feature"}
```

Root cause:

- The audit script initially checked:

```text
.../<scene>/prefix_000007/feature.sft
```

- Actual streaming feature layout is:

```text
.../<scene>/window_0000/prefix_000007/feature.sft
```

Investigation:

- Listed AOSS directories directly.
- Confirmed scene directory contains:
  - `window_0000`
  - `prefix_index.npy`
- Listed inside `window_0000` and found:
  - `prefix_000007/feature.sft`

Fix:

- Updated `streaming_feature_path()` to include `window_{window_id:04d}` and `prefix_{tail:06d}`.

Status:

- Fixed and verified.
- Final audit succeeded.

Impact on previous experiments:

- This was only in the newly created audit script, not in training.
- It did not invalidate training results.

### 3.2 Confirmed issue: `682` vs `684` scene discrepancy

Type: data integrity issue.

Observed values:

- Configured data scene root contains `683` scenes.
- Streaming feature scene count:
  - prefix 8: `682`
  - prefix 12/16/24: `681`
- Target feature scene count:
  - all prefixes: `681`

Final excluded scenes:

- `c24f94007b`
  - reason: `missing_images_dir`
  - base scene is incomplete.
- `a003a6585e`
  - prefix 8 earlier had streaming feature but target was missing.
  - in `h1,2,4` manifests it is excluded as `no_valid_prefix_sample` or `not_enough_valid_frames` depending on prefix.

Root cause:

- The configured AOSS scene root itself lists `683`, not `684`.
- One scene is structurally incomplete.
- One scene lacks usable C target/long-horizon samples.

Status:

- Resolved for C training by excluding both scenes.
- Final manifests use 681 valid scenes.

Remaining unknown:

- The identity of the "original 684th scene" cannot be recovered from this session unless an external original 684-scene list is provided.

Impact:

- Using the old 682-scene manifest without target checking could have included a scene with missing target features.
- The final C retraining did not include invalid scenes.

### 3.3 Confirmed issue: environment variable root override can break streaming paths

Type: configuration/environment issue.

Symptom:

- First 500-batch benchmark failed with:

```text
FileNotFoundError: Streaming prefix VFM feature not found at
fj:s3://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX/dino/scannetpp/5654092cc2/window_0003/prefix_000007/feature.sft
```

Expected path should include:

```text
dino_prefix_8_m0p35_r0p5
```

Root cause:

- `InsScene15KDataset.__init__` uses:

```python
streaming_feat_root = os.environ.get("INSCENE_STREAMING_FEAT_ROOT", streaming_feat_root)
```

- The benchmark wrapper had exported `INSCENE_STREAMING_FEAT_ROOT` as the parent:

```text
fj:s3://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX
```

- This overrode the Hydra argument that had the correct concrete root.

Fix:

- In `run_aoss_streaming_benchmark.sh`, set:

```bash
INSCENE_STREAMING_FEAT_ROOT="${STREAM_ROOT_BASE}/${VFM}_prefix_${PREFIX_LEN}_m0p35_r0p5"
```

- In `launch_c_retrain_40gpu.sh`, use `STREAM_ROOT_BASE` for the sweep script but `unset INSCENE_STREAMING_FEAT_ROOT` so Hydra override wins.

Status:

- Fixed.
- Verified by successful benchmarks and formal retraining.

Impact:

- If future launch scripts set `INSCENE_STREAMING_FEAT_ROOT` to the parent directory, training will fail even if Hydra overrides are correct.

### 3.4 Confirmed issue: `action_horizons=1` manifest is not sufficient for C2/C3

Type: data/sample-construction issue.

Symptom:

- Initial prefix manifests were being generated with `--action-horizons 1`.
- C2/C3 configs use:

```yaml
action_horizons: [1, 2, 4]
```

Root cause:

- Manifest windows must contain enough future frames for the largest requested horizon.
- A manifest built only for horizon 1 may not be valid for horizon 4.

Fix:

- Stopped the horizon-1-only `p12/p16/p24` manifest generation.
- Rebuilt formal manifests with:

```bash
--action-horizons 1,2,4
```

Status:

- Fixed.
- Final formal C manifests are `h1,2,4` manifests.

Impact:

- The earlier horizon-1-only manifest generation was intentionally stopped and not used for formal C training.

### 3.5 Confirmed issue: over-aggressive launch caused CUDA OOM and stale contexts

Type: infrastructure/capacity planning issue.

Initial aggressive config:

- `SLOTS_PER_NODE=10`
- `BATCH_SIZE=8`
- `NUM_WORKERS=4`
- some GPUs got two processes.

Symptom:

- Multiple jobs failed immediately with:

```text
RuntimeError: CUDA error: out of memory
```

Example:

- `c_retrain_20260722_action_dynamics_aether_p12_layer1.log`

Root cause:

- `aether` and `cogvideox` have 3072-channel features/probes.
- `batch_size=8` plus extra per-node concurrency was too aggressive.
- Some H800 GPUs filled to around 80 GB.

Follow-on issue:

- After killing training processes, `nvidia-smi` showed `[Not Found]` CUDA contexts occupying GPU memory.
- `nvidia-smi --gpu-reset` was not supported on these nodes.
- Contexts later naturally released.

Fix:

- Stopped the overloaded run.
- Waited until stale contexts released.
- Relaunched with safe config:
  - `SLOTS_PER_NODE=8`
  - `BATCH_SIZE=4`
  - `NUM_WORKERS=4`
  - `1 proc/GPU`

Status:

- Resolved.
- Safe run completed successfully for the tasks it retained.

Impact:

- The failed aggressive launch logs should not be used for results.
- Final valid results come from:
  - `c_retrain_safe_20260722_1435`
  - `c_retrain_rescue_20260723_1748`

### 3.6 Confirmed infrastructure issue: two nodes were reclaimed

Nodes lost:

- `10.119.20.37`
- `10.119.19.88`

Symptom:

```text
ssh: connect to host ... No route to host
```

Impact:

- Some jobs assigned to those nodes were interrupted or never started.

Resolution:

- Parsed all 72 expected tasks.
- Found:
  - 62 done successfully in the safe run.
  - 10 needed rescue.
- Rescheduled those 10 tasks on the remaining 3 nodes:
  - `10.119.17.135`
  - `10.119.16.151`
  - `10.119.31.225`

Final status:

- Rescue run completed 10/10 successfully.

### 3.7 Confirmed behavior: GPU utilization remains low

Type: expected performance bottleneck / I/O behavior.

Observed:

- GPU utilization was often low even while jobs were progressing.
- Earlier benchmark showed single-process DINO C1 around `0.82 it/s`.
- Dual-process DINO C1 on one GPU gave around `2.13 it/s` combined in a controlled benchmark.

Root cause:

- The probe heads are relatively lightweight.
- Training often waits for AOSS feature reads / DataLoader batches.
- This is not necessarily a hard failure.

Status:

- Accepted for formal safe run to prioritize stability.

Remaining risk:

- Full 2 proc/GPU across heavy VFMs should not be used blindly because the aggressive launch already showed memory risk.

## 4. Configuration And Runtime Changes

### 4.1 Data roots

Recommended data root:

```bash
INSCENE_DATA_ROOT='fj:s3://baiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2'
```

This root directly contains scene directories.

Streaming feature base:

```bash
STREAM_ROOT_BASE='fj:s3://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX'
```

Concrete streaming feature roots follow:

```text
${STREAM_ROOT_BASE}/${vfm}_prefix_${prefix}_m0p35_r0p5
```

Target feature roots:

```text
fj:s3://baiqiao/InsScene-15K/FEAT_TARGET_BY_PREFIX/${vfm}_prefix_${prefix}_m0p35_r0p5
```

Important:

- Do not set `INSCENE_STREAMING_FEAT_ROOT` to the parent directory during training.
- Either unset it or set it to the concrete prefix root.

### 4.2 Manifest configuration

Formal C retraining used:

```text
manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl
manifests/c_retrain_20260722/inscene_c_h124_prefix12.jsonl
manifests/c_retrain_20260722/inscene_c_h124_prefix16.jsonl
manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl
```

All were built with:

```bash
--action-horizons 1,2,4
```

This supports C1/C2/C3 from a common stricter sample set.

### 4.3 Formal training configuration

Aggressive config attempted and rejected:

```bash
SLOTS_PER_NODE=10
BATCH_SIZE=8
NUM_WORKERS=4
```

Reason rejected:

- Caused CUDA OOM on heavy VFMs.

Final safe config:

```bash
SLOTS_PER_NODE=8
BATCH_SIZE=4
BATCH_SIZE_VAL=4
NUM_WORKERS=4
NUM_WORKERS_VAL=2
```

Rationale:

- 1 process per GPU.
- Avoids double-process memory risk.
- Keeps AOSS pressure bounded.

### 4.4 Checkpoint / autoresume

All formal retraining jobs were launched with:

```bash
autoresume=false
```

Reason:

- Old C checkpoints were invalid due to the previous target bug.
- Resuming from those checkpoints would contaminate retraining.

Run folders were also changed:

```text
paths.run_folder_name=${RUN_ID}/${tag}
```

This avoids writing into old bad run directories.

### 4.5 Rescue run configuration

Rescue run:

```bash
RUN_ID=c_retrain_rescue_20260723_1748
NODES_CSV=10.119.17.135,10.119.16.151,10.119.31.225
JOB_LIST=logs/inscene15k_streaming/c_retrain_safe_20260722_1435/rescue_20260723_1748/rescue_jobs.tsv
SLOTS_PER_NODE=16
BATCH_SIZE=4
NUM_WORKERS=4
```

Although `SLOTS_PER_NODE=16`, only 10 rescue jobs were scheduled total, distributed `4/3/3` over three nodes.

## 5. Data And Evaluation Flow Changes

### 5.1 Dataset reading

Training now uses sample-level local manifests through `INSCENE_MANIFEST`.

Effect:

- Avoids remote AOSS directory exploration at dataset startup.
- Reduces startup time.
- Makes scene/sample inclusion reproducible.

### 5.2 Scene filtering

Final C manifests include 681 scenes.

Excluded scenes:

- `a003a6585e`
- `c24f94007b`

Reasons:

- `c24f94007b`: incomplete base data, `missing_images_dir`.
- `a003a6585e`: missing/invalid for C target or long-horizon samples.

### 5.3 Sample construction

Final manifests are built with:

```bash
--prefix-lengths 8|12|16|24
--action-horizons 1,2,4
```

Sample counts:

- prefix 8: `2724`
- prefix 12: `2724`
- prefix 16: `2723`
- prefix 24: `2721`

The small differences come from legal temporal-window availability under longer prefix/future requirements.

### 5.4 Metric/evaluation flow

This session did not run final metric evaluation over completed C checkpoints.

It completed C probing training.

Next step should be:

- run C1/C2/C3 evaluation scripts over the final checkpoints;
- summarize metrics into the slide/report table;
- ensure the evaluation uses the same fixed loader and manifest assumptions.

### 5.5 Log parsing

Implemented ad hoc parsing for:

- job completion via `END ... rc=0`;
- `train_epoch_end`;
- `train_batch_end`;
- epochs/hour speed table.

Output:

```text
logs/inscene15k_streaming/c_retrain_safe_20260722_1435/epoch_speed.tsv
```

## 6. Executed Experiments

### 6.1 Manifest audits

Purpose:

- resolve 682/684 discrepancy;
- ensure feature roots are complete;
- generate formal C manifests.

Key outputs:

- `manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl`
- `manifests/c_retrain_20260722/inscene_c_h124_prefix12.jsonl`
- `manifests/c_retrain_20260722/inscene_c_h124_prefix16.jsonl`
- `manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl`

Final result:

- success.
- 681 valid scenes.
- no duplicate sample IDs.

Result credibility:

- High for the checked DINO/default-layer paths.
- Directory-level audits showed the same 681-scene valid set across all 6 VFMs/prefix roots.
- Sample file checks were run across all VFM/prefix default-layer roots.

### 6.2 Single-process AOSS benchmark

Run tag:

```text
single_nw4_500_20260722T0852Z
```

Config:

- VFM: `dino`
- probe: `action_dynamics`
- prefix: `8`
- `num_workers=4`
- `BATCHES=500`
- `PROCESSES=1`
- `GPU_IDS=0`

Purpose:

- confirm longer stability beyond 50-batch smoke test.

Result:

- exit code: `0`
- throughput: `0.82 it/s`
- GPU0 peak memory: `1794 MB`
- no error keywords.
- estimated batch wall-time:
  - P50: `1.47s`
  - P95: `1.97s`
  - P99: `1.97s`

Credibility:

- Good for DINO C1/prefix8 steady-state behavior.
- Not a pure cold-cache test.
- Not a pure DataLoader data-time measurement; it is batch wall-time from progress logs.

### 6.3 Single-card dual-process benchmark

Run tag:

```text
two_proc_gpu0_nw4_500_20260722T0907Z
```

Config:

- VFM: `dino`
- probe: `action_dynamics`
- prefix: `8`
- `num_workers=4` per process
- 2 processes on one GPU
- `BATCHES=500` per process

Result:

- both training processes exit code `0`.
- per-process throughput:
  - `1.06 it/s`
  - `1.07 it/s`
- combined throughput: `2.13 it/s`
- peak GPU memory: about `3585 MB`
- P99 batch wall-time: about `1.50s` per process.

Important note:

- The wrapper itself returned non-zero once because the script was edited while it was still running, causing a shell heredoc parse issue. The training processes themselves completed successfully with rc `0`.

Conclusion:

- Single-card dual-process can help when data wait dominates.
- It should not be generalized blindly to heavier VFMs.

### 6.4 Aggressive formal C retrain attempt

Run ID:

```text
c_retrain_20260722
```

Config:

- `SLOTS_PER_NODE=10`
- `BATCH_SIZE=8`
- `NUM_WORKERS=4`

Purpose:

- maximize throughput over 5 nodes/40 H800 GPUs.

Result:

- failed due to CUDA OOM, especially for heavy 3072-channel VFMs.
- stopped and discarded.

Credibility:

- Not valid for results.
- Useful as capacity evidence.

### 6.5 Safe formal C retrain

Run ID:

```text
c_retrain_safe_20260722_1435
```

Config:

- 5 nodes initially.
- `SLOTS_PER_NODE=8`
- `BATCH_SIZE=4`
- `NUM_WORKERS=4`
- one process per GPU.
- `autoresume=false`.
- manifest-driven data loading.

Initial status:

- 40 jobs active.
- no errors after 20+ minutes.

Later infrastructure event:

- two nodes were reclaimed:
  - `10.119.20.37`
  - `10.119.19.88`

Final result from this run:

- 62/72 tasks completed successfully.
- 10 tasks required rescue.

Credibility:

- Valid for completed 62 tasks.

### 6.6 Rescue C retrain

Run ID:

```text
c_retrain_rescue_20260723_1748
```

Config:

- available nodes:
  - `10.119.17.135`
  - `10.119.16.151`
  - `10.119.31.225`
- 10 rescue jobs from:

```text
logs/inscene15k_streaming/c_retrain_safe_20260722_1435/rescue_20260723_1748/rescue_jobs.tsv
```

Result:

- 10/10 tasks completed successfully.

Credibility:

- Valid replacement for jobs interrupted or not started due to reclaimed nodes.

### 6.7 Final C probing completion check

Final result:

- total expected tasks: `72`
- done ok: `72`
- failed: `0`
- incomplete: `0`

Breakdown:

- `c_retrain_safe_20260722_1435`: 62 completed.
- `c_retrain_rescue_20260723_1748`: 10 completed.

Current process state:

- no `c_retrain` training processes still running.

## 7. Experimental Results And Main Findings

### 7.1 Feature completeness

Evidence:

- directory-level AOSS audit across all 6 VFMs and 4 prefixes.
- sample file check for default layer feature files.
- final manifest builds with audit files.

Finding:

- C-series features are sufficiently complete for the 681-scene final manifest set.

Confidence:

- High for default-layer C training used in this session.

### 7.2 Training completion

Evidence:

- completion parser over logs:

```json
{
  "counts": {
    "done_ok": 72
  },
  "total": 72
}
```

Finding:

- C-series probing training is complete.

Confidence:

- High.

### 7.3 Training speed

Per-task speed table:

```text
logs/inscene15k_streaming/c_retrain_safe_20260722_1435/epoch_speed.tsv
```

At the time the table was generated:

- total tasks: `72`
- with speed statistics: `62`
- done ok: `23`
- running: `40`
- queued: `9`
- done bad: `0`

Average epochs/hour at that time:

- `vjepa2`: `9.43`
- `f3r`: `7.13` for the 2 jobs with speed at that time.
- `wan`: `4.66`
- `dino`: `4.32`
- `cogvideox`: `3.12`
- `aether`: `2.41`

Slowest observed:

- `aether` and `cogvideox`, especially large prefixes.

Fastest observed:

- `vjepa2 action_dynamics`, around `10-11 epoch/hour`.

Confidence:

- Good as an operational speed snapshot.
- The table was generated before all jobs completed, so it should not be treated as a final runtime table unless regenerated after completion.

### 7.4 GPU utilization

Evidence:

- `nvidia-smi` showed low instantaneous utilization.
- Benchmarks showed data-wait behavior.

Finding:

- GPU utilization is low because probe compute is lightweight and AOSS/DataLoader waiting dominates.

Confidence:

- High as an operational observation.

### 7.5 Multiprocess scheduling

Evidence:

- DINO single-card dual-process benchmark succeeded and improved aggregate throughput.
- Aggressive formal run with heavier VFMs and batch size 8 caused OOM.

Finding:

- Multiprocess-per-GPU is useful only under conservative memory conditions.
- It should not be combined with large batch size or heavy 3072-channel VFMs without staged testing.

Confidence:

- High for this cluster/session.

## 8. Project Design Analysis And Improvement Ideas

### 8.1 Manifest-first training

Status: implemented.

Recommendation:

- Keep manifest-driven training as the default for AOSS-backed InsScene training.

Reason:

- Avoids expensive remote directory scans.
- Enables reproducible scene/sample sets.
- Records exclusion reasons and checksums.

### 8.2 Audit sidecars

Status: implemented.

Recommendation:

- Keep `included_scenes.txt`, `excluded_scenes.jsonl`, and `.meta.json` for every manifest used in a paper result.

Completion standard:

- Manifest metadata includes SHA256, scene/sample counts, split counts, roots, prefix, horizons, VFM, and feature layer.

### 8.3 Conservative launch defaults

Status: implemented.

Recommendation:

- Use:

```bash
SLOTS_PER_NODE=8
BATCH_SIZE=4
NUM_WORKERS=4
```

- Only increase concurrency after a 20-30 minute stability check.

### 8.4 Full C evaluation

Status: completed in follow-up eval run `c_eval_20260724_1903`.

What was run:

- Evaluated all retrained C1/C2/C3 checkpoints using `vidfm3d/eval_diag.py`.
- Used only checkpoints from:
  - `c_retrain_safe_20260722_1435`
  - `c_retrain_rescue_20260723_1748`
- Deduplicated safe/rescue overlaps and preferred rescue checkpoints.
- Used final `h1,2,4` manifests:
  - `inscene_c_h124_prefix8.jsonl`
  - `inscene_c_h124_prefix12.jsonl`
  - `inscene_c_h124_prefix16.jsonl`
  - `inscene_c_h124_prefix24.jsonl`

Results:

- 72/72 eval summaries completed.
- Probe counts:
  - `action_dynamics`: 24
  - `path_integration`: 24
  - `counterfactual`: 24
- No missing summaries.
- No zero-valid or `n=0` summaries.

Output artifacts:

- Raw 72-row table:
  - `logs/inscene15k_streaming/results/c_eval_20260724_1903/c_series_raw_metrics.csv`
- Best-prefix-per-task/VFM table:
  - `logs/inscene15k_streaming/results/c_eval_20260724_1903/c_series_best_by_probe_vfm.csv`
- Markdown summary:
  - `logs/inscene15k_streaming/results/c_eval_20260724_1903/c_series_final_metrics.md`
- Validation JSON:
  - `logs/inscene15k_streaming/results/c_eval_20260724_1903/c_series_validation.json`

Headline best results:

- C1 `action_dynamics`: DINO prefix 8, `global_R@1=0.4783`.
- C2 `path_integration`: DINO prefix 8, `global_R@1=0.1944`.
- C3 `counterfactual`: DINO prefix 8, `intervention_validity=0.6120`.

Completion standard:

- Completed. The final tables are ready for slides/reporting.

### 8.5 Final runtime table

Status: partially done.

Recommendation:

- Regenerate epochs/hour after all jobs are complete, including rescue jobs.

Completion standard:

- One final table combining:
  - 62 tasks from `c_retrain_safe_20260722_1435`;
  - 10 tasks from `c_retrain_rescue_20260723_1748`.

### 8.6 Node-loss resilient scheduler

Status: partially implemented through manual rescue.

Recommendation:

- Add automatic job state tracking and resubmission.
- Keep a canonical expected job matrix and detect:
  - done ok;
  - failed;
  - currently running;
  - missing.

Completion standard:

- Relaunch script can resume only missing/failed jobs automatically.

### 8.7 Cold vs steady-state benchmark separation

Status: suggested, partially recorded in benchmark conditions.

Recommendation:

- Record cache state explicitly for future performance claims.
- If possible, run cold-start and steady-state AOSS benchmarks separately.

## 9. Current Project Status

Current code status:

- The current code can run C-series training successfully with manifest-driven data loading.
- New scripts are untracked:
  - `scripts/build_inscene_streaming_manifest.py`
  - `scripts/run_aoss_streaming_benchmark.sh`
  - `scripts/launch_c_retrain_40gpu.sh`
  - `scripts/run_c_eval_slots.py`
- New manifest directory is untracked:
  - `manifests/c_retrain_20260722/`

Recommended config:

```bash
SLOTS_PER_NODE=8
BATCH_SIZE=4
NUM_WORKERS=4
autoresume=false
```

Trusted results:

- C-series probing training completion is trustworthy:
  - 72/72 jobs done ok.
- C-series final validation metrics are now trustworthy for reporting:
  - 72/72 eval summaries completed in `c_eval_20260724_1903`.
  - C1/C2/C3 each have 24 valid summaries.
  - No missing summaries or zero-valid summaries.

Results needing next verification:

- Old C checkpoints remain invalid and should not be used.
- Runtime/epoch-speed table still should be regenerated if reporting training efficiency.

Most serious remaining blocker:

- No blocker remains for reporting C-series validation metrics.

Technical debt:

- Launcher/rescue tooling is functional but still a shell-based supervisor.
- Job status aggregation should be made more robust.
- Final speed table should be regenerated after all jobs and rescue jobs are complete.
- `scripts/run_c_eval_slots.py` should be folded into a more permanent evaluation launcher if C evals will be repeated.

## 10. TODOs

### P0: Must Handle Immediately

#### P0.1 Run C-series evaluation on retrained checkpoints

Status: completed.

What was run:

- Ran C evaluation for:
  - C1 `action_dynamics`
  - C2 `path_integration`
  - C3 `counterfactual`

Why:

- Training was complete, but final C metrics needed to be produced from fixed-target retrained checkpoints.

Recommended implementation:

- Implemented through `scripts/run_c_eval_slots.py`.
- Enumerated checkpoints under:
  - `logs/inscene15k_streaming/runs/c_retrain_safe_20260722_1435/...`
  - `logs/inscene15k_streaming/runs/c_retrain_rescue_20260723_1748/...`
- Evaluate with fixed loader and correct target roots/manifests.

Completion standard:

- Every one of the 72 retrained C jobs has an evaluation result: completed.
- No old invalid C checkpoints were used.

Validation:

- 72/72 summaries exist.
- All summaries have `n>0`.
- Probe counts are 24/24/24 for C1/C2/C3.

#### P0.2 Build a final C results table

Status: completed.

What:

- Merge safe-run and rescue-run evaluation outputs.

Why:

- Final C slide/report needs one canonical table.

Completion standard:

- One 72-row raw table per VFM/prefix/probe:
  - `logs/inscene15k_streaming/results/c_eval_20260724_1903/c_series_raw_metrics.csv`
- One best-prefix table for slides:
  - `logs/inscene15k_streaming/results/c_eval_20260724_1903/c_series_best_by_probe_vfm.csv`
- Markdown summary:
  - `logs/inscene15k_streaming/results/c_eval_20260724_1903/c_series_final_metrics.md`

### P1: Should Complete Before Next Experimental Round

#### P1.1 Regenerate final epoch-speed table

What:

- Recompute `epochs/hour` after all jobs completed.

Why:

- Existing `epoch_speed.tsv` was generated mid-run.

Recommended implementation:

- Extend the current parser to combine safe and rescue runs.

Completion standard:

- 72 rows, one per task.
- Includes wall-clock time, epochs/hour, run ID, rc.

#### P1.2 Commit or otherwise preserve new scripts/manifests

What:

- Decide whether to add the new scripts and manifests to version control or artifact storage.

Why:

- They are required to reproduce the successful C retraining.

Completion standard:

- Scripts are tracked or archived.
- Manifest SHA256s are documented.

#### P1.3 Add automatic rescue/resume mode to launcher

What:

- Add a `--resume-missing` mode or equivalent.

Why:

- Node reclamation is likely in this environment.

Completion standard:

- The launcher can compute missing/failed jobs and relaunch only those.

### P2: Research And Engineering Enhancements

#### P2.1 More systematic AOSS performance profiling

What:

- Add direct DataLoader `data_time` instrumentation.

Why:

- Current logs provide batch wall-time, not pure data loading time.

Completion standard:

- P50/P95/P99 data time per job.
- AOSS timeout/retry counters if available.

#### P2.2 Controlled node-level concurrency experiments

What:

- Test:
  - 1 proc/GPU
  - selected 2 proc/GPU
  - VFM-specific concurrency limits

Why:

- DINO tolerated dual-process well, but heavy VFMs OOMed under aggressive settings.

Completion standard:

- Safe concurrency profile per VFM/probe/prefix.

#### P2.3 Manifest compression or schema split

What:

- Split scene-level metadata from sample-level records or move to Parquet.

Why:

- JSONL records are large due to repeated scene metadata.

Completion standard:

- Faster load and smaller artifact size without changing sample semantics.

## 11. Key File Index

| File or directory | Role in this session | Modified | Main content/change | Current status |
| --- | --- | ---: | --- | --- |
| `scripts/build_inscene_streaming_manifest.py` | Offline sample manifest builder and feature/data auditor | Yes | Added audit files, exclusion reasons, feature checks, metadata, checksum | New/untracked; validated and used |
| `scripts/run_aoss_streaming_benchmark.sh` | AOSS benchmark wrapper | Yes | Added stable training exit handling, GPU stats, conditions, summary parsing | New/untracked; validated and used |
| `scripts/launch_c_retrain_40gpu.sh` | Multi-node C retrain launcher | Yes | Added bounded remote supervisors, safe config support, `JOB_LIST`, `NODES_CSV` | New/untracked; validated and used |
| `scripts/run_c_eval_slots.py` | Multi-slot C evaluation launcher | Yes | Discovers safe/rescue checkpoints, deduplicates jobs, prefers rescue, runs `eval_diag.py` with manifests | New/untracked; validated and used for `c_eval_20260724_1903` |
| `vidfm3d/data/components/inscene15k_dataset.py` | Dataset loader | Not newly edited in final part; existing modified state used | Manifest loading and flat target fallback were used/verified | Functioned correctly in training |
| `vidfm3d/utils/feature_layers.py` | Feature filename/default layer mapping | No | Used to audit expected feature filenames | Verified defaults for all VFMs |
| `scripts/run_streaming_probe_sweep.sh` | Underlying train command generator | No | Used by launcher with controlled env/overrides | Worked for formal training |
| `manifests/c_retrain_20260722/` | Final C manifests | Yes/new artifacts | `h1,2,4` manifests for p8/p12/p16/p24 and audit metadata | Used by formal training |
| `logs/inscene15k_streaming/c_retrain_safe_20260722_1435/` | Main safe formal C retrain run | Generated | 62 successful C jobs, launch logs, checkpoints under run folders | Valid training artifacts |
| `logs/inscene15k_streaming/c_retrain_rescue_20260723_1748/` | Rescue retrain run after node loss | Generated | 10 successful rescue jobs | Valid training artifacts |
| `logs/inscene15k_streaming/c_retrain_safe_20260722_1435/rescue_20260723_1748/rescue_jobs.tsv` | Rescue job list | Generated | The 10 jobs not completed in the original safe run | Used successfully |
| `logs/inscene15k_streaming/c_retrain_safe_20260722_1435/epoch_speed.tsv` | Mid-run speed table | Generated | Epochs/hour for tasks with progress at generation time | Useful but should be regenerated now |
| `logs/inscene15k_streaming/runs/c_eval_20260724_1903/` | Final C eval run directory | Generated | 72 `eval/val_summary.json` and prediction dumps | Complete and valid |
| `logs/inscene15k_streaming/results/c_eval_20260724_1903/` | Final C metrics tables | Generated | Raw CSV, best-by-task/VFM CSV, markdown summary, validation JSON | Ready for slides/report |
| `/mnt/local_cache/inscene/benchmarks/single_nw4_500_20260722T0852Z/` | Single-process benchmark logs | Generated | 500-batch stable benchmark | Valid operational evidence |
| `/mnt/local_cache/inscene/benchmarks/two_proc_gpu0_nw4_500_20260722T0907Z/` | Dual-process benchmark logs | Generated | 2-process GPU sharing benchmark | Training valid; wrapper parse issue noted |
| `docs/c_retrain_session_report_20260724.md` | This report | Yes | Full session summary | Current document |

## 12. One-page Summary

This session turned the C-series experiments from a broken/invalid state into a completed retraining run.

Main accomplishments:

- Added a manifest builder/auditor that records included/excluded scenes, exclusion reasons, metadata, and checksums.
- Resolved the `682/684` scene discrepancy:
  - final C-valid set is 681 scenes;
  - excluded `a003a6585e` and `c24f94007b`.
- Fixed audit path assumptions for streaming features:
  - actual layout is `window_xxxx/prefix_xxxxxx/feature*.sft`.
- Built final C manifests for prefixes `8/12/16/24` with `action_horizons=1,2,4`.
- Benchmarked AOSS training:
  - single process DINO C1: `0.82 it/s`;
  - dual process DINO C1: `2.13 it/s` combined.
- Implemented a multi-node launcher.
- Attempted an aggressive config, observed OOM, and backed off to a safe config.
- Completed formal C retraining:
  - 72/72 tasks succeeded;
  - 62 completed in `c_retrain_safe_20260722_1435`;
  - 10 completed in `c_retrain_rescue_20260723_1748`.
- Completed formal C evaluation:
  - 72/72 eval summaries succeeded in `c_eval_20260724_1903`;
  - C1/C2/C3 each have 24 valid summaries;
  - no missing summaries and no zero-valid summaries.
- Generated final C metrics tables:
  - `logs/inscene15k_streaming/results/c_eval_20260724_1903/c_series_raw_metrics.csv`;
  - `logs/inscene15k_streaming/results/c_eval_20260724_1903/c_series_best_by_probe_vfm.csv`;
  - `logs/inscene15k_streaming/results/c_eval_20260724_1903/c_series_final_metrics.md`.
- Updated the result canvas with C-series best-prefix results.
- Confirmed no C retrain processes remain.

Key fixes:

- Manifest-driven dataset startup avoids AOSS directory scanning.
- Flat target feature layout is supported by the loader.
- `INSCENE_STREAMING_FEAT_ROOT` must not be set to the parent feature directory during training.
- Old invalid C checkpoints were avoided with `autoresume=false` and new run folders.

Main conclusions:

- C probing training is now complete and reliable.
- The old C checkpoints remain invalid and should not be used.
- Final C metrics are now available for reporting.
- Best C results by current primary metrics:
  - C1 `action_dynamics`: DINO prefix 8, `global_R@1=0.4783`;
  - C2 `path_integration`: DINO prefix 8, `global_R@1=0.1944`;
  - C3 `counterfactual`: DINO prefix 8, `intervention_validity=0.6120`.
- GPU utilization remains low because data loading/AOSS wait dominates, but safe training completed successfully.

Next most important step:

- Use the generated C metrics in slides/reporting, and regenerate the final epoch-speed/runtime table if training efficiency numbers are needed.
