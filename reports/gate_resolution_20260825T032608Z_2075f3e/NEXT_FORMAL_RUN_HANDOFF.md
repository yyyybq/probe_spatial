# Next Formal Run Handoff

Source gate tag: `gate_resolution_20260825T032608Z_2075f3e`  
State: **BLOCKED before own-GPU gate; do not launch formal experiments yet.**

## Resume point

Do not repeat gates 1–3 unless the worktree, credentials, scheduler state, or input artifacts have changed. Resume at gate 4 only after establishing a legitimate scheduler identity for this user/project.

Required identity resolution:

1. Activate a SCO profile whose create/describe response reports `ownership.user_name=yinbaiqiao`; or obtain explicit project-administrator documentation naming an authorized service identity for this user and run.
2. Confirm create permission for workspace `aigc`, pool `zoetrope`, one `N4lS.Iq.I80.1` GPU worker, PV mount `/mnt/umm`, and the selected project image.
3. Do not use the current credential while its authoritative API ownership remains `guchenyang`. Do not enter, copy, reuse, stop, or kill an existing job.
4. Submit exactly one short, isolated, normal-priority 1-GPU gate. Verify ownership from the returned job record before entering it or allowing the gate command to run.

The historical compatible resolved resource configuration and the unresolved 403 evidence are in `SCHEDULER_SELECTED_RESOLVED_CONFIGS.json` and `SCHEDULER_AUDIT.md`. Historical validity of a configuration is not authorization to reuse a historical job or credential.

## Frozen code and manifests

- Base commit: `2075f3eb270cb0e8907c3927b6b079c64965b75a`
- Actual code identity: source-only snapshot manifest SHA256 `fa80790c33a30b316d60fe34714c2fd175978db956726c76c76c3f8425bf691e`
- Snapshot tar SHA256: `5cdbbe4c75ef704a8748c9520d1350da664d9e0ae59c83849a1482185a64250e`
- Snapshot: `/mnt/umm/users/yinbaiqiao/probe_spatial/artifacts/gate_resolution_20260825T032608Z_2075f3e/source_snapshot.tar`
- Split manifest: `/mnt/umm/users/yinbaiqiao/probe_spatial/configs/splits/inscene15k_v1.json`
- Split SHA256: `e01a2194143405ed8f08613ec0bc672c7c237fa9bb21d2b8da441e7ad56bf086`
- Canonical C manifest: `/mnt/umm/users/yinbaiqiao/probe_spatial/manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl`
- C manifest SHA256: `772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c`

Before using the snapshot as the experiment identity, verify both snapshot hashes and `HANDOFF_INPUTS.sha256`. If live source differs from `SOURCE_SNAPSHOT_MANIFEST.csv`, either restore an exact isolated copy from the snapshot or freeze a new immutable identity and document the difference. Never label dirty `HEAD` alone as the code identity.

## Exact environment

```bash
export PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
export INSCENE_DATA_ROOT='fj:s3://baiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2'
export INSCENE_STREAMING_FEAT_ROOT='fj:s3://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX/dino_prefix_8_m0p35_r0p5'
export INSCENE_TARGET_FEAT_ROOT='fj:s3://baiqiao/InsScene-15K/FEAT_TARGET_BY_PREFIX/dino_prefix_8_m0p35_r0p5'
export INSCENE_SPLIT_MANIFEST='/mnt/umm/users/yinbaiqiao/probe_spatial/configs/splits/inscene15k_v1.json'
export AOSS_CONF='/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf'
export AOSS_API_HOST='aoss-internal.cn-fz-01.fjscmsapi-oss.com'
```

Expected software: Python 3.11.15, PyTorch `2.6.0+cu124`, CUDA build 12.4. Confirm actual driver/runtime versions inside the allocated worker.

## Gates 4–8 to execute

Use the already prepared script semantics from:

`/mnt/umm/users/yinbaiqiao/probe_spatial/reports/recovery_20260822T034513Z_2075f3eb/run_gpu_aoss_gate_sco.sh`

If path/environment adaptation is necessary, copy it into the new gate tag and record its SHA/diff; do not modify the prior recovery report directory and do not change test semantics.

On the compliant compute worker, record status, elapsed time, exact path, exception, byte count, and hash where practical for:

1. `nvidia-smi`, exact Conda Python, Torch import, `torch.cuda.is_available()`, device identity, and a small CUDA tensor forward.
2. Known local split manifest.
3. Exact known AOSS `target_indices.npy`; read only the known object, never scan a large parent prefix.
4. Exact known streaming-prefix feature.
5. Exact known target-isolated feature.
6. The same reads through project `aoss_fs`.
7. One `InsScene15KDataset` sample, one actual C sample, and one B2 sample.

Only after all critical compute/AOSS/DataLoader steps pass, build a new 1–5-scene DINO prefix-8 strict-B2 smoke manifest under the new gate's `artifacts/` directory. Verify on real data:

- `n_valid > 0`;
- target object raw pixels are zero for every accepted strict sample;
- changing target-frame object mask does not change the past-only centroid;
- changing a past visible object mask does change the centroid;
- target-frame depth/object pixels do not enter centroid construction;
- GT is identical across VFMs for the same canonical sample;
- p8/p12/p16/p24 retain the same scene, object, and target moment;
- DINO p8 streaming feature, target-isolated feature, object query, GT target, and DataLoader all load successfully.

Do not substitute synthetic tests for this real smoke. If any gate fails, stop before formal launch and emit another blocked gate report.

## Runner scope

Current runner SHA256: `ee95d1d4d2ba554a919d2668d7be555222dfd1a0ba440a5a83d7a67d1eba9486`.

Existing-cache DINO p8 strict-B2 and C baseline gates may proceed without `--streaming-window-obs-len` or V-JEPA2 `predictor_target`. Do not perform new cross-prefix cache extraction requiring a shared full observation window, or new V-JEPA2 predictor-target extraction. No original implementation of either interface was recovered. Any future replacement must be explicitly labeled `RECONSTRUCTED_INTERFACE` and supplied with the requested leakage/alignment regression tests.

## Additional repository blocker

Before formal expansion, resolve the 34 substantive 0/1-byte source/config files listed in:

`/mnt/umm/users/yinbaiqiao/probe_spatial/reports/gate_resolution_20260825T032608Z_2075f3e/CURRENT_TINY_SOURCE_GATE_INVENTORY.tsv`

They share mtime `2026-08-25 02:56:38 UTC` and include formal launcher scripts and configs. This gate window did not investigate or restore them because it was explicitly prohibited from redoing empty-file recovery. Preserve the current snapshot first; use a separate authorized recovery task, evidence-backed last-good sources, and hashes. Do not guess their contents.

## Formal identity and launch condition

After, and only after, gates 1–8 are all PASS, freeze the final code/manifests/environment again and mint:

`RUN_TAG=paper_validity_exec_<UTC_TIMESTAMP>_<IMMUTABLE_CODE_HASH>`

Do not reuse `paper_validity_20260821T200224Z_2075f3e`, `recovery_20260822T034513Z_2075f3eb`, or `paper_strict_b2_postrecovery_20260822T0425Z_2075f3e`. Create `reports/<RUN_TAG>/RUN_PROVENANCE.md` before training, recording source identity, restored-file hashes, manifest hashes, feature roots, Conda/Python/Torch/CUDA, scheduler image, model/layer/timestep, and seed.

Then follow the requested order: DINO p8 strict-B2 smoke train/eval, DINO p8 B2 baselines, DINO p8 C1/C2 fair baselines, metric/sample-ID validation, and only then expansion to all models/prefixes. Preserve canonical sample IDs, use paired common sets for primary comparisons and prefix AUC, and never mix old invalid C checkpoints into the run.

## Current non-results

There are no new real strict-B2 numbers, B2 baseline numbers, C1/C2 fair-baseline numbers, common-set tables, prefix AUC, difficulty curves, correlations, or capacity-pilot results in this gate tag. No formal or gate background job from this window remains running, and no canonical result was overwritten.
