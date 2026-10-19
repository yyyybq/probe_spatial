# Q00 Phase 1 direct-node execution log

Run tag: `q00_phase1_wave_b_e_20260829T075939Z_2075f3eb`  
Frozen bundle: `reports/q00_phase1_runs/q00_phase1_wave_b_e_20260829T075939Z_2075f3eb/bundle.json`  
Bundle SHA256: `5a0d765b973ccf73b52b8827dc45eb2f8a8803b37a3e419b396a438b511fe4af`

## 2026-08-30 safety handoff

- Direct host: `pt-314e0f9747cc44dc9e91dc07705c52a4-worker-0` (`10.119.27.217`).
- Repository: `/mnt/umm/users/yinbaiqiao/probe_spatial`.
- Repository commit from the shared local checkout: `2075f3eb270cb0e8907c3927b6b079c64965b75a`; the direct host image has no `git` executable, so commit identity is checked from the shared checkout and frozen bundle provenance.
- Python: `/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python`; PyTorch `2.6.0+cu124`; CUDA runtime `12.4`; `torch.cuda.is_available()=true`; eight visible NVIDIA H800 GPUs.
- GPUs 0–2 were occupied by unrelated `vagen-lite` processes (~19.7 GiB each). GPUs 3–7 were fully free and are the only eligible direct-node lanes.
- `/mnt/umm` had 32 TiB free. `/mnt/local_cache` did not yet exist; the outer launcher creates only the versioned cache namespace required by the frozen runner.
- Protected `experiments/registry.csv` and seven `results/canonical/*.csv` SHA256 values matched the frozen pre-image exactly.

## Duplicate-writer check

- SCO jobs `pt-ez076v1l`, `pt-9eu2jfbb`, and `pt-iy371t7w` were each `STARTING`, with `start_time=null` and worker replicas `0/1`.
- Exact reconciliation found zero output directories, summaries, predictions, preflights, commands, runner status files, or audit locks for all 99 frozen Wave-B rows.
- All three SCO jobs were cancelled and independently confirmed `SUSPENDED` before any direct-node execution.

## Execution policy

- The frozen bundle and frozen execution scripts are not modified.
- New outer wrapper: `scripts/q00_phase1_direct_node_launcher.sh`; frozen direct-launcher SHA256 before execution: `2a7b044859969b041b5f6050b05c084c510564124b05c27550c18381af84bd6e`.
- The wrapper uses per-action `flock`, refuses pre-existing logs, checks exact bundle SHA and host identity, and refuses GPUs using more than 1 GiB before launch.
- Canary assignment: LingBot→GPU 3, MiniMax-H3→GPU 4, Wan→GPU 5.
- Remaining 96 are forbidden until the frozen canary audit and lock pass.
- All eval outputs remain in the frozen versioned run directories and staging paths; canonical promotion is a separate, audited action.

## Direct-node preflight result

- The direct host independently matched 11/11 live execution/runtime files to the bundle inventory.
- All 99 checkpoint paths were readable and matched frozen sizes; all four frozen manifests matched SHA256.
- The three canary checkpoint SHA256 values matched the inventory.
- AOSS `contains` checks passed for each canary's exact sample prefix index and streaming tensor; Wan's exact target directory, target-index object, and target tensor also passed.
- The versioned cache directory `/mnt/local_cache/inscene/aoss_cache/q00_phase1_wave_b_e_20260829T075939Z_2075f3eb/shared` was created on the 55 TiB-free node root filesystem. No legacy cache was reused.
- Remote `git` and `nvcc` executables are absent. Commit identity was recovered from `.git/HEAD` plus `.git/refs/heads/main` and matched the bundle; eval uses the frozen precompiled PyTorch runtime, so neither absence changes the eval contract.

## Canary attempt 1 — wrapper import failure

- Start/end: `2026-08-30T03:43:29Z`–`03:43:34Z`.
- LingBot, MiniMax-H3, and Wan each exited with rc=1 before the frozen runner entered `main()`.
- Exact error in all three preserved logs: `ModuleNotFoundError: No module named 'vidfm3d'` at the parent runner import.
- No runner status, preflight, output directory, summary, prediction, audit input, or canonical write was created; GPU 3/4/5 memory stayed at zero.
- Scoped repair: the outer wrapper now sets `PYTHONPATH=/mnt/umm/users/yinbaiqiao/probe_spatial` only for the parent frozen runner and gate entrypoints, and performs an import smoke test before launch. Frozen scripts and shard files remain unchanged.
- Retry logs and locks are versioned with `DIRECT_ATTEMPT=r2`; attempt-1 logs are retained.
- Repaired wrapper SHA256: `11f58f31a884b30add642a7929f1b65c1af342a57180db81970baaafba8ae371`.

## Canary attempt 2 — frozen shape-contract failure

- Start/end: `2026-08-30T03:45:55Z`–`03:46:11Z`.
- The parent runner, CUDA, script/runtime inventory, shard identity, checkpoint, manifest, and prefix-index gates were reached. All three canaries then failed closed before evaluator startup because the original frozen runner incorrectly required the raw cache tensor's first dimension to equal the manifest frame prefix.
- Observed exact tensors:
  - LingBot: `[13,30,52,2048]` for an 8-frame prefix.
  - MiniMax-H3: `[22,15,26,5376]` for a 16-frame prefix.
  - Wan: `[21,30,52,1536]` for a 12-frame prefix.
- Each canary exited rc=2. Only versioned launcher evidence and `q00_failure.json` were written; no preflight, evaluator, summary, prediction, audit, promotion, registry, or canonical write occurred.
- The three launcher-status SHA256 values are `ff77c4ba...`, `28722ff8...`, and `ff9fa1a1...`; the three failure SHA256 values are `de3dd949...`, `ac342130...`, and `1ebb315a...`. Full digests are frozen in the successor bundle's `provenance/direct_shape_contract.json`.

## Exact shape diagnosis

- A 99/99 remote sidecar/index audit found no missing artifacts and only three fixed raw token-grid shapes: LingBot `[13,30,52,2048]`, MiniMax-H3 `[22,15,26,5376]`, and Wan `[21,30,52,1536]`.
- The extractor repeat-tail pads every raw prefix to the model input length (49/73/81) before tokenization. The frozen dataset then maps the fixed raw temporal-token axis back to the manifest prefix via `input_length`, `valid_length`, `pad_mode`, and derived `vfm_idx`.
- Therefore raw cache `T` and manifest prefix `P` are distinct. The original `T==P` gate was a deterministic validation bug, not a feature or experiment failure.
- The corrected contract remains strict: full campaign/model_variant/protocol/model identity, exact T/H/W/C, exact input length, `repeat_tail`, exact raw IDs/window/tail, derived `vfm_idx`, finite dtype, and target `[M,H,W,C]` with exact target-index count and matching H/W/C.
- Validator tests passed for all 99 parents in both the audit and promoter; 56 mutated shape/input/pad/target contracts and two full-identity mutations were all rejected.

## Successor bundle lineage

- The original bundle remains unchanged and retains the failed canary evidence.
- First candidate `q00_phase1_wave_b_e_direct_shapefix_20260830T035250Z_2075f3eb` was created but independently rejected before execution because it froze only exact shapes and omitted input-length/pad/index/target-spatial gates. It must never be launched.
- Active successor run tag: `q00_phase1_wave_b_e_direct_shapecontract_20260830T040218Z_2075f3eb`.
- Active bundle: `reports/q00_phase1_runs/q00_phase1_wave_b_e_direct_shapecontract_20260830T040218Z_2075f3eb/bundle.json`.
- Active bundle SHA256: `ed5ee041e18efd45ede9c40ff79e488fcf3c1a9ec2bc7a2fa3ff918a6df0cd5c`.
- Frozen corrected script SHA256 values:
  - runner: `1aa7e716385da8ad3150c8ad15ac8482237113f5453130b9356f77fc57d3616e`
  - auditor: `520be31512a6fe9138e43952845900ea0ecb0de96bf05d5e71f3056e0b643c45`
  - promoter: `3430d70f322e6d39a8b7c194086d8bd7c67bef9a736725be990261b2ed0f535d`
- The successor's 99 rows have the same ordered universe IDs, expected 105 atomic IDs, checkpoint inventory, and feature-root inventory as the original. Exactly eight execution-namespace/path fields changed: run tag, two frozen-copy paths, and five staging output paths. No scientific or full-identity field changed.
- Frozen shape evidence: `provenance/direct_shape_contract.json`, SHA256 `dcc12c77479693617ba8e06d1de9ae3ac1d4cffb8dbe34e8f8ce5fba27cf3725`.
- A concurrent worktree edit changed `features/run_inscene15k.py` after diagnosis. This file is not executed during Wave-B eval and was not overwritten. Its observed historical SHA remains evidence-only inside the contract JSON; live eval is bound solely to the successor's frozen runtime tree.
- Active outer direct-node wrapper: `scripts/q00_phase1_direct_node_shapefix_launcher.sh`, SHA256 `f529ac4901b1adb1fa967aa303437dbe27188c3e9f1c3def2ead518ed64b56cc`.
- Before launch, all 659 source hashes in the active bundle and all eight protected pre-image hashes matched.

## Direct-node GPU reallocation before successor canaries

- At `2026-08-30T04:10:43Z`, the earlier free GPU 3/4/5 assignment was no longer valid. GPU 0–3 and 7 had active unrelated processes and were rejected.
- GPUs 4/5/6 each showed one `ray::WorkerDict` PID started at `03:45:29Z`, but process state was `ZNl` (defunct), parent PID 1, with no live compute holder. Each card retained about 29 GiB free.
- The external wrapper was tightened to require at least 24 GiB free and to reject every holder whose live `ps` state is not zombie. It never kills, pauses, or changes another process.
- Successor canary assignment is LingBot→GPU 4, MiniMax-H3→GPU 5, Wan→GPU 6. The bundle, shard, checkpoint, manifest, runtime, audit, and promotion contracts are unchanged.

## Successor canaries — real manifest/cache lineage failure

- Start/end: `2026-08-30T04:13:51Z`–`04:14:33Z`.
- All three GPUs were fully free at launch (81,081 MiB free each), and bundle/script/CUDA/checkpoint/manifest/sample-shape gates passed.
- The full-val prefix-index audit then failed all three canaries before evaluator startup:
  - LingBot p8: manifest `00a231a370_w002_p8` expected raw IDs `[382,...,410]`; cache key `(window=2,tail=7)` records `[356,...,378]`.
  - MiniMax-H3 p16: manifest `00a231a370_w003_p16` expected `[517,...,562]`; cache records `[505,...,553]`.
  - Wan p12: manifest `00a231a370_w002_p12` expected `[382,...,431]`; cache records `[356,...,392]`.
- Each canary exited rc=2 after 14–16 seconds. No evaluator, summary, prediction, audit, promotion, registry write, or canonical write occurred. Remaining 96 remain locked.
- This gate is not being relaxed. Initial source inspection shows the historical dataset loader selected cache entries only by `(window_id, tail)` and never checked the cached `actual_input_raw_ids` against manifest `streaming_indices`; therefore a checkpoint could train cleanly while consuming a different temporal window than its label/action row.
- A full 99-row/cache-group reconciliation is in progress. Rows lacking exact cache/manifest lineage will be removed from eval-only and reclassified for exact feature reconstruction plus clean retraining; no alternate manifest or existing provisional result will be substituted.

## Full Wave-B lineage reconciliation

- The read-only direct-node audit covered all 2,208 validation associations (276 rows for each of eight model/prefix cache groups) and found 1,206 exact associations and 1,002 raw-ID mismatches. No cache group was fully exact. Aggregated over both models at each prefix, the counts are p8 240/312, p12 276/276, p16 348/204, and p24 342/210 (exact/mismatch); each individual model group follows the corresponding 120/156, 138/138, 174/102, or 171/105 pattern.
- The collision is systematic. Frozen h124 manifests build observation windows of `prefix + max(action_horizons)` = 12/16/20/28 frames, while the generic `FEAT_STREAMING` prefix index was produced from a fixed 24-frame window. Reusing `(scene, window_id, tail)` therefore aliases different raw-frame windows.
- The historical loader checked only `(window_id, tail)` and never compared cached `actual_input_raw_ids` with manifest `streaming_indices`. The 99 checkpoints consequently trained on a mixture of correct and wrong temporal windows and cannot be repaired by evaluation alone.
- All 99 Wave-B rows are reclassified from eval-only to `FEATURE_PARTIAL_OR_MISSING` / `FEATURE_EXTRACTION` / `FEATURE_THEN_RETRAIN`. Remaining-96 execution and all promotion remain forbidden.
- The Wan p8 manifest-exact versioned streaming root is reusable for six path-integration rows, but its target cache covers only horizon 1. Those rows still require isolated horizon-2 and horizon-4 target extraction followed by clean retraining; their legacy checkpoints are invalid for the frozen manifest.
- Frozen machine evidence:
  - parent classification: `/tmp/q00_waveb_reclassification_99_v2.csv`, SHA256 `8e9cb9c754a05bc2f794182353ce1ad4677ce91f249c963958fb24049dd70a81`;
  - row-level lineage: `/tmp/q00_wave_b_prefix_index_manifest_lineage_20260830.csv`, SHA256 `e3281e5dc4c77c07b7294e6d13e87289b4700c7787b0012c7cffaed425ef8c0f`;
  - Wan p8 partial reuse: `/tmp/q00_waveb_wan_p8_exact_root_reuse_6.csv`, SHA256 `0a88211bcb9fd3b1dcdd20a61f8de9e17469a2cecf3fbf700a9eedf149868bc4`.

## Wave-E legacy B2 packaging

- User authorization to place the 30 frozen `PACKAGE` rows only in an independent legacy/protocol namespace is treated as acceptance of the explicitly reconstructed cohort, not acceptance into the strict main protocol.
- Packaging command: `/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -B scripts/q00_phase1_package_legacy_b2.py`.
- Output namespace: `results/canonical_protocols/legacy_b2_shared_hidden_8_12_16_24_val_v1`.
- Audit result: 30/30 parents, 30 unique protocol-canonical IDs, 150/150 bound artifacts, zero non-finite metrics, zero summary/prediction metric mismatches, and one exact prediction-label identity per prefix cohort.
- These rows have `protocol_canonical_valid=true`, `main_canonical_valid=false`, and comparability limited to this legacy namespace. They are not labeled as strict `inscene15k_v1`.
- Registry and all seven protected main canonical tables retained their exact frozen pre-image SHA256 values.

## Phase-1 reconciliation outputs

- Wave-E full lineage resolution replaced the preliminary classification with:
  `PACKAGE=30`, `EVAL_ONLY=0`, `CLEAN_RETRAIN=17`,
  `FEATURE_THEN_RETRAIN=182`, `PERMANENTLY_BLOCKED=20`.
- The corrected legacy B2 package remains protocol-canonical only. Its 30 rows are now
  explicitly `scientifically_comparable=false` and `analysis_eligible=false`; the
  preliminary pre-lineage package was moved to a clearly marked superseded directory.
- A follow-on audit of all 642 main-canonical parents found:
  - 444 confirmed generic shared24-vs-h124 lineage errors;
  - 6 exact/versioned Dataset-canary-safe Wan p8 action-dynamics parents;
  - 192 BY_PREFIX parents with no run-bound raw-ID proof.
- Main protected canonical membership remains 642, but strict global-analysis eligibility
  is temporarily 6/642. The 636 quarantined rows are not silently deleted or rewritten.
- Required Phase-1 outputs were generated under
  `results/analysis/q00_1096_phase1/` and `reports/q00_phase1_wave_b_e.md`.
- The deterministic replacement queue contains 843 rows: 444 canonical scientific
  remediations plus 399 missing-or-noncanonical completions. Of the latter, 243 have no
  numeric result and 156 have provisional or protocol-unfrozen numeric evidence. All rows remain
  `dispatch_eligible=false`; no training or feature extraction was started by the builder.

## Wan p8 h2/h4 target remediation canary

- The lowest-cost scientifically useful recovery target is the Wan p8 C2/C3 cohort. The
  existing immutable manifest-exact streaming cache and horizon-1 target tensors remain
  reusable, but horizons 2 and 4 are absent. A full repair requires 5,448 new target
  forwards across 681 scenes; the first gate is deliberately limited to one train scene
  plus one validation scene (8 manifest rows, 16 new forwards).
- New extractor/auditor contracts were implemented and tested without GPU execution:
  extractor SHA256 `95c574eeccdced2ca7ff88138877d4771003464cfd0b29563fd8072ee1f2c07f`;
  independent auditor SHA256 `1d0d49bc85f43cc0c823341bcbf4a6aaf34cf75bc6af371d8446b8615b8a52bc`;
  direct runner SHA256 `ce9196ffcb7a8c4d4484e574fe6b9247c157567eca358e85c912a87fcc889efc`.
  Their focused suites passed 19/19; independent race/marker/dtype/rehash/symlink
  mutations passed 9/9.
- A first freeze attempt, run tag
  `q00_wan_p8_target_h124_canary_20260830T055448Z_2075f3eb`, failed before publication
  because the mounted filesystem returned `EINVAL` for `renameat2(RENAME_NOREPLACE)`.
  Its hidden `.incomplete-*` staging directory is retained as evidence; no final bundle or
  feature output namespace was created.
- Candidate bundle `q00_wan_p8_target_h124_canary_20260830T060318Z_2075f3eb`
  (SHA256 `70e7609dfbbb7651bc599a69c1ba523f39075d24c0334df4859c4045133cb581`)
  was fully written but was never executed: fault injection showed that its prepare script
  could expose a partial `bundle.json` if the final copy failed. It is preserved as an
  obsolete freeze, and its target namespace remains absent.
- The publication path was repaired to reserve the destination by atomic `mkdir`, write and
  `fsync` a private marker, verify its SHA, and atomically no-replace publish `bundle.json`
  via a same-directory hard link. Race/fault tests passed 6/6, including a four-byte partial
  copy failure with no final marker exposed.
- Active frozen bundle:
  `reports/q00_phase2_runs/q00_wan_p8_target_h124_canary_20260830T060728Z_2075f3eb/bundle.json`;
  SHA256 `34da6ba6963dd7b68fd7e66082adfe5a6ed557f541f2732e118c8148e5d3cf0a`.
  Two independent audits found 26/26 internal source hashes, 16/16 input images,
  28/28 h1 source artifacts, 31/31 model snapshot artifacts (28,935,653,511 bytes),
  and 8/8 protected pre-images exact. The planned CLI, git commit, two-scene selection,
  external realpaths, and absent versioned output namespace also passed.

## Wan p8 canary direct-node CPU validation and resource stop

- Exact direct-node command: `ssh -i /mnt/umm/users/yinbaiqiao/id_rsa -o UserKnownHostsFile=/tmp/known_hosts -o StrictHostKeyChecking=accept-new -o BatchMode=yes root@10.119.27.217 /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -B /mnt/umm/users/yinbaiqiao/probe_spatial/reports/q00_phase2_runs/q00_wan_p8_target_h124_canary_20260830T060728Z_2075f3eb/runtime_tree/scripts/run_wan_target_h124_direct_canary.py --bundle /mnt/umm/users/yinbaiqiao/probe_spatial/reports/q00_phase2_runs/q00_wan_p8_target_h124_canary_20260830T060728Z_2075f3eb --bundle-sha256 34da6ba6963dd7b68fd7e66082adfe5a6ed557f541f2732e118c8148e5d3cf0a --repo-root /mnt/umm/users/yinbaiqiao/probe_spatial --action validate`.
- CPU-only validation ran from `2026-08-30T06:13:07Z` to `06:17:08Z` and returned
  `PASS`/rc=0. Before/after hashes were identical for all source, model, input-image, and
  protected inventories. Status SHA256 is
  `7ac662783a7c8097efc6acb3e6f998607a6ccb2d08d60dbf6f9cb0fdfaea9d50`;
  log SHA256 is `f786d64d66a55ff1fd4442baa29bd90952a2143c0e412e3a71a05935293f94a8`.
- A single GPU gate was sampled at `2026-08-30T06:17:30Z`. All eight H800 devices had
  live unrelated Ray/vagen holders. Free memory ranged from 4,015 to 23,883 MiB, below
  the frozen 75,000 MiB threshold on every device. Exact evidence is
  `reports/q00_phase2_runs/q00_wan_p8_target_h124_canary_20260830T060728Z_2075f3eb_gpu_gate_20260830T061730Z.json`.
- Stop state: `BLOCKED_RESOURCE`. The gate was not relaxed; no foreign process was
  interrupted; no feature extraction, training, evaluation, audit promotion, registry
  write, or canonical write occurred. The versioned target namespace is still absent.
  Recovery is deterministic: when one H800 has no live holder and at least 75,000 MiB
  free, run the same frozen runner with `--action extract --gpu <eligible-index>`; it will
  create an exclusive output namespace and must end with the independent audit PASS before
  any full extraction or C2/C3 clean-train canary is authorized.

## Wan p8 canary handoff to SCO H800

- The user authorized moving this bounded feature-extraction canary to the H800 SCO pool.
  No training or evaluation was authorized or launched.
- The direct bundle above was retired for extraction after an actual `/mnt/umm` publication
  test showed that its frozen `renameat2(RENAME_NOREPLACE)` primitive returned `EINVAL` on
  this filesystem. CPU validation remained valid, but the bundle was not used for GPU work.
- The publication implementation was changed only at the filesystem boundary: immutable
  JSON markers use private write/fsync/hash followed by a no-replace hard link, and scene
  directories use an atomic directory reservation plus per-file hard-link publication.
  Tensor generation, manifest selection, model execution, and target semantics were not
  changed. An independent race/fault/durability suite on the actual `/mnt/umm` device passed
  12/12, including concurrent writers, partial publication, foreign targets, and fsync order.
- The SCO execution contract fixes one H800 (`N4lS.Iq.I80.1`), image
  `registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617`, AFS mount
  `019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm`, priority `HIGHEST`, reserved quota,
  and zero retries. The image digest was unavailable and is explicitly recorded as
  `TAG_PIN_ONLY_ACCEPTED_FOR_THIS_CANARY` rather than represented as a cryptographic pin.

## Retired first SCO submission

- Bundle: `reports/q00_phase2_runs/q00_wan_p8_target_h124_canary_20260830T092823Z_2075f3eb/bundle.json`;
  SHA256 `05bb70bb5a91cc353f7fd6a4cc97f984e04c9a9f675977f18126614de0f90c7c`.
- The CPU-only receipt passed, but the first submitter version inspected the scheduler
  description before its resource fields converged. It therefore rejected the record with
  `SCO worker spec mismatch`, stopped the newly created job `pt-67y61dvq`, and published a
  terminal failure receipt (SHA256
  `c919253b7adae8d448ac39cb05dda303b00252f0e0d49dbf6e703518bf59b28e`).
- Independent reconciliation found `start_time=null`, no live worker, no output namespace,
  and no extractor/auditor artifact. The bundle and job were retired and never reused.
- A preceding submit attempt on the shared mount was also rejected before `sco create`
  because the root-squashed worker identity could not write the launcher directory. Later
  bundles pre-create only that mutable launcher directory with compatible permissions;
  immutable bundle inputs remain hash-locked.

## SCO H800 extraction attempt — storage failure

- Active bundle:
  `reports/q00_phase2_runs/q00_wan_p8_target_h124_canary_20260830T094052Z_2075f3eb/bundle.json`;
  SHA256 `a48e37ddf1bed2ae24565e2c17ffc8bfd166bda76ca726ee8c8a1f619b565856`.
- CPU validation completed `PASS/COMPLETE` before submission. Its terminal status SHA256 is
  `44746e4500a14fce8446059fd675d28040227201c9108a70d1df860f76215c6c`;
  source 28, input-image 16, model 31, and protected 8 inventories were identical before
  and after validation.
- Exactly one SCO job was submitted: `pt-xz63jp91`, display name
  `bq_q00_wh124_20260830094052`. It ran from `2026-08-30T09:47:25Z` to
  `09:53:00Z` on worker `pt-ce9d9b4a74f1435bbe961141ef41f27f-worker-0` and one
  NVIDIA H800. The frozen preflight observed 81,559 MiB total, 81,081 MiB free, zero
  compute holders, CUDA available, and exactly one visible CUDA device.
- Submission and worker lineage passed before any output claim: submit claim SHA256
  `d90214328bef7e6e537496932c7b627e6310995ebdf71386e18c000e38e89ad6`,
  submit status SHA256
  `41e8d27328bb59f949fbd9aad3b271e1c413ae5e90831ce7c1c8b61e1976e27d`,
  scheduler-description SHA256
  `718e5a7936d3a3d35c778be5f683630fb596abec1c5c9ffd98e252086afb9330`,
  and worker-binding SHA256
  `85ebca5b2a6a8ac8ac9ab1a3d11eb51d6ba118cce057191c615fa2b90d104f1f`.
- The model loaded successfully and extraction reached the first train scene. Writing
  `feature_t749_layer5.sft` then failed with
  `safetensors_rust.SafetensorError: I/O error: No space left on device (os error 28)`.
  The extract log SHA256 is
  `822e4a8cbc645fc3d1adab8184f80ed373d3b688baff687e001cecd38c73a5e2`.
- The frozen runner terminated `FAIL / CHILD_FAIL_POST_PASS`, child rc=1. Its terminal
  receipt SHA256 is
  `8189bff489511afbc940291a753c1ec44711ddb6d345ce11d1fe1954c48290d2`.
  Failure postflight independently rehashed source, input images, model snapshot, and all
  eight protected files; every before/after value remained exact.
- The independent feature auditor was not run (`audit_returncode=null`, report absent).
  There is no scene completion, shard completion, root completion, clean target inventory,
  or downstream authorization. The exclusive output namespace contains only the launch and
  shard/scene claims plus a hidden incomplete first scene: one layer-0 tensor and sidecar,
  and a partial layer-5 temporary tensor. It is retained as immutable failure evidence and
  must not be reused as a resume source.
- The scheduler terminal state is `FAILED`. Main registry/canonical files were not written;
  their eight SHA256 values remain exactly equal to the frozen pre-images.

## Current stop state

- Stop state: `BLOCKED_STORAGE`, fail closed. No retry, full extraction, clean training,
  evaluation, audit promotion, registry update, or canonical update was launched.
- `df` on the same shared QuarkFS mount reported about 31 TiB available and inode use near
  zero after the failure. The image has no quota client, and the active worker is no longer
  available for a read-only capacity check. Therefore the evidence supports an unresolved
  worker/path quota or storage-backend write limit, not a claim that the global filesystem
  is full.
- This is not evidence that the feature root is generally unwritable: the prior exact Wan
  root on the same mount contains 9,534 target files / 78,335,039,422 bytes and a 1.5 TiB
  streaming root produced by the same H800 image/specification. The unknown is the current
  write allowance or backend state for this SCO worker/identity/path, not the target tensor
  format.
- The failed namespace will remain untouched. A retry requires a new versioned bundle and
  output namespace, plus a bounded storage gate on the same SCO mount that proves a complete
  scene-sized write/fsync/readback cycle and records worker-local capacity/quota evidence.
  Only after that gate passes may the same two-scene extraction canary be resubmitted.

## Storage recovery diagnostic — PASS

- After the user confirmed that the QuarkFS quota had been cleared, one new, non-scientific
  storage-only diagnostic was run in a fresh namespace. Bundle:
  `reports/q00_phase2_runs/q00_wan_p8_h124_storage_diag_20260831T015510Z_2075f3eb/bundle.json`;
  SHA256 `03a2ec8a72a5e570a09f20f760bef18a5a3cdc8b12ad5014527766a83f62b0ce`.
- Exactly one H800 SCO job, `pt-9glwnaw8`, ran from `2026-08-31T01:57:18Z` to
  `01:57:28Z` on `pt-1c8c5d639d2944dab1fff88b666aec60-worker-0`, using the same
  image, mount, and one-H800 worker specification as the failed feature canary.
- The diagnostic wrote six real FP16 safetensors with the extraction tensor shape
  `[12,30,52,1536]`, then fsynced, SHA-checked, opened, shape/dtype/finite-checked, and
  read back every file. Payload total: `345,047,568` bytes. Terminal status is `PASS`,
  SHA256 `46f20a4eda2b81bf50284b0f7964b642ddde3f3a59ecd13b6ec16b843b27d984`.
- Source, protected registry/canonical tables, and the diagnostic's input contract were
  unchanged before/after. It did not load a model, extract target features, train, evaluate,
  package, promote, or modify canonical data. The retained diagnostic output is evidence only.

## Fresh two-scene h2/h4 extraction canary — frozen preflight

- Fresh immutable bundle:
  `reports/q00_phase2_runs/q00_wan_p8_target_h124_canary_20260831T020035Z_2075f3eb/bundle.json`;
  SHA256 `e71dbcbd5d9e3a941fcb3ac5c65b8f188ffb40aaae69f84bd22587fcb5aaaac2`.
- Before CPU validation, 31/31 frozen source hashes, the eight protected pre-images, the
  selected two-scene/eight-row h124 manifest contract, 16 selected input images, and 28 h1
  source artifacts matched exactly. The new target output root was absent.
- A single direct-node CPU-only validator started at `2026-08-31T02:04:21Z`. It is rehashing
  the frozen 31-file model snapshot and all inputs before any SCO submission. It does not use
  CUDA, run target forwards, or write canonical data. Submission remains prohibited until its
  terminal `PASS/COMPLETE` receipt is present.

## Fresh two-scene h2/h4 extraction canary — PASS

- The CPU-only validator reached `PASS / COMPLETE` at `2026-08-31T02:07:55Z` (status SHA256
  `663a3bdbd52affffbb2fcffdb3698623b2f80135f6a90297c352fa9f714fe2a4`; log SHA256
  `f786d64d66a55ff1fd4442baa29bd90952a2143c0e412e3a71a05935293f94a8`). It confirmed
  source h1 `28 / 230,058,822 B`, input images `16 / 971,045 B`, model snapshot
  `31 / 28,935,653,511 B`, and protected files `8 / 8` unchanged before/after.
- A first local submitter invocation containing an unsupported `--repo-root` argument failed
  in argparse before creating any claim or SCO job. A second local invocation then correctly
  failed on the root-squashed, root-owned mutable launcher receipt directory. These are
  retained fail-closed operational receipts; neither created an output writer.
- The exact frozen submitter was then run on the specified direct node under the same
  root-squashed identity as the CPU validator. It created exactly one job, `pt-sz84eh0f`, with
  immutable submit claim SHA256
  `ef709c9fcd760334a7204c79f2ed302f07a0d2bdfc7fe168badde38d537237f2`, scheduler-description
  SHA256 `e08abb47d92f341d27cfc195fb636a3a82d9e9f9a0ff675b80258315c84868ef`, and worker-binding
  SHA256 `1ac2a13d596ff441f6da4046c78d5f726c2a21cccbb9ec557554a069dadd63c0`.
- `pt-sz84eh0f` ran one frozen H800 worker from `2026-08-31T02:10:51Z` to `02:21:15Z` on
  `pt-006c0b77ee1945b8886e4515709659d9-worker-0`, with the exact bundle, image, mount,
  one-GPU specification, CUDA visibility, manifest, source-h1 root, and no-retry contract.
- The isolated target root completed both selected scenes (`00777c41d4`, `00a231a370`): eight
  h124 rows, horizons 2 and 4, six layers `[0,5,10,15,25,29]`, and 16 new Wan forwards. All
  12 published target tensors are finite FP16 `[12,30,52,1536]`; the final output inventory
  contains 28 bound artifacts including tensors, target indices, and sidecars.
- Frozen runner terminal receipt: `PASS / COMPLETE`, rc=0, audit rc=0, SHA256
  `6c355245460e3eee48ad7758bc77aa030219dde51efeb0f8bfeb2f8e9d6c19ba`.
  Independent auditor result: `PASS / ok=true`, SHA256
  `05bb6cfa15a626b742ee1f30f328dd22086af646adc4be99433a7de3421adf4b`.
  It revalidated SCO job lineage, full manifest/selection identity, h1 source lineage,
  model and input-image inventories, all produced tensors/sidecars, and the eight protected
  registry/canonical hashes before and after execution.
- This result is only the bounded h2/h4 target-feature extraction canary. It does not train a
  probe, run eval, package a result, promote anything, or change `experiments/registry.csv` or
  any main canonical table. A full 681-scene h2/h4 target extraction remains a separately
  frozen next wave, followed by a clean C2/C3 train+eval canary if and only if its own gates
  pass.

## Display-first loose result delivery — 2026-08-31

- The execution priority was explicitly changed to a 1096-row display table.  No clean
  retraining, full feature extraction, canonical remediation, registry update, or canonical
  promotion was launched under this section.
- CPU-only generation produced `results/analysis/q00_1096_loose/all_results_1096_loose.csv`
  and its Parquet companion.  It retains all exact-bound canonical, raw historical,
  provisional, and legacy candidates; known feature/window and manifest issues are warnings,
  not visibility gates.  The initial table has 1,096 frozen parents and 833 finite selected
  results (642 T0 canonical, 31 T1 raw historical, 160 T2 provisional, 263 T5 no numeric).
- A new isolated runner, `scripts/run_q00_1096_legacy_as_is_loose.py` (SHA256
  `0d237834ac12ea39bc9f400689b303dc3d5478467444e1c2e8c7f66f7641ad9c`), was used only for
  `LEGACY_AS_IS_LOOSE` evaluation.  It uses the exact inventory checkpoint and historical
  manifest/cache fields, forbids sibling-checkpoint fallback, writes only below
  `results/analysis/q00_1096_loose/eval_runs/`, and never promotes or writes registry/canonical
  tables.
- First direct-node batch tag: `direct_loose_canary_20260831T064626Z`; five exact
  checkpoint-available missing parents ran independently on H800 GPUs 0, 1, 2, 4, and 5.
  The outputs were initially created as `root:root` by the authorized root SSH session on the
  QuarkFS client; after all five terminal receipts existed and no runner remained, ownership of
  this exact output tree only was changed to workspace UID/GID `20325:20325` so the workspace
  can ingest the receipts.  No content was modified.
- PASS: `videomae_vjepa_layer_sweep_streaming_mask_videomae_multi_layer_l0_p8_hall_s42`
  produced `mean_loss_identity=5.4966277806223305`, `n=276`, receipt SHA256
  `9eb42f166bd020d20a7ec544465593a1500f7a7a07bcafa8e2bd9021ae56c095`.
  PASS: `videomae_vjepa_layer_sweep_streaming_depth_vjepa2_multi_layer_l0_p24_hall_s42`
  produced `mean_loss_depth=0.5426851573089758`, `n=276`, receipt SHA256
  `34e90bf110e5e5c2ddb6dce81e86c2a08f43605aa8e480103f3f35d421c846fa`.
  Both retained their known mismatch warnings and have `scientifically_comparable=false`.
- FAIL (retained as terminal loose-eval evidence, no fallback): LingBot view-consistency and
  MiniMax-H3 streaming-depth each encountered an unreadable historical generic streaming
  feature path; Wan path-integration reached evaluator loading but the current historical
  configuration did not provide checkpoint keys `probe.constant_context` and
  `probe.constant_action`.  Their receipt SHA256 values are respectively
  `4028e495053cf7444364c7b47eb3a8bc18a2fb9ef8a6e71000b6a258cbae95d0`,
  `1717c791c6eeaa2903ef5ca15f5a95b7b7a88849027e19e30d0109dbed2d907a`, and
  `a5f6a4abc21e48de547d16b1425768f67fc3ab93996bc3304bbfc4cdcd2e89f4`.
- Every receipt records unchanged checkpoint metadata and exact before/after protected hashes;
  no registry or main canonical file changed.  The loose table is refreshed only from the two
  PASS receipts; failed receipts remain under `eval_runs/` and do not create numeric values.

## Loose evaluation batch 2 — V-JEPA2 layer-0 mask expansion

- The first batch established that the historical V-JEPA2 layer-0 generic streaming cache is
  readable in loose mode.  The next bounded expansion therefore evaluated only the remaining
  layer-0 `streaming_mask` prefixes, rather than launching a broad sweep.
- Direct-node tag: `direct_loose_vjepa_l0_mask_20260831T071805Z`; exact historical checkpoints
  and h124 manifests were used for p12, p16, and p24 independently.  The updated runner SHA256
  is `4e0a84379e19ac99b1430cab2af1b156c2b7e2f8f12ee4446cc0f130c94425eb`; it hands the completed
  output tree back to workspace UID/GID `20325:20325` after the terminal receipt, preventing the
  root-owned QuarkFS visibility problem from the first batch.
- All three completed `PASS`, each with `n=276`, retained mismatch warnings, and
  `scientifically_comparable=false`:
  - p12: `mean_loss_identity=6.734466847734175`, receipt SHA256
    `593810b0ce380786d29421358cdb6c29c9086fee4cde1939c7ac7526fcf70467`.
  - p16: `mean_loss_identity=8.611402434499368`, receipt SHA256
    `e2b20dc0854a10b4327ceb0f03680a2c59dd084e244724b2c82b2f0d8e582cee`.
  - p24: `mean_loss_identity=10.462539253027543`, receipt SHA256
    `443fdbd2e78245d17383a5121412bfe1273e90dca65bf54cb88213b28d799dca`.
- All three receipts attest that their exact checkpoint metadata and all eight protected
  registry/canonical hashes were unchanged before/after.  They are display-only T4 candidates;
  no promotion or canonical mutation occurred.

## Loose evaluation queue refresh — 2026-08-31

- The current-missing checkpoint inventory was rebuilt from the refreshed loose table.  It has
  203 checkpoint-available parents: 178 `READY_LEGACY_AS_IS_LOOSE`, 22 with a missing original
  manifest, and three `HOLD_PREVIOUS_LOOSE_EVAL_FAILURE`.  The held rows are the prior LingBot,
  MiniMax-H3, and Wan canaries; they will not be silently retried with the same historical
  checkpoint/cache combination.
- A read-only direct-node preflight for one new V-JEPA2 layer-5 loose-eval canary used the
  authorized SSH command at 07:40 UTC and returned `No route to host`.  No GPU process, eval
  run, SCO job, output directory, registry update, or canonical mutation was created.  The
  queue remains deterministic and can resume from a new, unique run tag after node reachability
  is restored.

## Loose evaluation batch 3 — replacement H800 node, V-JEPA2 layer 5

- Replacement direct node `10.119.22.92` passed the read-only preflight at 08:19 UTC: eight H800
  GPUs, exact shared repository/interpreter/checkpoint inventory, and no pre-existing loose-eval
  writer.  GPU 1 first ran the exact layer-5 streaming-depth p24 canary under tag
  `direct_loose_vjepa_l5_depth_20260831T075500Z` and returned `PASS`,
  `mean_loss_depth=0.48257098271363025`, `n=276`; terminal receipt SHA256
  `67395d45820338c115e44dd5c1703edd0803ac9af8cc4a882cca18a91fd64112`.
- That cache-layer canary authorized only a bounded three-lane mask expansion on GPUs 1/2/3,
  tag `direct_loose_vjepa_l5_mask_20260831T082900Z`.  All used their exact inventory
  checkpoint, historical h124 manifest, and isolated output directory; all have unchanged
  checkpoint and protected-file before/after evidence:
  - p12: `mean_loss_identity=5.637044731473577`, `n=276`, receipt SHA256
    `021cc1250cdd3e735e777e896809c34f282efecc79061948377799b3696afd06`.
  - p16: `mean_loss_identity=6.104188323452853`, `n=276`, receipt SHA256
    `66056ee530e08f63b4e9c3ebca59e091d4b625eb11b0c24e68cdfbcaf9c7fe4a`.
  - p24: `mean_loss_identity=7.834646143127179`, `n=276`, receipt SHA256
    `10a6b1d39b95fa48893eee3052519bf3ffa9f2779605d7640b070a6f5c743d7e`.
- These four values remain T4 display-only loose evaluations with the known window/manifest
  mismatch warnings.  No training, feature reconstruction, registry write, canonical write, or
  promotion occurred.

## Loose evaluation batch 4 — active recovery point

- After the table refresh, the exact V-JEPA2 layer-11 streaming-depth p24 canary was launched on
  GPU 1 of `10.119.22.92` under unique tag `direct_loose_vjepa_l11_depth_20260831T084500Z`.
  Its shared output namespace was atomically created at 08:44 UTC.  Subsequent independent SSH
  diagnostics encountered `Connection timed out` and then `No route to host`, while the original
  SSH/evaluator session remained open without a terminal receipt.
- No duplicate submission, retry, fallback, promotion, training, or canonical write was made.
  This unique run must be reconciled from its terminal receipt after node reachability returns;
  it must not be relaunched into the same scientific/output identity while its current writer
  status is unknown.

## Loose evaluation batches 5–7 — replacement H800 `10.119.22.88`

- The user supplied replacement direct node `10.119.22.88`.  Read-only preflight found eight
  H800 GPUs, the shared repository and exact `probe-spatial` Python runtime.  The node had
  unrelated resident jobs using about 3.4 GiB per GPU; every selected lane still had at least
  75 GiB free before launch.  Each loose run used a distinct tag and an initially absent output
  namespace.  No SCO job was submitted.
- Batch 5 tag `direct_loose_vjepa_l17_depth_20260831T090300Z` evaluated the exact historical
  V-JEPA2 layer-17 streaming-depth p24 checkpoint/manifest on GPU 0.  It returned `PASS`,
  `mean_loss_depth=0.4081234490849834`, `n=276`, receipt SHA256
  `338f4b76e772216d510cbd634b389a70ad05acea4407ca8ca60a929dfc3a614d`.
- Batch 6 tag `direct_loose_vjepa_l17_mask_20260831T091800Z` evaluated three exact historical
  V-JEPA2 layer-17 streaming-mask checkpoints in parallel, one physical GPU per parent:
  - p12: `mean_loss_identity=4.451103322747825`, `n=276`, receipt SHA256
    `19817e12603fb6b5cecf32ea34367be8e1d43850535d4529b5d82eead227e3fe`.
  - p16: `mean_loss_identity=5.188931540950485`, `n=276`, receipt SHA256
    `7ad50661db520e03542c029dd3e3e4df8c39fcd9625c1b623c3eba8212048e9d`.
  - p24: `mean_loss_identity=6.352830769575161`, `n=276`, receipt SHA256
    `248a0d29e0c9980d0bd1c3fcddc9e8749154fb727b7bd641f683d189c3efe58c`.
- Batch 7 tag `direct_loose_vjepa_l11_mask_20260831T100000Z` evaluated only the two
  `READY_LEGACY_AS_IS_LOOSE`, strict-final V-JEPA2 layer-11 streaming-mask parents.  It did
  not touch the separate, unreachable-node layer-11 depth p24 recovery point:
  - p16: `mean_loss_identity=4.731891154379085`, `n=276`, receipt SHA256
    `f3dd6f34887c91313dcd4f2b14e6e81647966305c67f7d4eb54525e1377bf7ed`.
  - p24: `mean_loss_identity=5.56270208587681`, `n=276`, receipt SHA256
    `1c0b8b00e111a689baf0231f179806a699f162205472c1923860f9992758e214`.
- All six terminal receipts record `LEGACY_AS_IS_LOOSE`, finite metrics, their exact historical
  checkpoint and h124 manifest, known mismatch warnings, and identical before/after hashes for
  the registry plus all seven main canonical tables.  They are display-only T4 candidates:
  no training, feature reconstruction, promotion, registry mutation, or canonical mutation
  occurred.
- After every completed batch the 1,096-row loose table was rebuilt.  To keep this CPU-only
  refresh bounded as receipts accumulate, its builder now indexes terminal loose receipts once
  per refresh instead of recursively rereading the same evaluation tree for every parent.  This
  is a performance-only change; candidate eligibility and source precedence are unchanged.
  Current builder SHA256 is `aa6bc489f96a83dd1f4c2c06ef245cd468239ae485c15993c882b7de9cdb3241`.

## Loose evaluation batch 8 — V-JEPA2 layer 23 compact expansion

- Tag `direct_loose_vjepa_l23_20260831T102000Z` used three independent, preflight-clean H800
  lanes on `10.119.22.88` for the remaining strict-final layer-23 parents that were still
  missing a finite result.  Each used the exact inventory checkpoint and historical manifest;
  all three completed `PASS` with `n=276` and matching protected pre/post hashes:
  - streaming-depth p24: `mean_loss_depth=0.4067929792015449`, receipt SHA256
    `2fc26ee58b9acffe98aa453c7e90065e2ae6da75eabe7e08caf2e11ba5b14943`.
  - streaming-mask p16: `mean_loss_identity=4.71299280917299`, receipt SHA256
    `b0d0119d33a4065602212d3f85c3a64c62ad620c38211f60fd04259783e0f057`.
  - streaming-mask p24: `mean_loss_identity=5.998464121334795`, receipt SHA256
    `90f3537cbaf6fde38caab4488a06d24cb3c843760a2f86cdf4df0becc093cea7`.
- These are T4 `LEGACY_AS_IS_LOOSE` display values with the known label/feature-window and
  manifest provenance warnings retained.  No source was promoted and no protected table was
  modified.  The post-batch loose result table contains 851 finite selected results and 245
  explicit no-numeric rows.

## Loose evaluation batch 9 — readable historical non-final checkpoints

- Tag `direct_loose_vjepa_nonfinal_mask_20260831T103000Z` ran the two remaining V-JEPA2
  streaming-mask p12 parents that share the already-proven readable historical cache path.
  Both checkpoints stop at epoch 46/step 115056, so their non-final status remains source
  metadata and a warning; it was not converted into a canonical claim.  Both completed
  `LEGACY_AS_IS_LOOSE` postflight with finite metrics, `n=276`, and unchanged protected hashes:
  - layer 11: `mean_loss_identity=4.400224043623261`, receipt SHA256
    `9bd86962b7e6e41a1f792f7d1d96148890e051fad9b3ff774e889c32b2079a60`.
  - layer 23: `mean_loss_identity=4.255215444858523`, receipt SHA256
    `053644719f82f5feb9dee4771f6d49350aabda8079ad57e56e86761d5eead94b`.
- The table was immediately rebuilt.  It now has 853 finite selected results and 243 explicit
  no-numeric parents.  As throughout this loose-evaluation line, no training, feature
  reconstruction, promotion, registry write, or canonical-table write occurred.

## Loose evaluation batch 10 — Wan C1 historical-runtime compatibility retry

- The exact historical Wan C1 action-dynamics checkpoint
  `wan_cog_layer_sweep_action_dynamics_wan_run_dir_discovered_l0_p12_hall_s42` first reached
  the evaluator but failed strict checkpoint loading because the current uncommitted probe class
  has two newly registered, unused full-conditioning parameters absent from the historical
  checkpoint: `probe.constant_context` and `probe.constant_action`.  The initial terminal
  receipt remains retained at SHA256
  `2c9181ba152e25ddcd5be0e1047276cc8addef52ff3434ff3d5ef64f3508d0b6`.
- A CPU-only guard verified the exact missing-key set, no unexpected keys, full conditioning,
  strict load after removing only those two runtime-only parameters in memory, and unchanged
  checkpoint bytes/metadata.  This is not `strict=False`, does not synthesize weights, and is
  enabled only for `LEGACY_AS_IS_LOOSE` Wan C1/C2 rows with that exact historical compatibility
  signature.
- The uniquely tagged retry `direct_loose_wan_c1_compat_20260831T110000Z` then ran on the
  supplied H800 node `10.119.22.88` against the exact inventory checkpoint and historical
  manifest.  It returned `PASS`, `global_R@1=0.05027933046221733`, `n=179`; terminal receipt
  SHA256 `f1956944e82bd431d693d4ff0e7da8fa2b1e5eaa3a685e3aa4bd306d31ac7f91` links the prior
  failed receipt and records the scoped compatibility action.  Registry and all main canonical
  tables remained byte-identical before/after.
- The display table was rebuilt after this receipt.  It now has 854 finite selected results and
  242 explicit no-numeric parents.  The loose candidate preserves `n`, summary hash/path, and
  the compatibility metadata, so this value is visibly T4/display-only rather than silently
  upgraded to a canonical or scientifically-comparable result.  No training, feature
  reconstruction, promotion, registry write, or canonical-table write occurred.

## Loose evaluation batch 11 — Wan C2 compatibility retry

- The exact historical Wan C2 path-integration checkpoint
  `wan_cog_layer_sweep_path_integration_wan_run_dir_discovered_l0_p12_hall_s42` had the same
  recorded two-parameter full-conditioning compatibility signature as Wan C1.  Its original
  failed loose receipt was retained and revalidated before this single retry; no failure from a
  different checkpoint, feature root, or metric contract was eligible.
- After a read-only GPU/namespace/process preflight on `10.119.22.88`, tag
  `direct_loose_wan_c2_compat_20260831T112500Z` ran on GPU 0 using the exact checkpoint,
  historical p12 h124 manifest, streaming and target roots, and `action_horizons=[1,2,4]`.
  It completed `PASS`, `global_R@1=0.03171641752123833`, `n=276`; terminal receipt SHA256
  `ac0b1208ae7ad45b2a6ed56551c5397867b07dcae4eb68afa6f52094c9eae255`, summary SHA256
  `809b0ca7c8179c7efeb17b124072ebe229cc26dfc0d0649bb8615b3a262c9654`.
- The receipt links the original failed attempt, records
  `removed_runtime_only_legacy_parameters` for precisely the two unused constants, keeps strict
  checkpoint loading, and proves checkpoint metadata plus registry/canonical protected hashes
  were unchanged before/after.  This is a T4 display-only result with explicit mismatch and
  compatibility warnings, not a canonical result.
- The table was refreshed to 855 finite selected results and 241 explicit no-numeric parents.
  No training, feature reconstruction, promotion, registry write, or canonical-table write
  occurred.

## Loose evaluation batch 12 — Wan C3 counterfactual canary (fail closed)

- Tag `direct_loose_wan_c3_p8_20260831T113500Z` evaluated the exact historical Wan C3 p8/layer0
  checkpoint on GPU 1 of `10.119.22.88`.  It loaded the historical manifest and checkpoint but
  stopped at the first dataloader fetch because the historical prefix-alias root lacks the exact
  requested layer-0 streaming tensor.  The missing path is recorded in the evaluator log;
  no fallback feature root or sibling configuration was used.
- Terminal receipt SHA256 `46dd5bb4dae848ca46af1fff7c0fbec0c95163d186014e7b1d873844ad324393`
  records `FAIL`, return code 1, the exact missing-feature error, and unchanged protected hashes.
  This is a storage/feature availability failure rather than the Wan C1/C2 historical
  state-dict compatibility case, so it is not retried and does not authorize C3 expansion.

## Loose evaluation batch 13 — Wan C1/C2 bounded prefix/layer wave

- After the independent C1/C2 canaries passed, four separate parent configurations were launched
  on GPUs 0–3 of `10.119.22.88` under one shared batch tag but distinct per-parent namespaces:
  C1 l0/p16, C2 l0/p16, C1 l10/p12, and C2 l10/p12.  Each used the exact inventory checkpoint,
  historical h124 manifest, its declared feature roots, and no shared run directory.
- All four completed `PASS` with strict in-memory historical-runtime compatibility only for the
  exact two unused Wan constants; each receipt proves unchanged checkpoint metadata and unchanged
  registry plus all seven main canonical tables:
  - C1 l0/p16: `global_R@1=0.054347824305295944`, `n=276`, receipt
    `52fde9b796ec639612456899833aabad138f81e22768182f8579531c9080ab49`.
  - C2 l0/p16: `global_R@1=0.014492753893136978`, `n=276`, receipt
    `fe27b5ceca14511094da27bfdf4833ba17ba5cb0b732db656b70cbc744ceb38a`.
  - C1 l10/p12: `global_R@1=0.08379888534545898`, `n=179`, receipt
    `e240f7c3965d7fb2600315d982df4918c396cc7fa47dfb47cea420c7551b64b8`.
  - C2 l10/p12: `global_R@1=0.033582091331481934`, `n=276`, receipt
    `5f6507fc86333f456854a14fe493ff2b36e8fdfc0866ec1a291287d6343c6b3a`.
- All four are T4 `LEGACY_AS_IS_LOOSE` display-only values with the known mismatch and
  compatibility warnings preserved.  The table was refreshed to 859 finite selected results and
  237 explicit no-numeric parents; no training, feature reconstruction, promotion, registry
  write, or canonical-table write occurred.

## Loose evaluation batch 14 — cross-model readability screen and LingBot C1 retry

- A three-parent screen used exact strict-final checkpoint/config/manifest records for LingBot C1,
  LingBot B2, and MiniMax A2.  All three produced terminal `FAIL` receipts with unchanged
  protected hashes and are held rather than retried with substitutes:
  - MiniMax A2 p24 lacks its exact historical `feature_layer10.sft` under `FEAT_STREAMING`;
    receipt `b53f804bab5a2f85c452e3869099af5f50f1a2dc3dd7ad6265eb48b76258effc`.
  - LingBot B2 has a zero-byte historical Hydra config; receipt
    `4acad802062d73e47db8e87b016b3c64e794552a113450d1ea6f2630dd8fa431`.
  - LingBot C1 first exposed the exact two-constant historical architecture difference.
- Independent source/config/log review confirmed the LingBot C1 transition is the same narrowly
  guarded `full`-conditioning case as Wan: only `probe.constant_context` and
  `probe.constant_action` are missing, no key is unexpected, and neither is read in the full
  `ActionDynamicsProbe` forward branch.  The loose-only guard was generalized from a model-name
  check to the same exact action/path proof.  Its one retry
  `direct_loose_lingbot_c1_compat_20260831T122500Z` passed strict loading but then failed at the
  exact missing LingBot streaming feature; receipt
  `4ccc94fc1a3aa9fa15ff059e3e14162759748e8ef3d3fb3299f86b034baceb6b`.
  That is a feature-readability blocker, so LingBot action/path is not expanded.

## Loose evaluation batch 15 — Wan C1/C2 eight-lane coverage wave

- Eight independent Wan C1/C2 parents, spanning p8/p12/p16/p24 and layers 0/10/15/25, ran one
  per H800 on `10.119.22.88`.  All were `PASS`, retained strict in-memory two-constant
  compatibility evidence, and proved unchanged checkpoint metadata plus protected registry/main
  canonical hashes.  All are T4 display-only values:
  - C1 l0/p24: `0.07971014827489853`, `n=276`, `de107afaae8b0f5490420b6a4f4b749619c9b847ebd2e88a464881328381d707`.
  - C1 l10/p16: `0.08695652335882187`, `n=276`, `6120b570c2ccac66fa3828eb7cd0693de9774e0e1bc81d46f95263c9f095c635`.
  - C1 l15/p12: `0.06145251542329788`, `n=179`, `af7d9f3cdf82e5b1b068172150e68b57b7050445b9b5ffc54181bbdb53100fa8`.
  - C1 l25/p12: `0.07262569665908813`, `n=179`, `d1113d850ef7a696b76933ee81e11aa07cf46f82abc52d9c17ac9a9f1b772773`.
  - C2 l0/p8: `0.009661835618317127`, `n=276`, `087c1dbdf26a93acb0909a76ed6b51319a43ba0dd3d6293c3faafedde1be1239`.
  - C2 l10/p16: `0.021739130839705467`, `n=276`, `54f548c91d40b0f73b57b6310588b68c185b38eda02cbdf7193a0f96817aa62c`.
  - C2 l15/p12: `0.03171641752123833`, `n=276`, `f6a11952830abb7b7edfce8fb32b1aa871788aacebcc887e96a8bcc121e8ea58`.
  - C2 l25/p12: `0.033582091331481934`, `n=276`, `783cc107c3bfe7c88b994e8d18aa59e678d91be15201d41558a2602ff05a5e2f`.
- The table refresh reached 867 finite selected results and 229 explicit no-numeric parents.  No
  training, feature reconstruction, promotion, registry write, or canonical-table write occurred.

## Loose evaluation batch 16 — Wan C1/C2 layer-5/layer-29 eight-lane coverage wave

- An initial outer-shell launch attempt at `2026-08-31T11:46:38Z` failed before any evaluator
  process or output receipt was created because its nested remote quoting was malformed.  Its
  empty `direct_loose_wan_c1c2_wave3_20260831T114638Z` namespace is retained and was not reused.
- The retry tag `direct_loose_wan_c1c2_wave3_retry_20260831T115500Z` used the supplied direct
  H800 host `10.119.22.88`, one exact parent per physical GPU.  The node had unrelated 24 GiB
  processes on every card but each loose evaluator stayed below 4 GiB; no OOM or interference was
  observed.  All eight evaluators reached terminal `PASS`, with exact historical checkpoints and
  manifests, finite `global_R@1`, and unchanged checkpoint/protected registry/canonical hashes:
  - C1 l5/p12: `0.07821229100227356`, `n=179`.
  - C1 l5/p16: `0.07608695328235626`, `n=276`.
  - C1 l5/p24: `0.12318840622901917`, `n=276`.
  - C1 l29/p12: `0.07821229100227356`, `n=179`.
  - C1 l29/p16: `0.06884057819843292`, `n=276`.
  - C1 l29/p24: `0.09057971090078354`, `n=276`.
  - C2 l5/p8: `0.02415458858013153`, `n=276`.
  - C2 l29/p8: `0.01690821163356304`, `n=276`.
- These receipts retain the narrowly guarded strict historical-runtime compatibility record for
  the two unused full-conditioning constants.  They are T4 `LEGACY_AS_IS_LOOSE` display values,
  not canonical/scientifically-comparable claims.  The table was rebuilt to 875 finite selected
  results and 221 explicit no-numeric parents; no training, feature reconstruction, promotion,
  registry write, or canonical-table write occurred.

## Loose evaluation batch 17 — Wan C1/C2 p16/p24 and p8 completion wave

- Tag `direct_loose_wan_c1c2_wave4_20260831T120200Z` used all eight H800 lanes on
  `10.119.22.88` after a read-only duplicate-writer and input preflight.  Every evaluator used
  its exact historical checkpoint/manifest and emitted a terminal `PASS`; checkpoint snapshots
  and the protected registry/main canonical hashes were unchanged before and after each parent.
- New display-only T4 values (`global_R@1`) are:
  - C1 l15/p16: `0.06159420311450958`, n=276.
  - C1 l25/p16: `0.054347824305295944`, n=276.
  - C1 l10/p24: `0.09420289844274521`, n=276.
  - C1 l15/p24: `0.09420289844274521`, n=276.
  - C1 l25/p24: `0.09057971090078354`, n=276.
  - C2 l10/p8: `0.01690821163356304`, n=276.
  - C2 l15/p8: `0.026570048183202744`, n=276.
  - C2 l25/p8: `0.00845410581678152`, n=276.
- All retain the exact two-key, full-conditioning, strict in-memory legacy compatibility record
  and the known label/feature-mismatch warning.  The 1,096-row loose table was refreshed to 883
  finite selected results and 213 explicit no-numeric parents.  No training, reconstruction,
  promotion, registry write, or canonical-table write occurred.

## Loose evaluation batch 18 — Wan C2 p12/p16/p24 coverage wave

- Tag `direct_loose_wan_c2_wave5_20260831T121100Z` evaluated eight remaining exact Wan
  path-integration parents in parallel on `10.119.22.88`.  All eight reached terminal `PASS`
  with `n=276`, exact historical checkpoint/manifest provenance, finite primary `global_R@1`, and
  unchanged checkpoint/protected registry/canonical hashes:
  - l5/p12 `0.026119403541088104`; l29/p12 `0.03731343150138855`.
  - l5/p16 `0.019323671236634254`; l15/p16 `0.028985507786273956`;
    l25/p16 `0.02053140103816986`; l29/p16 `0.02053140103816986`.
  - l0/p24 `0.025362318381667137`; l5/p24 `0.03381642326712608`.
- These remain T4 `LEGACY_AS_IS_LOOSE` results with explicit known-mismatch and strict
  runtime-compatibility metadata.  The loose table refreshed to 891 finite selected results and
  205 explicit no-numeric parents.  No training, feature reconstruction, promotion, registry
  write, or canonical-table write occurred.

## Loose evaluation batch 19 — Wan C2 p24 completion

- Tag `direct_loose_wan_c2_wave6_20260831T121900Z` completed the two final ready Wan
  path-integration p24 parents on independent H800 lanes of `10.119.22.88`:
  - l10/p24: `global_R@1=0.04106280207633972`, n=276.
  - l15/p24: `global_R@1=0.05797101557254791`, n=276.
- Both are terminal `PASS` receipts with exact historical checkpoint/manifest, finite metric,
  strict runtime-only two-key compatibility evidence, and unchanged checkpoint plus protected
  registry/main canonical hashes.  This completes the currently preflight-readable Wan C1/C2
  grid; values remain T4/display-only with known mismatch warnings.  The 1,096-row loose table
  was refreshed to 893 finite selected results and 203 explicit no-numeric parents.  No training,
  feature reconstruction, promotion, registry write, or canonical-table write occurred.

## Loose evaluation batch 20 — cross-model legacy readability screen (fail closed)

- Tag `direct_loose_crossmodel_wave1_20260831T122800Z` launched eight exact, current-missing
  parent configurations to identify reusable historical input paths.  No row was substituted or
  retried.  All eight returned terminal `FAIL` before producing a finite metric; protected hashes
  remained unchanged.
- Two LingBot parents had zero-byte `.hydra/config.yaml` files and were rejected by the loose
  runner before evaluator launch: streaming-mask l10/p12 and ego-belief-v2 l10/p12.
- The other six reached their exact evaluator/dataloader paths and then rejected the exact first
  missing historical feature; they do not justify a broad retry of their model family:
  - LingBot streaming-depth l10/p12: missing AOSS `feature_layer10.sft` at prefix 11.
  - LingBot view-consistency l10/p16: missing AOSS `feature_layer10.sft` at prefix 15.
  - MiniMax-H3 streaming-depth l40/p12, streaming-mask l10/p12, and view-consistency l20/p24:
    each missing its exact AOSS `feature_layer*.sft` at the requested prefix.
  - Wan counterfactual l0/p12: missing exact local prefix-alias
    `feature_t749_layer0.sft`.
- A CPU-only inventory correction now requires non-empty historical `run_manifest.json`,
  `.hydra/overrides.yaml`, and `.hydra/config.yaml`, preventing the empty-config failure class
  from being re-queued.  No evaluator, checkpoint, feature root, registry, or canonical table
  was modified.

## Loose evaluation batch 21 — Wan C3 p12 readable-layer completion

- A CPU-only first-feature screen on `10.119.22.88` checked all 148 checkpoint-backed
  no-numeric parents against their exact historical first validation artifact.  Report:
  `eval_runs/first_feature_screen_20260831T123500Z.json`, SHA256
  `1dd1378f0622287e53f907604a6b4ad591e36ce52e4c46506edafd7b91ab323a`.
  It found 121 definitely missing feature artifacts, 23 legacy roots requiring a separate path
  adapter, and four readable Wan counterfactual p12 layers.
- Tag `direct_loose_wan_c3_p12_wave2_20260831T124100Z` evaluated those four exact parents in
  parallel on H800 GPUs 0--3 of `10.119.22.88`.  All reached terminal `PASS`, processed 276
  validation samples, retained their historical checkpoint/manifest identity, and left the
  checkpoint plus all protected registry/canonical hashes unchanged:
  - l10/p12: `intervention_validity=0.4099099099099099`.
  - l15/p12: `intervention_validity=0.39864864864864863`.
  - l25/p12: `intervention_validity=0.3963963963963964`.
  - l29/p12: `intervention_validity=0.39414414414414417`.
- All four are display-only T4 `LEGACY_AS_IS_LOOSE` results with the recorded known
  label/feature-window mismatch; no training, feature reconstruction, promotion, registry
  write, or canonical-table write occurred.  The loose table is now 897 finite selected results
  and 199 explicit no-numeric parents.
