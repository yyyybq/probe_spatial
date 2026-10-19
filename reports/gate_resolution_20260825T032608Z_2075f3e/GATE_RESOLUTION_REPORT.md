# Gate Resolution Report

Gate tag: `gate_resolution_20260825T032608Z_2075f3e`  
Repository: `/mnt/umm/users/yinbaiqiao/probe_spatial`  
Base commit: `2075f3eb270cb0e8907c3927b6b079c64965b75a` (`main`, dirty worktree)  
Final verdict: **BLOCKED — formal validity experiments were not launched.**

## Outcome

The current source state was frozen and the runner and scheduler gaps were audited. The existing-cache DINO p8 path does not depend on either missing historical runner interface, so the runner is not the immediate blocker for the requested strict-B2 gate. The immediate blocker is scheduler identity ownership: the only local SCO profile is labeled `yinbaiqiao`, but the scheduler API assigns jobs created with that credential to `guchenyang`. Using that credential would violate the explicit prohibition on using another user's token or job.

Because gate 4 (own 1-GPU allocation) did not pass, gates 5–8 were intentionally not run. There is no compute-node CUDA result, no compute-node AOSS read result, no real DataLoader result, and no real strict-B2 smoke result in this gate tag. No formal `RUN_TAG` was minted and no training job was started.

## Gate status

| Order | Gate | Status | Evidence |
|---:|---|---|---|
| 1 | Snapshot current worktree | **PASS** | Source-only snapshot and manifests described below |
| 2 | Runner version audit | **PASS_FOR_EXISTING_CACHE_CONSUMPTION** | `RUNNER_VERSION_AUDIT.md` |
| 3 | Scheduler/403 audit | **COMPLETE_WITH_BLOCKER** | `SCHEDULER_AUDIT.md` |
| 4 | Obtain own 1-GPU allocation | **BLOCKED_IDENTITY_OWNERSHIP_MISMATCH** | API owner is `guchenyang`; no job submitted in this window |
| 5 | Compute-node CUDA | **NOT_RUN_PREREQUISITE_FAILED** | `GPU_AOSS_GATE.json` |
| 6 | Compute-node exact-object AOSS | **NOT_RUN_PREREQUISITE_FAILED** | `AOSS_COMPUTE_READ_AUDIT.csv` |
| 7 | Real C/B2 DataLoader | **NOT_RUN_PREREQUISITE_FAILED** | `AOSS_COMPUTE_READ_AUDIT.csv` |
| 8 | 1–5 scene strict-B2 real smoke | **NOT_RUN_PREREQUISITE_FAILED** | `STRICT_B2_REAL_SMOKE.md` |

`all_gates_pass=false`. The mandated stop condition was observed.

## Immutable source snapshot

An automatic commit was not appropriate because the worktree contains a large, mixed set of pre-existing tracked and untracked changes whose ownership and intended commit scope cannot safely be inferred. Instead, the gate uses a source-only immutable tar snapshot. It contains source, configuration, scripts, tests, and bounded text metadata; it does not archive large feature data.

- Snapshot: `artifacts/gate_resolution_20260825T032608Z_2075f3e/source_snapshot.tar`
- Files: 723
- Uncompressed payload bytes: 4,767,476
- Snapshot manifest SHA256: `fa80790c33a30b316d60fe34714c2fd175978db956726c76c76c3f8425bf691e`
- Tar SHA256: `5cdbbe4c75ef704a8748c9520d1350da664d9e0ae59c83849a1482185a64250e`
- Inventory: `SOURCE_SNAPSHOT_MANIFEST.csv`
- Hash record: `SOURCE_SNAPSHOT.sha256`

The initial porcelain-v2 status, ordinary and binary Git diffs, tracked-source hashes, and untracked-source inventory are preserved in this report directory. The eight required recovery/validity handoff inputs still match `HANDOFF_INPUTS.sha256` (8/8 unchanged).

## Runner audit result

Current `features/run_inscene15k.py`:

- 82,812 bytes
- SHA256 `ee95d1d4d2ba554a919d2668d7be555222dfd1a0ba440a5a83d7a67d1eba9486`
- Git blob `4b01d7a7f880441ddca0b928874a677045c76fda`

The current runner supports `streaming_prefix`, `target_isolated`, motion-uniform sampling controls, and the other modes documented in `RUNNER_VERSION_AUDIT.md`. Searches across all reachable refs, reflog-visible and unreachable Git objects, dangling trees, Cursor Local History/project records, Codex sessions, source bundles, launchers, run folders, and archive candidates found no original implementation containing either exact missing interface:

- `--streaming-window-obs-len`: **not recovered**
- `predictor_target`: **not recovered**

No interface was guessed or reconstructed. The current runner was not represented as the missing newer historical runner.

This gap does not block a DINO p8 strict-B2 real gate that only consumes the already existing streaming-prefix and target-isolated caches and builds GT from RGB/mask/depth geometry. It does block any new cross-prefix extraction that promises a shared full observation window and any new V-JEPA2 predictor-target extraction. Such work requires either an original evidence-backed source or a separately approved implementation labeled `RECONSTRUCTED_INTERFACE` with regression tests.

## Scheduler and GPU result

The local environment has one SCO profile. It can list workspace jobs, so the credential is not simply expired. However, the scheduler's authoritative ownership field assigns the prior gate job created through this profile to:

- API owner: `guchenyang`
- API owner ID: `feaf23e2-610e-4e20-aba5-9a3b6871633e`
- Recent 500 jobs owned by `yinbaiqiao`: 0

The earlier job `pt-yqzcx6n0` failed before container start (`start_time=null`). Its exposed schema contains no event/reason field. The one retry returned HTTP 403 with no response body. Because the same API identity has successful jobs with an equivalent one-GPU configuration before and after that attempt, the 403 cannot be proven to be a permanent role denial; its precise backend cause remains unresolved. The independent ownership mismatch is nevertheless conclusive for this window.

No existing job was entered, reused, killed, or occupied. No new scheduler job was submitted. Login-node `torch.cuda.is_available()==False` was recorded only as a login-node observation and was not treated as evidence about cluster GPU availability.

## Data, AOSS, DataLoader, and strict-B2 result

The exact environment values recovered from historical artifacts are frozen in `GPU_AOSS_GATE.json`, including the Conda Python, data/feature roots, split manifest, canonical C manifest, AOSS config, and API host. Local manifest existence and hashes were verified.

Remote AOSS reads were not repeated from the login node and were not promoted from the earlier timeout into a compute-node result. Since no compliant compute allocation was available, exact-object reads of `target_indices.npy`, a streaming feature, and a target-isolated feature through project `aoss_fs` were not run. Likewise, a real `InsScene15KDataset`, C sample, B2 sample, and 1–5 scene strict manifest were not run. `AOSS_COMPUTE_READ_AUDIT.csv` records every required step as `NOT_RUN_PREREQUISITE_FAILED` rather than fabricating success.

Consequently, this window makes no real-data claim for target-pixel hiding, past-only centroid invariance, VFM-independent GT, fixed-target cross-prefix pairing, or feature loading. Existing unit-test coverage is not substituted for the required real smoke.

## Repository verification

- Conda Python: 3.11.15
- PyTorch: `2.6.0+cu124`; build CUDA 12.4
- Core restored/gate files: `py_compile` **PASS**
- Runner `--help`: **PASS**
- Strict-manifest builder `--help`: **PASS**
- `tests/test_probe_contracts.py tests/test_paper_validity.py`: **27 passed**, 8 warnings
- `git diff --check`: **PASS**
- Required old handoff files: **8/8 hashes unchanged**

No gate/formal probe process was left running. An unrelated user-owned monitoring process was observed and deliberately left untouched.

## Current-worktree drift discovered during snapshot

The snapshot inventory found 34 substantive Python/Markdown/JSON/YAML/shell source or configuration files of size 0 or 1 byte, all with mtime `2026-08-25 02:56:38 UTC`, in addition to expected empty `__init__.py` files. These include formal launch/prepare scripts and several model/task configs. The complete list is `CURRENT_TINY_SOURCE_GATE_INVENTORY.tsv`.

They were not restored or investigated in this window because the request explicitly forbids redoing empty-file forensics or recovery. Their current tiny contents are preserved in the immutable snapshot. They do not invalidate the completed gate diagnostics, but they are an additional blocker for any formal expansion that relies on those files. A separately authorized recovery window must compare them against the already recovered hashes/history and determine why they changed after the prior handoff.

## Formal experiment status

No strict-B2 result, B2 baseline, C1/C2 fair baseline, common intersection, prefix AUC, difficulty curve, task correlation, or capacity pilot was launched or generated. No old canonical result or old report was overwritten. The prior tags `paper_validity_20260821T200224Z_2075f3e` and `recovery_20260822T034513Z_2075f3eb` remain inputs, not output targets.

The next formal window must first resolve the scheduler identity and the newly observed tiny-file drift, then resume at gate 4 using the exact handoff in `NEXT_FORMAL_RUN_HANDOFF.md`. Only after gates 4–8 all pass may it mint a fresh `paper_validity_exec_<UTC>_<IMMUTABLE_CODE_HASH>` and begin DINO p8 smoke training/evaluation.
