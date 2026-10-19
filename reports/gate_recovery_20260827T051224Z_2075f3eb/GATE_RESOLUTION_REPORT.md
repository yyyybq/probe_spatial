# Recovery and Gate Resolution Report

Tag: `gate_recovery_20260827T051224Z_2075f3eb`  
Repository: `/mnt/umm/users/yinbaiqiao/probe_spatial`  
Base HEAD: `2075f3eb270cb0e8907c3927b6b079c64965b75a` (`main`, dirty worktree)  
Outcome: **34/34 tiny repository files recovered; GPU/AOSS/DataLoader/strict-B2 gate blocked at scheduler create permission.**

## Executive result

The 34 substantive files found at 0 or 1 byte in the preceding gate were backed up and restored from evidence-backed original sources. Thirty-three were restored from dangling Git tree `e7fc7f857a2dd27eed3626c64d18f083bd1c0d87`; every selected blob is independently corroborated by an identical Cursor Local History SHA256. `scripts/restore_tstep_from_aoss.sh`, absent from that tree, was restored from its 2,277-byte Cursor Local History entry `vIbd.sh` dated 2026-08-21 05:10:15.159 UTC. No file was reconstructed.

The user explicitly confirmed on 2026-08-27 that scheduler identity `guchenyang` is an authorized project/shared identity. The prior ownership-policy blocker is therefore resolved. Two new one-GPU submissions were attempted using that authorized local SCO profile. Both were rejected by the create API with HTTP 403 before a job ID was allocated. Read APIs still work, both target pools are active, and both expose the requested one-H800 SKU. No job was created and no GPU was occupied.

Because the scheduler create prerequisite failed, compute-node CUDA, AOSS exact-object reads, real DataLoader samples, and real strict-B2 smoke were not run. No formal experiment was launched. A source-only final recovery checkpoint has nevertheless been frozen so the restored code cannot be lost while scheduler access is repaired.

## File recovery

- Before state: 3 zero-byte files and 31 one-byte files, all with filesystem mtime `2026-08-25 02:56:38 UTC`.
- Current backup: `artifacts/gate_recovery_20260827T051224Z_2075f3eb/pre_restore_tiny.tar`, SHA256 `6f85616d7577593eb698379997731aac4eee9797b3ea38baac3b7f75667de293`.
- Verified recovery-source archive: `artifacts/gate_recovery_20260827T051224Z_2075f3eb/recovery_sources.tar`, SHA256 `97a4e405c2431aa3abf3ba0d34fed05f03b0de89dcd32a1003ad43b7c7c6ed9e`.
- Git-object recoveries: 33.
- Cursor-history-only recoveries: 1.
- Reconstructed files: 0.
- Remaining substantive 0/1-byte files under repository code/config roots: 0.

Per-file source blob/history ID, timestamps, before/final sizes, and hashes are in `FILE_RECOVERY_MATRIX.csv`. Final hashes are also in `RESTORED_TINY_FILES.sha256`.

The common truncation timestamp and the new Cursor history entries labeled `Undo Create Diff` are consistent with the already established Cursor batch-overwrite mechanism. This round did not find evidence that quota or ENOSPC caused these 34 truncations.

## Recovery validation

- 25 restored shell scripts: `bash -n` PASS.
- 9 restored YAML files: PyYAML parse PASS.
- Core runner, strict-manifest builder, spatial diagnostics, and new gate helpers: `py_compile` PASS.
- `tests/test_probe_contracts.py tests/test_paper_validity.py`: 27 passed, 8 warnings, 32.00 seconds.
- `git diff --check`: PASS.
- Previous gate deliverables: 7/7 SHA256 unchanged.
- Required recovery/validity handoff inputs: 8/8 SHA256 unchanged.

## Scheduler attempts

The local SCO profile file remained `/mnt/umm/users/yinbaiqiao/.config/sco/profiles/default.toml`, mtime 2026-08-06 05:34:00 UTC, SHA256 `29384ecf933952d4e83028ed688b69de43f2b5aa989b4b6d5308c8526c4158f5`. Credential values were not printed.

Attempt 1:

- Display name: `bq_gate_recovery_08270524`
- Pool/spec: `zoetrope` / `N4lS.Iq.I80.1`
- Scheduling: `NORMAL`, `RESERVED`
- Result: HTTP 403 from create API; no job ID.

After that failure, read-only checks established that `zoetrope` and `h800` are ACTIVE and both advertise `N4lS.Iq.I80.1`. Recent successful authorized-identity jobs used `h800`, `HIGHEST`, `RESERVED`, so one controlled compatibility attempt was made:

- Display name: `bq_gate_recovery_08270535`
- Pool/spec: `h800` / `N4lS.Iq.I80.1`
- Scheduling: `HIGHEST`, `RESERVED`
- Result: HTTP 403 from create API; no job ID.

The retry aligned pool, priority, quota, image, mount, workspace, framework, node count, and SKU with known successful project settings. The repeated pre-ID 403 therefore supports a current create-permission/credential-policy failure for this local AK/SK, not a CUDA, AOSS, startup-command, or worker-capacity failure. A successful job under the same service-side `user_name` does not prove that it used the same AK/SK.

The external helper `/mnt/umm/users/yinbaiqiao/submit.sh` is currently one byte, mtime 2026-08-27 03:06:59 UTC. Cursor preserves its prior 2,583-byte version with SHA256 `4fc264d5554804b16e4a762867344247b7e3501044cb77c080ab2d520ae372f5`, but that helper only wraps the same `sco acp jobs create` call and carries no alternate authentication. It was not modified because it is outside this repository and restoring it would not resolve the 403.

## Gate matrix

| Gate | Status | Result |
|---|---|---|
| Recover 34 tiny repository files | **PASS** | 34 originals restored; zero reconstructed |
| Repository syntax/tests | **PASS** | shell/YAML/py_compile/diff checks and 27 tests pass |
| Authorized scheduler identity | **PASS_BY_USER_ATTESTATION** | `guchenyang` is an authorized project/shared identity |
| Create isolated 1-GPU job | **BLOCKED_SCHEDULER_HTTP_403** | Two bounded attempts; neither returned a job ID |
| Compute CUDA | **NOT_RUN_PREREQUISITE_FAILED** | No compute worker |
| Compute AOSS exact objects | **NOT_RUN_PREREQUISITE_FAILED** | No compute worker |
| Real DataLoader C/B2 samples | **NOT_RUN_PREREQUISITE_FAILED** | AOSS/compute gate not available |
| Real strict-B2 smoke | **NOT_RUN_PREREQUISITE_FAILED** | Prerequisite gates not passed |
| Formal experiments | **NOT_RUN** | Stop condition observed |

## Frozen recovery code identity

The dirty worktree is not identified by HEAD alone. A source-only immutable checkpoint was created after recovery:

- Manifest: `reports/gate_recovery_20260827T051224Z_2075f3eb/FINAL_SOURCE_MANIFEST.csv`
- Manifest SHA256: `ee640a94b09849ceaab20947e69e2db09977469462f4c8fc9aa28560a6568ce1`
- Snapshot: `artifacts/gate_recovery_20260827T051224Z_2075f3eb/final_source_snapshot.tar`
- Snapshot SHA256: `1335b4b4af35b28749c937c4676ea6917bee7591314796fff9036dd0193b0476`
- Files: 739
- Payload bytes: 5,219,428

The snapshot includes all 34 restored files and the adapted gate helpers, while excluding reports, results, logs, checkpoints, and large feature data. This is a recovery checkpoint identity, not a formal experiment `RUN_TAG`, because the real environment gates have not passed. Details are in `FINAL_CODE_IDENTITY.md`.

## Remaining blocker

The project administrator must restore `acp jobs create` permission for the AK/SK in the current local profile, or issue/activate a replacement credential for the same authorized project identity. Read-only access is insufficient. Once repaired, submit exactly one new short one-GPU job and resume with the frozen gate script. Do not reuse or enter existing jobs.

No gate or formal background process remains, no old canonical result was overwritten, and no old validity/recovery report was modified.
