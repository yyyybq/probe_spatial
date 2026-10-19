# Next Actions

## External scheduler action

Ask the project/SCO administrator to restore `acp jobs create` permission for the AK/SK currently stored in `/mnt/umm/users/yinbaiqiao/.config/sco/profiles/default.toml`, or issue a replacement credential for the same authorized `guchenyang` project/shared identity. Provide these facts:

- job-list, describe, AEC2-list, and worker-spec reads succeed;
- both `zoetrope` and `h800` are ACTIVE;
- both expose `N4lS.Iq.I80.1`;
- create returns HTTP 403 with no body under both NORMAL and known-compatible HIGHEST settings;
- neither attempt created a job ID.

Do not retry until the credential or ACL changes.

## Resume command semantics

After the administrator confirms the fix, create exactly one new one-GPU job using the frozen script:

`/mnt/umm/users/yinbaiqiao/probe_spatial/reports/gate_recovery_20260827T051224Z_2075f3eb/run_gpu_aoss_gate_sco.sh`

Use the project image, `/mnt/umm` PV, one `N4lS.Iq.I80.1`, and no fault-tolerance retry. Verify the returned job owner is the authorized shared identity and that a new job ID exists before monitoring it. Do not enter or reuse existing jobs.

The script will run in order: CUDA/NVIDIA checks, exact known AOSS objects through project `aoss_fs`, one real DINO p8 DataLoader sample, and then a 1–5 scene strict-B2 manifest/validation. If any prerequisite fails, it stops before strict smoke.

## Code identity

Use recovery source-manifest hash:

`ee640a94b09849ceaab20947e69e2db09977469462f4c8fc9aa28560a6568ce1`

Verify `FINAL_SOURCE_SNAPSHOT.sha256` and `RESTORED_TINY_FILES.sha256` before resuming. The external `/mnt/umm/users/yinbaiqiao/submit.sh` is still one byte; either restore it separately from Cursor history or continue using the direct SCO command. It does not provide alternate authentication and is not required by the frozen gate script.

Only after CUDA, AOSS, DataLoader, and real strict-B2 gates all pass should the next window mint a new `paper_validity_exec_<UTC>_<IMMUTABLE_CODE_HASH>` formal tag. No formal experiment is authorized from the current blocked state.
