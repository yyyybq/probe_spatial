# Scheduler and HTTP 403 audit

Gate tag: `gate_resolution_20260825T032608Z_2075f3e`  
Verdict: **BLOCKED_IDENTITY_OWNERSHIP_MISMATCH. No job was submitted or entered in this window.**

## Local and API identity

- POSIX identity: `uid=20325(yinbaiqiao)`, primary group `yinbaiqiao`.
- Host: `app-cfc1ce01cbb442d6bbc689ab96668806-6dcdcb5669-w5vsd`.
- Scheduler CLI: `sco v1.2.0`, profile `default`, local profile username `yinbaiqiao`, zone `cn-fz-01a`.
- The profile has one AK/SK pair present (values were not printed; only presence and fingerprints were checked). It has no configured subscription/resource-group override.
- The profile can currently list the `aigc` workspace, so the credential is not simply expired.
- Scheduler API ownership for the recovery job created from this profile is `user_name=guchenyang`, user ID `feaf23e2-610e-4e20-aba5-9a3b6871633e`. Recent project jobs invoked against paths under this repository have the same ownership.
- In the most recent 500 jobs, zero records have `ownership.user_name=yinbaiqiao`.
- The machine has exactly one SCO profile. There is no locally available profile whose scheduler API ownership is proven to be `yinbaiqiao`.

The local profile label is not sufficient proof of job ownership; the scheduler response is authoritative. Under the explicit prohibition on other users' tokens/jobs, this profile cannot be used to create the requested “own” allocation.

## Prior failed gate and retry

`pt-yqzcx6n0` / `bq_repo_recovery_gate_08220418`:

- Created 2026-08-22 04:18:31 UTC; terminal state `FAILED` at 04:18:39 UTC.
- `start_time` is null, proving the container command did not start.
- Resolved resource: workspace `aigc`, resource group `default`, pool `zoetrope`, zone `cn-fz-01a`, `N4lS.Iq.I80.1` (1 H800), one replica, reserved quota, `HIGHEST` priority.
- Image: `registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617`.
- Mount: PV AFS volume `019ec9f9-6d12-7d49-aad4-864b15c9eb06` at `/mnt/umm`.
- The available describe/list schema exposes no event or scheduler-reason field. No worker or workload log exists because no start occurred. Therefore the first failure's internal reason is **not evidenced**, not guessed.

The single retry returned only:

`failed to get response (create job) from lepton server, status code: 403`

No response body, missing permission, namespace, quota, or role reason was returned. The exact 125-byte record is preserved in the prior recovery snapshot.

The 403 cannot be classified as an expired token: the same credential currently performs workspace job-list reads, and the API identity has later successful jobs. It also cannot be proven to be a permanent submission-role denial because successful one-GPU submissions with the same API ownership/config exist after the retry. Evidence supports **unresolved transient/backend/quota-or-policy denial**, with an independent and currently decisive ownership mismatch.

An IAM user-list query was attempted read-only and returned an explicit denial for missing `iam.user.list`. No credential refresh, profile mutation, or alternate-user token lookup was attempted.

## Historical successful resolved configurations

`pt-5v8b334j` (`bq_wanp8_full_gate_0822`) succeeded immediately before the failed recovery gate using the same image, mount, workspace, pool, 1-GPU worker spec, priority and reserved quota. `pt-l59q5ufx` and later project jobs also succeeded with this base configuration. Selected resolved JSON is in `SCHEDULER_SELECTED_RESOLVED_CONFIGS.json`; the full read-only 500-job response is `SCHEDULER_JOBS_500.json`.

This proves the resource specification is historically valid. It does not authorize reuse/exec and does not establish that the current profile belongs to `yinbaiqiao`.

## Required resolution

Before scheduler gate 4 can run, the user/project administrator must provide or activate a legitimate SCO profile whose API-created job reports `ownership.user_name=yinbaiqiao` (or explicitly document that a named project service identity is the authorized owner for this user and this run). The profile must also have create permission for workspace `aigc`, pool `zoetrope`, one `N4lS.Iq.I80.1`, the mounted PV, and the selected image.

Once that identity is available, submit exactly one short, isolated 1-GPU gate job. Do not use `HIGHEST` unless that identity is explicitly authorized; use normal priority and a finite gate command. Verify ownership from the create/describe response before entering or running the gate. No existing job may be entered, stopped, killed, copied, or reused.

