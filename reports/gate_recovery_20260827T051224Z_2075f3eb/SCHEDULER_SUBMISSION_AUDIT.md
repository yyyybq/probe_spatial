# Scheduler Submission Audit

Tag: `gate_recovery_20260827T051224Z_2075f3eb`  
Verdict: **AUTHORIZED_IDENTITY_CONFIRMED; CREATE API BLOCKED_HTTP_403.**

## Authorization clarification

The user explicitly confirmed that scheduler identity `guchenyang` is an authorized project/shared identity. A job owned by that service identity is permitted for this project. The earlier interpretation that the owner mismatch itself violated policy is superseded by this authorization statement.

## Read-only checks

- SCO CLI: v1.2.0.
- Workspace: `aigc` is readable.
- Pools `zoetrope` and `h800`: ACTIVE.
- Both pools expose `N4lS.Iq.I80.1` (one H800, 14 vCPU, 240 GiB, zone `cn-fz-01a`).
- The profile can list and describe jobs.
- Recent successful jobs owned by `guchenyang` exist, including project paths under this repository.
- Local profile SHA256: `29384ecf933952d4e83028ed688b69de43f2b5aa989b4b6d5308c8526c4158f5`; mtime 2026-08-06 05:34:00 UTC. Secret values were never printed.

## Bounded create attempts

| Attempt | Display name | Pool | Spec | Priority/quota | Result |
|---:|---|---|---|---|---|
| 1 | `bq_gate_recovery_08270524` | `zoetrope` | `N4lS.Iq.I80.1` | NORMAL/RESERVED | HTTP 403; no job ID |
| 2 | `bq_gate_recovery_08270535` | `h800` | `N4lS.Iq.I80.1` | HIGHEST/RESERVED | HTTP 403; no job ID |

Both used one worker node, the project image `registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617`, PV `019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm`, and no fault-tolerance retry. The returned error in both cases was:

`failed to get response (create job) from lepton server, status code: 403`

No response body or missing permission name was exposed. The latest job list contains neither display name, confirming no job was created.

## Interpretation

Resource discovery and job-list APIs work while create is denied. Matching the second attempt to a recently successful project pool/priority/quota configuration did not change the outcome. The strongest supported conclusion is that the AK/SK stored in this local profile currently lacks create authorization or is subject to a create-specific policy denial. It may not be the same AK/SK used for other jobs that resolve to the same service-side `user_name`.

This is not evidence of missing cluster GPUs, AOSS failure, bad worker startup code, or quota exhaustion: the API rejected the request before creating a schedulable job.

## Required administrative action

Grant `acp jobs create` for workspace `aigc` to the current authorized project credential, or replace/refresh the local profile with a valid authorized project AK/SK. After the administrator change, perform one short single-GPU gate submission. Do not repeatedly retry before the credential or ACL changes.

`/mnt/umm/users/yinbaiqiao/submit.sh` is one byte but is not an authentication path; its preserved original merely invokes the same CLI. Restoring that external helper is separate repository-recovery work and cannot fix this 403.
