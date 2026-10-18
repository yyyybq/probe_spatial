#!/usr/bin/env python3
"""Submit v2 evaluations of canonical-V2 checkpoints as one-GPU Zoetrope jobs.

Reads artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_r1/plan.json
(built by scripts/build_eval_v2_canonical_items_20261017.py).  Each job runs the
SHA-bound runner run_eval_single.py on --per-job items in sequence and exits
non-zero if any of them failed.  Every item is validated on the login node
(hashes only) before anything is submitted.  Default: dry run.

  --mode canary      one item per job, keys given with --keys
  --mode production  all plan items not yet evaluated, --per-job items per job
                     (refused unless every canary item has an EVAL_COMPLETE.json)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_r1"
RUNNER = CAMPAIGN / "run_eval_single.py"
RUN_ID = "main_v2_eval_v2_canonical_20261017_r1"
OUTPUT_ROOT = ROOT / "logs/inscene15k_streaming/runs" / RUN_ID
PY = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
SCO = Path.home() / ".sco/bin/sco"
IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
JOB_ID_RE = re.compile(r"pt-[a-z0-9]+")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def validate(entry: dict, mode: str) -> None:
    done = subprocess.run([PY, str(RUNNER), "--item", entry["path"], "--item-sha256", entry["sha256"],
                           "--run-id", RUN_ID, "--mode", mode, "--validate-only"], text=True, capture_output=True)
    if done.returncode:
        raise SystemExit(f"validation failed for {entry['path']}:\n{done.stdout}\n{done.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("canary", "production"), required=True)
    parser.add_argument("--keys", nargs="*", default=[])
    parser.add_argument("--per-job", type=int, default=4)
    parser.add_argument("--limit-jobs", type=int, default=0)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--revision", choices=("r1", "r2", "r3", "r4", "t1", "c3"), default="r1",
                        help="r*: canonical-V2 campaigns; t1: timestep campaign artifacts/tstep_eval_v2_20261018_t1")
    args = parser.parse_args()
    global CAMPAIGN, RUNNER, RUN_ID, OUTPUT_ROOT
    if args.revision == "c3":
        CAMPAIGN = ROOT / "artifacts/c3_from_c1_eval_20261018_c3"
        RUN_ID = "c3_from_c1_eval_20261018_c3"
    elif args.revision.startswith("t"):
        CAMPAIGN = ROOT / f"artifacts/tstep_eval_v2_20261018_{args.revision}"
        RUN_ID = f"tstep_eval_v2_20261018_{args.revision}"
    else:
        CAMPAIGN = ROOT / f"artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_{args.revision}"
        RUN_ID = f"main_v2_eval_v2_canonical_20261017_{args.revision}"
    RUNNER = CAMPAIGN / "run_eval_single.py"
    OUTPUT_ROOT = ROOT / "logs/inscene15k_streaming/runs" / RUN_ID
    plan = json.loads((CAMPAIGN / "plan.json").read_text())
    ledger_path = CAMPAIGN / "submissions.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "schema": "MAIN_V2_EVAL_V2_SUBMISSIONS_V1", "run_id": RUN_ID, "runner_sha256": sha256(RUNNER), "jobs": {}}
    if ledger["runner_sha256"] != sha256(RUNNER):
        raise SystemExit("runner changed since the first submission; refusing")
    submitted = {k for job in ledger["jobs"].values() if job["mode"] == args.mode for k in job["keys"]}
    if args.mode == "canary":
        keys = args.keys
        per_job = 1
    else:
        canary_keys = {k for job in ledger["jobs"].values() if job["mode"] == "canary" for k in job["keys"]}
        if not canary_keys:
            raise SystemExit("no canary submitted yet")
        missing = [k for k in canary_keys if not (OUTPUT_ROOT / "canary" / k / "EVAL_COMPLETE.json").is_file()]
        if missing:
            raise SystemExit(f"canary not complete for {len(missing)} item(s), e.g. {missing[:3]}")
        keys = sorted(plan["items"])
        per_job = args.per_job
    keys = [k for k in keys if k not in submitted]
    for k in keys:
        if k not in plan["items"]:
            raise SystemExit(f"{k} is not in the plan")
        validate(plan["items"][k], args.mode)
    batches = [keys[i:i + per_job] for i in range(0, len(keys), per_job)]
    if args.limit_jobs:
        batches = batches[: args.limit_jobs]
    print(f"mode={args.mode} items={sum(len(b) for b in batches)} jobs={len(batches)} (validated)")
    for n, batch in enumerate(batches):
        steps = []
        for k in batch:
            e = plan["items"][k]
            steps.append(shlex.join([PY, str(RUNNER), "--item", e["path"], "--item-sha256", e["sha256"],
                                     "--run-id", RUN_ID, "--mode", args.mode]) + " || rc=1")
        command = f"cd {shlex.quote(str(ROOT))} && rc=0; " + "; ".join(steps) + "; exit $rc"
        tag = "c3c1" if args.revision == "c3" else "tstep" if args.revision.startswith("t") else "evalv2"
        name = f"bq-{tag}-{args.mode[:4]}-{len(ledger['jobs']):04d}"
        if not args.submit:
            print(f"  {name}: {', '.join(batch)}")
            continue
        argv = [str(SCO), "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope",
                f"--job-name={name}", "--priority=HIGHEST", "--quota-type=reserved",
                f"--container-image-url={IMAGE}", f"--storage-mount={MOUNT}", "--training-framework=pytorch",
                "--worker-nodes=1", "--worker-spec=N4lS.Iq.I80.1", f"--command={command}"]
        done = subprocess.run(argv, text=True, capture_output=True, timeout=180)
        match = JOB_ID_RE.search(done.stdout + done.stderr)
        ledger["jobs"][name] = {"mode": args.mode, "keys": batch, "job_id": match.group(0) if match else None,
                                "returncode": done.returncode, "command": command, "submitted_utc": now()}
        ledger["updated_utc"] = now()
        atomic_json(ledger_path, ledger)
        print(f"  {name}\t{ledger['jobs'][name]['job_id'] or 'SUBMIT_ERROR'}", flush=True)
        if done.returncode or not match:
            print(done.stdout, done.stderr)
            raise SystemExit("submission failed; stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
