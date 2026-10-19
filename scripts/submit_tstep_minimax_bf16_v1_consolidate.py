#!/usr/bin/env python3
"""Submit the consolidation + full-verification job for the MiniMax bf16 re-extraction.

Runs after all 32 production shards (stages 2/3) have SUCCEEDED:

  1. queries the SCO state of every production_* job in
     logs/tstep_minimax_bf16_v1/submissions.json;
  2. runs scripts/consolidate_tstep_minimax_bf16_v1_shards.py --check on the
     login node (read-only) -> <consolidation dir>/check_<UTC>.json;
  3. with --submit, and only if every shard SUCCEEDED and the check PASSed,
     creates one 1-GPU Zoetrope job that runs (as root, which the root-owned
     shard and scene directories require)
        consolidate ... --apply  -> <consolidation dir>/<tag>/apply_report.json
        verify_tstep_minimax_bf16_v1_full.py -> <tag>/verify_full_report.json
     and records it under key ``consolidation`` in the extraction ledger.

Default is a dry run.  After the job SUCCEEDED, build the stage-5 matrix
(scripts/submit_tstep_minimax_bf16_v1_retrain.py), which also writes the strict
alignment audits for the bf16 namespaces.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit_tstep_minimax_bf16_v1_extract import (  # noqa: E402
    AOSS_CONF, LOG_ROOT, PRODUCTION_TAG, PY_PROBE, ROOT, job_state, record, sco_create, sha256,
)

CONSOLIDATOR = ROOT / "scripts/consolidate_tstep_minimax_bf16_v1_shards.py"
FULL_VERIFIER = ROOT / "scripts/verify_tstep_minimax_bf16_v1_full.py"
CONSOLIDATION_DIR = LOG_ROOT / PRODUCTION_TAG / "consolidation"
EXPECTED_SHARDS = 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--retry-reason", default="", help="required to re-submit after a FAILED consolidation job")
    parser.add_argument("--skip-states", action="store_true", help="dry run only: do not query SCO")
    args = parser.parse_args()
    ledger_path = LOG_ROOT / "submissions.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    shards = {k: v for k, v in ledger["jobs"].items() if k.startswith("production_")}
    if len(shards) != EXPECTED_SHARDS:
        raise SystemExit(f"ledger has {len(shards)} production shards, expected {EXPECTED_SHARDS}")
    states = {} if args.skip_states else {k: job_state(v["job_id"]) for k, v in sorted(shards.items())}
    if states:
        print("shard states:", dict(Counter(states.values())))
        for key, state in sorted(states.items()):
            if state != "SUCCEEDED":
                print(f"  {key} {shards[key]['job_id']} {state}")

    CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)
    check_report = CONSOLIDATION_DIR / f"check_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    check = subprocess.run([PY_PROBE, str(CONSOLIDATOR), "--report", str(check_report)],
                           text=True, capture_output=True)
    print(check.stdout[-4000:])
    check_status = json.loads(check_report.read_text(encoding="utf-8"))["status"]
    print(f"check report: {check_report} status={check_status}")

    previous = ledger["jobs"].get("consolidation")
    attempt = 1
    if previous and previous.get("job_id"):
        state = job_state(previous["job_id"])
        if state != "FAILED":
            raise SystemExit(f"consolidation job {previous['job_id']} is {state}; nothing to do")
        if not args.retry_reason:
            raise SystemExit(f"consolidation job {previous['job_id']} FAILED; pass --retry-reason to re-submit")
        attempt = int(previous.get("attempt", 1)) + 1
    tag = f"r{attempt}"
    out = CONSOLIDATION_DIR / tag
    command = f"cd {shlex.quote(str(ROOT))} && " + " && ".join([
        shlex.join(["env", f"AOSS_CONF={AOSS_CONF}", PY_PROBE, str(CONSOLIDATOR), "--apply",
                    "--report", str(out / "apply_report.json")]),
        shlex.join(["env", f"AOSS_CONF={AOSS_CONF}", "AOSS_CACHE_DIR=/tmp/aoss_verify_full", PY_PROBE,
                    str(FULL_VERIFIER), "--report", str(out / "verify_full_report.json")]),
    ])
    name = f"bq-mmx-bf16v1-consolidate-{tag}"
    print(f"job {name} (1 GPU):\n{command}")
    if not args.submit:
        print("dry run: no SCO job created (pass --submit)")
        return 0
    if args.skip_states or any(state != "SUCCEEDED" for state in states.values()):
        raise SystemExit("every production shard must be SUCCEEDED before consolidation")
    if check_status != "PASS":
        raise SystemExit("consolidation check did not PASS; see the check report")
    out.mkdir(parents=True, exist_ok=True)
    rec = sco_create(name, command, 1)
    rec.update({"stage": "consolidation", "attempt": attempt, "tag": tag, "check_report": str(check_report),
                "check_report_sha256": sha256(check_report),
                "consolidator_sha256": sha256(CONSOLIDATOR), "full_verifier_sha256": sha256(FULL_VERIFIER),
                "apply_report": str(out / "apply_report.json"),
                "verify_full_report": str(out / "verify_full_report.json")})
    if previous:
        prior = {k: v for k, v in previous.items() if k != "previous_attempts"}
        prior.update({"state": "FAILED", "retry_reason": args.retry_reason})
        rec["previous_attempts"] = list(previous.get("previous_attempts", [])) + [prior]
    record(ledger, "consolidation", rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
