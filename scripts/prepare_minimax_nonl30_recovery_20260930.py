#!/usr/bin/env python3
"""Render the isolated MiniMax BF16-V2 non-L30 Zoetrope resume jobs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917/minimax_nonl30_recovery_20260930_r1"
RUNNER = ROOT / "runtime/feature_upload_recovery_nonl30_v1/minimax_nonl30_recovery.py"
OLD = ["pt-4lma0exb", "pt-z7wd113c", "pt-ili01g8p", "pt-9cozgsr6", "pt-d7l8a52q", "pt-q0j47v8u", "pt-yrla3xen", "pt-k6luh9rd"]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    jobs = []
    scripts = BASE / "submit_scripts"; scripts.mkdir(exist_ok=True)
    for shard, old in enumerate(OLD):
        script = scripts / f"shard-{shard:02d}.sh"
        script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                          f"cd {ROOT}\nexec /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python {RUNNER} --old-job {old} --shard {shard}\n")
        script.chmod(0o555)
        name = f"bq-mm-bf16v2-nonl30-s{shard:02d}-r1"
        argv = ["/mnt/umm/users/yinbaiqiao/.sco/bin/sco", "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope", f"--job-name={name}", "--priority=HIGHEST", "--quota-type=reserved", "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617", "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm", "--training-framework=pytorch", "--worker-nodes=1", "--worker-spec=N4lS.Iq.I80.1", f"--command=cd {ROOT} && bash {script}"]
        jobs.append({"shard": shard, "prior_job": old, "job_name": name, "script": str(script), "script_sha256": sha(script), "sco_argv": argv})
    ledger = {"status": "READY_NOT_SUBMITTED", "created_utc": datetime.now(timezone.utc).isoformat(), "pool": "zoetrope", "worker_spec": "N4lS.Iq.I80.1", "runner": str(RUNNER), "runner_sha256": sha(RUNNER), "scope": {"streaming_layers": [10,20,40,49], "target_layers": [20,40,49], "excluded_remote_verified_layer": 30}, "jobs": jobs}
    (BASE / "submission_plan.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({"plan": str(BASE / 'submission_plan.json'), "jobs": len(jobs), "runner_sha256": ledger["runner_sha256"]}))

if __name__ == "__main__": main()
