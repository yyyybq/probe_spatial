#!/usr/bin/env python3
"""Submit the MiniMax-H3 bf16 seeded-v1 re-extraction jobs (stages 1-3).

Stage 1 (``--canary``): one 1-GPU Zoetrope job that runs scene 0 only through
all four (timestep, mode) steps from the frozen runtime and then verifies the
published caches against the fp16 seeded-v1 originals
(scripts/verify_tstep_minimax_bf16_v1_canary.py).

Stages 2/3 (``--production``): 32 one-GPU jobs = 2 timesteps x 2 modes x 8
scene shards (scripts/run_tstep_minimax_bf16_v1_extract_shard.sh).  Refused
unless the canary report says PASS.  Each shard owns its local root, done-index
and step lock (the frozen sync loop's done-index merge is not locked, so shards
must not share one); all shards publish into the shared AOSS namespaces.

Ledger: logs/tstep_minimax_bf16_v1/submissions.json (+ sco_submit.log).
Default is a dry run that prints the exact commands.
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
RELEASE_NAME = "tstep_minimax_bf16_v1_extractor_r2"
RELEASE = ROOT / "runtime" / RELEASE_NAME
LAUNCHER = ROOT / "scripts/run_tstep_minimax_bf16_v1_extract.sh"
SHARD_LAUNCHER = ROOT / "scripts/run_tstep_minimax_bf16_v1_extract_shard.sh"
SHARD_COUNT = 8
VERIFIER = ROOT / "scripts/verify_tstep_minimax_bf16_v1_canary.py"
LOG_ROOT = ROOT / "logs/tstep_minimax_bf16_v1"
DONE_DIR = ROOT / "logs/tstep_canonical_v2_seeded_v1/aoss_done"
LOCK_DIR = ROOT / "logs/tstep_canonical_v2_seeded_v1/step_locks"
PY_PROBE = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
# Jobs run as root in the container (no ~/.petreloss.conf); vidfm3d.utils.aoss_fs
# (used by the verifier) needs AOSS_CONF, the sync loop reads the same file itself.
AOSS_CONF = "/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf"
SCO = Path.home() / ".sco/bin/sco"
CONTAINER_IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
STORAGE_MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
JOB_ID_RE = re.compile(r"pt-[a-z0-9]+")
NAMESPACES = {
    "stream": "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1",
    "target": "FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1",
}
TIMESTEPS = (500, 1000)
STEPS = [(t, mode) for t in TIMESTEPS for mode in ("stream", "target")]
CANARY_TAG = "canary_20261016"
CANARY_SCENE = "00777c41d4"  # scene 0 of the 682 collected ScanNet++ scenes
PRODUCTION_TAG = "production_20261016"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def frozen_tree_sha256() -> str:
    manifest = json.loads((RELEASE / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise SystemExit("frozen runtime manifest is not FROZEN")
    check = subprocess.run([PY_PROBE, str(ROOT / "scripts/freeze_tstep_minimax_bf16_v1_extractor.py"), "--verify",
                            "--release", RELEASE_NAME],
                           text=True, capture_output=True)
    if check.returncode:
        raise SystemExit(f"frozen runtime verification failed:\n{check.stdout}\n{check.stderr}")
    report = json.loads(check.stdout)
    if report["status"] != "PASS" or report["source_tree_sha256"] != manifest["source_tree_sha256"]:
        raise SystemExit(f"frozen runtime verification mismatch: {report}")
    return manifest["source_tree_sha256"]


def ensure_done_indices() -> list[Path]:
    paths = []
    for mode, namespace in NAMESPACES.items():
        for t in TIMESTEPS:
            path = DONE_DIR / f"{namespace}_minimax_h3_t{t}.txt"
            if path.is_symlink():
                raise SystemExit(f"done-index is a symlink: {path}")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                os.chmod(path, 0o644)
            paths.append(path)
    return paths


def step_command(tree_sha: str, t: int, mode: str, run_tag: str, n_gpus: int, scene_end: int | None) -> str:
    env = [
        f"EXPECTED_SOURCE_TREE_SHA256={tree_sha}", f"T={t}", f"MODE={mode}", f"RUN_TAG={run_tag}",
        f"N_GPUS={n_gpus}", "SYNC_TRUST_EXTRACT=0", f"AOSS_CONF={AOSS_CONF}",
    ]
    if scene_end is not None:
        env += ["SCENE_START=0", f"SCENE_END={scene_end}"]
    return shlex.join(["env", *env, "bash", str(LAUNCHER)])


def canary_tag(attempt: int) -> str:
    return CANARY_TAG if attempt == 1 else f"{CANARY_TAG}_r{attempt}"


def canary_command(tree_sha: str, tag: str) -> str:
    steps = " && ".join(step_command(tree_sha, t, mode, tag, 1, 1) for t, mode in STEPS)
    verify = shlex.join([
        "env", f"AOSS_CONF={AOSS_CONF}",
        PY_PROBE, str(VERIFIER), "--run-tag", tag, "--scene-count", "1",
        "--report", str(LOG_ROOT / tag / "verify_report.json"),
    ])
    return f"cd {shlex.quote(str(ROOT))} && {steps} && {verify}"


def job_state(job_id: str) -> str:
    completed = subprocess.run(
        [str(SCO), "acp", "jobs", "describe", job_id, "--workspace-name=aigc", "-o", "json"],
        text=True, capture_output=True, timeout=120,
    )
    try:
        return str(json.loads(completed.stdout).get("state", "UNKNOWN"))
    except json.JSONDecodeError:
        return "UNKNOWN"


def sco_create(name: str, command: str, n_gpus: int) -> dict:
    argv = [
        str(SCO), "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope",
        f"--job-name={name}", "--priority=HIGHEST", "--quota-type=reserved",
        f"--container-image-url={CONTAINER_IMAGE}", f"--storage-mount={STORAGE_MOUNT}",
        "--training-framework=pytorch", "--worker-nodes=1", f"--worker-spec=N4lS.Iq.I80.{n_gpus}",
        f"--command={command}",
    ]
    completed = subprocess.run(argv, text=True, capture_output=True, timeout=180)
    text = completed.stdout + "\n" + completed.stderr
    match = JOB_ID_RE.search(text)
    return {
        "job_name": name, "command": command, "worker_spec": f"N4lS.Iq.I80.{n_gpus}",
        "job_id": match.group(0) if match else None, "returncode": completed.returncode,
        "stdout": completed.stdout, "stderr": completed.stderr, "submitted_utc": now(),
    }


def load_ledger(tree_sha: str) -> dict:
    path = LOG_ROOT / "submissions.json"
    if path.exists():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        if ledger.get("source_tree_sha256") != tree_sha:
            # A new frozen release may replace the old one only before any
            # production shard exists; canary attempts keep their own tree sha.
            if any(key.startswith("production_") for key in ledger["jobs"]):
                raise SystemExit("ledger has production jobs on a different frozen tree; refusing")
            ledger.setdefault("tree_history", []).append({
                "source_tree_sha256": ledger.get("source_tree_sha256"),
                "release": ledger.get("release", "tstep_minimax_bf16_v1_extractor"),
                "replaced_utc": now(),
            })
            ledger["source_tree_sha256"] = tree_sha
            ledger["release"] = RELEASE_NAME
        return ledger
    return {
        "schema": "TSTEP_MINIMAX_BF16_V1_EXTRACT_SUBMISSIONS_V1",
        "source_tree_sha256": tree_sha,
        "release": RELEASE_NAME,
        "launcher_sha256": sha256(LAUNCHER),
        "verifier_sha256": sha256(VERIFIER),
        "resource_pool": "zoetrope",
        "namespaces": NAMESPACES,
        "layers": [20, 25],
        "jobs": {},
    }


def record(ledger: dict, key: str, rec: dict) -> None:
    ledger["jobs"][key] = rec
    ledger["updated_utc"] = now()
    atomic_json(LOG_ROOT / "submissions.json", ledger)
    with (LOG_ROOT / "sco_submit.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{rec['submitted_utc']}] {key} {rec['job_name']} job_id={rec['job_id']} rc={rec['returncode']}\n")
    print(f"{key}\t{rec['job_id'] or 'SUBMIT_ERROR'}", flush=True)
    if rec["returncode"] or not rec["job_id"]:
        print(rec["stdout"], rec["stderr"], flush=True)
        raise SystemExit(f"submission failed for {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--canary", action="store_true")
    group.add_argument("--production", action="store_true")
    parser.add_argument("--submit", action="store_true", help="create the SCO job(s); default prints the plan")
    parser.add_argument("--steps", default="all", help="production only: comma list like 500:stream,1000:target")
    parser.add_argument("--retry-reason", default="",
                        help="canary only: re-submit after a FAILED canary under a new run tag (required then)")
    args = parser.parse_args()
    for path in (LAUNCHER, VERIFIER):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    tree_sha = frozen_tree_sha256()
    ensure_done_indices()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    ledger = load_ledger(tree_sha)

    if args.canary:
        previous = ledger["jobs"].get("canary")
        attempt = 1
        if previous and previous.get("job_id"):
            state = job_state(previous["job_id"])
            if state != "FAILED":
                raise SystemExit(f"canary {previous['job_id']} is {state}; only a FAILED canary may be retried")
            if not args.retry_reason:
                raise SystemExit(f"canary {previous['job_id']} FAILED; pass --retry-reason to re-submit")
            attempt = int(previous.get("attempt", 1)) + 1
        tag = canary_tag(attempt)
        command = canary_command(tree_sha, tag)
        name = f"bq-mmx-bf16v1-canary-1016-r{attempt}" if attempt > 1 else "bq-mmx-bf16v1-canary-1016"
        print(f"canary attempt {attempt} job {name} (1 GPU), run tag {tag}:\n{command}")
        if not args.submit:
            return 0
        # A retried canary may reuse scene-0 units that an earlier attempt on the
        # SAME frozen release already published (the verifier re-checks every
        # published file); anything else in the canonical indices is refused.
        reused = {}
        for t, mode in STEPS:
            done = DONE_DIR / f"{NAMESPACES[mode]}_minimax_h3_t{t}.txt"
            lines = [line for line in done.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                continue
            if any(not line.startswith(f"{CANARY_SCENE}/") for line in lines):
                raise SystemExit(f"done-index {done} holds entries outside the canary scene; refusing")
            if not previous or previous.get("release") != RELEASE_NAME:
                raise SystemExit(f"done-index {done} has entries from a different frozen release; refusing")
            reused[f"t{t}_{mode}"] = len(lines)
        rec = sco_create(name, command, 1)
        rec.update({"stage": "canary", "run_tag": tag, "scene_span": "0:1", "attempt": attempt,
                    "release": RELEASE_NAME, "source_tree_sha256": tree_sha,
                    "reused_published_units": reused,
                    "ads_wrapper_sha256": sha256(ROOT / "scripts/ads_cli_in_logdir.sh"),
                    "launcher_sha256": sha256(LAUNCHER), "verifier_sha256": sha256(VERIFIER)})
        if previous:
            prior = {k: v for k, v in previous.items() if k != "previous_attempts"}
            prior.update({"state": "FAILED", "retry_reason": args.retry_reason})
            rec["previous_attempts"] = list(previous.get("previous_attempts", [])) + [prior]
        record(ledger, "canary", rec)
        return 0

    active_tag = ledger["jobs"].get("canary", {}).get("run_tag", CANARY_TAG)
    report_path = LOG_ROOT / active_tag / "verify_report.json"
    if not report_path.is_file():
        raise SystemExit(f"canary verification report missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise SystemExit(f"canary report is {report.get('status')}; production refused")
    if sha256(SHARD_LAUNCHER) != ledger.setdefault("shard_launcher_sha256", sha256(SHARD_LAUNCHER)):
        raise SystemExit("shard launcher changed since the first production submission; refusing")
    wanted = STEPS if args.steps == "all" else [
        (int(item.split(":")[0]), item.split(":")[1]) for item in args.steps.split(",")
    ]
    for t, mode in wanted:
        if (t, mode) not in STEPS:
            raise SystemExit(f"unknown step {t}:{mode}")
        for shard in range(SHARD_COUNT):
            shard_id = f"t{t}_{mode}_s{shard}of{SHARD_COUNT}"
            key = f"production_{shard_id}"
            state_dir = LOG_ROOT / PRODUCTION_TAG / "shards" / shard_id
            log_dir = LOG_ROOT / PRODUCTION_TAG / f"minimax_h3_{shard_id}"
            done_index = state_dir / "aoss_done" / f"{NAMESPACES[mode]}_minimax_h3_t{t}.txt"
            lock = state_dir / "step.lock"
            command = f"cd {shlex.quote(str(ROOT))} && " + shlex.join([
                "env", f"EXPECTED_SOURCE_TREE_SHA256={tree_sha}", f"T={t}", f"MODE={mode}",
                f"SHARD_INDEX={shard}", f"SHARD_COUNT={SHARD_COUNT}", f"RUN_TAG={PRODUCTION_TAG}",
                "SYNC_TRUST_EXTRACT=0", f"AOSS_CONF={AOSS_CONF}", "bash", str(SHARD_LAUNCHER),
            ])
            name = f"bq-mmx-bf16v1-{mode}-t{t}-s{shard}-1016"
            print(f"{key}: {name} (1 GPU)")
            if not args.submit:
                continue
            if key in ledger["jobs"] and ledger["jobs"][key].get("job_id"):
                print(f"  already submitted: {ledger['jobs'][key]['job_id']}")
                continue
            done_index.parent.mkdir(parents=True, exist_ok=True)
            if done_index.is_symlink():
                raise SystemExit(f"done-index is a symlink: {done_index}")
            if not done_index.exists():
                done_index.touch()
                os.chmod(done_index, 0o644)
            if lock.exists():
                owner = lock.read_text(encoding="utf-8").strip()
                if owner != str(log_dir):
                    raise SystemExit(f"step lock {lock} is owned by {owner}")
            else:
                lock.write_text(str(log_dir) + "\n", encoding="utf-8")
            rec = sco_create(name, command, 1)
            rec.update({"stage": "production", "run_tag": PRODUCTION_TAG, "timestep": t, "mode": mode,
                        "shard_index": shard, "shard_count": SHARD_COUNT, "log_dir": str(log_dir),
                        "done_index": str(done_index), "step_owner_lock": str(lock),
                        "shard_launcher_sha256": sha256(SHARD_LAUNCHER),
                        "release": RELEASE_NAME, "source_tree_sha256": tree_sha,
                        "canary_report_sha256": sha256(report_path)})
            record(ledger, key, rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
