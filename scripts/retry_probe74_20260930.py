#!/usr/bin/env python3
"""Freeze and submit replacements for failed/stale CogVideoX/LingBot probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
OUT = BASE / "retry_probe74_zoetrope_20260930_r1"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
SOURCES = {
    "cog84": BASE / "single_gpu_cog84_h800_20260928_r1",
    "cog12": BASE / "single_gpu_cog12_ego_fp32_zoetrope_20260928_r3",
    "ling48": BASE / "single_gpu_lingbot48_20260924_r1",
    "ling32": BASE / "single_gpu_lingbot32_zoetrope_20260928_r3",
}


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.chmod(tmp, 0o644); os.replace(tmp, path)


def sco(*args):
    p = subprocess.run([SCO, "acp", "jobs", *args], text=True, capture_output=True, timeout=180)
    if p.returncode: raise RuntimeError(p.stdout + "\n" + p.stderr)
    return p.stdout


def live():
    jobs = {}
    for page in range(1, 8):
        rows = json.loads(sco("list", "--workspace-name=aigc", "--page-size=500", f"--page-token={page}", "-o", "json"))
        jobs.update({r["name"]: r for r in rows})
        if len(rows) < 500: break
    return jobs


def active_job_map():
    rows = load(BASE / "probe552_progress/experiments_552.json")["experiments"]
    mapping = {r["unique_key"]: r.get("training", {}).get("job_id") for r in rows if r.get("training", {}).get("job_id")}
    for rel in ("single_gpu_cog12_ego_fp32_zoetrope_20260928_r3/submissions.json",
                "single_gpu_lingbot32_zoetrope_20260928_r3/submissions.json"):
        for key, record in load(BASE / rel)["jobs"].items(): mapping[key] = record["job_id"]
    for rel in ("h800_node_release_20260928_r1/migration.json", "h800_node_release_20260928_r2/migration.json"):
        for record in load(BASE / rel).get("zoetrope_state_root_repairs", []):
            if record.get("state") == "ZOETROPE_SUBMITTED": mapping[record["unique_key"]] = record["new_job_id"]
    return mapping


def completed_keys():
    rows = load(BASE / "probe552_progress/experiments_552.json")["experiments"]
    done = {r["unique_key"] for r in rows if r.get("training", {}).get("status") == "SUCCEEDED"}
    done.update(p.stem for p in BASE.glob("single_gpu_*/state/production/*/completed/*.json"))
    return done


def item_index():
    index = {}
    for group, root in SOURCES.items():
        for path in sorted((root / "items").glob("*.jsonl")):
            item = json.loads(path.read_text())
            index.setdefault(item["unique_key"], (group, root, path, item))
    return index


def latest_epoch_checkpoint(run_slug, key):
    root = ROOT / "logs/inscene15k_streaming/runs" / run_slug / "production" / run_slug / key
    candidates = [p for p in root.glob("**/checkpoints/epoch=*.ckpt") if p.is_file() and p.stat().st_size > 1_000_000]
    return str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else None


def prepare():
    if (OUT / "plan.json").exists():
        print(json.dumps({"status": "EXISTS", "plan": str(OUT / "plan.json")})); return
    jobs = live(); mapping = active_job_map(); done = completed_keys(); items = item_index()
    selected = []
    for key, job_id in sorted(mapping.items()):
        if key in done or not (key.startswith("cogvideox__") or key.startswith("lingbot__")): continue
        job = jobs.get(job_id, {}); state = job.get("state")
        if state not in {"FAILED", "RUNNING"}: continue
        group, source, old_item_path, item = items[key]
        source_plan = load(source / "plan.json")
        run_slug = source_plan["run_id"]
        checkpoint = latest_epoch_checkpoint(run_slug, key)
        selected.append({"unique_key": key, "group": group, "old_job_id": job_id, "old_state": state,
                         "old_pool": job.get("resource_pool", {}).get("name"), "source_item": str(old_item_path),
                         "source_item_sha256": sha(old_item_path), "resume_checkpoint": checkpoint})
    counts = {s: sum(r["old_state"] == s for r in selected) for s in ("FAILED", "RUNNING")}
    # The original audit found 54 FAILED and 20 RUNNING. Do not duplicate a
    # formerly stale job if it has since reached scheduler SUCCEEDED.
    if counts["FAILED"] != 54 or counts["RUNNING"] > 20:
        raise RuntimeError(f"fresh scope mismatch: selected={len(selected)} counts={counts}")
    for group in sorted({r["group"] for r in selected}):
        source = SOURCES[group]; old_plan = load(source / "plan.json"); old_run = old_plan["run_id"]
        group_root = OUT / group; new_run = f"main_v2_retry_probe74_zoetrope_20260930_r1_{group}"
        template = (source / "run_single.py").read_text()
        template = re.sub(r'^ITEM_ROOT = ROOT / ".*"$', f'ITEM_ROOT = Path("{group_root / "items"}")', template, flags=re.M)
        template = re.sub(r'^STATE_ROOT = ROOT / ".*"$', f'STATE_ROOT = Path("{group_root / "state"}")', template, flags=re.M)
        template = template.replace(str(ROOT / "logs/inscene15k_streaming/runs" / old_run),
                                    str(ROOT / "logs/inscene15k_streaming/runs" / new_run))
        template = template.replace(old_run, new_run)
        runner = group_root / "run_single.py"; runner.parent.mkdir(parents=True, exist_ok=True); runner.write_text(template)
        os.chmod(runner, 0o755)
        for ordinal, record in enumerate(r for r in selected if r["group"] == group):
            item = json.loads(Path(record["source_item"]).read_text())
            overrides = list(item["scientific_overrides"])
            overrides.append("callbacks.model_checkpoint.save_top_k=1")
            if record["resume_checkpoint"]: overrides.append(f"ckpt_path='{record['resume_checkpoint']}'")
            item["scientific_overrides"] = overrides
            path = group_root / "items" / f"{ordinal:04d}.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
            record.update(item_path=str(path), item_sha256=sha(path), run_id=new_run, runner=str(runner), runner_sha256=sha(runner))
    plan = {"schema": "PROBE74_RETRY_V1", "created_utc": now(), "resource_pool": "zoetrope",
            "selection": "fresh scheduler FAILED plus stale RUNNING, completion receipts excluded",
            "checkpoint_policy": "resume latest complete epoch checkpoint when present; save_top_k=1",
            "counts": counts, "jobs": selected}
    save(OUT / "plan.json", plan)
    print(json.dumps({"status": "FROZEN", "selected": len(selected), "counts": counts,
                      "resume": sum(bool(r["resume_checkpoint"]) for r in selected), "plan": str(OUT / "plan.json")}))


def submit():
    plan = load(OUT / "plan.json"); ledger_path = OUT / "submissions.json"
    ledger = load(ledger_path) if ledger_path.exists() else {"schema": "PROBE74_RETRY_SUBMISSIONS_V1", "resource_pool": "zoetrope", "jobs": {}}
    for ordinal, record in enumerate(plan["jobs"]):
        key = record["unique_key"]
        if key in ledger["jobs"] and ledger["jobs"][key].get("job_id"): continue
        if sha(record["item_path"]) != record["item_sha256"] or sha(record["runner"]) != record["runner_sha256"]: raise RuntimeError("frozen SHA mismatch")
        command = f"{PYTHON} {record['runner']} --item {record['item_path']} --item-sha256 {record['item_sha256']} --run-id {record['run_id']}"
        name = f"bq-r74c-z-{ordinal:02d}-{hashlib.sha256(key.encode()).hexdigest()[:6]}"
        out = sco("create", "--workspace-name=aigc", "--aec2-name=zoetrope", f"--job-name={name}", "--priority=HIGHEST",
                  "--quota-type=reserved", "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
                  "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm", "--training-framework=pytorch",
                  "--worker-nodes=1", "--worker-spec=N4lS.Iq.I80.1", f"--command={command}")
        ids = list(dict.fromkeys(re.findall(r"pt-[a-z0-9]+", out)))
        if len(ids) != 1: raise RuntimeError(f"ambiguous create: {out}")
        ledger["jobs"][key] = {**record, "job_name": name, "job_id": ids[0], "command": command, "state": "SUBMITTED", "submitted_utc": now()}
        save(ledger_path, ledger)
        print(json.dumps({"submitted": len(ledger["jobs"]), "key": key, "job_id": ids[0]}), flush=True)
    # Only after every replacement has a durable job ID, stop the 20 stale originals.
    running = [r["old_job_id"] for r in plan["jobs"] if r["old_state"] == "RUNNING"]
    if len(ledger["jobs"]) != len(plan["jobs"]): raise RuntimeError("replacement submission incomplete; refusing stops")
    stop_path = OUT / "stops.json"
    if not stop_path.exists():
        out = sco("stop", "--workspace-name=aigc", *running)
        save(stop_path, {"requested_utc": now(), "old_running_job_ids": running, "scheduler_output": out})
    print(json.dumps({"submitted": len(ledger["jobs"]), "old_running_stop_requested": len(running)}))


def status():
    jobs = live(); ledger = load(OUT / "submissions.json")
    from collections import Counter
    states = Counter(jobs.get(r["job_id"], {}).get("state", "NOT_VISIBLE") for r in ledger["jobs"].values())
    old = Counter(jobs.get(r["old_job_id"], {}).get("state", "NOT_VISIBLE") for r in ledger["jobs"].values() if r["old_state"] == "RUNNING")
    print(json.dumps({"replacements": states, "old_running": old}, sort_keys=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("action", choices=("prepare", "submit", "status")); a = p.parse_args()
    {"prepare": prepare, "submit": submit, "status": status}[a.action]()
