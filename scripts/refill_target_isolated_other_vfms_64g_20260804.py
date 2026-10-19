#!/usr/bin/env python3
"""Refill native 64-worker target-isolated feature extraction shards."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

KEY = "/mnt/umm/users/yinbaiqiao/id_rsa"
PROJ = "/mnt/umm/users/yinbaiqiao/probe_spatial"
RUN = Path(PROJ) / "logs/target_isolated_other_vfms_20260804_0418"
NODES = [
    "10.119.17.255",
    "10.119.19.201",
    "10.119.19.208",
    "10.119.19.209",
    "10.119.19.40",
    "10.119.20.137",
    "10.119.20.240",
    "10.119.22.132",
]
N_SCENES = 682
TOTAL_WORKERS = 64


def ssh(node: str, command: str, timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-n",
            "-i",
            KEY,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=/tmp/known_hosts",
            f"root@{node}",
            command,
        ],
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def worker_task(worker: int) -> tuple[str, int, int]:
    specs = [
        ("vjepa2", N_SCENES),
        ("videomae", N_SCENES),
        ("f3r", N_SCENES),
        ("aether", N_SCENES),
    ]
    total = sum(count for _, count in specs)
    lo = total * worker // TOTAL_WORKERS
    hi = total * (worker + 1) // TOTAL_WORKERS
    pos = 0
    for vfm, count in specs:
        task_lo, task_hi = pos, pos + count
        seg_lo, seg_hi = max(lo, task_lo), min(hi, task_hi)
        if seg_lo < seg_hi:
            return vfm, seg_lo - task_lo, seg_hi - task_lo
        pos = task_hi
    raise RuntimeError(f"worker {worker} has no task")


EXPECTED = {i: worker_task(i) for i in range(TOTAL_WORKERS)}


def finished_workers() -> set[int]:
    done: set[int] = set()
    for path in RUN.glob("node_*/gpu*_worker*.log"):
        if path.name.endswith(".supervisor.log"):
            continue
        text = path.read_text(errors="ignore")
        match = re.search(r"worker_index=(\d+)", text)
        if match and "finished=" in text and "Traceback" not in text and "OSError:" not in text:
            done.add(int(match.group(1)))
    return done


def active_workers() -> set[int]:
    active: set[int] = set()
    for node in NODES:
        try:
            proc = ssh(
                node,
                "ps -eo cmd | grep 'features.run_inscene15k' | grep 'target_isolated' | grep -v grep || true",
            )
        except Exception:
            continue
        for line in proc.stdout.splitlines():
            start_match = re.search(r"--start (\d+) --end (\d+)", line)
            vfm_match = re.search(r"--vfm (\w+)", line)
            if not start_match or not vfm_match:
                continue
            key = (vfm_match.group(1), int(start_match.group(1)), int(start_match.group(2)))
            for worker, expected in EXPECTED.items():
                if key == expected:
                    active.add(worker)
                    break
    return active


def gpu_memory(node: str, gpu: int) -> int | None:
    try:
        proc = ssh(
            node,
            f"nvidia-smi -i {gpu} --query-gpu=memory.used --format=csv,noheader,nounits",
            timeout=15,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(float(proc.stdout.strip().splitlines()[0]))
    except Exception:
        return None


def launch_worker(worker: int) -> dict:
    node_index, gpu = divmod(worker, 8)
    node = NODES[node_index]
    log_dir = RUN / f"node_{node_index}"
    log_dir.mkdir(parents=True, exist_ok=True)
    remote = (
        f"cd '{PROJ}' && mkdir -p '{log_dir}' && "
        f"setsid -f bash -lc 'exec > {log_dir}/gpu{gpu}_worker{worker}.supervisor.log 2>&1 < /dev/null; "
        f"RUN_ROOT={RUN} bash scripts/launch_target_isolated_other_vfms_64g_20260804.sh run-worker {node_index} {gpu}' ; "
        "echo DETACHED"
    )
    proc = ssh(node, remote, timeout=20)
    return {
        "worker": worker,
        "node": node,
        "gpu": gpu,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip()[:500],
    }


def iteration() -> dict:
    done = finished_workers()
    active = active_workers()
    launched = []
    skipped_busy = []
    for worker in range(TOTAL_WORKERS):
        if worker in done or worker in active:
            continue
        node_index, gpu = divmod(worker, 8)
        node = NODES[node_index]
        mem = gpu_memory(node, gpu)
        if mem is None or mem >= 1500:
            skipped_busy.append({"worker": worker, "node": node, "gpu": gpu, "mem": mem})
            continue
        launched.append(launch_worker(worker))
        time.sleep(0.1)
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "done": sorted(done),
        "active": sorted(active),
        "launched": launched,
        "skipped_busy": skipped_busy,
        "remaining": sorted(set(range(TOTAL_WORKERS)) - done - active - {x["worker"] for x in launched}),
    }
    (RUN / "refill_status.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    while True:
        report = iteration()
        print(
            f"{report['time_utc']} done={len(report['done'])} active={len(report['active'])} "
            f"launched={len(report['launched'])} remaining={len(report['remaining'])}",
            flush=True,
        )
        if len(report["done"]) == TOTAL_WORKERS:
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
