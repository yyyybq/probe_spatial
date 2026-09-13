#!/usr/bin/env python3
"""Run a recoverable, GPU-bounded D-series training stage."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.d_series.feature_store import MANIFEST_SHA256
from scripts.d_series.run_d1 import REVISIONS


@dataclass(frozen=True)
class Task:
    name: str
    experiment: str
    model: str
    branch: str
    seed: int
    output_dir: Path
    command: tuple[str, ...]
    reference: str | None = None


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def completed(task: Task):
    receipt_path = task.output_dir / "run_receipt.json"
    metrics_path = task.output_dir / "per_sample_metrics.csv"
    checkpoint_path = task.output_dir / "checkpoint.pt"
    if not task.output_dir.exists():
        return False
    if not all(path.is_file() for path in (receipt_path, metrics_path, checkpoint_path)):
        raise RuntimeError(f"partial output requires inspection; refusing overwrite: {task.output_dir}")
    receipt = json.loads(receipt_path.read_text())
    expected = {"status": "complete_development_evaluation", "experiment": task.experiment,
                "model": task.model, "branch": task.branch, "seed": task.seed,
                "manifest_sha256": MANIFEST_SHA256, "model_revision": REVISIONS[task.model]}
    if task.reference is not None:
        expected["reference"] = task.reference
    mismatch = {key: (receipt.get(key), value) for key, value in expected.items() if receipt.get(key) != value}
    if mismatch:
        raise RuntimeError(f"completed-output receipt mismatch at {receipt_path}: {mismatch}")
    return True


def d1_tasks(args):
    tasks = []
    model_specs = [(model, "isolated_target", branch) for model in REVISIONS for branch in ("spatial", "pooled", "linear_spatial")]
    model_specs += [("vjepa2_vitl256", "native_target", branch) for branch in ("spatial", "pooled")]
    for model, reference, branch in model_specs:
        for seed in (0, 1, 2):
            name = f"d1_{model}_{reference}_{branch}_s{seed}"
            output = args.run_root / "d1" / name
            command = (sys.executable, "scripts/d_series/run_d1.py", "--model", model, "--reference", reference,
                       "--branch", branch, "--seed", str(seed), "--scene-root", str(args.scene_root),
                       "--feature-root", str(args.feature_root), "--output-dir", str(output))
            tasks.append(Task(name, "D1", model, branch, seed, output, command, reference))
    return tasks


def downstream_tasks(args):
    tasks = []
    for model in REVISIONS:
        for seed in (0, 1, 2):
            source = args.run_root / "d1" / f"d1_{model}_isolated_target_pooled_s{seed}" / "checkpoint.pt"
            name = f"d2_{model}_s{seed}"
            output = args.run_root / "d2" / name
            command = (sys.executable, "scripts/d_series/run_d2.py", "--model", model, "--seed", str(seed),
                       "--scene-root", str(args.scene_root), "--feature-root", str(args.feature_root),
                       "--d1-checkpoint", str(source), "--output-dir", str(output))
            tasks.append(Task(name, "D2", model, "pooled", seed, output, command))
    for branch in ("spatial", "pooled"):
        for seed in (0, 1, 2):
            source = args.run_root / "d1" / f"d1_vjepa2_vitl256_native_target_{branch}_s{seed}" / "checkpoint.pt"
            name = f"d3_vjepa2_vitl256_{branch}_s{seed}"
            output = args.run_root / "d3" / name
            command = (sys.executable, "scripts/d_series/run_d3.py", "--branch", branch, "--seed", str(seed),
                       "--scene-root", str(args.scene_root), "--feature-root", str(args.feature_root),
                       "--native-d1-checkpoint", str(source), "--output-dir", str(output))
            tasks.append(Task(name, "D3", "vjepa2_vitl256", branch, seed, output, command))
    return tasks


def run_task(task, gpu_queue, log_dir):
    if completed(task):
        return {"task": task.name, "status": "resume_skip_complete", "returncode": 0}
    task.output_dir.parent.mkdir(parents=True, exist_ok=True)
    gpu = gpu_queue.get()
    started = utc_now()
    log_path = log_dir / f"{task.name}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        with log_path.open("w") as handle:
            handle.write(json.dumps({"task": task.name, "gpu": gpu, "started_utc": started,
                                     "command": list(task.command)}) + "\n")
            handle.flush()
            result = subprocess.run(task.command, cwd=REPO_ROOT, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, text=True)
    finally:
        gpu_queue.put(gpu)
    return {"task": task.name, "gpu": gpu, "started_utc": started, "ended_utc": utc_now(),
            "status": "complete" if result.returncode == 0 else "failed", "returncode": result.returncode,
            "log": str(log_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("d1", "d2d3"), required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch_cuda_available():
        raise RuntimeError("training campaign requires CUDA")
    gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one device")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    tasks = d1_tasks(args) if args.stage == "d1" else downstream_tasks(args)
    if args.stage == "d2d3":
        required = [task for task in d1_tasks(args) if task.branch != "linear_spatial"]
        if not all(completed(task) for task in required):
            raise RuntimeError("D2/D3 stage requires every main D1 checkpoint and receipt")
    gpu_queue = queue.Queue()
    for gpu in gpu_ids:
        gpu_queue.put(gpu)
    results = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(run_task, task, gpu_queue, args.report_dir) for task in tasks]
        for future in as_completed(futures):
            value = future.result(); results.append(value); print(json.dumps(value), flush=True)
    status = {"schema_version": 1, "stage": args.stage, "ended_utc": utc_now(), "tasks": sorted(results, key=lambda row: row["task"]),
              "failed": sum(row["returncode"] != 0 for row in results)}
    (args.report_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    if status["failed"]:
        raise SystemExit(f"{status['failed']} training tasks failed")


def torch_cuda_available():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
