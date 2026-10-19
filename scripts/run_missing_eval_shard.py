#!/usr/bin/env python3
"""Run a shard of missing eval_diag jobs on one worker."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
PYTHON = os.environ.get("PYTHON_BIN", "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python")
RUN_ID = os.environ["RUN_ID"]
SHARD_TSV = Path(os.environ["SHARD_TSV"])
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT / f"logs/probing_jobs_{RUN_ID}"))
SLOTS = int(os.environ.get("SLOTS", "8"))
GPUS = [x for x in os.environ.get("GPU_IDS_CSV", "0,1,2,3,4,5,6,7").split(",") if x]
DATA_ROOT = os.environ.get("INSCENE_DATA_ROOT", "aoss://baiqiao/InsScene-15K/data")


def hydra_path(path: str) -> str:
    return path.replace("=", r"\=")


def rows() -> list[dict[str, str]]:
    with SHARD_TSV.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def resolve_scene_manifest(row: dict[str, str]) -> str:
    raw_path = row.get("scene_manifest", "")
    if not raw_path:
        raise FileNotFoundError(
            f"missing scene_manifest for {row.get('job_key', '<unknown>')}; "
            "refusing to fall back to canonical"
        )
    raw = Path(raw_path)
    candidate = raw if raw.is_absolute() else ROOT / raw
    if not candidate.exists():
        raise FileNotFoundError(
            f"scene_manifest does not exist for {row.get('job_key', '<unknown>')}: "
            f"{raw_path}; refusing to fall back to canonical"
        )
    return str(candidate)


def run_job(row: dict[str, str], slot: int) -> dict[str, object]:
    gpu = GPUS[slot % len(GPUS)]
    key = row["job_key"]
    out_name = f"{RUN_ID}_{key}"
    run_folder = f"{RUN_ID}/{out_name}"
    summary = ROOT / "logs/inscene15k_streaming/runs" / run_folder / "eval/val_summary.json"
    log_dir = ROOT / "logs/inscene15k_streaming/eval_jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{RUN_ID}_slot{slot}_{key}.log"
    status = {
        "key": key,
        "slot": slot,
        "gpu": gpu,
        "summary": str(summary),
        "log": str(log_path),
    }
    if summary.exists():
        try:
            existing = json.loads(summary.read_text())
            if int(existing.get("n", 0)) > 0:
                return {**status, "rc": 0, "skipped": True}
        except Exception:
            pass

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PROJECT_ROOT": str(ROOT),
            "CUDA_VISIBLE_DEVICES": gpu,
            "INSCENE_DATA_ROOT": DATA_ROOT,
            "INSCENE_STREAMING_FEAT_ROOT": row["streaming_feat_root"],
            "AOSS_CACHE_DIR": f"/mnt/local_cache/inscene/aoss_cache/{RUN_ID}/{key}",
        }
    )
    if row.get("target_feat_root"):
        env["INSCENE_TARGET_FEAT_ROOT"] = row["target_feat_root"]
    try:
        scene_manifest = resolve_scene_manifest(row)
    except FileNotFoundError as exc:
        with log_path.open("w") as log:
            print(
                f"FAIL_CLOSED {datetime.utcnow().isoformat()}Z slot={slot} gpu={gpu} "
                f"key={key} error={exc}",
                file=log,
                flush=True,
            )
        return {**status, "rc": 2, "fail_closed": True, "error": str(exc)}
    if scene_manifest:
        env["INSCENE_MANIFEST"] = scene_manifest

    cmd = [
        PYTHON,
        "vidfm3d/eval_diag.py",
        f"experiment={row['experiment']}",
        f"vfm_name={row['vfm_name']}",
        f"video_channels={row['video_channels']}",
        f"streaming_feat_root={row['streaming_feat_root']}",
        f"feature_layer={row['feature_layer']}",
        f"feature_timestep={row['feature_timestep']}",
        f"prefix_len={row['prefix_len']}",
        f"job_name={out_name}",
        f"paths.run_folder_name={run_folder}",
        "ckpt_path=" + hydra_path(row["ckpt_path"]),
        "+eval_split=val",
        "train=false",
        "test=false",
        "data.data_module.num_workers_val=2",
        "data.data_module.batch_size_per_device_val=4",
        "logger.wandb.offline=true",
        "++bootstrap_reps=0",
    ]
    if row.get("feat_postfix"):
        cmd.append(f"feat_postfix={row['feat_postfix']}")
    if row.get("target_feat_root"):
        cmd.append(f"++target_feat_root={row['target_feat_root']}")
    extra = json.loads(row.get("extra_overrides_json") or "{}")
    for key_extra, value in sorted(extra.items()):
        cmd.append(f"{key_extra}={value}")

    start = time.time()
    with log_path.open("w") as log:
        print(
            f"START {datetime.utcnow().isoformat()}Z slot={slot} gpu={gpu} "
            f"key={key} ckpt={row['ckpt_path']}",
            file=log,
            flush=True,
        )
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(
            f"END {datetime.utcnow().isoformat()}Z rc={proc.returncode} elapsed={time.time() - start:.1f}s",
            file=log,
            flush=True,
        )
    return {**status, "rc": proc.returncode, "elapsed": time.time() - start}


def main() -> None:
    jobs = rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = OUT_DIR / "launcher" / f"{SHARD_TSV.stem}_status.jsonl"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a") as status:
        print(
            json.dumps(
                {
                    "event": "shard_start",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "run_id": RUN_ID,
                    "shard": str(SHARD_TSV),
                    "jobs": len(jobs),
                    "slots": SLOTS,
                    "gpus": GPUS,
                }
            ),
            file=status,
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=SLOTS) as executor:
            futures = [
                executor.submit(run_job, row, index % SLOTS)
                for index, row in enumerate(jobs)
            ]
            for future in futures:
                print(
                    json.dumps(
                        {
                            "event": "job_done",
                            "time": datetime.utcnow().isoformat() + "Z",
                            **future.result(),
                        }
                    ),
                    file=status,
                    flush=True,
                )
        print(
            json.dumps({"event": "shard_done", "time": datetime.utcnow().isoformat() + "Z"}),
            file=status,
            flush=True,
        )


if __name__ == "__main__":
    main()
