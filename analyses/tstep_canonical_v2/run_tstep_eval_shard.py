#!/usr/bin/env python3
"""Run one shard of official tstep A1/A2/B2 held-out eval jobs."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
PYTHON = os.environ.get("PYTHON_BIN", "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python")
RUN_ID = os.environ["RUN_ID"]
SHARD_TSV = Path(os.environ["SHARD_TSV"])
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT / f"artifacts/{RUN_ID}"))
SLOTS = int(os.environ.get("SLOTS", "8"))
GPUS = [x for x in os.environ.get("GPU_IDS_CSV", "0,1,2,3,4,5,6,7").split(",") if x]
DATA_ROOT = os.environ.get("INSCENE_DATA_ROOT", "/mnt/umm/users/yinbaiqiao/InsScene-15K/data")


def hydra_path(path: str) -> str:
    return path.replace("=", r"\=")


def rows() -> list[dict[str, str]]:
    with SHARD_TSV.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def resolve_scene_manifest(row: dict[str, str]) -> str:
    raw = Path(row["scene_manifest"])
    candidate = raw if raw.is_absolute() else ROOT / raw
    if not candidate.is_file():
        raise FileNotFoundError(f"missing scene_manifest for {row['job_key']}: {raw}")
    return str(candidate)


def run_job(row: dict[str, str], slot: int) -> dict[str, object]:
    gpu = GPUS[slot % len(GPUS)]
    key = row["job_key"]
    out_name = f"{RUN_ID}_{key}"
    run_folder = f"{RUN_ID}/{out_name}"
    summary = ROOT / "logs/inscene15k_streaming/runs" / run_folder / "eval/val_summary.json"
    log_dir = ROOT / "logs/inscene15k_streaming/eval_tstep_jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{RUN_ID}_slot{slot}_{key}.log"
    status = {
        "key": key,
        "unique_key": row["unique_key"],
        "probe": row["probe"],
        "evaluator": row["evaluator"],
        "metric_name": row["metric_name"],
        "slot": slot,
        "gpu": gpu,
        "summary": str(summary),
        "log": str(log_path),
    }
    if summary.is_file():
        try:
            existing = json.loads(summary.read_text())
            n = int(existing.get("n_samples", existing.get("n", 0)))
            if n > 0 and row["metric_name"] in existing:
                return {**status, "rc": 0, "skipped": True}
        except Exception:
            pass
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT),
        "PROJECT_ROOT": str(ROOT),
        "CUDA_VISIBLE_DEVICES": gpu,
        "INSCENE_DATA_ROOT": DATA_ROOT,
        "INSCENE_STREAMING_FEAT_ROOT": row["streaming_feat_root"],
        "INSCENE_TRUST_EXPANDED_MANIFEST": "1",
        "PROBE_FP16_PASSTHROUGH": "1",
        "AOSS_CACHE_DIR": f"/mnt/local_cache/inscene/aoss_cache/{RUN_ID}/{key}",
        "MPLCONFIGDIR": f"/tmp/tstep-eval-mpl-{RUN_ID}-{slot}",
    })
    try:
        scene_manifest = resolve_scene_manifest(row)
    except FileNotFoundError as exc:
        with log_path.open("w") as log:
            print(f"FAIL_CLOSED {datetime.now(timezone.utc).isoformat()} {exc}", file=log)
        return {**status, "rc": 2, "fail_closed": True, "error": str(exc)}
    env["INSCENE_MANIFEST"] = scene_manifest
    cmd = [
        PYTHON,
        row["entrypoint"],
        f"experiment={row['experiment']}",
        f"vfm_name={row['vfm_name']}",
        f"video_channels={row['video_channels']}",
        f"streaming_feat_root={row['streaming_feat_root']}",
        f"feature_layer={row['feature_layer']}",
        f"feature_timestep={row['feature_timestep']}",
        f"prefix_len={row['prefix_len']}",
        f"feat_postfix={row['feat_postfix']}",
        f"job_name={out_name}",
        f"paths.run_folder_name={run_folder}",
        "ckpt_path=" + hydra_path(row["ckpt_path"]),
        "+eval_split=val",
        "train=false",
        "test=false",
        "data.data_module.num_workers_val=2",
        "data.data_module.batch_size_per_device_val=4",
        "logger.wandb.offline=true",
    ]
    if row["evaluator"] == "diag":
        cmd.append("++bootstrap_reps=0")
    extra = json.loads(row.get("extra_overrides_json") or "{}")
    for key_extra, value in sorted(extra.items()):
        cmd.append(f"{key_extra}={value}")
    start = time.time()
    with log_path.open("w") as log:
        print(
            f"START {datetime.now(timezone.utc).isoformat()} slot={slot} gpu={gpu} "
            f"key={key} entry={row['entrypoint']} ckpt={row['ckpt_path']}",
            file=log,
            flush=True,
        )
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(
            f"END {datetime.now(timezone.utc).isoformat()} rc={proc.returncode} "
            f"elapsed={time.time() - start:.1f}s",
            file=log,
            flush=True,
        )
    return {**status, "rc": proc.returncode, "elapsed": time.time() - start}


def main() -> None:
    jobs = rows()
    status_path = OUT_DIR / "launcher" / f"{SHARD_TSV.stem}_status.jsonl"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a") as status:
        print(json.dumps({
            "event": "shard_start",
            "time": datetime.now(timezone.utc).isoformat(),
            "run_id": RUN_ID,
            "shard": str(SHARD_TSV),
            "jobs": len(jobs),
            "slots": SLOTS,
            "gpus": GPUS,
        }), file=status, flush=True)
        with ThreadPoolExecutor(max_workers=SLOTS) as executor:
            futures = [executor.submit(run_job, row, index % SLOTS) for index, row in enumerate(jobs)]
            for future in futures:
                print(json.dumps({
                    "event": "job_done",
                    "time": datetime.now(timezone.utc).isoformat(),
                    **future.result(),
                }), file=status, flush=True)
        print(json.dumps({
            "event": "shard_done",
            "time": datetime.now(timezone.utc).isoformat(),
        }), file=status, flush=True)


if __name__ == "__main__":
    main()
