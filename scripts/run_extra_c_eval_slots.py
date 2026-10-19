#!/usr/bin/env python3
"""Run extra C-series eval jobs for VideoMAE and Qwen2.5-VL checkpoints."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
PYTHON = os.environ.get(
    "PYTHON_BIN", "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
)
RUN_TAG = os.environ["RUN_TAG"]
SLOT_COUNT = int(os.environ["SLOT_COUNT"])
SLOT_IDS = [int(x) for x in os.environ["SLOT_IDS"].split(",") if x]
GPUS = [x for x in os.environ["GPUS"].split(",") if x]

DATA_ROOT = "fj:s3://baiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2"
LOG_DIR = ROOT / "logs/inscene15k_streaming/eval_jobs"
MANIFEST_ROOT = ROOT / "logs/inscene15k_streaming/manifests/c_retrain_20260725_extra_fast"
STREAM_ROOT_BASE = "fj:s3://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX"
TARGET_ROOT_BASE = "fj:s3://baiqiao/InsScene-15K/FEAT_TARGET_BY_PREFIX"
PROBES = ["action_dynamics", "path_integration", "counterfactual"]
SOURCES = [
    ("c_retrain_extra_videomae_qwen_localfeat_20260725_1826", {"videomae"}),
    ("qwen_resume_20260726_1755", {"qwen2_5_vl"}),
]


def parse_key(key: str) -> tuple[str, str, int, str]:
    for probe in PROBES:
        prefix = probe + "_"
        if key.startswith(prefix):
            rem = key[len(prefix) :]
            match = re.match(r"(.+)_p(\d+)_layer(.+)$", rem)
            if not match:
                raise ValueError(f"Cannot parse eval key: {key}")
            return probe, match.group(1), int(match.group(2)), match.group(3)
    raise ValueError(f"Unknown probe in eval key: {key}")


def read_overrides(run_dir: Path) -> dict[str, str]:
    overrides: dict[str, str] = {}
    overrides_path = run_dir / ".hydra/overrides.yaml"
    for line in overrides_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("- ") and "=" in line:
            key, value = line[2:].split("=", 1)
            overrides[key] = value
    return overrides


def checkpoint_step(path: Path) -> tuple[int, float]:
    match = re.search(r"step=(\d+)", path.name)
    step = int(match.group(1)) if match else -1
    return step, path.stat().st_mtime


def selected_jobs() -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for run, allowed_vfms in SOURCES:
        root = ROOT / "logs/inscene15k_streaming/runs" / run
        if not root.exists():
            continue
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            prefix = run + "_"
            if not run_dir.name.startswith(prefix):
                continue
            key = run_dir.name[len(prefix) :]
            probe, vfm, prefix_len, layer = parse_key(key)
            if vfm not in allowed_vfms:
                continue
            ckpts = sorted(run_dir.glob("VideoProbe3D/*/checkpoints/*.ckpt"))
            if not ckpts:
                continue
            manifest = MANIFEST_ROOT / f"{vfm}_c_h124_prefix{prefix_len}.jsonl"
            if not manifest.exists():
                raise FileNotFoundError(f"Missing manifest for {key}: {manifest}")
            jobs.append(
                {
                    "key": key,
                    "probe": probe,
                    "vfm": vfm,
                    "prefix": prefix_len,
                    "layer": layer,
                    "source_run": run,
                    "run_dir": run_dir,
                    "ckpt": max(ckpts, key=checkpoint_step),
                    "manifest": manifest,
                    "overrides": read_overrides(run_dir),
                }
            )
    return sorted(jobs, key=lambda item: str(item["key"]))


def hydra_path(path: Path) -> str:
    return str(path).replace("=", r"\=")


def aoss_stream_root(vfm: str, prefix: int) -> str:
    return f"{STREAM_ROOT_BASE}/{vfm}_prefix_{prefix}_m0p35_r0p5"


def aoss_target_root(vfm: str, prefix: int) -> str:
    return f"{TARGET_ROOT_BASE}/{vfm}_prefix_{prefix}_m0p35_r0p5"


def run_job(job: dict[str, object], gpu: str, slot_id: int) -> dict[str, object]:
    key = str(job["key"])
    probe = str(job["probe"])
    vfm = str(job["vfm"])
    prefix = int(job["prefix"])
    layer = str(job["layer"])
    overrides = dict(job["overrides"])  # type: ignore[arg-type]
    out_name = f"{RUN_TAG}_{key}"
    out_dir = ROOT / "logs/inscene15k_streaming/runs" / RUN_TAG / out_name
    summary = out_dir / "eval/val_summary.json"
    log_path = LOG_DIR / f"{RUN_TAG}_slot{slot_id}_{key}.log"

    if summary.exists():
        try:
            existing = json.loads(summary.read_text())
            if int(existing.get("n", 0)) > 0:
                return {"key": key, "rc": 0, "skipped": True, "summary": str(summary)}
        except Exception:
            pass

    streaming_root = aoss_stream_root(vfm, prefix)
    target_root = aoss_target_root(vfm, prefix)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PROJECT_ROOT": str(ROOT),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "INSCENE_DATA_ROOT": DATA_ROOT,
            "INSCENE_MANIFEST": str(job["manifest"]),
            "INSCENE_STREAMING_FEAT_ROOT": streaming_root,
            "INSCENE_TARGET_FEAT_ROOT": target_root,
            "AOSS_CACHE_DIR": f"/mnt/local_cache/inscene/aoss_cache/{RUN_TAG}/{key}",
        }
    )
    cmd = [
        PYTHON,
        "vidfm3d/eval_diag.py",
        f"experiment={overrides.get('experiment', f'inscene15k_streaming/{probe}_{vfm}_v1')}",
        f"vfm_name={overrides.get('vfm_name', vfm)}",
        f"video_channels={overrides['video_channels']}",
        f"streaming_feat_root={streaming_root}",
        f"feature_layer={overrides.get('feature_layer', layer)}",
        f"feature_timestep={overrides.get('feature_timestep', '749')}",
        f"prefix_len={prefix}",
        f"target_feat_root={target_root}",
        f"job_name={out_name}",
        f"paths.run_folder_name={RUN_TAG}/{out_name}",
        "ckpt_path=" + hydra_path(Path(job["ckpt"]).resolve()),  # type: ignore[arg-type]
        "+eval_split=val",
        "train=false",
        "test=false",
        "data.data_module.num_workers_val=2",
        "data.data_module.batch_size_per_device_val=4",
        "logger.wandb.offline=true",
    ]

    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        print(
            f"START {datetime.utcnow().isoformat()}Z gpu={gpu} slot={slot_id} "
            f"key={key} ckpt={job['ckpt']} manifest={job['manifest']}",
            file=log,
            flush=True,
        )
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(
            f"END {datetime.utcnow().isoformat()}Z rc={proc.returncode} "
            f"elapsed={time.time() - start:.1f}s",
            file=log,
            flush=True,
        )

    return {
        "key": key,
        "rc": proc.returncode,
        "summary": str(summary),
        "log": str(log_path),
        "elapsed": time.time() - start,
    }


def worker(slot_id: int, gpu: str) -> None:
    jobs = [job for index, job in enumerate(selected_jobs()) if index % SLOT_COUNT == slot_id]
    status_path = LOG_DIR / f"{RUN_TAG}_slot{slot_id}_status.jsonl"
    with status_path.open("a") as status:
        print(
            json.dumps(
                {
                    "event": "slot_start",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "slot": slot_id,
                    "gpu": gpu,
                    "jobs": len(jobs),
                }
            ),
            file=status,
            flush=True,
        )
        for job in jobs:
            result = run_job(job, gpu, slot_id)
            print(
                json.dumps({"event": "job_done", "time": datetime.utcnow().isoformat() + "Z", **result}),
                file=status,
                flush=True,
            )
        print(
            json.dumps({"event": "slot_done", "time": datetime.utcnow().isoformat() + "Z", "slot": slot_id}),
            file=status,
            flush=True,
        )


def main() -> None:
    if len(SLOT_IDS) != len(GPUS):
        raise SystemExit(f"SLOT_IDS and GPUS length mismatch: {SLOT_IDS} vs {GPUS}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = selected_jobs()
    if len(jobs) != 24:
        raise SystemExit(f"Expected 24 extra eval jobs, found {len(jobs)}")
    with ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        futures = [executor.submit(worker, slot_id, gpu) for slot_id, gpu in zip(SLOT_IDS, GPUS)]
        for future in futures:
            future.result()
    print(f"DONE {RUN_TAG} slots={SLOT_IDS}")


if __name__ == "__main__":
    main()
