#!/usr/bin/env python3
"""Run C-series eval jobs assigned to one or more global slots.

The script discovers retrained checkpoints from the safe/rescue C retraining
runs, deduplicates by experiment key, prefers rescue checkpoints, and launches
`vidfm3d/eval_diag.py` for the slots requested through the environment.
"""

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

LOG_DIR = ROOT / "logs/inscene15k_streaming/eval_jobs"
DATA_ROOT = "fj:s3://baiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2"
PROBES = ["action_dynamics", "path_integration", "counterfactual"]
RUNS = ["c_retrain_safe_20260722_1435", "c_retrain_rescue_20260723_1748"]


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
    for line in (run_dir / ".hydra/overrides.yaml").read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("- ") and "=" in line:
            key, value = line[2:].split("=", 1)
            overrides[key] = value
    return overrides


def selected_jobs() -> list[dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    for run in RUNS:
        root = ROOT / "logs/inscene15k_streaming/runs" / run
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            prefix = run + "_"
            if not run_dir.name.startswith(prefix):
                continue
            key = run_dir.name[len(prefix) :]
            ckpts = sorted(run_dir.glob("VideoProbe3D/*/checkpoints/*.ckpt"))
            if not ckpts:
                continue
            if key not in selected or "rescue" in run:
                selected[key] = {
                    "key": key,
                    "source_run": run,
                    "run_dir": run_dir,
                    "ckpt": ckpts[-1],
                    "overrides": read_overrides(run_dir),
                }
    return [selected[key] for key in sorted(selected)]


def hydra_path(path: Path) -> str:
    return str(path).replace("=", r"\=")


def run_job(job: dict[str, object], gpu: str, slot_id: int) -> dict[str, object]:
    key = str(job["key"])
    probe, vfm, prefix, layer_from_name = parse_key(key)
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

    manifest = ROOT / "manifests/c_retrain_20260722" / f"inscene_c_h124_prefix{prefix}.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PROJECT_ROOT": str(ROOT),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "INSCENE_DATA_ROOT": DATA_ROOT,
            "INSCENE_MANIFEST": str(manifest),
            "INSCENE_STREAMING_FEAT_ROOT": overrides.get("streaming_feat_root", ""),
            "INSCENE_TARGET_FEAT_ROOT": overrides.get("target_feat_root", ""),
        }
    )
    cmd = [
        PYTHON,
        "vidfm3d/eval_diag.py",
        f"experiment={overrides.get('experiment', f'inscene15k_streaming/{probe}_{vfm}_v1')}",
        f"vfm_name={overrides.get('vfm_name', vfm)}",
        f"video_channels={overrides['video_channels']}",
        f"streaming_feat_root={overrides['streaming_feat_root']}",
        f"feature_layer={overrides.get('feature_layer', layer_from_name)}",
        f"feature_timestep={overrides.get('feature_timestep', '749')}",
        f"prefix_len={prefix}",
        f"target_feat_root={overrides['target_feat_root']}",
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
    with log_path.open("w") as log:
        print(
            f"START {datetime.utcnow().isoformat()}Z gpu={gpu} slot={slot_id} "
            f"key={key} ckpt={job['ckpt']}",
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
    with ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        futures = [executor.submit(worker, slot_id, gpu) for slot_id, gpu in zip(SLOT_IDS, GPUS)]
        for future in futures:
            future.result()
    print(f"DONE {RUN_TAG} slots={SLOT_IDS}")


if __name__ == "__main__":
    main()
