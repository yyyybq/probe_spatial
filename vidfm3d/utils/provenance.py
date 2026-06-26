"""Reproducibility metadata for training and evaluation runs."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()
    except Exception:
        return "unknown"


def write_run_manifest(cfg, output_dir: str) -> Path | None:
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return None
    root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    status = _git(root, "status", "--short")
    split_manifest = os.environ.get("INSCENE_SPLIT_MANIFEST")
    split_digest = None
    if split_manifest and Path(split_manifest).is_file():
        import hashlib
        split_digest = hashlib.sha256(Path(split_manifest).read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "git": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "branch", "--show-current"),
            "dirty": bool(status and status != "unknown"),
            "status": status.splitlines(),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "scheduler": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_NODELIST", "SLURM_PROCID", "SLURM_NTASKS",
            )
            if os.environ.get(key) is not None
        },
        "data": {
            "split_manifest": split_manifest,
            "split_manifest_sha256": split_digest,
            "data_root": os.environ.get("INSCENE_DATA_ROOT"),
            "feature_root": os.environ.get("INSCENE_FEAT_ROOT"),
            "context_feature_root": os.environ.get("INSCENE_CONTEXT_FEAT_ROOT"),
            "target_feature_root": os.environ.get("INSCENE_TARGET_FEAT_ROOT"),
            "shuffled_feature_root": os.environ.get("INSCENE_SHUFFLED_FEAT_ROOT"),
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    path = Path(output_dir) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)
    return path
