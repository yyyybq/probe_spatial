"""Reproducibility metadata for training and evaluation runs."""

from __future__ import annotations

import json
import os
import platform
import shutil
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


def _sha256(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _realpath(path: str | None) -> str | None:
    if not path:
        return None
    if str(path).startswith(("aoss://", "s3://", "fj:s3://")):
        return str(path)
    return str(Path(path).expanduser().resolve(strict=False))


def _feature_root_record(name: str, path: str | None) -> dict[str, str | bool | None]:
    real = _realpath(path)
    exists = None
    is_symlink = None
    if path and not str(path).startswith(("aoss://", "s3://", "fj:s3://")):
        p = Path(path)
        exists = p.exists()
        is_symlink = p.is_symlink()
    return {
        "name": name,
        "path": path,
        "realpath": real,
        "exists": exists,
        "is_symlink": is_symlink,
    }


def write_run_manifest(cfg, output_dir: str) -> Path | None:
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return None
    root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    status = _git(root, "status", "--short")
    split_manifest = os.environ.get("INSCENE_SPLIT_MANIFEST")
    scene_manifest = os.environ.get("INSCENE_MANIFEST") or split_manifest
    split_digest = _sha256(split_manifest)
    scene_digest = _sha256(scene_manifest)
    feature_roots = [
        _feature_root_record("streaming", os.environ.get("INSCENE_STREAMING_FEAT_ROOT")),
        _feature_root_record("feature", os.environ.get("INSCENE_FEAT_ROOT")),
        _feature_root_record("context", os.environ.get("INSCENE_CONTEXT_FEAT_ROOT")),
        _feature_root_record("target", os.environ.get("INSCENE_TARGET_FEAT_ROOT")),
        _feature_root_record("shuffled", os.environ.get("INSCENE_SHUFFLED_FEAT_ROOT")),
    ]
    extractor = {
        "vfm_name": cfg.get("vfm_name"),
        "feature_layer": cfg.get("feature_layer"),
        "feature_timestep": cfg.get("feature_timestep"),
        "feat_postfix": cfg.get("feat_postfix"),
        "prefix_len": cfg.get("prefix_len"),
        "video_channels": cfg.get("video_channels"),
        "streaming_feat_root": cfg.get("streaming_feat_root"),
        "target_feat_root": cfg.get("target_feat_root"),
    }
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
            "scene_manifest": scene_manifest,
            "scene_manifest_sha256": scene_digest,
            "scene_manifest_realpath": _realpath(scene_manifest),
            "data_root": os.environ.get("INSCENE_DATA_ROOT"),
            "feature_root": os.environ.get("INSCENE_FEAT_ROOT"),
            "context_feature_root": os.environ.get("INSCENE_CONTEXT_FEAT_ROOT"),
            "target_feature_root": os.environ.get("INSCENE_TARGET_FEAT_ROOT"),
            "shuffled_feature_root": os.environ.get("INSCENE_SHUFFLED_FEAT_ROOT"),
            "feature_roots": feature_roots,
        },
        "extractor": extractor,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    path = Path(output_dir) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)

    provenance_dir = Path(output_dir) / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = provenance_dir / "run_provenance.json"
    provenance_tmp = provenance_path.with_name(f".{provenance_path.name}.tmp-{os.getpid()}")
    provenance_tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(provenance_tmp, provenance_path)
    if scene_manifest and Path(scene_manifest).is_file():
        shutil.copy2(scene_manifest, provenance_dir / "manifest_snapshot.jsonl")
    git_snapshot = {
        "commit": payload["git"]["commit"],
        "branch": payload["git"]["branch"],
        "dirty": payload["git"]["dirty"],
        "status": payload["git"]["status"],
        "diff": _git(root, "diff", "--"),
    }
    git_path = provenance_dir / "git_snapshot.json"
    git_tmp = git_path.with_name(f".{git_path.name}.tmp-{os.getpid()}")
    git_tmp.write_text(json.dumps(git_snapshot, indent=2, default=str) + "\n")
    os.replace(git_tmp, git_path)
    return path
