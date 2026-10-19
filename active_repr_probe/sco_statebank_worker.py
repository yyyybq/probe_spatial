#!/usr/bin/env python3
"""Dedicated real-renderer state-bank worker for the 48-state canary.

This intentionally contains no model loading or checkpoint recovery.  It
first produces one genuine state to audit visibility/occlusion and branch
restoration, then uses the same frozen task subset and scene cache to produce
exactly 48 states.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .build_state_bank import build
from .io_utils import read_jsonl, sha256_file, stable_hash, write_json
from .protocol import freeze_protocol, validate_state_bank


ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
ADS = Path("/mnt/umm/users/yinbaiqiao/aoss/ads-cli")
CONF = Path("/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf")
HOST = "aoss-internal.cn-fz-01.fjscmsapi-oss.com"
BUCKET = "baiqiao"


def _source_manifest() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    names = ["sco_statebank_worker.py", "build_state_bank.py", "geometry.py", "protocol.py", "io_utils.py"]
    return {name: sha256_file(package / name) for name in names}


def _bind_verified_renderer_runtime() -> None:
    """Bind the same prebuilt gsplat/runtime paths as the successful SCO entry."""
    vagen_env = "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite"
    os.environ.setdefault("CAMBRIAN_SRC", "/mnt/umm/users/yinbaiqiao/cambrian-s")
    os.environ.setdefault("CUDA_HOME", vagen_env)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer")
    os.environ.setdefault("VAGEN_GSPLAT_PREBUILT", "1")
    os.environ["PATH"] = f"{vagen_env}/bin:{os.environ.get('PATH', '')}"
    os.environ["CPATH"] = f"{vagen_env}/targets/x86_64-linux/include:{os.environ.get('CPATH', '')}"
    os.environ["CPLUS_INCLUDE_PATH"] = f"{vagen_env}/targets/x86_64-linux/include:{os.environ.get('CPLUS_INCLUDE_PATH', '')}"
    os.environ["LIBRARY_PATH"] = f"{vagen_env}/targets/x86_64-linux/lib:{os.environ.get('LIBRARY_PATH', '')}"


def _credentials(conf: Path) -> tuple[str, str]:
    parser = configparser.RawConfigParser()
    parser.read(conf)
    access = parser.get("fj", "access_key", fallback="").strip()
    secret = parser.get("fj", "secret_key", fallback="").strip()
    if not access or not secret:
        raise RuntimeError("worker AOSS configuration has no usable [fj] credentials")
    return access, secret


def _redact(text: str, access: str, secret: str) -> str:
    return re.sub(r"s3://[^/@:\\s]+:[^/@\\s]+@", "s3://<redacted>@", text.replace(access, "<redacted>").replace(secret, "<redacted>"))


def _upload(out: Path, prefix: str, conf: Path, ads: Path) -> dict[str, Any]:
    access, secret = _credentials(conf)
    destination = f"s3://{access}:{secret}@{BUCKET}.{HOST}/{prefix.strip('/')}/"
    result = subprocess.run([str(ads), "--quiet", "sync", str(out) + "/", destination], text=True, capture_output=True, check=False, timeout=1800)
    log = out / "aoss_upload.log"
    safe = _redact(result.stdout + result.stderr, access, secret)
    log.write_text(safe, encoding="utf-8")
    return {"status": "UPLOADED" if result.returncode == 0 else "UPLOAD_FAILED", "returncode": result.returncode, "aoss_uri": f"s3://{BUCKET}/{prefix.strip('/')}/", "log": str(log)}


def _args(out: Path, seeds: Path, task_subset: Path, scene_cache: Path, target_rows: int, steps_per_episode: int) -> SimpleNamespace:
    return SimpleNamespace(
        vagen_root=ROOT,
        env_config=ROOT / "examples/train/active_spatial/env_config_h800_7b_6types.yaml",
        output_dir=out,
        episode_seeds_file=seeds,
        start_seed=0,
        episodes=0,
        steps_per_episode=steps_per_episode,
        target_rows=target_rows,
        max_per_scene=48,
        seed=41,
        split_seed=20260018,
        enable_visibility_labels=True,
        enable_collision_labels=True,
        jsonl_path=task_subset,
        gs_root=scene_cache,
        render_backend="local",
        gpu_device=0,
    )


def _semantic_check(bank: Path, expected_rows: int | None = None) -> dict[str, Any]:
    rows = list(read_jsonl(bank / "states.jsonl"))
    errors: list[str] = []
    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")
    for row in rows:
        sid = row.get("sample_id", "<missing>")
        history = row.get("history", [])
        if len(history) != 3 or [item.get("env_step") for item in history] != list(range(history[0].get("env_step", 0), history[0].get("env_step", 0) + 3)) if history else True:
            errors.append(f"{sid}: non-contiguous W=3 history")
        for item in history:
            image = bank / item.get("image_path", "")
            if not image.is_file() or sha256_file(image) != item.get("image_sha256"):
                errors.append(f"{sid}: missing or mismatched RGB image")
        visibility = row.get("labels", {}).get("visibility", {})
        required = {"status", "in_fov", "is_visible", "occlusion_ratio", "occlusion_type", "screen_coverage"}
        if visibility.get("status") != "VERIFIED" or not required <= set(visibility):
            errors.append(f"{sid}: visibility/occlusion not a checked renderer result")
        branches = row.get("counterfactual_actions", [])
        if not branches or not all(branch.get("state_restored_exact") is True for branch in branches):
            errors.append(f"{sid}: a counterfactual branch did not restore the exact state")
    return {
        "valid": not errors,
        "rows": len(rows),
        "errors": errors,
        "state_bank_validation": validate_state_bank(bank / "states.jsonl"),
        "sample_id_hash": stable_hash([row.get("sample_id") for row in rows]),
        "occlusion_types": sorted({str(row["labels"]["visibility"].get("occlusion_type")) for row in rows if row.get("labels", {}).get("visibility", {}).get("status") == "VERIFIED"}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-state-subset", type=Path, required=True)
    parser.add_argument("--scene-cache", type=Path, required=True)
    parser.add_argument("--upload-prefix", required=True)
    parser.add_argument("--aoss-conf", type=Path, default=CONF)
    parser.add_argument("--ads-cli", type=Path, default=ADS)
    args = parser.parse_args()
    _bind_verified_renderer_runtime()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    if not args.source_state_subset.is_file() or not args.scene_cache.is_dir():
        raise RuntimeError("frozen task subset or previously verified scene cache is unavailable")
    write_json(out / "code_identity.json", {"created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": _source_manifest(), "python": sys.executable, "hostname": os.uname().nodename})
    subset_text = args.source_state_subset.read_text(encoding="utf-8")
    task_subset = out / "frozen_task_subset.jsonl"
    task_subset.write_text(subset_text, encoding="utf-8")
    write_json(out / "input_identity.json", {"source_task_subset": str(args.source_state_subset), "task_subset_sha256": sha256_file(task_subset), "scene_cache": str(args.scene_cache)})
    smoke_seeds = out / "semantic_smoke_seeds.txt"
    smoke_seeds.write_text("0\n1\n2\n3\n", encoding="utf-8")
    smoke = out / "semantic_smoke"
    smoke_meta = build(_args(smoke, smoke_seeds, task_subset, args.scene_cache, 1, 5))
    smoke_check = _semantic_check(smoke, expected_rows=1)
    write_json(out / "semantic_smoke_validation.json", {"builder": smoke_meta, **smoke_check})
    if not smoke_check["valid"]:
        raise RuntimeError(f"real semantic smoke failed: {smoke_check['errors'][:3]}")
    seeds = out / "episode_seeds.txt"
    # The frozen 16-row task subset is reset deterministically by these seeds.
    # Ten real rollout observations per episode leave enough W=3 windows to
    # retain exactly 48 rows despite legitimate early terminations.
    seeds.write_text("".join(f"{seed}\n" for seed in range(16)), encoding="utf-8")
    state_bank = out / "state_bank"
    state_meta = build(_args(state_bank, seeds, task_subset, args.scene_cache, 48, 10))
    state_check = _semantic_check(state_bank, expected_rows=48)
    freeze_protocol(state_bank, {"state_bank_source": "real_active_spatial_renderer", "label_version": "active_repr_geometry_v1", "frozen_task_subset_sha256": sha256_file(task_subset), "state_bank_hash": state_meta.get("state_bank_hash")})
    write_json(out / "state_bank_result.json", {"builder": state_meta, **state_check, "status": "COMPLETE" if state_check["valid"] else "FAILED"})
    if not state_check["valid"]:
        raise RuntimeError(f"48-state semantic gate failed: {state_check['errors'][:3]}")
    receipt = _upload(out, args.upload_prefix, args.aoss_conf, args.ads_cli)
    write_json(out / "aoss_upload_receipt.json", receipt)
    if receipt["status"] != "UPLOADED":
        raise RuntimeError("state bank is valid locally but AOSS artifact upload failed")
    print(json.dumps({"status": "COMPLETE", "rows": state_check["rows"], "sample_id_hash": state_check["sample_id_hash"]}))


if __name__ == "__main__":
    main()
