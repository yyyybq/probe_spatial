#!/usr/bin/env python3
"""Recover and load only the exact c8@300 HF actor export from AOSS."""
from __future__ import annotations

import argparse
import configparser
import importlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import sha256_file, write_json


ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
ADS = Path("/mnt/umm/users/yinbaiqiao/aoss/ads-cli")
CONF = Path("/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf")
RELATIVE = "VAGEN-Lite/exps/vagen_active_spatial/c8_fwdfirst_rewscale_lfp_server_v19/checkpoints/global_step_300/actor/huggingface"
HOST = "aoss-internal.cn-fz-01.fjscmsapi-oss.com"
BUCKET = "baiqiao"


def _credentials(conf: Path) -> tuple[str, str]:
    parser = configparser.RawConfigParser()
    parser.read(conf)
    access = parser.get("fj", "access_key", fallback="").strip()
    secret = parser.get("fj", "secret_key", fallback="").strip()
    if not access or not secret:
        raise RuntimeError("worker AOSS [fj] credentials unavailable")
    return access, secret


def _redact(text: str, access: str, secret: str) -> str:
    return re.sub(r"s3://[^/@:\\s]+:[^/@\\s]+@", "s3://<redacted>@", text.replace(access, "<redacted>").replace(secret, "<redacted>"))


def _verify_layout(stage: Path, *, full_shard_sha256: bool) -> dict[str, Any]:
    config = stage / "config.json"
    index = stage / "model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        raise RuntimeError("recovered c8 export lacks config.json or model.safetensors.index.json")
    weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("c8 safetensors index has no weight_map")
    names = sorted(set(weight_map.values()))
    missing = [name for name in names if not (stage / name).is_file() or (stage / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"c8 index references missing/empty shards: {missing[:5]}")
    value = json.loads(config.read_text(encoding="utf-8"))
    shards = [{"name": name, "bytes": (stage / name).stat().st_size} for name in names]
    if full_shard_sha256:
        for item in shards:
            item["sha256"] = sha256_file(stage / item["name"])
    return {"status": "INDEX_AND_COMPONENTS_VERIFIED", "verification_strength": "full_shard_sha256" if full_shard_sha256 else "index_referenced_nonzero_files", "config_sha256": sha256_file(config), "index_sha256": sha256_file(index), "model_type": value.get("model_type"), "architectures": value.get("architectures"), "indexed_tensor_count": len(weight_map), "required_shards": shards}


def _formal_load(stage: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ["CAMBRIAN_SRC"] = "/mnt/umm/users/yinbaiqiao/cambrian-s"
    sys.path.insert(0, str(ROOT))
    importlib.import_module("vagen.models.cambrian_register")
    tokenizer = AutoTokenizer.from_pretrained(stage, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(stage, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    norm = getattr(inner, "norm", None)
    if layers is None or norm is None:
        raise RuntimeError("formal Cambrian model loaded but lacks model.layers/model.norm probe boundaries")
    return {"status": "FORMAL_LOAD_PASSED", "model_class": f"{type(model).__module__}.{type(model).__name__}", "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}", "layers": len(layers), "has_norm": True, "cuda": torch.cuda.get_device_name(0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--reuse-staged", action="store_true", help="Use a prior task-owned staged cache; do not download it again.")
    parser.add_argument("--aoss-conf", type=Path, default=CONF)
    parser.add_argument("--ads-cli", type=Path, default=ADS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log = args.output_dir / "restore.log"
    if args.reuse_staged:
        if not args.stage_dir.is_dir():
            raise RuntimeError("requested reusable c8 stage directory does not exist")
        result = None
        log.write_text("reused prior task-owned c8 stage; no download performed\n", encoding="utf-8")
    else:
        args.stage_dir.mkdir(parents=True, exist_ok=False)
        access, secret = _credentials(args.aoss_conf)
        source = f"s3://{access}:{secret}@{BUCKET}.{HOST}/{RELATIVE}/"
        result = subprocess.run([str(args.ads_cli), "--quiet", "sync", source, str(args.stage_dir) + "/"], text=True, capture_output=True, check=False, timeout=7200)
        log.write_text(_redact(result.stdout + result.stderr, access, secret), encoding="utf-8")
        if result.returncode != 0:
            report = {"status": "RESTORE_FAILED", "identity": "c8_fwdfirst_rewscale_lfp_server_v19 global_step_300 actor/huggingface", "aoss_uri": f"s3://{BUCKET}/{RELATIVE}/", "stage_dir": str(args.stage_dir), "restore_returncode": result.returncode, "restore_log": str(log)}
            write_json(args.output_dir / "c8_recovery.json", report)
            raise RuntimeError("AOSS restore command failed")
    report: dict[str, Any] = {"created_at": datetime.now(timezone.utc).isoformat(), "identity": "c8_fwdfirst_rewscale_lfp_server_v19 global_step_300 actor/huggingface", "aoss_uri": f"s3://{BUCKET}/{RELATIVE}/", "stage_dir": str(args.stage_dir), "restore_returncode": None if result is None else result.returncode, "restore_log": str(log), "reused_staged_cache": bool(args.reuse_staged)}
    report["layout"] = _verify_layout(args.stage_dir, full_shard_sha256=not args.reuse_staged)
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("c8 staged and verified, but worker has no CUDA device for formal load")
    report["formal_load"] = _formal_load(args.stage_dir)
    report["status"] = "LOAD_PASSED"
    write_json(args.output_dir / "c8_recovery.json", report)
    print(json.dumps({"status": report["status"], "shards": len(report["layout"]["required_shards"]), "layers": report["formal_load"]["layers"]}))


if __name__ == "__main__":
    main()
