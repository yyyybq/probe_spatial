#!/usr/bin/env python3
"""Run a bounded, evidence-producing Active Spatial representation canary on SCO.

This is deliberately a *worker* entrypoint.  It uses the already deployed
SCO image/mount and VAGEN's native ``fj:s3://`` InteriorGS implementation;
the submitter never injects credentials.  Remote checkpoint candidates are
only queried under the formal archive layout established by the existing
restore script.  A failed candidate query is recorded as such and is never
replaced by a similarly named checkpoint.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .build_state_bank import scene_split
from .io_utils import stable_hash, write_json


ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
DEFAULT_CONF = Path("/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf")
DEFAULT_ADS = Path("/mnt/umm/users/yinbaiqiao/aoss/ads-cli")
HOST = "aoss-internal.cn-fz-01.fjscmsapi-oss.com"
BUCKET = "baiqiao"
REMOTE_GS_ROOT = "fj:s3://baiqiao/InteriorGS"

MODELS = {
    "qwen_pretrain_baseline": {
        "identity": "Qwen/Qwen2.5-VL-7B-Instruct (registered snapshot cc594898137f460bfe9f0759e9844b3ce807cfb5)",
        "backend": "qwen2_5_vl",
        "path": "/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "aoss_relative": None,
    },
    "v46_step200": {
        "identity": "v46_7b_nodelta_w3_rerun global_step_200 actor/huggingface",
        "backend": "qwen2_5_vl",
        "path": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/v46_7b_nodelta_w3_rerun/checkpoints/global_step_200/actor/huggingface",
        "aoss_relative": "VAGEN-Lite/exps/vagen_active_spatial/v46_7b_nodelta_w3_rerun/checkpoints/global_step_200/actor/huggingface",
    },
    "v50_step150": {
        "identity": "v50_7b_w3_format001_stable global_step_150 frozen HF export",
        "backend": "qwen2_5_vl",
        "path": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/unified_eval_runs/frozen_ckpts/v50_step150",
        "aoss_relative": "VAGEN-Lite/exps/unified_eval_runs/frozen_ckpts/v50_step150",
    },
    "cambrian_pretrain_baseline": {
        "identity": "Cambrian-S-7B-LFP pretrain baseline (c8 init)",
        "backend": "cambrians",
        "path": "/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP",
        "aoss_relative": None,
    },
    "c8_step300": {
        "identity": "c8_fwdfirst_rewscale_lfp_server_v19 global_step_300 actor/huggingface",
        "backend": "cambrians",
        "path": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/c8_fwdfirst_rewscale_lfp_server_v19/checkpoints/global_step_300/actor/huggingface",
        "aoss_relative": "VAGEN-Lite/exps/vagen_active_spatial/c8_fwdfirst_rewscale_lfp_server_v19/checkpoints/global_step_300/actor/huggingface",
    },
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_layout(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"status": "NOT_STAGED", "reason": "path_not_present_on_worker"}
    config = path / "config.json"
    weights = sorted(path.glob("*.safetensors")) + sorted(path.glob("pytorch_model*.bin"))
    indexes = sorted(path.glob("*.index.json"))
    files = [{"name": p.name, "bytes": p.stat().st_size} for p in [config, *indexes, *weights] if p.is_file()]
    return {
        "status": "OBJECTS_VERIFIED" if config.is_file() and (weights or indexes) else "INCOMPLETE",
        "config_present": config.is_file(),
        "weight_or_index_present": bool(weights or indexes),
        "files": files,
    }


def _credentials(conf: Path) -> tuple[str, str]:
    parser = configparser.RawConfigParser()
    parser.read(conf)
    access = parser.get("fj", "access_key", fallback="").strip()
    secret = parser.get("fj", "secret_key", fallback="").strip()
    if not access or not secret:
        raise RuntimeError("[fj] AOSS credentials unavailable in worker-injected configuration")
    return access, secret


def _redact(text: str, access: str, secret: str) -> str:
    text = text.replace(access, "<redacted>").replace(secret, "<redacted>")
    return re.sub(r"s3://[^/@:\\s]+:[^/@\\s]+@", "s3://<redacted>@", text)


def _archive_dryrun(relative: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """Target exactly one established archive prefix without copying objects."""
    access, secret = _credentials(args.aoss_conf)
    source = f"s3://{access}:{secret}@{BUCKET}.{HOST}/{relative.strip('/')}/"
    log = out_dir / "aoss_query" / (relative.replace("/", "__") + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.ads_cli), "--dryrun", "--depth", "1", "sync", source, str(out_dir / ".aoss_dryrun_sink") + "/"]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=180, check=False)
        safe = _redact(completed.stdout + completed.stderr, access, secret)
        log.write_text(safe, encoding="utf-8")
        found = re.search(r"Found:\s*(\d+)", safe)
        return {
            "status": "REMOTE_OBJECTS_FOUND" if completed.returncode == 0 and found and int(found.group(1)) > 0 else "REMOTE_QUERY_EMPTY_OR_FAILED",
            "aoss_uri": f"s3://{BUCKET}/{relative.strip('/')}/",
            "returncode": completed.returncode,
            "objects_reported": int(found.group(1)) if found else None,
            "log": str(log),
        }
    except Exception as exc:
        log.write_text(_redact(repr(exc), access, secret), encoding="utf-8")
        return {"status": "REMOTE_QUERY_FAILED", "aoss_uri": f"s3://{BUCKET}/{relative.strip('/')}/", "log": str(log), "reason": repr(exc)}


def _make_verified_task_subset(jsonl: Path, ledger: Path, task_output: Path, seeds_output: Path, split_seed: int) -> dict[str, Any]:
    """Select real task rows only from scenes with prior renderer success."""
    wanted = {"train": 8, "val": 4, "test": 4}
    ledger_rows = json.loads(ledger.read_text(encoding="utf-8")).get("scenes", [])
    approved = {str(row["scene_id"]) for row in ledger_rows if row.get("renderer_smoke_status") == "success"}
    candidates: dict[str, list[tuple[int, dict[str, Any]]]] = {key: [] for key in wanted}
    with jsonl.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            scene = str(row.get("scene_id") or "")
            split = scene_split(scene, split_seed)
            if scene not in approved or split not in wanted:
                continue
            if len(candidates[split]) < 80:
                candidates[split].append((index, row))
    if any(not candidates[key] for key in wanted):
        raise RuntimeError(f"verified-renderer task subset lacks a split: {{key: len(value) for key, value in candidates.items()}}")
    selected: list[tuple[int, dict[str, Any]]] = []
    for split in ("train", "val", "test"):
        source = candidates[split]
        selected.extend(source[index % len(source)] for index in range(wanted[split]))
    task_output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for _, row in selected), encoding="utf-8")
    seeds = list(range(len(selected)))
    seeds_output.write_text("".join(f"{seed}\n" for seed in seeds), encoding="utf-8")
    return {"source_indices": [index for index, _ in selected], "scene_ids": [str(row.get("scene_id")) for _, row in selected], "seeds": seeds, "count": len(seeds), "split_seed": split_seed, "approved_renderer_scenes": sorted(approved), "task_subset_hash": stable_hash(selected)}


def _run(command: list[str], log: Path, env: dict[str, str]) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        return proc.wait()


def _stage_verified_scenes(task_subset: Path, scene_ids: list[str], out: Path, cache_root: Path, args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    """Reuse VAGEN's official ledger+stage-one implementation in a new cache."""
    source_map = out / "state_scene_sources.json"
    source_map.write_text(json.dumps({"canary": str(task_subset)}, indent=2) + "\n", encoding="utf-8")
    ledger = out / "state_scene_ledger.json"
    pipeline = args.vagen_root / "scripts/r1_aoss_scene_pipeline.py"
    build_log = out / "logs" / "scene_ledger_build.log"
    build = [sys.executable, str(pipeline), "build-ledger", "--sources", str(source_map), "--output", str(ledger)]
    result = {"ledger": str(ledger), "build_returncode": _run(build, build_log, env), "build_log": str(build_log), "scenes": {}}
    if result["build_returncode"] != 0:
        return result
    unique = list(dict.fromkeys(scene_ids))
    for scene in unique:
        log = out / "logs" / "scene_stage" / f"{scene}.log"
        command = [sys.executable, str(pipeline), "stage-one", "--ledger", str(ledger), "--scene-id", scene, "--cache-root", str(cache_root), "--log-dir", str(out / "scene_download_logs"), "--conf", str(args.aoss_conf), "--ads-cli", str(args.ads_cli)]
        rc = _run(command, log, env)
        result["scenes"][scene] = {"returncode": rc, "log": str(log), "cache": str(cache_root / "ready" / scene)}
    result["all_staged"] = all(row["returncode"] == 0 for row in result["scenes"].values())
    write_json(out / "scene_stage_result.json", result)
    return result


def _upload(out_dir: Path, prefix: str, args: argparse.Namespace) -> dict[str, Any]:
    access, secret = _credentials(args.aoss_conf)
    source = str(out_dir) + "/"
    destination = f"s3://{access}:{secret}@{BUCKET}.{HOST}/{prefix.strip('/')}/"
    log = out_dir / "aoss_upload.log"
    result = subprocess.run([str(args.ads_cli), "--quiet", "sync", source, destination], text=True, capture_output=True, timeout=1800, check=False)
    safe = _redact(result.stdout + result.stderr, access, secret)
    log.write_text(safe, encoding="utf-8")
    found = re.search(r"copied:\s*(\d+)", safe)
    return {"status": "UPLOADED" if result.returncode == 0 else "UPLOAD_FAILED", "aoss_uri": f"s3://{BUCKET}/{prefix.strip('/')}/", "returncode": result.returncode, "copied": int(found.group(1)) if found else None, "log": str(log)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="qwen_pretrain_baseline,v50_step150,cambrian_pretrain_baseline")
    parser.add_argument("--vagen-root", type=Path, default=ROOT)
    parser.add_argument("--env-config", type=Path, default=ROOT / "examples/train/active_spatial/env_config_h800_7b_6types.yaml")
    parser.add_argument("--aoss-conf", type=Path, default=DEFAULT_CONF)
    parser.add_argument("--ads-cli", type=Path, default=DEFAULT_ADS)
    parser.add_argument("--verified-scene-ledger", type=Path, default=ROOT / "exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/required_scene_ledger.json")
    parser.add_argument("--reuse-scene-cache", type=Path, help="Previously validated task-owned cache root; stage-one performs resume validation only.")
    parser.add_argument("--upload-prefix")
    parser.add_argument("--skip-archive-query", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    # Exact renderer binding from the previously successful Active Spatial SCO
    # evaluation entry.  It uses the shared precompiled gsplat extension and
    # conda CUDA toolchain; this is intentionally not a pip-install fallback.
    vagen_env = "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite"
    extension_root = "/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer"
    env.update({
        "PYTHONPATH": f"{args.vagen_root}:{Path(__file__).resolve().parents[1]}:{env.get('PYTHONPATH', '')}",
        "PATH": f"{vagen_env}/bin:{env.get('PATH', '')}",
        "AOSS_CONF": str(args.aoss_conf),
        "AOSS_PLY_CACHE_DIR": str(out / "renderer_cache"),
        "VAGEN_ENV_ROOT": vagen_env,
        "CUDA_HOME": vagen_env,
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "CPATH": f"{vagen_env}/targets/x86_64-linux/include:{env.get('CPATH', '')}",
        "CPLUS_INCLUDE_PATH": f"{vagen_env}/targets/x86_64-linux/include:{env.get('CPLUS_INCLUDE_PATH', '')}",
        "LIBRARY_PATH": f"{vagen_env}/targets/x86_64-linux/lib:{env.get('LIBRARY_PATH', '')}",
        "TORCH_EXTENSIONS_DIR": extension_root,
        "VAGEN_GSPLAT_PREBUILT": "1",
    })
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    if any(name not in MODELS for name in requested):
        raise SystemExit(f"unknown model id in {requested}; valid={sorted(MODELS)}")
    inventory: dict[str, Any] = {"created_at": utc(), "worker": {"python": sys.executable, "hostname": os.uname().nodename}, "aoss_conf_present": args.aoss_conf.is_file(), "resources": {}}
    for name, spec in MODELS.items():
        local = _checkpoint_layout(Path(spec["path"]))
        remote = {"status": "AOSS_URI_NOT_LOCATED_FROM_EXISTING_EVIDENCE"}
        if spec["aoss_relative"] and not args.skip_archive_query:
            remote = _archive_dryrun(str(spec["aoss_relative"]), args, out)
        inventory["resources"][name] = {"role": name, "identity": spec["identity"], "backend": spec["backend"], "runtime_cache_path": spec["path"], "local_stage": local, "remote_discovery": remote}
    write_json(out / "resource_inventory.json", inventory)

    # The state bank itself is the renderer/AOSS smoke: it calls the native
    # gs_io client for each selected scene, writes real RGB frames, and replays
    # every action branch before retaining a label.
    jsonl = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_6types.jsonl")
    seeds_file = out / "episode_seeds.txt"
    task_subset = out / "verified_renderer_tasks.jsonl"
    # Chosen once from the successful R1 renderer ledger: 6/2/2 scene groups
    # across train/val/test, then expanded to 8/4/4 real episodes.
    seed_record = _make_verified_task_subset(jsonl, args.verified_scene_ledger, task_subset, seeds_file, 20260018)
    scene_cache_root = args.reuse_scene_cache or (out / "staged_interiorgs")
    scene_stage = _stage_verified_scenes(task_subset, list(seed_record["scene_ids"]), out, scene_cache_root, args, env)
    state_dir = out / "state_bank"
    state_cmd = [sys.executable, "-m", "active_repr_probe.build_state_bank", "--vagen-root", str(args.vagen_root), "--env-config", str(args.env_config), "--output-dir", str(state_dir), "--episode-seeds-file", str(seeds_file), "--steps-per-episode", "5", "--enable-visibility-labels", "--enable-collision-labels", "--jsonl-path", str(task_subset), "--gs-root", str(scene_cache_root / "ready"), "--gpu-device", "0", "--split-seed", "20260018"]
    state_rc = _run(state_cmd, out / "logs" / "state_bank.log", env) if scene_stage.get("all_staged") else 9
    state_manifest = json.loads((state_dir / "state_bank_manifest.json").read_text()) if (state_dir / "state_bank_manifest.json").is_file() else {"status": "FAILED", "returncode": state_rc}
    state_manifest["selected_seed_record"] = seed_record
    state_manifest["scene_stage"] = scene_stage
    state_manifest["worker_returncode"] = state_rc
    write_json(out / "state_bank_result.json", state_manifest)

    runs: dict[str, Any] = {"state_bank": state_manifest, "models": {}}
    if state_rc == 0 and int(state_manifest.get("rows", 0)) == 48:
        for name in requested:
            spec = MODELS[name]
            layout = inventory["resources"][name]["local_stage"]
            entry: dict[str, Any] = {"identity": spec["identity"], "layout_before_load": layout}
            if layout.get("status") != "OBJECTS_VERIFIED":
                entry.update({"status": "NOT_RUN", "reason": "exact local staged model did not pass object inventory"})
                runs["models"][name] = entry
                continue
            feat = out / "features" / f"{name}.pt"
            cmd = [sys.executable, "-m", "active_repr_probe.extract_features", "--state-bank", str(state_dir / "states.jsonl"), "--checkpoint", str(spec["path"]), "--backend", str(spec["backend"]), "--vagen-root", str(args.vagen_root), "--output", str(feat), "--limit", "48"]
            rc = _run(cmd, out / "logs" / f"features_{name}.log", env)
            entry.update({"feature_returncode": rc, "feature_log": str(out / "logs" / f"features_{name}.log")})
            manifest = feat.with_suffix(feat.suffix + ".manifest.json")
            if rc == 0 and manifest.is_file():
                entry.update({"status": "FORWARD_PASSED", "feature_manifest": str(manifest), "feature": str(feat)})
                probe_dir = out / "probes" / name
                probe_cmd = [sys.executable, "-m", "active_repr_probe.train_probe", "--features", str(feat), "--state-bank", str(state_dir / "states.jsonl"), "--output-dir", str(probe_dir), "--epochs", "20", "--run-label-shuffle"]
                probe_rc = _run(probe_cmd, out / "logs" / f"probe_{name}.log", env)
                entry.update({"probe_returncode": probe_rc, "probe_log": str(out / "logs" / f"probe_{name}.log"), "probe_results": str(probe_dir / "results.json") if (probe_dir / "results.json").is_file() else None})
                text_feat = out / "features" / f"{name}.text_only.pt"
                text_cmd = [sys.executable, "-m", "active_repr_probe.extract_features", "--state-bank", str(state_dir / "states.jsonl"), "--checkpoint", str(spec["path"]), "--backend", str(spec["backend"]), "--vagen-root", str(args.vagen_root), "--output", str(text_feat), "--limit", "48", "--input-mode", "text_only"]
                text_rc = _run(text_cmd, out / "logs" / f"features_{name}_text_only.log", env)
                entry.update({"text_only_feature_returncode": text_rc, "text_only_feature_log": str(out / "logs" / f"features_{name}_text_only.log")})
                if text_rc == 0 and text_feat.with_suffix(text_feat.suffix + ".manifest.json").is_file():
                    text_probe_dir = out / "probes" / f"{name}_text_only"
                    text_probe_cmd = [sys.executable, "-m", "active_repr_probe.train_probe", "--features", str(text_feat), "--state-bank", str(state_dir / "states.jsonl"), "--output-dir", str(text_probe_dir), "--epochs", "20", "--control", "text_only"]
                    text_probe_rc = _run(text_probe_cmd, out / "logs" / f"probe_{name}_text_only.log", env)
                    entry.update({"text_only_probe_returncode": text_probe_rc, "text_only_probe_results": str(text_probe_dir / "results.json") if (text_probe_dir / "results.json").is_file() else None})
            else:
                entry.update({"status": "FORWARD_FAILED"})
            runs["models"][name] = entry
    else:
        runs["state_bank_gate"] = "FAILED_OR_NOT_48_ROWS; no model forward attempted"
    runs["finished_at"] = utc()
    write_json(out / "canary_result.json", runs)
    if args.upload_prefix:
        upload = _upload(out, args.upload_prefix, args)
        write_json(out / "aoss_upload_receipt.json", upload)
    print(json.dumps({"status": "COMPLETE", "output": str(out), "state_rows": state_manifest.get("rows"), "models": {key: value.get("status") for key, value in runs["models"].items()}}, ensure_ascii=False))
    if state_rc != 0 or int(state_manifest.get("rows", 0)) != 48:
        raise SystemExit("CANARY_STATE_BANK_GATE_FAILED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(f"CANARY_WORKER_FAILED: {exc}")
