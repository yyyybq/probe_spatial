#!/usr/bin/env python3
"""Run one SHA-bound canonical-V2 B2 (ego_belief_v2) re-train on exactly one GPU.

Copy of the 2026-09-28 single_gpu_cog12_ego_fp32 runner generalised to the two
immutable training packs the original 16 cells were trained from:

  cogvideox  runtime/main_v2_probe_v2_r7_single_v3 + probe_validation_cog96 overlay
  aether     runtime/main_v2_probe_v2_aetherfix_r1 + probe552 aether_target8 overlay

The only scientific change versus the original runs is PROBE_FP16_PASSTHROUGH=0:
features are widened to fp32 in the loader instead of being handed to the GPU as
fp16 (the setting under which these 16 heads diverged to non-finite weights).
Everything else (pack, overlay, item, config SHA, overrides) is pinned by hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
CAMPAIGN = ROOT / "artifacts/main_v2_probe_v2_816_20260917/single_gpu_b2_fp32_20261016_r1"
ITEM_ROOT = CAMPAIGN / "items"
STATE_ROOT = CAMPAIGN / "state"
OUTPUT_ROOT = ROOT / "logs/inscene15k_streaming/runs/main_v2_single_gpu_b2_fp32_20261016_r1"
PYTHON = Path("/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python")
RUN_ID = "main_v2_single_gpu_b2_fp32_20261016_r1"
PROBE = "ego_belief_v2"
MODEL_PACKS = {
    "cogvideox": {
        "pack": ROOT / "runtime/main_v2_probe_v2_r7_single_v3",
        "source_sha_file_sha256": "e073c6c29674c6e75e493c7d194c388b0523898fcffef5d810e15faf0a705c38",
        "overlay": ROOT / "artifacts/main_v2_probe_v2_816_20260917/probe_validation_cog96_20260924_r1/overlay/manifest.json",
        "overlay_sha256": "bfa35b55a3912897cde848d9cffc3a804a5fe763991b9c48dbb3d6d65d1cf5d9",
        "original_campaign": "single_gpu_cog84_h800_20260928_r1",
    },
    "aether": {
        "pack": ROOT / "runtime/main_v2_probe_v2_aetherfix_r1",
        "source_sha_file_sha256": "d3faa935676630ae4398785b357d6b211c18ccc3dfc25087856f0e751a7e2621",
        "overlay": ROOT / "artifacts/main_v2_probe_v2_816_20260917/probe552_validation_aether_target8_20260923_r4/overlay/manifest.json",
        "overlay_sha256": "bc5f4b0952b6c3c7f7e29ada5743a6ec0b7412101fb8e48d93fc8f741321a99a",
        "original_campaign": "single_gpu_aether24_20260923_r1",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", type=Path, required=True)
    parser.add_argument("--item-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-gpu-check", action="store_true", help="validate-only on a GPU-less login node")
    args = parser.parse_args()
    item_path = args.item.resolve()
    if args.run_id != RUN_ID:
        raise RuntimeError("run ID mismatch")
    if item_path.parent != ITEM_ROOT.resolve() or item_path.is_symlink():
        raise RuntimeError("item path is outside the frozen item root")
    if not item_path.is_file() or sha256(item_path) != args.item_sha256:
        raise RuntimeError("item SHA mismatch")
    rows = [json.loads(line) for line in item_path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("one-item file does not contain exactly one row")
    item = rows[0]
    if item["probe"] != PROBE:
        raise RuntimeError(f"this campaign re-trains {PROBE} only, got {item['probe']}")
    if item["model"] not in MODEL_PACKS:
        raise RuntimeError(f"model {item['model']} is outside the B2 fp32 re-train scope")
    pack_info = MODEL_PACKS[item["model"]]
    pack = pack_info["pack"]
    source = pack / "source"
    source_sha_file = pack / "source.sha256"
    overlay = pack_info["overlay"]
    if sha256(source_sha_file) != pack_info["source_sha_file_sha256"]:
        raise RuntimeError("source checksum-file identity mismatch")
    source_check = subprocess.run(["sha256sum", "--status", "-c", str(source_sha_file)], cwd=source)
    if source_check.returncode:
        raise RuntimeError("immutable source verification failed")
    config = source / "configs/experiment" / f"{item['experiment_config']}.yaml"
    if not config.is_file() or sha256(config) != item["experiment_config_sha256"]:
        raise RuntimeError("experiment config SHA mismatch")
    if sha256(overlay) != pack_info["overlay_sha256"]:
        raise RuntimeError("receipt-union overlay SHA mismatch")
    execution_manifest = Path(item["execution_manifest"]["path"])
    if sha256(execution_manifest) != item["execution_manifest"]["sha256"]:
        raise RuntimeError("execution manifest SHA mismatch")
    if args.validate_only and args.skip_gpu_check:
        visible = ["<unchecked>"]
    else:
        visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
        if not visible:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                text=True, capture_output=True, check=False,
            )
            visible = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        if len(visible) != 1:
            raise RuntimeError(f"single-GPU launcher requires one visible GPU, got {visible!r}")

    if args.validate_only:
        print(json.dumps({
            "status": "PASS", "unique_key": item["unique_key"], "model": item["model"],
            "item_sha256": args.item_sha256,
            "config_sha256": item["experiment_config_sha256"],
            "pack": str(pack),
            "source_sha_file_sha256": pack_info["source_sha_file_sha256"],
            "overlay_sha256": pack_info["overlay_sha256"],
            "visible_gpu_count": len(visible),
        }, sort_keys=True))
        return 0

    key = item["unique_key"]
    mode = "canary" if args.canary else "production"
    claim = STATE_ROOT / "claims" / mode / key
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"exclusive writer claim exists for {mode}/{key}") from exc
    state_dir = STATE_ROOT / mode / args.run_id
    output = OUTPUT_ROOT / mode / args.run_id / key
    output.mkdir(parents=True, exist_ok=True)
    complete = state_dir / "completed" / f"{key}.json"
    if complete.exists() and not args.canary:
        claim.rmdir()
        raise RuntimeError(f"production completion receipt already exists for {key}")

    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": visible[0],
        "PROJECT_ROOT": str(source),
        "PYTHONPATH": str(source),
        "PATH": str(PYTHON.parent) + ":" + environment.get("PATH", ""),
        "PROBE_OWNER_OVERLAY_MANIFEST": str(overlay),
        "PROBE_OWNER_LOCAL_MIRROR_ROOT": "/mnt/umm/users/yinbaiqiao/InsScene-15K",
        "INSCENE_DATA_ROOT": "/mnt/umm/users/yinbaiqiao/InsScene-15K/data",
        "INSCENE_LOCAL_SCENE_ROOT": "/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2",
        "INSCENE_MANIFEST": str(execution_manifest),
        "INSCENE_MANIFEST_SHA256": item["execution_manifest"]["sha256"],
        "INSCENE_STREAMING_FEAT_ROOT": "aoss://baiqiao/PROBE816_OWNER_OVERLAY_ONLY",
        "INSCENE_TARGET_FEAT_ROOT": "aoss://baiqiao/PROBE816_OWNER_OVERLAY_ONLY",
        "INSCENE_TARGET_INDEX_ROOT": "/mnt/umm/users/yinbaiqiao/InsScene-15K/PROBE816_OWNER_OVERLAY_ONLY",
        "INSCENE_TARGET_LAYOUT": "flat",
        "INSCENE_TRUST_EXPANDED_MANIFEST": "1",
        "INSCENE_PROBE_TASK": item["probe"],
        "INSCENE_METADATA_CACHE_MAX_SCENES": "32",
        "INSCENE_LABEL_CACHE_ROOT": str(STATE_ROOT / "label_sidecar_v1"),
        # The one deliberate change versus the 2026-09 production runs.
        "PROBE_FP16_PASSTHROUGH": "0",
        "AOSS_CACHE_DIR": str(Path("/mnt/local_cache/inscene/main_v2_probe_v2_r7") / args.run_id / key),
        "AOSS_FS_METRICS": str(output / "aoss_metrics.jsonl"),
        "AOSS_FS_METRICS_MODE": "aggregate",
        "AOSS_LOG_LOCAL_HITS": "0",
        "MPLCONFIGDIR": str(Path("/tmp") / f"mpl-{args.run_id}"),
        "OMP_NUM_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "offline",
    })
    overrides = [f"experiment={item['experiment_config']}"] + list(item["scientific_overrides"])
    overrides += [
        "paths.log_dir=/mnt/umm/users/yinbaiqiao/probe_spatial/logs",
        f"paths.run_folder_name=main_v2_single_gpu_b2_fp32_20261016_r1/{mode}/{args.run_id}/{key}",
        f"job_name={key}",
        "autoresume=false",
        f"data.data_module.num_workers={0 if args.canary else 1}",
        f"data.data_module.num_workers_val={0 if args.canary else 1}",
    ]
    if args.canary:
        overrides += [
            "logger=canary_csv", "trainer.max_epochs=1", "++trainer.max_steps=2",
            "++trainer.limit_train_batches=2", "++trainer.limit_val_batches=0",
            "++trainer.num_sanity_val_steps=0", "++trainer.log_every_n_steps=1",
            "++trainer.enable_progress_bar=false",
        ]
    else:
        overrides += [
            f"logger.wandb.name={args.run_id}_{key}",
            f"logger.wandb.group={args.run_id}",
            "logger.wandb.offline=true",
        ]
    command = [str(PYTHON), "vidfm3d/train.py"] + overrides
    start = {
        "schema": "MAIN_V2_PROBE_B2_FP32_RETRAIN_STATE_V1", "at": now(), "state": "STARTING",
        "mode": mode, "unique_key": key, "model": item["model"], "item_sha256": args.item_sha256,
        "pack": str(pack), "source_sha_file_sha256": pack_info["source_sha_file_sha256"],
        "overlay": str(overlay), "overlay_sha256": pack_info["overlay_sha256"],
        "feature_transfer_precision": {"PROBE_FP16_PASSTHROUGH": "0"},
        "original_campaign": pack_info["original_campaign"],
        "output_dir": str(output), "command": command,
    }
    atomic_json(state_dir / "experiments" / f"{key}.json", start)
    atomic_json(output / "START_RECEIPT.json", start)
    child: subprocess.Popen[Any] | None = None

    def terminate(_signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    rc = 99
    try:
        with (output / "train.log").open("a", encoding="utf-8") as log:
            log.write(f"BEGIN {now()} mode={mode} key={key}\n")
            log.flush()
            child = subprocess.Popen(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
            rc = child.wait()
            log.write(f"END {now()} rc={rc}\n")
        terminal = {**start, "at": now(), "state": "SUCCEEDED" if rc == 0 else "FAILED", "returncode": rc}
        atomic_json(state_dir / "experiments" / f"{key}.json", terminal)
        if rc == 0:
            receipt = {**terminal, "completed_utc": now()}
            atomic_json(output / "TRAIN_COMPLETE.json", receipt)
            atomic_json(complete, receipt)
        return rc
    finally:
        claim.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
