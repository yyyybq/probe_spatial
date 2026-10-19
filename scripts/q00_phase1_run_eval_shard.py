#!/usr/bin/env python3
"""Run an exact Q00 Wave-B shard with fail-closed artifact/feature preflight."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from vidfm3d.utils import aoss_fs


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
RUN_TAG = os.environ["RUN_TAG"]
BUNDLE = Path(os.environ["BUNDLE"]).resolve()
RUNTIME_TREE = (BUNDLE / "provenance/runtime_tree").resolve()
SHARD_TSV = Path(os.environ["SHARD_TSV"]).resolve()
SLOTS = int(os.environ.get("SLOTS", "1"))
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
REQUESTED_PYTHON = os.environ.get("PYTHON_BIN", PYTHON)
if Path(REQUESTED_PYTHON).resolve() != Path(PYTHON).resolve():
    raise RuntimeError(
        f"PYTHON_BIN must resolve to the frozen probe-spatial interpreter: "
        f"{REQUESTED_PYTHON!r} != {PYTHON!r}"
    )
DATA_ROOT = "aoss://baiqiao/InsScene-15K/data"
STREAM_ROOT = "aoss://baiqiao/InsScene-15K/FEAT_STREAMING"
TARGET_ROOT = "aoss://baiqiao/InsScene-15K/FEAT_TARGET"
SANITIZED_ENV_PREFIXES = ("INSCENE_", "PROBE_", "FEAT_", "Q00_")
# Streaming caches store model-token grids, not one feature slice per raw
# prefix frame.  Extraction pads each prefix to the model's frozen input
# length before the VAE/patch embedding, so the token-grid shape is fixed per
# model and independent of the 8/12/16/24-frame manifest prefix.  These exact
# shapes are derived from the frozen extractors and confirmed by all three
# direct-node canaries; keep the raw-frame cardinality gate on prefix_index.
STREAMING_TOKEN_SHAPES = {
    "lingbot": [13, 30, 52, 2048],
    "minimax_h3": [22, 15, 26, 5376],
    "wan": [21, 30, 52, 1536],
}
STREAMING_INPUT_LENGTHS = {"lingbot": 49, "minimax_h3": 73, "wan": 81}
STREAMING_FULL_IDENTITIES = {
    "lingbot": (
        "lingbot_video_layer_benchmark", "multi_layer",
        "historical_candidate_v1_val_protocol", "lingbot",
    ),
    "minimax_h3": (
        "minimax_h3_layer_benchmark", "multi_layer",
        "historical_candidate_v1_val_protocol", "minimax_h3",
    ),
    "wan": (
        "wan_cog_layer_sweep", "run_dir_discovered",
        "historical_candidate_v1_val_protocol", "wan",
    ),
}


def parse_gpu_ids_csv(raw: str) -> list[str]:
    parts = raw.split(",")
    if not parts:
        raise ValueError("GPU_IDS_CSV is empty")
    values: list[int] = []
    for part in parts:
        token = part.strip()
        if not token or re.fullmatch(r"[0-9]+", token) is None:
            raise ValueError(
                "GPU_IDS_CSV must be a comma-separated list of non-negative integers"
            )
        values.append(int(token))
    if len(set(values)) != len(values):
        raise ValueError("GPU_IDS_CSV contains duplicate device IDs")
    return [str(value) for value in values]


GPUS = parse_gpu_ids_csv(os.environ.get("GPU_IDS_CSV", "0"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_rows() -> list[dict[str, str]]:
    with SHARD_TSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or len({row["universe_id"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate-universe shard")
    if any(row["run_tag"] != RUN_TAG for row in rows):
        raise ValueError("run-tag mismatch in shard")
    return rows


def frozen_script_check() -> dict[str, Any]:
    inventory = json.loads((BUNDLE / "provenance/script_inventory.json").read_text())
    checks = []
    for item in inventory:
        current = ROOT / item["path"].split("provenance/scripts/", 1)[-1]
        actual = sha256(current) if current.is_file() else ""
        checks.append({"path": str(current), "expected": item["sha256"], "actual": actual, "ok": actual == item["sha256"]})
    result = {"ok": all(item["ok"] for item in checks), "checks": checks}
    if not result["ok"]:
        raise RuntimeError("current execution scripts do not match frozen bundle")
    return result


def frozen_runtime_check() -> dict[str, Any]:
    inventory_path = BUNDLE / "provenance/runtime_inventory.json"
    if not inventory_path.is_file():
        raise RuntimeError(f"missing frozen runtime inventory: {inventory_path}")
    metadata_path = BUNDLE / "bundle.json"
    metadata = json.loads(metadata_path.read_text())
    expected_inventory_sha = str(
        metadata.get("source_hashes", {}).get(
            "provenance/runtime_inventory.json", ""
        )
    )
    actual_inventory_sha = sha256(inventory_path)
    if not expected_inventory_sha or actual_inventory_sha != expected_inventory_sha:
        raise RuntimeError("frozen runtime inventory SHA does not match bundle metadata")
    inventory = json.loads(inventory_path.read_text())
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError("empty or invalid frozen runtime inventory")
    checks = []
    for item in inventory:
        relative = str(item.get("path", ""))
        frozen_relative = str(item.get("frozen_path", ""))
        frozen = (ROOT / frozen_relative).resolve()
        expected_frozen = (RUNTIME_TREE / relative).resolve()
        try:
            expected_frozen.relative_to(RUNTIME_TREE)
            contained = True
        except ValueError:
            contained = False
        frozen_sha = sha256(frozen) if frozen.is_file() else ""
        expected_sha = str(item.get("sha256", ""))
        expected_frozen_sha = str(item.get("frozen_sha256", ""))
        expected_size = int(item.get("size_bytes", -1))
        actual_size = frozen.stat().st_size if frozen.is_file() else -1
        ok = (
            bool(relative)
            and bool(frozen_relative)
            and contained
            and frozen == expected_frozen
            and frozen_sha == expected_sha == expected_frozen_sha
            and actual_size == expected_size
        )
        checks.append({
            "path": relative,
            "frozen_path": str(frozen),
            "expected_sha256": expected_frozen_sha,
            "actual_sha256": frozen_sha,
            "expected_size_bytes": expected_size,
            "actual_size_bytes": actual_size,
            "ok": ok,
        })
    result = {
        "ok": all(item["ok"] for item in checks),
        "inventory_path": str(inventory_path),
        "inventory_expected_sha256": expected_inventory_sha,
        "inventory_actual_sha256": actual_inventory_sha,
        "runtime_tree": str(RUNTIME_TREE),
        "count": len(checks),
        "checks": checks,
    }
    if not result["ok"]:
        raise RuntimeError("frozen runtime tree does not match its inventory")
    return result


def frozen_runtime_file(relative: str) -> Path:
    candidate = (RUNTIME_TREE / relative).resolve()
    try:
        candidate.relative_to(RUNTIME_TREE)
    except ValueError as exc:
        raise RuntimeError(f"runtime path escapes frozen tree: {relative}") from exc
    inventory = json.loads(
        (BUNDLE / "provenance/runtime_inventory.json").read_text()
    )
    matches = [item for item in inventory if str(item.get("path", "")) == relative]
    if len(matches) != 1:
        raise RuntimeError(f"runtime file is not uniquely inventoried: {relative}")
    inventoried = (ROOT / str(matches[0].get("frozen_path", ""))).resolve()
    if inventoried != candidate or not candidate.is_file():
        raise RuntimeError(f"runtime file is not the frozen inventoried copy: {relative}")
    if sha256(candidate) != str(matches[0].get("frozen_sha256", "")):
        raise RuntimeError(f"runtime file SHA mismatch: {relative}")
    return candidate


def cuda_startup_check() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    numeric_gpus = [int(value) for value in GPUS]
    checks = {
        "cuda_available": available,
        "positive_device_count": device_count > 0,
        "slots_within_gpu_list": 1 <= SLOTS <= len(GPUS),
        "device_count_covers_lanes": device_count >= SLOTS,
        "all_declared_gpu_ids_visible": bool(numeric_gpus)
        and max(numeric_gpus) < device_count,
    }
    result = {
        "ok": all(checks.values()),
        "checks": checks,
        "torch_cuda_available": available,
        "torch_cuda_device_count": device_count,
        "launcher_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_ids": GPUS,
        "lane_gpu_ids": GPUS[:SLOTS],
        "slots": SLOTS,
    }
    if not result["ok"]:
        raise RuntimeError(f"CUDA startup gate failed: {result}")
    return result


def require_canary_gate_for_remaining(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Allow only exact frozen one-row canaries or the exact 96-row shard.

    Classification is by content SHA rather than pathname, so copying or
    renaming ``wave_b_remaining_96.tsv`` cannot bypass the canary lock.
    """
    metadata = json.loads((BUNDLE / "bundle.json").read_text())
    source_hashes = metadata.get("source_hashes", {})
    actual_shard_sha = sha256(SHARD_TSV)
    remaining_sha = str(source_hashes.get("inputs/wave_b_remaining_96.tsv", ""))
    canary_hashes = {
        str(digest): name
        for name, digest in source_hashes.items()
        if name.startswith("inputs/canary_shards/") and name.endswith(".tsv")
    }
    if actual_shard_sha in canary_hashes:
        if len(rows) != 1 or rows[0].get("is_canary") != "true":
            raise RuntimeError("frozen canary shard content/row contract mismatch")
        return {
            "classification": "EXACT_ONE_ROW_CANARY",
            "required": False,
            "ok": True,
            "shard_sha256": actual_shard_sha,
            "frozen_source": canary_hashes[actual_shard_sha],
        }
    if not remaining_sha or actual_shard_sha != remaining_sha:
        raise RuntimeError(
            "runner accepts only the three exact frozen canary shard hashes or "
            "the exact frozen remaining-96 hash"
        )
    if len(rows) != 96 or any(row.get("is_canary") != "false" for row in rows):
        raise RuntimeError("remaining-96 shard content/row contract mismatch")
    audit_path = BUNDLE / "audit/canary_audit.json"
    results_path = BUNDLE / "audit/canary_results.csv"
    lock_path = BUNDLE / "audit/canary_audit_lock.json"
    if not all(path.is_file() for path in (audit_path, results_path, lock_path)):
        raise RuntimeError("remaining-96 shard requires completed canary audit+lock")
    audit = json.loads(audit_path.read_text())
    lock = json.loads(lock_path.read_text())
    checks = {
        "audit_pass": audit.get("ok") is True,
        "three_successes": int(audit.get("successful_summaries", -1)) == 3,
        "zero_failures": int(audit.get("failed_summaries", -1)) == 0,
        "audit_sha": lock.get("audit_sha256") == sha256(audit_path),
        "results_sha": lock.get("results_csv_sha256") == sha256(results_path),
        "canary_manifest_sha": lock.get("job_manifest_sha256")
        == sha256(BUNDLE / "inputs/wave_b_canary_3.tsv"),
        "bundle_sha": lock.get("bundle_json_sha256") == sha256(BUNDLE / "bundle.json"),
    }
    result = {
        "classification": "EXACT_REMAINING_96",
        "required": True,
        "ok": all(checks.values()),
        "shard_sha256": actual_shard_sha,
        "checks": checks,
        "audit_path": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "lock_path": str(lock_path),
        "lock_sha256": sha256(lock_path),
    }
    if not result["ok"]:
        raise RuntimeError(f"remaining-96 canary gate failed: {result}")
    return result


def local_artifact_preflight(row: dict[str, str]) -> dict[str, Any]:
    fields = {
        "experiment_config": "experiment_config_sha256",
        "hydra_config": "hydra_config_sha256",
        "hydra_overrides": "hydra_overrides_sha256",
        "run_manifest": "run_manifest_sha256",
        "config_tree": "config_tree_sha256",
        "resolved_config_frozen": "resolved_config_sha256",
        "train_log": "train_log_sha256",
        "launcher_manifest": "launcher_manifest_sha256",
    }
    checks = []
    for path_field, hash_field in fields.items():
        path = ROOT / row[path_field]
        actual = sha256(path) if path.is_file() else ""
        checks.append({
            "kind": path_field,
            "path": row[path_field],
            "expected_sha256": row[hash_field],
            "actual_sha256": actual,
            "ok": actual == row[hash_field],
        })
    if not all(item["ok"] for item in checks):
        raise ValueError("config/log/launcher artifact changed after freeze")
    resolved_path = (ROOT / row["resolved_config_frozen"]).resolve()
    resolved_root = (BUNDLE / "inputs/resolved_training_configs").resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("resolved training config escapes frozen bundle") from exc
    resolved = json.loads(resolved_path.read_text())
    contract = json.loads(row["resolved_eval_contract_json"])
    if (
        row.get("resolved_config_source_kind") != "run_manifest.config_resolved"
        or not isinstance(resolved, dict)
        or not resolved
        or not isinstance(contract, dict)
        or not all(contract.get("checks", {}).values())
    ):
        raise ValueError("resolved training-config contract is absent or not PASS")
    for key, value in contract.get("critical_eval_contract", {}).items():
        if resolved.get(key) != value:
            raise ValueError(f"resolved critical eval contract changed: {key}")
    return {
        "ok": True,
        "checks": checks,
        "resolved_config_source_kind": row["resolved_config_source_kind"],
        "resolved_config_sha256": row["resolved_config_sha256"],
        "resolved_eval_contract": contract,
    }


def clean_completion_preflight(row: dict[str, str]) -> dict[str, Any]:
    if row["checkpoint_status"] != "STRICT_CLEAN_FINAL_READY":
        raise ValueError("checkpoint status is not frozen strict-clean-final")
    expected_clean = "strict_pass=true;clean_marker=true;bad_marker=false;epoch_complete=true"
    if row["clean_completion_status"] != expected_clean:
        raise ValueError("frozen clean-completion status is not strict PASS")
    log_path = ROOT / row["train_log"]
    text = log_path.read_text(errors="replace")
    if "Starting training!" not in text:
        raise ValueError("training log lacks a start marker")
    final_attempt = text.rsplit("Starting training!", 1)[-1]
    completion_marker = f"`max_epochs={int(row['expected_max_epochs'])}` reached"
    forbidden = (
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "ProcessExitedException",
        "Killed\n",
        "[ERROR]",
    )
    bad_hits = [marker for marker in forbidden if marker in final_attempt]
    val_counts = [
        int(value)
        for value in re.findall(
            r"Loaded (\d+) records from scene manifest .* for split=val",
            final_attempt,
        )
    ]
    if completion_marker not in final_attempt or bad_hits:
        raise ValueError(
            f"final training attempt is not clean-complete: "
            f"completion={completion_marker in final_attempt} bad={bad_hits}"
        )
    if val_counts != [int(row["expected_n"])]:
        raise ValueError(
            f"final training attempt val-manifest cardinality mismatch: {val_counts}"
        )
    return {
        "ok": True,
        "checkpoint_status": row["checkpoint_status"],
        "clean_completion_status": row["clean_completion_status"],
        "expected_max_epochs": int(row["expected_max_epochs"]),
        "completion_marker": completion_marker,
        "bad_markers_in_final_attempt": bad_hits,
        "observed_val_counts_in_final_attempt": val_counts,
    }


def hydra_path(path: str) -> str:
    return path.replace("=", r"\=")


def frozen_manifest_path(row: dict[str, str]) -> Path:
    path = (ROOT / row["frozen_manifest"]).resolve()
    allowed_root = (BUNDLE / "inputs/manifests").resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError("frozen manifest path escapes bundle") from exc
    if not path.is_file() or sha256(path) != row["manifest_sha256"]:
        raise ValueError("frozen manifest missing or SHA mismatch")
    return path


def manifest_sample(row: dict[str, str]) -> dict[str, Any]:
    original_manifest = ROOT / row["manifest"]
    if not original_manifest.is_file() or sha256(original_manifest) != row["manifest_sha256"]:
        raise ValueError("original manifest missing or SHA mismatch")
    manifest = frozen_manifest_path(row)
    wanted = row["sample_id"]
    sample = None
    manifest_rows: list[dict[str, Any]] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        manifest_rows.append(item)
        if item.get("sample_id") == wanted:
            sample = item
    train_rows = [item for item in manifest_rows if str(item.get("split")) == "train"]
    val_rows = [item for item in manifest_rows if str(item.get("split")) == "val"]
    val_count = len(val_rows)
    if sample is None or str(sample.get("split")) != "val":
        raise ValueError("frozen canary sample absent from manifest val split")
    identity_checks = {
        "scene_id": str(sample.get("scene_id")) == row["sample_scene_id"],
        "source": str(sample.get("source")) == row["sample_source"],
        "streaming_window_id": str(sample.get("streaming_window_id")) == row["sample_window_id"],
        "streaming_tail": str(sample.get("streaming_tail")) == row["sample_tail"],
    }
    if not all(identity_checks.values()):
        raise ValueError(f"manifest sample full identity changed: {identity_checks}")
    if val_count != int(row["expected_n"]):
        raise ValueError(f"manifest val n mismatch: {val_count} != {row['expected_n']}")
    train_ids = {str(item.get("sample_id", "")) for item in train_rows}
    val_ids = {str(item.get("sample_id", "")) for item in val_rows}
    train_scenes = {str(item.get("scene_id", "")) for item in train_rows}
    val_scenes = {str(item.get("scene_id", "")) for item in val_rows}
    all_ids = [str(item.get("sample_id", "")) for item in manifest_rows]
    manifest_contract = {
        "train_n": len(train_rows) == int(row["manifest_train_n"]),
        "total_n": len(manifest_rows) == int(row["manifest_total_n"]),
        "all_sample_ids_unique": bool(all_ids) and all(all_ids)
        and len(set(all_ids)) == len(all_ids),
        "sample_overlap_zero": not (train_ids & val_ids)
        and int(row["manifest_train_val_sample_overlap_n"]) == 0,
        "scene_overlap_zero": not (train_scenes & val_scenes)
        and int(row["manifest_train_val_scene_overlap_n"]) == 0,
    }
    if not all(manifest_contract.values()):
        raise ValueError(f"frozen manifest uniqueness/leakage contract failed: {manifest_contract}")
    if [int(x) for x in sample["streaming_indices"]] != json.loads(row["sample_streaming_indices_json"]):
        raise ValueError("manifest sample streaming identity changed")
    return sample


def materialized_file_audit(path: str) -> tuple[Path, dict[str, Any]]:
    # A successful byte materialization is stronger evidence than a separate
    # remote ``contains`` probe and avoids treating a transient metadata-query
    # failure as permission to fall back.  Missing/unreadable objects still
    # raise and fail the job closed.
    local = Path(aoss_fs.materialize(path)).resolve()
    if not local.is_file() or local.stat().st_size <= 0:
        raise FileNotFoundError(f"materialized object is absent or empty: {path}")
    return local, {
        "source_path": path,
        "materialized_path": str(local),
        "sha256": sha256(local),
        "size_bytes": local.stat().st_size,
    }


def tensor_audit(path: str) -> dict[str, Any]:
    local, artifact = materialized_file_audit(path)
    tensors = aoss_fs.load_safetensors(str(local))
    if set(tensors) != {"feat"}:
        raise ValueError(f"unexpected safetensor keys at {path}: {sorted(tensors)}")
    feat = tensors["feat"]
    finite = bool(torch.isfinite(feat).all().item())
    if feat.ndim != 4 or feat.numel() == 0 or not finite:
        raise ValueError(f"invalid feature tensor at {path}: shape={tuple(feat.shape)} finite={finite}")
    return {
        **artifact,
        "shape": list(feat.shape),
        "dtype": str(feat.dtype),
        "finite": finite,
    }


def feature_preflight(row: dict[str, str], sample: dict[str, Any]) -> dict[str, Any]:
    stream_root = row["streaming_feat_root"]
    target_root = row["target_feat_root"]
    if stream_root != STREAM_ROOT or "${" in stream_root:
        raise ValueError(f"streaming feature root is not frozen exact root: {stream_root}")
    target_required = row["probe"] in {"action_dynamics", "path_integration", "counterfactual"}
    if target_required and target_root != TARGET_ROOT:
        raise ValueError(f"target feature root is not frozen exact root: {target_root}")
    source = str(sample["source"])
    scene = str(sample["scene_id"])
    window_id = int(sample["streaming_window_id"])
    tail = int(sample["streaming_tail"])
    scene_root = aoss_fs.join(stream_root, row["model"], source, scene)
    index_path = aoss_fs.join(scene_root, "prefix_index.npy")
    index_local, index_artifact = materialized_file_audit(index_path)
    raw_records = aoss_fs.load_npy(
        str(index_local), allow_pickle=True
    ).tolist()
    if isinstance(raw_records, dict):
        raw_records = list(raw_records.values())
    records = {(int(item.get("window_id", -1)), int(item["tail"])): item for item in raw_records if isinstance(item, dict) and "tail" in item}
    record = records.get((window_id, tail))
    if record is None:
        raise ValueError(f"prefix_index lacks exact key {(window_id, tail)}")
    expected_indices = [int(value) for value in sample["streaming_indices"]]
    actual_indices = [int(value) for value in record.get("actual_input_raw_ids", record.get("indices", []))]
    valid_length = int(record.get("valid_length", -1))
    input_length = int(record.get("input_length", -1))
    pad_mode = str(record.get("pad_mode", ""))
    expected_input_length = STREAMING_INPUT_LENGTHS.get(row["model"])
    full_identity = (
        row["campaign"], row["model_variant"], row["protocol"], row["model"]
    )
    if full_identity != STREAMING_FULL_IDENTITIES.get(row["model"]):
        raise ValueError(f"unsupported streaming feature full identity: {full_identity}")
    if actual_indices != expected_indices:
        raise ValueError(f"sample identity mismatch: expected={expected_indices} actual={actual_indices}")
    if valid_length != len(expected_indices):
        raise ValueError("prefix_index valid_length mismatch")
    if input_length != expected_input_length or pad_mode != "repeat_tail":
        raise ValueError(
            f"prefix_index model-input contract mismatch: model={row['model']} "
            f"input_length={input_length} expected={expected_input_length} "
            f"pad_mode={pad_mode!r}"
        )
    feature_name = f"feature{row['feat_postfix']}.sft"
    stream_path = aoss_fs.join(scene_root, f"window_{window_id:04d}", f"prefix_{tail:06d}", feature_name)
    stream_tensor = tensor_audit(stream_path)
    expected_shape = STREAMING_TOKEN_SHAPES.get(row["model"])
    if (
        expected_shape is None
        or int(row["video_channels"]) != expected_shape[-1]
        or stream_tensor["shape"] != expected_shape
    ):
        raise ValueError(
            f"streaming feature shape violates frozen model-token contract: "
            f"model={row['model']} actual={stream_tensor['shape']} "
            f"expected={expected_shape}"
        )
    positions = torch.arange(len(expected_indices), dtype=torch.float32)
    if input_length > valid_length and len(expected_indices) > 0:
        positions[-1] = float(input_length - 1)
    derived_vfm_indices = (
        positions / float(input_length - 1) * float(expected_shape[0] - 1)
    ).round().long().clamp(0, expected_shape[0] - 1).tolist()
    if (
        len(derived_vfm_indices) != len(expected_indices)
        or not derived_vfm_indices
        or min(derived_vfm_indices) < 0
        or max(derived_vfm_indices) >= expected_shape[0]
        or derived_vfm_indices[-1] != expected_shape[0] - 1
    ):
        raise ValueError("derived streaming token-index contract mismatch")
    result: dict[str, Any] = {
        "streaming_root": stream_root,
        "streaming_root_kind": "remote_uri_materialized_for_content_audit",
        "aoss_cache_dir": os.environ.get("AOSS_CACHE_DIR"),
        "prefix_index": index_path,
        "prefix_index_materialized": index_artifact,
        "sample_id": sample["sample_id"],
        "window_id": window_id,
        "tail": tail,
        "expected_streaming_indices": expected_indices,
        "actual_streaming_indices": actual_indices,
        "streaming_tensor": stream_tensor,
        "streaming_model_contract": {
            "model": row["model"],
            "full_identity": list(full_identity),
            "expected_token_shape": expected_shape,
            "input_length": input_length,
            "valid_length": valid_length,
            "pad_mode": pad_mode,
            "derived_vfm_indices": derived_vfm_indices,
        },
    }
    if target_required:
        target_scene = aoss_fs.join(target_root, row["model"], source, scene)
        prefixed = aoss_fs.join(target_scene, f"prefix_{tail:06d}")
        target_dir = prefixed if aoss_fs.isdir(prefixed) else target_scene
        target_index_path = aoss_fs.join(target_dir, "target_indices.npy")
        target_index_local, target_index_artifact = materialized_file_audit(
            target_index_path
        )
        target_indices = [
            int(value)
            for value in aoss_fs.load_npy(str(target_index_local)).tolist()
        ]
        expected_target = int(row["sample_target_raw_id"])
        if target_indices.count(expected_target) != 1:
            raise ValueError(f"target identity mismatch: expected raw {expected_target}, indices={target_indices}")
        target_path = aoss_fs.join(target_dir, feature_name)
        target_tensor = tensor_audit(target_path)
        if (
            int(target_tensor["shape"][0]) != len(target_indices)
            or target_tensor["shape"][1:] != expected_shape[1:]
        ):
            raise ValueError(
                f"target feature shape violates frozen model-token contract: "
                f"actual={target_tensor['shape']} expected=[M,{expected_shape[1:]}]"
            )
        result.update({
            "target_root": target_root,
            "target_root_kind": "remote_uri_materialized_for_content_audit",
            "target_indices_path": target_index_path,
            "target_indices_materialized": target_index_artifact,
            "target_indices": target_indices,
            "expected_target_raw_id": expected_target,
            "target_tensor": target_tensor,
        })
    return result


def full_streaming_index_audit(row: dict[str, str]) -> dict[str, Any]:
    """Verify every frozen val row against the cache's actual prefix index.

    This closes the lineage gap left by checking only one tensor canary: the
    evaluator still has to read every feature object, while this independent
    audit proves that every manifest sample maps to the exact raw-frame IDs
    recorded by the feature cache.
    """
    manifest = frozen_manifest_path(row)
    val_rows: list[dict[str, Any]] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        if sample.get("split") == "val":
            val_rows.append(sample)
    if len(val_rows) != int(row["expected_n"]):
        raise ValueError("full prefix-index audit val cardinality mismatch")
    index_cache: dict[tuple[str, str], tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]] = {}
    identity_ledger: list[dict[str, Any]] = []
    for sample in val_rows:
        source = str(sample.get("source", ""))
        scene = str(sample.get("scene_id", ""))
        if not source or not scene:
            raise ValueError("manifest row lacks feature source/scene identity")
        cache_key = (source, scene)
        if cache_key not in index_cache:
            index_path = aoss_fs.join(
                row["streaming_feat_root"], row["model"], source, scene,
                "prefix_index.npy",
            )
            local, artifact = materialized_file_audit(index_path)
            raw_records = aoss_fs.load_npy(str(local), allow_pickle=True).tolist()
            if isinstance(raw_records, dict):
                raw_records = list(raw_records.values())
            records: dict[tuple[int, int], dict[str, Any]] = {}
            for item in raw_records:
                if not isinstance(item, dict) or "tail" not in item:
                    continue
                key = (int(item.get("window_id", -1)), int(item["tail"]))
                if key in records:
                    raise ValueError(f"duplicate prefix-index identity {key}: {index_path}")
                records[key] = item
            if not records:
                raise ValueError(f"empty/invalid prefix index: {index_path}")
            index_cache[cache_key] = (records, artifact)
        records, artifact = index_cache[cache_key]
        window_id = int(sample["streaming_window_id"])
        tail = int(sample["streaming_tail"])
        record = records.get((window_id, tail))
        if record is None:
            raise ValueError(
                f"prefix index lacks frozen sample {sample.get('sample_id')} "
                f"key={(window_id, tail)}"
            )
        expected = [int(value) for value in sample["streaming_indices"]]
        actual = [
            int(value)
            for value in record.get("actual_input_raw_ids", record.get("indices", []))
        ]
        tail_raw = int(sample["streaming_tail_raw"])
        expected_input_length = STREAMING_INPUT_LENGTHS.get(row["model"])
        if (
            actual != expected
            or int(record.get("valid_length", -1)) != len(expected)
            or int(record.get("input_length", -1)) != expected_input_length
            or str(record.get("pad_mode", "")) != "repeat_tail"
            or not actual
            or actual[-1] != tail_raw
        ):
            raise ValueError(
                f"full prefix-index identity mismatch for {sample.get('sample_id')}: "
                f"expected={expected} actual={actual} tail_raw={tail_raw}"
            )
        identity_ledger.append({
            "sample_id": str(sample["sample_id"]),
            "source": source,
            "scene_id": scene,
            "window_id": window_id,
            "tail": tail,
            "tail_raw": tail_raw,
            "actual_input_raw_ids": actual,
            "input_length": int(record["input_length"]),
            "valid_length": int(record["valid_length"]),
            "pad_mode": str(record["pad_mode"]),
            "prefix_index_sha256": artifact["sha256"],
        })
    ledger_payload = "\n".join(
        json.dumps(item, sort_keys=True, separators=(",", ":"))
        for item in identity_ledger
    ) + "\n"
    artifacts = [
        {
            "source": key[0],
            "scene_id": key[1],
            **value[1],
        }
        for key, value in sorted(index_cache.items())
    ]
    return {
        "ok": True,
        "manifest_val_records_checked": len(identity_ledger),
        "unique_prefix_indexes_checked": len(artifacts),
        "identity_ledger_sha256": hashlib.sha256(ledger_payload.encode()).hexdigest(),
        "prefix_index_artifacts": artifacts,
        "contract": "actual_input_raw_ids==manifest.streaming_indices;valid_length;input_length;repeat_tail;tail_raw",
    }


def feature_artifacts_after_evaluator(feature: dict[str, Any]) -> dict[str, Any]:
    """Re-hash every materialized preflight artifact after evaluation."""
    candidates: list[tuple[str, dict[str, Any]]] = []
    for name in (
        "prefix_index_materialized",
        "streaming_tensor",
        "target_indices_materialized",
        "target_tensor",
    ):
        item = feature.get(name)
        if isinstance(item, dict):
            candidates.append((name, item))
    full = feature.get("full_streaming_index_audit", {})
    for index, item in enumerate(full.get("prefix_index_artifacts", [])):
        if isinstance(item, dict):
            candidates.append((f"full_prefix_index_{index:04d}", item))
    expected_by_path: dict[str, tuple[str, int, list[str]]] = {}
    for name, item in candidates:
        path = str(item.get("materialized_path", ""))
        expected_hash = str(item.get("sha256", ""))
        expected_size = int(item.get("size_bytes", -1))
        if not path or not expected_hash or expected_size <= 0:
            raise ValueError(f"incomplete preflight content audit for {name}")
        previous = expected_by_path.get(path)
        if previous is not None:
            if previous[:2] != (expected_hash, expected_size):
                raise ValueError(f"conflicting preflight hashes for {path}")
            previous[2].append(name)
        else:
            expected_by_path[path] = (expected_hash, expected_size, [name])
    checks = []
    for path_text, (expected_hash, expected_size, names) in sorted(expected_by_path.items()):
        path = Path(path_text)
        actual_hash = sha256(path) if path.is_file() else ""
        actual_size = path.stat().st_size if path.is_file() else -1
        checks.append({
            "path": path_text,
            "artifact_names": names,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "expected_size_bytes": expected_size,
            "actual_size_bytes": actual_size,
            "ok": actual_hash == expected_hash and actual_size == expected_size,
        })
    result = {"ok": bool(checks) and all(item["ok"] for item in checks), "checks": checks}
    result["unique_artifacts_checked"] = len(checks)
    if not result["ok"]:
        raise ValueError("materialized feature artifact changed during evaluation")
    return result


def checkpoint_preflight(row: dict[str, str], deep: bool) -> dict[str, Any]:
    checkpoint = ROOT / row["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_sha = sha256(checkpoint)
    if actual_sha != row["checkpoint_sha256"] or checkpoint.stat().st_size != int(row["checkpoint_size_bytes"]):
        raise ValueError("checkpoint inventory mismatch")
    result: dict[str, Any] = {
        "path": str(checkpoint),
        "sha256": actual_sha,
        "size_bytes": checkpoint.stat().st_size,
        "expected_final_epoch": int(row["expected_final_epoch"]),
        "expected_final_step": int(row["expected_final_step"]),
        "deep_metadata_loaded": deep,
    }
    if deep:
        data = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        epoch = int(data.get("epoch", -1))
        step = int(data.get("global_step", -1))
        result.update({"checkpoint_epoch": epoch, "checkpoint_global_step": step})
        if epoch != int(row["expected_final_epoch"]) or step != int(row["expected_final_step"]):
            raise ValueError(f"checkpoint metadata is not expected final: epoch={epoch} step={step}")
        del data
    return result


def command(
    row: dict[str, str], gpu: str
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    output_dir = ROOT / row["output_directory"]
    run_folder = f"{RUN_TAG}/{output_dir.name}"
    out_name = output_dir.name
    evaluator_relative = (
        "vidfm3d/eval_a1_streaming.py"
        if row["evaluator"] == "a1"
        else "vidfm3d/eval_diag.py"
    )
    executable = frozen_runtime_file(evaluator_relative)
    cmd = [
        PYTHON,
        str(executable),
        f"experiment={row['experiment']}",
        f"vfm_name={row['vfm_name']}",
        f"video_channels={row['video_channels']}",
        f"streaming_feat_root={row['streaming_feat_root']}",
        f"feature_layer={row['feature_layer']}",
        f"feature_timestep={row['feature_timestep']}",
        f"prefix_len={row['prefix_len']}",
        f"job_name={out_name}",
        f"paths.run_folder_name={run_folder}",
        "ckpt_path=" + hydra_path(str((ROOT / row["checkpoint"]).resolve())),
        "+eval_split=val",
        "train=false",
        "test=false",
        "data.data_module.num_workers_val=2",
        "data.data_module.batch_size_per_device_val=4",
        "logger.wandb.offline=true",
    ]
    if row["evaluator"] == "diag":
        cmd.append("++bootstrap_reps=0")
    if row.get("feat_postfix"):
        cmd.append(f"feat_postfix={row['feat_postfix']}")
    if row.get("target_feat_root"):
        cmd.append(f"++target_feat_root={row['target_feat_root']}")
    for key, value in sorted(json.loads(row["extra_overrides_json"] or "{}").items()):
        cmd.append(f"{key}={value}")
    removed_parent_keys = sorted(
        key
        for key in os.environ
        if any(key.startswith(prefix) for prefix in SANITIZED_ENV_PREFIXES)
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(key.startswith(prefix) for prefix in SANITIZED_ENV_PREFIXES)
    }
    env.update({
        "PYTHONPATH": str(RUNTIME_TREE),
        "PYTHONNOUSERSITE": "1",
        "PROJECT_ROOT": str(ROOT),
        "CUDA_VISIBLE_DEVICES": gpu,
        "INSCENE_DATA_ROOT": DATA_ROOT,
        "INSCENE_MANIFEST": str(frozen_manifest_path(row)),
        "INSCENE_TRUST_EXPANDED_MANIFEST": "1",
        "INSCENE_STREAMING_FEAT_ROOT": row["streaming_feat_root"],
        "AOSS_CACHE_DIR": f"/mnt/local_cache/inscene/aoss_cache/{RUN_TAG}/shared",
        "AOSS_FS_METRICS": str(output_dir / "eval/aoss_metrics.jsonl"),
        "MPLCONFIGDIR": f"/tmp/matplotlib-{RUN_TAG}-{row['universe_id']}",
        "Q00_UNIVERSE_ID": row["universe_id"],
        "Q00_CAMPAIGN": row["campaign"],
        "Q00_MODEL_VARIANT": row["model_variant"],
        "Q00_PROTOCOL": row["protocol"],
        "Q00_MANIFEST_SHA256": row["manifest_sha256"],
        "Q00_CONFIG_SHA256": row["experiment_config_sha256"],
        "Q00_RESOLVED_CONFIG_SHA256": row["resolved_config_sha256"],
        "Q00_RESOLVED_CONFIG_SOURCE_KIND": row["resolved_config_source_kind"],
        "Q00_REQUIRE_CUDA": "1",
        "Q00_RUNTIME_TREE": str(RUNTIME_TREE),
        "PROBE_FP16_PASSTHROUGH": "0",
    })
    if row.get("target_feat_root"):
        env["INSCENE_TARGET_FEAT_ROOT"] = row["target_feat_root"]
    exact_reintroduced = sorted(
        key
        for key in env
        if any(key.startswith(prefix) for prefix in SANITIZED_ENV_PREFIXES)
    )
    allowed_reintroduced = {
        "INSCENE_DATA_ROOT",
        "INSCENE_MANIFEST",
        "INSCENE_TRUST_EXPANDED_MANIFEST",
        "INSCENE_STREAMING_FEAT_ROOT",
        "PROBE_FP16_PASSTHROUGH",
        "Q00_UNIVERSE_ID",
        "Q00_CAMPAIGN",
        "Q00_MODEL_VARIANT",
        "Q00_PROTOCOL",
        "Q00_MANIFEST_SHA256",
        "Q00_CONFIG_SHA256",
        "Q00_RESOLVED_CONFIG_SHA256",
        "Q00_RESOLVED_CONFIG_SOURCE_KIND",
        "Q00_REQUIRE_CUDA",
        "Q00_RUNTIME_TREE",
    }
    if row.get("target_feat_root"):
        allowed_reintroduced.add("INSCENE_TARGET_FEAT_ROOT")
    if set(exact_reintroduced) != allowed_reintroduced:
        raise ValueError(
            f"child environment has unexpected experiment controls: "
            f"{sorted(set(exact_reintroduced) ^ allowed_reintroduced)}"
        )
    env_policy = {
        "sanitized_prefixes": list(SANITIZED_ENV_PREFIXES),
        "removed_parent_keys": removed_parent_keys,
        "exact_reintroduced_keys": exact_reintroduced,
        "policy": "strip all inherited INSCENE_/PROBE_/FEAT_/Q00_ then add exact frozen values only",
    }
    return cmd, env, env_policy


def exact_value(summary: dict[str, Any], key: str) -> Any:
    if key not in summary:
        raise ValueError(f"summary missing required field {key}")
    return summary[key]


def frozen_manifest_val_rows(row: dict[str, str]) -> dict[str, dict[str, Any]]:
    manifest = frozen_manifest_path(row)
    ordered: list[dict[str, Any]] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if str(item.get("split")) == "val":
            ordered.append(item)
    ids = [str(item.get("sample_id", "")) for item in ordered]
    if (
        len(ordered) != int(row["expected_n"])
        or any(not value for value in ids)
        or len(set(ids)) != len(ids)
        or hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
        != row["manifest_val_ordered_ids_sha256"]
    ):
        raise ValueError("frozen manifest val identity ledger mismatch")
    return dict(zip(ids, ordered))


def prediction_audit(row: dict[str, str], expected_n: int) -> dict[str, Any]:
    prediction_path = ROOT / row["prediction_path"]
    if row["prediction_required"] != "true":
        return {"required": False, "path": row["prediction_path"]}
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    records = torch.load(prediction_path, map_location="cpu", weights_only=False)
    if not isinstance(records, list) or len(records) != expected_n:
        raise ValueError(f"prediction record count mismatch: {type(records)} / {len(records) if isinstance(records, list) else 'n/a'}")
    checked_tensors = 0
    checked_numeric_scalars = 0
    manifest_ids: list[str] = []
    manifest_rows = frozen_manifest_val_rows(row)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("prediction record is not a dictionary")
        manifest_id = str(record.get("manifest_sample_id", ""))
        if not manifest_id:
            raise ValueError("prediction record lacks manifest_sample_id")
        manifest_ids.append(manifest_id)
        manifest_row = manifest_rows.get(manifest_id)
        if manifest_row is None:
            raise ValueError(
                f"prediction manifest_sample_id is absent from frozen val manifest: "
                f"{manifest_id}"
            )
        required_record_fields = {
            "scene_path", "streaming_window_id", "streaming_tail_raw"
        }
        required_manifest_fields = {
            "scene_dir", "streaming_window_id", "streaming_tail_raw"
        }
        if (
            not required_record_fields.issubset(record)
            or not required_manifest_fields.issubset(manifest_row)
            or not str(manifest_row["scene_dir"])
        ):
            raise ValueError(
                f"prediction/frozen-manifest identity fields missing for {manifest_id}"
            )
        identity_checks = {
            "scene_path": str(record["scene_path"])
            == str(manifest_row["scene_dir"]),
            "streaming_window_id": int(record["streaming_window_id"])
            == int(manifest_row["streaming_window_id"]),
            "streaming_tail_raw": int(record["streaming_tail_raw"])
            == int(manifest_row["streaming_tail_raw"]),
        }
        if not all(identity_checks.values()):
            raise ValueError(
                f"prediction/frozen-manifest identity mismatch for {manifest_id}: "
                f"{identity_checks}"
            )
        for value in record.values():
            if isinstance(value, torch.Tensor):
                checked_tensors += 1
                if value.numel() and not bool(torch.isfinite(value).all().item()):
                    raise ValueError("prediction tensor contains non-finite values")
            elif isinstance(value, float):
                checked_numeric_scalars += 1
                if not math.isfinite(value):
                    raise ValueError("prediction scalar contains non-finite value")
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("prediction manifest_sample_id values are not unique")
    manifest_ordered_sha = hashlib.sha256(("\n".join(manifest_ids) + "\n").encode()).hexdigest()
    valid_subset_signature = ""
    if row["family"] == "B2":
        signature_rows = []
        for record in records:
            polar = record.get("polar_gt")
            if not isinstance(polar, torch.Tensor) or "target_bin" not in record or "scene_path" not in record:
                raise ValueError("B2 record lacks frozen valid-subset signature fields")
            signature_rows.append({
                "scene_path": str(record["scene_path"]),
                "target_bin": int(record["target_bin"]),
                "polar_gt": [round(float(value), 6) for value in polar.tolist()],
            })
        payload = json.dumps(signature_rows, sort_keys=True, separators=(",", ":"))
        valid_subset_signature = hashlib.sha256(payload.encode()).hexdigest()
        if valid_subset_signature != row["expected_valid_subset_signature_sha256"]:
            raise ValueError("B2 ordered construct-valid subset signature changed")
    else:
        if manifest_ordered_sha != row["manifest_val_ordered_ids_sha256"]:
            raise ValueError("prediction ordered manifest identity differs from frozen manifest")
    if row["family"] == "C2":
        if any(not bool(record["valid"].all().item()) for record in records):
            raise ValueError("C2 contains invalid horizon(s), indicating missing context/target features")
    if row["family"] == "C3":
        threshold = float(json.loads(row["extra_overrides_json"])["counterfactual_min_overlap"])
        for record in records:
            actual = record["valid"].bool()
            construct = record["overlap"].float() >= threshold
            if not torch.equal(actual, construct):
                raise ValueError("C3 invalid horizon is not explained solely by frozen overlap construction")
    records_checked = len(records)
    del records
    return {
        "required": True,
        "path": row["prediction_path"],
        "sha256": sha256(prediction_path),
        "n": expected_n,
        "checked_finite_tensors": checked_tensors,
        "checked_finite_numeric_scalars": checked_numeric_scalars,
        "manifest_ids_ordered_sha256": manifest_ordered_sha,
        "frozen_manifest_identity_records_checked": records_checked,
        "frozen_manifest_identity_fields": [
            "scene_path==scene_dir",
            "streaming_window_id",
            "streaming_tail_raw",
        ],
        "valid_subset_signature_sha256": valid_subset_signature,
    }


def summary_audit(row: dict[str, str], gpu: str) -> dict[str, Any]:
    summary_path = ROOT / row["summary_path"]
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    cuda_execution = {
        "execution_device": exact_value(summary, "execution_device"),
        "cuda_available": exact_value(summary, "cuda_available"),
        "cuda_device_count": exact_value(summary, "cuda_device_count"),
        "cuda_device_name": exact_value(summary, "cuda_device_name"),
        "cuda_device_uuid": exact_value(summary, "cuda_device_uuid"),
        "CUDA_VISIBLE_DEVICES": exact_value(summary, "CUDA_VISIBLE_DEVICES"),
        "Q00_REQUIRE_CUDA": "1",
    }
    cuda_checks = {
        "execution_device": str(cuda_execution["execution_device"]) == "cuda:0",
        "cuda_available": cuda_execution["cuda_available"] is True,
        "cuda_device_count": int(cuda_execution["cuda_device_count"]) == 1,
        "cuda_device_name": bool(str(cuda_execution["cuda_device_name"] or "").strip()),
        "CUDA_VISIBLE_DEVICES": str(cuda_execution["CUDA_VISIBLE_DEVICES"]) == gpu,
    }
    if not all(cuda_checks.values()):
        raise ValueError(
            f"summary does not prove single-lane CUDA execution: {cuda_checks} / "
            f"{cuda_execution}"
        )
    cuda_execution["checks"] = cuda_checks
    n_key = "n_samples" if row["evaluator"] == "a1" else "n"
    n = int(exact_value(summary, n_key))
    manifest_n = int(row["expected_n"])
    expected_result_n = int(row["expected_result_n"])
    cardinality: dict[str, Any] = {"manifest_val_n": manifest_n, "summary_n": n}
    if n != expected_result_n:
        raise ValueError(f"result n differs from frozen family contract: {n} != {expected_result_n}")
    if row["evaluator"] == "a1":
        if n != manifest_n:
            raise ValueError(f"A1 summary n mismatch: {n} != {manifest_n}")
    else:
        loader_n = int(exact_value(summary, "loader_n_samples"))
        valid_n = int(exact_value(summary, "valid_record_n"))
        invalid_n = int(exact_value(summary, "invalid_record_n"))
        cardinality.update({"loader_n_samples": loader_n, "valid_record_n": valid_n, "invalid_record_n": invalid_n})
        if loader_n != manifest_n or valid_n != n or invalid_n != loader_n - valid_n:
            raise ValueError(f"diag loader/record accounting mismatch: {cardinality}")
        if row["family"] == "B2":
            if not (0 < n <= manifest_n):
                raise ValueError(f"B2 valid-subset n out of contract: {cardinality}")
        elif n != manifest_n or invalid_n != 0:
            raise ValueError(f"diag summary n mismatch: {n} != {manifest_n}")
    loader_ids = [str(value) for value in exact_value(summary, "loader_manifest_ids")]
    valid_ids = [str(value) for value in exact_value(summary, "valid_manifest_ids")]
    invalid_ids = [str(value) for value in exact_value(summary, "invalid_manifest_ids")]
    if (
        len(loader_ids) != manifest_n
        or len(set(loader_ids)) != manifest_n
        or hashlib.sha256(("\n".join(loader_ids) + "\n").encode()).hexdigest() != row["manifest_val_ordered_ids_sha256"]
        or len(valid_ids) != n
        or len(invalid_ids) != manifest_n - n
        or set(valid_ids).isdisjoint(set(invalid_ids)) is False
        or set(valid_ids) | set(invalid_ids) != set(loader_ids)
    ):
        raise ValueError("manifest/valid/invalid per-row identity ledger mismatch")
    expected_probe = row["probe"]
    strict_identity = {
        "probe_type": str(exact_value(summary, "probe_type")) == expected_probe,
        "vfm_name": str(exact_value(summary, "vfm_name")) == row["model"],
        "feature_layer": str(exact_value(summary, "feature_layer")) == row["feature_layer"],
        "feat_postfix": str(exact_value(summary, "feat_postfix")) == row["feat_postfix"],
        "feature_timestep": str(exact_value(summary, "feature_timestep")) == row["feature_timestep"],
        "prefix_len": str(exact_value(summary, "prefix_len")) == row["prefix_len"],
        "streaming_feat_root": str(exact_value(summary, "streaming_feat_root")) == row["streaming_feat_root"],
        "job_name": str(exact_value(summary, "job_name")) == Path(row["output_directory"]).name,
        "q00_universe_id": str(exact_value(summary, "q00_universe_id")) == row["universe_id"],
        "q00_campaign": str(exact_value(summary, "q00_campaign")) == row["campaign"],
        "q00_model_variant": str(exact_value(summary, "q00_model_variant")) == row["model_variant"],
        "q00_protocol": str(exact_value(summary, "q00_protocol")) == row["protocol"],
        "q00_manifest_sha256": str(exact_value(summary, "q00_manifest_sha256")) == row["manifest_sha256"],
        "q00_config_sha256": str(exact_value(summary, "q00_config_sha256")) == row["experiment_config_sha256"],
        "q00_resolved_config_sha256": str(
            exact_value(summary, "q00_resolved_config_sha256")
        ) == row["resolved_config_sha256"],
        "q00_resolved_config_source_kind": str(
            exact_value(summary, "q00_resolved_config_source_kind")
        ) == row["resolved_config_source_kind"],
    }
    ckpt = str(exact_value(summary, "ckpt_path"))
    strict_identity["ckpt_path"] = Path(ckpt).resolve() == (ROOT / row["checkpoint"]).resolve()
    if row["target_feat_root"]:
        strict_identity["target_feat_root"] = str(exact_value(summary, "target_feat_root")) == row["target_feat_root"]
    if not all(strict_identity.values()):
        raise ValueError(f"strict summary identity mismatch: {strict_identity}")
    expected_heads = json.loads(row["expected_active_heads_json"])
    if expected_heads and list(exact_value(summary, "active_heads")) != expected_heads:
        raise ValueError(f"A1 active-head contract mismatch: {summary.get('active_heads')} != {expected_heads}")

    required_values: dict[str, float] = {}
    for name in json.loads(row["required_summary_metrics_json"]):
        try:
            number = float(exact_value(summary, name))
        except (TypeError, ValueError):
            raise ValueError(f"required summary metric is non-numeric: {name}")
        if not math.isfinite(number):
            raise ValueError(f"required summary metric is non-finite: {name}={number}")
        required_values[name] = number
    for name in json.loads(row["count_equals_expected_n_json"]):
        if int(required_values[name]) != n:
            raise ValueError(f"A1 valid-count contract mismatch: {name}={required_values[name]} n={n}")
    if row["family"] == "C2" and int(required_values.get("n_loop", 0)) > 0:
        loop_value = float(exact_value(summary, "loop_closure_error"))
        if not math.isfinite(loop_value):
            raise ValueError("C2 loop_closure_error is non-finite despite n_loop > 0")
        required_values["loop_closure_error"] = loop_value

    metrics = []
    for output in json.loads(row["expected_outputs_json"]):
        name = output["metric_name"]
        value = summary.get(name, summary.get(f"mean_{name}"))
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"missing/non-numeric metric {name}: {value!r}")
        if not math.isfinite(number):
            raise ValueError(f"non-finite metric {name}: {number}")
        metrics.append({**output, "metric_value": number})
    prediction = prediction_audit(row, n)
    valid_ordered_sha = hashlib.sha256(("\n".join(valid_ids) + "\n").encode()).hexdigest()
    if prediction.get("manifest_ids_ordered_sha256") != valid_ordered_sha:
        raise ValueError("prediction identity order differs from summary valid-ID ledger")
    return {
        "ok": True,
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "prediction": prediction,
        "n": n,
        "cuda_execution": cuda_execution,
        "cardinality": cardinality,
        "lineage": {
            "loader_manifest_ids_ordered_sha256": row["manifest_val_ordered_ids_sha256"],
            "valid_manifest_ids_ordered_sha256": valid_ordered_sha,
            "invalid_manifest_ids": invalid_ids,
        },
        "strict_identity": strict_identity,
        "required_summary_values": required_values,
        "metrics": metrics,
    }


def run_one(row: dict[str, str], slot: int) -> dict[str, Any]:
    gpu = GPUS[slot % len(GPUS)]
    output_dir = ROOT / row["output_directory"]
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_path = eval_dir / "q00_eval.log"
    status: dict[str, Any] = {"universe_id": row["universe_id"], "model": row["model"], "gpu": gpu, "slot": slot}
    if (ROOT / row["summary_path"]).exists():
        return {**status, "ok": False, "rc": 3, "error": "refusing pre-existing summary in versioned output"}
    start = time.time()
    try:
        sample = manifest_sample(row)
        feature = feature_preflight(row, sample)
        feature["full_streaming_index_audit"] = full_streaming_index_audit(row)
        preflight = {
            "ok": True,
            "time": now(),
            "universe_id": row["universe_id"],
            "full_identity": {key: row[key] for key in ("campaign", "model", "model_variant", "family", "probe", "layer", "prefix", "horizon", "seed", "protocol")},
            "execution_hashes": {
                "bundle_json": sha256(BUNDLE / "bundle.json"),
                "shard_tsv": sha256(SHARD_TSV),
                "script_inventory": sha256(BUNDLE / "provenance/script_inventory.json"),
                "runtime_inventory": sha256(BUNDLE / "provenance/runtime_inventory.json"),
            },
            "manifest": {
                "original_path": row["manifest"],
                "frozen_path": row["frozen_manifest"],
                "sha256": row["manifest_sha256"],
                "train_n": int(row["manifest_train_n"]),
                "expected_n": int(row["expected_n"]),
                "total_n": int(row["manifest_total_n"]),
                "train_val_sample_overlap_n": int(row["manifest_train_val_sample_overlap_n"]),
                "train_val_scene_overlap_n": int(row["manifest_train_val_scene_overlap_n"]),
            },
            "checkpoint": checkpoint_preflight(row, deep=True),
            "clean_completion": clean_completion_preflight(row),
            "local_artifacts": local_artifact_preflight(row),
            "feature": feature,
            "fallback_allowed": False,
        }
        runtime_before = frozen_runtime_check()
        preflight["frozen_runtime_before_evaluator"] = runtime_before
        write_json(ROOT / row["preflight_path"], preflight)
        cmd, env, env_policy = command(row, gpu)
        identity_env_keys = [
            "PYTHONPATH", "PROJECT_ROOT", "CUDA_VISIBLE_DEVICES", "Q00_REQUIRE_CUDA",
            "Q00_RUNTIME_TREE", "INSCENE_DATA_ROOT", "INSCENE_MANIFEST",
            "INSCENE_TRUST_EXPANDED_MANIFEST", "INSCENE_STREAMING_FEAT_ROOT",
            "PROBE_FP16_PASSTHROUGH", "Q00_UNIVERSE_ID", "Q00_CAMPAIGN",
            "Q00_MODEL_VARIANT", "Q00_PROTOCOL", "Q00_MANIFEST_SHA256",
            "Q00_CONFIG_SHA256", "Q00_RESOLVED_CONFIG_SHA256",
            "Q00_RESOLVED_CONFIG_SOURCE_KIND", "AOSS_CACHE_DIR",
            "AOSS_FS_METRICS",
        ]
        if row.get("target_feat_root"):
            identity_env_keys.append("INSCENE_TARGET_FEAT_ROOT")
        write_json(ROOT / row["command_path"], {
            "time": now(), "cwd": str(RUNTIME_TREE), "argv": cmd,
            "identity_env": {key: env[key] for key in identity_env_keys},
            "environment_policy": env_policy,
            "frozen_evaluator_sha256": sha256(Path(cmd[1])),
            "argv_sha256": hashlib.sha256("\0".join(cmd).encode()).hexdigest(),
        })
        try:
            with log_path.open("w") as log:
                print(f"START {now()} universe_id={row['universe_id']} gpu={gpu}", file=log, flush=True)
                proc = subprocess.run(
                    cmd,
                    cwd=RUNTIME_TREE,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                print(f"END {now()} rc={proc.returncode} elapsed={time.time()-start:.1f}", file=log, flush=True)
        finally:
            runtime_after = frozen_runtime_check()
        if proc.returncode:
            raise RuntimeError(f"evaluator rc={proc.returncode}")
        text = log_path.read_text(errors="replace")
        forbidden = ("using dummy", "allow_missing_vfm=True", "fall back to canonical", "refusing to fall back")
        hits = [item for item in forbidden if item in text]
        if hits:
            raise RuntimeError(f"fallback marker(s) in eval log: {hits}")
        postflight = summary_audit(row, gpu)
        postflight["checkpoint_after_evaluator"] = checkpoint_preflight(row, deep=False)
        postflight["feature_artifacts_after_evaluator"] = (
            feature_artifacts_after_evaluator(feature)
        )
        postflight["frozen_runtime_after_evaluator"] = runtime_after
        write_json(eval_dir / "q00_postflight.json", postflight)
        return {**status, "ok": True, "rc": 0, "elapsed": time.time() - start, "summary": row["summary_path"]}
    except Exception as exc:
        failure = {"ok": False, "time": now(), "universe_id": row["universe_id"], "error": repr(exc)}
        write_json(eval_dir / "q00_failure.json", failure)
        with log_path.open("a") as log:
            print(f"FAIL_CLOSED {now()} error={exc!r}", file=log, flush=True)
        return {**status, "ok": False, "rc": 2, "elapsed": time.time() - start, "error": repr(exc)}


def main() -> None:
    rows = read_rows()
    if not GPUS or SLOTS < 1 or SLOTS > len(GPUS):
        raise SystemExit("invalid SLOTS/GPU_IDS_CSV")
    cuda_check = cuda_startup_check()
    canary_gate = require_canary_gate_for_remaining(rows)
    launcher = BUNDLE / "launcher"
    launcher.mkdir(parents=True, exist_ok=True)
    # Use a fresh versioned cache namespace for both parent preflight and child
    # evaluator.  This prevents inherited/stale cache roots from changing the
    # artifact bytes that were audited before evaluation.
    os.environ["AOSS_CACHE_DIR"] = (
        f"/mnt/local_cache/inscene/aoss_cache/{RUN_TAG}/shared"
    )
    os.environ["AOSS_FS_METRICS"] = str(
        launcher / f"{SHARD_TSV.stem}_preflight_aoss_metrics.jsonl"
    )
    script_check = frozen_script_check()
    runtime_check = frozen_runtime_check()
    write_json(launcher / f"{SHARD_TSV.stem}_script_check.json", script_check)
    write_json(launcher / f"{SHARD_TSV.stem}_runtime_check.json", runtime_check)
    write_json(launcher / f"{SHARD_TSV.stem}_cuda_check.json", cuda_check)
    write_json(launcher / f"{SHARD_TSV.stem}_canary_gate.json", canary_gate)
    status_path = launcher / f"{SHARD_TSV.stem}_status.jsonl"
    if status_path.exists():
        raise SystemExit(f"refusing existing status file: {status_path}")
    with status_path.open("w") as handle:
        print(json.dumps({"event": "shard_start", "time": now(), "shard": str(SHARD_TSV), "rows": len(rows), "slots": SLOTS, "gpus": GPUS, "cuda": cuda_check, "canary_gate": canary_gate}), file=handle, flush=True)
        # One serial lane per GPU.  This prevents a short job on one worker
        # thread from taking the next row assigned to a still-busy GPU.
        lanes: list[list[dict[str, str]]] = [[] for _ in range(SLOTS)]
        for index, row in enumerate(rows):
            lanes[index % SLOTS].append(row)

        def run_lane(slot: int, lane: list[dict[str, str]]) -> list[dict[str, Any]]:
            return [run_one(row, slot) for row in lane]

        with ThreadPoolExecutor(max_workers=SLOTS) as pool:
            futures = [pool.submit(run_lane, slot, lane) for slot, lane in enumerate(lanes)]
            failed = 0
            for future in futures:
                for result in future.result():
                    failed += not result["ok"]
                    print(json.dumps({"event": "job_done", "time": now(), **result}), file=handle, flush=True)
        print(json.dumps({"event": "shard_done", "time": now(), "failed": failed}), file=handle, flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
