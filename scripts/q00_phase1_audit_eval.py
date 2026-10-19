#!/usr/bin/env python3
"""Audit Q00 Wave-B canary or full staging outputs before promotion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = "aoss://baiqiao/InsScene-15K/data"
SANITIZED_ENV_PREFIXES = ("INSCENE_", "PROBE_", "FEAT_", "Q00_")
ENVIRONMENT_POLICY = (
    "strip all inherited INSCENE_/PROBE_/FEAT_/Q00_ then add exact frozen values only"
)
RESOLVED_CONFIG_SOURCE_KIND = "run_manifest.config_resolved"
FROZEN_PYTHON = Path(
    "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
).resolve()
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def content_audit_ok(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    digest = str(item.get("sha256", ""))
    try:
        size = int(item.get("size_bytes", 0))
    except (TypeError, ValueError):
        return False
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest) and size > 0


def exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"boolean is not an integer contract value: {label}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-integral contract value {label}: {value!r}") from exc
    if str(number) != str(value) and not isinstance(value, int):
        raise ValueError(f"non-canonical integer contract value {label}: {value!r}")
    return number


def torch_round_indices(
    positions: list[int], input_length: int, token_length: int
) -> list[int]:
    if input_length <= 1 or token_length <= 0:
        raise ValueError("invalid model-token index dimensions")
    return [
        max(0, min(token_length - 1, int(round(
            float(position) / float(input_length - 1) * float(token_length - 1)
        ))))
        for position in positions
    ]


def is_sha256(value: Any) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value)) is not None


def expected_command_env(bundle: Path, row: dict[str, str]) -> dict[str, str]:
    output = (ROOT / row["output_directory"]).resolve()
    expected = {
        "PYTHONPATH": str((bundle / "provenance/runtime_tree").resolve()),
        "PROJECT_ROOT": str(ROOT),
        "Q00_REQUIRE_CUDA": "1",
        "Q00_RUNTIME_TREE": str((bundle / "provenance/runtime_tree").resolve()),
        "INSCENE_DATA_ROOT": DATA_ROOT,
        "INSCENE_MANIFEST": str((ROOT / row["frozen_manifest"]).resolve()),
        "INSCENE_TRUST_EXPANDED_MANIFEST": "1",
        "INSCENE_STREAMING_FEAT_ROOT": row["streaming_feat_root"],
        "PROBE_FP16_PASSTHROUGH": "0",
        "Q00_UNIVERSE_ID": row["universe_id"],
        "Q00_CAMPAIGN": row["campaign"],
        "Q00_MODEL_VARIANT": row["model_variant"],
        "Q00_PROTOCOL": row["protocol"],
        "Q00_MANIFEST_SHA256": row["manifest_sha256"],
        "Q00_CONFIG_SHA256": row["experiment_config_sha256"],
        "Q00_RESOLVED_CONFIG_SHA256": row["resolved_config_sha256"],
        "Q00_RESOLVED_CONFIG_SOURCE_KIND": row["resolved_config_source_kind"],
        "AOSS_CACHE_DIR": (
            f"/mnt/local_cache/inscene/aoss_cache/{row['run_tag']}/shared"
        ),
        "AOSS_FS_METRICS": str(output / "eval/aoss_metrics.jsonl"),
    }
    if row.get("target_feat_root"):
        expected["INSCENE_TARGET_FEAT_ROOT"] = row["target_feat_root"]
    return expected


def validate_environment_policy(
    policy: Any, expected_env: dict[str, str], *, universe_id: str
) -> None:
    if not isinstance(policy, dict):
        raise ValueError(f"command environment policy is not an object: {universe_id}")
    removed = policy.get("removed_parent_keys")
    reintroduced = policy.get("exact_reintroduced_keys")
    expected_reintroduced = sorted(
        key
        for key in expected_env
        if any(key.startswith(prefix) for prefix in SANITIZED_ENV_PREFIXES)
    )
    if (
        policy.get("sanitized_prefixes") != list(SANITIZED_ENV_PREFIXES)
        or policy.get("policy") != ENVIRONMENT_POLICY
        or not isinstance(removed, list)
        or any(not isinstance(key, str) for key in removed)
        or removed != sorted(set(removed))
        or any(
            not any(key.startswith(prefix) for prefix in SANITIZED_ENV_PREFIXES)
            for key in removed
        )
        or reintroduced != expected_reintroduced
    ):
        raise ValueError(f"command environment sanitization policy mismatch: {universe_id}")


def validate_local_artifacts(
    bundle: Path, row: dict[str, str], local_artifacts: Any
) -> None:
    universe_id = row["universe_id"]
    if not isinstance(local_artifacts, dict) or local_artifacts.get("ok") is not True:
        raise ValueError(f"local artifact preflight is not PASS: {universe_id}")
    contract = {
        "experiment_config": "experiment_config_sha256",
        "hydra_config": "hydra_config_sha256",
        "hydra_overrides": "hydra_overrides_sha256",
        "run_manifest": "run_manifest_sha256",
        "config_tree": "config_tree_sha256",
        "resolved_config_frozen": "resolved_config_sha256",
        "train_log": "train_log_sha256",
        "launcher_manifest": "launcher_manifest_sha256",
    }
    raw_checks = local_artifacts.get("checks")
    if not isinstance(raw_checks, list):
        raise ValueError(f"local artifact checks are not a list: {universe_id}")
    checks: dict[str, dict[str, Any]] = {}
    for item in raw_checks:
        if not isinstance(item, dict) or not item.get("kind"):
            raise ValueError(f"invalid local artifact check: {universe_id}")
        kind = str(item["kind"])
        if kind in checks:
            raise ValueError(f"duplicate local artifact check {kind}: {universe_id}")
        checks[kind] = item
    if set(checks) != set(contract):
        raise ValueError(f"local artifact check set mismatch: {universe_id}")
    for kind, hash_field in contract.items():
        item = checks[kind]
        expected_hash = row[hash_field]
        if (
            item.get("ok") is not True
            or item.get("path") != row[kind]
            or item.get("expected_sha256") != expected_hash
            or item.get("actual_sha256") != expected_hash
            or not is_sha256(expected_hash)
        ):
            raise ValueError(f"local artifact mismatch {kind}: {universe_id}")
    resolved_path = (ROOT / row["resolved_config_frozen"]).resolve()
    resolved_root = (bundle / "inputs/resolved_training_configs").resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"resolved config escapes frozen bundle: {universe_id}") from exc
    if not resolved_path.is_file() or sha256(resolved_path) != row["resolved_config_sha256"]:
        raise ValueError(f"resolved config content/hash mismatch: {universe_id}")
    resolved = json.loads(resolved_path.read_text())
    run_manifest = json.loads((ROOT / row["run_manifest"]).read_text())
    frozen_contract = json.loads(row["resolved_eval_contract_json"])
    checks_contract = frozen_contract.get("checks") if isinstance(frozen_contract, dict) else None
    critical = (
        frozen_contract.get("critical_eval_contract")
        if isinstance(frozen_contract, dict)
        else None
    )
    model = resolved.get("model", {}) if isinstance(resolved, dict) else {}
    trainer = resolved.get("trainer", {}) if isinstance(resolved, dict) else {}
    expected_training_probe = "pixalign" if row["family"] == "A1" else row["probe"]
    target_required = bool(row.get("target_feat_root"))
    extra_overrides = json.loads(row.get("extra_overrides_json") or "{}")
    resolved_identity = {
        "run_manifest_config": isinstance(run_manifest, dict)
        and run_manifest.get("config") == resolved,
        "canonical_frozen_bytes": resolved_path.read_text()
        == json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        "seed": str(resolved.get("seed")) == row["seed"],
        "probe": isinstance(model, dict)
        and str(model.get("probe_type")) == expected_training_probe,
        "vfm_name": str(resolved.get("vfm_name")) == row["vfm_name"] == row["model"],
        "video_channels": str(resolved.get("video_channels")) == row["video_channels"],
        "prefix_len": str(resolved.get("prefix_len")) == row["prefix_len"],
        "feature_layer": str(resolved.get("feature_layer")) == row["feature_layer"],
        "feature_timestep": str(resolved.get("feature_timestep"))
        == row["feature_timestep"],
        "feat_postfix": str(resolved.get("feat_postfix")) == row["feat_postfix"],
        "streaming_feat_root": resolved.get("streaming_feat_root")
        == row["streaming_feat_root"],
        "max_epochs": isinstance(trainer, dict)
        and str(trainer.get("max_epochs")) == row["expected_max_epochs"],
        "target_feat_root": (
            resolved.get("target_feat_root") == row["target_feat_root"]
            if target_required
            else resolved.get("target_feat_root") in (None, "")
        ),
        "critical_eval_contract": extra_overrides == critical,
    }
    if (
        row.get("resolved_config_source_kind") != RESOLVED_CONFIG_SOURCE_KIND
        or local_artifacts.get("resolved_config_source_kind")
        != RESOLVED_CONFIG_SOURCE_KIND
        or local_artifacts.get("resolved_config_sha256") != row["resolved_config_sha256"]
        or local_artifacts.get("resolved_eval_contract") != frozen_contract
        or not isinstance(resolved, dict)
        or not resolved
        or not isinstance(checks_contract, dict)
        or not checks_contract
        or any(value is not True for value in checks_contract.values())
        or not isinstance(critical, dict)
        or any(resolved.get(key) != value for key, value in critical.items())
        or not all(resolved_identity.values())
    ):
        raise ValueError(f"resolved training-config contract mismatch: {universe_id}")


def validate_full_streaming_index(row: dict[str, str], feature: dict[str, Any]) -> None:
    universe_id = row["universe_id"]
    audit = feature.get("full_streaming_index_audit")
    if not isinstance(audit, dict):
        raise ValueError(f"missing full streaming-index audit: {universe_id}")
    artifacts = audit.get("prefix_index_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"invalid prefix-index artifact list: {universe_id}")
    unique_n = exact_int(
        audit.get("unique_prefix_indexes_checked"),
        label=f"{universe_id}.unique_prefix_indexes_checked",
    )
    artifact_keys: list[tuple[str, str]] = []
    for item in artifacts:
        if (
            not isinstance(item, dict)
            or not str(item.get("source", ""))
            or not str(item.get("scene_id", ""))
            or not str(item.get("source_path", ""))
            or not str(item.get("materialized_path", ""))
            or not content_audit_ok(item)
        ):
            raise ValueError(f"invalid full prefix-index content audit: {universe_id}")
        artifact_keys.append((str(item["source"]), str(item["scene_id"])))
    if (
        audit.get("ok") is not True
        or exact_int(
            audit.get("manifest_val_records_checked"),
            label=f"{universe_id}.manifest_val_records_checked",
        )
        != exact_int(row["expected_n"], label=f"{universe_id}.expected_n")
        or unique_n <= 0
        or unique_n != len(artifacts)
        or len(set(artifact_keys)) != len(artifact_keys)
        or not is_sha256(audit.get("identity_ledger_sha256"))
        or audit.get("contract")
        != "actual_input_raw_ids==manifest.streaming_indices;valid_length;input_length;repeat_tail;tail_raw"
    ):
        raise ValueError(f"full streaming-index lineage contract mismatch: {universe_id}")


def validate_feature_tensor_contract(row: dict[str, str], feature: dict[str, Any]) -> None:
    universe_id = row["universe_id"]

    def tensor_shape(item: Any, *, label: str) -> list[int]:
        if not isinstance(item, dict) or item.get("finite") is not True:
            raise ValueError(f"missing/non-finite {label}: {universe_id}")
        raw_shape = item.get("shape")
        if not isinstance(raw_shape, list) or len(raw_shape) != 4:
            raise ValueError(f"{label} is not an exact 4-D tensor: {universe_id}")
        shape = [
            exact_int(value, label=f"{universe_id}.{label}.shape[{index}]")
            for index, value in enumerate(raw_shape)
        ]
        if any(value <= 0 for value in shape) or not str(item.get("dtype", "")):
            raise ValueError(f"invalid {label} shape/dtype: {universe_id}: {shape}")
        return shape

    try:
        expected_indices = [int(value) for value in json.loads(row["sample_streaming_indices_json"])]
        feature_expected = [int(value) for value in feature.get("expected_streaming_indices", [])]
        feature_actual = [int(value) for value in feature.get("actual_streaming_indices", [])]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid streaming-index sample ledger: {universe_id}") from exc
    stream_shape = tensor_shape(feature.get("streaming_tensor"), label="streaming_tensor")
    expected_stream_shape = STREAMING_TOKEN_SHAPES.get(row["model"])
    expected_input_length = STREAMING_INPUT_LENGTHS.get(row["model"])
    full_identity = (
        row["campaign"], row["model_variant"], row["protocol"], row["model"]
    )
    positions = list(range(len(expected_indices)))
    if expected_input_length is not None and expected_input_length > len(expected_indices) and positions:
        positions[-1] = expected_input_length - 1
    expected_vfm_indices = (
        torch_round_indices(positions, expected_input_length, expected_stream_shape[0])
        if expected_input_length is not None and expected_stream_shape is not None
        else []
    )
    expected_model_contract = {
        "model": row["model"],
        "full_identity": list(full_identity),
        "expected_token_shape": expected_stream_shape,
        "input_length": expected_input_length,
        "valid_length": len(expected_indices),
        "pad_mode": "repeat_tail",
        "derived_vfm_indices": expected_vfm_indices,
    }
    if (
        expected_stream_shape is None
        or full_identity != STREAMING_FULL_IDENTITIES.get(row["model"])
        or stream_shape != expected_stream_shape
        or expected_stream_shape[-1]
        != exact_int(row["video_channels"], label=f"{universe_id}.video_channels")
        or feature.get("streaming_model_contract") != expected_model_contract
        or feature_expected != expected_indices
        or feature_actual != expected_indices
        or len(expected_indices) != exact_int(row["prefix_len"], label=f"{universe_id}.prefix_len")
        or exact_int(feature.get("window_id"), label=f"{universe_id}.feature.window_id")
        != exact_int(row["sample_window_id"], label=f"{universe_id}.sample_window_id")
        or exact_int(feature.get("tail"), label=f"{universe_id}.feature.tail")
        != exact_int(row["sample_tail"], label=f"{universe_id}.sample_tail")
    ):
        raise ValueError(f"streaming tensor/sample shape-lineage mismatch: {universe_id}")
    if row.get("target_feat_root"):
        target_shape = tensor_shape(feature.get("target_tensor"), label="target_tensor")
        try:
            target_indices = [int(value) for value in feature.get("target_indices", [])]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid target-index ledger: {universe_id}") from exc
        if (
            not target_indices
            or target_shape[0] != len(target_indices)
            or target_shape[1:] != expected_stream_shape[1:]
            or target_indices.count(exact_int(
                row["sample_target_raw_id"], label=f"{universe_id}.sample_target_raw_id"
            )) != 1
            or exact_int(
                feature.get("expected_target_raw_id"),
                label=f"{universe_id}.feature.expected_target_raw_id",
            )
            != exact_int(
                row["sample_target_raw_id"], label=f"{universe_id}.sample_target_raw_id"
            )
        ):
            raise ValueError(f"target tensor shape/identity mismatch: {universe_id}")


def validate_feature_artifacts_after_evaluator(
    feature: dict[str, Any], postflight_value: Any, *, universe_id: str
) -> None:
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
    by_path: dict[str, dict[str, Any]] = {}
    for name, item in candidates:
        path = str(item.get("materialized_path", ""))
        digest = str(item.get("sha256", ""))
        size = exact_int(item.get("size_bytes"), label=f"{universe_id}.{name}.size_bytes")
        if not path or not is_sha256(digest) or size <= 0:
            raise ValueError(f"invalid preflight feature artifact {name}: {universe_id}")
        existing = by_path.get(path)
        if existing is None:
            by_path[path] = {"sha256": digest, "size_bytes": size, "names": [name]}
        elif existing["sha256"] == digest and existing["size_bytes"] == size:
            existing["names"].append(name)
        else:
            raise ValueError(f"conflicting preflight feature artifact path: {universe_id}")
    expected_checks = [
        {
            "path": path,
            "artifact_names": value["names"],
            "expected_sha256": value["sha256"],
            "actual_sha256": value["sha256"],
            "expected_size_bytes": value["size_bytes"],
            "actual_size_bytes": value["size_bytes"],
            "ok": True,
        }
        for path, value in sorted(by_path.items())
    ]
    expected = {
        "ok": bool(expected_checks),
        "checks": expected_checks,
        "unique_artifacts_checked": len(expected_checks),
    }
    if not expected_checks or postflight_value != expected:
        raise ValueError(f"postflight feature artifact re-hash mismatch: {universe_id}")


def validate_summary_strict_identity(
    row: dict[str, str], summary: dict[str, Any], postflight: dict[str, Any]
) -> None:
    universe_id = row["universe_id"]
    expected_values = {
        "probe_type": row["probe"],
        "vfm_name": row["model"],
        "feature_layer": row["feature_layer"],
        "feat_postfix": row["feat_postfix"],
        "feature_timestep": row["feature_timestep"],
        "prefix_len": row["prefix_len"],
        "streaming_feat_root": row["streaming_feat_root"],
        "job_name": Path(row["output_directory"]).name,
        "q00_universe_id": universe_id,
        "q00_campaign": row["campaign"],
        "q00_model_variant": row["model_variant"],
        "q00_protocol": row["protocol"],
        "q00_manifest_sha256": row["manifest_sha256"],
        "q00_config_sha256": row["experiment_config_sha256"],
        "q00_resolved_config_sha256": row["resolved_config_sha256"],
        "q00_resolved_config_source_kind": row["resolved_config_source_kind"],
    }
    if row.get("target_feat_root"):
        expected_values["target_feat_root"] = row["target_feat_root"]
    checks = {
        key: key in summary and str(summary[key]) == str(value)
        for key, value in expected_values.items()
    }
    checkpoint = summary.get("ckpt_path")
    try:
        checkpoint_ok = checkpoint is not None and Path(str(checkpoint)).resolve() == (
            ROOT / row["checkpoint"]
        ).resolve()
    except (OSError, RuntimeError, ValueError):
        checkpoint_ok = False
    checks["ckpt_path"] = checkpoint_ok
    locked = postflight.get("strict_identity")
    if (
        not all(checks.values())
        or not isinstance(locked, dict)
        or set(locked) != set(checks)
        or any(value is not True for value in locked.values())
    ):
        raise ValueError(
            f"summary/postflight strict identity mismatch {universe_id}: {checks}"
        )


def allowed_execution_shards(bundle_metadata: dict[str, Any]) -> dict[str, dict[str, str]]:
    source_hashes = bundle_metadata.get("source_hashes", {})
    contracts: dict[str, dict[str, str]] = {}
    canary_sources = {
        name: str(digest)
        for name, digest in source_hashes.items()
        if name.startswith("inputs/canary_shards/") and name.endswith(".tsv")
    }
    if len(canary_sources) != 3 or len(set(canary_sources.values())) != 3:
        raise ValueError("bundle does not contain exactly three unique one-row canary shards")
    for source, digest in canary_sources.items():
        if not is_sha256(digest):
            raise ValueError(f"invalid frozen canary shard hash: {source}")
        contracts[digest] = {"classification": "EXACT_ONE_ROW_CANARY", "source": source}
    remaining_source = "inputs/wave_b_remaining_96.tsv"
    remaining_hash = str(source_hashes.get(remaining_source, ""))
    if not is_sha256(remaining_hash) or remaining_hash in contracts:
        raise ValueError("bundle lacks one unique exact remaining-96 shard hash")
    contracts[remaining_hash] = {
        "classification": "EXACT_REMAINING_96",
        "source": remaining_source,
    }
    return contracts


def validate_launcher_canary_gate(
    bundle: Path, shard_sha: str, shard_contracts: dict[str, dict[str, str]]
) -> dict[str, str]:
    contract = shard_contracts.get(shard_sha)
    if contract is None:
        raise ValueError("execution shard is not an exact one-row canary or remaining-96")
    gate_path = (bundle / "launcher" / f"{Path(contract['source']).stem}_canary_gate.json").resolve()
    if not gate_path.is_file():
        raise ValueError(f"missing launcher canary gate: {gate_path}")
    gate = json.loads(gate_path.read_text())
    classification = contract["classification"]
    if (
        not isinstance(gate, dict)
        or gate.get("classification") != classification
        or gate.get("ok") is not True
        or gate.get("shard_sha256") != shard_sha
    ):
        raise ValueError(f"launcher canary gate identity mismatch: {gate_path}")
    if classification == "EXACT_ONE_ROW_CANARY":
        expected = {
            "classification": classification,
            "required": False,
            "ok": True,
            "shard_sha256": shard_sha,
            "frozen_source": contract["source"],
        }
        if gate != expected:
            raise ValueError(f"one-row canary gate contract mismatch: {gate_path}")
    else:
        audit_path = (bundle / "audit/canary_audit.json").resolve()
        results_path = (bundle / "audit/canary_results.csv").resolve()
        lock_path = (bundle / "audit/canary_audit_lock.json").resolve()
        audit_payload = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
        lock_payload = json.loads(lock_path.read_text()) if lock_path.is_file() else {}
        expected_check_keys = {
            "audit_pass",
            "three_successes",
            "zero_failures",
            "audit_sha",
            "results_sha",
            "canary_manifest_sha",
            "bundle_sha",
        }
        expected_gate_keys = {
            "classification",
            "required",
            "ok",
            "shard_sha256",
            "checks",
            "audit_path",
            "audit_sha256",
            "lock_path",
            "lock_sha256",
        }
        checks = gate.get("checks")
        if (
            set(gate) != expected_gate_keys
            or gate.get("required") is not True
            or not isinstance(checks, dict)
            or set(checks) != expected_check_keys
            or any(value is not True for value in checks.values())
            or gate.get("audit_path") != str(audit_path)
            or gate.get("audit_sha256") != sha256(audit_path)
            or gate.get("lock_path") != str(lock_path)
            or gate.get("lock_sha256") != sha256(lock_path)
            or not results_path.is_file()
            or audit_payload.get("ok") is not True
            or exact_int(
                audit_payload.get("successful_summaries"),
                label="canary_audit.successful_summaries",
            )
            != 3
            or exact_int(
                audit_payload.get("failed_summaries"),
                label="canary_audit.failed_summaries",
            )
            != 0
            or lock_payload.get("audit_sha256") != sha256(audit_path)
            or lock_payload.get("results_csv_sha256") != sha256(results_path)
            or lock_payload.get("job_manifest_sha256")
            != sha256(bundle / "inputs/wave_b_canary_3.tsv")
            or lock_payload.get("bundle_json_sha256") != sha256(bundle / "bundle.json")
        ):
            raise ValueError(f"remaining-96 canary gate/lock mismatch: {gate_path}")
    return {
        "path": str(gate_path),
        "sha256": sha256(gate_path),
        "classification": classification,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--scope", choices=("canary", "full"), default="full")
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    bundle_metadata_path = bundle / "bundle.json"
    bundle_metadata = json.loads(bundle_metadata_path.read_text())
    shard_contracts = allowed_execution_shards(bundle_metadata)
    manifest = bundle / "inputs" / ("wave_b_canary_3.tsv" if args.scope == "canary" else "wave_b_99.tsv")
    jobs = rows(manifest)
    expected_count = 3 if args.scope == "canary" else 99
    problems: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    registry_ids: list[str] = []
    frozen_hash_problems = []
    for name, expected_sha in bundle_metadata.get("source_hashes", {}).items():
        if name.startswith(("inputs/", "provenance/")):
            path = bundle / name
        else:
            path = ROOT / name
        actual_sha = sha256(path) if path.is_file() else ""
        if actual_sha != expected_sha:
            frozen_hash_problems.append({"path": str(path), "expected": expected_sha, "actual": actual_sha})
    protected_before_path = bundle / "inputs/protected_before.json"
    protected_before = json.loads(protected_before_path.read_text())
    protected_drift = []
    for relative, metadata in protected_before.items():
        path = ROOT / relative
        actual_sha = sha256(path) if path.is_file() else ""
        if actual_sha != metadata.get("sha256"):
            protected_drift.append({
                "path": relative,
                "expected": metadata.get("sha256", ""),
                "actual": actual_sha,
            })
    expected_ids_path = bundle / "inputs/expected_registry_ids.json"
    frozen_expected_ids = json.loads(expected_ids_path.read_text())
    for row in jobs:
        output = ROOT / row["output_directory"] / "eval"
        preflight_path = output / "q00_preflight.json"
        postflight_path = output / "q00_postflight.json"
        command_path = output / "q00_command.json"
        summary_path = ROOT / row["summary_path"]
        failure_path = output / "q00_failure.json"
        record: dict[str, Any] = {
            "universe_id": row["universe_id"],
            "model": row["model"],
            "family": row["family"],
            "probe": row["probe"],
            "layer": row["layer"],
            "prefix": row["prefix"],
            "is_canary": row["is_canary"],
            "summary_path": row["summary_path"],
            "status": "FAIL",
        }
        try:
            if failure_path.exists():
                raise ValueError(f"failure marker exists: {failure_path}")
            for path in (preflight_path, postflight_path, command_path, summary_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            preflight = json.loads(preflight_path.read_text())
            postflight = json.loads(postflight_path.read_text())
            command = json.loads(command_path.read_text())
            summary = json.loads(summary_path.read_text())
            if preflight.get("ok") is not True or postflight.get("ok") is not True:
                raise ValueError("preflight/postflight not PASS")
            identity = preflight.get("full_identity", {})
            for key in ("campaign", "model", "model_variant", "protocol", "family", "probe", "layer", "prefix", "horizon", "seed"):
                if str(identity.get(key)) != row[key]:
                    raise ValueError(f"preflight identity mismatch: {key}")
            if preflight.get("manifest", {}).get("sha256") != row["manifest_sha256"]:
                raise ValueError("manifest SHA mismatch")
            if preflight.get("manifest", {}).get("frozen_path") != row["frozen_manifest"]:
                raise ValueError("evaluator did not bind the frozen manifest copy")
            if (
                int(preflight.get("manifest", {}).get("train_val_sample_overlap_n", -1)) != 0
                or int(preflight.get("manifest", {}).get("train_val_scene_overlap_n", -1)) != 0
            ):
                raise ValueError("manifest train/val leakage gate failed")
            if preflight.get("checkpoint", {}).get("sha256") != row["checkpoint_sha256"]:
                raise ValueError("checkpoint SHA mismatch")
            if postflight.get("checkpoint_after_evaluator", {}).get("sha256") != row["checkpoint_sha256"]:
                raise ValueError("checkpoint changed during/after evaluator")
            if preflight.get("clean_completion", {}).get("ok") is not True:
                raise ValueError("strict clean/final training-completion proof missing")
            validate_local_artifacts(bundle, row, preflight.get("local_artifacts"))
            runtime_before = preflight.get("frozen_runtime_before_evaluator", {})
            runtime_after = postflight.get("frozen_runtime_after_evaluator", {})
            if runtime_before.get("ok") is not True or runtime_after.get("ok") is not True:
                raise ValueError("frozen runtime before/after evaluator gate missing")
            if (
                runtime_before.get("inventory_actual_sha256")
                != bundle_metadata["source_hashes"]["provenance/runtime_inventory.json"]
                or runtime_after.get("inventory_actual_sha256")
                != bundle_metadata["source_hashes"]["provenance/runtime_inventory.json"]
            ):
                raise ValueError("runtime inventory hash differs from frozen bundle")
            execution_hashes = preflight.get("execution_hashes", {})
            shard_sha = str(execution_hashes.get("shard_tsv", ""))
            shard_contract = shard_contracts.get(shard_sha)
            expected_shard_class = (
                "EXACT_ONE_ROW_CANARY"
                if row["is_canary"] == "true"
                else "EXACT_REMAINING_96"
            )
            expected_shard_source = (
                f"inputs/canary_shards/canary_{row['model']}.tsv"
                if row["is_canary"] == "true"
                else "inputs/wave_b_remaining_96.tsv"
            )
            if not (
                execution_hashes.get("bundle_json") == sha256(bundle_metadata_path)
                and execution_hashes.get("script_inventory") == bundle_metadata["source_hashes"]["provenance/script_inventory.json"]
                and execution_hashes.get("runtime_inventory") == bundle_metadata["source_hashes"]["provenance/runtime_inventory.json"]
                and shard_contract is not None
                and shard_contract["classification"] == expected_shard_class
                and shard_contract["source"] == expected_shard_source
            ):
                raise ValueError("execution bundle/shard hash chain mismatch")
            canary_gate = validate_launcher_canary_gate(bundle, shard_sha, shard_contracts)
            feature = preflight.get("feature", {})
            if feature.get("sample_id") != row["sample_id"]:
                raise ValueError("feature sample ID mismatch")
            expected_aoss_cache = (
                f"/mnt/local_cache/inscene/aoss_cache/{row['run_tag']}/shared"
            )
            if (
                feature.get("aoss_cache_dir") != expected_aoss_cache
                or feature.get("streaming_root") != row["streaming_feat_root"]
                or feature.get("streaming_root_kind") != "remote_uri_materialized_for_content_audit"
                or not content_audit_ok(feature.get("prefix_index_materialized"))
                or not content_audit_ok(feature.get("streaming_tensor"))
                or feature.get("streaming_tensor", {}).get("finite") is not True
            ):
                raise ValueError("streaming feature non-finite/missing")
            if row["target_feat_root"] and (
                feature.get("target_root") != row["target_feat_root"]
                or feature.get("target_root_kind") != "remote_uri_materialized_for_content_audit"
                or not content_audit_ok(feature.get("target_indices_materialized"))
                or not content_audit_ok(feature.get("target_tensor"))
                or feature.get("target_tensor", {}).get("finite") is not True
            ):
                raise ValueError("target feature non-finite/missing")
            validate_feature_tensor_contract(row, feature)
            validate_full_streaming_index(row, feature)
            validate_feature_artifacts_after_evaluator(
                feature,
                postflight.get("feature_artifacts_after_evaluator"),
                universe_id=row["universe_id"],
            )
            env = command.get("identity_env", {})
            expected_env = expected_command_env(bundle, row)
            if (
                not isinstance(env, dict)
                or set(env) != set(expected_env) | {"CUDA_VISIBLE_DEVICES"}
                or any(env.get(key) != value for key, value in expected_env.items())
                or re.fullmatch(r"[0-9]+", str(env.get("CUDA_VISIBLE_DEVICES", ""))) is None
                or feature.get("aoss_cache_dir") != env.get("AOSS_CACHE_DIR")
            ):
                raise ValueError("command identity mismatch")
            validate_environment_policy(
                command.get("environment_policy"), expected_env, universe_id=row["universe_id"]
            )
            evaluator = (
                bundle / "provenance/runtime_tree/vidfm3d/eval_a1_streaming.py"
                if row["evaluator"] == "a1"
                else bundle / "provenance/runtime_tree/vidfm3d/eval_diag.py"
            )
            argv = command.get("argv", [])
            argv_sha256 = (
                hashlib.sha256("\0".join(argv).encode()).hexdigest()
                if isinstance(argv, list) and all(isinstance(item, str) for item in argv)
                else ""
            )
            if (
                command.get("cwd") != str(bundle / "provenance/runtime_tree")
                or len(argv) < 2
                or Path(argv[0]).resolve() != FROZEN_PYTHON
                or Path(argv[1]).resolve() != evaluator.resolve()
                or command.get("frozen_evaluator_sha256") != sha256(evaluator)
                or command.get("argv_sha256") != argv_sha256
            ):
                raise ValueError("command did not execute the frozen evaluator/runtime")
            validate_summary_strict_identity(row, summary, postflight)
            n = int(summary.get("n_samples", summary.get("n", 0)))
            manifest_n = int(row["expected_n"])
            expected_result_n = int(row["expected_result_n"])
            if int(postflight.get("n", -1)) != n:
                raise ValueError("summary/postflight n mismatch")
            if n != expected_result_n:
                raise ValueError(f"summary n differs from frozen family contract: {n} != {expected_result_n}")
            if row["family"] == "B2":
                if not (0 < n <= manifest_n):
                    raise ValueError("B2 construct-valid subset n mismatch")
            elif n != manifest_n:
                raise ValueError("non-B2 summary must cover the complete manifest val set")
            cardinality = postflight.get("cardinality", {})
            if cardinality.get("manifest_val_n") != manifest_n:
                raise ValueError("postflight manifest cardinality mismatch")
            if row["evaluator"] == "diag":
                if (
                    int(cardinality.get("loader_n_samples", -1)) != manifest_n
                    or int(cardinality.get("valid_record_n", -1)) != n
                    or int(cardinality.get("invalid_record_n", -1)) != manifest_n - n
                ):
                    raise ValueError("postflight loader/valid/invalid accounting mismatch")
                if row["family"] != "B2" and int(cardinality.get("invalid_record_n", -1)) != 0:
                    raise ValueError("non-B2 evaluation dropped manifest rows")
            lineage = postflight.get("lineage", {})
            if lineage.get("loader_manifest_ids_ordered_sha256") != row["manifest_val_ordered_ids_sha256"]:
                raise ValueError("loader manifest ordered identity differs from frozen manifest")
            expected = json.loads(row["expected_outputs_json"])
            actual_metrics = {item["registry_id"]: item for item in postflight.get("metrics", [])}
            if set(actual_metrics) != {item["registry_id"] for item in expected}:
                raise ValueError("metric/registry ID schema mismatch")
            for item in expected:
                actual = actual_metrics[item["registry_id"]]
                if actual.get("metric_name") != item["metric_name"] or not math.isfinite(float(actual["metric_value"])):
                    raise ValueError(f"metric contract failure: {item}")
                registry_ids.append(item["registry_id"])
            current_summary_sha = sha256(summary_path)
            if current_summary_sha != postflight.get("summary_sha256"):
                raise ValueError("summary changed after postflight")
            prediction = postflight.get("prediction", {})
            prediction_path = ROOT / row["prediction_path"]
            if row["prediction_required"] == "true":
                if not prediction_path.is_file() or sha256(prediction_path) != prediction.get("sha256"):
                    raise ValueError("prediction missing or changed after postflight")
                if int(prediction.get("n", -1)) != n:
                    raise ValueError("prediction n mismatch")
                if int(prediction.get("frozen_manifest_identity_records_checked", -1)) != n:
                    raise ValueError("prediction-to-frozen-manifest identity audit incomplete")
                if (
                    int(prediction.get("checked_finite_tensors", 0))
                    + int(prediction.get("checked_finite_numeric_scalars", 0))
                    < 1
                ):
                    raise ValueError("prediction finite audit did not inspect numeric payloads")
                if prediction.get("manifest_ids_ordered_sha256") != lineage.get("valid_manifest_ids_ordered_sha256"):
                    raise ValueError("prediction and summary valid-ID order differ")
                if row["family"] == "B2":
                    expected_signature = row["expected_valid_subset_signature_sha256"]
                    if not expected_signature or prediction.get("valid_subset_signature_sha256") != expected_signature:
                        raise ValueError("B2 construct-valid ordered signature mismatch")
                elif prediction.get("manifest_ids_ordered_sha256") != row["manifest_val_ordered_ids_sha256"]:
                    raise ValueError("non-B2 prediction lineage differs from complete manifest")
            cuda_contract = {
                "execution_device": str(summary.get("execution_device", "")).startswith("cuda"),
                "cuda_available": summary.get("cuda_available") is True,
                "cuda_device_count": int(summary.get("cuda_device_count", 0)) == 1,
                "cuda_device_name": bool(summary.get("cuda_device_name")),
                "CUDA_VISIBLE_DEVICES": str(summary.get("CUDA_VISIBLE_DEVICES", "")) == str(env.get("CUDA_VISIBLE_DEVICES", "")),
            }
            if not all(cuda_contract.values()):
                raise ValueError(f"strict CUDA execution contract mismatch: {cuda_contract}")
            for name in json.loads(row["required_summary_metrics_json"]):
                if name not in summary:
                    raise ValueError(f"current summary missing contracted field {name}")
                value = float(summary[name])
                if not math.isfinite(value):
                    raise ValueError(f"current summary contracted field non-finite {name}")
                locked = float(postflight.get("required_summary_values", {}).get(name, float("nan")))
                if value != locked:
                    raise ValueError(f"summary/postflight value mismatch {name}")
            record.update({
                "status": "PASS",
                "n": n,
                "summary_sha256": current_summary_sha,
                "prediction_sha256": prediction.get("sha256", ""),
                "preflight_sha256": sha256(preflight_path),
                "postflight_sha256": sha256(postflight_path),
                "command_sha256": sha256(command_path),
                "manifest_sha256": row["manifest_sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "execution_shard_sha256": shard_sha,
                "canary_gate_path": canary_gate["path"],
                "canary_gate_sha256": canary_gate["sha256"],
                "canary_gate_classification": canary_gate["classification"],
                "expected_registry_ids": ";".join(item["registry_id"] for item in expected),
                "metrics_json": json.dumps(postflight["metrics"], sort_keys=True, separators=(",", ":")),
            })
        except Exception as exc:
            record["error"] = repr(exc)
            problems.append({"universe_id": row["universe_id"], "error": repr(exc)})
        results.append(record)

    counts = Counter(item["status"] for item in results)
    duplicate_universe = [key for key, count in Counter(item["universe_id"] for item in results).items() if count > 1]
    duplicate_registry = [key for key, count in Counter(registry_ids).items() if count > 1]
    model_counts = Counter(item["model"] for item in results if item["status"] == "PASS")
    expected_atomic = 3
    if args.scope == "canary":
        expected_atomic = sum(len(json.loads(row["expected_outputs_json"])) for row in jobs)
    else:
        expected_atomic = 105
    expected_id_contract_ok = (
        sorted(registry_ids) == sorted(frozen_expected_ids)
        if args.scope == "full"
        else set(registry_ids) <= set(frozen_expected_ids)
    )
    ok = (
        len(jobs) == expected_count
        and counts.get("PASS", 0) == expected_count
        and not problems
        and not duplicate_universe
        and not duplicate_registry
        and len(registry_ids) == expected_atomic
        and not frozen_hash_problems
        and not protected_drift
        and expected_id_contract_ok
    )
    if args.scope == "canary":
        ok = ok and model_counts == Counter({"lingbot": 1, "minimax_h3": 1, "wan": 1})
    audit = {
        "ok": ok,
        "scope": args.scope,
        "expected_universe_ids": expected_count,
        "actual_result_rows": len(results),
        "successful_summaries": counts.get("PASS", 0),
        "failed_summaries": counts.get("FAIL", 0),
        "expected_atomic_registry_ids": expected_atomic,
        "actual_atomic_registry_ids": len(registry_ids),
        "expected_id_contract_ok": expected_id_contract_ok,
        "duplicate_universe_ids": duplicate_universe,
        "duplicate_registry_ids": duplicate_registry,
        "non_finite_metrics": 0 if ok else sum("finite" in item.get("error", "") for item in problems),
        "manifest_mismatches": sum("manifest" in item.get("error", "").lower() for item in problems),
        "feature_mismatches": sum("feature" in item.get("error", "").lower() for item in problems),
        "model_pass_counts": dict(model_counts),
        "problems": problems,
        "frozen_hash_problems": frozen_hash_problems,
        "protected_source_of_truth_drift": protected_drift,
        "input_hashes": {
            "bundle_json": sha256(bundle_metadata_path),
            "job_manifest": sha256(manifest),
            "expected_registry_ids": sha256(expected_ids_path),
            "script_inventory": sha256(bundle / "provenance/script_inventory.json"),
            "runtime_inventory": sha256(bundle / "provenance/runtime_inventory.json"),
        },
        "results": results,
    }
    out = bundle / "audit"
    write_json(out / f"{args.scope}_audit.json", audit)
    fields = sorted({key for item in results for key in item})
    with (out / f"{args.scope}_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    result_csv = out / f"{args.scope}_results.csv"
    lock = {
        "scope": args.scope,
        "audit_sha256": sha256(out / f"{args.scope}_audit.json"),
        "results_csv_sha256": sha256(result_csv),
        "job_manifest_sha256": sha256(manifest),
        "expected_registry_ids_sha256": sha256(expected_ids_path),
        "expected_registry_ids": sorted(
            item["registry_id"] for row in jobs for item in json.loads(row["expected_outputs_json"])
        ),
        "bundle_json_sha256": sha256(bundle_metadata_path),
    }
    write_json(out / f"{args.scope}_audit_lock.json", lock)
    print(json.dumps({key: audit[key] for key in ("ok", "scope", "expected_universe_ids", "successful_summaries", "failed_summaries", "actual_atomic_registry_ids")}, sort_keys=True))
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
