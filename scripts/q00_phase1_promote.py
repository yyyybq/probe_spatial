#!/usr/bin/env python3
"""Promote an exact, audited Q00 Phase-1 Wave-B bundle.

This importer is deliberately bundle-specific.  It never scans historical run
directories and never calls the legacy evaluation-summary updater.  The only
accepted inputs are the frozen 99-row TSV, its exact expected-ID list, the
frozen reconciliation/wave tables, and a full audit that cryptographically
binds every summary and prediction artifact.

Promotion is append-only at the logical-row level.  Existing registry and
canonical rows must remain byte-for-byte equal after CSV parsing and in the
same order.  The protected files are replaced only after a second hash check
while holding an exclusive lock.  A journal and pre-images support rollback on
any caught failure during the multi-file commit.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NAMES = (
    "all_results.csv",
    "a1_results.csv",
    "a2_results.csv",
    "b2_results.csv",
    "c1_results.csv",
    "c2_results.csv",
    "c3_results.csv",
)
PROTECTED_RELATIVE = (
    "experiments/registry.csv",
    *(f"results/canonical/{name}" for name in CANONICAL_NAMES),
)
FAMILY_FILE = {
    "A1": "a1_results.csv",
    "A2": "a2_results.csv",
    "B2": "b2_results.csv",
    "C1": "c1_results.csv",
    "C2": "c2_results.csv",
    "C3": "c3_results.csv",
}
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
REGISTRY_FIELDS = [
    "experiment_id",
    "family",
    "probe",
    "model",
    "backend",
    "model_variant",
    "layer",
    "prefix",
    "horizon",
    "seed",
    "train_split",
    "eval_split",
    "dataset_manifest",
    "config",
    "checkpoint",
    "launcher_job_id",
    "raw_artifact_path",
    "canonical_metric_path",
    "metric_name",
    "metric_value",
    "status",
    "status_reason",
    "supersedes",
    "git_commit",
    "notes",
]
SIMPLE_CANONICAL_FIELDS = [
    "experiment_id",
    "family",
    "probe",
    "model",
    "model_variant",
    "layer",
    "prefix",
    "horizon",
    "seed",
    "eval_split",
    "metric_name",
    "metric_value",
    "status",
    "raw_artifact_path",
]
CANONICAL_FIELDS = {
    "all_results.csv": [
        "family",
        "probe",
        "model",
        "layer",
        "prefix",
        "metric_name",
        "metric_value",
        "experiment_id",
        "raw_artifact_path",
    ],
    "a1_results.csv": SIMPLE_CANONICAL_FIELDS,
    "a2_results.csv": SIMPLE_CANONICAL_FIELDS,
    "b2_results.csv": [
        "experiment_id",
        "model",
        "model_variant",
        "layer",
        "prefix",
        "seed",
        "eval_split",
        "angular_error_mean",
        "angular_error_median",
        "top1",
        "top3",
        "mean_log_dist_err",
        "status",
        "raw_artifact_path",
    ],
    "c1_results.csv": [
        "experiment_id",
        "model",
        "layer",
        "prefix",
        "seed",
        "eval_split",
        "global_R@1",
        "global_R@5",
        "no_action_R@1",
        "shuffled_action_R@1",
        "last_observation_R@1",
        "action_gain_vs_no_action",
        "action_gain_vs_shuffle",
        "gain_over_copy",
        "action_gain_cos",
        "status",
        "raw_artifact_path",
    ],
    "c2_results.csv": [
        "experiment_id",
        "model",
        "layer",
        "prefix",
        "horizon",
        "seed",
        "eval_split",
        "metric_name",
        "metric_value",
        "global_R@5",
        "final_pose_error",
        "drift_rate",
        "loop_closure_error",
        "status",
        "raw_artifact_path",
    ],
    "c3_results.csv": [
        "experiment_id",
        "model",
        "layer",
        "prefix",
        "horizon",
        "seed",
        "eval_split",
        "metric_name",
        "metric_value",
        "status",
        "raw_artifact_path",
    ],
}
IDENTITY_FIELDS = (
    "campaign",
    "model_variant",
    "protocol",
    "family",
    "probe",
    "model",
    "layer",
    "prefix",
    "horizon",
    "seed",
)
EXPECTED_PARENT_ROWS = 99
EXPECTED_ATOMIC_ROWS = 105
STRICT_CHECKPOINT_STATUS = "STRICT_CLEAN_FINAL_READY"
STRICT_CLEAN_COMPLETION_STATUS = (
    "strict_pass=true;clean_marker=true;bad_marker=false;epoch_complete=true"
)
DATA_ROOT = "aoss://baiqiao/InsScene-15K/data"
SANITIZED_ENV_PREFIXES = ("INSCENE_", "PROBE_", "FEAT_", "Q00_")
ENVIRONMENT_POLICY = (
    "strip all inherited INSCENE_/PROBE_/FEAT_/Q00_ then add exact frozen values only"
)
RESOLVED_CONFIG_SOURCE_KIND = "run_manifest.config_resolved"
FROZEN_PYTHON = Path(
    "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
).resolve()


class PromotionError(RuntimeError):
    """Fail-closed validation or promotion error."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise PromotionError(f"missing JSON input: {path}")
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - error path is message-only
        raise PromotionError(f"invalid JSON input {path}: {exc}") from exc


def read_csv(path: Path, delimiter: str = ",") -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise PromotionError(f"missing CSV/TSV input: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise PromotionError(f"atomic-copy source missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.q00-phase1-{os.getpid()}")
    shutil.copy2(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def clean(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower().replace("@", "at").replace("/", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_.+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def exact_registry_id(job: Mapping[str, str], output: Mapping[str, Any]) -> str:
    """Stable ID containing campaign, real model variant, and protocol."""
    return "_".join(
        [
            "q00",
            clean(job["family"]),
            clean(output["probe_label"]),
            clean(job["model"]),
            clean(job["campaign"]),
            clean(job["model_variant"]),
            clean(job["protocol"]),
            f"l{clean(job['layer'])}",
            f"p{clean(job['prefix'])}",
            f"h{clean(job['horizon'])}",
            f"s{clean(job['seed'])}",
        ]
    )


def safe_project_path(value: str, *, label: str) -> Path:
    if not value:
        raise PromotionError(f"empty project path for {label}")
    raw = Path(value)
    if raw.is_absolute():
        raise PromotionError(f"absolute path forbidden for {label}: {value}")
    resolved = (ROOT / raw).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise PromotionError(f"path escapes project root for {label}: {value}") from exc
    return resolved


def ensure_under(path: Path, parent: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise PromotionError(f"{label} is outside frozen output root: {path}") from exc


def numeric(value: Any, *, label: str, required: bool = True) -> float | None:
    if value is None or str(value).strip() == "":
        if required:
            raise PromotionError(f"missing numeric value: {label}")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        if required:
            raise PromotionError(f"non-numeric value for {label}: {value!r}") from exc
        return None
    if not math.isfinite(number):
        if required:
            raise PromotionError(f"non-finite value for {label}: {number}")
        return None
    return number


def number_text(value: float) -> str:
    return format(float(value), ".17g")


def exact_int(value: Any, *, label: str) -> int:
    number = numeric(value, label=label, required=True)
    assert number is not None
    if not number.is_integer():
        raise PromotionError(f"non-integral value for {label}: {value!r}")
    return int(number)


def torch_round_indices(
    positions: list[int], input_length: int, token_length: int
) -> list[int]:
    if input_length <= 1 or token_length <= 0:
        raise PromotionError("invalid model-token index dimensions")
    return [
        max(0, min(token_length - 1, int(round(
            float(position) / float(input_length - 1) * float(token_length - 1)
        ))))
        for position in positions
    ]


def numbers_equal(left: Any, right: Any) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def content_audit_ok(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    digest = str(value.get("sha256", ""))
    try:
        size = exact_int(value.get("size_bytes"), label="materialized_content.size_bytes")
    except PromotionError:
        return False
    return bool(re.fullmatch(r"[0-9a-f]{64}", digest)) and size > 0


def is_sha256(value: Any) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value)) is not None


def expected_command_env(bundle: Path, job: Mapping[str, str]) -> dict[str, str]:
    runtime_tree = (bundle / "provenance/runtime_tree").resolve()
    output = safe_project_path(job["output_directory"], label="output_directory")
    expected = {
        "PYTHONPATH": str(runtime_tree),
        "PROJECT_ROOT": str(ROOT),
        "Q00_REQUIRE_CUDA": "1",
        "Q00_RUNTIME_TREE": str(runtime_tree),
        "INSCENE_DATA_ROOT": DATA_ROOT,
        "INSCENE_MANIFEST": str(
            safe_project_path(job["frozen_manifest"], label="frozen_manifest")
        ),
        "INSCENE_TRUST_EXPANDED_MANIFEST": "1",
        "INSCENE_STREAMING_FEAT_ROOT": job["streaming_feat_root"],
        "PROBE_FP16_PASSTHROUGH": "0",
        "Q00_UNIVERSE_ID": job["universe_id"],
        "Q00_CAMPAIGN": job["campaign"],
        "Q00_MODEL_VARIANT": job["model_variant"],
        "Q00_PROTOCOL": job["protocol"],
        "Q00_MANIFEST_SHA256": job["manifest_sha256"],
        "Q00_CONFIG_SHA256": job["experiment_config_sha256"],
        "Q00_RESOLVED_CONFIG_SHA256": job["resolved_config_sha256"],
        "Q00_RESOLVED_CONFIG_SOURCE_KIND": job["resolved_config_source_kind"],
        "AOSS_CACHE_DIR": (
            f"/mnt/local_cache/inscene/aoss_cache/{job['run_tag']}/shared"
        ),
        "AOSS_FS_METRICS": str(output / "eval/aoss_metrics.jsonl"),
    }
    if job.get("target_feat_root"):
        expected["INSCENE_TARGET_FEAT_ROOT"] = job["target_feat_root"]
    return expected


def validate_environment_policy(
    policy: Any, expected_env: Mapping[str, str], *, universe_id: str
) -> None:
    if not isinstance(policy, dict):
        raise PromotionError(f"command environment policy is not an object: {universe_id}")
    removed = policy.get("removed_parent_keys")
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
        or policy.get("exact_reintroduced_keys") != expected_reintroduced
    ):
        raise PromotionError(f"command environment policy mismatch: {universe_id}")


def validate_full_streaming_index(job: Mapping[str, str], feature: Mapping[str, Any]) -> None:
    universe_id = job["universe_id"]
    audit = feature.get("full_streaming_index_audit")
    if not isinstance(audit, dict):
        raise PromotionError(f"missing full streaming-index audit: {universe_id}")
    artifacts = audit.get("prefix_index_artifacts")
    if not isinstance(artifacts, list):
        raise PromotionError(f"invalid prefix-index artifact list: {universe_id}")
    unique_n = exact_int(
        audit.get("unique_prefix_indexes_checked"),
        label=f"{universe_id}.unique_prefix_indexes_checked",
    )
    keys: list[tuple[str, str]] = []
    for item in artifacts:
        if (
            not isinstance(item, dict)
            or not str(item.get("source", ""))
            or not str(item.get("scene_id", ""))
            or not str(item.get("source_path", ""))
            or not str(item.get("materialized_path", ""))
            or not content_audit_ok(item)
        ):
            raise PromotionError(f"invalid full prefix-index content audit: {universe_id}")
        keys.append((str(item["source"]), str(item["scene_id"])))
    if (
        audit.get("ok") is not True
        or exact_int(
            audit.get("manifest_val_records_checked"),
            label=f"{universe_id}.manifest_val_records_checked",
        )
        != exact_int(job["expected_n"], label=f"{universe_id}.expected_n")
        or unique_n <= 0
        or unique_n != len(artifacts)
        or len(set(keys)) != len(keys)
        or not is_sha256(audit.get("identity_ledger_sha256"))
        or audit.get("contract")
        != "actual_input_raw_ids==manifest.streaming_indices;valid_length;input_length;repeat_tail;tail_raw"
    ):
        raise PromotionError(f"full streaming-index lineage mismatch: {universe_id}")


def validate_feature_tensor_contract(
    job: Mapping[str, str], feature: Mapping[str, Any]
) -> None:
    universe_id = job["universe_id"]

    def tensor_shape(item: Any, *, label: str) -> list[int]:
        if not isinstance(item, dict) or item.get("finite") is not True:
            raise PromotionError(f"missing/non-finite {label}: {universe_id}")
        raw_shape = item.get("shape")
        if not isinstance(raw_shape, list) or len(raw_shape) != 4:
            raise PromotionError(f"{label} is not an exact 4-D tensor: {universe_id}")
        shape = [
            exact_int(value, label=f"{universe_id}.{label}.shape[{index}]")
            for index, value in enumerate(raw_shape)
        ]
        if any(value <= 0 for value in shape) or not str(item.get("dtype", "")):
            raise PromotionError(f"invalid {label} shape/dtype: {universe_id}: {shape}")
        return shape

    try:
        expected_indices = [
            int(value) for value in json.loads(job["sample_streaming_indices_json"])
        ]
        feature_expected = [
            int(value) for value in feature.get("expected_streaming_indices", [])
        ]
        feature_actual = [
            int(value) for value in feature.get("actual_streaming_indices", [])
        ]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PromotionError(f"invalid streaming-index sample ledger: {universe_id}") from exc
    stream_shape = tensor_shape(feature.get("streaming_tensor"), label="streaming_tensor")
    expected_stream_shape = STREAMING_TOKEN_SHAPES.get(job["model"])
    expected_input_length = STREAMING_INPUT_LENGTHS.get(job["model"])
    full_identity = (
        job["campaign"], job["model_variant"], job["protocol"], job["model"]
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
        "model": job["model"],
        "full_identity": list(full_identity),
        "expected_token_shape": expected_stream_shape,
        "input_length": expected_input_length,
        "valid_length": len(expected_indices),
        "pad_mode": "repeat_tail",
        "derived_vfm_indices": expected_vfm_indices,
    }
    if (
        expected_stream_shape is None
        or full_identity != STREAMING_FULL_IDENTITIES.get(job["model"])
        or stream_shape != expected_stream_shape
        or expected_stream_shape[-1]
        != exact_int(job["video_channels"], label=f"{universe_id}.video_channels")
        or feature.get("streaming_model_contract") != expected_model_contract
        or feature_expected != expected_indices
        or feature_actual != expected_indices
        or len(expected_indices)
        != exact_int(job["prefix_len"], label=f"{universe_id}.prefix_len")
        or exact_int(feature.get("window_id"), label=f"{universe_id}.feature.window_id")
        != exact_int(job["sample_window_id"], label=f"{universe_id}.sample_window_id")
        or exact_int(feature.get("tail"), label=f"{universe_id}.feature.tail")
        != exact_int(job["sample_tail"], label=f"{universe_id}.sample_tail")
    ):
        raise PromotionError(f"streaming tensor/sample shape-lineage mismatch: {universe_id}")
    if job.get("target_feat_root"):
        target_shape = tensor_shape(feature.get("target_tensor"), label="target_tensor")
        try:
            target_indices = [int(value) for value in feature.get("target_indices", [])]
        except (TypeError, ValueError) as exc:
            raise PromotionError(f"invalid target-index ledger: {universe_id}") from exc
        if (
            not target_indices
            or target_shape[0] != len(target_indices)
            or target_shape[1:] != expected_stream_shape[1:]
            or target_indices.count(exact_int(
                job["sample_target_raw_id"], label=f"{universe_id}.sample_target_raw_id"
            )) != 1
            or exact_int(
                feature.get("expected_target_raw_id"),
                label=f"{universe_id}.feature.expected_target_raw_id",
            )
            != exact_int(
                job["sample_target_raw_id"], label=f"{universe_id}.sample_target_raw_id"
            )
        ):
            raise PromotionError(f"target tensor shape/identity mismatch: {universe_id}")


def validate_feature_artifacts_after_evaluator(
    feature: Mapping[str, Any], postflight_value: Any, *, universe_id: str
) -> None:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
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
    if isinstance(full, dict):
        for index, item in enumerate(full.get("prefix_index_artifacts", [])):
            if isinstance(item, dict):
                candidates.append((f"full_prefix_index_{index:04d}", item))
    by_path: dict[str, dict[str, Any]] = {}
    for name, item in candidates:
        path = str(item.get("materialized_path", ""))
        digest = str(item.get("sha256", ""))
        size = exact_int(item.get("size_bytes"), label=f"{universe_id}.{name}.size_bytes")
        if not path or not is_sha256(digest) or size <= 0:
            raise PromotionError(f"invalid preflight feature artifact {name}: {universe_id}")
        previous = by_path.get(path)
        if previous is None:
            by_path[path] = {"sha256": digest, "size_bytes": size, "names": [name]}
        elif previous["sha256"] == digest and previous["size_bytes"] == size:
            previous["names"].append(name)
        else:
            raise PromotionError(f"conflicting preflight feature path: {universe_id}")
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
        raise PromotionError(f"postflight feature artifact re-hash mismatch: {universe_id}")


def execution_shard_contracts(metadata: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    source_hashes = metadata.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise PromotionError("bundle lacks source hashes for execution-shard contract")
    contracts: dict[str, dict[str, str]] = {}
    canary_sources = {
        str(name): str(digest)
        for name, digest in source_hashes.items()
        if str(name).startswith("inputs/canary_shards/") and str(name).endswith(".tsv")
    }
    if len(canary_sources) != 3 or len(set(canary_sources.values())) != 3:
        raise PromotionError("bundle does not contain three unique one-row canary shards")
    for source, digest in canary_sources.items():
        if not is_sha256(digest):
            raise PromotionError(f"invalid frozen canary shard hash: {source}")
        contracts[digest] = {"classification": "EXACT_ONE_ROW_CANARY", "source": source}
    remaining_source = "inputs/wave_b_remaining_96.tsv"
    remaining_hash = str(source_hashes.get(remaining_source, ""))
    if not is_sha256(remaining_hash) or remaining_hash in contracts:
        raise PromotionError("bundle lacks one unique exact remaining-96 shard")
    contracts[remaining_hash] = {
        "classification": "EXACT_REMAINING_96",
        "source": remaining_source,
    }
    return contracts


def validate_launcher_canary_gate(
    bundle: Path,
    shard_sha: str,
    contracts: Mapping[str, Mapping[str, str]],
    record: Mapping[str, Any],
) -> tuple[str, str]:
    contract = contracts.get(shard_sha)
    if contract is None:
        raise PromotionError("execution shard is not an exact canary/remaining-96 shard")
    gate_path = (
        bundle / "launcher" / f"{Path(contract['source']).stem}_canary_gate.json"
    ).resolve()
    gate = read_json(gate_path)
    classification = contract["classification"]
    if (
        gate.get("classification") != classification
        or gate.get("ok") is not True
        or gate.get("shard_sha256") != shard_sha
    ):
        raise PromotionError(f"launcher canary-gate identity mismatch: {gate_path}")
    if classification == "EXACT_ONE_ROW_CANARY":
        expected = {
            "classification": classification,
            "required": False,
            "ok": True,
            "shard_sha256": shard_sha,
            "frozen_source": contract["source"],
        }
        if gate != expected:
            raise PromotionError(f"one-row canary-gate contract mismatch: {gate_path}")
    else:
        audit_path = (bundle / "audit/canary_audit.json").resolve()
        results_path = (bundle / "audit/canary_results.csv").resolve()
        lock_path = (bundle / "audit/canary_audit_lock.json").resolve()
        audit_payload = read_json(audit_path)
        lock_payload = read_json(lock_path)
        checks = gate.get("checks")
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
            raise PromotionError(f"remaining-96 canary gate/lock mismatch: {gate_path}")
    gate_hash = sha256(gate_path)
    if (
        record.get("execution_shard_sha256") != shard_sha
        or record.get("canary_gate_path") != str(gate_path)
        or record.get("canary_gate_sha256") != gate_hash
        or record.get("canary_gate_classification") != classification
    ):
        raise PromotionError("full audit does not bind the per-job launcher canary gate")
    return str(gate_path), gate_hash


def summary_metric(summary: Mapping[str, Any], name: str, *, required: bool = True) -> float | None:
    candidates = [name]
    if name == "Auc_30":
        candidates.append("mean_Auc_30")
    present = [(key, summary[key]) for key in candidates if key in summary]
    if not present:
        return numeric(None, label=f"summary.{name}", required=required)
    values = [numeric(value, label=f"summary.{key}", required=required) for key, value in present]
    finite_values = [value for value in values if value is not None]
    if len(finite_values) > 1 and not all(numbers_equal(finite_values[0], item) for item in finite_values[1:]):
        raise PromotionError(f"conflicting summary aliases for {name}: {present}")
    return finite_values[0] if finite_values else None


def parse_json_list(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise PromotionError(f"{label} is not a JSON list")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PromotionError(f"invalid JSON list in {label}: {exc}") from exc
    if not isinstance(parsed, list):
        raise PromotionError(f"{label} did not decode to a list")
    return parsed


def row_counter(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> Counter[tuple[str, ...]]:
    return Counter(tuple(str(row.get(field, "")) for field in fields) for row in rows)


def file_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256(path) for name, path in paths.items()}


def verify_hashes(paths: Mapping[str, Path], expected: Mapping[str, str], *, label: str) -> None:
    actual = file_hashes(paths)
    if actual != dict(expected):
        mismatches = {
            key: {"expected": expected.get(key, ""), "actual": actual.get(key, "")}
            for key in sorted(set(actual) | set(expected))
            if actual.get(key) != expected.get(key)
        }
        raise PromotionError(f"{label} SHA256 mismatch: {json.dumps(mismatches, sort_keys=True)}")


def protected_paths() -> dict[str, Path]:
    return {relative: ROOT / relative for relative in PROTECTED_RELATIVE}


def frozen_protected_hashes(bundle: Path) -> dict[str, str]:
    data = read_json(bundle / "inputs/protected_before.json")
    if not isinstance(data, dict):
        raise PromotionError("protected_before.json is not an object")
    expected_keys = set(PROTECTED_RELATIVE)
    if set(data) != expected_keys:
        raise PromotionError(
            f"protected file set mismatch: expected={sorted(expected_keys)} actual={sorted(data)}"
        )
    return {key: str(data[key]["sha256"]) for key in PROTECTED_RELATIVE}


def verify_script_inventory(bundle: Path) -> None:
    inventory_path = bundle / "provenance/script_inventory.json"
    inventory = read_json(inventory_path)
    if not isinstance(inventory, list) or not inventory:
        raise PromotionError("empty/invalid frozen script inventory")
    failures = []
    for item in inventory:
        frozen_path = str(item.get("path", ""))
        if "provenance/scripts/" not in frozen_path:
            failures.append({"path": frozen_path, "error": "unrecognized frozen path"})
            continue
        frozen = safe_project_path(frozen_path, label="script_inventory.frozen_path")
        current = ROOT / frozen_path.split("provenance/scripts/", 1)[1]
        actual = sha256(current) if current.is_file() else ""
        frozen_actual = sha256(frozen) if frozen.is_file() else ""
        if actual != item.get("sha256") or frozen_actual != item.get("sha256"):
            failures.append(
                {
                    "path": str(current),
                    "frozen_path": str(frozen),
                    "expected": item.get("sha256", ""),
                    "actual": actual,
                    "frozen_actual": frozen_actual,
                }
            )
    if failures:
        raise PromotionError(f"live scripts do not match frozen bundle: {json.dumps(failures, sort_keys=True)}")


def verify_runtime_inventory(bundle: Path) -> None:
    inventory_path = bundle / "provenance/runtime_inventory.json"
    inventory = read_json(inventory_path)
    if not isinstance(inventory, list) or not inventory:
        raise PromotionError("empty/invalid frozen runtime inventory")
    failures = []
    for item in inventory:
        relative = str(item.get("path", ""))
        frozen_relative = str(item.get("frozen_path", ""))
        if not relative or not frozen_relative:
            failures.append({"path": relative, "error": "missing runtime path"})
            continue
        current = safe_project_path(relative, label="runtime.path")
        frozen = safe_project_path(frozen_relative, label="runtime.frozen_path")
        current_hash = sha256(current) if current.is_file() else ""
        frozen_hash = sha256(frozen) if frozen.is_file() else ""
        expected = str(item.get("sha256", ""))
        expected_frozen = str(item.get("frozen_sha256", ""))
        if not (current_hash == expected == expected_frozen == frozen_hash):
            failures.append(
                {
                    "path": relative,
                    "current": current_hash,
                    "expected": expected,
                    "frozen": frozen_hash,
                    "expected_frozen": expected_frozen,
                }
            )
    if failures:
        raise PromotionError(f"live runtime tree differs from frozen bundle: {json.dumps(failures[:20], sort_keys=True)}")


def validate_frozen_sources(bundle: Path, jobs: Sequence[Mapping[str, str]]) -> None:
    metadata = read_json(bundle / "bundle.json")
    source_root = bundle / "inputs/source_of_truth"
    source_specs = (
        (
            "experiments/task_status_1096_before_expand_20260822.csv",
            "universe_sha256",
        ),
        (
            "results/analysis/q00_1096/universe_1096_reconciled.csv",
            "reconciliation_sha256",
        ),
        (
            "results/analysis/q00_1096/execution_waves.csv",
            "execution_waves_sha256",
        ),
    )
    for relative, key in source_specs:
        path = source_root / relative
        if sha256(path) != metadata.get(key):
            raise PromotionError(f"frozen source hash mismatch: {relative}")
    _, reconciliation_rows = read_csv(
        source_root / "results/analysis/q00_1096/universe_1096_reconciled.csv"
    )
    _, wave_rows = read_csv(source_root / "results/analysis/q00_1096/execution_waves.csv")
    reconciliation = {row["universe_id"]: row for row in reconciliation_rows}
    waves = {row["universe_id"]: row for row in wave_rows}
    if len(reconciliation) != int(metadata.get("universe_rows", -1)):
        raise PromotionError("frozen reconciliation row count does not match bundle metadata")
    for job in jobs:
        universe_id = job["universe_id"]
        source = reconciliation.get(universe_id)
        wave = waves.get(universe_id)
        if source is None or wave is None or wave.get("wave") != "Wave B":
            raise PromotionError(f"job is not an exact frozen Wave-B row: {universe_id}")
        for key in IDENTITY_FIELDS:
            if str(source.get(key, "")) != str(job.get(key, "")):
                raise PromotionError(
                    f"frozen full-identity mismatch for {universe_id}: {key} "
                    f"job={job.get(key)!r} source={source.get(key)!r}"
                )
        if source.get("checkpoint_path") != job.get("checkpoint"):
            raise PromotionError(f"checkpoint provenance mismatch: {universe_id}")
        if source.get("manifest") != job.get("manifest"):
            raise PromotionError(f"manifest provenance mismatch: {universe_id}")
        source_job_contract = {
            "manifest_sha256": "manifest_sha256",
            "checkpoint_status": "checkpoint_status",
            "clean_completion_status": "clean_completion_status",
            "expected_max_epochs": "expected_max_epochs",
            "expected_final_epoch": "expected_final_epoch",
            "checkpoint_step": "expected_final_step",
        }
        for source_key, job_key in source_job_contract.items():
            if str(source.get(source_key, "")) != str(job.get(job_key, "")):
                raise PromotionError(
                    f"frozen job/provenance contract mismatch for {universe_id}: "
                    f"source.{source_key}={source.get(source_key)!r} "
                    f"job.{job_key}={job.get(job_key)!r}"
                )
        logical_layer = exact_int(
            source.get("layer"), label=f"{universe_id}.source.logical_layer"
        )
        expected_feature_layer = (
            logical_layer - 1
            if (source.get("model"), logical_layer)
            in {("lingbot", 48), ("minimax_h3", 50)}
            else logical_layer
        )
        expected_timestep = exact_int(
            source.get("default_timestep"),
            label=f"{universe_id}.source.default_timestep",
        )
        expected_postfix = f"_t{expected_timestep}_layer{expected_feature_layer}"
        if (
            exact_int(job.get("feature_layer"), label=f"{universe_id}.feature_layer")
            != expected_feature_layer
            or exact_int(
                job.get("feature_timestep"), label=f"{universe_id}.feature_timestep"
            )
            != expected_timestep
            or job.get("feat_postfix") != expected_postfix
        ):
            raise PromotionError(
                f"logical/feature-layer or default-timestep provenance mismatch: {universe_id}"
            )
        if source.get("registry_variant_collision_group"):
            raise PromotionError(f"collision-group row cannot use automatic promotion: {universe_id}")


def output_contract(job: Mapping[str, str]) -> list[tuple[str, str]]:
    family, probe = job["family"], job["probe"]
    if family == "A1" and probe == "streaming_depth":
        return [("streaming_depth/depth", "mean_loss_depth"), ("streaming_depth/camera", "Auc_30")]
    if family == "A1" and probe == "streaming_mask":
        return [("streaming_mask/mask", "mean_loss_identity")]
    metrics = {
        "A2": "balanced_acc",
        "B2": "mean_ang_err_deg",
        "C1": "global_R@1",
        "C2": "global_R@1",
        "C3": "intervention_validity",
    }
    if family not in metrics:
        raise PromotionError(f"unsupported family/probe output contract: {family}/{probe}")
    return [(probe, metrics[family])]


def expected_outputs(jobs: Sequence[Mapping[str, str]]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ordered_ids: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for job in jobs:
        outputs = parse_json_list(job.get("expected_outputs_json", ""), label=f"{job['universe_id']}.expected_outputs")
        if not outputs:
            raise PromotionError(f"no expected output for {job['universe_id']}")
        frozen_contract = [
            (str(item.get("probe_label", "")), str(item.get("metric_name", "")))
            for item in outputs
            if isinstance(item, dict)
        ]
        if frozen_contract != output_contract(job):
            raise PromotionError(
                f"frozen output contract mismatch for {job['universe_id']}: {frozen_contract}"
            )
        for raw_output in outputs:
            if not isinstance(raw_output, dict):
                raise PromotionError(f"non-object expected output for {job['universe_id']}")
            output = {str(key): value for key, value in raw_output.items()}
            for required in ("probe_label", "metric_name", "registry_id"):
                if not output.get(required):
                    raise PromotionError(f"missing {required} in expected output for {job['universe_id']}")
            derived = exact_registry_id(job, output)
            if output["registry_id"] != derived:
                raise PromotionError(
                    f"non-exact expected registry ID for {job['universe_id']}: "
                    f"frozen={output['registry_id']} required={derived}"
                )
            if derived in by_id:
                raise PromotionError(f"duplicate exact registry ID in frozen jobs: {derived}")
            combined = {**job, **output, "registry_id": derived}
            by_id[derived] = combined
            ordered_ids.append(derived)
    if len(ordered_ids) != EXPECTED_ATOMIC_ROWS:
        raise PromotionError(f"expected {EXPECTED_ATOMIC_ROWS} atomic IDs, found {len(ordered_ids)}")
    return ordered_ids, by_id


def validate_job_paths(bundle: Path, jobs: Sequence[Mapping[str, str]], run_tag: str) -> None:
    frozen_root = (ROOT / "logs/inscene15k_streaming/runs" / run_tag).resolve()
    path_owners: dict[Path, str] = {}
    for job in jobs:
        universe_id = job["universe_id"]
        if job.get("run_tag") != run_tag:
            raise PromotionError(f"run-tag mismatch: {universe_id}")
        manifest_train_n = exact_int(
            job.get("manifest_train_n"), label=f"{universe_id}.manifest_train_n"
        )
        manifest_val_n = exact_int(job.get("expected_n"), label=f"{universe_id}.expected_n")
        manifest_total_n = exact_int(
            job.get("manifest_total_n"), label=f"{universe_id}.manifest_total_n"
        )
        sample_overlap_n = exact_int(
            job.get("manifest_train_val_sample_overlap_n"),
            label=f"{universe_id}.manifest_train_val_sample_overlap_n",
        )
        scene_overlap_n = exact_int(
            job.get("manifest_train_val_scene_overlap_n"),
            label=f"{universe_id}.manifest_train_val_scene_overlap_n",
        )
        expected_max_epochs = exact_int(
            job.get("expected_max_epochs"), label=f"{universe_id}.expected_max_epochs"
        )
        expected_final_epoch = exact_int(
            job.get("expected_final_epoch"), label=f"{universe_id}.expected_final_epoch"
        )
        expected_evaluator = "a1" if job["family"] == "A1" else "diag"
        target_required = job["family"] in {"C1", "C2", "C3"}
        try:
            resolved_eval_contract = json.loads(job.get("resolved_eval_contract_json", ""))
        except json.JSONDecodeError as exc:
            raise PromotionError(
                f"invalid resolved_eval_contract_json: {universe_id}"
            ) from exc
        resolved_checks = (
            resolved_eval_contract.get("checks")
            if isinstance(resolved_eval_contract, dict)
            else None
        )
        resolved_critical = (
            resolved_eval_contract.get("critical_eval_contract")
            if isinstance(resolved_eval_contract, dict)
            else None
        )
        internal_contract = {
            "evaluator": job.get("evaluator") == expected_evaluator,
            "vfm_name": job.get("vfm_name") == job["model"],
            "feature_layer_present": bool(job.get("feature_layer")),
            "feat_postfix_present": bool(job.get("feat_postfix")),
            "feature_timestep_present": bool(job.get("feature_timestep")),
            "prefix_len": job.get("prefix_len") == job["prefix"],
            "prediction_required": job.get("prediction_required") == "true",
            "target_feat_root": bool(job.get("target_feat_root")) == target_required,
            "manifest_train_nonempty": manifest_train_n > 0,
            "manifest_val_nonempty": manifest_val_n > 0,
            "manifest_total_covers_train_val": manifest_total_n >= manifest_train_n + manifest_val_n,
            "manifest_train_val_sample_overlap_zero": sample_overlap_n == 0,
            "manifest_train_val_scene_overlap_zero": scene_overlap_n == 0,
            "checkpoint_status": job.get("checkpoint_status") == STRICT_CHECKPOINT_STATUS,
            "clean_completion_status": job.get("clean_completion_status")
            == STRICT_CLEAN_COMPLETION_STATUS,
            "expected_max_epochs_positive": expected_max_epochs > 0,
            "expected_final_epoch_nonnegative": expected_final_epoch >= 0,
            "resolved_config_source_kind": job.get("resolved_config_source_kind")
            == RESOLVED_CONFIG_SOURCE_KIND,
            "resolved_config_sha256": is_sha256(job.get("resolved_config_sha256")),
            "resolved_eval_checks": isinstance(resolved_checks, dict)
            and bool(resolved_checks)
            and all(value is True for value in resolved_checks.values()),
            "resolved_eval_critical": isinstance(resolved_critical, dict),
        }
        mismatches = sorted(key for key, value in internal_contract.items() if not value)
        if mismatches:
            raise PromotionError(
                f"frozen job internal contract mismatch for {universe_id}: {mismatches}"
            )
        output = safe_project_path(job["output_directory"], label="output_directory")
        ensure_under(output, frozen_root, label="output_directory")
        previous_owner = path_owners.setdefault(output, universe_id)
        if previous_owner != universe_id:
            raise PromotionError(
                f"cross-universe output-directory collision: {previous_owner} / {universe_id}"
            )
        for key in ("summary_path", "prediction_path", "preflight_path", "command_path"):
            artifact = safe_project_path(job[key], label=key)
            ensure_under(artifact, output, label=key)
            previous_owner = path_owners.setdefault(artifact, universe_id)
            if previous_owner != universe_id:
                raise PromotionError(
                    f"cross-universe artifact-path collision: {previous_owner} / {universe_id} / {key}"
                )
        for key in (
            "manifest",
            "frozen_manifest",
            "checkpoint",
            "hydra_config",
            "hydra_overrides",
            "run_manifest",
            "config_tree",
            "resolved_config_frozen",
            "experiment_config",
            "train_log",
            "launcher_manifest",
        ):
            path = safe_project_path(job[key], label=key)
            if key == "frozen_manifest":
                ensure_under(path, bundle / "inputs/manifests", label=key)
            elif key == "resolved_config_frozen":
                ensure_under(path, bundle / "inputs/resolved_training_configs", label=key)


def validate_summary_identity(job: Mapping[str, str], summary: Mapping[str, Any]) -> int:
    universe_id = job["universe_id"]
    expected_n = exact_int(job["expected_n"], label=f"{universe_id}.expected_n")
    expected_result_n = exact_int(
        job["expected_result_n"], label=f"{universe_id}.expected_result_n"
    )
    if job["evaluator"] == "a1":
        if "n_samples" not in summary:
            raise PromotionError(f"A1 summary lacks n_samples: {universe_id}")
        n = exact_int(summary["n_samples"], label=f"{universe_id}.summary.n_samples")
        if n != expected_result_n or n != expected_n:
            raise PromotionError(
                f"A1 summary n mismatch for {universe_id}: "
                f"n={n} expected_result={expected_result_n} manifest={expected_n}"
            )
    elif job["evaluator"] == "diag":
        if "n" not in summary:
            raise PromotionError(f"diag summary lacks n: {universe_id}")
        n = exact_int(summary["n"], label=f"{universe_id}.summary.n")
        try:
            loader_n = exact_int(
                summary["loader_n_samples"], label=f"{universe_id}.summary.loader_n_samples"
            )
            valid_n = exact_int(
                summary["valid_record_n"], label=f"{universe_id}.summary.valid_record_n"
            )
            invalid_n = exact_int(
                summary["invalid_record_n"], label=f"{universe_id}.summary.invalid_record_n"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PromotionError(f"diag summary lacks exact loader accounting: {universe_id}") from exc
        if loader_n != expected_n or valid_n != n or invalid_n != loader_n - valid_n:
            raise PromotionError(
                f"diag loader accounting mismatch for {universe_id}: "
                f"manifest={expected_n} loader={loader_n} valid={valid_n} invalid={invalid_n} n={n}"
            )
        if n != expected_result_n:
            raise PromotionError(
                f"summary n differs from frozen family contract for {universe_id}: "
                f"{n} != {expected_result_n}"
            )
        if job["family"] == "B2":
            if not 0 < n <= expected_n:
                raise PromotionError(f"B2 valid-subset n out of range: {universe_id}")
        elif n != expected_n or invalid_n != 0:
            raise PromotionError(
                f"non-B2 diag did not cover complete manifest for {universe_id}: "
                f"n={n} manifest={expected_n} invalid={invalid_n}"
            )
    else:
        raise PromotionError(f"unknown evaluator {job['evaluator']!r}: {universe_id}")

    expected_job_name = Path(job["output_directory"]).name
    strict_identity = {
        "probe_type": str(summary.get("probe_type", "")) == job["probe"],
        "vfm_name": str(summary.get("vfm_name", "")) == job["model"],
        "feature_layer": str(summary.get("feature_layer", "")) == job["feature_layer"],
        "feat_postfix": str(summary.get("feat_postfix", "")) == job["feat_postfix"],
        "feature_timestep": str(summary.get("feature_timestep", "")) == job["feature_timestep"],
        "prefix_len": str(summary.get("prefix_len", "")) == job["prefix_len"],
        "streaming_feat_root": str(summary.get("streaming_feat_root", ""))
        == job["streaming_feat_root"],
        "job_name": str(summary.get("job_name", "")) == expected_job_name,
        "q00_universe_id": str(summary.get("q00_universe_id", "")) == universe_id,
        "q00_campaign": str(summary.get("q00_campaign", "")) == job["campaign"],
        "q00_model_variant": str(summary.get("q00_model_variant", "")) == job["model_variant"],
        "q00_protocol": str(summary.get("q00_protocol", "")) == job["protocol"],
        "q00_manifest_sha256": str(summary.get("q00_manifest_sha256", ""))
        == job["manifest_sha256"],
        "q00_config_sha256": str(summary.get("q00_config_sha256", ""))
        == job["experiment_config_sha256"],
        "q00_resolved_config_sha256": str(
            summary.get("q00_resolved_config_sha256", "")
        )
        == job["resolved_config_sha256"],
        "q00_resolved_config_source_kind": str(
            summary.get("q00_resolved_config_source_kind", "")
        )
        == job["resolved_config_source_kind"],
    }
    if job.get("target_feat_root"):
        strict_identity["target_feat_root"] = (
            str(summary.get("target_feat_root", "")) == job["target_feat_root"]
        )
    checkpoint_value = str(summary.get("ckpt_path", ""))
    if not checkpoint_value:
        raise PromotionError(f"summary checkpoint missing: {universe_id}")
    checkpoint = Path(checkpoint_value)
    checkpoint = checkpoint.resolve() if checkpoint.is_absolute() else (ROOT / checkpoint).resolve()
    strict_identity["ckpt_path"] = checkpoint == safe_project_path(job["checkpoint"], label="checkpoint")
    mismatches = sorted(key for key, value in strict_identity.items() if not value)
    if mismatches:
        raise PromotionError(f"strict summary identity mismatch for {universe_id}: {mismatches}")

    expected_heads = [
        str(value)
        for value in parse_json_list(
            job["expected_active_heads_json"], label=f"{universe_id}.expected_active_heads"
        )
    ]
    if expected_heads and [str(value) for value in summary.get("active_heads", [])] != expected_heads:
        raise PromotionError(f"A1 active-head schema mismatch: {universe_id}")

    required_values: dict[str, float] = {}
    for name in parse_json_list(
        job["required_summary_metrics_json"], label=f"{universe_id}.required_summary_metrics"
    ):
        key = str(name)
        value = summary_metric(summary, key, required=True)
        assert value is not None
        required_values[key] = value
    for name in parse_json_list(
        job["count_equals_expected_n_json"], label=f"{universe_id}.count_equals_expected_n"
    ):
        key = str(name)
        if key not in required_values or not numbers_equal(required_values[key], n):
            raise PromotionError(f"summary count contract mismatch for {universe_id}: {key}")
    if job["family"] == "C2" and required_values.get("n_loop", 0) > 0:
        summary_metric(summary, "loop_closure_error", required=True)

    loader_ids = [str(value) for value in summary.get("loader_manifest_ids", [])]
    valid_ids = [str(value) for value in summary.get("valid_manifest_ids", [])]
    invalid_ids = [str(value) for value in summary.get("invalid_manifest_ids", [])]
    loader_ordered_hash = hashlib.sha256(("\n".join(loader_ids) + "\n").encode()).hexdigest()
    if (
        len(loader_ids) != expected_n
        or len(set(loader_ids)) != expected_n
        or loader_ordered_hash != job["manifest_val_ordered_ids_sha256"]
        or len(valid_ids) != n
        or len(invalid_ids) != expected_n - n
        or not set(valid_ids).isdisjoint(set(invalid_ids))
        or set(valid_ids) | set(invalid_ids) != set(loader_ids)
    ):
        raise PromotionError(f"summary manifest identity ledger mismatch: {universe_id}")
    return n


def audit_metric_map(record: Mapping[str, Any], *, universe_id: str) -> dict[str, dict[str, Any]]:
    values = parse_json_list(record.get("metrics_json", ""), label=f"{universe_id}.audit.metrics_json")
    output: dict[str, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict) or not item.get("registry_id"):
            raise PromotionError(f"invalid audited metric entry: {universe_id}")
        registry_id = str(item["registry_id"])
        if registry_id in output:
            raise PromotionError(f"duplicate audited metric ID {registry_id}: {universe_id}")
        output[registry_id] = item
    return output


def validate_full_audit(
    bundle: Path,
    jobs: Sequence[Mapping[str, str]],
    ordered_ids: Sequence[str],
    *,
    jobs_sha256: str,
    expected_ids_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, Any]], str]:
    audit_path = bundle / "audit/full_audit.json"
    audit = read_json(audit_path)
    audit_sha256 = sha256(audit_path)
    audit_lock_path = bundle / "audit/full_audit_lock.json"
    audit_lock = read_json(audit_lock_path)
    results_csv_path = bundle / "audit/full_results.csv"
    results_csv_fields, results_csv_rows = read_csv(results_csv_path)
    metadata = read_json(bundle / "bundle.json")
    shard_contracts = execution_shard_contracts(metadata)
    if audit.get("ok") is not True or audit.get("scope") != "full":
        raise PromotionError("full audit is not an exact PASS")
    scalar_expectations = {
        "expected_universe_ids": EXPECTED_PARENT_ROWS,
        "actual_result_rows": EXPECTED_PARENT_ROWS,
        "successful_summaries": EXPECTED_PARENT_ROWS,
        "failed_summaries": 0,
        "expected_atomic_registry_ids": EXPECTED_ATOMIC_ROWS,
        "actual_atomic_registry_ids": EXPECTED_ATOMIC_ROWS,
    }
    for key, expected in scalar_expectations.items():
        if int(audit.get(key, -1)) != expected:
            raise PromotionError(f"full audit scalar mismatch: {key}={audit.get(key)!r} expected={expected}")
    input_hashes = audit.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise PromotionError("full audit lacks exact input_hashes object")
    expected_input_hashes = {
        "bundle_json": sha256(bundle / "bundle.json"),
        "job_manifest": jobs_sha256,
        "expected_registry_ids": expected_ids_sha256,
        "script_inventory": sha256(bundle / "provenance/script_inventory.json"),
        "runtime_inventory": sha256(bundle / "provenance/runtime_inventory.json"),
    }
    if input_hashes != expected_input_hashes:
        raise PromotionError(
            "full audit input hash chain mismatch: "
            f"expected={json.dumps(expected_input_hashes, sort_keys=True)} "
            f"actual={json.dumps(input_hashes, sort_keys=True)}"
        )
    if input_hashes.get("job_manifest") != jobs_sha256:
        raise PromotionError("full audit is not bound to the frozen 99-row TSV SHA256")
    if input_hashes.get("expected_registry_ids") != expected_ids_sha256:
        raise PromotionError("full audit is not bound to expected_registry_ids.json SHA256")
    if (
        audit.get("expected_id_contract_ok") is not True
        or audit.get("duplicate_universe_ids")
        or audit.get("duplicate_registry_ids")
        or audit.get("problems")
        or audit.get("frozen_hash_problems")
        or audit.get("protected_source_of_truth_drift")
        or int(audit.get("non_finite_metrics", -1)) != 0
    ):
        raise PromotionError("full audit contains ID/hash/metric problems")

    expected_lock = {
        "scope": "full",
        "audit_sha256": audit_sha256,
        "results_csv_sha256": sha256(results_csv_path),
        "job_manifest_sha256": jobs_sha256,
        "expected_registry_ids_sha256": expected_ids_sha256,
        "expected_registry_ids": sorted(ordered_ids),
        "bundle_json_sha256": sha256(bundle / "bundle.json"),
    }
    if audit_lock != expected_lock:
        raise PromotionError(
            "full-audit lock mismatch: "
            f"expected={json.dumps(expected_lock, sort_keys=True)} "
            f"actual={json.dumps(audit_lock, sort_keys=True)}"
        )
    if not results_csv_fields or len(results_csv_rows) != EXPECTED_PARENT_ROWS:
        raise PromotionError("full_results.csv is not exactly 99 result rows")
    runtime_inventory = read_json(bundle / "provenance/runtime_inventory.json")
    if not isinstance(runtime_inventory, list) or not runtime_inventory:
        raise PromotionError("runtime inventory is not a list")
    runtime_by_path: dict[str, dict[str, Any]] = {}
    for item in runtime_inventory:
        if not isinstance(item, dict) or not item.get("path"):
            raise PromotionError("runtime inventory contains an invalid entry")
        relative = str(item["path"])
        if relative in runtime_by_path:
            raise PromotionError(f"runtime inventory contains duplicate path: {relative}")
        runtime_by_path[relative] = item

    results = audit.get("results")
    if not isinstance(results, list) or len(results) != EXPECTED_PARENT_ROWS:
        raise PromotionError("full audit does not contain exactly 99 result records")
    by_universe: dict[str, dict[str, Any]] = {}
    for record in results:
        if not isinstance(record, dict) or not record.get("universe_id"):
            raise PromotionError("invalid full-audit result record")
        universe_id = str(record["universe_id"])
        if universe_id in by_universe:
            raise PromotionError(f"duplicate full-audit universe ID: {universe_id}")
        if record.get("status") != "PASS":
            raise PromotionError(f"non-PASS full-audit record: {universe_id}")
        by_universe[universe_id] = record

    job_ids = {job["universe_id"] for job in jobs}
    if set(by_universe) != job_ids:
        raise PromotionError("full-audit universe-ID set differs from frozen TSV")

    csv_by_universe: dict[str, dict[str, str]] = {}
    for record in results_csv_rows:
        universe_id = record.get("universe_id", "")
        if not universe_id or universe_id in csv_by_universe:
            raise PromotionError("full_results.csv has missing/duplicate universe ID")
        csv_by_universe[universe_id] = record
    if set(csv_by_universe) != job_ids:
        raise PromotionError("full_results.csv universe-ID set differs from frozen TSV")
    csv_lock_fields = (
        "status",
        "summary_path",
        "summary_sha256",
        "prediction_sha256",
        "preflight_sha256",
        "postflight_sha256",
        "command_sha256",
        "manifest_sha256",
        "checkpoint_sha256",
        "execution_shard_sha256",
        "canary_gate_path",
        "canary_gate_sha256",
        "canary_gate_classification",
        "expected_registry_ids",
        "metrics_json",
        "n",
    )
    for universe_id, record in by_universe.items():
        csv_record = csv_by_universe[universe_id]
        for key in csv_lock_fields:
            if str(csv_record.get(key, "")) != str(record.get(key, "")):
                raise PromotionError(f"full audit JSON/CSV mismatch for {universe_id}: {key}")

    artifact_hashes: dict[str, str] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for job in jobs:
        universe_id = job["universe_id"]
        record = by_universe[universe_id]
        for key in ("model", "family", "probe", "layer", "prefix"):
            if str(record.get(key, "")) != job[key]:
                raise PromotionError(f"full-audit row identity mismatch for {universe_id}: {key}")
        summary_relative = job["summary_path"]
        prediction_relative = job["prediction_path"]
        if record.get("summary_path") != summary_relative:
            raise PromotionError(f"audited summary path mismatch: {universe_id}")
        if record.get("prediction_path") not in (None, "", prediction_relative):
            raise PromotionError(f"audited prediction path mismatch: {universe_id}")
        if record.get("manifest_sha256") != job["manifest_sha256"]:
            raise PromotionError(f"audited manifest hash mismatch: {universe_id}")
        if record.get("checkpoint_sha256") != job["checkpoint_sha256"]:
            raise PromotionError(f"audited checkpoint hash mismatch: {universe_id}")

        output = safe_project_path(job["output_directory"], label="output_directory")
        artifact_specs = (
            ("summary", summary_relative, "summary_sha256"),
            ("prediction", prediction_relative, "prediction_sha256"),
            ("preflight", job["preflight_path"], "preflight_sha256"),
            ("postflight", str(Path(job["output_directory"]) / "eval/q00_postflight.json"), "postflight_sha256"),
            ("command", job["command_path"], "command_sha256"),
        )
        resolved_artifacts: dict[str, Path] = {}
        for label, relative, record_hash_key in artifact_specs:
            path = safe_project_path(relative, label=f"{universe_id}.{label}")
            ensure_under(path, output, label=f"{universe_id}.{label}")
            expected_hash = str(record.get(record_hash_key, ""))
            if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                raise PromotionError(f"missing/invalid audited {label} SHA256: {universe_id}")
            actual_hash = sha256(path) if path.is_file() else ""
            if actual_hash != expected_hash:
                raise PromotionError(f"{label} changed after full audit: {universe_id}")
            existing_hash = artifact_hashes.setdefault(relative, expected_hash)
            if existing_hash != expected_hash:
                raise PromotionError(f"conflicting audit hashes for artifact {relative}")
            resolved_artifacts[label] = path

        summary_path = resolved_artifacts["summary"]
        prediction_path = resolved_artifacts["prediction"]
        preflight = read_json(resolved_artifacts["preflight"])
        postflight = read_json(resolved_artifacts["postflight"])
        command = read_json(resolved_artifacts["command"])

        if preflight.get("ok") is not True or preflight.get("universe_id") != universe_id:
            raise PromotionError(f"invalid exact preflight: {universe_id}")
        preflight_identity = preflight.get("full_identity", {})
        for key in IDENTITY_FIELDS:
            if str(preflight_identity.get(key, "")) != job[key]:
                raise PromotionError(f"preflight full-identity mismatch for {universe_id}: {key}")
        preflight_manifest = preflight.get("manifest", {})
        if not isinstance(preflight_manifest, dict):
            raise PromotionError(f"preflight manifest is not an object: {universe_id}")
        expected_manifest_contract = {
            "original_path": job["manifest"],
            "frozen_path": job["frozen_manifest"],
            "sha256": job["manifest_sha256"],
            "train_n": exact_int(
                job.get("manifest_train_n"), label=f"{universe_id}.manifest_train_n"
            ),
            "expected_n": exact_int(job.get("expected_n"), label=f"{universe_id}.expected_n"),
            "total_n": exact_int(
                job.get("manifest_total_n"), label=f"{universe_id}.manifest_total_n"
            ),
            "train_val_sample_overlap_n": exact_int(
                job.get("manifest_train_val_sample_overlap_n"),
                label=f"{universe_id}.manifest_train_val_sample_overlap_n",
            ),
            "train_val_scene_overlap_n": exact_int(
                job.get("manifest_train_val_scene_overlap_n"),
                label=f"{universe_id}.manifest_train_val_scene_overlap_n",
            ),
        }
        if (
            expected_manifest_contract["train_n"] <= 0
            or expected_manifest_contract["expected_n"] <= 0
            or expected_manifest_contract["total_n"]
            < expected_manifest_contract["train_n"] + expected_manifest_contract["expected_n"]
            or expected_manifest_contract["train_val_sample_overlap_n"] != 0
            or expected_manifest_contract["train_val_scene_overlap_n"] != 0
            or preflight_manifest != expected_manifest_contract
        ):
            raise PromotionError(f"preflight manifest mismatch: {universe_id}")
        preflight_checkpoint = preflight.get("checkpoint", {})
        checkpoint_path_value = Path(str(preflight_checkpoint.get("path", "")))
        checkpoint_path_value = (
            checkpoint_path_value.resolve()
            if checkpoint_path_value.is_absolute()
            else (ROOT / checkpoint_path_value).resolve()
        )
        if (
            checkpoint_path_value != safe_project_path(job["checkpoint"], label="checkpoint")
            or preflight_checkpoint.get("sha256") != job["checkpoint_sha256"]
            or exact_int(
                preflight_checkpoint.get("size_bytes"),
                label=f"{universe_id}.preflight.checkpoint_size_bytes",
            )
            != exact_int(job["checkpoint_size_bytes"], label=f"{universe_id}.checkpoint_size_bytes")
            or exact_int(
                job["checkpoint_size_bytes"], label=f"{universe_id}.checkpoint_size_bytes"
            )
            <= 0
            or exact_int(
                preflight_checkpoint.get("expected_final_epoch"),
                label=f"{universe_id}.preflight.expected_final_epoch",
            )
            != exact_int(job["expected_final_epoch"], label=f"{universe_id}.expected_final_epoch")
            or exact_int(
                preflight_checkpoint.get("expected_final_step"),
                label=f"{universe_id}.preflight.expected_final_step",
            )
            != exact_int(job["expected_final_step"], label=f"{universe_id}.expected_final_step")
            or exact_int(
                preflight_checkpoint.get("checkpoint_epoch"),
                label=f"{universe_id}.preflight.checkpoint_epoch",
            )
            != exact_int(job["expected_final_epoch"], label=f"{universe_id}.expected_final_epoch")
            or exact_int(
                preflight_checkpoint.get("checkpoint_global_step"),
                label=f"{universe_id}.preflight.checkpoint_global_step",
            )
            != exact_int(job["expected_final_step"], label=f"{universe_id}.expected_final_step")
            or preflight_checkpoint.get("deep_metadata_loaded") is not True
        ):
            raise PromotionError(f"preflight checkpoint mismatch: {universe_id}")
        clean_completion = preflight.get("clean_completion", {})
        expected_max_epochs = exact_int(
            job.get("expected_max_epochs"), label=f"{universe_id}.expected_max_epochs"
        )
        expected_clean_completion = {
            "ok": True,
            "checkpoint_status": STRICT_CHECKPOINT_STATUS,
            "clean_completion_status": STRICT_CLEAN_COMPLETION_STATUS,
            "expected_max_epochs": expected_max_epochs,
            "completion_marker": f"`max_epochs={expected_max_epochs}` reached",
            "bad_markers_in_final_attempt": [],
            "observed_val_counts_in_final_attempt": [expected_manifest_contract["expected_n"]],
        }
        if (
            job.get("checkpoint_status") != STRICT_CHECKPOINT_STATUS
            or job.get("clean_completion_status") != STRICT_CLEAN_COMPLETION_STATUS
            or expected_max_epochs <= 0
            or clean_completion != expected_clean_completion
        ):
            raise PromotionError(f"preflight strict clean-completion mismatch: {universe_id}")
        local_artifacts = preflight.get("local_artifacts", {})
        if not isinstance(local_artifacts, dict) or local_artifacts.get("ok") is not True:
            raise PromotionError(f"preflight local-artifact audit missing: {universe_id}")
        raw_local_checks = local_artifacts.get("checks", [])
        local_checks = {
            str(item.get("kind", "")): item
            for item in raw_local_checks
            if isinstance(item, dict)
        }
        local_contract = {
            "experiment_config": (job["experiment_config"], job["experiment_config_sha256"]),
            "hydra_config": (job["hydra_config"], job["hydra_config_sha256"]),
            "hydra_overrides": (job["hydra_overrides"], job["hydra_overrides_sha256"]),
            "run_manifest": (job["run_manifest"], job["run_manifest_sha256"]),
            "config_tree": (job["config_tree"], job["config_tree_sha256"]),
            "resolved_config_frozen": (
                job["resolved_config_frozen"],
                job["resolved_config_sha256"],
            ),
            "train_log": (job["train_log"], job["train_log_sha256"]),
            "launcher_manifest": (job["launcher_manifest"], job["launcher_manifest_sha256"]),
        }
        if (
            not isinstance(raw_local_checks, list)
            or len(raw_local_checks) != len(local_checks)
            or set(local_checks) != set(local_contract)
        ):
            raise PromotionError(f"preflight local-artifact set mismatch: {universe_id}")
        for kind, (relative, expected_hash) in local_contract.items():
            check = local_checks[kind]
            if not (
                check.get("ok") is True
                and check.get("path") == relative
                and check.get("expected_sha256") == expected_hash
                and check.get("actual_sha256") == expected_hash
            ):
                raise PromotionError(f"preflight local-artifact mismatch for {universe_id}: {kind}")
        resolved_path = safe_project_path(
            job["resolved_config_frozen"], label="resolved_config_frozen"
        )
        ensure_under(
            resolved_path,
            bundle / "inputs/resolved_training_configs",
            label="resolved_config_frozen",
        )
        resolved = read_json(resolved_path)
        run_manifest_payload = read_json(
            safe_project_path(job["run_manifest"], label="run_manifest")
        )
        try:
            frozen_resolved_contract = json.loads(job["resolved_eval_contract_json"])
        except json.JSONDecodeError as exc:
            raise PromotionError(f"invalid resolved eval contract: {universe_id}") from exc
        resolved_checks = (
            frozen_resolved_contract.get("checks")
            if isinstance(frozen_resolved_contract, dict)
            else None
        )
        resolved_critical = (
            frozen_resolved_contract.get("critical_eval_contract")
            if isinstance(frozen_resolved_contract, dict)
            else None
        )
        resolved_model = resolved.get("model", {}) if isinstance(resolved, dict) else {}
        resolved_trainer = resolved.get("trainer", {}) if isinstance(resolved, dict) else {}
        expected_training_probe = "pixalign" if job["family"] == "A1" else job["probe"]
        target_required = bool(job.get("target_feat_root"))
        try:
            frozen_extra_overrides = json.loads(job.get("extra_overrides_json") or "{}")
        except json.JSONDecodeError as exc:
            raise PromotionError(f"invalid frozen extra overrides: {universe_id}") from exc
        resolved_identity = {
            "run_manifest_config": isinstance(run_manifest_payload, dict)
            and run_manifest_payload.get("config") == resolved,
            "canonical_frozen_bytes": resolved_path.read_text()
            == json.dumps(resolved, indent=2, sort_keys=True) + "\n",
            "seed": str(resolved.get("seed")) == job["seed"],
            "probe": isinstance(resolved_model, dict)
            and str(resolved_model.get("probe_type")) == expected_training_probe,
            "vfm_name": str(resolved.get("vfm_name")) == job["vfm_name"] == job["model"],
            "video_channels": str(resolved.get("video_channels")) == job["video_channels"],
            "prefix_len": str(resolved.get("prefix_len")) == job["prefix_len"],
            "feature_layer": str(resolved.get("feature_layer")) == job["feature_layer"],
            "feature_timestep": str(resolved.get("feature_timestep"))
            == job["feature_timestep"],
            "feat_postfix": str(resolved.get("feat_postfix")) == job["feat_postfix"],
            "streaming_feat_root": resolved.get("streaming_feat_root")
            == job["streaming_feat_root"],
            "max_epochs": isinstance(resolved_trainer, dict)
            and str(resolved_trainer.get("max_epochs")) == job["expected_max_epochs"],
            "target_feat_root": (
                resolved.get("target_feat_root") == job["target_feat_root"]
                if target_required
                else resolved.get("target_feat_root") in (None, "")
            ),
            "critical_eval_contract": frozen_extra_overrides == resolved_critical,
        }
        if (
            sha256(resolved_path) != job["resolved_config_sha256"]
            or job.get("resolved_config_source_kind") != RESOLVED_CONFIG_SOURCE_KIND
            or local_artifacts.get("resolved_config_source_kind")
            != RESOLVED_CONFIG_SOURCE_KIND
            or local_artifacts.get("resolved_config_sha256")
            != job["resolved_config_sha256"]
            or local_artifacts.get("resolved_eval_contract") != frozen_resolved_contract
            or not isinstance(resolved, dict)
            or not resolved
            or not isinstance(resolved_checks, dict)
            or not resolved_checks
            or any(value is not True for value in resolved_checks.values())
            or not isinstance(resolved_critical, dict)
            or any(resolved.get(key) != value for key, value in resolved_critical.items())
            or not all(resolved_identity.values())
        ):
            failed_resolved = sorted(
                key for key, value in resolved_identity.items() if not value
            )
            raise PromotionError(
                f"resolved training-config contract mismatch: {universe_id}: "
                f"{failed_resolved}"
            )
        feature = preflight.get("feature", {})
        expected_aoss_cache = (
            f"/mnt/local_cache/inscene/aoss_cache/{job['run_tag']}/shared"
        )
        if (
            feature.get("sample_id") != job["sample_id"]
            or feature.get("streaming_root") != job["streaming_feat_root"]
            or feature.get("aoss_cache_dir") != expected_aoss_cache
            or feature.get("streaming_root_kind")
            != "remote_uri_materialized_for_content_audit"
            or not content_audit_ok(feature.get("prefix_index_materialized"))
            or not content_audit_ok(feature.get("streaming_tensor"))
            or feature.get("streaming_tensor", {}).get("finite") is not True
        ):
            raise PromotionError(f"preflight streaming-feature mismatch: {universe_id}")
        if job.get("target_feat_root") and (
            feature.get("target_root") != job["target_feat_root"]
            or feature.get("target_root_kind")
            != "remote_uri_materialized_for_content_audit"
            or not content_audit_ok(feature.get("target_indices_materialized"))
            or not content_audit_ok(feature.get("target_tensor"))
            or feature.get("target_tensor", {}).get("finite") is not True
        ):
            raise PromotionError(f"preflight target-feature mismatch: {universe_id}")
        validate_feature_tensor_contract(job, feature)
        validate_full_streaming_index(job, feature)
        runtime_before = preflight.get("frozen_runtime_before_evaluator", {})
        if (
            runtime_before.get("ok") is not True
            or runtime_before.get("inventory_expected_sha256")
            != expected_input_hashes["runtime_inventory"]
            or runtime_before.get("inventory_actual_sha256")
            != expected_input_hashes["runtime_inventory"]
            or exact_int(runtime_before.get("count"), label=f"{universe_id}.runtime_before.count")
            != len(runtime_inventory)
            or len(runtime_before.get("checks", [])) != len(runtime_inventory)
            or any(check.get("ok") is not True for check in runtime_before.get("checks", []))
        ):
            raise PromotionError(f"frozen runtime pre-evaluator gate mismatch: {universe_id}")
        execution_hashes = preflight.get("execution_hashes", {})
        shard_sha = str(execution_hashes.get("shard_tsv", ""))
        shard_contract = shard_contracts.get(shard_sha)
        expected_shard_class = (
            "EXACT_ONE_ROW_CANARY"
            if job.get("is_canary") == "true"
            else "EXACT_REMAINING_96"
        )
        expected_shard_source = (
            f"inputs/canary_shards/canary_{job['model']}.tsv"
            if job.get("is_canary") == "true"
            else "inputs/wave_b_remaining_96.tsv"
        )
        if (
            execution_hashes.get("bundle_json") != expected_input_hashes["bundle_json"]
            or execution_hashes.get("script_inventory") != expected_input_hashes["script_inventory"]
            or execution_hashes.get("runtime_inventory") != expected_input_hashes["runtime_inventory"]
            or shard_contract is None
            or shard_contract["classification"] != expected_shard_class
            or shard_contract["source"] != expected_shard_source
        ):
            raise PromotionError(f"preflight execution hash chain mismatch: {universe_id}")
        gate_path, gate_hash = validate_launcher_canary_gate(
            bundle, shard_sha, shard_contracts, record
        )
        try:
            gate_binding_path = str(Path(gate_path).resolve().relative_to(ROOT))
        except ValueError as exc:
            raise PromotionError(f"launcher gate escapes project root: {gate_path}") from exc
        previous_gate_hash = artifact_hashes.setdefault(gate_binding_path, gate_hash)
        if previous_gate_hash != gate_hash:
            raise PromotionError(f"conflicting launcher gate hash: {gate_path}")
        if expected_shard_class == "EXACT_REMAINING_96":
            for gate_dependency in (
                bundle / "audit/canary_audit.json",
                bundle / "audit/canary_results.csv",
                bundle / "audit/canary_audit_lock.json",
            ):
                try:
                    dependency_path = str(gate_dependency.resolve().relative_to(ROOT))
                except ValueError as exc:
                    raise PromotionError(
                        f"launcher gate dependency escapes project root: {gate_dependency}"
                    ) from exc
                dependency_hash = sha256(gate_dependency)
                previous_dependency_hash = artifact_hashes.setdefault(
                    dependency_path, dependency_hash
                )
                if previous_dependency_hash != dependency_hash:
                    raise PromotionError(
                        f"conflicting launcher gate dependency hash: {dependency_path}"
                    )

        runtime_tree = (bundle / "provenance/runtime_tree").resolve()
        expected_env = expected_command_env(bundle, job)
        identity_env = command.get("identity_env", {})
        if (
            set(identity_env) != set(expected_env) | {"CUDA_VISIBLE_DEVICES"}
            or any(identity_env.get(key) != value for key, value in expected_env.items())
            or re.fullmatch(r"[0-9]+", str(identity_env.get("CUDA_VISIBLE_DEVICES", "")))
            is None
            or feature.get("aoss_cache_dir") != identity_env.get("AOSS_CACHE_DIR")
        ):
            raise PromotionError(f"command full-identity environment mismatch: {universe_id}")
        validate_environment_policy(
            command.get("environment_policy"), expected_env, universe_id=universe_id
        )
        evaluator_relative = (
            "vidfm3d/eval_a1_streaming.py"
            if job["evaluator"] == "a1"
            else "vidfm3d/eval_diag.py"
        )
        evaluator_inventory = runtime_by_path.get(evaluator_relative)
        evaluator_path = (runtime_tree / evaluator_relative).resolve()
        checkpoint_override = str(safe_project_path(job["checkpoint"], label="checkpoint")).replace(
            "=", r"\="
        )
        expected_argv_tail = [
            str(evaluator_path),
            f"experiment={job['experiment']}",
            f"vfm_name={job['vfm_name']}",
            f"video_channels={job['video_channels']}",
            f"streaming_feat_root={job['streaming_feat_root']}",
            f"feature_layer={job['feature_layer']}",
            f"feature_timestep={job['feature_timestep']}",
            f"prefix_len={job['prefix_len']}",
            f"job_name={Path(job['output_directory']).name}",
            f"paths.run_folder_name={job['run_tag']}/{Path(job['output_directory']).name}",
            f"ckpt_path={checkpoint_override}",
            "+eval_split=val",
            "train=false",
            "test=false",
            "data.data_module.num_workers_val=2",
            "data.data_module.batch_size_per_device_val=4",
            "logger.wandb.offline=true",
        ]
        if job["evaluator"] == "diag":
            expected_argv_tail.append("++bootstrap_reps=0")
        if job.get("feat_postfix"):
            expected_argv_tail.append(f"feat_postfix={job['feat_postfix']}")
        if job.get("target_feat_root"):
            expected_argv_tail.append(f"++target_feat_root={job['target_feat_root']}")
        extra_overrides = json.loads(job.get("extra_overrides_json") or "{}")
        if not isinstance(extra_overrides, dict):
            raise PromotionError(f"job extra_overrides_json is not an object: {universe_id}")
        expected_argv_tail.extend(
            f"{key}={value}" for key, value in sorted(extra_overrides.items())
        )
        actual_argv = command.get("argv")
        if (
            not isinstance(actual_argv, list)
            or len(actual_argv) < 2
            or not all(isinstance(value, str) for value in actual_argv)
        ):
            raise PromotionError(f"command argv schema mismatch: {universe_id}")
        argv_sha256 = hashlib.sha256("\0".join(actual_argv).encode()).hexdigest()
        if (
            evaluator_inventory is None
            or command.get("cwd") != str(runtime_tree)
            or identity_env.get("PYTHONPATH") != str(runtime_tree)
            or identity_env.get("Q00_RUNTIME_TREE") != str(runtime_tree)
            or identity_env.get("Q00_REQUIRE_CUDA") != "1"
            or command.get("frozen_evaluator_sha256")
            != evaluator_inventory.get("frozen_sha256")
            or Path(actual_argv[0]).resolve() != FROZEN_PYTHON
            or actual_argv[1:] != expected_argv_tail
            or command.get("argv_sha256") != argv_sha256
        ):
            raise PromotionError(f"command frozen-runtime contract mismatch: {universe_id}")

        summary = read_json(summary_path)
        n = validate_summary_identity(job, summary)
        if int(record.get("n", -1)) != n:
            raise PromotionError(f"audited n differs from current summary: {universe_id}")
        summaries[universe_id] = summary

        if postflight.get("ok") is not True or int(postflight.get("n", -1)) != n:
            raise PromotionError(f"invalid exact postflight: {universe_id}")
        checkpoint_after = postflight.get("checkpoint_after_evaluator", {})
        checkpoint_after_path = Path(str(checkpoint_after.get("path", "")))
        checkpoint_after_path = (
            checkpoint_after_path.resolve()
            if checkpoint_after_path.is_absolute()
            else (ROOT / checkpoint_after_path).resolve()
        )
        if (
            checkpoint_after_path != safe_project_path(job["checkpoint"], label="checkpoint")
            or checkpoint_after.get("sha256") != job["checkpoint_sha256"]
            or exact_int(
                checkpoint_after.get("size_bytes"), label=f"{universe_id}.checkpoint_after.size"
            )
            != exact_int(job["checkpoint_size_bytes"], label=f"{universe_id}.checkpoint_size")
            or exact_int(
                checkpoint_after.get("expected_final_epoch"),
                label=f"{universe_id}.checkpoint_after.expected_final_epoch",
            )
            != exact_int(job["expected_final_epoch"], label=f"{universe_id}.expected_final_epoch")
            or exact_int(
                checkpoint_after.get("expected_final_step"),
                label=f"{universe_id}.checkpoint_after.expected_final_step",
            )
            != exact_int(job["expected_final_step"], label=f"{universe_id}.expected_final_step")
            or checkpoint_after.get("deep_metadata_loaded") is not False
        ):
            raise PromotionError(f"checkpoint changed during/after evaluator: {universe_id}")
        validate_feature_artifacts_after_evaluator(
            feature,
            postflight.get("feature_artifacts_after_evaluator"),
            universe_id=universe_id,
        )
        cuda_checks = {
            "execution_device": str(summary.get("execution_device", "")) == "cuda:0",
            "cuda_available": summary.get("cuda_available") is True,
            "cuda_device_count": exact_int(
                summary.get("cuda_device_count"), label=f"{universe_id}.summary.cuda_device_count"
            )
            == 1,
            "cuda_device_name": bool(str(summary.get("cuda_device_name", "")).strip()),
            "CUDA_VISIBLE_DEVICES": str(summary.get("CUDA_VISIBLE_DEVICES", ""))
            == str(identity_env.get("CUDA_VISIBLE_DEVICES", ""))
            and re.fullmatch(r"[0-9]+", str(identity_env.get("CUDA_VISIBLE_DEVICES", "")))
            is not None,
        }
        if not all(cuda_checks.values()):
            raise PromotionError(f"summary CUDA execution contract mismatch: {universe_id}")
        cuda_execution = postflight.get("cuda_execution", {})
        locked_cuda_checks = cuda_execution.get("checks", {})
        if (
            str(cuda_execution.get("execution_device", "")) != str(summary["execution_device"])
            or cuda_execution.get("cuda_available") is not summary["cuda_available"]
            or not numbers_equal(cuda_execution.get("cuda_device_count"), 1)
            or str(cuda_execution.get("cuda_device_name", ""))
            != str(summary["cuda_device_name"])
            or str(cuda_execution.get("cuda_device_uuid", ""))
            != str(summary.get("cuda_device_uuid", ""))
            or str(cuda_execution.get("CUDA_VISIBLE_DEVICES", ""))
            != str(summary["CUDA_VISIBLE_DEVICES"])
            or cuda_execution.get("Q00_REQUIRE_CUDA") != "1"
            or set(locked_cuda_checks) != set(cuda_checks)
            or not all(value is True for value in locked_cuda_checks.values())
        ):
            raise PromotionError(f"postflight CUDA execution lock mismatch: {universe_id}")
        postflight_summary = Path(str(postflight.get("summary", "")))
        postflight_summary = (
            postflight_summary.resolve()
            if postflight_summary.is_absolute()
            else (ROOT / postflight_summary).resolve()
        )
        if (
            postflight_summary != summary_path
            or postflight.get("summary_sha256") != record.get("summary_sha256")
        ):
            raise PromotionError(f"postflight summary binding mismatch: {universe_id}")
        prediction_contract = postflight.get("prediction", {})
        if (
            job.get("prediction_required") != "true"
            or prediction_contract.get("required") is not True
            or prediction_contract.get("path") != prediction_relative
            or prediction_contract.get("sha256") != record.get("prediction_sha256")
            or int(prediction_contract.get("n", -1)) != n
            or (
                exact_int(
                    prediction_contract.get("checked_finite_tensors"),
                    label=f"{universe_id}.prediction.checked_finite_tensors",
                )
                + exact_int(
                    prediction_contract.get("checked_finite_numeric_scalars"),
                    label=f"{universe_id}.prediction.checked_finite_numeric_scalars",
                )
                < 1
            )
        ):
            raise PromotionError(f"postflight prediction binding mismatch: {universe_id}")
        cardinality = postflight.get("cardinality", {})
        if (
            int(cardinality.get("manifest_val_n", -1)) != int(job["expected_n"])
            or int(cardinality.get("summary_n", -1)) != n
        ):
            raise PromotionError(f"postflight cardinality mismatch: {universe_id}")
        if job["evaluator"] == "diag" and (
            exact_int(
                cardinality.get("loader_n_samples"), label=f"{universe_id}.cardinality.loader_n"
            )
            != exact_int(job["expected_n"], label=f"{universe_id}.expected_n")
            or exact_int(
                cardinality.get("valid_record_n"), label=f"{universe_id}.cardinality.valid_n"
            )
            != n
            or exact_int(
                cardinality.get("invalid_record_n"), label=f"{universe_id}.cardinality.invalid_n"
            )
            != exact_int(job["expected_n"], label=f"{universe_id}.expected_n") - n
        ):
            raise PromotionError(f"postflight loader/valid/invalid accounting mismatch: {universe_id}")
        lineage = postflight.get("lineage", {})
        valid_ids = [str(value) for value in summary.get("valid_manifest_ids", [])]
        valid_ordered_hash = hashlib.sha256(("\n".join(valid_ids) + "\n").encode()).hexdigest()
        if (
            lineage.get("loader_manifest_ids_ordered_sha256")
            != job["manifest_val_ordered_ids_sha256"]
            or lineage.get("valid_manifest_ids_ordered_sha256") != valid_ordered_hash
            or [str(value) for value in lineage.get("invalid_manifest_ids", [])]
            != [str(value) for value in summary.get("invalid_manifest_ids", [])]
            or prediction_contract.get("manifest_ids_ordered_sha256") != valid_ordered_hash
            or exact_int(
                prediction_contract.get("frozen_manifest_identity_records_checked"),
                label=f"{universe_id}.prediction.identity_records_checked",
            )
            != n
        ):
            raise PromotionError(f"postflight prediction/summary lineage mismatch: {universe_id}")
        if job["family"] == "B2":
            expected_signature = job.get("expected_valid_subset_signature_sha256", "")
            if (
                not re.fullmatch(r"[0-9a-f]{64}", expected_signature)
                or prediction_contract.get("valid_subset_signature_sha256") != expected_signature
            ):
                raise PromotionError(f"B2 frozen valid-subset signature mismatch: {universe_id}")
        elif prediction_contract.get("manifest_ids_ordered_sha256") != job[
            "manifest_val_ordered_ids_sha256"
        ]:
            raise PromotionError(f"non-B2 prediction does not cover frozen manifest order: {universe_id}")
        strict_postflight = postflight.get("strict_identity", {})
        expected_strict_identity_keys = {
            "probe_type",
            "vfm_name",
            "feature_layer",
            "feat_postfix",
            "feature_timestep",
            "prefix_len",
            "streaming_feat_root",
            "job_name",
            "q00_universe_id",
            "q00_campaign",
            "q00_model_variant",
            "q00_protocol",
            "q00_manifest_sha256",
            "q00_config_sha256",
            "q00_resolved_config_sha256",
            "q00_resolved_config_source_kind",
            "ckpt_path",
        }
        if job.get("target_feat_root"):
            expected_strict_identity_keys.add("target_feat_root")
        if (
            not isinstance(strict_postflight, dict)
            or set(strict_postflight) != expected_strict_identity_keys
            or any(value is not True for value in strict_postflight.values())
        ):
            raise PromotionError(f"postflight strict identity did not pass: {universe_id}")
        runtime_after = postflight.get("frozen_runtime_after_evaluator", {})
        if (
            runtime_after.get("ok") is not True
            or runtime_after.get("inventory_expected_sha256")
            != expected_input_hashes["runtime_inventory"]
            or runtime_after.get("inventory_actual_sha256")
            != expected_input_hashes["runtime_inventory"]
            or exact_int(runtime_after.get("count"), label=f"{universe_id}.runtime_after.count")
            != len(runtime_inventory)
            or len(runtime_after.get("checks", [])) != len(runtime_inventory)
            or any(check.get("ok") is not True for check in runtime_after.get("checks", []))
        ):
            raise PromotionError(f"frozen runtime post-evaluator gate mismatch: {universe_id}")
        required_locked = postflight.get("required_summary_values", {})
        if not isinstance(required_locked, dict):
            raise PromotionError(f"postflight required-summary lock is invalid: {universe_id}")
        for name in parse_json_list(
            job["required_summary_metrics_json"], label=f"{universe_id}.required_summary_metrics"
        ):
            key = str(name)
            summary_value = summary_metric(summary, key, required=True)
            if key not in required_locked or not numbers_equal(required_locked[key], summary_value):
                raise PromotionError(f"postflight required metric mismatch for {universe_id}: {key}")
        if job["family"] == "C2" and float(required_locked.get("n_loop", 0)) > 0:
            loop_value = summary_metric(summary, "loop_closure_error", required=True)
            if not numbers_equal(required_locked.get("loop_closure_error"), loop_value):
                raise PromotionError(f"postflight loop-closure metric mismatch: {universe_id}")

        expected_for_job = parse_json_list(
            job["expected_outputs_json"], label=f"{universe_id}.expected_outputs"
        )
        expected_for_job_ids = [str(item["registry_id"]) for item in expected_for_job]
        if str(record.get("expected_registry_ids", "")).split(";") != expected_for_job_ids:
            raise PromotionError(f"audited expected-ID sequence mismatch: {universe_id}")
        audited_metrics = audit_metric_map(record, universe_id=universe_id)
        if set(audited_metrics) != set(expected_for_job_ids):
            raise PromotionError(f"audited primary metric-ID set mismatch: {universe_id}")
        postflight_metrics = {
            str(item.get("registry_id", "")): item
            for item in postflight.get("metrics", [])
            if isinstance(item, dict)
        }
        if (
            len(postflight.get("metrics", [])) != len(postflight_metrics)
            or set(postflight_metrics) != set(expected_for_job_ids)
        ):
            raise PromotionError(f"postflight primary metric-ID set mismatch: {universe_id}")
        for output in expected_for_job:
            registry_id = str(output["registry_id"])
            audited = audited_metrics[registry_id]
            postflight_metric = postflight_metrics[registry_id]
            if (
                audited.get("metric_name") != output["metric_name"]
                or postflight_metric.get("metric_name") != output["metric_name"]
            ):
                raise PromotionError(f"audited metric-name mismatch: {registry_id}")
            summary_value = summary_metric(summary, str(output["metric_name"]), required=True)
            if not (
                numbers_equal(audited.get("metric_value"), summary_value)
                and numbers_equal(postflight_metric.get("metric_value"), summary_value)
            ):
                raise PromotionError(f"audited metric value differs from summary: {registry_id}")

    return by_universe, artifact_hashes, summaries, audit_sha256


def build_primary_registry_rows(
    jobs: Sequence[Mapping[str, str]],
    audit_by_universe: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    audit_sha256: str,
    git_commit: str,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows: list[dict[str, str]] = []
    by_id: dict[str, dict[str, str]] = {}
    for job in jobs:
        universe_id = job["universe_id"]
        summary = summaries[universe_id]
        audited = audit_metric_map(audit_by_universe[universe_id], universe_id=universe_id)
        for output in parse_json_list(job["expected_outputs_json"], label=f"{universe_id}.expected_outputs"):
            registry_id = str(output["registry_id"])
            metric_value = summary_metric(summary, str(output["metric_name"]), required=True)
            if not numbers_equal(metric_value, audited[registry_id]["metric_value"]):
                raise PromotionError(f"metric drift while building registry row: {registry_id}")
            summary_hash = str(audit_by_universe[universe_id]["summary_sha256"])
            prediction_hash = str(audit_by_universe[universe_id]["prediction_sha256"])
            notes = "; ".join(
                [
                    "source=q00_phase1_exact_frozen_eval",
                    f"q00_universe_id={universe_id}",
                    f"campaign={job['campaign']}",
                    f"model_variant={job['model_variant']}",
                    f"protocol={job['protocol']}",
                    f"run_tag={job['run_tag']}",
                    f"summary_sha256={summary_hash}",
                    f"prediction_sha256={prediction_hash}",
                    f"manifest_sha256={job['manifest_sha256']}",
                    f"eval_frozen_manifest={job['frozen_manifest']}",
                    f"checkpoint_sha256={job['checkpoint_sha256']}",
                    f"experiment_config_sha256={job['experiment_config_sha256']}",
                    f"run_manifest_sha256={job['run_manifest_sha256']}",
                    f"config_tree_sha256={job['config_tree_sha256']}",
                    f"resolved_config_source_kind={job['resolved_config_source_kind']}",
                    f"resolved_config_sha256={job['resolved_config_sha256']}",
                    f"execution_shard_sha256={audit_by_universe[universe_id]['execution_shard_sha256']}",
                    f"canary_gate_sha256={audit_by_universe[universe_id]['canary_gate_sha256']}",
                    f"full_audit_sha256={audit_sha256}",
                ]
            )
            row = {
                "experiment_id": registry_id,
                "family": job["family"],
                "probe": str(output["probe_label"]),
                "model": job["model"],
                "backend": job["model"],
                "model_variant": job["model_variant"],
                "layer": job["layer"],
                "prefix": job["prefix"],
                "horizon": job["horizon"],
                "seed": job["seed"],
                "train_split": "train",
                "eval_split": "val",
                "dataset_manifest": job["manifest"],
                "config": job["experiment_config"],
                "checkpoint": job["checkpoint"],
                "launcher_job_id": str(summary["job_name"]),
                "raw_artifact_path": job["summary_path"],
                "canonical_metric_path": f"results/canonical/{FAMILY_FILE[job['family']]}",
                "metric_name": str(output["metric_name"]),
                "metric_value": number_text(float(metric_value)),
                "status": "COMPLETE",
                "status_reason": "Q00 Phase-1 exact frozen eval passed full artifact audit",
                "supersedes": "",
                "git_commit": git_commit[:7],
                "notes": notes,
            }
            if registry_id in by_id:
                raise PromotionError(f"duplicate primary row while building registry: {registry_id}")
            by_id[registry_id] = row
            rows.append(row)
    if len(rows) != EXPECTED_ATOMIC_ROWS:
        raise PromotionError(f"primary registry row count is not {EXPECTED_ATOMIC_ROWS}: {len(rows)}")
    return rows, by_id


def required_summary_text(summary: Mapping[str, Any], name: str) -> str:
    value = summary_metric(summary, name, required=True)
    assert value is not None
    return number_text(value)


def optional_summary_text(summary: Mapping[str, Any], name: str) -> str:
    value = summary_metric(summary, name, required=False)
    return "" if value is None else number_text(value)


def build_canonical_additions(
    jobs: Sequence[Mapping[str, str]],
    primary_by_id: Mapping[str, Mapping[str, str]],
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    additions = {name: [] for name in CANONICAL_NAMES}
    for job in jobs:
        universe_id = job["universe_id"]
        summary = summaries[universe_id]
        outputs = parse_json_list(job["expected_outputs_json"], label=f"{universe_id}.expected_outputs")
        primary_rows = [primary_by_id[str(output["registry_id"])] for output in outputs]
        for row in primary_rows:
            additions["all_results.csv"].append(
                {
                    "family": row["family"],
                    "probe": row["probe"],
                    "model": row["model"],
                    "layer": row["layer"],
                    "prefix": row["prefix"],
                    "metric_name": row["metric_name"],
                    "metric_value": row["metric_value"],
                    "experiment_id": row["experiment_id"],
                    "raw_artifact_path": row["raw_artifact_path"],
                }
            )
        family = job["family"]
        if family in {"A1", "A2"}:
            name = FAMILY_FILE[family]
            additions[name].extend(
                [{field: row.get(field, "") for field in SIMPLE_CANONICAL_FIELDS} for row in primary_rows]
            )
            continue
        if len(primary_rows) != 1:
            raise PromotionError(f"non-A1 job must have one primary row: {universe_id}")
        primary = primary_rows[0]
        common = {
            "experiment_id": primary["experiment_id"],
            "model": primary["model"],
            "layer": primary["layer"],
            "prefix": primary["prefix"],
            "seed": primary["seed"],
            "eval_split": primary["eval_split"],
            "status": "COMPLETE",
            "raw_artifact_path": primary["raw_artifact_path"],
        }
        if family == "B2":
            additions["b2_results.csv"].append(
                {
                    **common,
                    "model_variant": primary["model_variant"],
                    "angular_error_mean": primary["metric_value"],
                    "angular_error_median": optional_summary_text(summary, "median_ang_err_deg"),
                    "top1": required_summary_text(summary, "top1"),
                    "top3": required_summary_text(summary, "top3"),
                    "mean_log_dist_err": required_summary_text(summary, "mean_log_dist_err"),
                }
            )
        elif family == "C1":
            global_r1 = float(required_summary_text(summary, "global_R@1"))
            no_action = float(required_summary_text(summary, "no_action_global_R@1"))
            shuffled = float(required_summary_text(summary, "shuffled_action_global_R@1"))
            last_observation = float(required_summary_text(summary, "last_observation_global_R@1"))
            additions["c1_results.csv"].append(
                {
                    **common,
                    "global_R@1": number_text(global_r1),
                    "global_R@5": required_summary_text(summary, "global_R@5"),
                    "no_action_R@1": number_text(no_action),
                    "shuffled_action_R@1": number_text(shuffled),
                    "last_observation_R@1": number_text(last_observation),
                    "action_gain_vs_no_action": number_text(global_r1 - no_action),
                    "action_gain_vs_shuffle": number_text(global_r1 - shuffled),
                    "gain_over_copy": number_text(global_r1 - last_observation),
                    "action_gain_cos": required_summary_text(summary, "action_gain_cos"),
                }
            )
        elif family == "C2":
            additions["c2_results.csv"].append(
                {
                    **common,
                    "horizon": "all",
                    "metric_name": "global_R@1",
                    "metric_value": primary["metric_value"],
                    "global_R@5": required_summary_text(summary, "global_R@5"),
                    "final_pose_error": required_summary_text(summary, "final_pose_error"),
                    "drift_rate": required_summary_text(summary, "drift_rate"),
                    "loop_closure_error": optional_summary_text(summary, "loop_closure_error"),
                }
            )
            for horizon in ("1", "2", "4"):
                additions["c2_results.csv"].append(
                    {
                        **common,
                        "horizon": horizon,
                        "metric_name": "pose_error",
                        "metric_value": required_summary_text(summary, f"pose_error_h{horizon}"),
                        "global_R@5": "",
                        "final_pose_error": "",
                        "drift_rate": "",
                        "loop_closure_error": "",
                    }
                )
        elif family == "C3":
            all_metrics = (
                ("all", "intervention_validity"),
                ("all", "intervention_margin"),
                ("all", "counterfactual_consistency"),
                ("all", "global_R@1"),
                ("1", "cf_cos"),
                ("2", "cf_cos"),
                ("4", "cf_cos"),
            )
            for horizon, metric_name in all_metrics:
                source_name = metric_name if horizon == "all" else f"cf_cos_h{horizon}"
                additions["c3_results.csv"].append(
                    {
                        **common,
                        "horizon": horizon,
                        "metric_name": metric_name,
                        "metric_value": required_summary_text(summary, source_name),
                    }
                )
        else:
            raise PromotionError(f"unsupported family while building canonical rows: {family}")

    expected_counts = {
        "all_results.csv": 105,
        "a1_results.csv": 19,
        "a2_results.csv": 15,
        "b2_results.csv": 1,
        "c1_results.csv": 28,
        "c2_results.csv": 128,
        "c3_results.csv": 70,
    }
    actual_counts = {name: len(rows) for name, rows in additions.items()}
    if actual_counts != expected_counts:
        raise PromotionError(
            f"canonical addition count mismatch: expected={expected_counts} actual={actual_counts}"
        )
    return additions


def stage_append_only(
    pre_dir: Path,
    stage_dir: Path,
    primary_rows: Sequence[Mapping[str, str]],
    canonical_additions: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, Any]:
    registry_source = pre_dir / "registry.csv"
    registry_fields, registry_before = read_csv(registry_source)
    if registry_fields != REGISTRY_FIELDS:
        raise PromotionError("registry schema differs from exact Phase-1 schema")
    existing_ids = [row["experiment_id"] for row in registry_before]
    if len(existing_ids) != len(set(existing_ids)):
        raise PromotionError("pre-promotion registry contains duplicate experiment IDs")
    expected_ids = {row["experiment_id"] for row in primary_rows}
    overlap = expected_ids & set(existing_ids)
    if overlap:
        raise PromotionError(f"expected promotion IDs already exist in registry: {sorted(overlap)}")
    registry_after = [dict(row) for row in registry_before] + [dict(row) for row in primary_rows]
    registry_stage = stage_dir / "registry.csv"
    write_csv(registry_stage, REGISTRY_FIELDS, registry_after)
    _, registry_roundtrip = read_csv(registry_stage)
    if registry_roundtrip[: len(registry_before)] != registry_before:
        raise PromotionError("staging rewrote or reordered an existing registry row")
    if row_counter(registry_roundtrip, REGISTRY_FIELDS) != (
        row_counter(registry_before, REGISTRY_FIELDS) + row_counter(primary_rows, REGISTRY_FIELDS)
    ):
        raise PromotionError("staged registry is not an exact append-only logical-row union")

    complete_keys = Counter(
        (
            row.get("family", ""),
            row.get("probe", ""),
            row.get("model", ""),
            row.get("model_variant", ""),
            row.get("layer", ""),
            row.get("prefix", ""),
            row.get("horizon", ""),
            row.get("seed", ""),
            row.get("eval_split", ""),
            row.get("metric_name", ""),
        )
        for row in registry_roundtrip
        if row.get("status") == "COMPLETE"
    )
    duplicates = [list(key) for key, count in complete_keys.items() if count > 1]
    if duplicates:
        raise PromotionError(f"promotion creates duplicate COMPLETE metric keys: {duplicates[:20]}")

    stage_canonical = stage_dir / "canonical"
    stage_canonical.mkdir()
    canonical_meta: dict[str, Any] = {}
    for name in CANONICAL_NAMES:
        source = pre_dir / name
        fields, before = read_csv(source)
        if fields != CANONICAL_FIELDS[name]:
            raise PromotionError(f"canonical schema differs from expected schema: {name}")
        additions = [dict(row) for row in canonical_additions[name]]
        for row in additions:
            if row.get("experiment_id") not in expected_ids:
                raise PromotionError(f"non-expected ID in canonical addition {name}: {row}")
        existing_canonical_ids = {row.get("experiment_id", "") for row in before}
        if expected_ids & existing_canonical_ids:
            raise PromotionError(f"expected ID already exists in canonical table {name}")
        after = [dict(row) for row in before] + additions
        destination = stage_canonical / name
        write_csv(destination, fields, after)
        _, roundtrip = read_csv(destination)
        if roundtrip[: len(before)] != before:
            raise PromotionError(f"staging rewrote or reordered an existing canonical row: {name}")
        if row_counter(roundtrip, fields) != row_counter(before, fields) + row_counter(additions, fields):
            raise PromotionError(f"canonical staging is not exact append-only union: {name}")
        canonical_meta[name] = {
            "before_rows": len(before),
            "added_rows": len(additions),
            "after_rows": len(roundtrip),
            "before_sha256": sha256(source),
            "staged_sha256": sha256(destination),
            "added_experiment_ids": sorted({row["experiment_id"] for row in additions}),
        }
    return {
        "registry": {
            "before_rows": len(registry_before),
            "added_rows": len(primary_rows),
            "after_rows": len(registry_roundtrip),
            "before_sha256": sha256(registry_source),
            "staged_sha256": sha256(registry_stage),
            "added_experiment_ids": sorted(expected_ids),
        },
        "canonical": canonical_meta,
    }


@contextmanager
def exclusive_promotion_lock() -> Iterator[Path]:
    lock_path = ROOT / "results/analysis/q00_1096/.q00_phase1_promotion.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PromotionError(f"another Q00 promotion holds {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} time={now()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def update_journal(path: Path, journal: dict[str, Any], state: str, **extra: Any) -> None:
    journal.update({"state": state, "updated_at": now(), **extra})
    write_json_atomic(path, journal)


def verify_mutable_inputs(binding_paths: Mapping[str, Path], binding_hashes: Mapping[str, str]) -> None:
    verify_hashes(binding_paths, binding_hashes, label="frozen/audited input")


def add_binding(
    paths: dict[str, Path],
    hashes: dict[str, str],
    *,
    key: str,
    path: Path,
    expected_hash: str,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise PromotionError(f"invalid binding SHA256 for {key}: {expected_hash!r}")
    if key in paths and (paths[key] != path or hashes[key] != expected_hash):
        raise PromotionError(f"conflicting frozen binding for {key}")
    paths[key] = path
    hashes[key] = expected_hash


def run_consistency(post_dir: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/check_project_consistency.py", "--strict-config"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (post_dir / "project_consistency.log").write_text(proc.stdout)
    if proc.returncode:
        raise PromotionError(f"post-promotion project consistency failed rc={proc.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all frozen/audited inputs and construct rows in memory without staging or promotion",
    )
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    expected_bundle_parent = (ROOT / "reports/q00_phase1_runs").resolve()
    ensure_under(bundle, expected_bundle_parent, label="bundle")

    jobs_path = bundle / "inputs/wave_b_99.tsv"
    expected_ids_path = bundle / "inputs/expected_registry_ids.json"
    _, jobs = read_csv(jobs_path, delimiter="\t")
    if len(jobs) != EXPECTED_PARENT_ROWS or len({job["universe_id"] for job in jobs}) != EXPECTED_PARENT_ROWS:
        raise PromotionError("frozen Wave-B TSV is not exactly 99 unique universe rows")
    metadata = read_json(bundle / "bundle.json")
    run_tag = str(metadata.get("run_tag", ""))
    if not run_tag or int(metadata.get("wave_b_parent_rows", -1)) != EXPECTED_PARENT_ROWS:
        raise PromotionError("bundle metadata does not describe the exact 99-row Wave-B run")
    if int(metadata.get("wave_b_atomic_registry_ids", -1)) != EXPECTED_ATOMIC_ROWS:
        raise PromotionError("bundle metadata atomic-ID count mismatch")
    source_hashes = metadata.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise PromotionError("bundle metadata lacks frozen source_hashes")

    validate_job_paths(bundle, jobs, run_tag)
    validate_frozen_sources(bundle, jobs)
    verify_script_inventory(bundle)
    verify_runtime_inventory(bundle)
    ordered_ids, expected_by_id = expected_outputs(jobs)
    expected_ids = read_json(expected_ids_path)
    if not isinstance(expected_ids, list) or expected_ids != ordered_ids:
        raise PromotionError("expected_registry_ids.json is not the exact ordered full-identity ID list")
    jobs_sha = sha256(jobs_path)
    expected_ids_sha = sha256(expected_ids_path)
    core_source_hashes = {
        "inputs/wave_b_99.tsv": jobs_sha,
        "inputs/expected_registry_ids.json": expected_ids_sha,
        "provenance/script_inventory.json": sha256(bundle / "provenance/script_inventory.json"),
        "provenance/runtime_inventory.json": sha256(bundle / "provenance/runtime_inventory.json"),
    }
    for key, expected_hash in core_source_hashes.items():
        if source_hashes.get(key) != expected_hash:
            raise PromotionError(f"bundle source_hashes mismatch for {key}")
    audit_by_universe, artifact_hashes, summaries, audit_sha = validate_full_audit(
        bundle,
        jobs,
        ordered_ids,
        jobs_sha256=jobs_sha,
        expected_ids_sha256=expected_ids_sha,
    )
    git_commit = (bundle / "provenance/git_commit.txt").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise PromotionError("invalid frozen git commit")
    if git_commit != metadata.get("git_commit"):
        raise PromotionError("git_commit.txt differs from audit-locked bundle metadata")
    primary_rows, primary_by_id = build_primary_registry_rows(
        jobs,
        audit_by_universe,
        summaries,
        audit_sha256=audit_sha,
        git_commit=git_commit,
    )
    if set(primary_by_id) != set(expected_by_id):
        raise PromotionError("constructed primary registry-ID set differs from frozen expected set")
    canonical_additions = build_canonical_additions(jobs, primary_by_id, summaries)

    protected = protected_paths()
    protected_snapshot = read_json(bundle / "inputs/protected_before.json")
    if protected_snapshot != metadata.get("protected_before"):
        raise PromotionError("protected_before.json differs from audit-locked bundle metadata")
    protected_frozen = frozen_protected_hashes(bundle)
    verify_hashes(protected, protected_frozen, label="protected pre-promotion")

    binding_paths: dict[str, Path] = {
        "bundle.json": bundle / "bundle.json",
        "wave_b_99.tsv": jobs_path,
        "expected_registry_ids.json": expected_ids_path,
        "full_audit.json": bundle / "audit/full_audit.json",
        "full_audit_lock.json": bundle / "audit/full_audit_lock.json",
        "full_results.csv": bundle / "audit/full_results.csv",
        "script_inventory.json": bundle / "provenance/script_inventory.json",
        "runtime_inventory.json": bundle / "provenance/runtime_inventory.json",
        "protected_before.json": bundle / "inputs/protected_before.json",
        "git_commit.txt": bundle / "provenance/git_commit.txt",
        "frozen_universe.csv": bundle
        / "inputs/source_of_truth/experiments/task_status_1096_before_expand_20260822.csv",
        "frozen_reconciliation.csv": bundle
        / "inputs/source_of_truth/results/analysis/q00_1096/universe_1096_reconciled.csv",
        "frozen_execution_waves.csv": bundle
        / "inputs/source_of_truth/results/analysis/q00_1096/execution_waves.csv",
    }
    binding_hashes = {name: sha256(path) for name, path in binding_paths.items()}
    for relative, expected_hash_raw in source_hashes.items():
        expected_hash = str(expected_hash_raw)
        path = (
            bundle / relative
            if str(relative).startswith(("inputs/", "provenance/"))
            else safe_project_path(str(relative), label=f"bundle_source:{relative}")
        )
        add_binding(
            binding_paths,
            binding_hashes,
            key=f"bundle_source:{relative}",
            path=path.resolve(),
            expected_hash=expected_hash,
        )
    for relative, expected_hash in artifact_hashes.items():
        key = f"artifact:{relative}"
        add_binding(
            binding_paths,
            binding_hashes,
            key=key,
            path=safe_project_path(relative, label=key),
            expected_hash=expected_hash,
        )
    local_artifact_hash_fields = {
        "hydra_config": "hydra_config_sha256",
        "hydra_overrides": "hydra_overrides_sha256",
        "experiment_config": "experiment_config_sha256",
        "run_manifest": "run_manifest_sha256",
        "config_tree": "config_tree_sha256",
        "resolved_config_frozen": "resolved_config_sha256",
        "train_log": "train_log_sha256",
        "launcher_manifest": "launcher_manifest_sha256",
    }
    for job in jobs:
        for path_field, hash_field in {
            "manifest": "manifest_sha256",
            "frozen_manifest": "manifest_sha256",
            **local_artifact_hash_fields,
        }.items():
            relative = job[path_field]
            key = f"{path_field}:{relative}"
            add_binding(
                binding_paths,
                binding_hashes,
                key=key,
                path=safe_project_path(relative, label=key),
                expected_hash=job[hash_field],
            )
    verify_mutable_inputs(binding_paths, binding_hashes)

    plan = {
        "ok": True,
        "validate_only": bool(args.validate_only),
        "bundle": str(bundle),
        "run_tag": run_tag,
        "parent_universe_rows": len(jobs),
        "primary_registry_rows": len(primary_rows),
        "canonical_additions": {name: len(rows) for name, rows in canonical_additions.items()},
        "jobs_tsv_sha256": jobs_sha,
        "expected_registry_ids_sha256": expected_ids_sha,
        "full_audit_sha256": audit_sha,
        "protected_before": protected_frozen,
    }
    if args.validate_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    with exclusive_promotion_lock() as lock_path:
        # Lock acquisition is followed by a fresh protected check before any
        # staging work.  Other compliant writers use the same lock.
        verify_hashes(protected, protected_frozen, label="protected after lock acquisition")
        promotion = bundle / "promotion"
        if promotion.exists():
            state = "UNKNOWN"
            journal_path = promotion / "journal.json"
            if journal_path.is_file():
                try:
                    state = str(read_json(journal_path).get("state", "UNKNOWN"))
                except PromotionError:
                    pass
            raise PromotionError(
                f"promotion directory already exists ({promotion}, journal state={state}); "
                "inspect/recover it before any retry"
            )
        pre = promotion / "pre"
        stage = promotion / "staging"
        post = promotion / "post"
        pre.mkdir(parents=True)
        stage.mkdir()
        post.mkdir()
        journal_path = promotion / "journal.json"
        journal: dict[str, Any] = {
            **plan,
            "lock_path": str(lock_path),
            "pid": os.getpid(),
            "created_at": now(),
            "committed_files": [],
        }
        update_journal(journal_path, journal, "PREPARING")

        commit_started = False
        try:
            for relative, source in protected.items():
                name = "registry.csv" if relative == "experiments/registry.csv" else Path(relative).name
                destination = pre / name
                shutil.copy2(source, destination)
                if sha256(destination) != protected_frozen[relative]:
                    raise PromotionError(f"pre-image copy hash mismatch: {relative}")
            stage_meta = stage_append_only(pre, stage, primary_rows, canonical_additions)
            stage_sources = {
                "experiments/registry.csv": stage / "registry.csv",
                **{
                    f"results/canonical/{name}": stage / "canonical" / name
                    for name in CANONICAL_NAMES
                },
            }
            stage_hashes = file_hashes(stage_sources)
            write_json_atomic(stage / "staging_audit.json", {"ok": True, **stage_meta, "stage_hashes": stage_hashes})
            update_journal(
                journal_path,
                journal,
                "STAGED",
                stage_hashes=stage_hashes,
                stage_metadata=stage_meta,
            )

            # Commit gate: inputs/artifacts and protected sources are checked a
            # second time under the exclusive lock immediately before replace.
            verify_script_inventory(bundle)
            verify_runtime_inventory(bundle)
            verify_mutable_inputs(binding_paths, binding_hashes)
            verify_hashes(protected, protected_frozen, label="protected second pre-commit check")
            commit_started = True
            update_journal(journal_path, journal, "COMMITTING")
            for relative in PROTECTED_RELATIVE:
                atomic_copy(stage_sources[relative], protected[relative])
                if sha256(protected[relative]) != stage_hashes[relative]:
                    raise PromotionError(f"post-copy hash mismatch: {relative}")
                journal["committed_files"].append(relative)
                update_journal(journal_path, journal, "COMMITTING")
            run_consistency(post)
            verify_hashes(protected, stage_hashes, label="protected post-commit")
            write_json_atomic(
                post / "metadata.json",
                {
                    "ok": True,
                    "committed_at": now(),
                    "parent_universe_rows": EXPECTED_PARENT_ROWS,
                    "primary_registry_rows": EXPECTED_ATOMIC_ROWS,
                    "old_registry_rows_changed": 0,
                    "old_canonical_rows_changed": 0,
                    "stage_hashes": stage_hashes,
                    "full_audit_sha256": audit_sha,
                },
            )
            update_journal(journal_path, journal, "COMMITTED", committed_at=now())
        except BaseException as exc:
            if commit_started:
                journal_errors: list[dict[str, str]] = []
                try:
                    update_journal(journal_path, journal, "ROLLING_BACK", error=repr(exc))
                except Exception as journal_exc:  # rollback must not depend on journal IO
                    journal_errors.append({"state": "ROLLING_BACK", "error": repr(journal_exc)})
                restore_sources = {
                    "experiments/registry.csv": pre / "registry.csv",
                    **{
                        f"results/canonical/{name}": pre / name
                        for name in CANONICAL_NAMES
                    },
                }
                rollback_errors = []
                for relative in PROTECTED_RELATIVE:
                    try:
                        atomic_copy(restore_sources[relative], protected[relative])
                    except Exception as rollback_exc:  # pragma: no cover - catastrophic IO failure
                        rollback_errors.append({"path": relative, "error": repr(rollback_exc)})
                if not rollback_errors:
                    try:
                        verify_hashes(protected, protected_frozen, label="protected rollback")
                    except Exception as verify_exc:  # pragma: no cover - catastrophic IO failure
                        rollback_errors.append({"path": "ALL", "error": repr(verify_exc)})
                try:
                    update_journal(
                        journal_path,
                        journal,
                        "ROLLED_BACK" if not rollback_errors else "ROLLBACK_FAILED",
                        error=repr(exc),
                        rollback_errors=rollback_errors,
                        journal_errors=journal_errors,
                    )
                except Exception as journal_exc:  # protected rollback already attempted
                    journal_errors.append({"state": "ROLLBACK_FINAL", "error": repr(journal_exc)})
                if rollback_errors:
                    raise PromotionError(
                        "promotion failed and rollback was incomplete: "
                        f"rollback_errors={rollback_errors}; journal_errors={journal_errors}; "
                        f"original={exc!r}"
                    ) from exc
            else:
                update_journal(journal_path, journal, "FAILED_BEFORE_COMMIT", error=repr(exc))
            raise

    print(
        json.dumps(
            {
                "ok": True,
                "added_parent_universe_ids": EXPECTED_PARENT_ROWS,
                "added_primary_registry_ids": EXPECTED_ATOMIC_ROWS,
                "old_registry_rows_changed": 0,
                "old_canonical_rows_changed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
