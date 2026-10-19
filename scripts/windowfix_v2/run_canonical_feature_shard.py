#!/usr/bin/env python3
"""Validate and run a reviewed canonical-v2 feature shard on an SCO worker.

This module never creates an SCO job.  It defaults to a read-only JSON command
preview; ``--execute`` is only for an already allocated, reviewed worker.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn


EXPECTED_PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"
EXPECTED_PREFIXES = [8, 12, 16, 24]
EXPECTED_HORIZONS = [1, 2, 4]
CERTIFIED_READY_VFMS = {
    "wan",
    "cogvideox",
    "minimax_h3",
    "dino",
    "vjepa2",
    "videomae",
}
CANARY_GATED_VFMS = {"aether", "f3r", "lingbot", "qwen2_5_vl"}
MLLM_VFMS = {"qwen2_5_vl"}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"ERROR: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def feature_protocol_sha(protocol: dict[str, Any]) -> str:
    window = protocol["canonical_window"]
    target = protocol["target"]
    payload = {
        "schema_version": 2,
        "protocol": protocol["canonical_protocol"],
        "sampling": window["sampling"],
        "canonical_obs_len": int(window["observations"]),
        "prefixes": [int(value) for value in protocol["streaming"]["prefix_lengths"]],
        "target_prefixes": [int(value) for value in target["prefix_lengths"]],
        "horizons": [int(value) for value in target["future_horizons"]],
        "include_anchors": True,
        "motion_step": float(window["motion_step"]),
        "rotation_weight": float(window["rotation_weight"]),
        "window_stride": int(window["window_stride"]),
        "max_windows_per_scene": int(window["max_windows_per_scene"]),
    }
    return canonical_json_sha(payload)


def read_regular_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot read {label}: {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must contain a JSON object: {path}")
    return value


def require_sha(path: Path, expected: Any, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        fail(f"bound {label} is missing or symlinked: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        fail(f"bound {label} has no valid SHA256")
    actual = sha256_file(path)
    if actual != expected:
        fail(f"bound {label} changed: expected={expected} actual={actual} path={path}")


def normalized_bound_path(value: Any) -> str:
    return str(Path(str(value or "")).resolve())


def validate_compact_binding(shard: dict[str, Any]) -> None:
    compact = shard.get("canonical_binding")
    if (
        not isinstance(compact, dict)
        or compact.get("schema_version") != 2
        or compact.get("mode") != "compact_obs28_source_v1"
    ):
        fail("compact canonical obs28 binding schema v2 is required")
    bindings: dict[str, dict[str, Any]] = {}
    for label in ("canonical_source", "namespace_lock", "label_lineage_audit"):
        binding = compact.get(label)
        if not isinstance(binding, dict):
            fail(f"missing compact binding: {label}")
        require_sha(Path(str(binding.get("path", ""))), binding.get("sha256"), label)
        bindings[label] = binding

    source = bindings["canonical_source"]
    source_path = Path(str(source["path"])).resolve()
    if (
        source.get("canonical_obs_len") != 28
        or source.get("source_prefix") != 24
        or int(source.get("rows", 0)) <= 0
        or int(source.get("scenes", 0)) <= 0
    ):
        fail("canonical source does not declare the complete p24/obs28 universe")
    sidecar_path = Path(str(source.get("sidecar_path", ""))).resolve()
    require_sha(sidecar_path, source.get("sidecar_sha256"), "canonical source sidecar")
    sidecar = read_regular_json(sidecar_path, "canonical source sidecar")
    if (
        sidecar.get("sha256") != source.get("sha256")
        or sidecar.get("prefix_lengths") != [24]
        or sidecar.get("action_horizons") != EXPECTED_HORIZONS
        or sidecar.get("num_samples") != source.get("rows")
        or sidecar.get("num_included_scenes") != source.get("scenes")
    ):
        fail("canonical source sidecar contract mismatch")

    lock_path = Path(str(bindings["namespace_lock"]["path"])).resolve()
    lock = read_regular_json(lock_path, "namespace lock")
    if lock.get("schema_version") != 2 or lock.get("canonical_protocol") != EXPECTED_PROTOCOL:
        fail("namespace lock protocol/schema mismatch")
    if lock.get("protocol_file_sha256") != shard.get("protocol_file_sha256"):
        fail("namespace lock protocol-file SHA mismatch")
    lock_source = lock.get("canonical_source")
    if not isinstance(lock_source, dict):
        fail("namespace lock has no canonical_source object")
    expected_lock_source = {
        "path": str(source_path),
        "sha256": source.get("sha256"),
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": source.get("sidecar_sha256"),
        "rows": source.get("rows"),
        "scenes": source.get("scenes"),
        "canonical_obs_len": 28,
        "source_prefix": 24,
    }
    actual_lock_source = dict(lock_source)
    actual_lock_source["path"] = normalized_bound_path(lock_source.get("path"))
    actual_lock_source["sidecar_path"] = normalized_bound_path(lock_source.get("sidecar_path"))
    if actual_lock_source != expected_lock_source:
        fail("namespace lock canonical-source binding mismatch")
    if lock.get("canonical_universe_sha256") != compact.get("canonical_universe_sha256"):
        fail("namespace lock canonical-universe SHA mismatch")
    if lock.get("feature_roots") != compact.get("feature_roots"):
        fail("namespace lock feature-root binding mismatch")
    policy = lock.get("override_policy", {})
    if policy.get("fail_closed") is not True or policy.get("row_roots_are_not_authoritative") is not True:
        fail("namespace lock does not reject row-level legacy roots")

    audit_path = Path(str(bindings["label_lineage_audit"]["path"])).resolve()
    audit = read_regular_json(audit_path, "label-lineage audit")
    if (
        audit.get("audit_schema_version") != 2
        or audit.get("audit_type") != "canonical_v2_compact_source_label_lineage"
        or audit.get("status") != "pass"
        or audit.get("problem_count") != 0
        or audit.get("fail_closed") is not True
    ):
        fail("label-lineage receipt is not a fail-closed full PASS")
    namespace = audit.get("namespace_binding", {})
    if (
        normalized_bound_path(namespace.get("path")) != str(lock_path)
        or namespace.get("sha256") != bindings["namespace_lock"].get("sha256")
        or namespace.get("feature_roots") != compact.get("feature_roots")
    ):
        fail("label-lineage receipt does not bind the current namespace lock")
    if (
        normalized_bound_path(audit.get("source_manifest")) != str(source_path)
        or audit.get("source_manifest_sha256") != source.get("sha256")
        or audit.get("source_sidecar_sha256") != source.get("sidecar_sha256")
        or audit.get("source_sidecar_declared_samples") != source.get("rows")
    ):
        fail("label-lineage receipt does not bind the canonical source")
    label_binding = bindings["label_lineage_audit"]
    if (
        audit.get("protocol") != EXPECTED_PROTOCOL
        or audit.get("protocol_file_sha256") != shard.get("protocol_file_sha256")
        or audit.get("canonical_universe_sha256") != compact.get("canonical_universe_sha256")
        or audit.get("label_lineage_universe_sha256")
        != label_binding.get("label_lineage_universe_sha256")
        or audit.get("input_anchor_future_mapping_sha256") != label_binding.get("mapping_sha256")
        or audit.get("per_prefix_mapping_sha256") != label_binding.get("per_prefix_mapping_sha256")
    ):
        fail("label-lineage receipt identity/mapping binding mismatch")
    full_counts = label_binding.get("full_raw_label_counts")
    if not isinstance(full_counts, dict) or any(
        audit.get("counts", {}).get(key) != value for key, value in full_counts.items()
    ):
        fail("label-lineage receipt full raw-label counts mismatch")


def validate_backend_canary_binding(shard: dict[str, Any]) -> None:
    vfm = str(shard.get("vfm"))
    binding = shard.get("backend_canary_binding")
    if vfm not in CANARY_GATED_VFMS:
        if binding is not None:
            fail(f"non-gated backend has an unexpected canary binding: {vfm}")
        return
    if not isinstance(binding, dict):
        fail(f"canary-gated backend lacks a PASS receipt binding: {vfm}")
    path = Path(str(binding.get("path", "")))
    require_sha(path, binding.get("sha256"), f"{vfm} backend canary receipt")
    receipt = read_regular_json(path, f"{vfm} backend canary receipt")
    source_bindings = shard["source_bindings"]
    extractor_label = "mllm_extractor" if vfm == "qwen2_5_vl" else "main_extractor"
    extractor = source_bindings.get(extractor_label, {})
    canary = source_bindings.get("backend_determinism_canary", {})
    expected = {
        "schema_version": 1,
        "audit_type": "canonical_v2_backend_cross_process_order_canary",
        "status": "pass",
        "vfm": vfm,
        "model_id": shard.get("model_id"),
        "timestep": shard.get("timestep"),
        "protocol": EXPECTED_PROTOCOL,
        "protocol_file_sha256": shard.get("protocol_file_sha256"),
        "extractor": extractor.get("path"),
        "extractor_sha256": extractor.get("sha256"),
        "canary_script": canary.get("path"),
        "canary_script_sha256": canary.get("sha256"),
    }
    if any(receipt.get(field) != value for field, value in expected.items()):
        fail(f"{vfm} backend canary identity mismatch")
    if int(receipt.get("layer", -10**9)) not in shard["layers"]:
        fail(f"{vfm} backend canary layer is not present in this shard")
    results = receipt.get("results")
    if not isinstance(results, dict) or set(results) != {"streaming_prefix", "target_isolated"}:
        fail(f"{vfm} backend canary lacks both extraction modes")
    for mode in ("streaming_prefix", "target_isolated"):
        comparison = results.get(mode, {}).get("comparison", {})
        digest = comparison.get("comparison_sha256")
        if (
            int(comparison.get("files_compared", 0)) <= 0
            or int(comparison.get("safetensors_compared", 0)) <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            fail(f"{vfm} backend canary has no valid {mode} comparison")


def validate_shard(path: Path) -> dict[str, Any]:
    shard = read_regular_json(path, "shard")
    if shard.get("schema_version") != 1 or shard.get("status") != "READY":
        fail("shard is not READY schema v1")
    stored_sha = shard.get("content_sha256")
    unhashed = dict(shard)
    unhashed.pop("content_sha256", None)
    if stored_sha != canonical_json_sha(unhashed):
        fail("shard content_sha256 mismatch")
    expected = {
        "canonical_protocol": EXPECTED_PROTOCOL,
        "canonical_obs_len": 28,
        "prefixes": EXPECTED_PREFIXES,
        "horizons": EXPECTED_HORIZONS,
        "include_anchors": True,
        "source": "scannetpp",
        "no_alias": True,
    }
    for field, value in expected.items():
        if shard.get(field) != value:
            fail(f"non-canonical shard field {field}: {shard.get(field)!r}")
    if shard.get("mode") not in {"streaming_prefix", "target_isolated"}:
        fail(f"unsupported mode: {shard.get('mode')!r}")
    if shard.get("vfm") not in CERTIFIED_READY_VFMS | CANARY_GATED_VFMS:
        fail(f"backend is not certified READY for canonical-v2: {shard.get('vfm')!r}")
    if not shard.get("layers") or any(not isinstance(layer, int) for layer in shard["layers"]):
        fail("layers must be a non-empty integer list")
    if int(shard.get("n_gpus", 0)) < 1:
        fail("n_gpus must be positive")
    if int(shard.get("scene_end", 0)) <= int(shard.get("scene_start", 0)):
        fail("invalid scene range")

    protocol_path = Path(str(shard.get("protocol_path", "")))
    require_sha(protocol_path, shard.get("protocol_file_sha256"), "protocol")
    protocol = read_regular_json(protocol_path, "protocol")
    if protocol.get("canonical_protocol") != EXPECTED_PROTOCOL:
        fail("bound protocol id mismatch")
    computed_feature_sha = feature_protocol_sha(protocol)
    if computed_feature_sha != shard.get("protocol_sha256"):
        fail("bound feature protocol SHA mismatch")
    declared_feature_sha = protocol.get("canonical_protocol_sha256")
    if declared_feature_sha is not None and declared_feature_sha != computed_feature_sha:
        fail("protocol declares a stale canonical_protocol_sha256")
    source_bindings = shard.get("source_bindings")
    if not isinstance(source_bindings, dict):
        fail("source_bindings missing")
    for label, binding in source_bindings.items():
        if not isinstance(binding, dict):
            fail(f"invalid source binding: {label}")
        require_sha(Path(str(binding.get("path", ""))), binding.get("sha256"), label)
    validate_compact_binding(shard)
    validate_backend_canary_binding(shard)
    return shard


def validate_batch(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    batch = read_regular_json(path, "batch")
    if batch.get("schema_version") != 1 or batch.get("status") != "READY":
        fail("batch is not READY schema v1")
    unhashed = dict(batch)
    stored_sha = unhashed.pop("content_sha256", None)
    if stored_sha != canonical_json_sha(unhashed):
        fail("batch content_sha256 mismatch")
    entries = batch.get("shards")
    if not isinstance(entries, list) or not entries:
        fail("batch contains no shard bindings")
    shards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            fail("invalid batch shard binding")
        shard_path = Path(str(entry.get("path", "")))
        require_sha(shard_path, entry.get("sha256"), "batch shard")
        shard = validate_shard(shard_path)
        if shard.get("shard_id") != entry.get("shard_id"):
            fail(f"batch shard id mismatch: {shard_path}")
        if shard.get("plan_id") != batch.get("plan_id") or shard.get("vfm") != batch.get("vfm"):
            fail(f"batch/shard plan or VFM mismatch: {shard_path}")
        if shard["shard_id"] in seen:
            fail(f"duplicate shard in batch: {shard['shard_id']}")
        seen.add(shard["shard_id"])
        shards.append(shard)
    return batch, shards


def ensure_no_symlink_components(path: Path, stop: Path) -> None:
    path_abs = path.absolute()
    stop_abs = stop.absolute()
    try:
        relative = path_abs.relative_to(stop_abs)
    except ValueError:
        fail(f"feature root escapes dataset root: {path_abs} vs {stop_abs}")
    current = stop_abs
    if current.is_symlink():
        fail(f"dataset root is a symlink: {current}")
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            fail(f"versioned feature root contains a symlink component: {current}")


def extractor_command(shard: dict[str, Any], start: int, end: int) -> list[str]:
    is_mllm = shard["vfm"] in MLLM_VFMS
    command = [
        str(shard["python"]),
        "-m",
        "features.run_inscene15k_mllm" if is_mllm else "features.run_inscene15k",
    ]
    if is_mllm:
        command.extend(
            [
                "--backend",
                "qwen2_5_vl",
                "--vfm-name",
                str(shard["vfm"]),
            ]
        )
    else:
        command.extend(["--vfm", str(shard["vfm"])])
    command.extend([
        "--mode",
        str(shard["mode"]),
        "--data-root",
        str(shard["dataset_data_root"]),
        "--out-root",
        str(shard["feature_root"]),
        "--source",
        "scannetpp",
        "--t",
        str(shard["timestep"]),
        "--output-layers",
        *[str(layer) for layer in shard["layers"]],
        "--prefix-lengths",
        "8,12,16,24",
        "--prefix-max-len",
        "24",
        "--streaming-window-obs-len",
        "28",
        "--streaming-motion-step",
        "0.35",
        "--streaming-rotation-weight",
        "0.5",
        "--streaming-window-stride",
        "8",
        "--streaming-max-windows-per-scene",
        "4",
        "--start",
        str(start),
        "--end",
        str(end),
    ])
    model_id = str(shard.get("model_id") or "").strip()
    if model_id:
        command.extend(["--model-id", model_id])
    if shard["mode"] == "target_isolated":
        command.extend(
            [
                "--target-from-streaming-windows",
                "--target-prefix-lengths",
                "8,12,16,24",
                "--target-horizons",
                "1,2,4",
            ]
        )
    if shard["vfm"] in {"wan", "minimax_h3"}:
        command.extend(["--prompt", "."])
    elif is_mllm:
        command.extend(["--prompt", "Describe the spatial layout of this scene."])
    return command


def shard_commands(shard: dict[str, Any]) -> list[dict[str, Any]]:
    start = int(shard["scene_start"])
    end = int(shard["scene_end"])
    n_gpus = int(shard["n_gpus"])
    commands: list[dict[str, Any]] = []
    for gpu in range(n_gpus):
        item_start = start + (end - start) * gpu // n_gpus
        item_end = start + (end - start) * (gpu + 1) // n_gpus
        if item_start < item_end:
            commands.append(
                {
                    "gpu": gpu,
                    "start": item_start,
                    "end": item_end,
                    "argv": extractor_command(shard, item_start, item_end),
                }
            )
    return commands


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def run_shard(shard: dict[str, Any], *, execute: bool) -> dict[str, Any]:
    commands = shard_commands(shard)
    preview = {"dry_run": not execute, "shard_id": shard["shard_id"], "commands": commands}
    if not execute:
        return preview

    dataset_root = Path(str(shard["dataset_data_root"])).absolute().parent
    feature_root = Path(str(shard["feature_root"])).absolute()
    ensure_no_symlink_components(feature_root, dataset_root)
    namespace = str(shard["feature_namespace"])
    if feature_root.name != namespace:
        fail(f"canonical feature-root basename mismatch: {feature_root.name!r} != {namespace!r}")
    forbidden = {"FEAT_STREAMING", "FEAT_TARGET", "FEAT_STREAMING_BY_PREFIX"}
    if any(part in forbidden for part in feature_root.parts):
        fail(f"legacy/alias namespace is forbidden: {feature_root}")
    if feature_root.exists() and (feature_root / ".SEALED").exists():
        fail(f"feature root is sealed and immutable: {feature_root}")
    feature_root.mkdir(parents=True, exist_ok=True)
    ensure_no_symlink_components(feature_root, dataset_root)

    lock_dir = feature_root / ".windowfix_v2_locks" / str(shard["vfm"]) / str(shard["source"])
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (lock_dir / f"{shard['mode']}.lock").open("a+")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-p{os.getpid()}"
    log_dir = (
        feature_root
        / ".windowfix_v2_logs"
        / str(shard["vfm"])
        / str(shard["source"])
        / str(shard["shard_id"])
        / attempt
    )
    log_dir.mkdir(parents=True, exist_ok=False)
    receipt = log_dir / "receipt.json"
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "shard_id": shard["shard_id"],
        "shard_content_sha256": shard["content_sha256"],
        "protocol_file_sha256": shard["protocol_file_sha256"],
        "protocol_sha256": shard["protocol_sha256"],
        "mode": shard["mode"],
        "vfm": shard["vfm"],
        "model_id": shard["model_id"],
        "timestep": shard["timestep"],
        "layers": shard["layers"],
        "prefixes": shard["prefixes"],
        "attempt": attempt,
        "status": "RUNNING",
    }
    write_json_atomic(receipt, provenance)

    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in shard.get("environment", {}).items()})
    environment.pop("EXTRACT_DONE_INDEX", None)
    environment["FEATURE_NAMESPACE"] = namespace
    project_root = str(shard["project_root"])
    extra_pythonpath = environment.pop("WINDOWFIX_EXTRA_PYTHONPATH", "")
    environment["PYTHONPATH"] = ":".join(
        item for item in (project_root, extra_pythonpath, environment.get("PYTHONPATH", "")) if item
    )

    processes: list[tuple[subprocess.Popen, Any, dict[str, Any], Path]] = []

    def terminate_children(_signum: int, _frame: Any) -> None:
        for process, _handle, _item, _log in processes:
            if process.poll() is None:
                process.terminate()

    old_handlers = {
        signum: signal.signal(signum, terminate_children) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for item in commands:
            log_path = log_dir / f"gpu{item['gpu']}_{item['start']}_{item['end']}.log"
            handle = log_path.open("wb")
            child_env = dict(environment)
            child_env["CUDA_VISIBLE_DEVICES"] = str(item["gpu"])
            process = subprocess.Popen(
                item["argv"],
                cwd=project_root,
                env=child_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, handle, item, log_path))
        results = []
        for process, handle, item, log_path in processes:
            returncode = process.wait()
            handle.close()
            results.append(
                {
                    "gpu": item["gpu"],
                    "start": item["start"],
                    "end": item["end"],
                    "returncode": returncode,
                    "log": str(log_path),
                }
            )
        failed = [item for item in results if item["returncode"] != 0]
        provenance.update({"status": "FAILED" if failed else "COMPLETE", "processes": results})
        write_json_atomic(receipt, provenance)
        if failed:
            fail(f"{len(failed)} extractor processes failed; receipt={receipt}")
    finally:
        for process, handle, _item, _log_path in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not handle.closed:
                handle.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    return {"dry_run": False, "shard_id": shard["shard_id"], "receipt": str(receipt)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--shard", type=Path)
    selection.add_argument("--batch", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.shard is not None:
        shard = validate_shard(args.shard.resolve())
        result = run_shard(shard, execute=args.execute)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    batch, shards = validate_batch(args.batch.resolve())
    if not args.execute:
        result = {
            "dry_run": True,
            "batch_id": batch["batch_id"],
            "serialization": "sequential",
            "shards": [run_shard(shard, execute=False) for shard in shards],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    completed = []
    for shard in shards:
        completed.append(run_shard(shard, execute=True))
    print(
        json.dumps(
            {
                "dry_run": False,
                "batch_id": batch["batch_id"],
                "serialization": "sequential",
                "shards": completed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
