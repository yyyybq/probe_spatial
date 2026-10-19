#!/usr/bin/env python3
"""Build-time primitives and fail-closed verifier for the v2 canary receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
RECEIPT_TYPE = "tstep_canonical_v2_canary_receipt"
SNAPSHOT_TYPE = "tstep_canonical_v2_canary_input_snapshot"
PROTOCOL_ID = "inscene_streaming_canonical_v2_obs28_h124"
PROTOCOL_SHA256 = "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
OBS_LEN = 28
PREFIXES = (8, 12, 16, 24)
HORIZONS = (1, 2, 4)
LAYERS = (15, 20, 25)
TIMESTEP = 499
VFM = "wan"
SOURCE = "scannetpp"
SCENE_ID = "033d0b9343"
STREAMING_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
STREAMING_AOSS_ROOT = f"aoss://baiqiao/InsScene-15K/{STREAMING_NAMESPACE}"
TARGET_AOSS_ROOT = f"aoss://baiqiao/InsScene-15K/{TARGET_NAMESPACE}"
STREAM_KEYS = 48
TARGET_KEYS = 3
DETERMINISTIC_NOISE_CONTRACT = {
    "contract": "inscene_deterministic_noise_v1",
    "wan_cogvideox_seed_scheme": "inscene_noise_seed_v1",
    "seed_payload": "canonical_json_utf8_sorted_compact({noise_seed_scheme,identity})",
    "seed_hash": "sha256_first_8_bytes_big_endian_mask_u63",
    "seed_excludes": ["timestep", "layer", "process_rank", "sample_order"],
    "streaming_identity": [
        "canonical_protocol",
        "canonical_protocol_sha256",
        "vfm",
        "model_id",
        "scene_id",
        "sample_kind=streaming_prefix",
        "window_id",
        "prefix_len",
    ],
    "target_identity": [
        "canonical_protocol",
        "canonical_protocol_sha256",
        "vfm",
        "model_id",
        "scene_id",
        "sample_kind=target_isolated",
        "frame_index",
    ],
    "generator_scope": "fresh_torch_generator_per_logical_sample_on_model_device",
    "wan_cogvideox_require_explicit_generator": True,
    "minimax_h3_seed_scheme": "minimax_h3_fixed_seed_v1",
    "minimax_h3_internal_seed": 0,
    "minimax_h3_seed_scope": "reset_per_forward",
}
SOURCE_PATHS = {
    "extractor": "features/run_inscene15k.py",
    "temporal_sampling": "vidfm3d/utils/temporal_sampling.py",
    "noise_seed_helper": "features/noise_seed.py",
    "wan_feature": "features/wan/wan_feature.py",
    "wan_i2v_feature": "features/wan/wan_feature_i2v.py",
    "wan_wrapper": "features/wan/extract_features.py",
    "cogvideox_feature": "features/cogvideox/cogvideox_feature.py",
    "cogvideox_i2v_feature": "features/cogvideox/cogvideox_feature_i2v.py",
    "cogvideox_wrapper": "features/cogvideox/extract_features.py",
    "minimax_h3_feature": "features/minimax_h3/minimax_h3_feature.py",
    "minimax_h3_wrapper": "features/minimax_h3/extract_features.py",
    "extraction_runner": "scripts/run_streaming_feat_extract_8gpu.sh",
    "sync_loop": "scripts/run_tstep_sync_loop.sh",
    "auditor": "scripts/audit_tstep_canonical_alignment.py",
    "canary_runner": "scripts/run_tstep_canonical_v2_canary.sh",
    "receipt_builder": "scripts/build_tstep_canonical_v2_canary_receipt.py",
    "receipt_verifier": "scripts/verify_tstep_canonical_v2_canary_receipt.py",
}
ADS_SUMMARY_RE = re.compile(r"\bads_rc=(\d+)\b.*\bfailed\s*[:=]\s*(\d+)\b", re.I)
SUPERVISOR_RESULT_RE = re.compile(r"\bextract_fail=(\d+)\s+sync_fail=(\d+)\b")
LOCAL_LEFT_RE = re.compile(r"\blocal_matching_sft_left=(\d+)\b")


class ReceiptError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_json(path: Path, value: dict[str, Any], mode: int = 0o644) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def require_regular(path: Path, label: str, *, mode: int | None = None) -> Path:
    raw_path = path if path.is_absolute() else (Path.cwd() / path)
    if raw_path.is_symlink():
        raise ReceiptError(f"{label} must not be a symlink: {raw_path}")
    try:
        path = raw_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ReceiptError(f"{label} is missing or unresolved: {raw_path}") from exc
    if not path.is_file():
        raise ReceiptError(f"{label} must be a regular file: {path}")
    actual_mode = stat.S_IMODE(path.stat().st_mode)
    if mode is not None and actual_mode != mode:
        raise ReceiptError(f"{label} must have mode {mode:04o}, got {actual_mode:04o}: {path}")
    if not os.access(path, os.R_OK):
        raise ReceiptError(f"{label} is not readable: {path}")
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    require_regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReceiptError(f"cannot parse {label} {path}: {exc!r}") from exc
    if not isinstance(value, dict):
        raise ReceiptError(f"{label} must contain a JSON object: {path}")
    return value


def project_path_record(path: Path, project_root: Path) -> dict[str, str]:
    path = path.resolve()
    try:
        recorded = path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        recorded = str(path)
    return {"path": recorded, "sha256": sha256_file(path)}


def resolve_project_record(record: dict[str, Any], project_root: Path, label: str) -> Path:
    if set(record) != {"path", "sha256"}:
        raise ReceiptError(f"{label} must contain exactly path and sha256")
    raw = Path(str(record["path"]))
    return raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()


def reject_invalid_markers(directory: Path) -> None:
    if not directory.is_dir():
        raise ReceiptError(f"canary receipt directory does not exist: {directory}")
    markers = sorted(path.name for path in directory.iterdir() if path.name.startswith("INVALID"))
    if markers:
        raise ReceiptError(f"canary directory is invalidated by: {', '.join(markers)}")


def validate_protocol_spec(path: Path) -> None:
    value = load_json(path, "protocol spec")
    expected = (
        value.get("canonical_protocol") == PROTOCOL_ID
        and value.get("canonical_protocol_sha256") == PROTOCOL_SHA256
        and value.get("canonical_window", {}).get("observations") == OBS_LEN
        and value.get("streaming", {}).get("prefix_lengths") == list(PREFIXES)
        and value.get("target", {}).get("future_horizons") == list(HORIZONS)
        and value.get("target", {}).get("include_anchors") is True
        and value.get("feature_namespaces", {}).get("streaming") == STREAMING_NAMESPACE
        and value.get("feature_namespaces", {}).get("target") == TARGET_NAMESPACE
        and value.get("feature_generation") == DETERMINISTIC_NOISE_CONTRACT
    )
    if not expected:
        raise ReceiptError(f"protocol spec does not implement {PROTOCOL_ID}: {path}")


def collect_input_hashes(
    project_root: Path, protocol_spec: Path, manifest_dir: Path
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, str]]]]:
    project_root = project_root.resolve()
    protocol_spec = require_regular(protocol_spec, "protocol spec")
    validate_protocol_spec(protocol_spec)
    source_records: dict[str, dict[str, str]] = {
        "protocol_spec": project_path_record(protocol_spec, project_root)
    }
    for name, relative in SOURCE_PATHS.items():
        path = require_regular(project_root / relative, f"source {name}")
        source_records[name] = project_path_record(path, project_root)

    manifest_records: dict[str, dict[str, dict[str, str]]] = {}
    for prefix in PREFIXES:
        manifest = require_regular(
            manifest_dir / f"inscene_c_h124_prefix{prefix}.jsonl",
            f"prefix-{prefix} manifest",
        )
        meta = require_regular(Path(f"{manifest}.meta.json"), f"prefix-{prefix} manifest meta")
        meta_value = load_json(meta, f"prefix-{prefix} manifest meta")
        manifest_hash = sha256_file(manifest)
        expected_meta = (
            meta_value.get("manifest_version") == 2
            and meta_value.get("canonical_protocol") == PROTOCOL_ID
            and meta_value.get("canonical_obs_len") == OBS_LEN
            and meta_value.get("prefix_length") == prefix
            and meta_value.get("action_horizons") == list(HORIZONS)
            and meta_value.get("streaming_feature_root") == STREAMING_AOSS_ROOT
            and meta_value.get("target_feature_root") == TARGET_AOSS_ROOT
            and meta_value.get("sha256") == manifest_hash
        )
        if not expected_meta:
            raise ReceiptError(f"prefix-{prefix} manifest metadata contract failed: {meta}")
        manifest_records[str(prefix)] = {
            "manifest": project_path_record(manifest, project_root),
            "meta": project_path_record(meta, project_root),
        }
    return source_records, manifest_records


def load_done_index(path: Path, label: str) -> list[str]:
    require_regular(path, label)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != len(set(lines)):
        raise ReceiptError(f"{label} contains duplicate keys: {path}")
    return lines


def expected_stream_keys(scene_id: str) -> set[str]:
    return {
        f"{scene_id}/window_{window:04d}/prefix_{prefix - 1:06d}/"
        f"feature_t{TIMESTEP}_layer{layer}.sft"
        for window in range(4)
        for prefix in PREFIXES
        for layer in LAYERS
    }


def expected_target_keys(scene_id: str) -> set[str]:
    return {
        f"{scene_id}/feature_t{TIMESTEP}_layer{layer}.sft" for layer in LAYERS
    }


def count_local_features(dataset_root: Path) -> dict[str, Any]:
    roots = {
        "stream": dataset_root / STREAMING_NAMESPACE / VFM / SOURCE / SCENE_ID,
        "target": dataset_root / TARGET_NAMESPACE / VFM / SOURCE / SCENE_ID,
    }
    by_mode: dict[str, int] = {}
    sidecars_by_mode: dict[str, int] = {}
    for mode, root in roots.items():
        by_mode[mode] = sum(1 for _ in root.rglob("*.sft")) if root.is_dir() else 0
        sidecars_by_mode[mode] = (
            sum(1 for _ in root.rglob("*.sft.manifest.json")) if root.is_dir() else 0
        )
    return {
        "local_sft": sum(by_mode.values()),
        "local_sft_by_mode": by_mode,
        "local_feature_sidecars": sum(sidecars_by_mode.values()),
        "local_feature_sidecars_by_mode": sidecars_by_mode,
    }


def collect_preflight(
    done_dir: Path, dataset_root: Path, project_root: Path, scene_id: str
) -> dict[str, Any]:
    stream_index = done_dir / f"{STREAMING_NAMESPACE}_{VFM}_t{TIMESTEP}.txt"
    target_index = done_dir / f"{TARGET_NAMESPACE}_{VFM}_t{TIMESTEP}.txt"
    stream_lines = load_done_index(stream_index, "streaming done-index")
    target_lines = load_done_index(target_index, "target done-index")
    local = count_local_features(dataset_root)
    if stream_lines or target_lines:
        raise ReceiptError(
            f"canary must start from empty done-indexes; got stream={len(stream_lines)} "
            f"target={len(target_lines)}"
        )
    if local["local_sft"] != 0 or local["local_feature_sidecars"] != 0:
        raise ReceiptError(
            "canary must start without local feature tensors/sidecars; "
            f"got {local}"
        )
    return {
        "scene_id": scene_id,
        "stream_keys": 0,
        "target_keys": 0,
        **local,
        "stream_done_index": project_path_record(stream_index, project_root),
        "target_done_index": project_path_record(target_index, project_root),
    }


def create_start_snapshot(
    *,
    output: Path,
    project_root: Path,
    protocol_spec: Path,
    manifest_dir: Path,
    done_dir: Path,
    dataset_root: Path,
    scene_id: str,
) -> dict[str, Any]:
    if scene_id != SCENE_ID:
        raise ReceiptError(f"canonical canary scene must be {SCENE_ID}, got {scene_id}")
    reject_invalid_markers(output.resolve().parent)
    source_hashes, manifest_hashes = collect_input_hashes(
        project_root, protocol_spec, manifest_dir
    )
    preflight = collect_preflight(done_dir, dataset_root, project_root, scene_id)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_type": SNAPSHOT_TYPE,
        "created_at_utc": utc_now(),
        "scene_id": scene_id,
        "protocol_id": PROTOCOL_ID,
        "source_sha256": source_hashes,
        "manifest_sha256": manifest_hashes,
        "preflight": preflight,
    }
    write_canonical_json(output, snapshot)
    return snapshot


def _manifest_stats(manifest: Path, scene_id: str) -> tuple[int, int, int]:
    total = 0
    selected = 0
    scenes: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            total += 1
            scene = str(row.get("scene_id") or Path(str(row["scene_dir"])).name)
            scenes.add(scene)
            if scene == scene_id:
                selected += 1
    return total, len(scenes), selected


def validate_audits(
    *,
    receipt_dir: Path,
    project_root: Path,
    manifest_dir: Path,
    scene_id: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    audit_dir = receipt_dir / "audit"
    if not audit_dir.is_dir() or stat.S_IMODE(audit_dir.stat().st_mode) & 0o111 == 0:
        raise ReceiptError(f"audit directory is not traversable: {audit_dir}")
    for prefix in PREFIXES:
        path = audit_dir / f"wan_scene_{scene_id}_p{prefix}.json"
        require_regular(path, f"prefix-{prefix} audit", mode=0o644)
        report = load_json(path, f"prefix-{prefix} audit")
        manifest = (manifest_dir / f"inscene_c_h124_prefix{prefix}.jsonl").resolve()
        manifest_hash = sha256_file(manifest)
        total_samples, total_scenes, selected_samples = _manifest_stats(manifest, scene_id)
        expected_fields = {
            "ok": True,
            "pass": True,
            "status": "pass",
            "problem_count": 0,
            "scope": "canary",
            "protocol_id": PROTOCOL_ID,
            "protocol": PROTOCOL_ID,
            "canonical_obs_len": OBS_LEN,
            "canonical_protocol_sha256": PROTOCOL_SHA256,
            "streaming_namespace": STREAMING_NAMESPACE,
            "target_namespace": TARGET_NAMESPACE,
            "feature_generation_contract": DETERMINISTIC_NOISE_CONTRACT["contract"],
            "vfm": VFM,
            "source": SOURCE,
            "manifest": str(manifest),
            "manifest_sha256": manifest_hash,
            "manifest_streaming_feature_root": STREAMING_AOSS_ROOT,
            "manifest_target_feature_root": TARGET_AOSS_ROOT,
            "manifest_sample_count": total_samples,
            "manifest_scene_count": total_scenes,
            "selected_scenes": [scene_id],
            "samples_checked": selected_samples,
            "scenes_checked": 1,
            "target_memberships_checked": selected_samples * (1 + len(HORIZONS)),
        }
        mismatches = {
            key: {"expected": expected, "actual": report.get(key)}
            for key, expected in expected_fields.items()
            if report.get(key) != expected
        }
        if report.get("prefixes") != list(PREFIXES) or report.get("horizons") != list(HORIZONS):
            mismatches["prefix_horizon_contract"] = {
                "expected": [list(PREFIXES), list(HORIZONS)],
                "actual": [report.get("prefixes"), report.get("horizons")],
            }
        if mismatches:
            raise ReceiptError(
                f"prefix-{prefix} audit semantic contract failed: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        result[str(prefix)] = {
            "path": f"audit/{path.name}",
            "sha256": sha256_file(path),
            "manifest_sha256": manifest_hash,
            "samples_checked": selected_samples,
        }
    return result


def parse_sync_log(path: Path, label: str) -> dict[str, Any]:
    require_regular(path, label)
    text = path.read_text(encoding="utf-8", errors="replace")
    if "sync failed" in text.lower():
        raise ReceiptError(f"{label} contains 'sync failed': {path}")
    summaries: list[dict[str, int]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if "ads_rc=" not in line:
            continue
        match = ADS_SUMMARY_RE.search(line)
        if match is None:
            raise ReceiptError(f"{label}:{lineno}: malformed ads summary")
        rc, failed = map(int, match.groups())
        if rc != 0 or failed != 0:
            raise ReceiptError(f"{label}:{lineno}: ads_rc={rc} failed={failed}")
        summaries.append({"ads_rc": rc, "failed": failed})
    if not summaries:
        raise ReceiptError(f"{label} contains no successful ads summary")
    if "sync loop complete; no matching local pairs or temps remain" not in text:
        raise ReceiptError(f"{label} lacks successful terminal marker")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "ads_summary_count": len(summaries),
        "all_ads_rc_zero": True,
        "all_ads_failed_zero": True,
        "terminal_empty": True,
    }


def parse_supervisor_log(path: Path, label: str) -> dict[str, Any]:
    require_regular(path, label)
    text = path.read_text(encoding="utf-8", errors="replace")
    results = [tuple(map(int, match.groups())) for match in SUPERVISOR_RESULT_RE.finditer(text)]
    local_left = [int(match.group(1)) for match in LOCAL_LEFT_RE.finditer(text)]
    if not results or any(extract != 0 or sync != 0 for extract, sync in results):
        raise ReceiptError(f"{label} lacks an all-zero extract/sync result: {path}")
    if not local_left or any(value != 0 for value in local_left):
        raise ReceiptError(f"{label} lacks local_matching_sft_left=0: {path}")
    if not any(line.strip() == "OK" for line in text.splitlines()):
        raise ReceiptError(f"{label} lacks terminal OK: {path}")
    h800_markers = [
        line
        for line in text.splitlines()
        if line.startswith("H800 gate PASS: 1 visible CUDA device(s):")
    ]
    if len(h800_markers) != 1 or "H800" not in h800_markers[0]:
        raise ReceiptError(f"{label} lacks the exact one-GPU H800 runtime gate: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "extract_fail": 0,
        "sync_fail": 0,
        "local_matching_sft_left": 0,
        "terminal_ok": True,
        "h800_runtime_gate": True,
    }


def _relativize_log_record(record: dict[str, Any], receipt_dir: Path) -> dict[str, Any]:
    value = dict(record)
    value["path"] = Path(str(value["path"])).resolve().relative_to(receipt_dir.resolve()).as_posix()
    return value


def collect_completion(
    *,
    receipt_dir: Path,
    project_root: Path,
    done_dir: Path,
    dataset_root: Path,
    scene_id: str,
) -> dict[str, Any]:
    stream_index = done_dir / f"{STREAMING_NAMESPACE}_{VFM}_t{TIMESTEP}.txt"
    target_index = done_dir / f"{TARGET_NAMESPACE}_{VFM}_t{TIMESTEP}.txt"
    stream_lines = load_done_index(stream_index, "streaming done-index")
    target_lines = load_done_index(target_index, "target done-index")
    expected_stream = expected_stream_keys(scene_id)
    expected_target = expected_target_keys(scene_id)
    if set(stream_lines) != expected_stream:
        raise ReceiptError(
            f"streaming done-index is not the exact canary set: "
            f"count={len(stream_lines)} missing={len(expected_stream - set(stream_lines))} "
            f"extra={len(set(stream_lines) - expected_stream)}"
        )
    if set(target_lines) != expected_target:
        raise ReceiptError(
            f"target done-index is not the exact canary set: "
            f"count={len(target_lines)} missing={len(expected_target - set(target_lines))} "
            f"extra={len(set(target_lines) - expected_target)}"
        )
    local = count_local_features(dataset_root)
    if local["local_sft"] != 0 or local["local_feature_sidecars"] != 0:
        raise ReceiptError(f"local feature cleanup is incomplete: {local}")
    sync = {
        mode: _relativize_log_record(
            parse_sync_log(receipt_dir / mode / "sync_loop.log", f"{mode} sync log"),
            receipt_dir,
        )
        for mode in ("stream", "target")
    }
    supervisor = {
        mode: _relativize_log_record(
            parse_supervisor_log(
                receipt_dir / mode / "supervisor.log", f"{mode} supervisor log"
            ),
            receipt_dir,
        )
        for mode in ("stream", "target")
    }
    return {
        "stream_keys": len(stream_lines),
        "target_keys": len(target_lines),
        **local,
        "stream_done_index": project_path_record(stream_index, project_root),
        "target_done_index": project_path_record(target_index, project_root),
        "sync": sync,
        "supervisor": supervisor,
    }


def _validate_hash_records(
    records: dict[str, dict[str, str]], expected: dict[str, dict[str, str]], label: str
) -> None:
    if records != expected:
        raise ReceiptError(f"{label} hashes changed or paths differ")


def build_receipt(
    *,
    receipt: Path,
    start_snapshot: Path,
    project_root: Path,
    protocol_spec: Path,
    manifest_dir: Path,
    done_dir: Path,
    dataset_root: Path,
    scene_id: str,
) -> dict[str, Any]:
    receipt_dir = receipt.resolve().parent
    reject_invalid_markers(receipt_dir)
    snapshot = load_json(start_snapshot, "canary start snapshot")
    if snapshot.get("snapshot_type") != SNAPSHOT_TYPE or snapshot.get("scene_id") != scene_id:
        raise ReceiptError("start snapshot identity mismatch")
    current_sources, current_manifests = collect_input_hashes(
        project_root, protocol_spec, manifest_dir
    )
    _validate_hash_records(snapshot.get("source_sha256", {}), current_sources, "source")
    if snapshot.get("manifest_sha256") != current_manifests:
        raise ReceiptError("manifest/meta hashes changed between canary start and finish")
    preflight = snapshot.get("preflight", {})
    if (
        preflight.get("stream_keys") != 0
        or preflight.get("target_keys") != 0
        or preflight.get("local_sft") != 0
        or preflight.get("local_feature_sidecars") != 0
    ):
        raise ReceiptError("start snapshot did not record an empty canary state")

    audits = validate_audits(
        receipt_dir=receipt_dir,
        project_root=project_root,
        manifest_dir=manifest_dir,
        scene_id=scene_id,
    )
    completion = collect_completion(
        receipt_dir=receipt_dir,
        project_root=project_root,
        done_dir=done_dir,
        dataset_root=dataset_root,
        scene_id=scene_id,
    )
    snapshot_record = {
        "path": start_snapshot.resolve().relative_to(receipt_dir).as_posix(),
        "sha256": sha256_file(start_snapshot),
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "status": "PASS",
        "created_at_utc": utc_now(),
        "scene_id": scene_id,
        "protocol_id": PROTOCOL_ID,
        "obs_len": OBS_LEN,
        "prefixes": list(PREFIXES),
        "horizons": list(HORIZONS),
        "modes": ["streaming_prefix", "target_isolated"],
        "layers": list(LAYERS),
        "vfm": VFM,
        "source": SOURCE,
        "timestep": TIMESTEP,
        "namespaces": {
            "streaming": STREAMING_NAMESPACE,
            "target": TARGET_NAMESPACE,
        },
        "source_sha256": current_sources,
        "manifest_sha256": current_manifests,
        "audit_artifacts": audits,
        "hash_stability": {
            "checked_at_start_and_finish": True,
            "unchanged": True,
            "start_snapshot": snapshot_record,
            "checked_source_count": len(current_sources),
            "checked_manifest_file_count": len(PREFIXES) * 2,
        },
        "completion": completion,
    }
    write_canonical_json(receipt, value)
    return value


def _resolve_receipt_relative(receipt_dir: Path, raw: str, label: str) -> Path:
    candidate = receipt_dir / raw
    if candidate.is_symlink():
        raise ReceiptError(f"{label} must not be a symlink: {candidate}")
    path = candidate.resolve()
    try:
        path.relative_to(receipt_dir.resolve())
    except ValueError as exc:
        raise ReceiptError(f"{label} escapes receipt directory: {raw}") from exc
    return candidate


def verify_receipt(
    *,
    receipt: Path,
    project_root: Path,
    protocol_spec: Path,
    manifest_dir: Path,
    done_dir: Path,
    dataset_root: Path,
    expected_scene: str,
) -> dict[str, Any]:
    receipt = require_regular(receipt, "canary receipt", mode=0o644)
    receipt_dir = receipt.parent
    reject_invalid_markers(receipt_dir)
    raw = receipt.read_bytes()
    try:
        value = json.loads(raw)
    except Exception as exc:
        raise ReceiptError(f"cannot parse canary receipt: {exc!r}") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ReceiptError("canary receipt is not canonical sorted compact JSON")
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "status": "PASS",
        "scene_id": expected_scene,
        "protocol_id": PROTOCOL_ID,
        "obs_len": OBS_LEN,
        "prefixes": list(PREFIXES),
        "horizons": list(HORIZONS),
        "modes": ["streaming_prefix", "target_isolated"],
        "layers": list(LAYERS),
        "vfm": VFM,
        "source": SOURCE,
        "timestep": TIMESTEP,
        "namespaces": {"streaming": STREAMING_NAMESPACE, "target": TARGET_NAMESPACE},
    }
    mismatches = {
        key: {"expected": expected, "actual": value.get(key)}
        for key, expected in expected_identity.items()
        if value.get(key) != expected
    }
    if mismatches:
        raise ReceiptError(f"receipt identity mismatch: {json.dumps(mismatches, sort_keys=True)}")

    current_sources, current_manifests = collect_input_hashes(
        project_root, protocol_spec, manifest_dir
    )
    if value.get("source_sha256") != current_sources:
        raise ReceiptError("receipt source/protocol hashes are stale")
    if value.get("manifest_sha256") != current_manifests:
        raise ReceiptError("receipt manifest/meta hashes are stale")

    stability = value.get("hash_stability", {})
    if (
        stability.get("checked_at_start_and_finish") is not True
        or stability.get("unchanged") is not True
        or stability.get("checked_source_count") != len(current_sources)
        or stability.get("checked_manifest_file_count") != len(PREFIXES) * 2
    ):
        raise ReceiptError("receipt hash-stability contract failed")
    snapshot_record = stability.get("start_snapshot", {})
    snapshot_path = _resolve_receipt_relative(
        receipt_dir, str(snapshot_record.get("path", "")), "start snapshot"
    )
    require_regular(snapshot_path, "canary start snapshot", mode=0o644)
    if snapshot_record.get("sha256") != sha256_file(snapshot_path):
        raise ReceiptError("start snapshot hash mismatch")
    snapshot = load_json(snapshot_path, "canary start snapshot")
    if (
        snapshot.get("snapshot_type") != SNAPSHOT_TYPE
        or snapshot.get("scene_id") != expected_scene
        or snapshot.get("source_sha256") != current_sources
        or snapshot.get("manifest_sha256") != current_manifests
        or snapshot.get("preflight", {}).get("stream_keys") != 0
        or snapshot.get("preflight", {}).get("target_keys") != 0
        or snapshot.get("preflight", {}).get("local_sft") != 0
        or snapshot.get("preflight", {}).get("local_feature_sidecars") != 0
    ):
        raise ReceiptError("start snapshot content contract failed")

    audits = validate_audits(
        receipt_dir=receipt_dir,
        project_root=project_root,
        manifest_dir=manifest_dir,
        scene_id=expected_scene,
    )
    if value.get("audit_artifacts") != audits:
        raise ReceiptError("receipt audit hashes/summaries are stale")
    completion = collect_completion(
        receipt_dir=receipt_dir,
        project_root=project_root,
        done_dir=done_dir,
        dataset_root=dataset_root,
        scene_id=expected_scene,
    )
    if value.get("completion") != completion:
        raise ReceiptError("receipt completion evidence is stale")
    return {
        "receipt": str(receipt),
        "receipt_sha256": sha256_file(receipt),
        "scene_id": expected_scene,
        "protocol_id": PROTOCOL_ID,
        "stream_keys": completion["stream_keys"],
        "target_keys": completion["target_keys"],
        "local_sft": completion["local_sft"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--protocol-spec", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--expected-scene", default=SCENE_ID)
    parser.add_argument("--done-dir", default="")
    parser.add_argument(
        "--dataset-root", default="/mnt/umm/users/yinbaiqiao/InsScene-15K"
    )
    parser.add_argument("--print-bindings-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    done_dir = (
        Path(args.done_dir).resolve()
        if args.done_dir
        else Path(args.receipt).resolve().parent.parent / "aoss_done"
    )
    try:
        bindings = verify_receipt(
            receipt=Path(args.receipt),
            project_root=project_root,
            protocol_spec=Path(args.protocol_spec),
            manifest_dir=Path(args.manifest_dir),
            done_dir=done_dir,
            dataset_root=Path(args.dataset_root),
            expected_scene=args.expected_scene,
        )
    except ReceiptError as exc:
        print(f"CANARY_RECEIPT_INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if args.print_bindings_json:
        print(json.dumps(bindings, sort_keys=True, separators=(",", ":")))
    else:
        print(
            f"CANARY_RECEIPT_VERIFIED receipt={bindings['receipt']} "
            f"sha256={bindings['receipt_sha256']} scene={bindings['scene_id']}"
        )


if __name__ == "__main__":
    main()
