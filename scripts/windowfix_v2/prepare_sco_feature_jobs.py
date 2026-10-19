#!/usr/bin/env python3
"""Render (but never submit) canonical-v2 feature extraction SCO jobs.

The input is the deduplicated 1096-table feature inventory, optionally enriched
by a cache audit.  This program is deliberately a renderer only: it writes
immutable shard JSON, SCO argv JSON, and shell-quoted commands for review.  It
does not import or invoke ``sco``.

One shard covers one (model, model-id, timestep) cache root and all required
layers.  All four canonical prefixes are always passed to the extractor even
when only one prefix needs new tensors.  That is required because
``prefix_index.npy`` is a scene-wide structural index and must never be
rewritten as a prefix subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
EXPECTED_PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"
EXPECTED_PREFIXES = (8, 12, 16, 24)
EXPECTED_HORIZONS = (1, 2, 4)
EXTRACT_ACTIONS = {
    "REEXTRACT",
    "REEXTRACT_MISMATCH",
    "REEXTRACT_MISSING",
    "REEXTRACT_CORRUPT",
    "MISSING",
    "MISMATCH",
    "CORRUPT",
}
REUSE_ACTIONS = {"REUSE_EXACT", "EXACT_REUSE", "READY_CANONICAL_V2"}
PENDING_ACTIONS = {"", "PENDING_CACHE_AUDIT", "PENDING", "REQUIRED"}
MAIN_EXTRACTOR_VFMS = {
    "wan",
    "cogvideox",
    "vjepa",
    "vjepa2",
    "dino",
    "videomae",
    "aether",
    "opensora",
    "f3r",
    "minimax_h3",
    "lingbot",
}
MLLM_VFMS = {"qwen2_5_vl", "bagel", "bagel_hf", "generic_hf"}
CERTIFIED_READY_VFMS = {
    "wan",
    "cogvideox",
    "minimax_h3",
    "dino",
    "vjepa2",
    "videomae",
}
CANARY_GATED_VFMS = {"aether", "f3r", "lingbot", "qwen2_5_vl"}
BACKEND_READINESS_BLOCKERS = {
    "vjepa": (
        "BLOCKED_CACHE_VALIDATOR_VJEPA: deterministic forward is not yet represented "
        "by explicit no-noise provenance, so seeded cache validation cannot certify resume"
    ),
    "aether": (
        "BLOCKED_GPU_CANARY_AETHER: sample-local Generator/provenance is implemented, "
        "but cross-process/order GPU equality is not yet certified"
    ),
    "f3r": (
        "BLOCKED_GPU_CANARY_F3R: random image-index generation is sample-local, "
        "but cross-rank/order GPU equality is not yet certified"
    ),
    "lingbot": (
        "BLOCKED_GPU_CANARY_LINGBOT: VAE and diffusion draws use a sample-local "
        "Generator, but GPU byte/tensor equality is not yet certified"
    ),
    "qwen2_5_vl": (
        "BLOCKED_GPU_CANARY_QWEN: canonical obs28 streaming/target extraction and "
        "strict no-noise provenance are implemented, but cross-process/order GPU "
        "equality is not yet certified"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: Any, label: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer (got {value!r})") from exc


def split_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in re.split(r"[;,]", str(value)) if item.strip()]


def slug(value: str, *, limit: int = 72) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    if not clean:
        clean = "unnamed"
    if len(clean) <= limit:
        return clean.lower()
    suffix = hashlib.sha256(value.encode()).hexdigest()[:10]
    return f"{clean[: limit - 11].rstrip('-._')}-{suffix}".lower()


def read_inventory(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"inventory must be a regular non-symlink file: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("specs", payload.get("rows"))
        if not isinstance(payload, list):
            raise ValueError("JSON inventory must be a list or contain specs/rows")
        rows = payload
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("inventory contains no specification rows")
    return [{str(k): v for k, v in row.items()} for row in rows]


def validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"protocol must be a regular non-symlink file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("canonical_protocol") != EXPECTED_PROTOCOL:
        failures.append("canonical_protocol")
    if payload.get("canonical_window", {}).get("observations") != 28:
        failures.append("canonical_window.observations")
    if tuple(payload.get("streaming", {}).get("prefix_lengths", [])) != EXPECTED_PREFIXES:
        failures.append("streaming.prefix_lengths")
    target = payload.get("target", {})
    if tuple(target.get("prefix_lengths", [])) != EXPECTED_PREFIXES:
        failures.append("target.prefix_lengths")
    if tuple(target.get("future_horizons", [])) != EXPECTED_HORIZONS:
        failures.append("target.future_horizons")
    if target.get("include_anchors") is not True:
        failures.append("target.include_anchors")
    namespaces = payload.get("feature_namespaces", {})
    if not str(namespaces.get("streaming", "")).startswith("FEAT_STREAMING_CANONICAL_V2"):
        failures.append("feature_namespaces.streaming")
    if not str(namespaces.get("target", "")).startswith("FEAT_TARGET_CANONICAL_V2"):
        failures.append("feature_namespaces.target")
    if payload.get("acceptance", {}).get("legacy_features_valid") is not False:
        failures.append("acceptance.legacy_features_valid")
    if failures:
        raise ValueError("protocol is not canonical-v2: " + ", ".join(failures))
    return payload, sha256_file(path)


def extractor_protocol_sha(protocol: dict[str, Any]) -> str:
    """Reproduce ``features.run_inscene15k.canonical_protocol_metadata`` SHA."""
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


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_compact_binding(
    *,
    canonical_source_arg: Path | None,
    namespace_lock_path: Path,
    label_audit_path: Path,
    protocol: dict[str, Any],
    protocol_file_sha: str,
) -> dict[str, Any]:
    """Bind one frozen obs28 source instead of four 55 MiB manifest copies."""
    for path, label in (
        (namespace_lock_path, "namespace lock"),
        (label_audit_path, "label-lineage audit"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    lock_sha = sha256_file(namespace_lock_path)
    audit_sha = sha256_file(label_audit_path)
    lock = json.loads(namespace_lock_path.read_text(encoding="utf-8"))
    audit = json.loads(label_audit_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or not isinstance(audit, dict):
        raise ValueError("namespace lock and label audit must contain JSON objects")
    if lock.get("schema_version") != 2:
        raise ValueError("compact namespace lock must use schema_version=2")
    if lock.get("canonical_protocol") != protocol["canonical_protocol"]:
        raise ValueError("namespace lock canonical protocol mismatch")
    if lock.get("protocol_file_sha256") != protocol_file_sha:
        raise ValueError("namespace lock protocol file SHA mismatch")
    policy = lock.get("override_policy", {})
    if policy.get("fail_closed") is not True or policy.get("row_roots_are_not_authoritative") is not True:
        raise ValueError("namespace lock is not fail-closed against row-level legacy roots")
    expected_roots = {
        kind: f"aoss://baiqiao/InsScene-15K/{protocol['feature_namespaces'][kind]}"
        for kind in ("streaming", "target")
    }
    if lock.get("feature_roots") != expected_roots:
        raise ValueError("namespace lock feature roots do not match the protocol namespaces")

    source_binding = lock.get("canonical_source")
    if not isinstance(source_binding, dict):
        raise ValueError("namespace lock has no canonical_source object")
    locked_source_path = Path(str(source_binding.get("path", ""))).resolve()
    if canonical_source_arg is not None:
        source = canonical_source_arg.resolve()
    else:
        source = locked_source_path
    if source != locked_source_path:
        raise ValueError("--canonical-source path differs from namespace lock canonical_source.path")
    if not source.is_file() or source.is_symlink():
        raise ValueError(
            "frozen canonical obs28 source is missing/non-regular; provide --canonical-source: "
            f"{source}"
        )
    expected_source_sha = str(source_binding.get("sha256", ""))
    if not _valid_sha256(expected_source_sha):
        raise ValueError("namespace lock has no valid canonical obs28 source SHA")
    actual_source_sha = sha256_file(source)
    if actual_source_sha != expected_source_sha:
        raise ValueError("canonical obs28 source SHA mismatch")
    expected_windows = int(protocol.get("expected_universe", {}).get("canonical_windows", 0))
    expected_scenes = int(protocol.get("expected_universe", {}).get("eligible_scenes", 0))
    expected_source_fields = {
        "rows": expected_windows,
        "scenes": expected_scenes,
        "canonical_obs_len": 28,
        "source_prefix": 24,
    }
    for field, expected in expected_source_fields.items():
        if source_binding.get(field) != expected:
            raise ValueError(f"namespace lock canonical_source.{field} mismatch")

    source_sidecar = Path(f"{source}.meta.json")
    if not source_sidecar.is_file() or source_sidecar.is_symlink():
        raise ValueError(f"canonical source sidecar is missing/non-regular: {source_sidecar}")
    sidecar_sha = sha256_file(source_sidecar)
    if sidecar_sha != source_binding.get("sidecar_sha256"):
        raise ValueError("namespace lock canonical source sidecar SHA mismatch")
    if Path(str(source_binding.get("sidecar_path", ""))).resolve() != source_sidecar.resolve():
        raise ValueError("namespace lock canonical source sidecar path mismatch")
    sidecar = json.loads(source_sidecar.read_text(encoding="utf-8"))
    if (
        sidecar.get("sha256") != actual_source_sha
        or sidecar.get("prefix_lengths") != [24]
        or [int(value) for value in sidecar.get("action_horizons", [])] != list(EXPECTED_HORIZONS)
        or sidecar.get("num_samples") != expected_windows
        or sidecar.get("num_included_scenes") != expected_scenes
    ):
        raise ValueError("canonical source sidecar is not the frozen p24/h124 universe")

    universe_digest = hashlib.sha256()
    label_lineage_universe_digest = hashlib.sha256()
    mapping_digest = hashlib.sha256()
    per_prefix_digest = {prefix: hashlib.sha256() for prefix in EXPECTED_PREFIXES}
    units: set[tuple[str, int]] = set()
    scene_ids: set[str] = set()
    selected_by_scene: dict[str, set[int]] = defaultdict(set)
    previous_unit: tuple[str, int] | None = None
    row_count = 0
    with source.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                scene = str(row.get("scene_id") or Path(str(row["scene_dir"])).name)
                window_id = int(row["streaming_window_id"])
                split = str(row["split"])
                window = [int(value) for value in row["streaming_window_indices"]]
                valid_frames = [str(value) for value in row["valid_frames"]]
                valid_pose_indices = [int(value) for value in row["valid_pose_indices"]]
            except Exception as exc:
                raise ValueError(f"canonical source line {line_number} is invalid: {exc}") from exc
            if len(window) != 28 or any(right <= left for left, right in zip(window, window[1:])):
                raise ValueError(f"canonical source line {line_number} is not a strict obs28 window")
            if len(valid_frames) != len(valid_pose_indices) or any(
                value < 0 or value >= len(valid_frames) for value in window
            ):
                raise ValueError(f"canonical source line {line_number} has an invalid raw-label map")
            if row.get("source") != protocol.get("source"):
                raise ValueError(f"canonical source line {line_number} dataset-source mismatch")
            if row.get("streaming_sampling") != protocol["canonical_window"]["sampling"]:
                raise ValueError(f"canonical source line {line_number} sampling mismatch")
            unit = (scene, window_id)
            if unit in units:
                raise ValueError(f"duplicate canonical source unit at line {line_number}: {unit}")
            if previous_unit is not None and unit <= previous_unit:
                raise ValueError(f"canonical source is not strictly scene/window ordered at line {line_number}")
            units.add(unit)
            scene_ids.add(scene)
            selected_by_scene[scene].update(window)
            previous_unit = unit
            source_prefix = row.get("streaming_indices")
            if source_prefix is None:
                raise ValueError(f"canonical source line {line_number} is not a frozen p24 row")
            source_prefix_values = [int(value) for value in source_prefix]
            if len(source_prefix_values) != 24 or source_prefix_values != window[:24]:
                raise ValueError(f"canonical source line {line_number} is not p24=window[:24]")
            if row.get("streaming_tail") != 23 or row.get("streaming_tail_raw") != window[23]:
                raise ValueError(f"canonical source line {line_number} has an invalid p24 tail")
            row_protocol = row.get("canonical_protocol")
            if row_protocol is not None and row_protocol != protocol["canonical_protocol"]:
                raise ValueError(f"canonical source line {line_number} protocol mismatch")
            canonical_payload = {
                "scene_id": scene,
                "split": split,
                "streaming_window_id": window_id,
                "streaming_window_indices": window,
            }
            universe_digest.update(
                (
                    json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode()
            )
            label_lineage_payload = {
                "scene": scene,
                "split": split,
                "window_id": window_id,
                "window": window,
            }
            label_lineage_universe_digest.update(
                (
                    json.dumps(label_lineage_payload, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode()
            )
            for prefix in EXPECTED_PREFIXES:
                mapping = {
                    "scene": scene,
                    "window_id": window_id,
                    "prefix": prefix,
                    "input": window[:prefix],
                    "anchor": window[prefix - 1],
                    "future": {
                        str(horizon): window[prefix - 1 + horizon]
                        for horizon in EXPECTED_HORIZONS
                    },
                }
                encoded = (
                    json.dumps(mapping, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode()
                mapping_digest.update(encoded)
                per_prefix_digest[prefix].update(encoded)
            row_count += 1
    if row_count != expected_windows or len(scene_ids) != expected_scenes:
        raise ValueError(
            f"canonical source universe mismatch: rows={row_count}/{expected_windows} "
            f"scenes={len(scene_ids)}/{expected_scenes}"
        )
    universe_sha = universe_digest.hexdigest()
    label_lineage_universe_sha = label_lineage_universe_digest.hexdigest()
    if universe_sha != lock.get("canonical_universe_sha256"):
        raise ValueError("canonical source universe SHA mismatch against namespace lock")

    if audit.get("audit_schema_version") != 2:
        raise ValueError("compact label-lineage audit must use audit_schema_version=2")
    if audit.get("audit_type") != "canonical_v2_compact_source_label_lineage":
        raise ValueError("wrong label-lineage audit type")
    if audit.get("status") != "pass" or audit.get("problem_count") != 0 or audit.get("fail_closed") is not True:
        raise ValueError("label-lineage audit is not a fail-closed full PASS")
    if audit.get("protocol") != protocol["canonical_protocol"] or audit.get("protocol_file_sha256") != protocol_file_sha:
        raise ValueError("label-lineage audit protocol binding mismatch")
    namespace_binding = audit.get("namespace_binding", {})
    if (
        Path(str(namespace_binding.get("path", ""))).resolve() != namespace_lock_path.resolve()
        or namespace_binding.get("sha256") != lock_sha
        or namespace_binding.get("feature_roots") != expected_roots
    ):
        raise ValueError("label-lineage audit does not bind the current namespace lock")
    counts = audit.get("counts", {})
    selected_raw_frames = sum(len(values) for values in selected_by_scene.values())
    expected_counts = {
        "canonical_windows": expected_windows,
        "derived_prefix_rows": expected_windows * len(EXPECTED_PREFIXES),
        "input_frame_references": expected_windows * sum(EXPECTED_PREFIXES),
        "anchor_references": expected_windows * len(EXPECTED_PREFIXES),
        "future_target_references": expected_windows * len(EXPECTED_PREFIXES) * len(EXPECTED_HORIZONS),
        "raw_scenes_checked": expected_scenes,
        "raw_frame_identities_checked": selected_raw_frames,
        "image_paths_checked": selected_raw_frames,
        "mask_paths_checked": selected_raw_frames,
        "depth_paths_checked": selected_raw_frames,
        "pose_rows_checked": selected_raw_frames,
    }
    for field, expected in expected_counts.items():
        if counts.get(field) != expected:
            raise ValueError(f"label-lineage audit count mismatch: {field}")
    if audit.get("scene_count") != expected_scenes:
        raise ValueError("label-lineage audit scene count mismatch")
    mapping_sha = mapping_digest.hexdigest()
    if audit.get("canonical_universe_sha256") != universe_sha:
        raise ValueError("label-lineage audit canonical universe SHA mismatch")
    if audit.get("label_lineage_universe_sha256") != label_lineage_universe_sha:
        raise ValueError("label-lineage audit internal universe SHA mismatch")
    if audit.get("input_anchor_future_mapping_sha256") != mapping_sha:
        raise ValueError("label-lineage audit mapping SHA mismatch")
    expected_per_prefix_sha = {
        str(prefix): per_prefix_digest[prefix].hexdigest() for prefix in EXPECTED_PREFIXES
    }
    if audit.get("per_prefix_mapping_sha256") != expected_per_prefix_sha:
        raise ValueError("label-lineage audit per-prefix mapping SHA mismatch")
    if (
        Path(str(audit.get("source_manifest", ""))).resolve() != source
        or audit.get("source_manifest_sha256") != actual_source_sha
        or audit.get("source_sidecar_sha256") != sidecar_sha
        or audit.get("source_sidecar_declared_samples") != expected_windows
    ):
        raise ValueError("label-lineage audit does not bind the canonical obs28 source")
    derivation = audit.get("derivation", {})
    expected_derivation = {
        "canonical_obs_len": 28,
        "prefixes": list(EXPECTED_PREFIXES),
        "input_rule": "streaming_window_indices[:prefix]",
        "anchor_rule": "streaming_window_indices[prefix-1]",
        "future_rule": "streaming_window_indices[prefix-1+horizon]",
        "horizons": list(EXPECTED_HORIZONS),
        "row_level_legacy_feature_roots_ignored": True,
    }
    if any(derivation.get(key) != value for key, value in expected_derivation.items()):
        raise ValueError("label-lineage audit derivation contract mismatch")
    if audit.get("feature_roots") != expected_roots:
        raise ValueError("label-lineage audit feature namespace mismatch")
    if (
        sha256_file(source) != actual_source_sha
        or sha256_file(source_sidecar) != sidecar_sha
        or sha256_file(namespace_lock_path) != lock_sha
        or sha256_file(label_audit_path) != audit_sha
    ):
        raise ValueError("a compact binding input changed during validation")

    return {
        "schema_version": 2,
        "mode": "compact_obs28_source_v1",
        "canonical_source": {
            "path": str(source),
            "sha256": actual_source_sha,
            "rows": row_count,
            "scenes": len(scene_ids),
            "canonical_obs_len": 28,
            "source_prefix": 24,
            "sidecar_path": str(source_sidecar),
            "sidecar_sha256": sidecar_sha,
        },
        "namespace_lock": {
            "path": str(namespace_lock_path),
            "sha256": lock_sha,
        },
        "label_lineage_audit": {
            "path": str(label_audit_path),
            "sha256": audit_sha,
            "mapping_sha256": mapping_sha,
            "label_lineage_universe_sha256": label_lineage_universe_sha,
            "per_prefix_mapping_sha256": expected_per_prefix_sha,
            "full_raw_label_counts": {
                key: expected_counts[key]
                for key in (
                    "raw_scenes_checked",
                    "raw_frame_identities_checked",
                    "image_paths_checked",
                    "mask_paths_checked",
                    "depth_paths_checked",
                    "pose_rows_checked",
                )
            },
        },
        "canonical_universe_sha256": universe_sha,
        "derivation": {
            "streaming": "canonical_window_indices[:prefix]",
            "anchor": "canonical_window_indices[prefix-1]",
            "future": "canonical_window_indices[prefix-1+h]",
            "prefixes": list(EXPECTED_PREFIXES),
            "horizons": list(EXPECTED_HORIZONS),
        },
        "feature_roots": expected_roots,
    }


def backend_for(vfm: str) -> str:
    if vfm in {"wan", "cogvideox"}:
        return "wan_cog"
    if vfm in {"vjepa", "vjepa2", "videomae"}:
        return "videomae_vjepa"
    if vfm == "minimax_h3":
        return "minimax_h3"
    if vfm == "lingbot":
        return "lingbot"
    if vfm in MLLM_VFMS:
        return "mllm"
    if vfm in MAIN_EXTRACTOR_VFMS:
        return "generic"
    return "unsupported"


def python_for(backend: str) -> str:
    if backend == "minimax_h3":
        return "/mnt/umm/users/yinbaiqiao/.conda/envs/minimax-h3/bin/python"
    if backend == "lingbot":
        return "/mnt/umm/users/yinbaiqiao/.conda/envs/lingbot-video/bin/python"
    return "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"


def mllm_canonical_ready(path: Path) -> tuple[bool, str]:
    if not path.is_file() or path.is_symlink():
        return False, f"canonical MLLM extractor missing: {path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    required = ("--streaming-window-obs-len", "canonical_protocol", "include_anchors")
    missing = [token for token in required if token not in text]
    if missing:
        return False, "MLLM extractor lacks canonical-v2 contract: " + ",".join(missing)
    return True, ""


def validate_backend_canary_receipts(
    directory: Path | None,
    *,
    protocol_file_sha: str,
    main_extractor: Path,
    mllm_extractor: Path,
    canary_script: Path,
) -> dict[str, dict[str, Any]]:
    """Return strictly verified canary bindings; missing receipts stay blocked."""
    if directory is None:
        return {}
    directory = directory.resolve()
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"backend canary directory must be regular: {directory}")
    bindings: dict[str, dict[str, Any]] = {}
    canary_sha = sha256_file(canary_script)
    for vfm in sorted(CANARY_GATED_VFMS):
        path = directory / "receipts" / f"{vfm}.json"
        if not path.exists():
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"backend canary receipt must be regular: {path}")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        extractor = mllm_extractor if vfm == "qwen2_5_vl" else main_extractor
        extractor_sha = sha256_file(extractor)
        failures = []
        expected = {
            "schema_version": 1,
            "audit_type": "canonical_v2_backend_cross_process_order_canary",
            "status": "pass",
            "vfm": vfm,
            "protocol": EXPECTED_PROTOCOL,
            "protocol_file_sha256": protocol_file_sha,
            "extractor": str(extractor),
            "extractor_sha256": extractor_sha,
            "canary_script": str(canary_script),
            "canary_script_sha256": canary_sha,
        }
        for field, value in expected.items():
            if receipt.get(field) != value:
                failures.append(field)
        results = receipt.get("results")
        if not isinstance(results, dict) or set(results) != {"streaming_prefix", "target_isolated"}:
            failures.append("results")
        else:
            for mode in ("streaming_prefix", "target_isolated"):
                comparison = results.get(mode, {}).get("comparison", {})
                if (
                    int(comparison.get("files_compared", 0)) <= 0
                    or int(comparison.get("safetensors_compared", 0)) <= 0
                    or not _valid_sha256(comparison.get("comparison_sha256"))
                ):
                    failures.append(f"results.{mode}.comparison")
        if failures:
            raise ValueError(
                f"backend canary receipt is not certifiable ({vfm}): "
                + ",".join(failures)
            )
        bindings[vfm] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "model_id": receipt.get("model_id"),
            "layer": int(receipt["layer"]),
            "timestep": int(receipt["timestep"]),
            "protocol_file_sha256": protocol_file_sha,
            "extractor_sha256": extractor_sha,
            "canary_script_sha256": canary_sha,
        }
    return bindings


def action_of(row: dict[str, Any]) -> str:
    return str(row.get("action") or row.get("cache_action") or "").strip().upper()


def normalize_rows(
    rows: Iterable[dict[str, Any]],
    *,
    protocol_id: str,
    protocol_file_sha: str,
    feature_protocol_sha: str,
    allow_pending: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        spec_id = str(row.get("spec_id") or row.get("feature_spec_id") or "").strip()
        if not spec_id:
            raise ValueError(f"inventory row {index} has no spec_id")
        if spec_id in seen:
            raise ValueError(f"duplicate spec_id: {spec_id}")
        seen.add(spec_id)
        declared_protocol = str(row.get("canonical_protocol") or row.get("protocol") or "").strip()
        if declared_protocol != protocol_id:
            raise ValueError(f"{spec_id}: canonical_protocol mismatch: {declared_protocol!r}")
        declared_feature_sha = str(row.get("protocol_sha256") or "").strip()
        if declared_feature_sha != feature_protocol_sha:
            raise ValueError(f"{spec_id}: protocol_sha256 mismatch")
        declared_file_sha = str(row.get("protocol_file_sha256") or "").strip()
        if declared_file_sha and declared_file_sha != protocol_file_sha:
            raise ValueError(f"{spec_id}: protocol_file_sha256 mismatch")
        vfm = str(row.get("vfm") or row.get("model") or "").strip()
        model = str(row.get("model") or vfm).strip()
        if not vfm:
            raise ValueError(f"{spec_id}: missing vfm/model")
        model_id = str(row.get("model_id") or model).strip()
        action = action_of(row)
        if action in PENDING_ACTIONS and not allow_pending:
            pending.append({"spec_id": spec_id, "action": action or "PENDING_CACHE_AUDIT"})
        elif action not in EXTRACT_ACTIONS | REUSE_ACTIONS | PENDING_ACTIONS:
            raise ValueError(f"{spec_id}: unsupported cache action {action!r}")
        normalized.append(
            {
                "spec_id": spec_id,
                "vfm": vfm,
                "model": model,
                "model_variant": str(row.get("model_variant") or "default").strip(),
                "model_id": model_id,
                "feature_layer": parse_int(row.get("feature_layer", row.get("layer")), f"{spec_id}.feature_layer"),
                "timestep": parse_int(row.get("timestep", row.get("feature_timestep", 0)), f"{spec_id}.timestep"),
                "prefix": parse_int(row.get("prefix"), f"{spec_id}.prefix"),
                "requires_streaming": parse_bool(row.get("requires_streaming", True)),
                "requires_target": parse_bool(row.get("requires_target", False)),
                "action": action or "PENDING_CACHE_AUDIT",
                "target_action": str(row.get("target_action") or "").strip().upper(),
                "campaigns": split_values(row.get("campaigns")),
                "tasks": split_values(row.get("tasks")),
            }
        )
    invalid_prefix = [r["spec_id"] for r in normalized if r["prefix"] not in EXPECTED_PREFIXES]
    if invalid_prefix:
        raise ValueError(f"non-canonical prefixes in inventory: {invalid_prefix[:8]}")
    return normalized, pending


def needs_stream(row: dict[str, Any], allow_pending: bool) -> bool:
    if not row["requires_streaming"]:
        return False
    return row["action"] in EXTRACT_ACTIONS or (allow_pending and row["action"] in PENDING_ACTIONS)


def needs_target(row: dict[str, Any], allow_pending: bool, target_policy: str) -> bool:
    if not row["requires_target"]:
        return False
    action = row["target_action"]
    if action:
        return action in EXTRACT_ACTIONS or (allow_pending and action in PENDING_ACTIONS)
    if target_policy == "all-required":
        return True
    return row["action"] in EXTRACT_ACTIONS or (allow_pending and row["action"] in PENDING_ACTIONS)


def build_groups(
    rows: list[dict[str, Any]], *, allow_pending: bool, target_policy: str
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["vfm"], row["model"], row["model_id"], row["timestep"])
        group = groups.setdefault(
            key,
            {
                "vfm": row["vfm"],
                "model": row["model"],
                "model_id": row["model_id"],
                "model_variants": set(),
                "timestep": row["timestep"],
                "layers": set(),
                "stream_layers": set(),
                "target_layers": set(),
                "prefixes_in_inventory": set(),
                "spec_ids": set(),
                "stream_spec_ids": set(),
                "target_spec_ids": set(),
                "actions": defaultdict(int),
            },
        )
        group["model_variants"].add(row["model_variant"])
        group["layers"].add(row["feature_layer"])
        group["prefixes_in_inventory"].add(row["prefix"])
        group["spec_ids"].add(row["spec_id"])
        group["actions"][row["action"]] += 1
        if needs_stream(row, allow_pending):
            group["stream_spec_ids"].add(row["spec_id"])
            group["stream_layers"].add(row["feature_layer"])
        if needs_target(row, allow_pending, target_policy):
            group["target_spec_ids"].add(row["spec_id"])
            group["target_layers"].add(row["feature_layer"])
    return groups


def environment_for(backend: str, project_root: Path) -> dict[str, str]:
    common = {
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if backend == "lingbot":
        common.update(
            {
                "HF_HOME": "/mnt/umm/shared_model/huggingface",
                "DIFFUSERS_ATTN_BACKEND": "_native_flash",
                "LINGBOT_MOE_PAD_BACKEND": "vectorized",
                "LINGBOT_MOE_EXPERT_BACKEND": "grouped_mm",
                "WINDOWFIX_EXTRA_PYTHONPATH": "/mnt/umm/users/yinbaiqiao/Consistency/lingbot-video",
            }
        )
    elif backend == "minimax_h3":
        common["HF_HOME"] = "/mnt/umm/shared_model/huggingface"
    elif backend in {"wan_cog", "videomae_vjepa", "generic", "mllm"}:
        common.update(
            {
                "HF_HOME": "/mnt/umm/users/yinbaiqiao/.cache/huggingface",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    common["PROJECT_ROOT"] = str(project_root)
    return common


def render_command(argv: list[str]) -> str:
    return shlex.join(argv)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--canonical-source", type=Path)
    parser.add_argument("--namespace-lock", required=True, type=Path)
    parser.add_argument("--label-audit", required=True, type=Path)
    parser.add_argument(
        "--backend-canary-dir",
        type=Path,
        help="Optional canary artifact root; only full PASS receipts unblock gated backends.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/InsScene-15K")
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--source", default="scannetpp")
    parser.add_argument("--scene-count", type=int, default=682)
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--workspace", default="aigc")
    parser.add_argument("--aec2-name", default="h800")
    parser.add_argument("--worker-spec", default="N4lS.Iq.I80.8")
    parser.add_argument("--priority", default="HIGHEST")
    parser.add_argument(
        "--container-image", default="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
    )
    parser.add_argument(
        "--storage-mount", default="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
    )
    parser.add_argument(
        "--target-policy",
        choices=("all-required", "reextract-only"),
        default="all-required",
        help="Without target_action, prepare every required target cache or mirror streaming action.",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Conservatively render pending cache-audit rows as extraction work.",
    )
    args = parser.parse_args()

    if args.source != "scannetpp":
        raise SystemExit("canonical-v2 feature jobs are ScanNet++ only")
    if args.n_gpus < 1 or args.scene_count < 1:
        raise SystemExit("n-gpus and scene-count must be positive")

    project_root = args.project_root.resolve()
    protocol_path = args.protocol.resolve()
    specs_path = args.specs.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to replace immutable output directory: {output_dir}")

    protocol, protocol_file_sha = validate_protocol(protocol_path)
    feature_protocol_sha = extractor_protocol_sha(protocol)
    compact_binding = validate_compact_binding(
        canonical_source_arg=args.canonical_source,
        namespace_lock_path=args.namespace_lock.resolve(),
        label_audit_path=args.label_audit.resolve(),
        protocol=protocol,
        protocol_file_sha=protocol_file_sha,
    )
    rows, pending = normalize_rows(
        read_inventory(specs_path),
        protocol_id=protocol["canonical_protocol"],
        protocol_file_sha=protocol_file_sha,
        feature_protocol_sha=feature_protocol_sha,
        allow_pending=args.allow_pending,
    )
    if pending:
        preview = ", ".join(item["spec_id"] for item in pending[:5])
        raise SystemExit(
            f"{len(pending)} specs still await cache audit ({preview}); "
            "run the exact-cache classifier first or use --allow-pending for conservative extraction"
        )

    main_extractor = project_root / "features/run_inscene15k.py"
    mllm_extractor = project_root / "features/run_inscene15k_mllm.py"
    plan_renderer = Path(__file__).resolve()
    runner = project_root / "scripts/windowfix_v2/run_canonical_feature_shard.py"
    audit_script = project_root / "scripts/windowfix_v2/audit_compact_alignment.py"
    base_audit_script = project_root / "scripts/audit_tstep_canonical_alignment.py"
    backend_canary_script = project_root / "scripts/windowfix_v2/run_backend_determinism_canary.py"
    for required in (
        main_extractor,
        runner,
        audit_script,
        base_audit_script,
        backend_canary_script,
    ):
        if not required.is_file() or required.is_symlink():
            raise SystemExit(f"missing regular bound source: {required}")
    source_bindings = {
        "plan_renderer": {"path": str(plan_renderer), "sha256": sha256_file(plan_renderer)},
        "main_extractor": {"path": str(main_extractor), "sha256": sha256_file(main_extractor)},
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "compact_audit": {"path": str(audit_script), "sha256": sha256_file(audit_script)},
        "base_alignment_audit": {
            "path": str(base_audit_script),
            "sha256": sha256_file(base_audit_script),
        },
        "backend_determinism_canary": {
            "path": str(backend_canary_script),
            "sha256": sha256_file(backend_canary_script),
        },
    }
    backend_sources = {
        "noise_seed": project_root / "features/noise_seed.py",
        "wan_feature": project_root / "features/wan/wan_feature.py",
        "wan_feature_i2v": project_root / "features/wan/wan_feature_i2v.py",
        "cogvideox_feature": project_root / "features/cogvideox/cogvideox_feature.py",
        "cogvideox_feature_i2v": project_root / "features/cogvideox/cogvideox_feature_i2v.py",
        "dino_feature": project_root / "features/dino/extract_features.py",
        "vjepa2_feature": project_root / "features/vjepa2/vjepa2_feature.py",
        "minimax_h3_feature": project_root / "features/minimax_h3/minimax_h3_feature.py",
        "aether_feature": project_root / "features/aether/aether_feature.py",
        "f3r_feature": project_root / "features/f3r/extract_features.py",
        "f3r_inference": project_root / "features/f3r/f3r/dust3r/inference_multiview.py",
        "f3r_model": project_root / "features/f3r/f3r/models/fast3r.py",
        "lingbot_feature": project_root / "features/lingbot/lingbot_feature.py",
    }
    for label, path in backend_sources.items():
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"missing regular backend source: {path}")
        source_bindings[label] = {"path": str(path), "sha256": sha256_file(path)}
    mllm_ready, mllm_reason = mllm_canonical_ready(mllm_extractor)
    if mllm_extractor.is_file() and not mllm_extractor.is_symlink():
        source_bindings["mllm_extractor"] = {
            "path": str(mllm_extractor),
            "sha256": sha256_file(mllm_extractor),
        }

    canary_bindings = validate_backend_canary_receipts(
        args.backend_canary_dir,
        protocol_file_sha=protocol_file_sha,
        main_extractor=main_extractor,
        mllm_extractor=mllm_extractor,
        canary_script=backend_canary_script,
    )

    groups = build_groups(rows, allow_pending=args.allow_pending, target_policy=args.target_policy)
    namespaces = protocol["feature_namespaces"]
    dataset_root = args.dataset_root.resolve()
    # The extractor fail-closes unless --out-root basename exactly equals the
    # protocol namespace.  The namespace's _SEEDED_V1 suffix is the immutable
    # cache version; protocol-file SHA is bound in every shard/receipt rather
    # than appended beneath the namespace.
    root_version = f"{namespaces['streaming']}|{namespaces['target']}"
    model_ids_by_vfm: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        model_ids_by_vfm[row["vfm"]].add(row["model_id"])
    collisions = {
        vfm: sorted(model_ids)
        for vfm, model_ids in model_ids_by_vfm.items()
        if len(model_ids) > 1
    }
    if collisions:
        raise SystemExit(
            "protocol namespaces store one model id per VFM; output collision risk: "
            + json.dumps(collisions, sort_keys=True)
        )
    canary_identity = canonical_json_sha(canary_bindings)[:10]
    plan_id = (
        f"windowfix-v2-{protocol_file_sha[:12]}-{sha256_file(specs_path)[:12]}-"
        f"{canary_identity}"
    )
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise SystemExit(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    (staging / "shards").mkdir()
    (staging / "batches").mkdir()
    (staging / "jobspecs").mkdir()

    shards: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    root_pairs: dict[tuple[str, str, str, int], dict[str, str]] = {}
    group_counter = 0
    for key in sorted(groups):
        group = groups[key]
        backend = backend_for(group["vfm"])
        model_key = slug(f"{group['vfm']}--{group['model']}--{group['model_id']}")
        cache_identity = f"{model_key}/t{group['timestep']}"
        stream_root = dataset_root / namespaces["streaming"]
        target_root = dataset_root / namespaces["target"]
        root_pairs[key] = {"streaming": str(stream_root), "target": str(target_root)}
        reasons: list[str] = []
        if backend == "unsupported":
            reasons.append(f"unsupported vfm: {group['vfm']}")
        if backend == "mllm" and not mllm_ready:
            reasons.append(mllm_reason)
        canary_binding = canary_bindings.get(group["vfm"])
        if canary_binding is not None and (
            canary_binding["model_id"] != group["model_id"]
            or canary_binding["timestep"] != group["timestep"]
            or canary_binding["layer"] not in group["layers"]
        ):
            reasons.append(
                "BLOCKED_CANARY_IDENTITY_MISMATCH: receipt model/timestep/layer "
                "does not match this extraction group"
            )
        readiness_reason = BACKEND_READINESS_BLOCKERS.get(group["vfm"])
        if readiness_reason and canary_binding is None:
            reasons.append(readiness_reason)
        elif group["vfm"] not in CERTIFIED_READY_VFMS and canary_binding is None:
            reasons.append(
                "BLOCKED_UNCERTIFIED_BACKEND: backend has no current canonical-v2 "
                "determinism/provenance certification"
            )
        modes: list[tuple[str, set[str], set[int], Path, str]] = []
        if group["stream_spec_ids"]:
            modes.append(
                (
                    "streaming_prefix",
                    group["stream_spec_ids"],
                    group["stream_layers"],
                    stream_root,
                    "streaming",
                )
            )
        if group["target_spec_ids"]:
            modes.append(
                (
                    "target_isolated",
                    group["target_spec_ids"],
                    group["target_layers"],
                    target_root,
                    "target",
                )
            )
        if reasons and modes:
            blocked.append(
                {
                    "vfm": group["vfm"],
                    "model": group["model"],
                    "model_id": group["model_id"],
                    "timestep": group["timestep"],
                    "layers": sorted(group["layers"]),
                    "modes": [mode for mode, *_ in modes],
                    "layers_by_mode": {
                        mode: sorted(mode_layers) for mode, _ids, mode_layers, _root, _kind in modes
                    },
                    "spec_ids": sorted(set().union(*(item[1] for item in modes))),
                    "status": "BLOCKED",
                    "reasons": reasons,
                }
            )
            continue
        for mode, selected_ids, selected_layers, out_root, kind in modes:
            group_counter += 1
            job_token = slug(f"{group['vfm']}-{group['timestep']}-{mode}-{group_counter}", limit=44)
            shard_id = f"wf2-{group_counter:03d}-{job_token}"
            shard_path = output_dir / "shards" / f"{shard_id}.json"
            shard = {
                "schema_version": SCHEMA_VERSION,
                "plan_id": plan_id,
                "shard_id": shard_id,
                "status": "READY",
                "canonical_protocol": protocol["canonical_protocol"],
                "protocol_path": str(protocol_path),
                "protocol_file_sha256": protocol_file_sha,
                "protocol_sha256": feature_protocol_sha,
                "canonical_obs_len": 28,
                "prefixes": list(EXPECTED_PREFIXES),
                "horizons": list(EXPECTED_HORIZONS),
                "include_anchors": True,
                "canonical_binding": compact_binding,
                "source_bindings": source_bindings,
                "backend": backend,
                "python": python_for(backend),
                "environment": environment_for(backend, project_root),
                "project_root": str(project_root),
                "dataset_data_root": str(dataset_root / "data"),
                "feature_namespace": namespaces[kind],
                "feature_root": str(out_root),
                "publish_uri": f"aoss://baiqiao/InsScene-15K/{namespaces[kind]}",
                "cache_identity": cache_identity,
                "source": args.source,
                "mode": mode,
                "vfm": group["vfm"],
                "model": group["model"],
                "model_id": group["model_id"],
                "model_variants": sorted(group["model_variants"]),
                "timestep": group["timestep"],
                "layers": sorted(selected_layers),
                "prefixes_in_inventory": sorted(group["prefixes_in_inventory"]),
                "selected_spec_ids": sorted(selected_ids),
                "all_group_spec_ids": sorted(group["spec_ids"]),
                "action_counts": dict(sorted(group["actions"].items())),
                "scene_start": 0,
                "scene_end": args.scene_count,
                "n_gpus": args.n_gpus,
                "no_alias": True,
                "reuse_contract": (
                    "Only regular old tensors with exact canonical raw IDs and verified "
                    "model/layer/timestep/shape/dtype may be copied before this fill job."
                ),
                "backend_canary_binding": canary_binding,
            }
            shard["content_sha256"] = canonical_json_sha(shard)
            shards.append(shard)
            staged_shard = staging / "shards" / f"{shard_id}.json"
            atomic_write_json(staged_shard, shard)

    # Every VFM owns one structural index subtree.  Serialize all of its
    # timesteps/modes inside one SCO batch so two jobs can never race while
    # atomically replacing the same prefix_index.npy or target_indices.npy.
    shards_by_vfm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shard in shards:
        shards_by_vfm[shard["vfm"]].append(shard)
    for batch_number, vfm in enumerate(sorted(shards_by_vfm), start=1):
        ordered = sorted(
            shards_by_vfm[vfm],
            key=lambda item: (
                int(item["timestep"]),
                0 if item["mode"] == "streaming_prefix" else 1,
                item["shard_id"],
            ),
        )
        batch_id = f"wf2-batch-{batch_number:02d}-{slug(vfm, limit=32)}"
        batch_path = output_dir / "batches" / f"{batch_id}.json"
        batch = {
            "schema_version": SCHEMA_VERSION,
            "plan_id": plan_id,
            "batch_id": batch_id,
            "status": "READY",
            "vfm": vfm,
            "serialization_scope": f"{namespaces['streaming']}/{vfm} and {namespaces['target']}/{vfm}",
            "serialization_reason": (
                "All timesteps and modes for one VFM share scene structural indexes; "
                "run them sequentially to prevent prefix_index/target_indices writer races."
            ),
            "shards": [
                {
                    "path": str(output_dir / "shards" / f"{item['shard_id']}.json"),
                    "sha256": sha256_file(staging / "shards" / f"{item['shard_id']}.json"),
                    "shard_id": item["shard_id"],
                    "mode": item["mode"],
                    "timestep": item["timestep"],
                }
                for item in ordered
            ],
        }
        batch["content_sha256"] = canonical_json_sha(batch)
        staged_batch = staging / "batches" / f"{batch_id}.json"
        atomic_write_json(staged_batch, batch)
        worker_argv = [
            python_for(backend_for(vfm)),
            str(runner),
            "--batch",
            str(batch_path),
            "--execute",
        ]
        worker_command = f"cd {shlex.quote(str(project_root))} && {render_command(worker_argv)}"
        job_name = slug(f"bq-wf2-{vfm}-serial-{batch_number}", limit=60)
        sco_argv = [
            "sco",
            "acp",
            "jobs",
            "create",
            f"--workspace-name={args.workspace}",
            f"--aec2-name={args.aec2_name}",
            f"--job-name={job_name}",
            f"--priority={args.priority}",
            f"--container-image-url={args.container_image}",
            f"--storage-mount={args.storage_mount}",
            "--training-framework=pytorch",
            "--worker-nodes=1",
            f"--worker-spec={args.worker_spec}",
            f"--command={worker_command}",
        ]
        jobspec = {
            "schema_version": SCHEMA_VERSION,
            "plan_id": plan_id,
            "job_name": job_name,
            "batch_id": batch_id,
            "batch_path": str(batch_path),
            "batch_sha256": sha256_file(staged_batch),
            "shard_count": len(ordered),
            "sco_argv": sco_argv,
            "sco_command": render_command(sco_argv),
            "submission_policy": "REVIEW_ONLY_NEVER_SUBMITTED_BY_PREPARER",
        }
        atomic_write_json(staging / "jobspecs" / f"{batch_id}.json", jobspec)

    audit_plans: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        if not group["target_spec_ids"]:
            continue
        roots = root_pairs[key]
        if any(item["model_id"] == group["model_id"] and item["timestep"] == group["timestep"] for item in blocked):
            continue
        for prefix in EXPECTED_PREFIXES:
            report = (
                output_dir
                / "audit"
                / slug(f"{group['vfm']}--{group['model_id']}--t{group['timestep']}")
                / f"prefix{prefix}.json"
            )
            argv = [
                python_for(backend_for(group["vfm"])),
                str(audit_script),
                "--canonical-source",
                compact_binding["canonical_source"]["path"],
                "--canonical-source-sha256",
                compact_binding["canonical_source"]["sha256"],
                "--namespace-lock",
                compact_binding["namespace_lock"]["path"],
                "--namespace-lock-sha256",
                compact_binding["namespace_lock"]["sha256"],
                "--label-audit",
                compact_binding["label_lineage_audit"]["path"],
                "--label-audit-sha256",
                compact_binding["label_lineage_audit"]["sha256"],
                "--prefix",
                str(prefix),
                "--streaming-index-root",
                str(Path(roots["streaming"]) / group["vfm"] / args.source),
                "--target-index-root",
                str(Path(roots["target"]) / group["vfm"] / args.source),
                "--vfm",
                group["vfm"],
                "--source",
                args.source,
                "--output",
                str(report),
            ]
            audit_plans.append(
                {
                    "vfm": group["vfm"],
                    "model_id": group["model_id"],
                    "timestep": group["timestep"],
                    "prefix": prefix,
                    "output": str(report),
                    "argv": argv,
                    "command": render_command(argv),
                }
            )

    jobspecs = sorted((staging / "jobspecs").glob("*.json"))
    launch_lines = [
        "# REVIEW ARTIFACT ONLY. This file is not executed by the preparer.",
        "# Each line below is a shell-quoted SCO create command.",
    ]
    for path in jobspecs:
        launch_lines.append(json.loads(path.read_text(encoding="utf-8"))["sco_command"])
    (staging / "sco_create_commands.txt").write_text("\n".join(launch_lines) + "\n", encoding="utf-8")
    atomic_write_json(staging / "audit_plan.json", {"schema_version": 1, "audits": audit_plans})
    atomic_write_json(staging / "blocked.json", {"schema_version": 1, "blocked": blocked})

    summary = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan_id,
        "status": "BLOCKED" if blocked else "READY_FOR_REVIEW",
        "submission_performed": False,
        "spec_inventory": {"path": str(specs_path), "sha256": sha256_file(specs_path), "rows": len(rows)},
        "protocol": {
            "path": str(protocol_path),
            "file_sha256": protocol_file_sha,
            "feature_protocol_sha256": feature_protocol_sha,
            "id": protocol["canonical_protocol"],
        },
        "canonical_binding": compact_binding,
        "source_bindings": source_bindings,
        "backend_canary_bindings": canary_bindings,
        "canonical_contract": {
            "obs_len": 28,
            "prefixes": list(EXPECTED_PREFIXES),
            "horizons": list(EXPECTED_HORIZONS),
            "include_anchors": True,
            "no_alias": True,
            "root_version": root_version,
        },
        "counts": {
            "specs": len(rows),
            "model_timestep_groups": len(groups),
            "ready_shards": len(shards),
            "ready_sco_batches": len(shards_by_vfm),
            "streaming_shards": sum(item["mode"] == "streaming_prefix" for item in shards),
            "target_shards": sum(item["mode"] == "target_isolated" for item in shards),
            "blocked_groups": len(blocked),
            "audit_commands": len(audit_plans),
        },
        "blocked": blocked,
        "notes": [
            "No SCO API or CLI was invoked.",
            "Every extraction shard emits all four canonical prefixes to preserve complete prefix_index.npy files.",
            "AOSS publication is intentionally separated until every generated full audit passes.",
        ],
    }
    atomic_write_json(staging / "summary.json", summary)
    lock_payload = {
        "schema_version": 1,
        "plan_id": plan_id,
        "summary_sha256": sha256_file(staging / "summary.json"),
        "shards": {
            path.name: sha256_file(path) for path in sorted((staging / "shards").glob("*.json"))
        },
        "batches": {
            path.name: sha256_file(path) for path in sorted((staging / "batches").glob("*.json"))
        },
        "jobspecs": {path.name: sha256_file(path) for path in jobspecs},
        "audit_plan_sha256": sha256_file(staging / "audit_plan.json"),
        "commands_sha256": sha256_file(staging / "sco_create_commands.txt"),
    }
    atomic_write_json(staging / "PLAN.lock.json", lock_payload)
    os.replace(staging, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
