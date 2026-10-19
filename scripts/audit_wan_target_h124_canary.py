#!/usr/bin/env python3
"""Independent, CPU-only audit of the two-scene Wan h124 target canary.

The audit deliberately does not import the extractor under test (or any of
its validation helpers).  Its only write is the explicitly requested report,
which is created with O_EXCL semantics after all read-only checks finish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors


MANIFEST_SHA256 = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
SOURCE_AUDIT_SHA256 = "c09a11b6b60d97d162d264365e46339b9d1626d40b80665c18fa4783dffc4010"
BASE_EXTRACTOR_SHA256 = "654644424e94ea8c9a1efecab5e1a65abf37321045d9b97a1f53e9bd668353d0"
LAYERS = (0, 5, 10, 15, 25, 29)
HORIZONS = (1, 2, 4)
FEATURE_SHAPE = (30, 52, 1536)
TIMESTEP = 749
INPUT_FRAMES = 81
BASE_SEED = 42
PROMPT = "."
EXPECTED_MANIFEST_ROWS = 2724
EXPECTED_MANIFEST_SCENES = 681
EXPECTED_SPLITS = {"train": 2448, "val": 276}
EXPECTED_CANARY_SCENES = 2
EXPECTED_CANARY_ROWS = 8
SHA_RE = re.compile(r"[0-9a-f]{64}")


class AuditError(RuntimeError):
    """A fail-closed contract violation."""


@dataclass(frozen=True)
class AuditContract:
    manifest_sha256: str = MANIFEST_SHA256
    source_audit_sha256: str = SOURCE_AUDIT_SHA256
    base_extractor_sha256: str = BASE_EXTRACTOR_SHA256
    layers: tuple[int, ...] = LAYERS
    horizons: tuple[int, ...] = HORIZONS
    feature_shape: tuple[int, ...] = FEATURE_SHAPE
    timestep: int = TIMESTEP
    input_frames: int = INPUT_FRAMES
    base_seed: int = BASE_SEED
    prompt: str = PROMPT
    manifest_rows: int = EXPECTED_MANIFEST_ROWS
    manifest_scenes: int = EXPECTED_MANIFEST_SCENES
    split_counts: tuple[tuple[str, int], ...] = tuple(EXPECTED_SPLITS.items())
    canary_scenes: int = EXPECTED_CANARY_SCENES
    canary_rows: int = EXPECTED_CANARY_ROWS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256_and_size(path: Path, label: str) -> tuple[str, int]:
    before = path.stat()
    before_link = path.lstat()
    digest = sha256(path)
    after = path.stat()
    after_link = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields) or any(
        getattr(before_link, key) != getattr(after_link, key) for key in fields
    ):
        raise AuditError(f"{label} changed while hashing: {path}")
    return digest, after.st_size


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise AuditError(f"unreadable JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path}")
    return value


def require_sha(value: Any, label: str) -> str:
    text = str(value)
    if SHA_RE.fullmatch(text) is None:
        raise AuditError(f"invalid SHA256 for {label}: {value!r}")
    return text


def deterministic_seed(base_seed: int, key: str) -> int:
    payload = hashlib.sha256(f"{base_seed}:{key}".encode()).digest()
    return int.from_bytes(payload[:4], "big") & 0x7FFFFFFF


def scene_id(row: dict[str, Any]) -> str:
    return str(row.get("scene_id") or Path(str(row["scene_dir"])).name)


def derive_scene_mapping(
    scene: str, rows: list[dict[str, Any]], contract: AuditContract
) -> dict[int, dict[str, Any]]:
    """Derive h1/h2/h4 only from tail and frozen window indices."""
    if len(rows) != 4 or len({str(row.get("sample_id")) for row in rows}) != 4:
        raise AuditError(f"{scene}: canary must contain four unique manifest rows")
    mapping: dict[int, dict[str, Any]] = {}
    horizon_counts: Counter[int] = Counter()
    for row in rows:
        if scene_id(row) != scene:
            raise AuditError(f"{scene}: scene grouping mismatch")
        try:
            tail = int(row["streaming_tail"])
            prefix = [int(value) for value in row["streaming_indices"]]
            window = [int(value) for value in row["streaming_window_indices"]]
            frames = [str(value) for value in row["valid_frames"]]
        except Exception as exc:
            raise AuditError(f"{row.get('sample_id')}: malformed frozen manifest row") from exc
        if prefix != window[: tail + 1]:
            raise AuditError(f"{row['sample_id']}: prefix != window[:tail+1]")
        for horizon in contract.horizons:
            position = tail + horizon
            if position < 0 or position >= len(window):
                raise AuditError(f"{row['sample_id']}: h{horizon} outside frozen window")
            raw_id = int(window[position])
            if raw_id < 0 or raw_id >= len(frames):
                raise AuditError(f"{row['sample_id']}: raw frame ID outside valid_frames")
            if raw_id in mapping:
                raise AuditError(f"{scene}: duplicated h124 raw ID {raw_id}")
            mapping[raw_id] = {
                "horizon": horizon,
                "sample_id": str(row["sample_id"]),
                "frame_name": frames[raw_id],
                "source": f"streaming_window_indices[{position}]",
                "seed": deterministic_seed(
                    contract.base_seed, f"target:{scene}:{raw_id}"
                ),
            }
            horizon_counts[horizon] += 1
    if len(mapping) != 12 or horizon_counts != Counter({1: 4, 2: 4, 4: 4}):
        raise AuditError(f"{scene}: expected 12 disjoint h1/h2/h4 targets")
    return mapping


def load_frozen_selection(
    manifest: Path, selection: Path, contract: AuditContract
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if sha256(manifest) != contract.manifest_sha256:
        raise AuditError("frozen manifest SHA mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) != contract.manifest_rows:
        raise AuditError(f"manifest row count {len(rows)} != {contract.manifest_rows}")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not item for item in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise AuditError("manifest sample IDs are empty or duplicated")
    grouped_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_all[scene_id(row)].append(row)
    if len(grouped_all) != contract.manifest_scenes:
        raise AuditError("manifest scene count mismatch")
    if Counter(str(row.get("split")) for row in rows) != Counter(dict(contract.split_counts)):
        raise AuditError("manifest split counts mismatch")

    raw_selected = [
        line.strip()
        for line in selection.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(raw_selected) != contract.canary_rows or len(set(raw_selected)) != len(raw_selected):
        raise AuditError("selection must contain exactly eight unique sample IDs")
    selected_set = set(raw_selected)
    missing = selected_set - set(sample_ids)
    if missing:
        raise AuditError(f"selection IDs absent from manifest: {sorted(missing)}")
    selected_rows = [row for row in rows if str(row["sample_id"]) in selected_set]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[scene_id(row)].append(row)
    if len(grouped) != contract.canary_scenes:
        raise AuditError("selection must resolve to exactly two scenes")
    for scene, scene_rows in grouped.items():
        if len(scene_rows) != len(grouped_all[scene]):
            raise AuditError(f"selection is a partial scene: {scene}")
        derive_scene_mapping(scene, scene_rows, contract)
    return rows, {scene: grouped[scene] for scene in sorted(grouped)}


def bundle_internal_hash_audit(bundle_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    root = bundle_path.parent.resolve()
    hashes = bundle.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise AuditError("bundle source_hashes is empty or invalid")
    checks = []
    for relative, expected_raw in sorted(hashes.items()):
        relative_path = Path(str(relative))
        if relative_path.is_absolute():
            raise AuditError(f"bundle source hash path is absolute: {relative}")
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise AuditError(f"bundle source hash escapes bundle: {relative}") from exc
        expected = require_sha(expected_raw, f"source_hashes[{relative}]")
        if not path.is_file():
            raise AuditError(f"bundle source-hash file is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise AuditError(f"bundle internal source SHA mismatch: {relative}")
        checks.append({"path": str(path), "relative": str(relative), "sha256": actual})
    return {"ok": True, "count": len(checks), "checks": checks}


def require_bundle_binding(
    path: Path, source_hashes: dict[str, Any], bundle_root: Path, label: str
) -> dict[str, str]:
    path = path.resolve()
    actual = sha256(path)
    matches = []
    for relative, expected in source_hashes.items():
        candidate = (bundle_root / str(relative)).resolve()
        if candidate == path and str(expected) == actual:
            matches.append(str(relative))
    if len(matches) != 1:
        raise AuditError(f"{label} is not uniquely bound by bundle source_hashes")
    return {"path": str(path), "sha256": actual, "bundle_key": matches[0]}


def external_root(bundle: dict[str, Any], *names: str) -> tuple[str, Path]:
    external = bundle.get("external_roots")
    if not isinstance(external, dict):
        external = {}
    candidates: list[tuple[str, Any]] = []
    for name in names:
        if name in bundle:
            candidates.append((name, bundle[name]))
        if name in external:
            candidates.append((f"external_roots.{name}", external[name]))
    normalized = {(key, str(Path(str(value)).resolve())) for key, value in candidates if value}
    paths = {value for _, value in normalized}
    if len(paths) != 1:
        raise AuditError(f"bundle must resolve exactly one {names} root, got {sorted(normalized)}")
    key, value = sorted(normalized)[0]
    return key, Path(value)


def parse_inventory(path: Path, base: Path | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    obj = read_object(path)
    raw = obj.get("artifacts")
    if raw is None:
        raw = obj.get("hashes", obj)
    entries: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        entries = [{"path": key, "sha256": value} for key, value in raw.items()]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise AuditError(f"unsupported inventory schema: {path}")
    result: dict[str, dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
            raise AuditError(f"malformed inventory entry: {path}")
        raw_path = Path(str(item["path"]))
        if not raw_path.is_absolute() and base is None:
            raise AuditError(
                f"relative inventory entry requires an explicit frozen base: {raw_path}"
            )
        logical = raw_path.absolute() if raw_path.is_absolute() else (base / raw_path).absolute()
        key = str(logical)
        if key in result:
            raise AuditError(f"duplicate inventory path: {key}")
        normalized = dict(item)
        normalized["path"] = key
        normalized["sha256"] = require_sha(item["sha256"], key)
        result[key] = normalized
    if not result:
        raise AuditError(f"empty inventory: {path}")
    return result, obj


def verify_current_inventory(
    inventory: dict[str, dict[str, Any]], label: str
) -> dict[str, Any]:
    checked = []
    for path_string, item in sorted(inventory.items()):
        path = Path(path_string)
        if not path.is_file():
            raise AuditError(f"{label} file missing: {path}")
        realpath = str(path.resolve())
        actual_sha, actual_size = stable_sha256_and_size(path, label)
        actual_symlink = path.is_symlink()
        if actual_sha != item["sha256"]:
            raise AuditError(f"{label} SHA changed: {path}")
        if "size_bytes" in item and int(item["size_bytes"]) != actual_size:
            raise AuditError(f"{label} size changed: {path}")
        if "realpath" in item and str(Path(str(item["realpath"])).resolve()) != realpath:
            raise AuditError(f"{label} realpath changed: {path}")
        if "is_symlink" in item and bool(item["is_symlink"]) != actual_symlink:
            raise AuditError(f"{label} symlink status changed: {path}")
        checked.append({"path": path_string, "sha256": actual_sha, "size_bytes": actual_size})
    return {"ok": True, "count": len(checked), "checked": checked}


def validate_canary_input_inventory(
    inventory: dict[str, dict[str, Any]],
    inventory_object: dict[str, Any],
    data_root: Path,
    groups: dict[str, list[dict[str, Any]]],
    contract: AuditContract,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Bind the 16 immutable h2/h4 source images to frozen manifest identity."""
    frozen_root = inventory_object.get("root")
    if not isinstance(frozen_root, str) or not Path(frozen_root).is_absolute():
        raise AuditError("canary input-image inventory root is not absolute")
    if Path(frozen_root).absolute() != data_root or Path(frozen_root).resolve() != data_root:
        raise AuditError("canary input-image inventory root differs from data_root")
    if len(inventory) != contract.canary_scenes * 8:
        raise AuditError(
            f"canary input-image inventory count {len(inventory)} != "
            f"{contract.canary_scenes * 8}"
        )

    required_fields = {
        "path", "realpath", "is_symlink", "size_bytes", "sha256",
        "scene", "sample_id", "horizon", "raw_id", "frame_name",
    }
    expected_paths: set[str] = set()
    by_scene: dict[str, dict[str, dict[str, Any]]] = {}
    for scene, rows in sorted(groups.items()):
        mapping = derive_scene_mapping(scene, rows, contract)
        per_scene: dict[str, dict[str, Any]] = {}
        for raw_id, identity in sorted(mapping.items()):
            if identity["horizon"] == 1:
                continue
            image = (
                data_root / "processed_scannetpp_v2" / "processed_scannetpp_v2"
                / scene / "images" / identity["frame_name"]
            ).absolute()
            logical = str(image)
            expected_paths.add(logical)
            record = inventory.get(logical)
            if record is None:
                raise AuditError(f"missing frozen canary input image: {logical}")
            if not required_fields.issubset(record):
                raise AuditError(f"incomplete canary input-image record: {logical}")
            expected_identity = {
                "scene": scene,
                "sample_id": identity["sample_id"],
                "horizon": identity["horizon"],
                "raw_id": raw_id,
                "frame_name": identity["frame_name"],
            }
            for key, expected in expected_identity.items():
                if record.get(key) != expected:
                    raise AuditError(f"canary input image {logical}: {key} mismatch")
            if record.get("is_symlink") != image.is_symlink():
                raise AuditError(f"canary input image symlink status mismatch: {logical}")
            if not image.is_file():
                raise AuditError(f"canary input image is not a regular file: {logical}")
            resolved = image.resolve()
            try:
                resolved.relative_to(data_root)
            except ValueError as exc:
                raise AuditError(f"canary input image escapes data_root: {logical}") from exc
            if str(resolved) != str(record.get("realpath")):
                raise AuditError(f"canary input image realpath mismatch: {logical}")
            per_scene[str(raw_id)] = dict(record)
        if len(per_scene) != 8:
            raise AuditError(f"{scene}: expected eight frozen h2/h4 input images")
        by_scene[scene] = per_scene
    if set(inventory) != expected_paths:
        raise AuditError("canary input-image inventory has extra or missing paths")
    return by_scene


def verify_final_output_inventory(
    records: list[dict[str, Any]], out_root: Path
) -> dict[str, Any]:
    """Last-pass rehash of every collected output artifact and sidecar."""
    root = out_root.resolve()
    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item["path"])):
        raw = Path(str(record["path"]))
        if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
            raise AuditError(f"final output is not an absolute regular non-symlink: {raw}")
        resolved = raw.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise AuditError(f"final output escapes output root: {raw}") from exc
        if raw.absolute() != resolved:
            raise AuditError(f"final output path is not realpath-stable: {raw}")
        key = str(resolved)
        if key in seen:
            raise AuditError(f"duplicate final output inventory path: {key}")
        seen.add(key)
        actual_sha, actual_size = stable_sha256_and_size(raw, "final output")
        if actual_sha != record.get("sha256"):
            raise AuditError(f"final output SHA changed during audit: {raw}")
        if int(record.get("size_bytes", -1)) != actual_size:
            raise AuditError(f"final output size changed during audit: {raw}")
        checked.append({"path": key, "sha256": actual_sha, "size_bytes": actual_size})
    if not checked:
        raise AuditError("final output inventory is empty")
    return {"ok": True, "count": len(checked), "checked": checked}


def audit_protected(path: Path, repo_root: Path) -> dict[str, Any]:
    before_obj = read_object(path)
    if not before_obj or not all(isinstance(key, str) for key in before_obj):
        raise AuditError("protected_before is empty or malformed")
    repo_root = repo_root.resolve()
    entries = {}
    for relative, expected in before_obj.items():
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise AuditError(f"protected path must be repo-relative: {relative}")
        target = (repo_root / relative_path).resolve()
        try:
            target.relative_to(repo_root)
        except ValueError as exc:
            raise AuditError(f"protected path escapes frozen repo root: {relative}") from exc
        entries[str(target)] = {
            "path": str(target),
            "sha256": require_sha(expected, f"protected:{relative}"),
            "repo_relative": relative,
        }
    result = verify_current_inventory(entries, "protected")
    result["before_equals_after"] = True
    return result


def frozen_repo_root(bundle: dict[str, Any]) -> tuple[str, Path]:
    """Resolve only the explicit external repo root, never ``__file__`` parents."""
    external = bundle.get("external_roots")
    if not isinstance(external, dict) or "repo_root" not in external:
        raise AuditError("bundle external_roots.repo_root is required")
    raw = Path(str(external["repo_root"]))
    if not raw.is_absolute():
        raise AuditError("bundle external_roots.repo_root must be absolute")
    absolute = raw.absolute()
    resolved = raw.resolve()
    if absolute != resolved:
        raise AuditError("bundle external_roots.repo_root must be a real, normalized path")
    if not resolved.is_dir() or resolved.is_symlink():
        raise AuditError("bundle external_roots.repo_root is not a real directory")
    if not (resolved / ".git").exists():
        raise AuditError("bundle external_roots.repo_root is not a git worktree")
    return "external_roots.repo_root", resolved


def validate_artifact_sidecar(
    artifact: Path, required: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    sidecar = Path(f"{artifact}.manifest.json")
    if not artifact.is_file() or not sidecar.is_file():
        raise AuditError(f"missing artifact/sidecar: {artifact}")
    metadata = read_object(sidecar)
    artifact_sha = sha256(artifact)
    size = artifact.stat().st_size
    if metadata.get("sha256") != artifact_sha:
        raise AuditError(f"artifact SHA differs from sidecar: {artifact}")
    if int(metadata.get("size_bytes", -1)) != size:
        raise AuditError(f"artifact size differs from sidecar: {artifact}")
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise AuditError(
                f"{artifact}: sidecar {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    record = {
        "path": str(artifact.resolve()),
        "sha256": artifact_sha,
        "size_bytes": size,
        "sidecar_path": str(sidecar.resolve()),
        "sidecar_sha256": sha256(sidecar),
        "sidecar_size_bytes": sidecar.stat().st_size,
    }
    return metadata, record


def tensor_contract(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if list(handle.keys()) != ["feat"]:
            raise AuditError(f"unexpected safetensors keys: {path}")
        view = handle.get_slice("feat")
        if tuple(view.get_shape()) != shape or str(view.get_dtype()) != "F16":
            raise AuditError(f"tensor header contract mismatch: {path}")
    tensor = load_safetensors(str(path), device="cpu")["feat"]
    if tensor.dtype != torch.float16 or tuple(tensor.shape) != shape:
        raise AuditError(f"loaded tensor contract mismatch: {path}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise AuditError(f"non-finite tensor: {path}")
    return tensor


def source_common_required(
    scene: str,
    rows: list[dict[str, Any]],
    h1_ids: list[int],
    model_id: Path,
    contract: AuditContract,
) -> dict[str, Any]:
    return {
        "canonical_manifest_sha256": contract.manifest_sha256,
        "extractor_sha256": contract.base_extractor_sha256,
        "mode": "manifest_exact_target",
        "model_id": str(model_id.resolve()),
        "prompt": contract.prompt,
        "timestep": contract.timestep,
        "layers": list(contract.layers),
        "num_frames": contract.input_frames,
        "base_seed": contract.base_seed,
        "target_raw_ids": h1_ids,
        "scene_id": scene,
        "sample_ids": sorted(str(row["sample_id"]) for row in rows),
    }


def output_common_required(
    scene: str,
    rows: list[dict[str, Any]],
    mapping: dict[int, dict[str, Any]],
    manifest: Path,
    source_root: Path,
    source_audit: Path,
    model_id: Path,
    base_extractor: Path,
    extractor: Path,
    git_commit: str,
    extracted_input_images_by_raw_id: dict[str, dict[str, Any]],
    contract: AuditContract,
) -> dict[str, Any]:
    all_ids = sorted(mapping)
    return {
        "schema_version": 3,
        "frame_index_schema": "canonical_manifest_exact_h124_v1",
        "mode": "manifest_exact_target_h124_extend",
        "canonical_manifest": str(manifest.resolve()),
        "canonical_manifest_sha256": contract.manifest_sha256,
        "source_h1_root": str(source_root.resolve()),
        "source_h1_audit": str(source_audit.resolve()),
        "source_h1_audit_sha256": contract.source_audit_sha256,
        "base_extractor": str(base_extractor.resolve()),
        "base_extractor_sha256": contract.base_extractor_sha256,
        "wrapper": str(extractor.resolve()),
        "wrapper_sha256": sha256(extractor),
        "model_id": str(model_id.resolve()),
        "prompt": contract.prompt,
        "timestep": contract.timestep,
        "layers": list(contract.layers),
        "num_frames": contract.input_frames,
        "effective_input_frames": contract.input_frames,
        "legacy_extractor_config_input_frames": 24,
        "base_seed": contract.base_seed,
        "seed_scheme": "sha256(base_seed:target:scene:raw_id)[:31bit]",
        "reused_horizons": [1],
        "extracted_horizons": [2, 4],
        "motion_window_generator_used": False,
        "git_commit": git_commit,
        "git_commit_verified": True,
        "extracted_input_images_by_raw_id": extracted_input_images_by_raw_id,
        "scene_id": scene,
        "sample_ids": sorted(str(row["sample_id"]) for row in rows),
        "target_raw_ids": all_ids,
        "target_horizon_by_raw_id": {
            str(raw_id): mapping[raw_id]["horizon"] for raw_id in all_ids
        },
        "target_sample_id_by_raw_id": {
            str(raw_id): mapping[raw_id]["sample_id"] for raw_id in all_ids
        },
        "target_frame_names": [mapping[raw_id]["frame_name"] for raw_id in all_ids],
        "target_id_sources": {
            str(raw_id): mapping[raw_id]["source"] for raw_id in all_ids
        },
        "sample_seeds": {
            str(raw_id): mapping[raw_id]["seed"] for raw_id in all_ids
        },
    }


def audit_scene(
    scene: str,
    rows: list[dict[str, Any]],
    out_root: Path,
    source_root: Path,
    source_audit: Path,
    manifest: Path,
    model_id: Path,
    base_extractor: Path,
    extractor: Path,
    git_commit: str,
    extracted_input_images_by_raw_id: dict[str, dict[str, Any]],
    contract: AuditContract,
) -> dict[str, Any]:
    mapping = derive_scene_mapping(scene, rows, contract)
    all_ids = sorted(mapping)
    h1_ids = [raw_id for raw_id in all_ids if mapping[raw_id]["horizon"] == 1]
    source_scene = source_root / "wan" / "scannetpp" / scene
    output_scene = out_root / "wan" / "scannetpp" / scene
    if not output_scene.is_dir() or output_scene.is_symlink():
        raise AuditError(f"missing or symlinked output scene: {output_scene}")
    expected_names = {"target_indices.npy", "target_indices.npy.manifest.json"}
    for layer in contract.layers:
        name = f"feature_t{contract.timestep}_layer{layer}.sft"
        expected_names.update({name, f"{name}.manifest.json"})
    actual_names = {item.name for item in output_scene.iterdir()}
    if actual_names != expected_names:
        raise AuditError(f"{scene}: unexpected output scene entries")
    output_scene_resolved = output_scene.resolve()
    try:
        output_scene_resolved.relative_to(out_root.resolve())
    except ValueError as exc:
        raise AuditError(f"{scene}: output scene escapes output root") from exc
    if output_scene.absolute() != output_scene_resolved:
        raise AuditError(f"{scene}: output scene path is not realpath-stable")
    invalid_entries = [
        item.name
        for item in output_scene.iterdir()
        if item.is_symlink()
        or not item.is_file()
        or item.resolve().parent != output_scene_resolved
    ]
    if invalid_entries:
        raise AuditError(
            f"{scene}: output entries are not regular non-symlink contained files: "
            f"{sorted(invalid_entries)}"
        )

    source_required = source_common_required(scene, rows, h1_ids, model_id, contract)
    source_index = source_scene / "target_indices.npy"
    _, source_index_record = validate_artifact_sidecar(
        source_index, {**source_required, "shape": [4], "dtype": "int64"}
    )
    source_ids = np.load(source_index, allow_pickle=False)
    if source_ids.dtype != np.int64 or source_ids.tolist() != h1_ids:
        raise AuditError(f"{scene}: immutable source h1 index mismatch")

    output_required = output_common_required(
        scene, rows, mapping, manifest, source_root, source_audit,
        model_id, base_extractor, extractor, git_commit,
        extracted_input_images_by_raw_id, contract,
    )
    output_required["source_h1_target_indices_sha256"] = source_index_record["sha256"]
    output_index = output_scene / "target_indices.npy"
    output_index_meta, output_index_record = validate_artifact_sidecar(
        output_index, {**output_required, "shape": [12], "dtype": "int64"}
    )
    output_ids = np.load(output_index, allow_pickle=False)
    if output_ids.dtype != np.int64 or output_ids.tolist() != all_ids:
        raise AuditError(f"{scene}: output indices do not match derived h124 IDs")

    source_artifacts = [source_index_record]
    output_artifacts = [output_index_record]
    h1_exact_rows = 0
    identity_reference = {
        "git_dirty": output_index_meta.get("git_dirty"),
        "legacy_config_discrepancy": output_index_meta.get("legacy_config_discrepancy"),
        "git_commit_verification_source": output_index_meta.get(
            "git_commit_verification_source"
        ),
    }
    if identity_reference["git_dirty"] not in (True, False, None):
        raise AuditError(f"{scene}: invalid output git_dirty")
    if not isinstance(identity_reference["legacy_config_discrepancy"], str):
        raise AuditError(f"{scene}: missing legacy config discrepancy annotation")
    verification_source = identity_reference["git_commit_verification_source"]
    if (
        not isinstance(verification_source, str)
        or not verification_source.strip()
        or verification_source.strip().lower() == "unknown"
    ):
        raise AuditError(f"{scene}: output git verification_source is absent")

    for layer in contract.layers:
        source_path = source_scene / f"feature_t{contract.timestep}_layer{layer}.sft"
        frame_names = [mapping[raw_id]["frame_name"] for raw_id in h1_ids]
        id_sources = {str(raw_id): mapping[raw_id]["source"] for raw_id in h1_ids}
        seeds = {str(raw_id): mapping[raw_id]["seed"] for raw_id in h1_ids}
        _, source_record = validate_artifact_sidecar(
            source_path,
            {
                **source_required,
                "target_frame_names": frame_names,
                "target_id_sources": id_sources,
                "sample_seeds": seeds,
                "tensors": {"feat": {"shape": [4, *contract.feature_shape], "dtype": "torch.float16"}},
            },
        )
        source_tensor = tensor_contract(source_path, (4, *contract.feature_shape))
        output_path = output_scene / f"feature_t{contract.timestep}_layer{layer}.sft"
        output_meta, output_record = validate_artifact_sidecar(
            output_path,
            {
                **output_required,
                **identity_reference,
                "source_h1_feature_sha256": source_record["sha256"],
                "tensors": {"feat": {"shape": [12, *contract.feature_shape], "dtype": "torch.float16"}},
            },
        )
        output_tensor = tensor_contract(output_path, (12, *contract.feature_shape))
        for source_position, raw_id in enumerate(h1_ids):
            output_position = all_ids.index(raw_id)
            if not torch.equal(output_tensor[output_position], source_tensor[source_position]):
                raise AuditError(f"{scene}/layer{layer}: h1 tensor row is not exact")
            h1_exact_rows += 1
        if output_meta.get("source_h1_feature_sha256") != source_record["sha256"]:
            raise AuditError(f"{scene}/layer{layer}: source feature binding mismatch")
        source_artifacts.append(source_record)
        output_artifacts.append(output_record)

    def flatten(records: Iterable[dict[str, Any]], scene_name: str) -> list[dict[str, Any]]:
        result = []
        for record in records:
            result.extend([
                {"path": record["path"], "sha256": record["sha256"], "scene": scene_name},
                {
                    "path": record["sidecar_path"],
                    "sha256": record["sidecar_sha256"],
                    "size_bytes": record["sidecar_size_bytes"],
                    "scene": scene_name,
                },
            ])
            result[-2]["size_bytes"] = record["size_bytes"]
        return result

    return {
        "scene": scene,
        "sample_ids": sorted(str(row["sample_id"]) for row in rows),
        "target_ids": all_ids,
        "horizon_counts": dict(sorted(Counter(item["horizon"] for item in mapping.values()).items())),
        "tensor_shape": [12, *contract.feature_shape],
        "tensor_dtype": "torch.float16",
        "finite_layers": len(contract.layers),
        "h1_exact_rows": h1_exact_rows,
        "expected_h1_exact_rows": 4 * len(contract.layers),
        "identity_reference": identity_reference,
        "source_inventory": flatten(source_artifacts, scene),
        "output_inventory": flatten(output_artifacts, scene),
    }


def exact_inventory_match(
    actual: list[dict[str, Any]], frozen: dict[str, dict[str, Any]], label: str
) -> None:
    actual_map: dict[str, dict[str, Any]] = {}
    for item in actual:
        path = str(Path(str(item["path"])).resolve())
        if path in actual_map:
            raise AuditError(f"duplicate {label} audit path: {path}")
        actual_map[path] = item
    if set(actual_map) != set(frozen):
        raise AuditError(
            f"{label} inventory paths differ: missing={sorted(set(frozen)-set(actual_map))}, "
            f"extra={sorted(set(actual_map)-set(frozen))}"
        )
    for path, item in actual_map.items():
        if item["sha256"] != frozen[path]["sha256"]:
            raise AuditError(f"{label} inventory SHA differs: {path}")
        if "scene" in frozen[path] and str(frozen[path]["scene"]) != str(item.get("scene")):
            raise AuditError(f"{label} inventory scene differs: {path}")


def audit_claims_and_shard(
    out_root: Path,
    shard_name: str,
    groups: dict[str, list[dict[str, Any]]],
    manifest: Path,
    selection: Path,
    source_root: Path,
    source_audit: Path,
    model_id: Path,
    extractor_sha: str,
    git_commit: str,
    contract: AuditContract,
    scene_results: list[dict[str, Any]],
) -> dict[str, Any]:
    provenance = out_root / "provenance"
    selected_scenes = sorted(groups)
    expected_output = [item for scene in scene_results for item in scene["output_inventory"]]
    expected_source = [item for scene in scene_results for item in scene["source_inventory"]]
    for scene in selected_scenes:
        claim_path = provenance / "scene_claims" / f"{scene}.json"
        complete_path = provenance / "scene_completions" / f"{scene}.json"
        claim = read_object(claim_path)
        complete = read_object(complete_path)
        required_claim = {
            "status": "CLAIMED", "scene": scene, "shard": shard_name,
            "manifest_sha256": contract.manifest_sha256, "wrapper_sha256": extractor_sha,
        }
        required_complete = {
            "status": "COMPLETE", "scene": scene, "shard": shard_name,
            "manifest_sha256": contract.manifest_sha256, "wrapper_sha256": extractor_sha,
            "target_count": 12, "layers": list(contract.layers),
        }
        for key, value in required_claim.items():
            if claim.get(key) != value:
                raise AuditError(f"{scene}: scene claim {key} mismatch")
        for key, value in required_complete.items():
            if complete.get(key) != value:
                raise AuditError(f"{scene}: scene completion {key} mismatch")
        per_scene_inventory = next(
            item["output_inventory"] for item in scene_results if item["scene"] == scene
        )
        canonical = [{"path": item["path"], "sha256": item["sha256"]} for item in per_scene_inventory]
        if complete.get("output_inventory_sha256") != canonical_json_sha256(canonical):
            raise AuditError(f"{scene}: scene completion inventory digest mismatch")

    shard_dir = provenance / "shards"
    claim_path = shard_dir / f"{shard_name}.claim.json"
    complete_path = shard_dir / f"{shard_name}.complete.json"
    output_inventory_path = shard_dir / f"{shard_name}.output_inventory.json"
    source_inventory_path = shard_dir / f"{shard_name}.source_inventory.json"
    claim = read_object(claim_path)
    complete = read_object(complete_path)
    claim_required = {
        "status": "CLAIMED", "shard": shard_name, "scenes": selected_scenes,
        "manifest_sha256": contract.manifest_sha256,
        "manifest": str(manifest.resolve()),
        "source_h1_root": str(source_root.resolve()),
        "source_h1_audit": str(source_audit.resolve()),
        "source_h1_audit_sha256": contract.source_audit_sha256,
        "selection_file": str(selection.resolve()),
        "selection_file_sha256": sha256(selection),
        "wrapper_sha256": extractor_sha,
        "git_commit": git_commit,
        "git_commit_verified": True,
    }
    for key, value in claim_required.items():
        if claim.get(key) != value:
            raise AuditError(f"shard claim {key} mismatch")
    runtime = claim.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("wrapper_sha256") != extractor_sha:
        raise AuditError("shard claim runtime wrapper mismatch")
    for container_name, container in (("shard claim", claim), ("shard runtime", runtime)):
        if container.get("git_commit") != git_commit:
            raise AuditError(f"{container_name} git_commit mismatch")
        if container.get("git_commit_verified") is not True:
            raise AuditError(f"{container_name} git_commit_verified is not true")
        source = container.get("git_commit_verification_source")
        if (
            not isinstance(source, str)
            or not source.strip()
            or source.strip().lower() == "unknown"
        ):
            raise AuditError(f"{container_name} git verification source is invalid")
    runtime_contract = runtime.get("contract")
    expected_runtime_contract = {
        "manifest_sha256": contract.manifest_sha256,
        "base_extractor_sha256": contract.base_extractor_sha256,
        "source_audit_sha256": contract.source_audit_sha256,
        "layers": list(contract.layers), "horizons": list(contract.horizons),
        "extract_horizons": [2, 4], "feature_shape": list(contract.feature_shape),
        "input_frames": contract.input_frames, "timestep": contract.timestep,
        "seed": contract.base_seed, "prompt": contract.prompt,
    }
    if runtime_contract != expected_runtime_contract:
        raise AuditError("shard runtime contract mismatch")
    argv = runtime.get("argv")
    if not isinstance(argv, list) or "--fail-on-existing" not in argv:
        raise AuditError("shard runtime did not record --fail-on-existing")
    if "--validate-only" in argv or "--finalize-only" in argv:
        raise AuditError("shard runtime mode is not extraction")

    complete_required = {
        "status": "COMPLETE", "shard": shard_name,
        "manifest_sha256": contract.manifest_sha256,
        "selection_file_sha256": sha256(selection), "wrapper_sha256": extractor_sha,
        "scene_count": contract.canary_scenes, "sample_count": contract.canary_rows,
        "new_forward_count": contract.canary_scenes * 8,
        "output_inventory": str(output_inventory_path.resolve()),
        "source_inventory": str(source_inventory_path.resolve()),
    }
    for key, value in complete_required.items():
        if complete.get(key) != value:
            raise AuditError(f"shard completion {key} mismatch")
    if complete.get("output_inventory_sha256") != sha256(output_inventory_path):
        raise AuditError("shard output inventory file SHA mismatch")
    if complete.get("source_inventory_sha256") != sha256(source_inventory_path):
        raise AuditError("shard source inventory file SHA mismatch")
    output_inventory, _ = parse_inventory(output_inventory_path)
    source_inventory, _ = parse_inventory(source_inventory_path)
    exact_inventory_match(expected_output, output_inventory, "shard output")
    exact_inventory_match(expected_source, source_inventory, "shard source")
    return {
        "ok": True,
        "scene_claims": len(selected_scenes),
        "scene_completions": len(selected_scenes),
        "shard_claim": str(claim_path),
        "shard_completion": str(complete_path),
        "output_inventory": str(output_inventory_path),
        "output_inventory_sha256": sha256(output_inventory_path),
        "source_inventory": str(source_inventory_path),
        "source_inventory_sha256": sha256(source_inventory_path),
    }


def contract_from_bundle(bundle: dict[str, Any]) -> AuditContract:
    contract = AuditContract()
    frozen = bundle.get("contract")
    if not isinstance(frozen, dict):
        raise AuditError("bundle contract is missing")
    expected_aliases = {
        "manifest_sha256": contract.manifest_sha256,
        "canonical_manifest_sha256": contract.manifest_sha256,
        "source_audit_sha256": contract.source_audit_sha256,
        "source_h1_audit_sha256": contract.source_audit_sha256,
        "base_extractor_sha256": contract.base_extractor_sha256,
        "layers": list(contract.layers), "horizons": list(contract.horizons),
        "feature_shape": list(contract.feature_shape), "input_frames": contract.input_frames,
        "timestep": contract.timestep, "seed": contract.base_seed,
        "base_seed": contract.base_seed, "prompt": contract.prompt,
    }
    checked = set()
    for key, expected in expected_aliases.items():
        if key in frozen:
            if frozen[key] != expected:
                raise AuditError(f"bundle contract {key} mismatch")
            checked.add(key)
    required_groups = [
        {"manifest_sha256", "canonical_manifest_sha256"},
        {"source_audit_sha256", "source_h1_audit_sha256"},
        {"layers"}, {"horizons"}, {"feature_shape"}, {"input_frames"},
        {"timestep"}, {"seed", "base_seed"}, {"prompt"}, {"base_extractor_sha256"},
    ]
    if any(not (group & checked) for group in required_groups):
        raise AuditError("bundle contract omits required frozen fields")
    return contract


def resolve_frozen_script(bundle_root: Path, source_hashes: dict[str, Any], basename: str) -> Path:
    matches = [
        (bundle_root / relative).resolve()
        for relative in source_hashes
        if Path(str(relative)).name == basename
    ]
    if len(matches) != 1 or not matches[0].is_file():
        raise AuditError(f"bundle must contain exactly one frozen {basename}")
    return matches[0]


def audit_sco_launch_lineage(
    bundle_path: Path,
    bundle: dict[str, Any],
    out_root: Path,
) -> dict[str, Any]:
    bundle_root = bundle_path.parent.resolve()
    bundle_sha = sha256(bundle_path)
    launcher = bundle_root / "launcher"
    paths = {
        "submit_claim": launcher / "sco_submit.claim.json",
        "submit_status": launcher / "sco_submit.status.json",
        "job_describe": launcher / "sco_job_describe.json",
        "worker_binding": launcher / "sco_worker_binding.json",
        "runner_running": launcher / "sco_extract.running.json",
        "output_claim": out_root / ".sco_launch_claim.json",
    }
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise AuditError(f"missing/unsafe SCO {label}: {path}")
    payloads = {label: read_object(path) for label, path in paths.items()}
    if not all(isinstance(value, dict) for value in payloads.values()):
        raise AuditError("SCO launch-lineage artifact is malformed")
    claim = payloads["submit_claim"]
    status = payloads["submit_status"]
    describe = payloads["job_describe"]
    binding = payloads["worker_binding"]
    running = payloads["runner_running"]
    output_claim = payloads["output_claim"]
    source_hashes = bundle["source_hashes"]
    submitter_sha = source_hashes.get(
        "runtime_tree/scripts/submit_wan_target_h124_sco_canary.py"
    )
    runner_sha = source_hashes.get(
        "runtime_tree/scripts/run_wan_target_h124_sco_canary.py"
    )
    if not isinstance(submitter_sha, str) or not isinstance(runner_sha, str):
        raise AuditError("frozen SCO runner/submitter hashes are absent")
    job_id = status.get("job_id")
    if not isinstance(job_id, str) or re.fullmatch(r"pt-[a-z0-9]+", job_id) is None:
        raise AuditError("invalid submitted SCO job ID")
    status_required = {
        "status": "PASS", "error": None, "returncode": 0, "job_ids": [job_id],
        "bundle": str(bundle_root), "bundle_sha256": bundle_sha,
        "submitter_sha256": submitter_sha,
        "claim_sha256": sha256(paths["submit_claim"]),
        "job_describe": str(paths["job_describe"].resolve()),
        "job_describe_sha256": sha256(paths["job_describe"]),
        "worker_binding": str(paths["worker_binding"].resolve()),
        "worker_binding_sha256": sha256(paths["worker_binding"]),
        "stop_returncode": None, "stop_output": None,
        "log": str((launcher / "sco_submit.log").resolve()),
        "log_sha256": sha256(launcher / "sco_submit.log"),
        "validate_status_sha256": sha256(launcher / "validate.status.json"),
    }
    for key, expected in status_required.items():
        if status.get(key) != expected:
            raise AuditError(f"SCO submit status {key} mismatch")
    claim_required = {
        "status": "CLAIMED", "bundle": str(bundle_root),
        "bundle_sha256": bundle_sha, "submitter_sha256": submitter_sha,
        "container_image_digest_policy": "TAG_PIN_ONLY_ACCEPTED_FOR_THIS_CANARY",
    }
    for key, expected in claim_required.items():
        if claim.get(key) != expected:
            raise AuditError(f"SCO submit claim {key} mismatch")
    if status.get("worker_command") != claim.get("worker_command"):
        raise AuditError("SCO worker command differs between claim and status")
    if status.get("sco_create_argv") != claim.get("sco_create_argv"):
        raise AuditError("SCO create argv differs between claim and status")
    if claim.get("validate_status_sha256") != sha256(launcher / "validate.status.json"):
        raise AuditError("SCO submit claim validation receipt mismatch")
    worker_command = claim.get("worker_command")
    if (
        not isinstance(worker_command, str)
        or claim.get("worker_command_sha256")
        != hashlib.sha256(worker_command.encode()).hexdigest()
    ):
        raise AuditError("SCO submit claim worker command SHA mismatch")
    if describe.get("name") != job_id:
        raise AuditError("SCO describe job ID mismatch")
    sco_contract = bundle.get("sco_execution_contract")
    if not isinstance(sco_contract, dict):
        raise AuditError("bundle SCO execution contract is absent")
    if describe.get("display_name") != sco_contract.get("job_name"):
        raise AuditError("SCO describe job name mismatch")
    if describe.get("workspace_name") != sco_contract.get("workspace"):
        raise AuditError("SCO describe workspace mismatch")
    if describe.get("framework") != "PYTORCH":
        raise AuditError("SCO describe framework mismatch")
    roles = describe.get("roles")
    if not isinstance(roles, list) or len(roles) != 1:
        raise AuditError("SCO describe role mismatch")
    if roles[0].get("startup_script") != claim.get("worker_command"):
        raise AuditError("SCO describe startup command mismatch")
    if roles[0].get("name") != "Worker":
        raise AuditError("SCO describe role name mismatch")
    if roles[0].get("image_path") != sco_contract.get("container_image"):
        raise AuditError("SCO describe image mismatch")
    specs = roles[0].get("resource_spec")
    if (
        not isinstance(specs, list) or len(specs) != 1
        or specs[0].get("name") != sco_contract.get("worker_spec")
        or specs[0].get("replicas") != 1
        or roles[0].get("total_replicas") != 1
    ):
        raise AuditError("SCO describe worker spec mismatch")
    scheduling = describe.get("scheduling")
    if (
        not isinstance(scheduling, dict)
        or scheduling.get("priority") != sco_contract.get("priority")
        or str(scheduling.get("quota_type", "")).lower()
        != str(sco_contract.get("quota_type", "")).lower()
    ):
        raise AuditError("SCO describe scheduling mismatch")
    pool = describe.get("resource_pool")
    if not isinstance(pool, dict) or pool.get("name") != sco_contract.get("aec2_name"):
        raise AuditError("SCO describe resource pool mismatch")
    mount_id, mount_path = str(sco_contract.get("storage_mount", "")).split(":", 1)
    mounts = describe.get("mount")
    if (
        not isinstance(mounts, list) or len(mounts) != 1
        or mounts[0].get("id") != mount_id or mounts[0].get("mount_path") != mount_path
    ):
        raise AuditError("SCO describe storage mount mismatch")
    lme = describe.get("lme")
    if not isinstance(lme, dict) or int(lme.get("max_retries", -1)) != 0:
        raise AuditError("SCO describe retry policy mismatch")
    host = binding.get("worker_hostname")
    if (
        binding.get("job_id") != job_id
        or not isinstance(host, str)
        or re.fullmatch(r"pt-[a-z0-9]+-worker-0", host) is None
        or str(binding.get("worker_status", "")).lower() != "running"
        or binding.get("get_workers_returncode") != 0
    ):
        raise AuditError("SCO scheduler worker binding mismatch")
    receipt = running.get("submission_receipt")
    if not isinstance(receipt, dict):
        raise AuditError("SCO runner lacks submission receipt")
    receipt_required = {
        "job_id": job_id, "worker_hostname": host,
        "submit_status": str(paths["submit_status"].resolve()),
        "submit_status_sha256": sha256(paths["submit_status"]),
        "submit_claim_sha256": sha256(paths["submit_claim"]),
        "job_describe_sha256": sha256(paths["job_describe"]),
        "worker_binding_sha256": sha256(paths["worker_binding"]),
        "container_image_digest_policy": "TAG_PIN_ONLY_ACCEPTED_FOR_THIS_CANARY",
    }
    if receipt != receipt_required:
        raise AuditError("SCO runner submission receipt mismatch")
    running_required = {
        "status": "RUNNING", "action": "extract", "host": host,
        "execution_backend": "sco_h800", "bundle_sha256": bundle_sha,
        "launcher_sha256": runner_sha,
    }
    for key, expected in running_required.items():
        if running.get(key) != expected:
            raise AuditError(f"SCO running marker {key} mismatch")
    expected_environment = {
        "Q00_SCO_EXECUTION_TAG": str(sco_contract.get("execution_tag")),
        "Q00_SCO_JOB_NAME": str(sco_contract.get("job_name")),
        "Q00_SCO_AEC2_NAME": str(sco_contract.get("aec2_name")),
        "Q00_SCO_WORKER_SPEC": str(sco_contract.get("worker_spec")),
    }
    if running.get("sco_environment") != expected_environment:
        raise AuditError("SCO running environment mismatch")
    output_required = {
        "status": "CLAIMED", "run_tag": bundle.get("run_tag"),
        "bundle": str(bundle_root), "bundle_sha256": bundle_sha, "host": host,
        "execution_backend": "sco_h800", "submitted_job_id": job_id,
        "submission_receipt_sha256": sha256(paths["submit_status"]),
        "submission_claim_sha256": sha256(paths["submit_claim"]),
        "job_describe_sha256": sha256(paths["job_describe"]),
        "worker_binding_sha256": sha256(paths["worker_binding"]),
        "runner_running": str(paths["runner_running"].resolve()),
        "runner_running_sha256": sha256(paths["runner_running"]),
    }
    for key, expected in output_required.items():
        if output_claim.get(key) != expected:
            raise AuditError(f"SCO output claim {key} mismatch")
    if output_claim.get("sco_environment") != expected_environment:
        raise AuditError("SCO output claim environment mismatch")
    gpu = output_claim.get("gpu")
    if (
        not isinstance(gpu, dict) or gpu.get("name") != "NVIDIA H800"
        or int(gpu.get("memory_total_mib", 0)) < 81000
        or int(gpu.get("memory_free_mib", 0)) < int(sco_contract.get("required_min_free_mib", 75000))
        or gpu.get("compute_holders") != []
        or gpu.get("torch_cuda_available") is not True
        or gpu.get("torch_device_count") != 1
        or gpu.get("torch_device_name") != "NVIDIA H800"
    ):
        raise AuditError("SCO output claim GPU contract mismatch")
    return {
        "ok": True,
        "job_id": job_id,
        "worker_hostname": host,
        "submit_status_sha256": sha256(paths["submit_status"]),
        "worker_binding_sha256": sha256(paths["worker_binding"]),
        "runner_running_sha256": sha256(paths["runner_running"]),
        "output_claim_sha256": sha256(paths["output_claim"]),
        "container_image_digest_policy": claim["container_image_digest_policy"],
    }


def run_audit(bundle_path: Path, out_root_cli: Path, shard_name: str) -> dict[str, Any]:
    bundle_path = bundle_path.resolve()
    bundle = read_object(bundle_path)
    bundle_root = bundle_path.parent.resolve()
    contract = contract_from_bundle(bundle)
    git_commit = str(bundle.get("git_commit", ""))
    if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise AuditError("bundle git_commit must be a verified full 40-hex commit")
    internal = bundle_internal_hash_audit(bundle_path, bundle)
    source_hashes = bundle["source_hashes"]

    manifest = bundle_root / "inputs/manifest.jsonl"
    selection = bundle_root / "inputs/canary_sample_ids.txt"
    source_audit = bundle_root / "inputs/source_h1_full_audit.json"
    source_inventory_path = bundle_root / "inputs/source_h1_inventory.json"
    input_image_inventory_path = bundle_root / "inputs/canary_input_image_inventory.json"
    model_inventory_path = bundle_root / "inputs/model_snapshot_inventory.json"
    protected_path = bundle_root / "inputs/protected_before.json"
    critical = {
        name: require_bundle_binding(path, source_hashes, bundle_root, name)
        for name, path in {
            "manifest": manifest, "selection": selection, "source_audit": source_audit,
            "source_inventory": source_inventory_path,
            "input_image_inventory": input_image_inventory_path,
            "model_inventory": model_inventory_path, "protected_before": protected_path,
        }.items()
    }
    base_extractor = resolve_frozen_script(bundle_root, source_hashes, "extract_wan_manifest_exact.py")
    extractor = resolve_frozen_script(bundle_root, source_hashes, "extract_wan_target_h124_extend_exact.py")
    auditor_matches = [
        (bundle_root / relative).resolve()
        for relative in source_hashes
        if Path(str(relative)).name == Path(__file__).name
    ]
    if len(auditor_matches) != 1 or sha256(auditor_matches[0]) != sha256(Path(__file__).resolve()):
        raise AuditError("running auditor is not the uniquely frozen bundle auditor")
    if sha256(base_extractor) != contract.base_extractor_sha256:
        raise AuditError("frozen base extractor SHA mismatch")
    if sha256(source_audit) != contract.source_audit_sha256:
        raise AuditError("frozen source h1 audit SHA mismatch")

    source_key, source_root = external_root(bundle, "source_h1_root", "source_root")
    data_key, data_root = external_root(bundle, "data_root")
    if data_key != "external_roots.data_root":
        raise AuditError("data_root must be frozen as external_roots.data_root")
    model_key, model_id = external_root(bundle, "model_id", "model_snapshot", "model_snapshot_root")
    out_key, frozen_out_root = external_root(bundle, "out_root", "output_root")
    repo_key, repo_root = frozen_repo_root(bundle)
    if out_root_cli.resolve() != frozen_out_root:
        raise AuditError("CLI out_root differs from frozen bundle out_root")
    out_root = frozen_out_root
    frozen_shard = bundle.get("shard_name") or (bundle.get("contract") or {}).get("shard_name")
    if frozen_shard != shard_name:
        raise AuditError("CLI shard name differs from frozen bundle shard")
    if not data_root.is_dir() or not source_root.is_dir() or not model_id.is_dir() or not out_root.is_dir():
        raise AuditError("data/source/model/output root is missing")
    if data_root.is_symlink() or out_root.is_symlink() or source_root.is_symlink() or model_id.is_symlink():
        raise AuditError("data/source/model/output top-level root must not be a symlink")
    if source_root == out_root or source_root in out_root.parents or out_root in source_root.parents:
        raise AuditError("output root overlaps immutable source")
    sco_launch = audit_sco_launch_lineage(bundle_path, bundle, out_root)

    _, groups = load_frozen_selection(manifest, selection, contract)
    source_audit_obj = read_object(source_audit)
    required_source_audit = {
        "ok": True, "manifest_sha256": contract.manifest_sha256,
        "target_root": str(source_root), "target_exact_raw": contract.manifest_rows,
        "problem_count": 0,
    }
    for key, value in required_source_audit.items():
        actual = source_audit_obj.get(key)
        if key == "target_root":
            actual = str(Path(str(actual)).resolve())
            value = str(Path(str(value)).resolve())
        if actual != value:
            raise AuditError(f"source h1 full audit {key} mismatch")

    source_before, _ = parse_inventory(source_inventory_path)
    input_images_before, input_images_obj = parse_inventory(input_image_inventory_path)
    input_images_by_scene = validate_canary_input_inventory(
        input_images_before, input_images_obj, data_root, groups, contract
    )
    model_before, model_inventory_obj = parse_inventory(model_inventory_path, model_id)
    if str(Path(str(model_inventory_obj.get("root", ""))).resolve()) != str(model_id):
        raise AuditError("model inventory root mismatch")
    source_after = verify_current_inventory(source_before, "source h1 before/after")
    input_images_after = verify_current_inventory(
        input_images_before, "canary input images before/after"
    )
    model_after = verify_current_inventory(model_before, "model snapshot before/after")
    protected_after = audit_protected(protected_path, repo_root)

    scene_results = [
        audit_scene(
            scene, rows, out_root, source_root, source_audit, manifest, model_id,
            base_extractor, extractor, git_commit, input_images_by_scene[scene], contract,
        )
        for scene, rows in groups.items()
    ]
    selected_source = [item for result in scene_results for item in result["source_inventory"]]
    exact_inventory_match(selected_source, source_before, "frozen source h1")
    shard = audit_claims_and_shard(
        out_root, shard_name, groups, manifest, selection, source_root,
        source_audit, model_id, sha256(extractor), git_commit, contract,
        scene_results,
    )
    # Rehash immutable inputs again after reading all large tensors.
    source_post = verify_current_inventory(source_before, "source h1 post-audit")
    input_images_post = verify_current_inventory(
        input_images_before, "canary input images post-audit"
    )
    model_post = verify_current_inventory(model_before, "model snapshot post-audit")
    protected_post = audit_protected(protected_path, repo_root)
    if source_post["checked"] != source_after["checked"]:
        raise AuditError("source h1 changed during independent audit")
    if input_images_post["checked"] != input_images_after["checked"]:
        raise AuditError("canary input images changed during independent audit")
    if model_post["checked"] != model_after["checked"]:
        raise AuditError("model snapshot changed during independent audit")
    if protected_post["checked"] != protected_after["checked"]:
        raise AuditError("protected files changed during independent audit")

    planned_cli = bundle.get("planned_cli")
    if not isinstance(planned_cli, (str, list)):
        raise AuditError("bundle planned_cli is absent")
    planned_text = " ".join(str(value) for value in planned_cli) if isinstance(planned_cli, list) else planned_cli
    for token in (str(manifest), str(selection), str(data_root), str(source_root), str(out_root), str(model_id), shard_name, "--fail-on-existing"):
        if token not in planned_text:
            raise AuditError(f"planned CLI omits frozen token: {token}")

    collected_output = [
        item for result in scene_results for item in result["output_inventory"]
    ]
    final_output_audit = verify_final_output_inventory(collected_output, out_root)
    sco_launch_post = audit_sco_launch_lineage(bundle_path, bundle, out_root)
    if sco_launch_post != sco_launch:
        raise AuditError("SCO launch-lineage artifacts changed during independent audit")

    return {
        "ok": True,
        "status": "PASS",
        "audit_kind": "independent_wan_target_h124_two_scene_canary_v1",
        "cpu_only": True,
        "train_run": False,
        "eval_run": False,
        "feature_extraction_run": False,
        "bundle": str(bundle_path),
        "bundle_sha256": sha256(bundle_path),
        "auditor": str(Path(__file__).resolve()),
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "contract": asdict(contract),
        "git_commit": git_commit,
        "bundle_internal_source_hashes": internal,
        "critical_bundle_bindings": critical,
        "external_root_keys": {
            "data": data_key, "source": source_key, "model": model_key, "output": out_key,
            "repo": repo_key,
        },
        "repo_root": str(repo_root),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "selection": str(selection),
        "selection_sha256": sha256(selection),
        "selected_scenes": sorted(groups),
        "selected_rows": sum(len(rows) for rows in groups.values()),
        "data_root": str(data_root),
        "canary_input_image_inventory_sha256": sha256(input_image_inventory_path),
        "canary_input_images_pre_post_unchanged": True,
        "canary_input_images_current_audit": input_images_post,
        "source_h1_root": str(source_root),
        "source_h1_inventory_sha256": sha256(source_inventory_path),
        "source_h1_pre_post_unchanged": True,
        "source_h1_current_audit": source_post,
        "model_id": str(model_id),
        "model_inventory_sha256": sha256(model_inventory_path),
        "model_pre_post_unchanged": True,
        "model_current_audit": model_post,
        "protected_before_sha256": sha256(protected_path),
        "protected_pre_post_unchanged": True,
        "protected_current_audit": protected_post,
        "output_root": str(out_root),
        "final_output_inventory_audit": final_output_audit,
        "shard_name": shard_name,
        "sco_launch_lineage": sco_launch,
        "sco_launch_lineage_pre_post_unchanged": True,
        "scene_results": scene_results,
        "shard_provenance": shard,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--shard-name", required=True)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args(argv)


def exclusive_report(path: Path, payload: dict[str, Any]) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise AuditError(f"refusing existing report: {path}") from exc


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = args.report.absolute()
    if report.exists():
        raise SystemExit(f"AUDIT FAILURE: refusing existing report: {report}")
    try:
        payload = run_audit(args.bundle, args.out_root, args.shard_name)
    except Exception as exc:
        payload = {
            "ok": False,
            "status": "FAIL",
            "audit_kind": "independent_wan_target_h124_two_scene_canary_v1",
            "auditor": str(Path(__file__).resolve()),
            "auditor_sha256": sha256(Path(__file__).resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        exclusive_report(report, payload)
        raise SystemExit(f"AUDIT FAILURE: {exc}") from exc
    exclusive_report(report, payload)
    print(json.dumps({"ok": True, "report": str(report), "scenes": payload["selected_scenes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
