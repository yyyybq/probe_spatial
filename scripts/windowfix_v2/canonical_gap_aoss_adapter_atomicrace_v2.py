#!/usr/bin/env python3
"""Canonical-v2 gap validation and receipts for the existing AOSS sync loop.

This module is deliberately *not* an uploader.  ``run_tstep_sync_loop.sh``
continues to own immutable hardlink staging, ``ads-cli sync``, exact upload
coverage checks, commit-before-delete ordering, retry, and cleanup.  The
adapter only teaches that accepted transaction engine how to:

* recognise complete, SHA-valid canonical gap upload units;
* bind every staged payload to its canonical unit and immutable worklist;
* persist the exact successful ``ads-cli`` transaction as a completion
  receipt; and
* safely clean up a local retry residue already covered by that receipt.

The receipt, rather than local ``exists()``, is the resume authority used by
``run_canonical_gap_shard.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.canonical_gap_worklist import (  # noqa: E402
    CANONICAL_PROTOCOL,
    CONTRACT,
    CanonicalGapWorklist,
    canonical_json_sha,
    sha256_file,
    stream_unit_semantic_matches,
    stream_unit_semantic_metadata,
    streaming_gap_unit_key,
    target_gap_unit_key,
)


RECEIPT_CONTRACT = "CANONICAL_V2_GAP_AOSS_COMPLETION_RECEIPT_V1"
VERIFICATION_METHOD = "ADS_CLI_SYNC_EXACT_COVERAGE_V1"


class GapSyncError(RuntimeError):
    """Fail-closed gap publication error."""


def _streaming_feature_filename(
    *, vfm: str, model_id: str, timestep: int, layer: int
) -> str:
    """Return the extractor's real streaming filename for one layer.

    The canonical unit identity is independent of the filename, but the
    existing extractors deliberately retain their historical per-backend
    layouts.  The gap adapter must therefore use the same convention as
    ``features.run_inscene15k`` rather than imposing a diffusion-style
    ``feature_t...`` name on every backend.  Doing the latter made completed
    DINO scenes look incomplete even though their forward pass had succeeded.
    """

    if vfm == "wan":
        prefix = "feature_t2v_14b" if model_id.endswith("14B-Diffusers") else "feature"
        return f"{prefix}_t{timestep}_layer{layer}.sft"
    if vfm == "cogvideox":
        # I2V is the canonical CogVideoX extractor here.  Retain the T2V
        # filename variant for a correctly declared future worklist.
        prefix = "feature"
        if "i2v" not in model_id.lower():
            prefix = f"feature_t2v_{model_id.split('-')[-1]}"
        return f"{prefix}_t{timestep}_layer{layer}.sft"
    if vfm == "dino":
        return "feature.sft"
    if vfm in {"vjepa2", "videomae", "qwen2_5_vl", "bagel", "bagel_hf", "generic_hf"}:
        return f"feature_layer{layer}.sft"
    if vfm == "f3r":
        return f"feature_l{layer}.sft"
    if vfm in {"aether", "opensora", "minimax_h3", "lingbot"}:
        return f"feature_t{timestep}_layer{layer}.sft"
    raise GapSyncError(f"unsupported streaming filename backend: {vfm!r}")


def _json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GapSyncError(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - the message is the contract
        raise GapSyncError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GapSyncError(f"{label} must contain one JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise GapSyncError(f"refusing symlink receipt path: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _completeness_only() -> bool:
    """Reclaim mode: do not re-hash or re-open payloads already written by extract."""
    return os.environ.get("GAP_SYNC_COMPLETENESS_ONLY", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
    }


def _ignore_temps_before_mtime() -> float | None:
    raw = os.environ.get("GAP_SYNC_IGNORE_TEMPS_BEFORE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    from datetime import datetime

    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GapSyncError(f"invalid GAP_SYNC_IGNORE_TEMPS_BEFORE: {raw}") from exc
    return parsed.timestamp()


def _signature(path: Path) -> list[int]:
    stat = path.stat(follow_symlinks=False)
    return [int(stat.st_dev), int(stat.st_ino), int(stat.st_size)]


def _stable_sha(path: Path) -> tuple[str, list[int]]:
    if path.is_symlink() or not path.is_file():
        raise GapSyncError(f"payload must be a regular non-symlink file: {path}")
    before = _signature(path)
    digest = sha256_file(path)
    after = _signature(path)
    if before != after:
        raise GapSyncError(f"payload changed while hashing: {path}")
    return digest, after


def _canonical_identity_sha(value: dict[str, Any]) -> str:
    return canonical_json_sha(value)


def _verified_worklist(
    path: Path,
    expected_sha256: str,
    *,
    mode: str,
    vfm: str,
    model_id: str,
    timestep: int,
    namespace: str,
    layers: Iterable[int],
    protocol_sha256: str,
    prompt: str,
    resize: Iterable[int],
    num_frames: int,
) -> tuple[CanonicalGapWorklist, dict[str, Any]]:
    worklist = CanonicalGapWorklist.load(
        path,
        expected_sha256=expected_sha256,
        mode=mode,
        vfm=vfm,
        model_id=model_id,
        timestep=timestep,
        feature_namespace=namespace,
        output_layers=layers,
        protocol_sha256=protocol_sha256,
        actual_prompt=prompt,
        resize=resize,
        num_frames=num_frames,
    )
    raw = _json(path, "gap worklist")
    return worklist, raw


def expected_upload_units(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return exact relative paths grouped by independently publishable unit."""
    mode = str(raw.get("mode"))
    vfm = str(raw.get("vfm"))
    model_id = str(raw.get("model_id"))
    timestep = int(raw.get("timestep"))
    worklist_sha = str(raw.get("_file_sha256") or "")
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise GapSyncError("gap worklist has no entries list")

    units: dict[str, dict[str, Any]] = {}
    if mode == "streaming_prefix":
        for entry in entries:
            if not isinstance(entry, dict):
                raise GapSyncError("gap worklist entry is not an object")
            scene = str(entry["scene"])
            window = int(entry["window_id"])
            prefix = int(entry["prefix"])
            layers = [int(value) for value in entry["layers"]]
            for layer in layers:
                feature_rel = (
                    f"{scene}/window_{window:04d}/prefix_{prefix - 1:06d}/"
                    f"{_streaming_feature_filename(vfm=vfm, model_id=model_id, timestep=timestep, layer=layer)}"
                )
                unit = streaming_gap_unit_key(scene, window, prefix, layer)
                units[unit] = {
                    "scene": scene,
                    "kind": "streaming_feature",
                    "extraction_units": [unit],
                    "paths": [
                        feature_rel,
                        f"{scene}/prefix_index.npy",
                    ],
                    "primary_paths": [feature_rel],
                    "entry": entry,
                    "layer": layer,
                }
    elif mode == "target_isolated":
        if len(worklist_sha) != 64:
            raise GapSyncError("target expected paths require worklist file SHA256")
        by_scene_layer: dict[tuple[str, int], set[int]] = defaultdict(set)
        for entry in entries:
            if not isinstance(entry, dict):
                raise GapSyncError("gap worklist entry is not an object")
            scene = str(entry["scene"])
            frame_index = int(entry["frame_index"])
            for layer in [int(value) for value in entry["layers"]]:
                by_scene_layer[(scene, layer)].add(frame_index)
        for (scene, layer), frames in sorted(by_scene_layer.items()):
            delta = f"{scene}/.canonical_v2_target_delta/{worklist_sha[:16]}"
            feature_rel = f"{delta}/feature_t{timestep}_layer{layer}.delta.sft"
            index_rel = f"{delta}/target_indices_layer{layer}.npy"
            unit = f"target_delta_layer/{scene}/layer_{layer}"
            units[unit] = {
                "scene": scene,
                "kind": "target_delta_layer",
                "extraction_units": [
                    target_gap_unit_key(scene, frame, layer) for frame in sorted(frames)
                ],
                "paths": [
                    feature_rel,
                    index_rel,
                ],
                "primary_paths": [feature_rel, index_rel],
                "layer": layer,
                "frames": sorted(frames),
            }
    else:
        raise GapSyncError(f"unsupported gap mode: {mode!r}")
    return {
        unit: {
            **meta,
            "paths": sorted(set(meta["paths"])),
            "primary_paths": sorted(set(meta["primary_paths"])),
        }
        for unit, meta in sorted(units.items())
    }


def expected_scene_payloads(raw: dict[str, Any]) -> dict[str, list[str]]:
    """Return the exact relative payload set required for each scene."""

    by_scene: dict[str, set[str]] = defaultdict(set)
    for meta in expected_upload_units(raw).values():
        scene = str(meta["scene"])
        for path in meta["paths"]:
            by_scene[scene].add(str(path))
    return {scene: sorted(paths) for scene, paths in sorted(by_scene.items())}


def expected_extraction_unit_keys(raw: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for meta in expected_upload_units(raw).values():
        result.update(str(unit) for unit in meta["extraction_units"])
    return result


def expected_scene_unit_keys(raw: dict[str, Any]) -> dict[str, set[str]]:
    by_scene: dict[str, set[str]] = defaultdict(set)
    for meta in expected_upload_units(raw).values():
        scene = str(meta["scene"])
        by_scene[scene].update(str(unit) for unit in meta["extraction_units"])
    return dict(by_scene)


def _legacy_expected_scene_payloads(raw: dict[str, Any]) -> dict[str, list[str]]:
    by_scene: dict[str, set[str]] = defaultdict(set)
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise GapSyncError("gap worklist has no entries list")
    mode = str(raw.get("mode"))
    vfm = str(raw.get("vfm"))
    model_id = str(raw.get("model_id"))
    timestep = int(raw.get("timestep"))
    worklist_sha = str(raw.get("_file_sha256") or "")
    for entry in entries:
        if not isinstance(entry, dict):
            raise GapSyncError("gap worklist entry is not an object")
        scene = str(entry["scene"])
        layers = [int(value) for value in entry["layers"]]
        if mode == "streaming_prefix":
            window = int(entry["window_id"])
            prefix = int(entry["prefix"])
            for layer in layers:
                by_scene[scene].add(
                    f"{scene}/window_{window:04d}/prefix_{prefix - 1:06d}/"
                    f"{_streaming_feature_filename(vfm=vfm, model_id=model_id, timestep=timestep, layer=layer)}"
                )
            by_scene[scene].add(f"{scene}/prefix_index.npy")
        elif mode == "target_isolated":
            if len(worklist_sha) != 64:
                raise GapSyncError("target expected paths require worklist file SHA256")
            delta = f"{scene}/.canonical_v2_target_delta/{worklist_sha[:16]}"
            for layer in layers:
                by_scene[scene].add(
                    f"{delta}/feature_t{timestep}_layer{layer}.delta.sft"
                )
                by_scene[scene].add(f"{delta}/target_indices_layer{layer}.npy")
        else:
            raise GapSyncError(f"unsupported gap mode: {mode!r}")
    return {scene: sorted(paths) for scene, paths in sorted(by_scene.items())}


def atomic_temps_for_worklist(
    *, root: Path, worklist_path: Path, worklist_sha256: str
) -> list[str]:
    """Find unfinished atomic writes owned by this exact worklist only.

    GAP namespaces can contain retry residue from an earlier immutable shard.
    Such residue must never be published or deleted by the current shard, but
    it also must not prevent a different, disjoint one-unit transaction from
    reaching its own verified receipt.  We therefore inspect only directories
    which the positive worklist itself owns.  A temp belonging to a selected
    feature/index remains a hard final-stop failure as before.
    """

    root = root.resolve()
    if sha256_file(worklist_path) != worklist_sha256:
        raise GapSyncError("gap worklist SHA mismatch while scanning atomic temps")
    raw = _json(worklist_path, "gap worklist")
    if raw.get("contract") != CONTRACT:
        raise GapSyncError("atomic-temp worklist contract mismatch")
    raw = dict(raw)
    raw["_file_sha256"] = worklist_sha256
    expected = expected_scene_payloads(raw)
    parents: set[Path] = set()
    for relative_paths in expected.values():
        for relative in relative_paths:
            path = root / relative
            try:
                path.parent.resolve().relative_to(root)
            except ValueError as exc:
                raise GapSyncError(f"atomic-temp path escapes feature root: {relative}") from exc
            parents.add(path.parent)
    found: set[str] = set()
    for parent in sorted(parents):
        if not parent.exists():
            continue
        if parent.is_symlink() or not parent.is_dir():
            raise GapSyncError(f"atomic-temp parent is unsafe: {parent}")
        for child in parent.iterdir():
            # All atomic writers in the extractor use a hidden basename and a
            # `.tmp-` suffix.  Do not classify unrelated ordinary files.
            if not child.name.startswith(".") or ".tmp-" not in child.name:
                continue
            # A writer may atomically rename its temp after iterdir().  That
            # disappearance is successful publication, not an unsafe object.
            try:
                child_stat = child.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(child_stat.st_mode):
                raise GapSyncError(f"atomic-temp candidate is unsafe: {child}")
            cutoff = _ignore_temps_before_mtime()
            if cutoff is not None and child_stat.st_mtime < cutoff:
                continue
            try:
                found.add(child.relative_to(root).as_posix())
            except ValueError as exc:  # pragma: no cover - parent containment above
                raise GapSyncError(f"atomic-temp candidate escapes root: {child}") from exc
    return sorted(found)


def _payload_row(
    root: Path,
    path: Path,
    *,
    unit: str,
    layer: int,
    identity: dict[str, Any],
    expected_metadata: dict[str, Any] | None = None,
    safetensor: bool,
    keep_local: bool = False,
) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise GapSyncError(f"payload escapes feature root: {path}") from exc
    sidecar = Path(f"{path}.manifest.json")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise GapSyncError(f"missing payload sidecar: {relative}.manifest.json")
    payload_stat = _signature(path)
    sidecar_stat = _signature(sidecar)
    metadata = _json(sidecar, "payload sidecar")
    if _completeness_only():
        payload_sha = str(metadata.get("sha256") or "")
        if len(payload_sha) != 64:
            raise GapSyncError(f"sidecar lacks extract-time sha256: {relative}")
        if metadata.get("size_bytes") not in (None, payload_stat[2]):
            raise GapSyncError(f"sidecar size mismatch: {relative}")
        sidecar_sha = sha256_file(sidecar)
    else:
        payload_sha, payload_stat = _stable_sha(path)
        sidecar_sha, sidecar_stat = _stable_sha(sidecar)
        if (
            metadata.get("schema_version") != 1
            or metadata.get("file") != path.name
            or metadata.get("size_bytes") != payload_stat[2]
            or metadata.get("sha256") != payload_sha
        ):
            raise GapSyncError(f"sidecar file/size/SHA mismatch: {relative}")
        if expected_metadata:
            bad = [key for key, value in expected_metadata.items() if metadata.get(key) != value]
            if bad:
                raise GapSyncError(f"sidecar identity mismatch {relative}: {','.join(bad)}")
        if safetensor:
            try:
                with safe_open(str(path), framework="np", device="cpu") as handle:
                    if "feat" not in handle.keys():
                        raise GapSyncError(f"safetensors lacks feat tensor: {relative}")
                    shape = list(handle.get_slice("feat").get_shape())
            except GapSyncError:
                raise
            except Exception as exc:
                raise GapSyncError(f"invalid safetensors {relative}: {exc}") from exc
            if metadata.get("tensors", {}).get("feat", {}).get("shape") != shape:
                raise GapSyncError(f"safetensors shape/sidecar mismatch: {relative}")
        if _signature(path) != payload_stat or _signature(sidecar) != sidecar_stat:
            raise GapSyncError(f"payload pair changed while validating: {relative}")
    return {
        "rel": relative,
        "sft_stat": payload_stat,
        "sidecar_stat": sidecar_stat,
        "payload_sha256": payload_sha,
        "sidecar_sha256": sidecar_sha,
        "unit": unit,
        "layer": int(layer),
        "structural": [],
        "keep_local": bool(keep_local),
        "canonical_unit_identity": identity,
        "canonical_unit_identity_sha256": _canonical_identity_sha(identity),
    }


def _prefix_records(path: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], list[Any]]:
    try:
        values = np.load(path, allow_pickle=True).tolist()
    except Exception as exc:
        raise GapSyncError(f"cannot load prefix index {path}: {exc}") from exc
    if not isinstance(values, list):
        raise GapSyncError(f"prefix index is not a record list: {path}")
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, dict):
            raise GapSyncError(f"prefix index contains a non-object record: {path}")
        record = {str(key): value for key, value in raw.items()}
        key = (int(record.get("window_id", -1)), int(record.get("valid_length", -1)))
        if key in records:
            raise GapSyncError(f"duplicate prefix record {key}: {path}")
        records[key] = record
    return records, values


def build_snapshot(
    *,
    root: Path,
    scan_root: Path,
    worklist_path: Path,
    worklist_sha256: str,
    mode: str,
    vfm: str,
    model_id: str,
    timestep: int,
    namespace: str,
    layers: list[int],
    protocol_sha256: str,
    prompt: str,
    resize: list[int],
    num_frames: int,
    max_scenes: int,
    exclude_units: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    root = root.resolve()
    scan_root = scan_root.resolve()
    try:
        scan_root.relative_to(root)
    except ValueError as exc:
        raise GapSyncError(f"scan root escapes feature root: {scan_root}") from exc
    worklist, raw = _verified_worklist(
        worklist_path,
        worklist_sha256,
        mode=mode,
        vfm=vfm,
        model_id=model_id,
        timestep=timestep,
        namespace=namespace,
        layers=layers,
        protocol_sha256=protocol_sha256,
        prompt=prompt,
        resize=resize,
        num_frames=num_frames,
    )
    raw = dict(raw)
    raw["_file_sha256"] = worklist.file_sha256
    expected_units = expected_upload_units(raw)

    snapshot: list[dict[str, Any]] = []
    invalid: list[str] = []
    completed_units = 0
    skipped = set(exclude_units or ())
    for upload_unit, unit_meta in sorted(expected_units.items()):
        if completed_units >= max_scenes:
            break
        if upload_unit in skipped:
            continue
        scene = str(unit_meta["scene"])
        scene_dir = root / scene
        try:
            scene_dir.resolve().relative_to(scan_root)
        except ValueError:
            continue
        expected_paths = [root / value for value in unit_meta["paths"]]
        primary_paths = [root / value for value in unit_meta["primary_paths"]]
        present_payloads = [
            path for path in expected_paths if path.is_file() and not path.is_symlink()
        ]
        present_sidecars = [
            path for path in expected_paths if Path(f"{path}.manifest.json").is_file()
        ]
        present_primary_payloads = [
            path for path in primary_paths if path.is_file() and not path.is_symlink()
        ]
        present_primary_sidecars = [
            path for path in primary_paths if Path(f"{path}.manifest.json").is_file()
        ]
        if not present_primary_payloads and not present_primary_sidecars:
            continue
        if (
            len(present_payloads) != len(expected_paths)
            or len(present_sidecars) != len(expected_paths)
        ):
            invalid.append(
                f"{upload_unit}\tincomplete_gap_unit="
                f"payloads:{len(present_payloads)}/{len(expected_paths)},"
                f"sidecars:{len(present_sidecars)}/{len(expected_paths)}"
            )
            continue

        unit_rows: list[dict[str, Any]] = []
        try:
            if mode == "streaming_prefix":
                entry = unit_meta["entry"]
                key = (int(entry["window_id"]), int(entry["prefix"]))
                layer = int(unit_meta["layer"])
                index_path = root / scene / "prefix_index.npy"
                index_expected = {
                    "feature_namespace": namespace,
                    "canonical_protocol": CANONICAL_PROTOCOL,
                    "canonical_protocol_sha256": protocol_sha256,
                    "canonical_obs_len": 28,
                    "gap_index_contract": "CANONICAL_V2_STATIC_PREFIX_INDEX_V1",
                    "canonical_source_sha256": worklist.canonical_source_sha256,
                }
                path = (
                    root
                    / scene
                    / f"window_{key[0]:04d}"
                    / f"prefix_{key[1] - 1:06d}"
                    / _streaming_feature_filename(
                        vfm=vfm,
                        model_id=model_id,
                        timestep=timestep,
                        layer=layer,
                    )
                )
                if _completeness_only():
                    semantic = {
                        "gap_contract": CONTRACT,
                        "canonical_source_sha256": worklist.canonical_source_sha256,
                        "canonical_protocol_sha256": protocol_sha256,
                        "layer": layer,
                        "gap_record_key": {
                            "scene": scene,
                            "window_id": int(entry["window_id"]),
                            "prefix": int(entry["prefix"]),
                        },
                    }
                else:
                    records, _ = _prefix_records(index_path)
                    record = records.get(key)
                    if record is None:
                        raise GapSyncError(f"prefix index lacks canonical gap key {key}")
                    semantic = stream_unit_semantic_metadata(worklist, scene, record, layer)
                    metadata = _json(Path(f"{path}.manifest.json"), "feature sidecar")
                    if not stream_unit_semantic_matches(metadata, semantic):
                        raise GapSyncError(
                            f"stream semantic identity mismatch: {path.relative_to(root)}"
                        )
                unit_rows.append(
                    _payload_row(
                        root,
                        path,
                        unit=upload_unit,
                        layer=layer,
                        identity={"kind": "streaming_feature", **semantic},
                        expected_metadata={
                            "frame_index_schema": "image_mask_intersection_v1",
                            "vfm": vfm,
                            "model_id": model_id,
                            "mode": mode,
                            "feature_namespace": namespace,
                            "timestep": timestep,
                            "prompt": prompt,
                            "resize": resize,
                            "canonical_protocol": CANONICAL_PROTOCOL,
                            "canonical_protocol_sha256": protocol_sha256,
                            "canonical_obs_len": 28,
                        },
                        safetensor=True,
                    )
                )
                unit_rows.append(
                    _payload_row(
                        root,
                        index_path,
                        unit=upload_unit,
                        layer=-2_147_483_648,
                        identity={
                            "kind": "streaming_prefix_index",
                            "scene": scene,
                            "canonical_source_sha256": worklist.canonical_source_sha256,
                            "canonical_protocol_sha256": protocol_sha256,
                        },
                        expected_metadata=index_expected,
                        safetensor=False,
                        keep_local=True,
                    )
                )
            else:
                layer = int(unit_meta["layer"])
                wanted = [int(value) for value in unit_meta["frames"]]
                delta = root / scene / ".canonical_v2_target_delta" / worklist.file_sha256[:16]
                index_path = delta / f"target_indices_layer{layer}.npy"
                feature_path = delta / f"feature_t{timestep}_layer{layer}.delta.sft"
                actual = np.load(index_path).astype(np.int64).tolist()
                if actual != wanted:
                    raise GapSyncError(
                        f"target delta index mismatch {scene}/layer{layer}: "
                        f"{len(actual)} != {len(wanted)}"
                    )
                index_sha = sha256_file(index_path)
                shared = {
                    "gap_contract": CONTRACT,
                    "gap_worklist_file_sha256": worklist.file_sha256,
                    "gap_worklist_content_sha256": worklist.content_sha256,
                    "canonical_source_sha256": worklist.canonical_source_sha256,
                    "vfm": vfm,
                    "model_id": model_id,
                    "timestep": timestep,
                    "layer": layer,
                    "prompt": prompt,
                    "resize": resize,
                    "num_frames": num_frames,
                    "target_indices_sha256": index_sha,
                }
                identity = {
                    "kind": "target_delta_layer",
                    "scene": scene,
                    "layer": layer,
                    "target_indices": wanted,
                    **shared,
                }
                unit_rows.append(
                    _payload_row(
                        root,
                        index_path,
                        unit=upload_unit,
                        layer=layer,
                        identity={**identity, "payload_role": "target_indices"},
                        expected_metadata=shared,
                        safetensor=False,
                    )
                )
                feature_row = _payload_row(
                    root,
                    feature_path,
                    unit=upload_unit,
                    layer=layer,
                    identity={**identity, "payload_role": "feature"},
                    expected_metadata={
                        **shared,
                        "frame_index_schema": "image_mask_intersection_v1",
                        "mode": mode,
                        "feature_namespace": namespace,
                        "canonical_protocol": CANONICAL_PROTOCOL,
                        "canonical_protocol_sha256": protocol_sha256,
                        "canonical_obs_len": 28,
                    },
                    safetensor=True,
                )
                feature_meta = _json(
                    Path(f"{feature_path}.manifest.json"), "target feature sidecar"
                )
                if feature_meta.get("tensors", {}).get("feat", {}).get("shape", [None])[0] != len(wanted):
                    raise GapSyncError(
                        f"target feature row count mismatch {scene}/layer{layer}"
                    )
                unit_rows.append(feature_row)
        except Exception as exc:
            invalid.append(f"{upload_unit}\t{type(exc).__name__}: {exc}")
            continue
        snapshot.extend(sorted(unit_rows, key=lambda value: value["rel"]))
        completed_units += 1
    return snapshot, invalid


def _receipt_unhashed(receipt: dict[str, Any]) -> dict[str, Any]:
    value = dict(receipt)
    value.pop("content_sha256", None)
    return value


def load_verified_receipt(
    path: Path,
    *,
    worklist_file_sha256: str | None = None,
    shard_content_sha256: str | None = None,
) -> dict[str, Any]:
    receipt = _json(path, "gap AOSS completion receipt")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("contract") != RECEIPT_CONTRACT
        or receipt.get("status") != "PASS_REMOTE_VERIFIED"
        or receipt.get("verification_method") != VERIFICATION_METHOD
        or receipt.get("content_sha256") != canonical_json_sha(_receipt_unhashed(receipt))
    ):
        raise GapSyncError(f"invalid gap AOSS completion receipt: {path}")
    lineage = receipt.get("lineage")
    if not isinstance(lineage, dict):
        raise GapSyncError("gap AOSS receipt has no lineage object")
    if worklist_file_sha256 and lineage.get("worklist_file_sha256") != worklist_file_sha256:
        raise GapSyncError("gap AOSS receipt worklist SHA mismatch")
    if shard_content_sha256 and lineage.get("shard_content_sha256") != shard_content_sha256:
        raise GapSyncError("gap AOSS receipt shard content SHA mismatch")
    if not isinstance(receipt.get("objects"), dict):
        raise GapSyncError("gap AOSS receipt has no objects map")
    return receipt


def receipt_object_matches_identity(
    receipt: dict[str, Any], relative_path: str, expected_identity: dict[str, Any]
) -> bool:
    """Return whether one receipt object proves the supplied unit identity.

    ``expected_identity`` may be a strict semantic subset for target deltas:
    their byte-level target-index digest is intentionally unavailable after
    local cleanup, while the canonical rows, model, layer, worklist and
    protocol remain identity-bearing.  The receipt's own canonical identity
    SHA is always rechecked before that subset comparison is accepted.
    """
    if not isinstance(relative_path, str) or not relative_path or relative_path.startswith("/"):
        return False
    if ".." in Path(relative_path).parts:
        return False
    item = receipt.get("objects", {}).get(relative_path)
    if not isinstance(item, dict):
        return False
    actual = item.get("canonical_unit_identity")
    actual_sha = item.get("canonical_unit_identity_sha256")
    if not isinstance(actual, dict) or actual_sha != canonical_json_sha(actual):
        return False

    def matches(actual_value: Any, expected_value: Any) -> bool:
        if isinstance(expected_value, dict):
            return isinstance(actual_value, dict) and all(
                key in actual_value and matches(actual_value[key], value)
                for key, value in expected_value.items()
            )
        return actual_value == expected_value

    return matches(actual, expected_identity)


def completed_scenes_from_receipt(
    receipt: dict[str, Any], raw_worklist: dict[str, Any], worklist_file_sha256: str
) -> set[str]:
    raw = dict(raw_worklist)
    raw["_file_sha256"] = worklist_file_sha256
    completed_units = completed_units_from_receipt(
        receipt, raw_worklist, worklist_file_sha256
    )
    scene_units = expected_scene_unit_keys(raw)
    return {
        scene
        for scene, units in scene_units.items()
        if units and units.issubset(completed_units)
    }


def completed_units_from_receipt(
    receipt: dict[str, Any], raw_worklist: dict[str, Any], worklist_file_sha256: str
) -> set[str]:
    raw = dict(raw_worklist)
    raw["_file_sha256"] = worklist_file_sha256
    expected = expected_upload_units(raw)
    objects = receipt.get("objects", {})
    completed: set[str] = set()
    for meta in expected.values():
        paths = [str(path) for path in meta["paths"]]
        if paths and all(path in objects for path in paths):
            completed.update(str(unit) for unit in meta["extraction_units"])
    return completed


def _verified_shard(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_file_sha256:
        raise GapSyncError("gap shard file SHA mismatch")
    shard = _json(path, "gap shard")
    content = shard.get("content_sha256")
    unhashed = dict(shard)
    unhashed.pop("content_sha256", None)
    if content != canonical_json_sha(unhashed):
        raise GapSyncError("gap shard content SHA mismatch")
    return shard


def commit_receipt(args: argparse.Namespace) -> None:
    shard = _verified_shard(args.shard, args.shard_file_sha256)
    if shard.get("content_sha256") != args.shard_content_sha256:
        raise GapSyncError("runtime gap shard content binding mismatch")
    worklist_path = Path(str(shard["worklist_path"]))
    if sha256_file(worklist_path) != shard.get("worklist_file_sha256"):
        raise GapSyncError("shard-bound worklist changed before receipt commit")
    worklist_raw = _json(worklist_path, "gap worklist")
    snapshot_rows = [
        json.loads(line)
        for line in args.snapshot.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    committed = {
        line.strip()
        for line in args.keys.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = {str(row["rel"]): row for row in snapshot_rows}
    if not committed or not committed.issubset(rows):
        raise GapSyncError("receipt commit keys are empty or escape the validated snapshot")
    if args.ads_rc != 0 or args.ads_failed != 0 or args.ads_found != args.ads_expected:
        raise GapSyncError("cannot commit receipt without exact successful ads-cli coverage")

    lineage = {
        "plan_id": shard.get("plan_id"),
        "shard_id": shard.get("shard_id"),
        "shard_file_sha256": args.shard_file_sha256,
        "shard_content_sha256": args.shard_content_sha256,
        "worklist_file_sha256": shard["worklist_file_sha256"],
        "worklist_content_sha256": shard["worklist_content_sha256"],
        "canonical_source_sha256": shard["canonical_source_sha256"],
        "protocol_file_sha256": shard["protocol_file_sha256"],
        "protocol_sha256": shard["protocol_sha256"],
        "feature_namespace": shard["feature_namespace"],
        "mode": shard["mode"],
        "vfm": shard["vfm"],
        "model_id": shard["model_id"],
        "timestep": int(shard["timestep"]),
        "source_bindings": shard["source_bindings"],
    }
    objects: dict[str, Any] = {}
    transactions: list[dict[str, Any]] = []
    if args.receipt.exists():
        previous = load_verified_receipt(
            args.receipt,
            worklist_file_sha256=shard["worklist_file_sha256"],
            shard_content_sha256=args.shard_content_sha256,
        )
        if previous.get("lineage") != lineage:
            raise GapSyncError("existing receipt lineage differs from current shard")
        objects.update(previous["objects"])
        transactions.extend(previous.get("transactions", []))
    for relative in sorted(committed):
        row = rows[relative]
        candidate = {
            "canonical_unit_identity": row["canonical_unit_identity"],
            "canonical_unit_identity_sha256": row["canonical_unit_identity_sha256"],
            "payload_sha256": row["payload_sha256"],
            "payload_size_bytes": int(row["sft_stat"][2]),
            "sidecar_sha256": row["sidecar_sha256"],
            "sidecar_size_bytes": int(row["sidecar_stat"][2]),
            "keep_local": bool(row.get("keep_local", False)),
        }
        old = objects.get(relative)
        if old is not None and old != candidate:
            raise GapSyncError(f"remote receipt owner conflict for {relative}")
        objects[relative] = candidate
    transaction = {
        "round": args.round,
        "ads_cli_rc": args.ads_rc,
        "ads_cli_found": args.ads_found,
        "ads_cli_expected": args.ads_expected,
        "ads_cli_failed": args.ads_failed,
        "staging_file_count": args.ads_expected,
        "committed_payload_count": len(committed),
        "committed_payloads_sha256": hashlib.sha256(
            "".join(f"{value}\n" for value in sorted(committed)).encode()
        ).hexdigest(),
    }
    transactions.append(transaction)
    done_sha = sha256_file(args.done_index)
    done_count = len(
        [line for line in args.done_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    )
    provisional = {
        "schema_version": 1,
        "contract": RECEIPT_CONTRACT,
        "status": "PASS_REMOTE_VERIFIED",
        "verification_method": VERIFICATION_METHOD,
        "lineage": lineage,
        "remote": {"base_uri": args.remote_base_uri},
        "done_index": {
            "path": str(args.done_index.resolve()),
            "sha256": done_sha,
            "entry_count": done_count,
        },
        "objects": dict(sorted(objects.items())),
        "transactions": transactions,
    }
    provisional["completed_scenes"] = sorted(
        completed_scenes_from_receipt(
            provisional, worklist_raw, shard["worklist_file_sha256"]
        )
    )
    provisional["completed_units"] = sorted(
        completed_units_from_receipt(
            provisional, worklist_raw, shard["worklist_file_sha256"]
        )
    )
    provisional["content_sha256"] = canonical_json_sha(provisional)
    _atomic_json(args.receipt, provisional)


def cleanup_confirmed(
    *, root: Path, receipt_path: Path, worklist_sha256: str, shard_content_sha256: str
) -> dict[str, int]:
    if not receipt_path.exists():
        return {"payloads": 0, "sidecars": 0, "already_absent": 0}
    receipt = load_verified_receipt(
        receipt_path,
        worklist_file_sha256=worklist_sha256,
        shard_content_sha256=shard_content_sha256,
    )
    root = root.resolve()
    pending: list[tuple[Path, Path, dict[str, Any]]] = []
    already_absent = 0
    # Validate *all* local residues before deleting the first byte.
    for relative, item in receipt["objects"].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise GapSyncError(f"unsafe receipt object path: {relative}")
        payload = root / relative_path
        sidecar = Path(f"{payload}.manifest.json")
        if not payload.exists() and not sidecar.exists():
            already_absent += 1
            continue
        if payload.exists():
            if _completeness_only():
                if _signature(payload)[2] != item.get("payload_size_bytes"):
                    raise GapSyncError(f"receipt-confirmed local payload size changed: {relative}")
            else:
                digest, stat = _stable_sha(payload)
                if digest != item.get("payload_sha256") or stat[2] != item.get("payload_size_bytes"):
                    raise GapSyncError(f"receipt-confirmed local payload changed: {relative}")
        if sidecar.exists():
            if _completeness_only():
                if _signature(sidecar)[2] != item.get("sidecar_size_bytes"):
                    raise GapSyncError(f"receipt-confirmed local sidecar size changed: {relative}")
            else:
                digest, stat = _stable_sha(sidecar)
                if digest != item.get("sidecar_sha256") or stat[2] != item.get("sidecar_size_bytes"):
                    raise GapSyncError(f"receipt-confirmed local sidecar changed: {relative}")
        pending.append((payload, sidecar, item))
    deleted_payloads = 0
    deleted_sidecars = 0
    for payload, sidecar, _item in pending:
        if bool(_item.get("keep_local", False)):
            continue
        if payload.exists():
            payload.unlink()
            deleted_payloads += 1
        if sidecar.exists():
            sidecar.unlink()
            deleted_sidecars += 1
    return {
        "payloads": deleted_payloads,
        "sidecars": deleted_sidecars,
        "already_absent": already_absent,
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--worklist-sha256", required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot")
    _common(snapshot)
    snapshot.add_argument("--scan-root", type=Path, required=True)
    snapshot.add_argument("--mode", required=True)
    snapshot.add_argument("--vfm", required=True)
    snapshot.add_argument("--model-id", required=True)
    snapshot.add_argument("--timestep", type=int, required=True)
    snapshot.add_argument("--namespace", required=True)
    snapshot.add_argument("--layers", nargs="+", type=int, required=True)
    snapshot.add_argument("--protocol-sha256", required=True)
    snapshot.add_argument("--prompt", required=True)
    snapshot.add_argument("--resize", nargs=2, type=int, required=True)
    snapshot.add_argument("--num-frames", type=int, required=True)
    snapshot.add_argument("--max-scenes", type=int, default=4)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--invalid-output", type=Path, required=True)

    commit = subparsers.add_parser("commit-receipt")
    commit.add_argument("--receipt", type=Path, required=True)
    commit.add_argument("--shard", type=Path, required=True)
    commit.add_argument("--shard-file-sha256", required=True)
    commit.add_argument("--shard-content-sha256", required=True)
    commit.add_argument("--snapshot", type=Path, required=True)
    commit.add_argument("--keys", type=Path, required=True)
    commit.add_argument("--done-index", type=Path, required=True)
    commit.add_argument("--round", required=True)
    commit.add_argument("--ads-rc", type=int, required=True)
    commit.add_argument("--ads-found", type=int, required=True)
    commit.add_argument("--ads-expected", type=int, required=True)
    commit.add_argument("--ads-failed", type=int, required=True)
    commit.add_argument("--remote-base-uri", required=True)

    cleanup = subparsers.add_parser("cleanup-confirmed")
    _common(cleanup)
    cleanup.add_argument("--receipt", type=Path, required=True)
    cleanup.add_argument("--shard-content-sha256", required=True)

    atomic_temps = subparsers.add_parser("atomic-temps")
    _common(atomic_temps)
    atomic_temps.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            rows, invalid = build_snapshot(
                root=args.root,
                scan_root=args.scan_root,
                worklist_path=args.worklist,
                worklist_sha256=args.worklist_sha256,
                mode=args.mode,
                vfm=args.vfm,
                model_id=args.model_id,
                timestep=args.timestep,
                namespace=args.namespace,
                layers=args.layers,
                protocol_sha256=args.protocol_sha256,
                prompt=args.prompt,
                resize=args.resize,
                num_frames=args.num_frames,
                max_scenes=args.max_scenes,
            )
            args.output.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            args.invalid_output.write_text(
                "".join(f"{line}\n" for line in invalid), encoding="utf-8"
            )
        elif args.command == "commit-receipt":
            commit_receipt(args)
        elif args.command == "atomic-temps":
            temps = atomic_temps_for_worklist(
                root=args.root,
                worklist_path=args.worklist,
                worklist_sha256=args.worklist_sha256,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{args.output.name}.tmp-", dir=args.output.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write("".join(f"{value}\n" for value in temps))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, args.output)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        else:
            result = cleanup_confirmed(
                root=args.root,
                receipt_path=args.receipt,
                worklist_sha256=args.worklist_sha256,
                shard_content_sha256=args.shard_content_sha256,
            )
            print(json.dumps(result, sort_keys=True))
    except GapSyncError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
