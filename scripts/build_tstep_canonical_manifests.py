#!/usr/bin/env python3
"""Build and audit canonical InsScene timestep-probing manifests.

The canonical protocol samples one motion-uniform 28-frame window per
``(scene, window_id)`` and derives every prefix from that same window.  This
prevents a shared ``window_id`` from silently referring to different source
frames at different prefix lengths.

The current reproducible source for those windows is the frozen prefix-24
manifest: it was generated with ``24 + max(1, 2, 4) == 28`` observations.  A
future extractor may provide an equivalent manifest made from canonical
``prefix_index.npy`` records; this tool intentionally consumes JSONL so it can
run on a login node without NumPy or feature downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


PROTOCOL_ID = "inscene_streaming_canonical_v2_obs28_h124"
CANONICAL_OBS_LEN = 28
ACTION_HORIZONS = (1, 2, 4)
DEFAULT_PREFIXES = (8, 12, 16, 24)
MANIFEST_NAME = "inscene_c_h124_prefix{prefix}.jsonl"
SUMMARY_NAME = "canonical_summary.json"
STREAMING_FEATURE_ROOT = (
    "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
)
TARGET_FEATURE_ROOT = (
    "aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
)


class ManifestError(ValueError):
    """Raised when a manifest violates the canonical temporal contract."""


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        items = value.replace(",", " ").split()
    else:
        items = value
    parsed = tuple(sorted({int(item) for item in items}))
    if not parsed:
        raise ManifestError("prefix list must not be empty")
    if any(prefix <= 0 or prefix + max(ACTION_HORIZONS) > CANONICAL_OBS_LEN for prefix in parsed):
        raise ManifestError(
            f"every prefix must satisfy 0 < prefix <= {CANONICAL_OBS_LEN - max(ACTION_HORIZONS)}"
        )
    return parsed


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot open manifest {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ManifestError(f"{path}:{line_number}: row must be a JSON object")
            yield line_number, row


def _scene_id(row: dict[str, Any], where: str) -> str:
    value = row.get("scene_id")
    if value is None and row.get("scene_dir"):
        value = Path(str(row["scene_dir"]).rstrip("/")).name
    scene = str(value or "").strip()
    if not scene:
        raise ManifestError(f"{where}: missing scene_id/scene_dir")
    return scene


def _int_list(row: dict[str, Any], field: str, where: str) -> list[int]:
    value = row.get(field)
    if not isinstance(value, list):
        raise ManifestError(f"{where}: {field} must be a list")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{where}: {field} must contain integers") from exc


def _validate_strictly_increasing(indices: list[int], where: str) -> None:
    if any(right <= left for left, right in zip(indices, indices[1:])):
        raise ManifestError(f"{where}: streaming_window_indices must be strictly increasing")


def load_split_map(paths: Sequence[Path]) -> dict[str, str]:
    """Load a scene-level split map, rejecting cross-manifest disagreement."""

    if not paths:
        raise ManifestError("at least one frozen split manifest is required")
    splits: dict[str, str] = {}
    for path in paths:
        for line_number, row in _read_jsonl(path):
            where = f"{path}:{line_number}"
            scene = _scene_id(row, where)
            split = str(row.get("split", "")).strip()
            if not split:
                raise ManifestError(f"{where}: missing split for scene {scene}")
            previous = splits.get(scene)
            if previous is not None and previous != split:
                raise ManifestError(
                    f"{where}: split mismatch for scene {scene}: {previous!r} != {split!r}"
                )
            splits[scene] = split
    if not splits:
        raise ManifestError("frozen split manifests contain no scenes")
    return splits


def load_canonical_windows(
    path: Path,
    split_map: dict[str, str],
    expected_shared_units: int = 0,
) -> list[dict[str, Any]]:
    """Load and validate one 28-frame canonical row per scene/window."""

    records: list[dict[str, Any]] = []
    unit_locations: dict[tuple[str, int], str] = {}
    sample_locations: dict[str, str] = {}
    scene_splits: dict[str, str] = {}

    for line_number, source_row in _read_jsonl(path):
        where = f"{path}:{line_number}"
        scene = _scene_id(source_row, where)
        try:
            window_id = int(source_row["streaming_window_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"{where}: invalid streaming_window_id") from exc
        if window_id < 0:
            raise ManifestError(f"{where}: streaming_window_id must be non-negative")

        window_indices = _int_list(source_row, "streaming_window_indices", where)
        if len(window_indices) != CANONICAL_OBS_LEN:
            raise ManifestError(
                f"{where}: expected {CANONICAL_OBS_LEN} canonical window indices, "
                f"found {len(window_indices)}"
            )
        _validate_strictly_increasing(window_indices, where)

        protocol = source_row.get("canonical_protocol")
        if protocol is not None and protocol != PROTOCOL_ID:
            raise ManifestError(f"{where}: incompatible canonical_protocol={protocol!r}")
        obs_len = source_row.get("canonical_obs_len")
        if obs_len is not None and int(obs_len) != CANONICAL_OBS_LEN:
            raise ManifestError(f"{where}: incompatible canonical_obs_len={obs_len!r}")
        sampling = str(source_row.get("streaming_sampling", "motion_uniform"))
        if sampling != "motion_uniform":
            raise ManifestError(f"{where}: expected motion_uniform sampling, found {sampling!r}")

        source_prefix = source_row.get("streaming_indices")
        if source_prefix is not None:
            prefix_indices = _int_list(source_row, "streaming_indices", where)
            if not prefix_indices or prefix_indices != window_indices[: len(prefix_indices)]:
                raise ManifestError(
                    f"{where}: source streaming_indices are not a prefix of the canonical window"
                )
            source_tail = int(source_row.get("streaming_tail", len(prefix_indices) - 1))
            if source_tail != len(prefix_indices) - 1:
                raise ManifestError(f"{where}: source streaming_tail does not match prefix length")
            source_tail_raw = int(source_row.get("streaming_tail_raw", prefix_indices[-1]))
            if source_tail_raw != prefix_indices[-1]:
                raise ManifestError(f"{where}: source streaming_tail_raw is misaligned")

        unit = (scene, window_id)
        previous_location = unit_locations.get(unit)
        if previous_location is not None:
            raise ManifestError(
                f"{where}: duplicate canonical unit {unit}; first seen at {previous_location}"
            )
        unit_locations[unit] = where

        source_sample = str(source_row.get("sample_id", "")).strip()
        if source_sample:
            previous_location = sample_locations.get(source_sample)
            if previous_location is not None:
                raise ManifestError(
                    f"{where}: duplicate source sample_id {source_sample!r}; "
                    f"first seen at {previous_location}"
                )
            sample_locations[source_sample] = where

        if scene not in split_map:
            raise ManifestError(f"{where}: scene {scene} is absent from frozen split manifests")
        split = split_map[scene]
        previous_split = scene_splits.get(scene)
        if previous_split is not None and previous_split != split:
            raise ManifestError(f"{where}: inconsistent split for scene {scene}")
        scene_splits[scene] = split

        row = dict(source_row)
        row["scene_id"] = scene
        row["split"] = split
        row["streaming_window_id"] = window_id
        row["streaming_window_indices"] = window_indices
        row["streaming_sampling"] = sampling
        # These legacy roots describe prefix-specific DINO caches and are not
        # valid provenance for the canonical timestep extraction campaign.
        row.pop("streaming_feature_root", None)
        row.pop("target_feature_root", None)
        records.append(row)

    records.sort(key=lambda row: (str(row["scene_id"]), int(row["streaming_window_id"])))
    if not records:
        raise ManifestError(f"canonical source manifest is empty: {path}")
    if expected_shared_units and len(records) != expected_shared_units:
        raise ManifestError(
            f"canonical universe has {len(records)} units, expected {expected_shared_units}"
        )
    return records


def _canonical_unit_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": str(row["scene_id"]),
        "split": str(row["split"]),
        "streaming_window_id": int(row["streaming_window_id"]),
        "streaming_window_indices": [int(value) for value in row["streaming_window_indices"]],
    }


def universe_sha256(records: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update((_compact_json(_canonical_unit_payload(row)) + "\n").encode("utf-8"))
    return digest.hexdigest()


def make_prefix_row(
    canonical_row: dict[str, Any],
    prefix: int,
    streaming_feature_root: str | None = STREAMING_FEATURE_ROOT,
    target_feature_root: str | None = TARGET_FEATURE_ROOT,
) -> dict[str, Any]:
    row = dict(canonical_row)
    scene = str(row["scene_id"])
    window_id = int(row["streaming_window_id"])
    window_indices = [int(value) for value in row["streaming_window_indices"]]
    prefix_indices = window_indices[:prefix]
    row.update(
        {
            "sample_id": f"{scene}_w{window_id:03d}_p{prefix}",
            "prefix_length": prefix,
            "streaming_tail": prefix - 1,
            "streaming_tail_raw": prefix_indices[-1],
            "streaming_indices": prefix_indices,
            "streaming_window_indices": window_indices,
            "canonical_protocol": PROTOCOL_ID,
            "canonical_obs_len": CANONICAL_OBS_LEN,
            "target_horizons": list(ACTION_HORIZONS),
            "include_anchors": True,
        }
    )
    row.pop("streaming_feature_root", None)
    row.pop("target_feature_root", None)
    if streaming_feature_root:
        row["streaming_feature_root"] = streaming_feature_root
    if target_feature_root:
        row["target_feature_root"] = target_feature_root
    return row


def _manifest_digest(
    records: Sequence[dict[str, Any]],
    prefix: int,
    output_path: Path | None,
    streaming_feature_root: str | None,
    target_feature_root: str | None,
) -> str:
    digest = hashlib.sha256()
    handle = None
    temporary_path: Path | None = None
    try:
        if output_path is not None:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.tmp-", dir=str(output_path.parent)
            )
            temporary_path = Path(temporary_name)
            handle = os.fdopen(fd, "wb")
        for record in records:
            row = make_prefix_row(
                record,
                prefix,
                streaming_feature_root=streaming_feature_root,
                target_feature_root=target_feature_root,
            )
            payload = (_compact_json(row) + "\n").encode("utf-8")
            digest.update(payload)
            if handle is not None:
                handle.write(payload)
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None
            os.replace(temporary_path, output_path)
            temporary_path = None
    finally:
        if handle is not None:
            handle.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _summary_hash(summary_without_hash: dict[str, Any]) -> str:
    return _sha256_bytes(_compact_json(summary_without_hash).encode("utf-8"))


def build_manifests(
    *,
    canonical_manifest: Path,
    split_manifests: Sequence[Path],
    output_dir: Path,
    prefixes: Sequence[int] = DEFAULT_PREFIXES,
    expected_shared_units: int = 2721,
    dry_run: bool = False,
    force: bool = False,
    streaming_feature_root: str | None = STREAMING_FEATURE_ROOT,
    target_feature_root: str | None = TARGET_FEATURE_ROOT,
) -> dict[str, Any]:
    prefixes = _parse_ints(prefixes)
    if streaming_feature_root != STREAMING_FEATURE_ROOT:
        raise ManifestError(
            f"streaming_feature_root must be seeded-v1 root {STREAMING_FEATURE_ROOT!r}"
        )
    if target_feature_root != TARGET_FEATURE_ROOT:
        raise ManifestError(
            f"target_feature_root must be seeded-v1 root {TARGET_FEATURE_ROOT!r}"
        )
    split_map = load_split_map(split_manifests)
    records = load_canonical_windows(
        canonical_manifest, split_map, expected_shared_units=expected_shared_units
    )
    shared_hash = universe_sha256(records)
    split_counts = dict(sorted(Counter(str(row["split"]) for row in records).items()))
    scene_count = len({str(row["scene_id"]) for row in records})

    targets: list[Path] = []
    for prefix in prefixes:
        manifest_path = output_dir / MANIFEST_NAME.format(prefix=prefix)
        targets.extend([manifest_path, manifest_path.with_suffix(manifest_path.suffix + ".meta.json")])
    targets.append(output_dir / SUMMARY_NAME)
    if not dry_run:
        existing = [path for path in targets if path.exists()]
        if existing and not force:
            raise ManifestError(
                "refusing to overwrite canonical outputs without --force: "
                + ", ".join(str(path) for path in existing[:4])
            )
        output_dir.mkdir(parents=True, exist_ok=True)

    source_info = {
        "canonical_manifest": canonical_manifest.name,
        "canonical_manifest_sha256": sha256_file(canonical_manifest),
        "frozen_split_manifests": [
            {"name": path.name, "sha256": sha256_file(path)}
            for path in sorted(split_manifests, key=lambda item: item.name)
        ],
    }
    manifest_summaries: dict[str, dict[str, Any]] = {}
    for prefix in prefixes:
        filename = MANIFEST_NAME.format(prefix=prefix)
        manifest_path = output_dir / filename
        checksum = _manifest_digest(
            records,
            prefix,
            None if dry_run else manifest_path,
            streaming_feature_root,
            target_feature_root,
        )
        metadata = {
            "manifest_version": 2,
            "canonical_protocol": PROTOCOL_ID,
            "canonical_obs_len": CANONICAL_OBS_LEN,
            "action_horizons": list(ACTION_HORIZONS),
            "prefix_length": prefix,
            "shared_unit_count": len(records),
            "num_samples": len(records),
            "num_scenes": scene_count,
            "split_counts": split_counts,
            "universe_sha256": shared_hash,
            "sha256": checksum,
            "streaming_feature_root": streaming_feature_root,
            "target_feature_root": target_feature_root,
            "source": source_info,
        }
        meta_filename = filename + ".meta.json"
        if not dry_run:
            _write_json_atomic(output_dir / meta_filename, metadata)
        manifest_summaries[str(prefix)] = {
            "filename": filename,
            "meta_filename": meta_filename,
            "rows": len(records),
            "scenes": scene_count,
            "split_counts": split_counts,
            "sha256": checksum,
        }

    summary: dict[str, Any] = {
        "summary_version": 1,
        "canonical_protocol": PROTOCOL_ID,
        "canonical_obs_len": CANONICAL_OBS_LEN,
        "action_horizons": list(ACTION_HORIZONS),
        "prefixes": list(prefixes),
        "shared_unit_count": len(records),
        "logical_sample_count": len(records) * len(prefixes),
        "scene_count": scene_count,
        "split_counts": split_counts,
        "universe_sha256": shared_hash,
        "streaming_feature_root": streaming_feature_root,
        "target_feature_root": target_feature_root,
        "manifests": manifest_summaries,
        "source": source_info,
    }
    summary["summary_sha256"] = _summary_hash(summary)
    if not dry_run:
        _write_json_atomic(output_dir / SUMMARY_NAME, summary)
    return summary


def _validate_prefix_row(
    row: dict[str, Any], prefix: int, where: str
) -> tuple[tuple[str, int], dict[str, Any]]:
    scene = _scene_id(row, where)
    try:
        window_id = int(row["streaming_window_id"])
        obs_len = int(row["canonical_obs_len"])
        tail = int(row["streaming_tail"])
        tail_raw = int(row["streaming_tail_raw"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"{where}: invalid canonical integer field") from exc
    if row.get("canonical_protocol") != PROTOCOL_ID:
        raise ManifestError(f"{where}: canonical_protocol mismatch")
    if obs_len != CANONICAL_OBS_LEN:
        raise ManifestError(f"{where}: canonical_obs_len mismatch")
    if row.get("streaming_feature_root") != STREAMING_FEATURE_ROOT:
        raise ManifestError(
            f"{where}: streaming_feature_root must be {STREAMING_FEATURE_ROOT!r}, "
            f"got {row.get('streaming_feature_root')!r}"
        )
    if row.get("target_feature_root") != TARGET_FEATURE_ROOT:
        raise ManifestError(
            f"{where}: target_feature_root must be {TARGET_FEATURE_ROOT!r}, "
            f"got {row.get('target_feature_root')!r}"
        )
    window_indices = _int_list(row, "streaming_window_indices", where)
    prefix_indices = _int_list(row, "streaming_indices", where)
    if len(window_indices) != CANONICAL_OBS_LEN:
        raise ManifestError(f"{where}: canonical window length is {len(window_indices)}, expected 28")
    _validate_strictly_increasing(window_indices, where)
    if prefix_indices != window_indices[:prefix]:
        raise ManifestError(f"{where}: streaming_indices do not match canonical prefix {prefix}")
    if tail != prefix - 1 or tail_raw != window_indices[prefix - 1]:
        raise ManifestError(f"{where}: streaming tail is misaligned for prefix {prefix}")
    expected_sample = f"{scene}_w{window_id:03d}_p{prefix}"
    if row.get("sample_id") != expected_sample:
        raise ManifestError(
            f"{where}: sample_id {row.get('sample_id')!r} != {expected_sample!r}"
        )
    split = str(row.get("split", "")).strip()
    if not split:
        raise ManifestError(f"{where}: missing split")
    common = _canonical_unit_payload(
        {
            "scene_id": scene,
            "split": split,
            "streaming_window_id": window_id,
            "streaming_window_indices": window_indices,
        }
    )
    return (scene, window_id), common


def audit_manifests(
    *,
    manifest_dir: Path,
    prefixes: Sequence[int] = DEFAULT_PREFIXES,
    expected_shared_units: int = 2721,
) -> dict[str, Any]:
    """Audit row alignment, sidecars, hashes, and the deterministic summary."""

    prefixes = _parse_ints(prefixes)
    paths = [manifest_dir / MANIFEST_NAME.format(prefix=prefix) for prefix in prefixes]
    for path in paths:
        if not path.is_file():
            raise ManifestError(f"missing canonical manifest: {path}")

    iterators = [_read_jsonl(path) for path in paths]
    seen_by_prefix = [set() for _ in prefixes]
    sample_ids_by_prefix = [set() for _ in prefixes]
    scene_split: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    shared_records: list[dict[str, Any]] = []
    previous_unit: tuple[str, int] | None = None

    for row_number, group in enumerate(itertools.zip_longest(*iterators), 1):
        if any(item is None for item in group):
            counts = [len(seen) for seen in seen_by_prefix]
            raise ManifestError(f"manifest row-count mismatch near row {row_number}: {counts}")
        common_reference: dict[str, Any] | None = None
        reference_unit: tuple[str, int] | None = None
        for index, item in enumerate(group):
            assert item is not None
            line_number, row = item
            prefix = prefixes[index]
            path = paths[index]
            where = f"{path}:{line_number}"
            unit, common = _validate_prefix_row(row, prefix, where)
            if unit in seen_by_prefix[index]:
                raise ManifestError(f"{where}: duplicate canonical unit {unit}")
            seen_by_prefix[index].add(unit)
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids_by_prefix[index]:
                raise ManifestError(f"{where}: duplicate sample_id {sample_id!r}")
            sample_ids_by_prefix[index].add(sample_id)
            if common_reference is None:
                common_reference = common
                reference_unit = unit
            elif unit != reference_unit or common != common_reference:
                raise ManifestError(
                    f"{where}: cross-prefix canonical window misalignment at row {row_number}"
                )

        assert common_reference is not None and reference_unit is not None
        if previous_unit is not None and reference_unit <= previous_unit:
            raise ManifestError(
                f"canonical rows are not in deterministic order: {reference_unit} after {previous_unit}"
            )
        previous_unit = reference_unit
        scene = reference_unit[0]
        split = str(common_reference["split"])
        old_split = scene_split.get(scene)
        if old_split is not None and old_split != split:
            raise ManifestError(f"scene {scene} changes split from {old_split!r} to {split!r}")
        scene_split[scene] = split
        split_counts[split] += 1
        shared_records.append(common_reference)

    shared_count = len(shared_records)
    if expected_shared_units and shared_count != expected_shared_units:
        raise ManifestError(
            f"canonical universe has {shared_count} units, expected {expected_shared_units}"
        )
    shared_hash = universe_sha256(shared_records)
    manifest_summaries: dict[str, dict[str, Any]] = {}
    sorted_split_counts = dict(sorted(split_counts.items()))
    for prefix, path in zip(prefixes, paths):
        checksum = sha256_file(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not meta_path.is_file():
            raise ManifestError(f"missing metadata sidecar: {meta_path}")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"invalid metadata sidecar {meta_path}: {exc}") from exc
        expected_metadata = {
            "canonical_protocol": PROTOCOL_ID,
            "canonical_obs_len": CANONICAL_OBS_LEN,
            "prefix_length": prefix,
            "shared_unit_count": shared_count,
            "num_samples": shared_count,
            "num_scenes": len(scene_split),
            "split_counts": sorted_split_counts,
            "universe_sha256": shared_hash,
            "sha256": checksum,
            "streaming_feature_root": STREAMING_FEATURE_ROOT,
            "target_feature_root": TARGET_FEATURE_ROOT,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ManifestError(
                    f"{meta_path}: {key}={metadata.get(key)!r}, expected {expected!r}"
                )
        manifest_summaries[str(prefix)] = {
            "filename": path.name,
            "meta_filename": meta_path.name,
            "rows": shared_count,
            "scenes": len(scene_split),
            "split_counts": sorted_split_counts,
            "sha256": checksum,
        }

    summary_path = manifest_dir / SUMMARY_NAME
    if not summary_path.is_file():
        raise ManifestError(f"missing canonical summary: {summary_path}")
    try:
        on_disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"invalid canonical summary {summary_path}: {exc}") from exc
    claimed_summary_hash = on_disk_summary.get("summary_sha256")
    unhashed_summary = dict(on_disk_summary)
    unhashed_summary.pop("summary_sha256", None)
    actual_summary_hash = _summary_hash(unhashed_summary)
    if claimed_summary_hash != actual_summary_hash:
        raise ManifestError(
            f"{summary_path}: summary_sha256 mismatch: {claimed_summary_hash!r} != {actual_summary_hash!r}"
        )
    expected_summary_fields = {
        "canonical_protocol": PROTOCOL_ID,
        "canonical_obs_len": CANONICAL_OBS_LEN,
        "action_horizons": list(ACTION_HORIZONS),
        "prefixes": list(prefixes),
        "shared_unit_count": shared_count,
        "logical_sample_count": shared_count * len(prefixes),
        "scene_count": len(scene_split),
        "split_counts": sorted_split_counts,
        "universe_sha256": shared_hash,
        "streaming_feature_root": STREAMING_FEATURE_ROOT,
        "target_feature_root": TARGET_FEATURE_ROOT,
        "manifests": manifest_summaries,
    }
    for key, expected in expected_summary_fields.items():
        if on_disk_summary.get(key) != expected:
            raise ManifestError(
                f"{summary_path}: {key} does not match audited manifests"
            )

    return {
        "ok": True,
        "canonical_protocol": PROTOCOL_ID,
        "canonical_obs_len": CANONICAL_OBS_LEN,
        "prefixes": list(prefixes),
        "shared_unit_count": shared_count,
        "logical_sample_count": shared_count * len(prefixes),
        "scene_count": len(scene_split),
        "split_counts": sorted_split_counts,
        "universe_sha256": shared_hash,
        "manifest_sha256": {
            prefix: manifest_summaries[str(prefix)]["sha256"] for prefix in prefixes
        },
        "summary_sha256": claimed_summary_hash,
    }


def _project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def _split_paths(args: argparse.Namespace, prefixes: Sequence[int]) -> list[Path]:
    if args.split_manifest:
        return [_project_path(value) for value in args.split_manifest]
    directory = _project_path(args.frozen_manifest_dir)
    return [directory / MANIFEST_NAME.format(prefix=prefix) for prefix in prefixes]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="generate aligned per-prefix manifests")
    build.add_argument(
        "--canonical-manifest",
        default="manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl",
        help="JSONL containing one validated 28-frame row per scene/window",
    )
    build.add_argument(
        "--frozen-manifest-dir",
        default="manifests/c_retrain_20260722",
        help="directory used to preserve and cross-check the frozen scene split",
    )
    build.add_argument(
        "--split-manifest",
        action="append",
        default=[],
        help="explicit frozen split manifest (repeatable; overrides --frozen-manifest-dir)",
    )
    build.add_argument("--output-dir", required=True)
    build.add_argument("--prefixes", default="8 12 16 24")
    build.add_argument("--expected-shared-units", type=int, default=2721)
    build.add_argument("--streaming-feature-root", default=STREAMING_FEATURE_ROOT)
    build.add_argument("--target-feature-root", default=TARGET_FEATURE_ROOT)
    build.add_argument("--dry-run", action="store_true")
    build.add_argument("--force", action="store_true")

    audit = subparsers.add_parser("audit", help="verify canonical manifests and hashes")
    audit.add_argument("--manifest-dir", required=True)
    audit.add_argument("--prefixes", default="8 12 16 24")
    audit.add_argument("--expected-shared-units", type=int, default=2721)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prefixes = _parse_ints(args.prefixes)
        if args.command == "build":
            summary = build_manifests(
                canonical_manifest=_project_path(args.canonical_manifest),
                split_manifests=_split_paths(args, prefixes),
                output_dir=_project_path(args.output_dir),
                prefixes=prefixes,
                expected_shared_units=args.expected_shared_units,
                dry_run=args.dry_run,
                force=args.force,
                streaming_feature_root=args.streaming_feature_root,
                target_feature_root=args.target_feature_root,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            result = audit_manifests(
                manifest_dir=_project_path(args.manifest_dir),
                prefixes=prefixes,
                expected_shared_units=args.expected_shared_units,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
