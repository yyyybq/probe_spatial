#!/usr/bin/env python3
"""Fail-closed audit for canonical-v2 manifests and raw label lineage.

This audit intentionally does not load any feature tensor.  It proves the
other half of the join: every canonical input/anchor/future position resolves
to one valid ScanNet++ frame and that frame has image, mask, depth, intrinsics
and trajectory sources.  Feature indexes are checked by the separate reuse /
post-extraction audit.
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
from typing import Any, Iterable, TextIO

import numpy as np


DEFAULT_PREFIXES = (8, 12, 16, 24)
MAX_PROBLEMS = 200


class AuditFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_ints(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    else:
        values = value
    result = [int(item) for item in values]
    if not result or len(result) != len(set(result)) or any(item <= 0 for item in result):
        raise AuditFailure("prefixes must be unique positive integers")
    return result


def read_jsonl(handle: TextIO, path: Path):
    for line_number, raw in enumerate(handle, 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuditFailure(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise AuditFailure(f"{path}:{line_number}: row is not an object")
        yield line_number, row


def as_int_list(row: dict[str, Any], key: str, where: str) -> list[int]:
    value = row.get(key)
    if not isinstance(value, list):
        raise AuditFailure(f"{where}: {key} is not a list")
    try:
        result = [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise AuditFailure(f"{where}: {key} is not an integer list") from exc
    return result


def uri_namespace(value: str) -> str:
    return str(value).rstrip("/").rsplit("/", 1)[-1]


class Problems:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.items: list[dict[str, Any]] = []

    def add(self, kind: str, **detail: Any) -> None:
        self.counts[kind] += 1
        if len(self.items) < MAX_PROBLEMS:
            self.items.append({"kind": kind, **detail})

    @property
    def count(self) -> int:
        return sum(self.counts.values())


def check_sidecar(path: Path, expected_sha: str, prefix: int, protocol: dict[str, Any], problems: Problems) -> None:
    sidecar = Path(f"{path}.meta.json")
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        problems.add("manifest_meta_unreadable", path=str(sidecar), error=repr(exc))
        return
    expected = {
        "canonical_protocol": protocol["canonical_protocol"],
        "canonical_obs_len": int(protocol["canonical_window"]["observations"]),
        "prefix_length": prefix,
        "action_horizons": [int(v) for v in protocol["target"]["future_horizons"]],
        "sha256": expected_sha,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            problems.add(
                "manifest_meta_mismatch",
                path=str(sidecar),
                field=key,
                expected=value,
                actual=meta.get(key),
            )


def check_raw_scene(
    *,
    scene: str,
    valid_frames: list[str],
    valid_pose_indices: list[int],
    selected_indices: set[int],
    raw_scene_root: Path,
    problems: Problems,
    counts: Counter[str],
) -> None:
    scene_root = raw_scene_root / scene
    meta_path = scene_root / "scene_iphone_metadata.npz"
    try:
        meta = np.load(meta_path, allow_pickle=True)
        meta_images = [str(item) for item in meta["images"].tolist()]
        intrinsics = meta["intrinsics"]
        trajectories = meta["trajectories"]
    except Exception as exc:
        problems.add("raw_metadata_unreadable", scene=scene, path=str(meta_path), error=repr(exc))
        return

    if len(valid_frames) != len(valid_pose_indices):
        problems.add(
            "valid_pose_length",
            scene=scene,
            valid_frames=len(valid_frames),
            valid_pose_indices=len(valid_pose_indices),
        )
        return
    counts["raw_scenes_checked"] += 1
    for local_index in sorted(selected_indices):
        counts["raw_frame_identities_checked"] += 1
        if local_index < 0 or local_index >= len(valid_frames):
            problems.add(
                "raw_local_index_oob",
                scene=scene,
                local_index=local_index,
                valid_frames=len(valid_frames),
            )
            continue
        frame_name = valid_frames[local_index]
        pose_index = valid_pose_indices[local_index]
        if pose_index < 0 or pose_index >= len(meta_images):
            problems.add(
                "raw_pose_index_oob",
                scene=scene,
                local_index=local_index,
                pose_index=pose_index,
            )
            continue
        if meta_images[pose_index] != frame_name:
            problems.add(
                "raw_pose_frame_mismatch",
                scene=scene,
                local_index=local_index,
                pose_index=pose_index,
                manifest_frame=frame_name,
                metadata_frame=meta_images[pose_index],
            )
        paths = {
            "image": scene_root / "images" / frame_name,
            "mask": scene_root / "refined_ins_ids" / f"{frame_name}.npy",
            "depth": scene_root / "depth" / f"{Path(frame_name).stem}.png",
        }
        for modality, path in paths.items():
            counts[f"{modality}_paths_checked"] += 1
            if not path.is_file():
                problems.add(
                    f"raw_{modality}_missing",
                    scene=scene,
                    local_index=local_index,
                    path=str(path),
                )
        try:
            intrinsic = np.asarray(intrinsics[pose_index])
            trajectory = np.asarray(trajectories[pose_index])
            if intrinsic.shape not in {(3, 3), (4, 4)} or not np.isfinite(intrinsic).all():
                raise ValueError(f"bad intrinsics shape/values: {intrinsic.shape}")
            if trajectory.shape != (4, 4) or not np.isfinite(trajectory).all():
                raise ValueError(f"bad trajectory shape/values: {trajectory.shape}")
            counts["pose_rows_checked"] += 1
        except Exception as exc:
            problems.add(
                "raw_camera_invalid",
                scene=scene,
                local_index=local_index,
                pose_index=pose_index,
                error=repr(exc),
            )


def audit(
    *,
    manifest_dir: Path,
    protocol_path: Path,
    prefixes: list[int],
    raw_scene_root: Path | None,
    namespace_lock_path: Path | None = None,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_file_sha = sha256_file(protocol_path)
    protocol_id = str(protocol["canonical_protocol"])
    obs_len = int(protocol["canonical_window"]["observations"])
    protocol_prefixes = [int(v) for v in protocol["streaming"]["prefix_lengths"]]
    target_prefixes = [int(v) for v in protocol["target"]["prefix_lengths"]]
    horizons = [int(v) for v in protocol["target"]["future_horizons"]]
    if prefixes != protocol_prefixes or prefixes != target_prefixes:
        raise AuditFailure(
            f"requested prefixes {prefixes} do not equal protocol prefixes "
            f"{protocol_prefixes}/{target_prefixes}"
        )
    required = max(prefix + horizon for prefix in prefixes for horizon in horizons)
    if obs_len < required:
        raise AuditFailure(f"canonical obs_len={obs_len} does not cover required position {required}")

    namespace_lock: dict[str, Any] | None = None
    if namespace_lock_path is not None:
        namespace_lock = json.loads(namespace_lock_path.read_text(encoding="utf-8"))
        if namespace_lock.get("schema_version") != 1:
            raise AuditFailure("namespace lock schema_version must be 1")
        if namespace_lock.get("protocol_file_sha256") != protocol_file_sha:
            raise AuditFailure("namespace lock was built for a different protocol file")
        expected_roots = {
            "streaming": protocol["feature_namespaces"]["streaming"],
            "target": protocol["feature_namespaces"]["target"],
        }
        for kind, namespace in expected_roots.items():
            if uri_namespace(namespace_lock.get("feature_roots", {}).get(kind, "")) != namespace:
                raise AuditFailure(f"namespace lock {kind} root does not match protocol")

    paths = [manifest_dir / f"inscene_c_h124_prefix{prefix}.jsonl" for prefix in prefixes]
    problems = Problems()
    manifest_sha: dict[str, str] = {}
    for prefix, path in zip(prefixes, paths):
        if not path.is_file():
            raise AuditFailure(f"missing manifest: {path}")
        digest = sha256_file(path)
        manifest_sha[str(prefix)] = digest
        check_sidecar(path, digest, prefix, protocol, problems)
        if namespace_lock is not None:
            locked = namespace_lock.get("manifest_sha256", {}).get(str(prefix))
            if locked != digest:
                problems.add(
                    "namespace_lock_manifest_sha256",
                    prefix=prefix,
                    expected=digest,
                    actual=locked,
                )

    counts: Counter[str] = Counter()
    mapping_digest = hashlib.sha256()
    universe_digest = hashlib.sha256()
    previous_unit: tuple[str, int] | None = None
    current_scene: str | None = None
    current_frames: list[str] = []
    current_pose_indices: list[int] = []
    current_signature = ""
    selected_for_scene: set[int] = set()
    scene_count = 0

    def flush_scene() -> None:
        nonlocal scene_count
        if current_scene is None:
            return
        scene_count += 1
        if raw_scene_root is not None:
            check_raw_scene(
                scene=current_scene,
                valid_frames=current_frames,
                valid_pose_indices=current_pose_indices,
                selected_indices=selected_for_scene,
                raw_scene_root=raw_scene_root,
                problems=problems,
                counts=counts,
            )

    handles = [path.open(encoding="utf-8") for path in paths]
    try:
        streams = [read_jsonl(handle, path) for handle, path in zip(handles, paths)]
        for ordinal, group in enumerate(itertools.zip_longest(*streams), 1):
            if any(item is None for item in group):
                problems.add("manifest_row_count_mismatch", ordinal=ordinal)
                break
            reference: tuple[str, int, list[int], str] | None = None
            group_frames: list[str] | None = None
            group_pose_indices: list[int] | None = None
            for prefix, path, item in zip(prefixes, paths, group):
                assert item is not None
                line_number, row = item
                where = f"{path}:{line_number}"
                try:
                    scene = str(row["scene_id"])
                    window_id = int(row["streaming_window_id"])
                    window = as_int_list(row, "streaming_window_indices", where)
                    input_ids = as_int_list(row, "streaming_indices", where)
                    valid_frames = [str(v) for v in row["valid_frames"]]
                    valid_pose_indices = as_int_list(row, "valid_pose_indices", where)
                    split = str(row["split"])
                    if row.get("canonical_protocol") != protocol_id:
                        raise AuditFailure(f"{where}: canonical_protocol mismatch")
                    if int(row.get("canonical_obs_len", -1)) != obs_len:
                        raise AuditFailure(f"{where}: canonical_obs_len mismatch")
                    if int(row.get("prefix_length", -1)) != prefix:
                        raise AuditFailure(f"{where}: prefix_length mismatch")
                    if window != sorted(set(window)) or len(window) != obs_len:
                        raise AuditFailure(f"{where}: canonical window is not {obs_len} strictly increasing IDs")
                    if input_ids != window[:prefix]:
                        raise AuditFailure(f"{where}: streaming_indices are not canonical prefix")
                    if int(row.get("streaming_tail", -1)) != prefix - 1:
                        raise AuditFailure(f"{where}: streaming_tail mismatch")
                    if int(row.get("streaming_tail_raw", -1)) != input_ids[-1]:
                        raise AuditFailure(f"{where}: streaming_tail_raw mismatch")
                    if [int(v) for v in row.get("target_horizons", [])] != horizons:
                        raise AuditFailure(f"{where}: target_horizons mismatch")
                    if row.get("include_anchors") is not True:
                        raise AuditFailure(f"{where}: include_anchors is not true")
                    if namespace_lock is None:
                        expected_stream_ns = protocol["feature_namespaces"]["streaming"]
                        expected_target_ns = protocol["feature_namespaces"]["target"]
                        if uri_namespace(row.get("streaming_feature_root", "")) != expected_stream_ns:
                            raise AuditFailure(f"{where}: streaming feature namespace mismatch")
                        if uri_namespace(row.get("target_feature_root", "")) != expected_target_ns:
                            raise AuditFailure(f"{where}: target feature namespace mismatch")
                    if any(index < 0 or index >= len(valid_frames) for index in window):
                        raise AuditFailure(f"{where}: canonical raw-ID outside valid_frames")
                    if len(valid_frames) != len(valid_pose_indices):
                        raise AuditFailure(f"{where}: valid frame/pose map length mismatch")
                except (KeyError, TypeError, ValueError, AuditFailure) as exc:
                    problems.add("manifest_contract", where=where, error=str(exc))
                    continue

                signature = compact_sha([valid_frames, valid_pose_indices])
                identity = (scene, window_id, window, split)
                if reference is None:
                    reference = identity
                    group_frames = valid_frames
                    group_pose_indices = valid_pose_indices
                elif identity != reference:
                    problems.add("cross_prefix_window_mismatch", where=where, ordinal=ordinal)
                elif compact_sha([valid_frames, valid_pose_indices]) != compact_sha(
                    [group_frames, group_pose_indices]
                ):
                    problems.add("cross_prefix_label_map_mismatch", where=where, ordinal=ordinal)

                anchor_position = prefix - 1
                target_positions = [anchor_position] + [anchor_position + h for h in horizons]
                target_ids = [window[position] for position in target_positions]
                counts["manifest_rows"] += 1
                counts["input_frame_references"] += len(input_ids)
                counts["anchor_references"] += 1
                counts["future_target_references"] += len(horizons)
                mapping_digest.update(
                    (json.dumps(
                        {
                            "scene": scene,
                            "window_id": window_id,
                            "prefix": prefix,
                            "input": input_ids,
                            "anchor": target_ids[0],
                            "future": dict(zip(map(str, horizons), target_ids[1:])),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ) + "\n").encode("utf-8")
                )

            if reference is None or group_frames is None or group_pose_indices is None:
                continue
            scene, window_id, window, split = reference
            unit = (scene, window_id)
            if previous_unit is not None and unit <= previous_unit:
                problems.add("canonical_order_or_duplicate", unit=list(unit), previous=list(previous_unit))
            previous_unit = unit
            universe_digest.update(
                (json.dumps(
                    {"scene": scene, "window_id": window_id, "window": window, "split": split},
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n").encode("utf-8")
            )
            if scene != current_scene:
                flush_scene()
                current_scene = scene
                current_frames = group_frames
                current_pose_indices = group_pose_indices
                current_signature = compact_sha([current_frames, current_pose_indices])
                selected_for_scene = set()
            elif compact_sha([group_frames, group_pose_indices]) != current_signature:
                problems.add("within_scene_label_map_mismatch", scene=scene, window_id=window_id)
            selected_for_scene.update(window)
            counts["canonical_windows"] += 1
        flush_scene()
    finally:
        for handle in handles:
            handle.close()

    expected_windows = int(protocol.get("expected_universe", {}).get("canonical_windows", 0))
    expected_scenes = int(protocol.get("expected_universe", {}).get("eligible_scenes", 0))
    expected_rows = expected_windows * len(prefixes)
    if expected_windows and counts["canonical_windows"] != expected_windows:
        problems.add(
            "canonical_window_count",
            expected=expected_windows,
            actual=counts["canonical_windows"],
        )
    if expected_scenes and scene_count != expected_scenes:
        problems.add("scene_count", expected=expected_scenes, actual=scene_count)
    if expected_rows and counts["manifest_rows"] != expected_rows:
        problems.add("manifest_row_count", expected=expected_rows, actual=counts["manifest_rows"])

    return {
        "audit_schema_version": 1,
        "audit_type": "canonical_v2_label_lineage",
        "status": "pass" if problems.count == 0 else "fail",
        "fail_closed": True,
        "protocol": protocol_id,
        "protocol_file": str(protocol_path),
        "protocol_file_sha256": protocol_file_sha,
        "manifest_dir": str(manifest_dir),
        "manifest_sha256": manifest_sha,
        "namespace_binding": (
            {
                "mode": "locked_override",
                "path": str(namespace_lock_path),
                "sha256": sha256_file(namespace_lock_path),
                "feature_roots": namespace_lock["feature_roots"],
            }
            if namespace_lock is not None
            else {"mode": "manifest_rows"}
        ),
        "raw_scene_root": str(raw_scene_root) if raw_scene_root is not None else None,
        "counts": dict(sorted(counts.items())),
        "scene_count": scene_count,
        "canonical_universe_sha256": universe_digest.hexdigest(),
        "input_anchor_future_mapping_sha256": mapping_digest.hexdigest(),
        "problem_count": problems.count,
        "problem_counts": dict(sorted(problems.counts.items())),
        "problems": problems.items,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/inscene_streaming_canonical_v2.json"),
    )
    parser.add_argument("--prefixes", default="8,12,16,24")
    parser.add_argument(
        "--raw-scene-root",
        type=Path,
        help="local directory containing one subdirectory per ScanNet++ scene",
    )
    parser.add_argument(
        "--namespace-lock",
        type=Path,
        help=(
            "small immutable overlay binding the unchanged canonical manifest "
            "hashes to isolated feature roots"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = audit(
            manifest_dir=args.manifest_dir.resolve(),
            protocol_path=args.protocol.resolve(),
            prefixes=parse_ints(args.prefixes),
            raw_scene_root=args.raw_scene_root.resolve() if args.raw_scene_root else None,
            namespace_lock_path=args.namespace_lock.resolve() if args.namespace_lock else None,
        )
    except Exception as exc:
        result = {
            "audit_schema_version": 1,
            "audit_type": "canonical_v2_label_lineage",
            "status": "fail",
            "fail_closed": True,
            "fatal_error": repr(exc),
        }
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
