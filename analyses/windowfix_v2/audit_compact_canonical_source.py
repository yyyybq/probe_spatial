#!/usr/bin/env python3
"""Audit canonical-v2 from one frozen 28-observation source manifest.

The historical p24+h124 manifest already contains the complete 28-observation
window.  Materialising four ~55 MiB JSONL files only to replace the prefix
fields and feature roots is unnecessary and, more importantly, creates four
mutable copies of the temporal source of truth.  This audit keeps the frozen
source byte-for-byte unchanged, derives every prefix/anchor/future mapping in
memory, and emits a small hash-bound receipt.

The row-level feature roots in the frozen source are deliberately ignored.
The protocol file is the only authority for the isolated canonical-v2 feature
namespaces.  Consumers must bind both the source SHA256 and this receipt SHA256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from analyses.windowfix_v2.audit_label_sources import (
        AuditFailure,
        Problems,
        as_int_list,
        check_raw_scene,
        compact_sha,
        parse_ints,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` invocation.
    from audit_label_sources import (  # type: ignore[no-redef]
        AuditFailure,
        Problems,
        as_int_list,
        check_raw_scene,
        compact_sha,
        parse_ints,
        sha256_file,
    )


def _json_line(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _load_protocol(path: Path, prefixes: list[int]) -> tuple[dict[str, Any], list[int]]:
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditFailure(f"cannot read protocol {path}: {exc}") from exc
    if protocol.get("canonical_protocol") != "inscene_streaming_canonical_v2_obs28_h124":
        raise AuditFailure("unexpected canonical protocol id")
    window = protocol.get("canonical_window", {})
    if int(window.get("observations", -1)) != 28:
        raise AuditFailure("canonical window must contain 28 observations")
    if str(window.get("sampling")) != "motion_uniform":
        raise AuditFailure("canonical sampling must be motion_uniform")
    stream_prefixes = [int(value) for value in protocol["streaming"]["prefix_lengths"]]
    target = protocol["target"]
    target_prefixes = [int(value) for value in target["prefix_lengths"]]
    horizons = [int(value) for value in target["future_horizons"]]
    if prefixes != stream_prefixes or prefixes != target_prefixes:
        raise AuditFailure("requested prefixes do not equal the protocol prefix set")
    if target.get("include_anchors") is not True:
        raise AuditFailure("canonical targets must include anchors")
    if max(prefix + horizon for prefix in prefixes for horizon in horizons) > 28:
        raise AuditFailure("canonical window does not cover every requested target")
    acceptance = protocol.get("acceptance", {})
    if acceptance.get("legacy_features_valid") is not False:
        raise AuditFailure("canonical-v2 must reject unproven legacy feature payloads")
    if acceptance.get("audit_gate") != "fail_closed":
        raise AuditFailure("canonical-v2 audit gate must be fail_closed")
    return protocol, horizons


def _check_source_sidecar(source: Path, digest: str) -> dict[str, Any]:
    sidecar_path = Path(f"{source}.meta.json")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditFailure(f"cannot read source sidecar {sidecar_path}: {exc}") from exc
    if sidecar.get("sha256") != digest:
        raise AuditFailure("frozen source sidecar SHA256 mismatch")
    if sidecar.get("prefix_lengths") != [24]:
        raise AuditFailure("frozen source sidecar is not the p24 source")
    if [int(value) for value in sidecar.get("action_horizons", [])] != [1, 2, 4]:
        raise AuditFailure("frozen source sidecar horizon mismatch")
    if int(sidecar.get("num_samples", -1)) != 2721:
        raise AuditFailure("frozen source sidecar row count mismatch")
    return sidecar


def audit(
    *,
    source_manifest: Path,
    protocol_path: Path,
    prefixes: list[int],
    raw_scene_root: Path | None,
    namespace_lock_path: Path | None = None,
) -> dict[str, Any]:
    protocol, horizons = _load_protocol(protocol_path, prefixes)
    protocol_sha = sha256_file(protocol_path)
    source_sha = sha256_file(source_manifest)
    source_sidecar = _check_source_sidecar(source_manifest, source_sha)
    expected_windows = int(protocol["expected_universe"]["canonical_windows"])
    expected_scenes = int(protocol["expected_universe"]["eligible_scenes"])
    obs_len = int(protocol["canonical_window"]["observations"])

    problems = Problems()
    counts: Counter[str] = Counter()
    universe_digest = hashlib.sha256()
    source_universe_digest = hashlib.sha256()
    mapping_digest = hashlib.sha256()
    per_prefix_digest = {prefix: hashlib.sha256() for prefix in prefixes}
    previous_unit: tuple[str, int] | None = None
    seen_units: set[tuple[str, int]] = set()
    seen_scenes: set[str] = set()
    current_scene: str | None = None
    current_frames: list[str] = []
    current_pose_indices: list[int] = []
    current_label_signature = ""
    selected_for_scene: set[int] = set()

    def flush_scene() -> None:
        if current_scene is None or raw_scene_root is None:
            return
        check_raw_scene(
            scene=current_scene,
            valid_frames=current_frames,
            valid_pose_indices=current_pose_indices,
            selected_indices=selected_for_scene,
            raw_scene_root=raw_scene_root,
            problems=problems,
            counts=counts,
        )

    with source_manifest.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            where = f"{source_manifest}:{line_number}"
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise AuditFailure("row is not an object")
                scene = str(row["scene_id"])
                split = str(row["split"])
                window_id = int(row["streaming_window_id"])
                window = as_int_list(row, "streaming_window_indices", where)
                source_input = as_int_list(row, "streaming_indices", where)
                valid_frames = [str(value) for value in row["valid_frames"]]
                valid_pose_indices = as_int_list(row, "valid_pose_indices", where)
                if row.get("source") != protocol.get("source"):
                    raise AuditFailure("source mismatch")
                if row.get("streaming_sampling") != protocol["canonical_window"]["sampling"]:
                    raise AuditFailure("sampling mismatch")
                if len(window) != obs_len or window != sorted(set(window)):
                    raise AuditFailure("window is not 28 strictly increasing raw IDs")
                if source_input != window[:24]:
                    raise AuditFailure("p24 source input is not the first 24 window IDs")
                if int(row.get("streaming_tail", -1)) != 23:
                    raise AuditFailure("p24 source tail mismatch")
                if int(row.get("streaming_tail_raw", -1)) != window[23]:
                    raise AuditFailure("p24 source raw tail mismatch")
                if len(valid_frames) != len(valid_pose_indices):
                    raise AuditFailure("valid frame/pose map length mismatch")
                if any(index < 0 or index >= len(valid_frames) for index in window):
                    raise AuditFailure("window raw ID is outside valid frame map")
                unit = (scene, window_id)
                if unit in seen_units:
                    raise AuditFailure("duplicate scene/window unit")
                if previous_unit is not None and unit <= previous_unit:
                    raise AuditFailure("source is not in strict scene/window order")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, AuditFailure) as exc:
                problems.add("source_contract", where=where, error=str(exc))
                continue

            signature = compact_sha([valid_frames, valid_pose_indices])
            if scene != current_scene:
                flush_scene()
                current_scene = scene
                current_frames = valid_frames
                current_pose_indices = valid_pose_indices
                current_label_signature = signature
                selected_for_scene = set()
                seen_scenes.add(scene)
            elif signature != current_label_signature:
                problems.add("within_scene_label_map_mismatch", scene=scene, window_id=window_id)

            seen_units.add(unit)
            previous_unit = unit
            selected_for_scene.update(window)
            counts["canonical_windows"] += 1
            universe_digest.update(
                _json_line(
                    {"scene": scene, "window_id": window_id, "window": window, "split": split}
                )
            )
            source_universe_digest.update(
                _json_line(
                    {
                        "scene_id": scene,
                        "split": split,
                        "streaming_window_id": window_id,
                        "streaming_window_indices": window,
                    }
                )
            )

            for prefix in prefixes:
                input_ids = window[:prefix]
                anchor = window[prefix - 1]
                future = {str(horizon): window[prefix - 1 + horizon] for horizon in horizons}
                mapping = {
                    "scene": scene,
                    "window_id": window_id,
                    "prefix": prefix,
                    "input": input_ids,
                    "anchor": anchor,
                    "future": future,
                }
                encoded = _json_line(mapping)
                mapping_digest.update(encoded)
                per_prefix_digest[prefix].update(encoded)
                counts["derived_prefix_rows"] += 1
                counts["input_frame_references"] += prefix
                counts["anchor_references"] += 1
                counts["future_target_references"] += len(horizons)
        flush_scene()

    if counts["canonical_windows"] != expected_windows:
        problems.add(
            "canonical_window_count",
            expected=expected_windows,
            actual=counts["canonical_windows"],
        )
    if len(seen_scenes) != expected_scenes:
        problems.add("scene_count", expected=expected_scenes, actual=len(seen_scenes))
    expected_rows = expected_windows * len(prefixes)
    if counts["derived_prefix_rows"] != expected_rows:
        problems.add(
            "derived_prefix_row_count",
            expected=expected_rows,
            actual=counts["derived_prefix_rows"],
        )
    final_protocol_sha = sha256_file(protocol_path)
    if final_protocol_sha != protocol_sha:
        problems.add(
            "protocol_changed_during_audit",
            before=protocol_sha,
            after=final_protocol_sha,
        )
    final_source_sha = sha256_file(source_manifest)
    if final_source_sha != source_sha:
        problems.add(
            "source_manifest_changed_during_audit",
            before=source_sha,
            after=final_source_sha,
        )

    namespace_binding: dict[str, Any] | None = None
    if namespace_lock_path is not None:
        try:
            lock = json.loads(namespace_lock_path.read_text(encoding="utf-8"))
            expected_roots = {
                kind: f"aoss://baiqiao/InsScene-15K/{protocol['feature_namespaces'][kind]}"
                for kind in ("streaming", "target")
            }
            if lock.get("schema_version") != 2:
                raise AuditFailure("compact namespace lock schema_version must be 2")
            if lock.get("canonical_protocol") != protocol["canonical_protocol"]:
                raise AuditFailure("compact namespace lock protocol mismatch")
            if lock.get("protocol_file_sha256") != protocol_sha:
                raise AuditFailure("compact namespace lock protocol SHA mismatch")
            if lock.get("canonical_source", {}).get("sha256") != source_sha:
                raise AuditFailure("compact namespace lock source SHA mismatch")
            if lock.get("canonical_universe_sha256") != source_universe_digest.hexdigest():
                raise AuditFailure("compact namespace lock universe SHA mismatch")
            if lock.get("feature_roots") != expected_roots:
                raise AuditFailure("compact namespace lock feature-root mismatch")
            namespace_binding = {
                "mode": "compact_locked_override",
                "path": str(namespace_lock_path),
                "sha256": sha256_file(namespace_lock_path),
                "feature_roots": expected_roots,
            }
        except Exception as exc:
            problems.add("namespace_lock_contract", error=str(exc))

    result = {
        "audit_schema_version": 2,
        "audit_type": "canonical_v2_compact_source_label_lineage",
        "status": "pass" if problems.count == 0 else "fail",
        "fail_closed": True,
        "protocol": protocol["canonical_protocol"],
        "protocol_file": str(protocol_path),
        "protocol_file_sha256": protocol_sha,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_sha,
        "source_sidecar_sha256": sha256_file(Path(f"{source_manifest}.meta.json")),
        "source_sidecar_declared_samples": source_sidecar["num_samples"],
        "derivation": {
            "canonical_obs_len": obs_len,
            "prefixes": prefixes,
            "input_rule": "streaming_window_indices[:prefix]",
            "anchor_rule": "streaming_window_indices[prefix-1]",
            "future_rule": "streaming_window_indices[prefix-1+horizon]",
            "horizons": horizons,
            "row_level_legacy_feature_roots_ignored": True,
        },
        "feature_roots": (
            namespace_binding["feature_roots"]
            if namespace_binding is not None
            else {
                "streaming": protocol["feature_namespaces"]["streaming"],
                "target": protocol["feature_namespaces"]["target"],
            }
        ),
        "namespace_binding": namespace_binding,
        "raw_scene_root": str(raw_scene_root) if raw_scene_root is not None else None,
        "counts": dict(sorted(counts.items())),
        "scene_count": len(seen_scenes),
        "canonical_universe_sha256": source_universe_digest.hexdigest(),
        "label_lineage_universe_sha256": universe_digest.hexdigest(),
        "input_anchor_future_mapping_sha256": mapping_digest.hexdigest(),
        "per_prefix_mapping_sha256": {
            str(prefix): per_prefix_digest[prefix].hexdigest() for prefix in prefixes
        },
        "problem_count": problems.count,
        "problem_counts": dict(sorted(problems.counts.items())),
        "problems": problems.items,
    }
    return result


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
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/inscene_streaming_canonical_v2.json"),
    )
    parser.add_argument("--prefixes", default="8,12,16,24")
    parser.add_argument("--raw-scene-root", type=Path)
    parser.add_argument("--namespace-lock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = audit(
            source_manifest=args.source_manifest.resolve(),
            protocol_path=args.protocol.resolve(),
            prefixes=parse_ints(args.prefixes),
            raw_scene_root=args.raw_scene_root.resolve() if args.raw_scene_root else None,
            namespace_lock_path=(
                args.namespace_lock.resolve() if args.namespace_lock else None
            ),
        )
    except Exception as exc:
        result = {
            "audit_schema_version": 2,
            "audit_type": "canonical_v2_compact_source_label_lineage",
            "status": "fail",
            "fail_closed": True,
            "fatal_error": repr(exc),
        }
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
