#!/usr/bin/env python3
"""Audit one derived prefix from a single frozen canonical obs28 source.

The source may be the historical prefix-24 JSONL because its authoritative
field is the 28-frame ``streaming_window_indices``.  Row-level feature roots
are intentionally replaced by the separately hashed namespace lock.  A small
temporary manifest containing only structural fields is derived as
``window[:prefix]`` and deleted after the existing strict alignment auditor
returns.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"
PREFIXES = (8, 12, 16, 24)
HORIZONS = (1, 2, 4)


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"ERROR: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        fail(f"{label} must be a regular non-symlink file: {path}")
    if len(expected) != 64 or sha256_file(path) != expected:
        fail(f"{label} SHA256 mismatch: {path}")


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot read {label}: {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must contain a JSON object")
    return value


def load_auditor(project_root: Path):
    path = project_root / "scripts/audit_tstep_canonical_alignment.py"
    if not path.is_file() or path.is_symlink():
        fail(f"strict alignment auditor missing/symlinked: {path}")
    spec = importlib.util.spec_from_file_location("windowfix_v2_base_alignment_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def derive_manifest(
    *, source: Path, output: Path, prefix: int, streaming_root: str, target_root: str
) -> tuple[int, int, str]:
    scenes: set[str] = set()
    units: set[tuple[str, int]] = set()
    count = 0
    with source.open(encoding="utf-8") as reader, output.open("w", encoding="utf-8") as writer:
        for line_number, raw in enumerate(reader, start=1):
            if not raw.strip():
                continue
            try:
                source_row = json.loads(raw)
                scene = str(source_row.get("scene_id") or Path(str(source_row["scene_dir"])).name)
                window_id = int(source_row["streaming_window_id"])
                window = [int(value) for value in source_row["streaming_window_indices"]]
                source_prefix = [int(value) for value in source_row["streaming_indices"]]
                split = str(source_row["split"])
            except Exception as exc:
                fail(f"canonical source line {line_number} is invalid: {exc}")
            if len(window) != 28 or any(b <= a for a, b in zip(window, window[1:])):
                fail(f"canonical source line {line_number} is not a strict obs28 window")
            if len(source_prefix) != 24 or source_prefix != window[:24]:
                fail(f"canonical source line {line_number} is not p24=window[:24]")
            unit = (scene, window_id)
            if unit in units:
                fail(f"duplicate canonical unit at source line {line_number}: {unit}")
            units.add(unit)
            scenes.add(scene)
            derived = {
                "sample_id": f"{scene}_w{window_id:03d}_p{prefix}",
                "scene_id": scene,
                "scene_dir": source_row.get("scene_dir", scene),
                "source": "scannetpp",
                "split": split,
                "prefix_length": prefix,
                "canonical_protocol": PROTOCOL,
                "canonical_obs_len": 28,
                "streaming_window_id": window_id,
                "streaming_obs_start": int(source_row.get("streaming_obs_start", window[0])),
                "streaming_sampling": "motion_uniform",
                "streaming_window_indices": window,
                "streaming_indices": window[:prefix],
                "streaming_tail": prefix - 1,
                "streaming_tail_raw": window[prefix - 1],
                "streaming_feature_root": streaming_root,
                "target_feature_root": target_root,
                "target_horizons": list(HORIZONS),
                "include_anchors": True,
            }
            writer.write(json.dumps(derived, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    digest = sha256_file(output)
    metadata = {
        "manifest_version": 2,
        "canonical_protocol": PROTOCOL,
        "canonical_obs_len": 28,
        "prefix_length": prefix,
        "action_horizons": list(HORIZONS),
        "streaming_feature_root": streaming_root,
        "target_feature_root": target_root,
        "num_samples": count,
        "num_scenes": len(scenes),
        "sha256": digest,
        "derivation": "canonical_obs28_source.streaming_window_indices[:prefix]",
    }
    Path(f"{output}.meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return count, len(scenes), digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-source", required=True, type=Path)
    parser.add_argument("--canonical-source-sha256", required=True)
    parser.add_argument("--namespace-lock", required=True, type=Path)
    parser.add_argument("--namespace-lock-sha256", required=True)
    parser.add_argument("--label-audit", required=True, type=Path)
    parser.add_argument("--label-audit-sha256", required=True)
    parser.add_argument("--prefix", required=True, type=int, choices=PREFIXES)
    parser.add_argument("--streaming-index-root", required=True, type=Path)
    parser.add_argument("--target-index-root", required=True, type=Path)
    parser.add_argument("--vfm", required=True)
    parser.add_argument("--source", default="scannetpp")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    source = args.canonical_source.resolve()
    lock_path = args.namespace_lock.resolve()
    label_path = args.label_audit.resolve()
    require_sha(source, args.canonical_source_sha256, "canonical obs28 source")
    require_sha(lock_path, args.namespace_lock_sha256, "namespace lock")
    require_sha(label_path, args.label_audit_sha256, "label-lineage audit")
    lock = read_json(lock_path, "namespace lock")
    label = read_json(label_path, "label-lineage audit")
    if lock.get("schema_version") != 2 or lock.get("canonical_protocol") != PROTOCOL:
        fail("namespace lock protocol/schema mismatch")
    lock_source = lock.get("canonical_source")
    if not isinstance(lock_source, dict):
        fail("namespace lock has no canonical_source object")
    if (
        Path(str(lock_source.get("path", ""))).resolve() != source
        or lock_source.get("sha256") != args.canonical_source_sha256
        or lock_source.get("rows") != 2721
        or lock_source.get("scenes") != 681
        or lock_source.get("canonical_obs_len") != 28
        or lock_source.get("source_prefix") != 24
    ):
        fail("namespace lock does not bind the complete p24/obs28 canonical source")
    sidecar_path = Path(str(lock_source.get("sidecar_path", ""))).resolve()
    require_sha(sidecar_path, str(lock_source.get("sidecar_sha256", "")), "canonical source sidecar")
    sidecar = read_json(sidecar_path, "canonical source sidecar")
    if (
        sidecar.get("sha256") != args.canonical_source_sha256
        or sidecar.get("prefix_lengths") != [24]
        or sidecar.get("action_horizons") != list(HORIZONS)
        or sidecar.get("num_samples") != 2721
        or sidecar.get("num_included_scenes") != 681
    ):
        fail("canonical source sidecar contract mismatch")
    if (
        label.get("audit_schema_version") != 2
        or label.get("audit_type") != "canonical_v2_compact_source_label_lineage"
        or label.get("status") != "pass"
        or label.get("problem_count") != 0
        or label.get("fail_closed") is not True
        or label.get("protocol") != PROTOCOL
        or label.get("protocol_file_sha256") != lock.get("protocol_file_sha256")
        or Path(str(label.get("source_manifest", ""))).resolve() != source
        or label.get("source_manifest_sha256") != args.canonical_source_sha256
        or label.get("source_sidecar_sha256") != lock_source.get("sidecar_sha256")
        or label.get("namespace_binding", {}).get("sha256") != args.namespace_lock_sha256
    ):
        fail("label-lineage audit is not a PASS bound to this namespace lock")
    if label.get("canonical_universe_sha256") != lock.get("canonical_universe_sha256"):
        fail("label-lineage audit canonical universe differs from namespace lock")
    counts = label.get("counts", {})
    required_counts = {
        "canonical_windows": 2721,
        "derived_prefix_rows": 10884,
        "input_frame_references": 163260,
        "anchor_references": 10884,
        "future_target_references": 32652,
        "raw_scenes_checked": 681,
    }
    if any(counts.get(field) != expected for field, expected in required_counts.items()):
        fail("label-lineage audit is not the complete canonical-v2 receipt")
    raw_count = counts.get("raw_frame_identities_checked")
    if not isinstance(raw_count, int) or raw_count <= 0 or any(
        counts.get(field) != raw_count
        for field in (
            "image_paths_checked",
            "mask_paths_checked",
            "depth_paths_checked",
            "pose_rows_checked",
        )
    ):
        fail("label-lineage audit lacks full raw label checks")
    roots = lock.get("feature_roots", {})
    streaming_root = str(roots.get("streaming", ""))
    target_root = str(roots.get("target", ""))
    if not streaming_root.endswith("/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"):
        fail("namespace lock streaming root is not the versioned canonical namespace")
    if not target_root.endswith("/FEAT_TARGET_CANONICAL_V2_SEEDED_V1"):
        fail("namespace lock target root is not the versioned canonical namespace")

    auditor = load_auditor(args.project_root.resolve())
    with tempfile.TemporaryDirectory(prefix=f"windowfix-v2-derived-p{args.prefix}-") as temp:
        derived = Path(temp) / f"derived_prefix{args.prefix}.jsonl"
        rows, scenes, derived_sha = derive_manifest(
            source=source,
            output=derived,
            prefix=args.prefix,
            streaming_root=streaming_root,
            target_root=target_root,
        )
        if rows != 2721 or scenes != 681:
            fail(f"derived manifest universe mismatch: rows={rows} scenes={scenes}")
        report = auditor.audit_alignment(
            manifest=derived,
            streaming_index_root=args.streaming_index_root,
            target_index_root=args.target_index_root,
            vfm=args.vfm,
            source=args.source,
            protocol_id=PROTOCOL,
            canonical_obs_len=28,
            prefixes=PREFIXES,
            horizons=HORIZONS,
            scene_ids=(),
        )
    report.update(
        {
            "binding_mode": "compact_obs28_source_v1",
            "derived_prefix": args.prefix,
            "derived_manifest_sha256": derived_sha,
            "manifest": str(source),
            "manifest_sha256": args.canonical_source_sha256,
            "manifest_meta": None,
            "canonical_source": {
                "path": str(source),
                "sha256": args.canonical_source_sha256,
                "sidecar_path": str(sidecar_path),
                "sidecar_sha256": lock_source["sidecar_sha256"],
            },
            "namespace_lock": {"path": str(lock_path), "sha256": args.namespace_lock_sha256},
            "label_lineage_audit": {
                "path": str(label_path),
                "sha256": args.label_audit_sha256,
                "mapping_sha256": label["input_anchor_future_mapping_sha256"],
                "per_prefix_mapping_sha256": label["per_prefix_mapping_sha256"],
            },
        }
    )
    auditor.write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "scope": report["scope"],
                "vfm": report["vfm"],
                "derived_prefix": args.prefix,
                "samples_checked": report["samples_checked"],
                "problem_count": report["problem_count"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
