#!/usr/bin/env python3
"""Freeze the complete probe816 MiniMax scene set as an isolated extractor view.

The first route was built from MiniMax's historical 1,694-row owner subset.  The
current canonical-v2 probing scope is the 2,721-row receipt-union manifest for
each prefix.  Select those manifests from the frozen catalog instead of
embedding hashes for the obsolete subset.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
ARTIFACT = (
    ROOT
    / "artifacts/main_v2_probe_v2_816_20260917"
    / "minimax_bf16_v2_production_route_v2"
)
VIEW = ARTIFACT / "data/processed_scannetpp_v2/processed_scannetpp_v2"
SOURCE = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2")
RECEIPT = ARTIFACT / "route_receipt.json"
MANIFEST_CATALOG = (
    ROOT
    / "artifacts/main_v2_probe_v2_816_20260917"
    / "execution_manifest_r3_receipt_union/manifest.json"
)
EXPECTED_ROWS_PER_PREFIX = 2721
PREFIXES = (8, 12, 16, 24)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o444)
    Path(temporary).replace(path)


def select_complete_manifests(catalog_path: Path) -> dict[int, tuple[Path, str]]:
    """Return the one complete receipt-union manifest for every probe prefix."""
    catalog = json.loads(catalog_path.read_text())
    if catalog.get("status") != "FROZEN_NOT_SUBMITTED":
        raise SystemExit(f"manifest catalog is not frozen: {catalog_path}")
    parent = catalog.get("parent_manifest", {})
    if parent.get("rows") != EXPECTED_ROWS_PER_PREFIX:
        raise SystemExit("manifest catalog parent is not the 2,721-row v2 scope")
    selected: dict[int, tuple[Path, str]] = {}
    for entry in catalog.get("availability_classes", []):
        prefix = int(entry.get("prefix", -1))
        if prefix not in PREFIXES or entry.get("rows") != EXPECTED_ROWS_PER_PREFIX:
            continue
        if prefix in selected:
            raise SystemExit(f"multiple complete manifests for prefix {prefix}")
        selected[prefix] = (Path(entry["path"]), str(entry["sha256"]))
    if set(selected) != set(PREFIXES):
        raise SystemExit(
            f"complete manifest prefixes differ: {sorted(selected)} != {list(PREFIXES)}"
        )
    return selected


def main() -> None:
    selected_manifests = select_complete_manifests(MANIFEST_CATALOG)
    manifests = {}
    scene_sets = {}
    streaming_units = set()
    target_all = set()
    target_training = set()
    scene_sources: dict[str, Path] = {}
    for prefix, (path, expected_sha) in sorted(selected_manifests.items()):
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise SystemExit(f"manifest SHA mismatch: {path}: {actual_sha}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != EXPECTED_ROWS_PER_PREFIX:
            raise SystemExit(f"manifest row count mismatch: {path}: {len(rows)}")
        keys = set()
        scenes = set()
        for row in rows:
            scene = str(row["scene_id"])
            if row.get("sample_id") != f"{scene}_w{int(row['streaming_window_id']):03d}_p{prefix}":
                raise SystemExit(f"sample identity mismatch: {row.get('sample_id')}")
            if len(row["streaming_indices"]) != prefix:
                raise SystemExit(f"prefix length mismatch: {row.get('sample_id')}")
            key = (scene, int(row["streaming_window_id"]), prefix)
            if key in keys:
                raise SystemExit(f"duplicate streaming key: {key}")
            keys.add(key)
            streaming_units.add(key)
            scenes.add(scene)
            source = Path(str(row["scene_dir"]).replace("fj:s3://baiqiao/InsScene-15K/data", "/mnt/umm/users/yinbaiqiao/InsScene-15K/data"))
            if source.name != scene or not source.is_dir():
                source = SOURCE / scene
            if not source.is_dir():
                raise SystemExit(f"missing source scene: {scene}: {source}")
            prior = scene_sources.setdefault(scene, source.resolve())
            if prior != source.resolve():
                raise SystemExit(f"scene source conflict: {scene}")
            window = [int(value) for value in row["streaming_window_indices"]]
            anchor = prefix - 1
            for horizon in (0, 1, 2, 4):
                if anchor + horizon < len(window):
                    target_all.add((scene, window[anchor + horizon]))
                    if horizon:
                        target_training.add((scene, window[anchor + horizon]))
        manifests[str(prefix)] = {
            "path": str(path),
            "sha256": actual_sha,
            "rows": len(rows),
        }
        scene_sets[prefix] = scenes

    reference = scene_sets[8]
    if len(reference) != 681 or any(scenes != reference for scenes in scene_sets.values()):
        raise SystemExit("the four frozen manifests do not share the exact 681-scene set")
    ordered_scenes = sorted(reference)
    extractor_source_scenes = {
        path.name
        for path in SOURCE.iterdir()
        if path.is_dir()
        and (path / "images").is_dir()
        and (path / "refined_ins_ids").is_dir()
        and (path / "scene_iphone_metadata.npz").is_file()
    }
    extra_source_scenes = sorted(extractor_source_scenes - reference)

    # The published view is read-only.  Temporarily reopen directories owned by
    # this user so rebuilding the exact same receipt remains idempotent.
    for directory in (ARTIFACT, VIEW.parent.parent, VIEW.parent, VIEW):
        if directory.exists():
            directory.chmod(0o755)
    VIEW.mkdir(parents=True, exist_ok=True)
    for scene in ordered_scenes:
        link = VIEW / scene
        target = scene_sources[scene]
        if link.is_symlink():
            if link.resolve() != target:
                raise SystemExit(f"existing data-view link mismatch: {link}")
        elif link.exists():
            raise SystemExit(f"data-view entry is not a symlink: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    actual_links = sorted(path.name for path in VIEW.iterdir())
    if actual_links != ordered_scenes:
        raise SystemExit("data-view membership differs from frozen manifest scenes")

    identity = {
        "ordered_scene_ids": ordered_scenes,
        "manifest_sha256_by_prefix": {
            prefix: spec["sha256"] for prefix, spec in manifests.items()
        },
        "protocol_sha256": "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f",
        "storage_contract": "MINIMAX_H3_FINITE_BF16_V1",
        "scope": "CANONICAL_V2_COMPLETE_2721",
    }
    receipt = {
        "schema": "MINIMAX_H3_BF16_PRODUCTION_ROUTE_V2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "identity_sha256": canonical_sha(identity),
        "data_view": str(ARTIFACT / "data"),
        "source_scene_root": str(SOURCE),
        "scene_count": len(ordered_scenes),
        "ordered_scene_ids": ordered_scenes,
        "excluded_source_scenes": extra_source_scenes,
        "frozen_manifests": manifests,
        "manifest_catalog": {
            "path": str(MANIFEST_CATALOG),
            "sha256": sha256(MANIFEST_CATALOG),
        },
        "streaming_records": len(streaming_units),
        "streaming_layer_items": len(streaming_units) * 5,
        "target_rows_all_protocol": len(target_all),
        "target_rows_training_required": len(target_training),
        "target_layer_items_all_protocol": len(target_all) * 4,
        "streaming_layers": [10, 20, 30, 40, 49],
        "target_layers": [20, 30, 40, 49],
        "logical_to_disk_layer": {"50": 49},
        "storage_contract": "MINIMAX_H3_FINITE_BF16_V1",
        "identity": identity,
    }
    atomic_json(RECEIPT, receipt)
    for directory in (VIEW, VIEW.parent, VIEW.parent.parent, ARTIFACT):
        directory.chmod(0o555)
    print(json.dumps({
        "receipt": str(RECEIPT),
        "receipt_sha256": sha256(RECEIPT),
        "scene_count": len(ordered_scenes),
        "streaming_records": len(streaming_units),
        "target_rows_all_protocol": len(target_all),
        "excluded_source_scenes": extra_source_scenes,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
