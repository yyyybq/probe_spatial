"""Fail-closed owner catalog for canonical-v2 816 probing.

A catalog is loadable only when it is a regular non-symlink file whose
manifest status is READY and whose route file SHA still matches. Missing
owners, unverified statuses, and dummy/legacy fallbacks are hard errors.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

READY = "READY"
OWNER_ELIGIBLE = frozenset(
    {
        "REFERENCE_SOURCE_READ_ONLY",
        "PASS_REMOTE_VERIFIED",
        "PASS_METADATA_RECOVERED",
    }
)
DISK_LAYER_OVERRIDES = {
    ("lingbot", 48): 47,
    ("minimax_h3", 50): 49,
}


class OwnerCatalogError(ValueError):
    """Catalog bytes, status, or owner identity failed a fail-closed check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def disk_layer(model: str, layer: int) -> int:
    return DISK_LAYER_OVERRIDES.get((model, int(layer)), int(layer))


def streaming_key(
    *,
    vfm: str,
    source: str,
    scene_id: str,
    feature_layer: int,
    timestep: int | None,
    window_id: int,
    prefix: int,
) -> tuple[Any, ...]:
    return (
        "streaming",
        str(vfm),
        str(source),
        str(scene_id),
        int(feature_layer),
        None if timestep is None else int(timestep),
        int(window_id),
        int(prefix),
    )


def target_key(
    *,
    vfm: str,
    source: str,
    scene_id: str,
    feature_layer: int,
    timestep: int | None,
    canonical_row_index: int,
    target_raw_id: int,
) -> tuple[Any, ...]:
    return (
        "target",
        str(vfm),
        str(source),
        str(scene_id),
        int(feature_layer),
        None if timestep is None else int(timestep),
        int(canonical_row_index),
        int(target_raw_id),
    )


def _regular(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise OwnerCatalogError(
            f"{label} must be an absolute regular non-symlink file: {path}"
        )
    return path.resolve()


def load_owner_catalog(path: str | Path) -> dict[str, Any]:
    manifest_path = _regular(Path(path), "owner catalog manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OwnerCatalogError(f"cannot parse owner catalog {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise OwnerCatalogError("owner catalog manifest must be a JSON object")
    if manifest.get("schema") != "canonical_v2_probe_owner_overlay_v1":
        raise OwnerCatalogError("owner catalog schema mismatch")
    if manifest.get("status") != READY:
        raise OwnerCatalogError(
            f"owner catalog is not READY ({manifest.get('status')}: "
            f"{(manifest.get('readiness') or {}).get('reason')})"
        )
    if manifest.get("fail_closed") is not True or manifest.get("audit_only") is True:
        raise OwnerCatalogError("owner catalog is not an executable fail-closed overlay")
    route_meta = manifest.get("route_file")
    if not isinstance(route_meta, dict):
        raise OwnerCatalogError("owner catalog route_file binding is missing")
    route_path = _regular(Path(str(route_meta["path"])), "owner catalog routes")
    recorded_sha = str(route_meta.get("sha256") or "")
    actual_sha = sha256_file(route_path)
    if recorded_sha != actual_sha:
        raise OwnerCatalogError(
            f"owner catalog route SHA mismatch: expected {recorded_sha}, found {actual_sha}"
        )
    owners: dict[tuple[Any, ...], dict[str, Any]] = {}
    opener = gzip.open if route_path.name.endswith(".gz") else open
    with opener(route_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise OwnerCatalogError(f"owner catalog has a blank route at line {line_number}")
            row = json.loads(line)
            key = tuple(row["key"])
            status = row.get("status")
            uri = row.get("source_uri")
            if status not in OWNER_ELIGIBLE or not uri:
                raise OwnerCatalogError(
                    f"owner catalog route {line_number} is not an eligible unique owner"
                )
            if key in owners:
                raise OwnerCatalogError(f"duplicate owner key: {key}")
            owners[key] = row
    expected_rows = int(route_meta.get("rows", -1))
    if expected_rows != len(owners):
        raise OwnerCatalogError(
            f"owner catalog row count mismatch: expected {expected_rows}, found {len(owners)}"
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "owners": owners,
    }


def resolve_owner(
    catalog: Mapping[str, Any],
    key: tuple[Any, ...],
    *,
    label: str,
) -> str:
    owners = catalog.get("owners")
    if not isinstance(owners, dict):
        raise OwnerCatalogError("owner catalog has not been loaded")
    row = owners.get(key)
    if row is None:
        raise OwnerCatalogError(f"no unique owner for {label}: {key}")
    uri = row.get("source_uri")
    if not uri or row.get("status") not in OWNER_ELIGIBLE:
        raise OwnerCatalogError(f"unverified owner for {label}: {key}")
    return str(uri)
