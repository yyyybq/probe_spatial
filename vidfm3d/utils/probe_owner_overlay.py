"""Immutable SHA-bound streaming/target owner overlay for main-v2 probing."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any

from vidfm3d.utils import aoss_fs


class OwnerOverlayError(RuntimeError):
    pass


_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"pid": None, "manifest": None, "stream": None, "target": None}
_VERIFIED: set[tuple[Any, ...]] = set()


def _exact_local_mirror(uri: str) -> Path | None:
    """Resolve an AOSS URI through an explicitly configured exact mirror.

    This is not a fuzzy/path-similarity fallback: only the canonical
    ``aoss://baiqiao/InsScene-15K/`` namespace is accepted, traversal is
    rejected, and ``materialize_verified`` still enforces the DB-bound size,
    header digest and full payload SHA (when present).
    """
    root_name = os.environ.get("PROBE_OWNER_LOCAL_MIRROR_ROOT")
    prefix = "aoss://baiqiao/InsScene-15K/"
    if not root_name or not uri.startswith(prefix):
        return None
    root = Path(root_name)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise OwnerOverlayError(f"local mirror root is not an absolute regular directory: {root}")
    relative = uri[len(prefix):]
    if not relative or any(part in {"", ".", ".."} for part in Path(relative).parts):
        raise OwnerOverlayError(f"invalid canonical mirror relative path: {uri}")
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise OwnerOverlayError(f"local mirror path escaped root: {uri}")
    return candidate if candidate.is_file() and not candidate.is_symlink() else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _regular(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise OwnerOverlayError(f"{label} is not an absolute regular file: {path}")
    return path.resolve()


def _connect(binding: dict[str, Any], label: str) -> sqlite3.Connection:
    path = _regular(Path(str(binding.get("path") or "")), label)
    expected = str(binding.get("sha256") or "")
    actual = _sha256(path)
    if actual != expected:
        raise OwnerOverlayError(f"{label} SHA mismatch: {actual} != {expected}")
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    status = con.execute("SELECT value FROM metadata WHERE key='status'").fetchone()
    if status is None or status[0] != "READY":
        raise OwnerOverlayError(f"{label} database is not READY")
    return con


def _state() -> dict[str, Any]:
    manifest_name = os.environ.get("PROBE_OWNER_OVERLAY_MANIFEST")
    if not manifest_name:
        raise OwnerOverlayError("PROBE_OWNER_OVERLAY_MANIFEST is required")
    pid = os.getpid()
    with _LOCK:
        if _STATE["pid"] == pid and _STATE["manifest"] == manifest_name:
            return _STATE
        path = _regular(Path(manifest_name), "owner overlay manifest")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "CANONICAL_V2_PROBE816_MIXED_OWNER_OVERLAY_V1"
            or manifest.get("status") != "READY"
            or manifest.get("fail_closed") is not True
        ):
            raise OwnerOverlayError("owner overlay manifest is not executable READY")
        for key in ("streaming_database", "target_database"):
            if not isinstance(manifest.get(key), dict):
                raise OwnerOverlayError(f"manifest lacks {key} binding")
        old_stream, old_target = _STATE.get("stream"), _STATE.get("target")
        if old_stream is not None:
            old_stream.close()
        if old_target is not None:
            old_target.close()
        _STATE.update({
            "pid": pid,
            "manifest": manifest_name,
            "stream": _connect(manifest["streaming_database"], "streaming owner"),
            "target": _connect(manifest["target_database"], "target owner"),
            "data": manifest,
        })
        _VERIFIED.clear()
        return _STATE


def resolve_streaming(
    vfm: str, source: str, scene_id: str, layer: int, timestep: int,
    window_id: int, prefix: int,
) -> dict[str, Any]:
    row = _state()["stream"].execute(
        "SELECT * FROM streaming_owners WHERE vfm=? AND source=? AND scene_id=? "
        "AND layer=? AND timestep=? AND window_id=? AND prefix=?",
        (vfm, source, scene_id, int(layer), int(timestep), int(window_id), int(prefix)),
    ).fetchone()
    if row is None:
        raise OwnerOverlayError("no exact streaming owner")
    return dict(row)


def resolve_targets(
    vfm: str, source: str, scene_id: str, layer: int, timestep: int,
    target_raw_ids: list[int],
) -> list[dict[str, Any]]:
    con = _state()["target"]
    result = []
    for raw_id in target_raw_ids:
        row = con.execute(
            "SELECT * FROM target_owners WHERE vfm=? AND source=? AND scene_id=? "
            "AND layer=? AND timestep=? AND target_raw_id=?",
            (vfm, source, scene_id, int(layer), int(timestep), int(raw_id)),
        ).fetchone()
        if row is None or str(row["status"]).startswith("PENDING"):
            raise OwnerOverlayError(f"no exact target owner for raw ID {raw_id}")
        result.append(dict(row))
    return result


def materialize_verified(row: dict[str, Any]) -> str:
    uri = str(row["source_uri"])
    local = _exact_local_mirror(uri)
    if local is None:
        local = Path(aoss_fs.materialize(uri))
    if local.is_symlink() or not local.is_file():
        raise OwnerOverlayError(f"materialized owner is not a regular file: {local}")
    expected_size = int(row["payload_size_bytes"])
    stat = local.stat()
    payload_sha = row.get("payload_sha256")
    key = (str(local), stat.st_size, stat.st_mtime_ns, expected_size,
           row.get("payload_header_sha256"), payload_sha)
    if key in _VERIFIED:
        return str(local)
    if stat.st_size != expected_size:
        raise OwnerOverlayError(f"payload size mismatch for {uri}")
    with local.open("rb") as f:
        raw_length = f.read(8)
        if len(raw_length) != 8:
            raise OwnerOverlayError(f"truncated safetensors payload: {uri}")
        header_size = struct.unpack("<Q", raw_length)[0]
        header = f.read(header_size)
    header_body_sha = hashlib.sha256(header).hexdigest()
    header_envelope_sha = hashlib.sha256(raw_length + header).hexdigest()
    expected_header = str(row.get("payload_header_sha256") or "")
    origin = str(row.get("origin") or "")
    # The remote streaming gate canonically hashes the 8-byte length envelope
    # plus JSON header.  Local re-extraction and target-route audits hash the
    # JSON header body.  Preserve the exact evidence contract per provenance;
    # do not accept an arbitrary one-of-two digest.
    if expected_header not in {"", "GAP_RECEIPT_FULL_SHA"}:
        observed_header = (
            header_envelope_sha
            if origin in {"gate_pass", "metadata_recovery", "minimax_contract_v2"}
            else header_body_sha
        )
        if observed_header != expected_header:
            raise OwnerOverlayError(
                f"payload header SHA mismatch for {uri} (origin={origin})"
            )
    if payload_sha and len(str(payload_sha)) == 64 and _sha256(local) != str(payload_sha):
        raise OwnerOverlayError(f"payload full SHA mismatch for {uri}")
    _VERIFIED.add(key)
    return str(local)


def load_manifest() -> dict[str, Any]:
    """Force manifest/DB verification and return the immutable manifest."""
    return dict(_state()["data"])
