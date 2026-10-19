#!/usr/bin/env python3
"""Extend the frozen Wan p8 exact h1 target cache to exact h1/h2/h4.

This is deliberately a target-only recovery tool.  It never writes to the
source h1 or streaming roots.  For every selected scene it:

1. derives h1/h2/h4 directly from the frozen manifest window indices;
2. verifies every reused h1 tensor against its sidecar, SHA256, identity,
   shape, dtype, and finite-value contract;
3. forwards only the disjoint h2/h4 frames with the pinned Wan extractor;
4. atomically publishes a new, flat, scene-level h1/h2/h4 target cache.

The output namespace is fail-closed.  Existing scene output, claims, shard
markers, or a root completion marker are fatal.  A crashed invocation leaves
an explicit claim/incomplete path and must not be resumed in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing this module is safe: Wan/diffusers is imported only inside its
# main(), while these are the exact, already-audited extraction primitives.
from scripts import extract_wan_manifest_exact as base  # noqa: E402


CANONICAL_MANIFEST_SHA256 = (
    "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
)
BASE_EXTRACTOR_SHA256 = (
    "654644424e94ea8c9a1efecab5e1a65abf37321045d9b97a1f53e9bd668353d0"
)
SOURCE_H1_AUDIT_SHA256 = (
    "c09a11b6b60d97d162d264365e46339b9d1626d40b80665c18fa4783dffc4010"
)
EXPECTED_GIT_COMMIT = "2075f3eb270cb0e8907c3927b6b079c64965b75a"
MODEL_SNAPSHOT = Path(
    "/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/"
    "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/"
    "0fad780a534b6463e45facd96134c9f345acfa5b"
)
LAYERS = (0, 5, 10, 15, 25, 29)
HORIZONS = (1, 2, 4)
EXTRACT_HORIZONS = (2, 4)
FEATURE_SHAPE = (30, 52, 1536)
INPUT_FRAMES = 81
TIMESTEP = 749
BASE_SEED = 42
PROMPT = "."
EXPECTED_ROWS = 2724
EXPECTED_SCENES = 681
EXPECTED_SPLITS = {"train": 2448, "val": 276}
ROOT_COMPLETE = "_COMPLETE.json"


class ContractError(RuntimeError):
    """Raised when an identity, lineage, or artifact gate fails."""


@dataclass(frozen=True)
class Contract:
    manifest_sha256: str = CANONICAL_MANIFEST_SHA256
    base_extractor_sha256: str = BASE_EXTRACTOR_SHA256
    source_audit_sha256: str = SOURCE_H1_AUDIT_SHA256
    layers: tuple[int, ...] = LAYERS
    horizons: tuple[int, ...] = HORIZONS
    extract_horizons: tuple[int, ...] = EXTRACT_HORIZONS
    feature_shape: tuple[int, ...] = FEATURE_SHAPE
    input_frames: int = INPUT_FRAMES
    timestep: int = TIMESTEP
    seed: int = BASE_SEED
    prompt: str = PROMPT


@dataclass
class SourceScene:
    scene: str
    source_root: Path
    target_ids: list[int]
    tensors: dict[int, torch.Tensor]
    artifact_sha256: dict[str, str]
    feature_sha256: dict[int, str]
    index_sha256: str


@dataclass(frozen=True)
class GitIdentity:
    repo_root: str
    commit: str
    verification_source: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise ContractError(f"Unreadable JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Expected JSON object: {path}")
    return value


def verify_git_identity(repo_root: Path, expected_commit: str) -> GitIdentity:
    """Resolve HEAD directly from .git without depending on the git CLI."""
    if not repo_root.is_absolute():
        raise ContractError(f"Repository root must be absolute: {repo_root}")
    try:
        resolved_root = repo_root.resolve(strict=True)
    except Exception as exc:
        raise ContractError(f"Unreadable repository root: {repo_root}: {exc}") from exc
    if resolved_root != repo_root:
        raise ContractError(
            f"Repository root must be its real canonical path: {repo_root} != "
            f"{resolved_root}"
        )
    if expected_commit != EXPECTED_GIT_COMMIT:
        raise ContractError(
            f"Expected git commit argument is not pinned: {expected_commit!r} != "
            f"{EXPECTED_GIT_COMMIT}"
        )
    git_dir = resolved_root / ".git"
    head_path = git_dir / "HEAD"
    if git_dir.is_symlink() or not git_dir.is_dir():
        raise ContractError(f"Repository .git is not a real directory: {git_dir}")
    if head_path.is_symlink() or not head_path.is_file():
        raise ContractError(f"Repository HEAD is not a real file: {head_path}")
    try:
        head_value = head_path.read_text().strip()
    except Exception as exc:
        raise ContractError(f"Unreadable repository HEAD: {head_path}: {exc}") from exc

    if head_value.startswith("ref: "):
        reference = head_value[5:]
        if not re.fullmatch(r"refs/[A-Za-z0-9._/-]+", reference) or ".." in reference:
            raise ContractError(f"Unsafe or malformed HEAD reference: {reference!r}")
        ref_path = git_dir / reference
        if ref_path.is_file() and not ref_path.is_symlink():
            commit = ref_path.read_text().strip()
            source = f"direct:.git/HEAD->.git/{reference}"
        else:
            packed_path = git_dir / "packed-refs"
            if packed_path.is_symlink() or not packed_path.is_file():
                raise ContractError(f"Unreadable HEAD reference: {ref_path}")
            matches = []
            for line in packed_path.read_text().splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                pieces = line.split(" ", 1)
                if len(pieces) == 2 and pieces[1] == reference:
                    matches.append(pieces[0])
            if len(matches) != 1:
                raise ContractError(
                    f"HEAD reference {reference!r} did not resolve uniquely in packed-refs"
                )
            commit = matches[0]
            source = f"direct:.git/HEAD->.git/packed-refs:{reference}"
    else:
        commit = head_value
        source = "direct:.git/HEAD(detached)"
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ContractError(f"Malformed git commit recovered from {source}: {commit!r}")
    if commit != expected_commit:
        raise ContractError(
            f"Repository commit mismatch: {commit} != expected {expected_commit}"
        )
    return GitIdentity(
        repo_root=str(resolved_root),
        commit=commit,
        verification_source=source,
    )


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    """Create a persistent claim with O_EXCL semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ContractError(f"Refusing existing claim/marker: {path}") from exc


def rename_noreplace(source: Path, destination: Path) -> None:
    """Publish a private file or flat directory without replacement.

    The project filesystem rejects ``renameat2(RENAME_NOREPLACE)``. Files use
    a same-directory hard link as their atomic validity point. Directories
    first reserve their final name with atomic ``mkdir`` and then hard-link
    every already-fsynced regular child. A crash may leave an explicit partial
    reserved directory, but never overwrites an existing path; the separate
    completion marker remains absent and all retries fail closed.
    """
    if source.is_symlink():
        raise ContractError(f"Refusing symlink publication source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        try:
            os.link(source, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise ContractError(
                f"Refusing existing no-replace publication target: {destination}"
            ) from exc
        except OSError as exc:
            raise ContractError(
                f"Atomic hard-link publication failed {source} -> {destination}: {exc}"
            ) from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        source.unlink()
        return

    if not source.is_dir():
        raise ContractError(
            f"Publication source is not a regular file/directory: {source}"
        )
    entries = sorted(source.iterdir(), key=lambda item: item.name)
    invalid = [
        item.name for item in entries if item.is_symlink() or not item.is_file()
    ]
    if invalid:
        raise ContractError(
            f"Directory publication has non-regular entries: {invalid}"
        )
    for item in entries:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(item, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    try:
        destination.mkdir(mode=0o755)
    except FileExistsError as exc:
        raise ContractError(
            f"Refusing existing no-replace publication target: {destination}"
        ) from exc
    try:
        for item in entries:
            target = destination / item.name
            try:
                os.link(item, target, follow_symlinks=False)
            except FileExistsError as exc:
                raise ContractError(
                    f"Refusing existing no-replace publication target: {target}"
                ) from exc
        destination_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        for item in entries:
            item.unlink()
        source.rmdir()
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        # Preserve both the private source and the reserved destination as
        # evidence. No completion marker can be published by the caller.
        raise


def atomic_json_noreplace(path: Path, payload: dict[str, Any]) -> None:
    """Durably stage JSON, then atomically publish without replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.incomplete-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        rename_noreplace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        # The unique unpublished temporary is ours; preserve any competing
        # final path byte-for-byte and remove only our private staging file.
        temporary.unlink(missing_ok=True)
        raise


def load_native_int64(path: Path, expected_length: int) -> list[int]:
    """Load an index vector and reject on-disk dtype coercion."""
    try:
        values = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ContractError(f"Unreadable target index {path}: {exc}") from exc
    if values.dtype != np.dtype(np.int64):
        raise ContractError(
            f"Target index native dtype mismatch: {path} has {values.dtype}, "
            "expected int64"
        )
    if values.ndim != 1 or values.shape != (expected_length,):
        raise ContractError(
            f"Target index native shape mismatch: {path} has {values.shape}, "
            f"expected ({expected_length},)"
        )
    return [int(value) for value in values.tolist()]


def wrapper_sha256() -> str:
    return sha256(Path(__file__).resolve())


def ensure_pinned_base_extractor(contract: Contract) -> None:
    path = Path(base.__file__).resolve()
    actual = sha256(path)
    if actual != contract.base_extractor_sha256:
        raise ContractError(
            f"Pinned base extractor mismatch: {path} {actual} != "
            f"{contract.base_extractor_sha256}"
        )
    if base.CANONICAL_SHA256 != contract.manifest_sha256:
        raise ContractError("Base extractor canonical manifest SHA changed")
    if int(base.INPUT_FRAMES) != contract.input_frames:
        raise ContractError(
            f"Base extractor INPUT_FRAMES={base.INPUT_FRAMES}, expected "
            f"effective value {contract.input_frames}"
        )
    if tuple(base.DEFAULT_LAYERS) != contract.layers:
        raise ContractError("Base extractor layer contract changed")


def scene_id(row: dict[str, Any]) -> str:
    return str(row.get("scene_id") or Path(str(row["scene_dir"])).name)


def direct_target_raw_id(row: dict[str, Any], horizon: int) -> tuple[int, str]:
    """Derive a target only from window[tail+h], ignoring shortcut fields."""
    horizon = int(horizon)
    if horizon not in HORIZONS:
        raise ContractError(f"Unsupported target horizon: {horizon}")
    try:
        tail = int(row["streaming_tail"])
        window = [int(value) for value in row["streaming_window_indices"]]
    except Exception as exc:
        raise ContractError(f"Malformed streaming window for {row.get('sample_id')}") from exc
    position = tail + horizon
    if position < 0 or position >= len(window):
        raise ContractError(
            f"{row.get('sample_id')}: target position {position} outside "
            f"window length {len(window)}"
        )
    raw_id = int(window[position])
    names = row.get("valid_frames")
    if not isinstance(names, list) or raw_id < 0 or raw_id >= len(names):
        raise ContractError(
            f"{row.get('sample_id')}: target raw ID {raw_id} outside valid_frames"
        )
    return raw_id, f"streaming_window_indices[{position}]"


def grouped_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[scene_id(row)].append(row)
    return {scene: grouped[scene] for scene in sorted(grouped)}


def validate_scene_rows(
    scene: str, rows: list[dict[str, Any]], contract: Contract
) -> dict[int, tuple[int, dict[str, Any], str]]:
    if len(rows) != 4:
        raise ContractError(f"{scene}: expected all four manifest rows, got {len(rows)}")
    if len({str(row["sample_id"]) for row in rows}) != 4:
        raise ContractError(f"{scene}: duplicate sample IDs")
    mapping: dict[int, tuple[int, dict[str, Any], str]] = {}
    by_horizon: dict[int, set[int]] = {horizon: set() for horizon in contract.horizons}
    for row in rows:
        if scene_id(row) != scene:
            raise ContractError(f"Scene grouping mismatch for {scene}")
        tail = int(row["streaming_tail"])
        prefix = [int(value) for value in row["streaming_indices"]]
        window = [int(value) for value in row["streaming_window_indices"]]
        if prefix != window[: tail + 1]:
            raise ContractError(f"{row['sample_id']}: prefix is not window[:tail+1]")
        for horizon in contract.horizons:
            raw_id, source = direct_target_raw_id(row, horizon)
            if raw_id in mapping:
                old_horizon, old_row, _ = mapping[raw_id]
                raise ContractError(
                    f"{scene}: raw target {raw_id} reused by "
                    f"h{old_horizon}/{old_row['sample_id']} and "
                    f"h{horizon}/{row['sample_id']}"
                )
            mapping[raw_id] = (horizon, row, source)
            by_horizon[horizon].add(raw_id)
    for horizon, target_ids in by_horizon.items():
        if len(target_ids) != 4:
            raise ContractError(f"{scene}: h{horizon} expected four targets")
    if len(mapping) != 12:
        raise ContractError(f"{scene}: expected 12 disjoint h124 targets")
    return mapping


def load_manifest_selection(
    manifest: Path,
    selection_file: Path | None,
    contract: Contract,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    manifest = manifest.resolve()
    actual_sha = sha256(manifest)
    if actual_sha != contract.manifest_sha256:
        raise ContractError(
            f"Canonical manifest SHA mismatch: {actual_sha} != {contract.manifest_sha256}"
        )
    rows = base.read_rows(manifest)
    if len(rows) != EXPECTED_ROWS:
        raise ContractError(f"Expected {EXPECTED_ROWS} manifest rows, got {len(rows)}")
    if len({str(row["sample_id"]) for row in rows}) != EXPECTED_ROWS:
        raise ContractError("Canonical manifest has duplicate sample IDs")
    all_groups = grouped_rows(rows)
    if len(all_groups) != EXPECTED_SCENES:
        raise ContractError(f"Expected {EXPECTED_SCENES} scenes, got {len(all_groups)}")
    if dict(Counter(str(row["split"]) for row in rows)) != EXPECTED_SPLITS:
        raise ContractError("Canonical split counts changed")
    for scene, scene_rows in all_groups.items():
        validate_scene_rows(scene, scene_rows, contract)

    if selection_file is None:
        selected_rows = rows
    else:
        selected = base.read_selection(selection_file.resolve())
        if selected is None:
            raise ContractError("Selection unexpectedly resolved to None")
        known = {str(row["sample_id"]) for row in rows}
        missing = selected - known
        if missing:
            raise ContractError(f"Selection contains unknown IDs: {sorted(missing)[:3]}")
        selected_rows = [row for row in rows if str(row["sample_id"]) in selected]
        if len(selected_rows) != len(selected):
            raise ContractError("Selection did not resolve one-to-one")
    selected_groups = grouped_rows(selected_rows)
    for scene, scene_rows in selected_groups.items():
        # This rejects partial-scene shards and is essential for a flat cache.
        if len(scene_rows) != len(all_groups[scene]):
            raise ContractError(f"Selection contains a partial scene: {scene}")
        validate_scene_rows(scene, scene_rows, contract)
    if not selected_groups:
        raise ContractError("Selection is empty")
    return rows, selected_groups


def validate_source_audit(
    audit_path: Path,
    source_h1_root: Path,
    contract: Contract,
) -> dict[str, Any]:
    audit_path = audit_path.resolve()
    actual_sha = sha256(audit_path)
    if actual_sha != contract.source_audit_sha256:
        raise ContractError(
            f"Source h1 audit SHA mismatch: {actual_sha} != {contract.source_audit_sha256}"
        )
    audit = read_json(audit_path)
    expected_root = source_h1_root.resolve()
    if audit.get("ok") is not True:
        raise ContractError("Source h1 full feature audit is not PASS")
    if audit.get("manifest_sha256") != contract.manifest_sha256:
        raise ContractError("Source h1 audit manifest SHA mismatch")
    if Path(str(audit.get("target_root", ""))).resolve() != expected_root:
        raise ContractError("Source h1 audit target root mismatch")
    if int(audit.get("target_exact_raw", -1)) != EXPECTED_ROWS:
        raise ContractError("Source h1 audit target coverage mismatch")
    if int(audit.get("problem_count", -1)) != 0:
        raise ContractError("Source h1 audit records problems")
    expected_counts = {
        "target_indices": EXPECTED_SCENES,
        "target_sft": EXPECTED_SCENES * len(contract.layers),
        "target_sidecars": EXPECTED_SCENES * len(contract.layers),
    }
    actual_counts = audit.get("file_counts") or {}
    for key, expected in expected_counts.items():
        if int(actual_counts.get(key, -1)) != expected:
            raise ContractError(
                f"Source h1 audit {key}={actual_counts.get(key)!r}, expected {expected}"
            )
    return audit


def validate_artifact_sidecar(
    artifact: Path,
    required: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    sidecar = Path(f"{artifact}.manifest.json")
    if not artifact.is_file() or not sidecar.is_file():
        raise ContractError(f"Missing artifact or sidecar: {artifact}")
    metadata = read_json(sidecar)
    artifact_sha = sha256(artifact)
    if metadata.get("sha256") != artifact_sha:
        raise ContractError(f"Artifact SHA does not match sidecar: {artifact}")
    if int(metadata.get("size_bytes", -1)) != artifact.stat().st_size:
        raise ContractError(f"Artifact size does not match sidecar: {artifact}")
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ContractError(
                f"{artifact}: sidecar field {key}={metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
    return metadata, artifact_sha, sha256(sidecar)


def expected_scene_details(
    scene: str,
    rows: list[dict[str, Any]],
    contract: Contract,
) -> tuple[
    dict[int, tuple[int, dict[str, Any], str]],
    list[int],
    list[int],
    list[str],
]:
    mapping = validate_scene_rows(scene, rows, contract)
    h1_ids = sorted(raw for raw, (horizon, _, _) in mapping.items() if horizon == 1)
    all_ids = sorted(mapping)
    sample_ids = sorted(str(row["sample_id"]) for row in rows)
    return mapping, h1_ids, all_ids, sample_ids


INPUT_IMAGE_RECORD_FIELDS = {
    "path",
    "realpath",
    "is_symlink",
    "size_bytes",
    "sha256",
    "scene",
    "sample_id",
    "horizon",
    "raw_id",
    "frame_name",
}


def stable_file_sha256_and_size(path: Path, label: str) -> tuple[str, int]:
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ContractError(f"{label}: file changed while hashing")
    return digest, after.st_size


def snapshot_extracted_input_image(
    data_root: Path,
    scene: str,
    row: dict[str, Any],
    raw_id: int,
    horizon: int,
) -> dict[str, Any]:
    names = base.frame_names(row)
    if raw_id < 0 or raw_id >= len(names):
        raise ContractError(f"{scene}/{raw_id}: input frame index is outside valid_frames")
    frame_name = names[raw_id]
    relative = Path(frame_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"{scene}/{raw_id}: unsafe input frame name {frame_name!r}")
    resolved_data_root = data_root.resolve(strict=True)
    path = (base.local_scene_dir(resolved_data_root, row) / "images" / relative).absolute()
    try:
        realpath = path.resolve(strict=True)
    except Exception as exc:
        raise ContractError(f"{scene}/{raw_id}: missing input image {path}: {exc}") from exc
    if not path.is_file() or not realpath.is_file():
        raise ContractError(f"{scene}/{raw_id}: input image is not a regular file: {path}")
    try:
        realpath.relative_to(resolved_data_root)
    except ValueError as exc:
        raise ContractError(f"{scene}/{raw_id}: input image escapes data root") from exc
    input_sha, input_size = stable_file_sha256_and_size(
        path, f"{scene}/{raw_id} input image"
    )
    if path.resolve(strict=True) != realpath:
        raise ContractError(f"{scene}/{raw_id}: input realpath changed while hashing")
    return {
        "path": str(path),
        "realpath": str(realpath),
        "is_symlink": path.is_symlink(),
        "size_bytes": input_size,
        "sha256": input_sha,
        "scene": scene,
        "sample_id": str(row["sample_id"]),
        "horizon": int(horizon),
        "raw_id": int(raw_id),
        "frame_name": frame_name,
    }


def validate_extracted_input_images(
    records: dict[str, Any],
    mapping: dict[int, tuple[int, dict[str, Any], str]],
    scene: str,
    contract: Contract,
) -> dict[str, dict[str, Any]]:
    expected_ids = sorted(
        raw_id
        for raw_id, (horizon, _, _) in mapping.items()
        if horizon in contract.extract_horizons
    )
    if not isinstance(records, dict) or set(records) != {
        str(raw_id) for raw_id in expected_ids
    }:
        raise ContractError(f"{scene}: extracted input image identity set mismatch")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_id in expected_ids:
        key = str(raw_id)
        record = records[key]
        if not isinstance(record, dict) or set(record) != INPUT_IMAGE_RECORD_FIELDS:
            raise ContractError(f"{scene}/{raw_id}: malformed extracted input record")
        horizon, row, _ = mapping[raw_id]
        expected_identity = {
            "scene": scene,
            "sample_id": str(row["sample_id"]),
            "horizon": int(horizon),
            "raw_id": int(raw_id),
            "frame_name": str(row["valid_frames"][raw_id]),
        }
        for field, expected in expected_identity.items():
            if record.get(field) != expected:
                raise ContractError(
                    f"{scene}/{raw_id}: extracted input {field} mismatch"
                )
        path_value = record.get("path")
        realpath_value = record.get("realpath")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise ContractError(f"{scene}/{raw_id}: input path is not absolute")
        path = Path(path_value)
        try:
            actual_realpath = path.resolve(strict=True)
        except Exception as exc:
            raise ContractError(f"{scene}/{raw_id}: input path disappeared: {exc}") from exc
        if realpath_value != str(actual_realpath):
            raise ContractError(f"{scene}/{raw_id}: input realpath mismatch")
        if record.get("is_symlink") != path.is_symlink():
            raise ContractError(f"{scene}/{raw_id}: input symlink status mismatch")
        if not path.is_file() or not actual_realpath.is_file():
            raise ContractError(f"{scene}/{raw_id}: input is no longer a regular file")
        actual_sha, actual_size = stable_file_sha256_and_size(
            path, f"{scene}/{raw_id} input image"
        )
        if path.resolve(strict=True) != actual_realpath:
            raise ContractError(f"{scene}/{raw_id}: input realpath changed while hashing")
        if record.get("size_bytes") != actual_size:
            raise ContractError(f"{scene}/{raw_id}: input size changed")
        if record.get("sha256") != actual_sha:
            raise ContractError(f"{scene}/{raw_id}: input SHA256 changed")
        normalized[key] = dict(record)
    return normalized


def validate_source_h1_scene(
    source_h1_root: Path,
    scene: str,
    rows: list[dict[str, Any]],
    model_id: Path,
    contract: Contract,
) -> SourceScene:
    mapping, h1_ids, _, sample_ids = expected_scene_details(scene, rows, contract)
    scene_root = source_h1_root / "wan" / "scannetpp" / scene
    index_path = scene_root / "target_indices.npy"
    common_required = {
        "canonical_manifest_sha256": contract.manifest_sha256,
        "extractor_sha256": contract.base_extractor_sha256,
        "mode": "manifest_exact_target",
        "model_id": str(model_id.resolve()),
        "prompt": contract.prompt,
        "timestep": contract.timestep,
        "layers": list(contract.layers),
        "num_frames": contract.input_frames,
        "base_seed": contract.seed,
        "target_raw_ids": h1_ids,
        "scene_id": scene,
        "sample_ids": sample_ids,
    }
    _, index_sha, index_sidecar_sha = validate_artifact_sidecar(
        index_path,
        {**common_required, "shape": [4], "dtype": "int64"},
    )
    indices = load_native_int64(index_path, 4)
    if indices != h1_ids:
        raise ContractError(f"{scene}: h1 target index mismatch")

    expected_frame_names = [
        str(mapping[raw_id][1]["valid_frames"][raw_id]) for raw_id in h1_ids
    ]
    expected_sources = {str(raw_id): mapping[raw_id][2] for raw_id in h1_ids}
    expected_seeds = {
        str(raw_id): base.deterministic_seed(
            contract.seed, f"target:{scene}:{raw_id}"
        )
        for raw_id in h1_ids
    }

    tensors: dict[int, torch.Tensor] = {}
    artifacts = {
        str(index_path.resolve()): index_sha,
        str(Path(f"{index_path}.manifest.json").resolve()): index_sidecar_sha,
    }
    feature_sha: dict[int, str] = {}
    for layer in contract.layers:
        path = scene_root / f"feature_t{contract.timestep}_layer{layer}.sft"
        metadata, file_sha, sidecar_sha = validate_artifact_sidecar(
            path,
            {
                **common_required,
                "target_frame_names": expected_frame_names,
                "target_id_sources": expected_sources,
                "sample_seeds": expected_seeds,
                "tensors": {
                    "feat": {
                        "shape": [4, *contract.feature_shape],
                        "dtype": "torch.float16",
                    }
                },
            },
        )
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if list(handle.keys()) != ["feat"]:
                raise ContractError(f"Unexpected safetensors keys: {path}")
            tensor_slice = handle.get_slice("feat")
            if tuple(tensor_slice.get_shape()) != (4, *contract.feature_shape):
                raise ContractError(f"Unexpected h1 tensor shape: {path}")
            if str(tensor_slice.get_dtype()) != "F16":
                raise ContractError(f"Unexpected h1 tensor dtype: {path}")
        tensor = load_safetensors(str(path), device="cpu")["feat"]
        if not torch.isfinite(tensor).all().item():
            raise ContractError(f"Non-finite h1 tensor: {path}")
        if metadata["sha256"] != file_sha:
            raise ContractError(f"Internal h1 SHA inconsistency: {path}")
        tensors[layer] = tensor
        feature_sha[layer] = file_sha
        artifacts[str(path.resolve())] = file_sha
        artifacts[str(Path(f"{path}.manifest.json").resolve())] = sidecar_sha
    return SourceScene(
        scene=scene,
        source_root=source_h1_root.resolve(),
        target_ids=h1_ids,
        tensors=tensors,
        artifact_sha256=artifacts,
        feature_sha256=feature_sha,
        index_sha256=index_sha,
    )


def assert_source_unchanged(source: SourceScene) -> None:
    for path_string, expected_sha in source.artifact_sha256.items():
        path = Path(path_string)
        actual = sha256(path)
        if actual != expected_sha:
            raise ContractError(
                f"Source h1 artifact changed during extraction: {path} "
                f"{actual} != {expected_sha}"
            )


def collect_new_features(
    scene: str,
    rows: list[dict[str, Any]],
    forward_one: Callable[[dict[str, Any], int], dict[int, torch.Tensor]],
    contract: Contract,
) -> dict[int, dict[int, torch.Tensor]]:
    mapping = validate_scene_rows(scene, rows, contract)
    requested = sorted(
        raw_id
        for raw_id, (horizon, _, _) in mapping.items()
        if horizon in contract.extract_horizons
    )
    if len(requested) != 8:
        raise ContractError(f"{scene}: expected eight new h2/h4 targets")
    outputs: dict[int, dict[int, torch.Tensor]] = {}
    for raw_id in requested:
        horizon, row, _ = mapping[raw_id]
        if horizon == 1:
            raise ContractError("Internal error: attempted to forward h1")
        values = forward_one(row, raw_id)
        if set(values) != set(contract.layers):
            raise ContractError(f"{scene}/{raw_id}: Wan output layer set mismatch")
        checked: dict[int, torch.Tensor] = {}
        for layer in contract.layers:
            value = values[layer].detach().cpu()
            if tuple(value.shape) != contract.feature_shape:
                raise ContractError(
                    f"{scene}/{raw_id}/layer{layer}: shape {tuple(value.shape)} "
                    f"!= {contract.feature_shape}"
                )
            if not torch.isfinite(value).all().item():
                raise ContractError(f"{scene}/{raw_id}/layer{layer}: non-finite output")
            checked[layer] = value
        outputs[raw_id] = checked
    return outputs


def merged_layer_tensor(
    source: SourceScene,
    new_features: dict[int, dict[int, torch.Tensor]],
    all_ids: list[int],
    layer: int,
) -> torch.Tensor:
    by_raw: dict[int, torch.Tensor] = {
        raw_id: source.tensors[layer][position]
        for position, raw_id in enumerate(source.target_ids)
    }
    for raw_id, values in new_features.items():
        if raw_id in by_raw:
            raise ContractError(f"New target collides with reused h1: {raw_id}")
        by_raw[raw_id] = values[layer]
    if set(by_raw) != set(all_ids):
        raise ContractError("Merged target identity set mismatch")
    merged = torch.stack([by_raw[raw_id] for raw_id in all_ids]).half().contiguous()
    if tuple(merged.shape) != (12, *source.tensors[layer].shape[1:]):
        raise ContractError(f"Merged layer {layer} shape mismatch")
    if not torch.isfinite(merged).all().item():
        raise ContractError(f"Merged layer {layer} contains non-finite values")
    return merged


def common_output_metadata(
    manifest: Path,
    source_h1_root: Path,
    source_audit: Path,
    model_id: Path,
    contract: Contract,
    git_identity: GitIdentity,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "frame_index_schema": "canonical_manifest_exact_h124_v1",
        "mode": "manifest_exact_target_h124_extend",
        "canonical_manifest": str(manifest.resolve()),
        "canonical_manifest_sha256": contract.manifest_sha256,
        "source_h1_root": str(source_h1_root.resolve()),
        "source_h1_audit": str(source_audit.resolve()),
        "source_h1_audit_sha256": contract.source_audit_sha256,
        "base_extractor": str(Path(base.__file__).resolve()),
        "base_extractor_sha256": contract.base_extractor_sha256,
        "wrapper": str(Path(__file__).resolve()),
        "wrapper_sha256": wrapper_sha256(),
        "model_id": str(model_id.resolve()),
        "prompt": contract.prompt,
        "timestep": contract.timestep,
        "layers": list(contract.layers),
        "num_frames": contract.input_frames,
        "effective_input_frames": contract.input_frames,
        "legacy_extractor_config_input_frames": 24,
        "legacy_config_discrepancy": (
            "legacy extractor_config.json said 24; pinned code and h1 sidecars say 81"
        ),
        "base_seed": contract.seed,
        "seed_scheme": "sha256(base_seed:target:scene:raw_id)[:31bit]",
        "reused_horizons": [1],
        "extracted_horizons": list(contract.extract_horizons),
        "motion_window_generator_used": False,
        "repo_root": git_identity.repo_root,
        "git_commit": git_identity.commit,
        "git_commit_verified": True,
        "git_commit_verification_source": git_identity.verification_source,
        "git_dirty": None,
    }


def output_scene_required_files(scene_root: Path, contract: Contract) -> set[str]:
    names = {"target_indices.npy", "target_indices.npy.manifest.json"}
    for layer in contract.layers:
        name = f"feature_t{contract.timestep}_layer{layer}.sft"
        names.add(name)
        names.add(f"{name}.manifest.json")
    return names


def validate_output_scene(
    scene_root: Path,
    scene: str,
    rows: list[dict[str, Any]],
    source: SourceScene,
    model_id: Path,
    contract: Contract,
    git_identity: GitIdentity,
    extracted_input_images_by_raw_id: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if scene_root.is_symlink() or not scene_root.is_dir():
        raise ContractError(f"{scene}: output scene root is not a real directory")
    mapping, _, all_ids, sample_ids = expected_scene_details(scene, rows, contract)
    entries = list(scene_root.iterdir())
    actual_names = {path.name for path in entries}
    expected_names = output_scene_required_files(scene_root, contract)
    if actual_names != expected_names:
        raise ContractError(
            f"{scene}: output entries differ; missing={sorted(expected_names-actual_names)}, "
            f"extra={sorted(actual_names-expected_names)}"
        )
    non_regular = [
        path.name for path in entries if path.is_symlink() or not path.is_file()
    ]
    if non_regular:
        raise ContractError(
            f"{scene}: output contains symlink/non-file entries: {sorted(non_regular)}"
        )
    frame_names = [
        str(mapping[raw_id][1]["valid_frames"][raw_id]) for raw_id in all_ids
    ]
    id_sources = {str(raw_id): mapping[raw_id][2] for raw_id in all_ids}
    sample_by_raw = {
        str(raw_id): str(mapping[raw_id][1]["sample_id"]) for raw_id in all_ids
    }
    sample_seeds = {
        str(raw_id): base.deterministic_seed(contract.seed, f"target:{scene}:{raw_id}")
        for raw_id in all_ids
    }
    index_path = scene_root / "target_indices.npy"
    if extracted_input_images_by_raw_id is None:
        extracted_input_images_by_raw_id = read_json(
            Path(f"{index_path}.manifest.json")
        ).get("extracted_input_images_by_raw_id")
    extracted_inputs = validate_extracted_input_images(
        extracted_input_images_by_raw_id, mapping, scene, contract
    )
    required = {
        "schema_version": 3,
        "frame_index_schema": "canonical_manifest_exact_h124_v1",
        "canonical_manifest_sha256": contract.manifest_sha256,
        "source_h1_root": str(source.source_root),
        "source_h1_audit_sha256": contract.source_audit_sha256,
        "base_extractor_sha256": contract.base_extractor_sha256,
        "wrapper_sha256": wrapper_sha256(),
        "mode": "manifest_exact_target_h124_extend",
        "model_id": str(model_id.resolve()),
        "prompt": contract.prompt,
        "timestep": contract.timestep,
        "layers": list(contract.layers),
        "num_frames": contract.input_frames,
        "effective_input_frames": contract.input_frames,
        "base_seed": contract.seed,
        "reused_horizons": [1],
        "extracted_horizons": list(contract.extract_horizons),
        "repo_root": git_identity.repo_root,
        "git_commit": git_identity.commit,
        "git_commit_verified": True,
        "git_commit_verification_source": git_identity.verification_source,
        "extracted_input_images_by_raw_id": extracted_inputs,
        "target_raw_ids": all_ids,
        "target_horizon_by_raw_id": {
            str(raw_id): int(mapping[raw_id][0]) for raw_id in all_ids
        },
        "target_sample_id_by_raw_id": sample_by_raw,
        "target_frame_names": frame_names,
        "target_id_sources": id_sources,
        "sample_seeds": sample_seeds,
        "scene_id": scene,
        "sample_ids": sample_ids,
    }
    _, index_sha, index_sidecar_sha = validate_artifact_sidecar(
        index_path, {**required, "shape": [12], "dtype": "int64"}
    )
    indices = load_native_int64(index_path, 12)
    if indices != all_ids:
        raise ContractError(f"{scene}: combined target indices mismatch")

    inventory = [
        {"path": str(index_path.resolve()), "sha256": index_sha},
        {
            "path": str(Path(f"{index_path}.manifest.json").resolve()),
            "sha256": index_sidecar_sha,
        },
    ]
    h1_position = {raw_id: position for position, raw_id in enumerate(source.target_ids)}
    combined_position = {raw_id: position for position, raw_id in enumerate(all_ids)}
    for layer in contract.layers:
        path = scene_root / f"feature_t{contract.timestep}_layer{layer}.sft"
        metadata, file_sha, sidecar_sha = validate_artifact_sidecar(
            path,
            {
                **required,
                "source_h1_feature_sha256": source.feature_sha256[layer],
                "tensors": {
                    "feat": {
                        "shape": [12, *contract.feature_shape],
                        "dtype": "torch.float16",
                    }
                },
            },
        )
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if list(handle.keys()) != ["feat"]:
                raise ContractError(f"Unexpected output safetensors keys: {path}")
            tensor_slice = handle.get_slice("feat")
            if tuple(tensor_slice.get_shape()) != (12, *contract.feature_shape):
                raise ContractError(f"Unexpected output tensor shape: {path}")
            if str(tensor_slice.get_dtype()) != "F16":
                raise ContractError(f"Unexpected output tensor dtype: {path}")
        tensor = load_safetensors(str(path), device="cpu")["feat"]
        if not torch.isfinite(tensor).all().item():
            raise ContractError(f"Non-finite output tensor: {path}")
        for raw_id in source.target_ids:
            if not torch.equal(
                tensor[combined_position[raw_id]],
                source.tensors[layer][h1_position[raw_id]],
            ):
                raise ContractError(f"{scene}/layer{layer}: reused h1 row changed")
        if metadata["sha256"] != file_sha:
            raise ContractError(f"Internal output SHA inconsistency: {path}")
        inventory.extend(
            [
                {"path": str(path.resolve()), "sha256": file_sha},
                {
                    "path": str(Path(f"{path}.manifest.json").resolve()),
                    "sha256": sidecar_sha,
                },
            ]
        )
    return inventory


def write_scene(
    out_root: Path,
    shard_name: str,
    scene: str,
    rows: list[dict[str, Any]],
    source: SourceScene,
    new_features: dict[int, dict[int, torch.Tensor]],
    manifest: Path,
    source_h1_root: Path,
    source_audit: Path,
    model_id: Path,
    contract: Contract,
    extracted_input_images_by_raw_id: dict[str, Any],
    git_identity: GitIdentity,
) -> list[dict[str, Any]]:
    mapping, _, all_ids, sample_ids = expected_scene_details(scene, rows, contract)
    parent = out_root / "wan" / "scannetpp"
    final_scene = parent / scene
    root_complete = out_root / ROOT_COMPLETE
    if root_complete.exists():
        raise ContractError(f"Refusing completed immutable root: {out_root}")
    if final_scene.exists():
        raise ContractError(f"Refusing existing scene output: {final_scene}")
    claim = out_root / "provenance" / "scene_claims" / f"{scene}.json"
    exclusive_json(
        claim,
        {
            "status": "CLAIMED",
            "scene": scene,
            "shard": shard_name,
            "manifest_sha256": contract.manifest_sha256,
            "wrapper_sha256": wrapper_sha256(),
            "pid": os.getpid(),
            "created_unix": time.time(),
        },
    )
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{scene}.incomplete-{shard_name}-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir()
    common = common_output_metadata(
        manifest,
        source_h1_root,
        source_audit,
        model_id,
        contract,
        git_identity,
    )
    horizon_by_raw = {str(raw_id): int(mapping[raw_id][0]) for raw_id in all_ids}
    sample_by_raw = {
        str(raw_id): str(mapping[raw_id][1]["sample_id"]) for raw_id in all_ids
    }
    frame_names = [
        str(mapping[raw_id][1]["valid_frames"][raw_id]) for raw_id in all_ids
    ]
    sources = {str(raw_id): mapping[raw_id][2] for raw_id in all_ids}
    seeds = {
        str(raw_id): base.deterministic_seed(contract.seed, f"target:{scene}:{raw_id}")
        for raw_id in all_ids
    }
    per_scene = {
        **common,
        "scene_id": scene,
        "sample_ids": sample_ids,
        "target_raw_ids": all_ids,
        "target_horizon_by_raw_id": horizon_by_raw,
        "target_sample_id_by_raw_id": sample_by_raw,
        "target_frame_names": frame_names,
        "target_id_sources": sources,
        "sample_seeds": seeds,
        "source_h1_target_indices_sha256": source.index_sha256,
        "extracted_input_images_by_raw_id": validate_extracted_input_images(
            extracted_input_images_by_raw_id, mapping, scene, contract
        ),
    }
    try:
        for layer in contract.layers:
            path = temporary / f"feature_t{contract.timestep}_layer{layer}.sft"
            tensor = merged_layer_tensor(source, new_features, all_ids, layer)
            base.atomic_safetensors(
                path,
                tensor,
                {
                    **per_scene,
                    "source_h1_feature_sha256": source.feature_sha256[layer],
                },
            )
        base.atomic_npy(
            temporary / "target_indices.npy",
            np.asarray(all_ids, dtype=np.int64),
            per_scene,
        )
        # Validate the complete temporary scene, including exact h1 equality,
        # before its only namespace publication operation.
        validate_output_scene(
            temporary,
            scene,
            rows,
            source,
            model_id,
            contract,
            git_identity,
            extracted_input_images_by_raw_id,
        )
        rename_noreplace(temporary, final_scene)
        # Paths participate in the inventory identity.  Re-read the published
        # namespace so completion cannot accidentally attest temporary paths.
        published_inventory = validate_output_scene(
            final_scene,
            scene,
            rows,
            source,
            model_id,
            contract,
            git_identity,
            extracted_input_images_by_raw_id,
        )
        completion = out_root / "provenance" / "scene_completions" / f"{scene}.json"
        atomic_json_noreplace(
            completion,
            {
                "status": "COMPLETE",
                "scene": scene,
                "shard": shard_name,
                "manifest_sha256": contract.manifest_sha256,
                "wrapper_sha256": wrapper_sha256(),
                "target_count": 12,
                "layers": list(contract.layers),
                "output_inventory_sha256": canonical_json_sha256(published_inventory),
                "completed_unix": time.time(),
            },
        )
    except Exception:
        # Deliberately preserve claim and incomplete output as evidence.  The
        # namespace is now tainted and all subsequent invocations fail closed.
        raise
    return published_inventory


def runtime_provenance(
    args: argparse.Namespace, contract: Contract, git_identity: GitIdentity
) -> dict[str, Any]:
    return {
        "argv": list(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "contract": asdict(contract),
        "wrapper_sha256": wrapper_sha256(),
        "base_extractor_sha256": sha256(Path(base.__file__).resolve()),
        "repo_root": git_identity.repo_root,
        "git_commit": git_identity.commit,
        "git_commit_verified": True,
        "git_commit_verification_source": git_identity.verification_source,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def ensure_safe_output_root(out_root: Path, source_h1_root: Path) -> None:
    """Reject broad, symlinked, or source-overlapping write targets."""
    unresolved = out_root.absolute()
    if unresolved.exists() and unresolved.is_symlink():
        raise ContractError(f"Output root must not be a symlink: {unresolved}")
    out_root = out_root.resolve()
    source_h1_root = source_h1_root.resolve()
    broad = {
        Path("/"),
        Path("/mnt"),
        Path("/mnt/umm"),
        Path("/mnt/umm/users"),
        Path("/mnt/umm/users/yinbaiqiao"),
        REPO_ROOT.resolve(),
    }
    if out_root in broad:
        raise ContractError(f"Refusing broad output root: {out_root}")
    if (
        out_root == source_h1_root
        or source_h1_root in out_root.parents
        or out_root in source_h1_root.parents
    ):
        raise ContractError(
            f"Output root overlaps immutable source h1 root: {out_root} / {source_h1_root}"
        )


def validate_shard_name(shard_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", shard_name):
        raise ContractError(f"Unsafe shard name: {shard_name!r}")


def ensure_output_preflight(
    out_root: Path,
    shard_name: str,
    scenes: Iterable[str],
) -> Path:
    if (out_root / ROOT_COMPLETE).exists():
        raise ContractError(f"Output root is already complete/immutable: {out_root}")
    shard_dir = out_root / "provenance" / "shards"
    shard_claim = shard_dir / f"{shard_name}.claim.json"
    shard_complete = shard_dir / f"{shard_name}.complete.json"
    existing_shard_state = list(shard_dir.glob(f"{shard_name}.*")) if shard_dir.exists() else []
    if shard_claim.exists() or shard_complete.exists() or existing_shard_state:
        raise ContractError(f"Refusing existing shard state: {shard_name}")
    for scene in scenes:
        checks = (
            out_root / "wan" / "scannetpp" / scene,
            out_root / "provenance" / "scene_claims" / f"{scene}.json",
            out_root / "provenance" / "scene_completions" / f"{scene}.json",
        )
        for path in checks:
            if path.exists():
                raise ContractError(f"Refusing existing output/claim: {path}")
    return shard_claim


def require_marker_fields(
    payload: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    """Require every provenance field and its exact value."""
    for key, value in expected.items():
        if key not in payload:
            raise ContractError(f"{label}: missing required field {key}")
        if payload[key] != value:
            raise ContractError(
                f"{label}: field {key}={payload[key]!r}, expected {value!r}"
            )


def require_marker_number(payload: dict[str, Any], key: str, label: str) -> None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label}: field {key} must be numeric")


def rehash_inventory(inventory: list[dict[str, Any]], label: str) -> None:
    """Re-read every attested artifact and reject duplicate or changed paths."""
    seen: set[str] = set()
    for position, item in enumerate(inventory):
        if not isinstance(item, dict):
            raise ContractError(f"{label}: inventory row {position} is not an object")
        path_string = item.get("path")
        expected_sha = item.get("sha256")
        if not isinstance(path_string, str) or not isinstance(expected_sha, str):
            raise ContractError(f"{label}: malformed inventory row {position}")
        path = Path(path_string)
        if not path.is_absolute() or path_string != str(path.resolve()):
            raise ContractError(f"{label}: inventory path is not canonical: {path}")
        if path_string in seen:
            raise ContractError(f"{label}: duplicate inventory path: {path}")
        seen.add(path_string)
        if not path.is_file():
            raise ContractError(f"{label}: missing inventory artifact: {path}")
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise ContractError(
                f"{label}: artifact changed: {path} {actual_sha} != {expected_sha}"
            )


def load_inventory_marker(path: Path, expected_sha: str, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ContractError(f"{label}: missing inventory file: {path}")
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise ContractError(
            f"{label}: inventory file SHA mismatch: {actual_sha} != {expected_sha}"
        )
    payload = read_json(path)
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("artifacts"), list
    ):
        raise ContractError(f"{label}: malformed inventory payload")
    artifacts = payload["artifacts"]
    rehash_inventory(artifacts, label)
    return artifacts


def validate_scene_markers(
    out_root: Path,
    scene: str,
    inventory: list[dict[str, Any]],
    contract: Contract,
) -> str:
    claim_path = out_root / "provenance" / "scene_claims" / f"{scene}.json"
    completion_path = (
        out_root / "provenance" / "scene_completions" / f"{scene}.json"
    )
    if not claim_path.is_file() or not completion_path.is_file():
        raise ContractError(f"Missing claim/completion provenance for {scene}")
    claim = read_json(claim_path)
    completion = read_json(completion_path)
    shard = claim.get("shard")
    if not isinstance(shard, str):
        raise ContractError(f"Scene claim {scene}: missing shard")
    validate_shard_name(shard)
    require_marker_fields(
        claim,
        {
            "status": "CLAIMED",
            "scene": scene,
            "shard": shard,
            "manifest_sha256": contract.manifest_sha256,
            "wrapper_sha256": wrapper_sha256(),
        },
        f"Scene claim {scene}",
    )
    if isinstance(claim.get("pid"), bool) or not isinstance(claim.get("pid"), int):
        raise ContractError(f"Scene claim {scene}: pid must be an integer")
    require_marker_number(claim, "created_unix", f"Scene claim {scene}")
    require_marker_fields(
        completion,
        {
            "status": "COMPLETE",
            "scene": scene,
            "shard": shard,
            "manifest_sha256": contract.manifest_sha256,
            "wrapper_sha256": wrapper_sha256(),
            "target_count": 12,
            "layers": list(contract.layers),
            "output_inventory_sha256": canonical_json_sha256(inventory),
        },
        f"Scene completion {scene}",
    )
    require_marker_number(completion, "completed_unix", f"Scene completion {scene}")
    return shard


def require_exact_directory_entries(
    directory: Path, expected_names: set[str], label: str
) -> None:
    if not directory.is_dir():
        raise ContractError(f"{label}: missing directory {directory}")
    actual_names = {path.name for path in directory.iterdir()}
    if actual_names != expected_names:
        raise ContractError(
            f"{label}: marker set mismatch; "
            f"missing={sorted(expected_names-actual_names)}, "
            f"extra={sorted(actual_names-expected_names)}"
        )
    non_files = [path.name for path in directory.iterdir() if not path.is_file()]
    if non_files:
        raise ContractError(f"{label}: non-file marker entries: {non_files}")


def validate_shard_markers(
    out_root: Path,
    manifest: Path,
    source_h1_root: Path,
    source_audit: Path,
    groups: dict[str, list[dict[str, Any]]],
    scene_owners: dict[str, str],
    scene_output_inventory: dict[str, list[dict[str, Any]]],
    scene_source_inventory: dict[str, list[dict[str, Any]]],
    contract: Contract,
    git_identity: GitIdentity,
) -> None:
    """Tie every scene marker to one complete, exact-selection shard."""
    shards = sorted(set(scene_owners.values()))
    shard_dir = out_root / "provenance" / "shards"
    expected_entries = {
        f"{shard}.{suffix}"
        for shard in shards
        for suffix in (
            "claim.json",
            "complete.json",
            "output_inventory.json",
            "source_inventory.json",
        )
    }
    require_exact_directory_entries(shard_dir, expected_entries, "Shard provenance")

    selected_union: set[str] = set()
    for shard in shards:
        owned = [scene for scene in groups if scene_owners[scene] == shard]
        claim_path = shard_dir / f"{shard}.claim.json"
        complete_path = shard_dir / f"{shard}.complete.json"
        claim = read_json(claim_path)
        complete = read_json(complete_path)
        selection_value = claim.get("selection_file")
        if selection_value is not None and not isinstance(selection_value, str):
            raise ContractError(f"Shard {shard}: invalid selection_file")
        selection_path = Path(selection_value).resolve() if selection_value else None
        selection_sha = sha256(selection_path) if selection_path else None
        require_marker_fields(
            claim,
            {
                "status": "CLAIMED",
                "shard": shard,
                "scenes": owned,
                "manifest_sha256": contract.manifest_sha256,
                "manifest": str(manifest.resolve()),
                "source_h1_root": str(source_h1_root.resolve()),
                "source_h1_audit": str(source_audit.resolve()),
                "source_h1_audit_sha256": contract.source_audit_sha256,
                "selection_file": str(selection_path) if selection_path else None,
                "selection_file_sha256": selection_sha,
                "wrapper_sha256": wrapper_sha256(),
                "repo_root": git_identity.repo_root,
                "git_commit": git_identity.commit,
                "git_commit_verified": True,
                "git_commit_verification_source": git_identity.verification_source,
            },
            f"Shard claim {shard}",
        )
        if not isinstance(claim.get("runtime"), dict):
            raise ContractError(f"Shard claim {shard}: runtime must be an object")
        require_marker_fields(
            claim["runtime"],
            {
                "repo_root": git_identity.repo_root,
                "git_commit": git_identity.commit,
                "git_commit_verified": True,
                "git_commit_verification_source": git_identity.verification_source,
            },
            f"Shard claim {shard} runtime",
        )
        require_marker_number(claim, "created_unix", f"Shard claim {shard}")

        if selection_path is None:
            selection_groups = groups
        else:
            if not selection_path.is_file():
                raise ContractError(f"Shard {shard}: missing selection file")
            _, selection_groups = load_manifest_selection(
                manifest, selection_path, contract
            )
        if list(selection_groups) != owned:
            raise ContractError(
                f"Shard {shard}: selection scenes do not equal owned scenes"
            )
        overlap = selected_union & set(owned)
        if overlap:
            raise ContractError(f"Shard {shard}: duplicate selection scenes {sorted(overlap)}")
        selected_union.update(owned)

        output_path = (shard_dir / f"{shard}.output_inventory.json").resolve()
        source_path = (shard_dir / f"{shard}.source_inventory.json").resolve()
        require_marker_fields(
            complete,
            {
                "status": "COMPLETE",
                "shard": shard,
                "manifest_sha256": contract.manifest_sha256,
                "selection_file": str(selection_path) if selection_path else None,
                "selection_file_sha256": selection_sha,
                "wrapper_sha256": wrapper_sha256(),
                "scene_count": len(owned),
                "sample_count": sum(len(groups[scene]) for scene in owned),
                "new_forward_count": len(owned) * 8,
                "output_inventory": str(output_path),
                "output_inventory_sha256": sha256(output_path),
                "source_inventory": str(source_path),
                "source_inventory_sha256": sha256(source_path),
            },
            f"Shard completion {shard}",
        )
        require_marker_number(complete, "completed_unix", f"Shard completion {shard}")
        shard_output = load_inventory_marker(
            output_path, complete["output_inventory_sha256"], f"Shard {shard} output"
        )
        shard_source = load_inventory_marker(
            source_path, complete["source_inventory_sha256"], f"Shard {shard} source"
        )
        expected_output = [
            item for scene in owned for item in scene_output_inventory[scene]
        ]
        expected_source = [
            item for scene in owned for item in scene_source_inventory[scene]
        ]
        if shard_output != expected_output:
            raise ContractError(f"Shard {shard}: output inventory ownership mismatch")
        if shard_source != expected_source:
            raise ContractError(f"Shard {shard}: source inventory ownership mismatch")

    if selected_union != set(groups):
        raise ContractError("Shard selections do not cover the full manifest exactly")


def extract_shard(args: argparse.Namespace, contract: Contract) -> None:
    manifest = args.manifest.resolve()
    source_h1_root = args.source_h1_root.resolve()
    source_audit = args.source_h1_audit.resolve()
    out_root = args.out_root.resolve()
    model_id = args.model_id.resolve()

    ensure_pinned_base_extractor(contract)
    git_identity = verify_git_identity(args.repo_root, args.expected_git_commit)
    validate_shard_name(args.shard_name)
    if not source_h1_root.is_dir():
        raise ContractError(f"Missing source h1 root: {source_h1_root}")
    if not model_id.is_dir():
        raise ContractError(f"Missing Wan model snapshot: {model_id}")
    if not args.data_root.resolve().is_dir():
        raise ContractError(f"Missing data root: {args.data_root.resolve()}")
    ensure_safe_output_root(out_root, source_h1_root)
    _, selected_groups = load_manifest_selection(
        manifest, args.selection_file, contract
    )
    validate_source_audit(source_audit, source_h1_root, contract)
    if model_id != MODEL_SNAPSHOT.resolve():
        raise ContractError(f"Wan model snapshot mismatch: {model_id}")

    if args.validate_only:
        for scene, rows in selected_groups.items():
            source = validate_source_h1_scene(
                source_h1_root, scene, rows, model_id, contract
            )
            assert_source_unchanged(source)
        print(
            json.dumps(
                {
                    "status": "VALIDATION_ONLY_PASS",
                    "scenes": len(selected_groups),
                    "rows": sum(len(rows) for rows in selected_groups.values()),
                    "new_forwards_not_run": len(selected_groups) * 8,
                },
                sort_keys=True,
            )
        )
        return

    if not args.fail_on_existing:
        raise ContractError("Extraction requires explicit --fail-on-existing")
    shard_claim = ensure_output_preflight(
        out_root, args.shard_name, selected_groups
    )
    exclusive_json(
        shard_claim,
        {
            "status": "CLAIMED",
            "shard": args.shard_name,
            "scenes": list(selected_groups),
            "manifest_sha256": contract.manifest_sha256,
            "manifest": str(manifest),
            "source_h1_root": str(source_h1_root),
            "source_h1_audit": str(source_audit),
            "source_h1_audit_sha256": contract.source_audit_sha256,
            "selection_file": (
                str(args.selection_file.resolve()) if args.selection_file else None
            ),
            "selection_file_sha256": (
                sha256(args.selection_file.resolve()) if args.selection_file else None
            ),
            "wrapper_sha256": wrapper_sha256(),
            "repo_root": git_identity.repo_root,
            "git_commit": git_identity.commit,
            "git_commit_verified": True,
            "git_commit_verification_source": git_identity.verification_source,
            "runtime": runtime_provenance(args, contract, git_identity),
            "created_unix": time.time(),
        },
    )

    # Heavy imports and GPU allocation occur only after every read-only gate and
    # exclusive output claim has succeeded.
    from features.wan.wan_feature import get_wan_featurizer

    model = get_wan_featurizer(model_id=str(model_id), null_prompt=contract.prompt)
    output_inventory: list[dict[str, Any]] = []
    source_inventory: list[dict[str, Any]] = []
    for scene, rows in selected_groups.items():
        source = validate_source_h1_scene(
            source_h1_root, scene, rows, model_id, contract
        )
        scene_mapping = validate_scene_rows(scene, rows, contract)
        extracted_input_images_by_raw_id: dict[str, dict[str, Any]] = {}

        def forward_one(row: dict[str, Any], raw_id: int) -> dict[int, torch.Tensor]:
            horizon, mapped_row, _ = scene_mapping[raw_id]
            if str(mapped_row["sample_id"]) != str(row["sample_id"]):
                raise ContractError(f"{scene}/{raw_id}: forward row identity mismatch")
            before = snapshot_extracted_input_image(
                args.data_root.resolve(), scene, row, raw_id, horizon
            )
            image = base.load_frame(args.data_root.resolve(), row, raw_id)
            values = base.forward_target(
                model,
                image,
                list(contract.layers),
                contract.seed,
                f"target:{scene}:{raw_id}",
            )
            after = snapshot_extracted_input_image(
                args.data_root.resolve(), scene, row, raw_id, horizon
            )
            if before != after:
                raise ContractError(
                    f"{scene}/{raw_id}: input image changed across Wan forward"
                )
            key = str(raw_id)
            if key in extracted_input_images_by_raw_id:
                raise ContractError(f"{scene}/{raw_id}: duplicate input image forward")
            extracted_input_images_by_raw_id[key] = before
            return values

        new_features = collect_new_features(
            scene, rows, forward_one, contract
        )
        validate_extracted_input_images(
            extracted_input_images_by_raw_id, scene_mapping, scene, contract
        )
        output_inventory.extend(
            write_scene(
                out_root,
                args.shard_name,
                scene,
                rows,
                source,
                new_features,
                manifest,
                source_h1_root,
                source_audit,
                model_id,
                contract,
                extracted_input_images_by_raw_id,
                git_identity,
            )
        )
        assert_source_unchanged(source)
        source_inventory.extend(
            {"path": path, "sha256": file_sha, "scene": scene}
            for path, file_sha in sorted(source.artifact_sha256.items())
        )

    if verify_git_identity(args.repo_root, args.expected_git_commit) != git_identity:
        raise ContractError("Repository identity changed during shard extraction")

    shard_dir = out_root / "provenance" / "shards"
    output_inventory_path = shard_dir / f"{args.shard_name}.output_inventory.json"
    source_inventory_path = shard_dir / f"{args.shard_name}.source_inventory.json"
    if output_inventory_path.exists() or source_inventory_path.exists():
        raise ContractError("Refusing existing shard inventory output")
    atomic_json_noreplace(
        output_inventory_path,
        {"schema_version": 1, "artifacts": output_inventory},
    )
    atomic_json_noreplace(
        source_inventory_path,
        {"schema_version": 1, "artifacts": source_inventory},
    )
    atomic_json_noreplace(
        shard_dir / f"{args.shard_name}.complete.json",
        {
            "status": "COMPLETE",
            "shard": args.shard_name,
            "manifest_sha256": contract.manifest_sha256,
            "selection_file": (
                str(args.selection_file.resolve()) if args.selection_file else None
            ),
            "selection_file_sha256": (
                sha256(args.selection_file.resolve()) if args.selection_file else None
            ),
            "wrapper_sha256": wrapper_sha256(),
            "scene_count": len(selected_groups),
            "sample_count": sum(len(rows) for rows in selected_groups.values()),
            "new_forward_count": len(selected_groups) * 8,
            "output_inventory": str(output_inventory_path.resolve()),
            "output_inventory_sha256": sha256(output_inventory_path),
            "source_inventory": str(source_inventory_path.resolve()),
            "source_inventory_sha256": sha256(source_inventory_path),
            "completed_unix": time.time(),
        },
    )
    print(
        json.dumps(
            {
                "status": "SHARD_COMPLETE",
                "shard": args.shard_name,
                "scenes": len(selected_groups),
                "samples": sum(len(rows) for rows in selected_groups.values()),
            },
            sort_keys=True,
        )
    )


def finalize_root(args: argparse.Namespace, contract: Contract) -> None:
    manifest = args.manifest.resolve()
    source_h1_root = args.source_h1_root.resolve()
    source_audit = args.source_h1_audit.resolve()
    out_root = args.out_root.resolve()
    model_id = args.model_id.resolve()
    complete_path = out_root / ROOT_COMPLETE
    if complete_path.exists():
        raise ContractError(f"Refusing existing root completion marker: {complete_path}")
    ensure_pinned_base_extractor(contract)
    git_identity = verify_git_identity(args.repo_root, args.expected_git_commit)
    if not source_h1_root.is_dir():
        raise ContractError(f"Missing source h1 root: {source_h1_root}")
    if not model_id.is_dir():
        raise ContractError(f"Missing Wan model snapshot: {model_id}")
    ensure_safe_output_root(out_root, source_h1_root)
    _, groups = load_manifest_selection(manifest, None, contract)
    validate_source_audit(source_audit, source_h1_root, contract)
    if model_id != MODEL_SNAPSHOT.resolve():
        raise ContractError(f"Wan model snapshot mismatch: {model_id}")

    scene_parent = out_root / "wan" / "scannetpp"
    if not scene_parent.is_dir():
        raise ContractError(f"Missing output scene root: {scene_parent}")
    actual_scenes = {path.name for path in scene_parent.iterdir() if path.is_dir()}
    if actual_scenes != set(groups):
        raise ContractError(
            f"Output scene set mismatch: missing={len(set(groups)-actual_scenes)}, "
            f"extra={len(actual_scenes-set(groups))}"
        )
    hidden_or_files = [
        path.name
        for path in scene_parent.iterdir()
        if not path.is_dir() or path.name.startswith(".")
    ]
    if hidden_or_files:
        raise ContractError(f"Incomplete/unexpected scene-root entries: {hidden_or_files[:5]}")

    output_inventory: list[dict[str, Any]] = []
    source_inventory: list[dict[str, Any]] = []
    scene_output_inventory: dict[str, list[dict[str, Any]]] = {}
    scene_source_inventory: dict[str, list[dict[str, Any]]] = {}
    scene_owners: dict[str, str] = {}
    for scene, rows in groups.items():
        source = validate_source_h1_scene(
            source_h1_root, scene, rows, model_id, contract
        )
        scene_output = validate_output_scene(
            scene_parent / scene,
            scene,
            rows,
            source,
            model_id,
            contract,
            git_identity,
        )
        scene_source = [
            {"path": path, "sha256": file_sha, "scene": scene}
            for path, file_sha in sorted(source.artifact_sha256.items())
        ]
        scene_output_inventory[scene] = scene_output
        scene_source_inventory[scene] = scene_source
        scene_owners[scene] = validate_scene_markers(
            out_root, scene, scene_output, contract
        )
        output_inventory.extend(scene_output)
        assert_source_unchanged(source)
        source_inventory.extend(scene_source)

    require_exact_directory_entries(
        out_root / "provenance" / "scene_claims",
        {f"{scene}.json" for scene in groups},
        "Scene claims",
    )
    require_exact_directory_entries(
        out_root / "provenance" / "scene_completions",
        {f"{scene}.json" for scene in groups},
        "Scene completions",
    )
    validate_shard_markers(
        out_root,
        manifest,
        source_h1_root,
        source_audit,
        groups,
        scene_owners,
        scene_output_inventory,
        scene_source_inventory,
        contract,
        git_identity,
    )

    provenance = out_root / "provenance"
    output_inventory_path = provenance / "full_output_inventory.json"
    source_inventory_path = provenance / "full_source_h1_inventory.json"
    if output_inventory_path.exists() or source_inventory_path.exists():
        raise ContractError("Refusing existing full-root inventory output")
    atomic_json_noreplace(
        output_inventory_path,
        {"schema_version": 1, "artifacts": output_inventory},
    )
    atomic_json_noreplace(
        source_inventory_path,
        {"schema_version": 1, "artifacts": source_inventory},
    )
    # Final, full re-read immediately before root publication.  This closes the
    # long-loop gap between per-scene validation and the immutable root marker.
    rehash_inventory(output_inventory, "Final collected output")
    rehash_inventory(source_inventory, "Final collected source h1")
    if verify_git_identity(args.repo_root, args.expected_git_commit) != git_identity:
        raise ContractError("Repository identity changed during finalization")
    root_payload = {
        "status": "COMPLETE_IMMUTABLE",
        "schema_version": 1,
        "canonical_manifest": str(manifest),
        "canonical_manifest_sha256": contract.manifest_sha256,
        "source_h1_root": str(source_h1_root),
        "source_h1_audit_sha256": contract.source_audit_sha256,
        "wrapper_sha256": wrapper_sha256(),
        "base_extractor_sha256": contract.base_extractor_sha256,
        "repo_root": git_identity.repo_root,
        "git_commit": git_identity.commit,
        "git_commit_verified": True,
        "git_commit_verification_source": git_identity.verification_source,
        "model_id": str(model_id),
        "scene_count": EXPECTED_SCENES,
        "sample_count": EXPECTED_ROWS,
        "target_rows_per_layer": EXPECTED_ROWS * len(contract.horizons),
        "target_sft_count": EXPECTED_SCENES * len(contract.layers),
        "layers": list(contract.layers),
        "horizons": list(contract.horizons),
        "reused_horizons": [1],
        "extracted_horizons": list(contract.extract_horizons),
        "output_inventory": str(output_inventory_path.resolve()),
        "output_inventory_sha256": sha256(output_inventory_path),
        "source_h1_inventory": str(source_inventory_path.resolve()),
        "source_h1_inventory_sha256": sha256(source_inventory_path),
        "completed_unix": time.time(),
    }
    atomic_json_noreplace(complete_path, root_payload)
    print(json.dumps(root_payload, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extend the pinned Wan p8 canonical h1 target cache to an exact, "
            "new h1/h2/h4 target namespace. No streaming or h1 forward is run."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source-h1-root", required=True, type=Path)
    parser.add_argument("--source-h1-audit", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--model-id", required=True, type=Path)
    parser.add_argument(
        "--repo-root",
        required=True,
        type=Path,
        help="Canonical absolute repository root whose .git/HEAD is verified.",
    )
    parser.add_argument(
        "--expected-git-commit",
        required=True,
        help=f"Required pinned repository commit ({EXPECTED_GIT_COMMIT}).",
    )
    parser.add_argument("--shard-name")
    parser.add_argument(
        "--fail-on-existing",
        action="store_true",
        help="Required for extraction; refuses every existing scene/shard claim.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify manifest/audit/source h1 only; no GPU load and no writes.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Full CPU audit and immutable root completion; no GPU load.",
    )
    args = parser.parse_args(argv)
    if args.validate_only and args.finalize_only:
        parser.error("--validate-only and --finalize-only are mutually exclusive")
    if not args.finalize_only:
        if args.data_root is None:
            parser.error("--data-root is required unless --finalize-only")
        if not args.shard_name:
            parser.error("--shard-name is required unless --finalize-only")
    if args.finalize_only and args.selection_file is not None:
        parser.error("--finalize-only always audits the full manifest")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    contract = Contract()
    if args.finalize_only:
        finalize_root(args, contract)
    else:
        extract_shard(args, contract)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    try:
        main()
    except ContractError as exc:
        raise SystemExit(f"CONTRACT FAILURE: {exc}") from exc
