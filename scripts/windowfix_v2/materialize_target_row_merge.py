#!/usr/bin/env python3
"""Materialize canonical-v2 target tensors from audited raw-ID row routes.

The default invocation is a read-only dry run.  ``--execute`` re-runs the full
route audit with source SHA verification, loads legacy and gap-only delta
payloads, reorders rows by the canonical target union, and publishes into the
dedicated ``FEAT_TARGET_CANONICAL_V2_MIXED_V1`` namespace.

Every destination is no-replace.  A per-layer receipt is published last and is
the commit marker; partial files are never accepted as complete.  This program
refuses every path containing ``SEEDED_V1``.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.windowfix_v2.audit_target_row_routes import (
    CANONICAL_PROTOCOL,
    CONTRACT,
    DINO_PREFIX_SOURCE_PRIORITIES,
    HORIZONS,
    MIXED_TARGET_NAMESPACE,
    PREFIXES,
    RouteAuditError,
    audit_plan,
    canonical_json_sha,
    read_json,
    safetensors_tensor_info,
    sha256_file,
    validate_destination_root,
)
from analyses.windowfix_v2.target_row_source import (
    is_remote as source_is_remote,
    read_tensor_row,
    tensor_layout,
)


class MergeError(ValueError):
    """Raised before an unsafe or incomplete merge can be committed."""


def _assert_materializer_duplicate_policy(plan: Mapping[str, Any]) -> None:
    """Independently enforce the one permitted multi-owner source family."""
    for unit in plan.get("units", []):
        if not isinstance(unit, dict):
            raise MergeError("target merge plan contains a non-object unit")
        if unit.get("duplicate_owner_policy", "forbid") != (
            "require_bitwise_equal_then_priority"
        ):
            continue
        if unit.get("vfm") != "dino":
            raise MergeError("conditional duplicate policy is restricted to vfm=dino")
        sources = unit.get("sources")
        if not isinstance(sources, list) or not sources:
            raise MergeError("DINO conditional duplicate policy requires prefix sources")
        for source in sources:
            if not isinstance(source, dict) or source.get("kind") != "legacy":
                raise MergeError("DINO conditional duplicate policy accepts only legacy prefix sources")
            source_id = str(source.get("source_id", ""))
            if source_id not in DINO_PREFIX_SOURCE_PRIORITIES:
                raise MergeError(f"forged/non-prefix DINO source_id: {source_id!r}")
            if int(source.get("priority", -1)) != DINO_PREFIX_SOURCE_PRIORITIES[source_id]:
                raise MergeError(f"DINO source priority mismatch for {source_id!r}")
    for unit in plan.get("units", []):
        for source in unit.get("sources", []):
            if source.get("kind") != "gap_delta":
                continue
            for field in ("expected_payload_sha256", "expected_index_sha256"):
                value = source.get(field)
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value.lower())
                ):
                    raise MergeError(f"gap_delta source requires {field}")


def _assert_routed_gap_hashes(result: Mapping[str, Any]) -> None:
    units = {_unit_key(unit): unit for unit in result["normalized_units"]}
    for route in result["routes"]:
        if route.get("source_kind") != "gap_delta":
            continue
        unit = units[_route_key(route)]
        sources = {
            str(source.get("source_id")): source for source in unit.get("sources", [])
        }
        source = sources.get(str(route.get("source_id")))
        if source is None:
            raise MergeError("routed gap_delta source disappeared from normalized plan")
        if route.get("source_payload_sha256") != source.get("expected_payload_sha256"):
            raise MergeError("routed gap_delta payload SHA differs from augmented plan")
        if route.get("source_index_sha256") != source.get("expected_index_sha256"):
            raise MergeError("routed gap_delta index SHA differs from augmented plan")


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temp_path(parent: Path, stem: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{stem}.tmp-", dir=parent)
    os.close(descriptor)
    return Path(name)


def _publish_noreplace(staged: Path, destination: Path) -> None:
    """Atomically publish a regular file without replacing an existing path."""
    if destination.is_symlink():
        raise MergeError(f"destination is a symlink: {destination}")
    try:
        os.link(staged, destination)
    except FileExistsError as exc:
        raise MergeError(f"destination already exists; overwrite forbidden: {destination}") from exc
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise MergeError("staging and destination must share one filesystem") from exc
        raise
    staged.unlink()
    _fsync_dir(destination.parent)


def _stage_bytes(parent: Path, stem: str, payload: bytes) -> Path:
    staged = _temp_path(parent, stem)
    try:
        with staged.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staged, 0o644)
        return staged
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def _stage_json(parent: Path, stem: str, value: Mapping[str, Any]) -> Path:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    return _stage_bytes(parent, stem, payload)


def _stage_npy(parent: Path, stem: str, values: Sequence[int]) -> Path:
    staged = _temp_path(parent, stem)
    try:
        with staged.open("wb") as handle:
            np.save(handle, np.asarray(values, dtype=np.int64), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staged, 0o644)
        return staged
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def _stage_safetensors(
    parent: Path, stem: str, tensor_key: str, tensor: torch.Tensor
) -> Path:
    staged = _temp_path(parent, stem)
    try:
        save_file(
            {tensor_key: tensor.contiguous()},
            str(staged),
            metadata={
                "contract": CONTRACT,
                "canonical_protocol": CANONICAL_PROTOCOL,
                "feature_namespace": MIXED_TARGET_NAMESPACE,
            },
        )
        with staged.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(staged, 0o644)
        return staged
    except Exception:
        staged.unlink(missing_ok=True)
        raise


class _SceneLock:
    def __init__(self, scene_dir: Path):
        self.path = scene_dir / ".target_row_merge.lock"
        self.descriptor: int | None = None

    def __enter__(self) -> "_SceneLock":
        try:
            self.descriptor = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
            )
        except FileExistsError as exc:
            raise MergeError(f"scene merge lock already exists: {self.path}") from exc
        os.write(self.descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(self.descriptor)
        _fsync_dir(self.path.parent)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
        self.path.unlink(missing_ok=True)
        _fsync_dir(self.path.parent)


def _canonical_provenance(source_path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with source_path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            record = json.loads(raw)
            source = str(record["source"])
            scene = str(record["scene_id"])
            window_id = int(record["streaming_window_id"])
            obs_start = int(record.get("streaming_obs_start", 0))
            window = [int(value) for value in record["streaming_window_indices"]]
            for prefix in PREFIXES:
                positions = [("anchor", 0, prefix - 1)] + [
                    ("future", horizon, prefix - 1 + horizon) for horizon in HORIZONS
                ]
                for role, horizon, position in positions:
                    result[(source, scene)].append(
                        {
                            "window_id": window_id,
                            "obs_start": obs_start,
                            "prefix_length": prefix,
                            "role": role,
                            "horizon": horizon,
                            "position": position,
                            "frame_index": window[position],
                            "canonical_protocol": CANONICAL_PROTOCOL,
                            "canonical_obs_len": 28,
                        }
                    )
    return result


def _unit_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        value["spec_id"], value["vfm"], value["model_id"], int(value["layer"]),
        int(value["timestep"]), value["source"], value["scene_id"],
    )


def _route_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return _unit_key(value)


def _ensure_no_seeded(path: Path) -> None:
    if any("SEEDED_V1" in component for component in path.parts):
        raise MergeError(f"refusing SEEDED_V1 path: {path}")


def _index_paths(scene_dir: Path) -> dict[str, Path]:
    index = scene_dir / "target_indices.npy"
    return {
        "index": index,
        "sidecar": Path(f"{index}.manifest.json"),
        "meta": scene_dir / "target_indices.meta.json",
        "ready": scene_dir / "target_indices.mixed_ready.json",
    }


def _verify_existing_index(
    paths: Mapping[str, Path],
    expected: Sequence[int],
    *,
    canonical_source_sha: str,
) -> str:
    existing = {name: path.exists() or path.is_symlink() for name, path in paths.items()}
    if not any(existing.values()):
        return "ABSENT"
    if not all(existing.values()):
        raise MergeError(f"partial target index transaction: {existing}")
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise MergeError("existing target index transaction contains a non-regular file")
    try:
        actual = np.load(paths["index"], allow_pickle=False)
    except Exception as exc:
        raise MergeError(f"cannot reload existing target index: {exc}") from exc
    if actual.dtype != np.int64 or actual.ndim != 1 or actual.tolist() != list(expected):
        raise MergeError("existing target index differs from canonical raw-ID order")
    digest = sha256_file(paths["index"])
    sidecar = read_json(paths["sidecar"], "existing target index sidecar")
    meta = read_json(paths["meta"], "existing target index metadata")
    ready = read_json(paths["ready"], "existing target index ready marker")
    if (
        sidecar.get("sha256") != digest
        or sidecar.get("shape") != [len(expected)]
        or sidecar.get("dtype") != "int64"
        or sidecar.get("feature_namespace") != MIXED_TARGET_NAMESPACE
        or meta.get("target_indices_sha256") != digest
        or meta.get("canonical_source_sha256") != canonical_source_sha
        or meta.get("target_count") != len(expected)
        or meta.get("include_anchors") is not True
        or meta.get("prefixes") != list(PREFIXES)
        or meta.get("horizons") != list(HORIZONS)
        or ready.get("status") != "PASS"
        or ready.get("target_indices_sha256") != digest
        or ready.get("canonical_source_sha256") != canonical_source_sha
    ):
        raise MergeError("existing target index transaction fails mixed contract")
    return "VERIFIED_EXISTING"


def _write_index_transaction(
    scene_dir: Path,
    expected: Sequence[int],
    *,
    canonical_source: Mapping[str, Any],
    protocol: Mapping[str, Any],
    provenance: Sequence[Mapping[str, Any]],
) -> str:
    paths = _index_paths(scene_dir)
    state = _verify_existing_index(
        paths, expected, canonical_source_sha=canonical_source["sha256"]
    )
    if state != "ABSENT":
        return state
    staged: list[Path] = []
    try:
        index_tmp = _stage_npy(scene_dir, "target_indices.npy", expected)
        staged.append(index_tmp)
        index_sha = sha256_file(index_tmp)
        index_sidecar = {
            "schema_version": 1,
            "file": "target_indices.npy",
            "feature_namespace": MIXED_TARGET_NAMESPACE,
            "size_bytes": index_tmp.stat().st_size,
            "sha256": index_sha,
            "shape": [len(expected)],
            "dtype": "int64",
            "contract": CONTRACT,
            "canonical_protocol": CANONICAL_PROTOCOL,
            "canonical_source_sha256": canonical_source["sha256"],
        }
        meta = {
            "schema_version": 2,
            "feature_namespace": MIXED_TARGET_NAMESPACE,
            "contract": CONTRACT,
            "protocol": CANONICAL_PROTOCOL,
            "protocol_sha256": protocol.get("canonical_protocol_sha256"),
            "canonical_source_sha256": canonical_source["sha256"],
            "obs_len": 28,
            "prefixes": list(PREFIXES),
            "horizons": list(HORIZONS),
            "include_anchors": True,
            "target_count": len(expected),
            "target_indices_sha256": index_sha,
            "provenance": list(provenance),
        }
        ready = {
            "schema_version": 1,
            "contract": CONTRACT,
            "status": "PASS",
            "feature_namespace": MIXED_TARGET_NAMESPACE,
            "canonical_source_sha256": canonical_source["sha256"],
            "target_indices_sha256": index_sha,
            "target_count": len(expected),
        }
        sidecar_tmp = _stage_json(scene_dir, "target_indices.npy.manifest.json", index_sidecar)
        meta_tmp = _stage_json(scene_dir, "target_indices.meta.json", meta)
        ready_tmp = _stage_json(scene_dir, "target_indices.mixed_ready.json", ready)
        staged.extend((sidecar_tmp, meta_tmp, ready_tmp))
        for temporary, destination in (
            (index_tmp, paths["index"]),
            (sidecar_tmp, paths["sidecar"]),
            (meta_tmp, paths["meta"]),
            (ready_tmp, paths["ready"]),
        ):
            _publish_noreplace(temporary, destination)
            staged.remove(temporary)
        _verify_existing_index(
            paths, expected, canonical_source_sha=canonical_source["sha256"]
        )
        return "WRITTEN"
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def _feature_paths(destination: Path) -> dict[str, Path]:
    return {
        "feature": destination,
        "sidecar": Path(f"{destination}.manifest.json"),
        "receipt": Path(f"{destination}.merge_receipt.json"),
    }


def _verify_existing_feature(
    paths: Mapping[str, Path],
    unit: Mapping[str, Any],
    expected: Sequence[int],
    *,
    plan_sha: str,
) -> bool:
    existing = {name: path.exists() or path.is_symlink() for name, path in paths.items()}
    if not any(existing.values()):
        return False
    if not all(existing.values()):
        raise MergeError(f"partial feature transaction: {existing}")
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise MergeError("feature transaction contains a non-regular file")
    info = safetensors_tensor_info(paths["feature"], unit["tensor_key"])
    digest = sha256_file(paths["feature"])
    sidecar = read_json(paths["sidecar"], "existing mixed feature sidecar")
    receipt = read_json(paths["receipt"], "existing mixed feature receipt")
    expected_shape = (len(expected), *unit["expected_tail_shape"])
    if (
        info.shape != expected_shape
        or info.dtype != unit["expected_dtype"]
        or sidecar.get("sha256") != digest
        or sidecar.get("feature_namespace") != MIXED_TARGET_NAMESPACE
        or sidecar.get("model_id") != unit["model_id"]
        or sidecar.get("layer") != unit["layer"]
        or sidecar.get("timestep") != unit["timestep"]
        or receipt.get("status") != "PASS"
        or receipt.get("output_payload_sha256") != digest
        or receipt.get("plan_sha256") != plan_sha
    ):
        raise MergeError("existing mixed feature does not match this merge unit")
    return True


def _merge_tensor(
    routes: list[dict[str, Any]], unit: Mapping[str, Any]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    layouts = {}
    rows: list[torch.Tensor] = []

    def _read_route(owner: Mapping[str, Any]) -> tuple[torch.Tensor, str]:
        location = str(owner["source_feature_path"])
        if not source_is_remote(location) and owner.get("source_payload_sha256_verified") is not True:
            raise MergeError("execute requires SHA-verified local source routes")
        layout = layouts.get(location)
        if layout is None:
            layout = tensor_layout(
                location,
                unit["tensor_key"],
                declared_size=int(owner["source_size_bytes"]),
            )
            if layout.header_sha256 != owner["source_tensor_header_sha256"]:
                raise MergeError("source safetensors header changed after route audit")
            if layout.shape[1:] != tuple(unit["expected_tail_shape"]):
                raise MergeError("source tail shape changed after route audit")
            if layout.dtype != unit["expected_dtype"]:
                raise MergeError("source dtype changed after route audit")
            layouts[location] = layout
        index = int(owner["source_row_index"])
        row, row_sha = read_tensor_row(layout, index)
        if tuple(row.shape) != tuple(unit["expected_tail_shape"]):
            raise MergeError("source row shape changed after route audit")
        return row, row_sha

    for expected_position, route in enumerate(routes):
        if int(route["canonical_row_index"]) != expected_position:
            raise MergeError("routes are not in canonical row order")
        row, row_sha = _read_route(route)
        route["source_row_sha256"] = row_sha
        verified_alternates = []
        if route.get("alternate_owners"):
            if unit["vfm"] != "dino" or route.get("source_id") not in DINO_PREFIX_SOURCE_PRIORITIES:
                raise MergeError("conditional duplicate owners are restricted to DINO prefix sources")
            if int(route.get("source_priority", -1)) != DINO_PREFIX_SOURCE_PRIORITIES[
                str(route["source_id"])
            ]:
                raise MergeError("selected DINO owner priority changed")
        for alternate in route.get("alternate_owners", []):
            if alternate.get("source_id") not in DINO_PREFIX_SOURCE_PRIORITIES:
                raise MergeError("forged DINO alternate owner source_id")
            if int(alternate.get("source_priority", -1)) != DINO_PREFIX_SOURCE_PRIORITIES[
                str(alternate["source_id"])
            ]:
                raise MergeError("DINO alternate owner priority changed")
            if int(alternate["target_raw_id"]) != int(route["target_raw_id"]):
                raise MergeError("alternate owner raw ID differs from selected owner")
            alternate_row, alternate_sha = _read_route(alternate)
            if not torch.equal(row, alternate_row):
                raise MergeError(
                    "duplicate target row owners are not bitwise equal: "
                    f"raw_id={route['target_raw_id']} selected={route['source_id']} "
                    f"alternate={alternate['source_id']}"
                )
            verified_alternates.append(
                {
                    "source_id": alternate["source_id"],
                    "source_kind": alternate["source_kind"],
                    "source_feature_path": alternate["source_feature_path"],
                    "source_row_index": int(alternate["source_row_index"]),
                    "source_row_sha256": alternate_sha,
                }
            )
        if verified_alternates:
            route["alternate_owners_bitwise_verified"] = verified_alternates
        routes[expected_position] = route
        rows.append(row)
    if not rows:
        raise MergeError("cannot materialize an empty target tensor")
    merged = torch.stack(rows, dim=0).contiguous()
    if tuple(merged.shape[1:]) != tuple(unit["expected_tail_shape"]):
        raise MergeError("merged tensor tail shape mismatch")
    if merged.is_floating_point() and not torch.isfinite(merged).all().item():
        raise MergeError("merged tensor contains NaN or Inf")
    return merged, rows


def _publish_feature_transaction(
    unit: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    *,
    plan_path: Path,
    plan_sha: str,
    audit_report: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(unit["destination_path"])
    _ensure_no_seeded(destination)
    scene_dir = destination.parent
    paths = _feature_paths(destination)
    if _verify_existing_feature(paths, unit, unit["expected_raw_ids"], plan_sha=plan_sha):
        return {"status": "VERIFIED_EXISTING", "destination": str(destination)}

    mutable_routes = [dict(route) for route in routes]
    merged, source_rows = _merge_tensor(mutable_routes, unit)
    routes = mutable_routes
    staged: list[Path] = []
    try:
        feature_tmp = _stage_safetensors(
            scene_dir, destination.name, unit["tensor_key"], merged
        )
        staged.append(feature_tmp)
        output_sha = sha256_file(feature_tmp)
        info = safetensors_tensor_info(feature_tmp, unit["tensor_key"])
        expected_shape = (len(unit["expected_raw_ids"]), *unit["expected_tail_shape"])
        if info.shape != expected_shape or info.dtype != unit["expected_dtype"]:
            raise MergeError("staged output header differs from merge contract")
        source_payloads: dict[str, dict[str, Any]] = {}
        for route in routes:
            for owner in (route, *route.get("alternate_owners", [])):
                path = str(owner["source_feature_path"])
                source_payloads[path] = {
                    "path": path,
                    "source_id": owner["source_id"],
                    "source_kind": owner["source_kind"],
                    "sha256": owner["source_payload_sha256"],
                    "sha256_verified": owner["source_payload_sha256_verified"],
                    "index_path": owner["source_index_path"],
                    "index_sha256": owner["source_index_sha256"],
                }
        lineage = [
            {
                "output_row_index": int(route["canonical_row_index"]),
                "target_raw_id": int(route["target_raw_id"]),
                "source_id": route["source_id"],
                "source_kind": route["source_kind"],
                "source_feature_path": route["source_feature_path"],
                "source_payload_sha256": route["source_payload_sha256"],
                "source_row_sha256": route["source_row_sha256"],
                "source_row_index": int(route["source_row_index"]),
                "alternate_owners_bitwise_verified": route.get(
                    "alternate_owners_bitwise_verified", []
                ),
            }
            for route in routes
        ]
        prompt = unit["prompt_contract"].get("value")
        sidecar = {
            "schema_version": 1,
            "frame_index_schema": "image_mask_intersection_v1",
            "feature_namespace": MIXED_TARGET_NAMESPACE,
            "contract": CONTRACT,
            "generation_contract": "mixed_legacy_exact_and_canonical_gap_v1",
            "canonical_protocol": CANONICAL_PROTOCOL,
            "canonical_protocol_sha256": audit_report.get("canonical_protocol_sha256"),
            "canonical_source_sha256": audit_report["canonical_source"]["sha256"],
            "vfm": unit["vfm"],
            "model_id": unit["model_id"],
            "mode": "target_isolated",
            "layer": unit["layer"],
            "layers": [unit["layer"]],
            "timestep": unit["timestep"],
            "prompt": prompt,
            "prompt_contract": unit["prompt_contract"],
            "preprocess_identity": unit["preprocess_identity"],
            "file": destination.name,
            "size_bytes": feature_tmp.stat().st_size,
            "sha256": output_sha,
            "tensors": {
                unit["tensor_key"]: {
                    "shape": list(info.shape),
                    "dtype": info.dtype,
                }
            },
            "target_indices_sha256": sha256_file(scene_dir / "target_indices.npy"),
            "row_lineage_sha256": canonical_json_sha(lineage),
            "source_payloads": list(source_payloads.values()),
        }
        for key, value in unit["preprocess_identity"].items():
            if "." not in key:
                sidecar[key] = value
        route_counts = Counter(route["source_kind"] for route in routes)
        receipt = {
            "schema_version": 1,
            "receipt_type": "canonical_v2_target_row_merge",
            "contract": CONTRACT,
            "status": "PASS",
            "committed": True,
            "seeded_namespace_written": False,
            "plan": str(plan_path),
            "plan_sha256": plan_sha,
            "plan_content_sha256": read_json(plan_path, "merge plan")["content_sha256"],
            "route_audit_content_sha256": audit_report["content_sha256"],
            "canonical_source_sha256": audit_report["canonical_source"]["sha256"],
            "spec_id": unit["spec_id"],
            "vfm": unit["vfm"],
            "model_id": unit["model_id"],
            "layer": unit["layer"],
            "timestep": unit["timestep"],
            "source": unit["source"],
            "scene_id": unit["scene_id"],
            "destination": str(destination),
            "output_payload_sha256": output_sha,
            "output_size_bytes": feature_tmp.stat().st_size,
            "target_indices_sha256": sidecar["target_indices_sha256"],
            "target_count": len(routes),
            "row_routes_sha256": canonical_json_sha(lineage),
            "rows_by_source_kind": dict(route_counts),
            "source_payloads": list(source_payloads.values()),
            "bitwise_rows_verified": len(routes),
            "duplicate_rows_bitwise_verified": sum(
                1 for route in routes if route.get("alternate_owners_bitwise_verified")
            ),
            "finite_rows_verified": len(routes),
        }
        sidecar_tmp = _stage_json(scene_dir, f"{destination.name}.manifest.json", sidecar)
        receipt_tmp = _stage_json(
            scene_dir, f"{destination.name}.merge_receipt.json", receipt
        )
        staged.extend((sidecar_tmp, receipt_tmp))
        _publish_noreplace(feature_tmp, paths["feature"])
        staged.remove(feature_tmp)
        _publish_noreplace(sidecar_tmp, paths["sidecar"])
        staged.remove(sidecar_tmp)

        # Reopen the committed tensor and prove every copied row is bitwise
        # identical to its selected source before publishing the receipt.
        committed = load_file(str(paths["feature"]), device="cpu")[unit["tensor_key"]]
        if tuple(committed.shape) != expected_shape:
            raise MergeError("committed output shape changed after publication")
        for row_index, route in enumerate(routes):
            if not torch.equal(committed[row_index], source_rows[row_index]):
                raise MergeError(
                    f"committed row {row_index} is not bitwise equal to routed source"
                )
        if sha256_file(paths["feature"]) != output_sha:
            raise MergeError("committed output SHA changed after publication")
        _publish_noreplace(receipt_tmp, paths["receipt"])
        staged.remove(receipt_tmp)
        return {
            "status": "WRITTEN",
            "destination": str(destination),
            "receipt": str(paths["receipt"]),
            "rows": len(routes),
            "rows_by_source_kind": dict(route_counts),
            "sha256": output_sha,
        }
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def materialize(
    plan_path: Path,
    *,
    execute: bool,
    receipt_path: Path | None = None,
    source_timeout_seconds: float | None = 30.0,
) -> dict[str, Any]:
    _assert_materializer_duplicate_policy(read_json(plan_path, "target row merge plan"))
    result = audit_plan(
        plan_path,
        verify_source_sha=execute,
        remote_policy="gate",
        source_timeout_seconds=source_timeout_seconds,
    )
    audit_report = result["report"]
    if audit_report["status"] != "PASS":
        raise MergeError(
            "target row routes are not complete and unambiguous: "
            f"status={audit_report['status']} counts={audit_report['counts']}"
        )
    _assert_routed_gap_hashes(result)
    output_root = validate_destination_root(audit_report["destination_root"])
    _ensure_no_seeded(output_root)
    routes_by_unit: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for route in result["routes"]:
        routes_by_unit[_route_key(route)].append(route)
    for routes in routes_by_unit.values():
        routes.sort(key=lambda row: int(row["canonical_row_index"]))
    for unit in result["normalized_units"]:
        routes = routes_by_unit.get(_unit_key(unit), [])
        if [int(row["target_raw_id"]) for row in routes] != unit["expected_raw_ids"]:
            raise MergeError(f"{unit['spec_id']}:{unit['scene_id']}: route order is not canonical")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "canonical_v2_target_row_merge_run",
        "contract": CONTRACT,
        "status": "DRY_RUN_PASS" if not execute else "RUNNING",
        "execute": execute,
        "feature_payload_written": False,
        "seeded_namespace_written": False,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "plan_content_sha256": read_json(plan_path, "merge plan")["content_sha256"],
        "route_audit_content_sha256": audit_report["content_sha256"],
        "canonical_source_sha256": audit_report["canonical_source"]["sha256"],
        "destination_root": str(output_root),
        "counts": audit_report["counts"],
        "units": [],
    }
    if not execute:
        return summary

    output_root.mkdir(parents=True, exist_ok=True)
    provenance = _canonical_provenance(
        Path(audit_report["canonical_source"]["path"])
    )
    for unit in result["normalized_units"]:
        destination = Path(unit["destination_path"])
        scene_dir = destination.parent
        _ensure_no_seeded(scene_dir)
        scene_dir.mkdir(parents=True, exist_ok=True)
        with _SceneLock(scene_dir):
            index_status = _write_index_transaction(
                scene_dir,
                unit["expected_raw_ids"],
                canonical_source=audit_report["canonical_source"],
                protocol=result["protocol"],
                provenance=provenance[(unit["source"], unit["scene_id"])],
            )
            unit_result = _publish_feature_transaction(
                unit,
                routes_by_unit[_unit_key(unit)],
                plan_path=plan_path,
                plan_sha=summary["plan_sha256"],
                audit_report=audit_report,
            )
            unit_result["index_status"] = index_status
            summary["units"].append(unit_result)

    summary["status"] = "PASS"
    summary["feature_payload_written"] = any(
        item["status"] == "WRITTEN" for item in summary["units"]
    )
    summary["counts"] = {
        **audit_report["counts"],
        "units_written": sum(item["status"] == "WRITTEN" for item in summary["units"]),
        "units_verified_existing": sum(
            item["status"] == "VERIFIED_EXISTING" for item in summary["units"]
        ),
    }
    summary["content_sha256"] = canonical_json_sha(
        {key: value for key, value in summary.items() if key != "content_sha256"}
    )
    if receipt_path is None:
        receipt_path = (
            output_root
            / "_MERGE_RECEIPTS"
            / f"{summary['plan_content_sha256']}.json"
        )
    if not receipt_path.is_absolute():
        raise MergeError("merge receipt path must be absolute")
    _ensure_no_seeded(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    staged_receipt = _stage_json(receipt_path.parent, receipt_path.name, summary)
    try:
        if receipt_path.exists() or receipt_path.is_symlink():
            existing = read_json(receipt_path, "existing merge run receipt")
            if existing.get("content_sha256") != summary["content_sha256"]:
                raise MergeError(f"different merge run receipt already exists: {receipt_path}")
            staged_receipt.unlink(missing_ok=True)
        else:
            _publish_noreplace(staged_receipt, receipt_path)
    finally:
        staged_receipt.unlink(missing_ok=True)
    summary["receipt"] = str(receipt_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--source-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        result = materialize(
            args.plan.resolve(),
            execute=args.execute,
            receipt_path=args.receipt.resolve() if args.receipt else None,
            source_timeout_seconds=args.source_timeout_seconds,
        )
    except (MergeError, RouteAuditError, OSError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
