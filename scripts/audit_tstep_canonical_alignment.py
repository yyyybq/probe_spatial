#!/usr/bin/env python3
"""Audit canonical streaming/target indexes without downloading feature tensors.

The audit is deliberately fail closed.  It loads only the small local
``prefix_index.npy``, ``target_indices.npy``, their ``.manifest.json``
sidecars, and ``target_indices.meta.json``
artifacts.  A full passing report can be consumed by the timestep probing
runner; a scene-filtered report is a canary only and is never trainable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


AUDIT_TYPE = "tstep_canonical_alignment"
AUDIT_SCHEMA_VERSION = 1
DEFAULT_PROTOCOL_ID = "inscene_streaming_canonical_v2_obs28_h124"
DEFAULT_PROTOCOL_SHA256 = (
    "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
)
DEFAULT_CANONICAL_OBS_LEN = 28
DEFAULT_PREFIXES = (8, 12, 16, 24)
DEFAULT_HORIZONS = (1, 2, 4)
STREAMING_FEATURE_ROOT = (
    "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
)
TARGET_FEATURE_ROOT = (
    "aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
)
STREAMING_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
MINIMAX_BF16_STREAMING_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1"
MINIMAX_BF16_TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1"
# Accepted (streaming, target) namespace pairs.  The frozen manifests always
# declare the seeded-v1 feature roots.  The 2026-10-16 MiniMax bf16
# re-extraction is a drop-in copy of seeded-v1 for MiniMax layers 20/25 (same
# unit sets, byte-identical structural indices, bf16 payloads), so its audits
# still bind the seeded-v1 manifest roots while every index sidecar and the
# index roots must carry the bf16 namespace.  ``tag`` keeps its audit files
# apart from the seeded-v1 ones; ``vfm``/``layers`` restrict where it is valid.
NAMESPACE_PAIRS: dict[tuple[str, str], dict[str, Any]] = {
    (STREAMING_NAMESPACE, TARGET_NAMESPACE): {"tag": "seeded_v1", "vfm": None, "layers": None},
    (MINIMAX_BF16_STREAMING_NAMESPACE, MINIMAX_BF16_TARGET_NAMESPACE): {
        "tag": "seeded_v1_minimax_bf16_v1", "vfm": "minimax_h3", "layers": (20, 25),
    },
}
FEATURE_GENERATION_CONTRACT = "inscene_deterministic_noise_v1"
NOISE_SEED_SCHEME = "inscene_noise_seed_v1"
MINIMAX_NOISE_SEED_SCHEME = "minimax_h3_fixed_seed_v1"
MAX_RECORDED_PROBLEMS = 200
HEX_DIGITS = frozenset("0123456789abcdef")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def namespace_pair_errors(
    streaming_namespace: str, target_namespace: str, vfm: str, layers: Iterable[int] = ()
) -> list[str]:
    """Return why (streaming, target) may not be audited/trained for VFM/layers."""

    pair = (streaming_namespace, target_namespace)
    contract = NAMESPACE_PAIRS.get(pair)
    if contract is None:
        return [f"unsupported namespace pair {pair}; accepted: {sorted(NAMESPACE_PAIRS)}"]
    errors = []
    if contract["vfm"] is not None and vfm != contract["vfm"]:
        errors.append(f"namespace pair {contract['tag']} is only valid for vfm={contract['vfm']}, got {vfm}")
    bad_layers = sorted(set(int(v) for v in layers) - set(contract["layers"] or ())) if contract["layers"] else []
    if bad_layers:
        errors.append(f"namespace pair {contract['tag']} holds layers {list(contract['layers'])} only, got {bad_layers}")
    return errors


def _words(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _ints(value: str) -> list[int]:
    return [int(item) for item in _words(value)]


def _scene_id(row: dict[str, Any]) -> str:
    value = row.get("scene_id")
    if value is not None and str(value):
        return str(value)
    scene_dir = row.get("scene_dir")
    if scene_dir is None:
        raise KeyError("scene_id")
    return Path(str(scene_dir)).name


def _int_list(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f"{label} must be a sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)):
            raise TypeError(f"{label} contains a boolean")
        integer = int(item)
        if isinstance(item, (float, np.floating)) and float(item) != integer:
            raise TypeError(f"{label} contains a non-integral value")
        result.append(integer)
    return result


def _scene_root(root: Path, vfm: str, source: str) -> Path:
    """Resolve the documented ROOT/VFM/SOURCE layout.

    Tests and one-off canaries may pass an already-resolved scene root; that is
    recognized only when its final two path components are exactly VFM/SOURCE.
    """

    if root.name == source and root.parent.name == vfm:
        return root
    return root / vfm / source


def _root_has_namespace(root: Path, namespace: str, vfm: str, source: str) -> bool:
    """Accept only NAMESPACE or NAMESPACE/VFM/SOURCE roots.

    Merely finding the namespace elsewhere in an absolute path is insufficient:
    an old or debug feature tree must never become trainable through a misleading
    ancestor directory name.
    """

    if root.name == source and root.parent.name == vfm:
        return root.parent.parent.name == namespace
    return root.name == namespace


class ProblemCollector:
    def __init__(self) -> None:
        self.count = 0
        self.items: list[dict[str, Any]] = []
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, kind: str, **detail: Any) -> None:
        self.count += 1
        self.counts[kind] += 1
        if len(self.items) < MAX_RECORDED_PROBLEMS:
            self.items.append({"kind": kind, **detail})


def _check_feature_generation_fields(
    value: dict[str, Any],
    *,
    expected_namespace: str,
    vfm: str,
    kind: str,
    problems: ProblemCollector,
    detail: dict[str, Any],
) -> None:
    if value.get("feature_namespace") != expected_namespace:
        problems.add(
            f"{kind}_feature_namespace",
            expected=expected_namespace,
            actual=value.get("feature_namespace"),
            **detail,
        )
    if value.get("feature_generation_contract") != FEATURE_GENERATION_CONTRACT:
        problems.add(
            f"{kind}_feature_generation_contract",
            expected=FEATURE_GENERATION_CONTRACT,
            actual=value.get("feature_generation_contract"),
            **detail,
        )
    if vfm in {"wan", "cogvideox"}:
        if value.get("noise_seed_scheme") != NOISE_SEED_SCHEME:
            problems.add(
                f"{kind}_noise_seed_scheme",
                expected=NOISE_SEED_SCHEME,
                actual=value.get("noise_seed_scheme"),
                **detail,
            )
    elif vfm == "minimax_h3":
        expected = {
            "noise_seed_scheme": MINIMAX_NOISE_SEED_SCHEME,
            "noise_seed": 0,
            "noise_seed_scope": "reset_per_forward",
        }
        for field, wanted in expected.items():
            if value.get(field) != wanted:
                problems.add(
                    f"{kind}_{field}",
                    expected=wanted,
                    actual=value.get(field),
                    **detail,
                )
    else:
        problems.add(f"{kind}_unsupported_vfm", vfm=vfm, **detail)


def _load_manifest(path: Path, problems: ProblemCollector) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception as exc:
                problems.add("manifest_json", line=lineno, detail=repr(exc))
                continue
            if not isinstance(row, dict):
                problems.add("manifest_row_type", line=lineno)
                continue
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                problems.add("manifest_sample_id_missing", line=lineno)
            elif sample_id in sample_ids:
                problems.add("manifest_sample_id_duplicate", line=lineno, sample_id=sample_id)
            sample_ids.add(sample_id)
            row["__line__"] = lineno
            rows.append(row)
    if not rows:
        problems.add("manifest_empty", path=str(path))
    return rows


def _check_manifest_sidecar(
    path: Path,
    *,
    manifest_sha256: str,
    rows: Sequence[dict[str, Any]],
    scenes: set[str],
    protocol_id: str,
    canonical_obs_len: int,
    horizons: Sequence[int],
    problems: ProblemCollector,
) -> dict[str, Any]:
    sidecar = Path(f"{path}.meta.json")
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise TypeError("manifest metadata must be a JSON object")
    except Exception as exc:
        problems.add("manifest_meta_unreadable", path=str(sidecar), detail=repr(exc))
        return {}
    _check_protocol_fields(
        meta,
        expected_protocol=protocol_id,
        expected_obs_len=canonical_obs_len,
        kind="manifest_meta",
        problems=problems,
        detail={"path": str(sidecar)},
    )
    if meta.get("manifest_version") != 2:
        problems.add(
            "manifest_meta_version",
            path=str(sidecar),
            expected=2,
            actual=meta.get("manifest_version"),
        )
    for field, expected_root in (
        ("streaming_feature_root", STREAMING_FEATURE_ROOT),
        ("target_feature_root", TARGET_FEATURE_ROOT),
    ):
        if meta.get(field) != expected_root:
            problems.add(
                f"manifest_meta_{field}",
                path=str(sidecar),
                expected=expected_root,
                actual=meta.get(field),
            )
    if meta.get("sha256") != manifest_sha256:
        problems.add(
            "manifest_meta_sha256",
            path=str(sidecar),
            expected=manifest_sha256,
            actual=meta.get("sha256"),
        )
    if meta.get("num_samples") != len(rows):
        problems.add(
            "manifest_meta_sample_count",
            path=str(sidecar),
            expected=len(rows),
            actual=meta.get("num_samples"),
        )
    if meta.get("num_scenes") != len(scenes):
        problems.add(
            "manifest_meta_scene_count",
            path=str(sidecar),
            expected=len(scenes),
            actual=meta.get("num_scenes"),
        )
    try:
        actual_horizons = _int_list(meta.get("action_horizons"), label="manifest action_horizons")
    except Exception as exc:
        actual_horizons = []
        problems.add("manifest_meta_horizons", path=str(sidecar), detail=repr(exc))
    if actual_horizons != list(horizons):
        problems.add(
            "manifest_meta_horizons",
            path=str(sidecar),
            expected=list(horizons),
            actual=actual_horizons,
        )
    prefix_lengths: set[int] = set()
    for row in rows:
        try:
            prefix_lengths.add(
                int(row.get("prefix_length", len(_int_list(row["streaming_indices"], label="streaming_indices"))))
            )
        except Exception:
            continue
    if len(prefix_lengths) != 1 or meta.get("prefix_length") not in prefix_lengths:
        problems.add(
            "manifest_meta_prefix_length",
            path=str(sidecar),
            expected=sorted(prefix_lengths),
            actual=meta.get("prefix_length"),
        )
    return meta


def _record_key(record: dict[str, Any]) -> tuple[int, int]:
    window_id = record.get("window_id", record.get("streaming_window_id"))
    tail = record.get("tail", record.get("streaming_tail"))
    if tail is None and record.get("prefix_length") is not None:
        tail = int(record["prefix_length"]) - 1
    return int(window_id), int(tail)


def _load_prefix_records(path: Path, scene: str, problems: ProblemCollector) -> dict[tuple[int, int], dict[str, Any]]:
    try:
        array = np.load(path, allow_pickle=True)
        raw_records = array.tolist()
        if isinstance(raw_records, dict):
            raw_records = [raw_records]
        if not isinstance(raw_records, list):
            raise TypeError(f"expected list-like object array, got {type(raw_records).__name__}")
    except Exception as exc:
        problems.add("prefix_index_unreadable", scene=scene, path=str(path), detail=repr(exc))
        return {}

    records: dict[tuple[int, int], dict[str, Any]] = {}
    for offset, record in enumerate(raw_records):
        if not isinstance(record, dict):
            problems.add("prefix_record_type", scene=scene, offset=offset)
            continue
        try:
            key = _record_key(record)
        except Exception as exc:
            problems.add("prefix_record_key", scene=scene, offset=offset, detail=repr(exc))
            continue
        if key in records:
            problems.add("prefix_record_duplicate", scene=scene, window=key[0], tail=key[1])
            continue
        records[key] = record
    return records


def _check_protocol_fields(
    value: dict[str, Any],
    *,
    expected_protocol: str,
    expected_obs_len: int,
    kind: str,
    problems: ProblemCollector,
    detail: dict[str, Any],
) -> None:
    # canonical v2 requires both fields; absence is not treated as a legacy
    # compatibility case because that would make stale indexes trainable.
    if value.get("canonical_protocol") != expected_protocol:
        problems.add(
            f"{kind}_protocol",
            expected=expected_protocol,
            actual=value.get("canonical_protocol"),
            **detail,
        )
    actual_obs_len = value.get("canonical_obs_len")
    if isinstance(actual_obs_len, (bool, np.bool_)):
        actual_obs_len = None
    try:
        actual_obs_len = int(actual_obs_len)
    except (TypeError, ValueError):
        actual_obs_len = None
    if actual_obs_len != expected_obs_len:
        problems.add(
            f"{kind}_obs_len",
            expected=expected_obs_len,
            actual=value.get("canonical_obs_len"),
            **detail,
        )


def _protocol_hash(value: dict[str, Any], *, canonical: bool) -> str:
    key = "canonical_protocol_sha256" if canonical else "protocol_sha256"
    fallback = "protocol_sha256" if canonical else "canonical_protocol_sha256"
    result = value.get(key, value.get(fallback, ""))
    return str(result).lower() if result is not None else ""


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= HEX_DIGITS


def _read_index_sidecar(
    path: Path,
    *,
    scene: str,
    protocol_id: str,
    canonical_obs_len: int,
    expected_namespace: str,
    vfm: str,
    kind: str,
    problems: ProblemCollector,
) -> dict[str, Any]:
    sidecar = Path(f"{path}.manifest.json")
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise TypeError("index sidecar must be a JSON object")
    except Exception as exc:
        problems.add(f"{kind}_sidecar_unreadable", scene=scene, path=str(sidecar), detail=repr(exc))
        return {}
    _check_protocol_fields(
        meta,
        expected_protocol=protocol_id,
        expected_obs_len=canonical_obs_len,
        kind=f"{kind}_sidecar",
        problems=problems,
        detail={"scene": scene, "path": str(sidecar)},
    )
    _check_feature_generation_fields(
        meta,
        expected_namespace=expected_namespace,
        vfm=vfm,
        kind=f"{kind}_sidecar",
        problems=problems,
        detail={"scene": scene, "path": str(sidecar)},
    )
    protocol_hash = _protocol_hash(meta, canonical=True)
    if protocol_hash != DEFAULT_PROTOCOL_SHA256:
        problems.add(
            f"{kind}_sidecar_protocol_sha256",
            scene=scene,
            path=str(sidecar),
            expected=DEFAULT_PROTOCOL_SHA256,
            actual=meta.get("canonical_protocol_sha256"),
        )
    try:
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
    except Exception as exc:
        problems.add(f"{kind}_index_stat", scene=scene, path=str(path), detail=repr(exc))
        return meta
    if meta.get("file") != path.name:
        problems.add(
            f"{kind}_sidecar_file",
            scene=scene,
            path=str(sidecar),
            expected=path.name,
            actual=meta.get("file"),
        )
    if meta.get("size_bytes") != actual_size:
        problems.add(
            f"{kind}_sidecar_size",
            scene=scene,
            path=str(sidecar),
            expected=actual_size,
            actual=meta.get("size_bytes"),
        )
    if meta.get("sha256") != actual_hash:
        problems.add(
            f"{kind}_sidecar_sha256",
            scene=scene,
            path=str(sidecar),
            expected=actual_hash,
            actual=meta.get("sha256"),
        )
    return meta


def _read_target_index(
    path: Path,
    meta_path: Path,
    *,
    scene: str,
    protocol_id: str,
    canonical_obs_len: int,
    expected_namespace: str,
    vfm: str,
    prefixes: Sequence[int],
    horizons: Sequence[int],
    problems: ProblemCollector,
) -> tuple[set[int], dict[str, Any], dict[str, Any]]:
    index_meta = _read_index_sidecar(
        path,
        scene=scene,
        protocol_id=protocol_id,
        canonical_obs_len=canonical_obs_len,
        expected_namespace=expected_namespace,
        vfm=vfm,
        kind="target_index",
        problems=problems,
    )
    try:
        values = np.load(path, allow_pickle=False)
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise TypeError(f"expected 1-D integer array, got shape={values.shape} dtype={values.dtype}")
        target_values = [int(item) for item in values.tolist()]
    except Exception as exc:
        problems.add("target_index_unreadable", scene=scene, path=str(path), detail=repr(exc))
        return set(), index_meta, {}

    if len(target_values) != len(set(target_values)):
        problems.add("target_index_duplicates", scene=scene, path=str(path))

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise TypeError("metadata JSON must be an object")
    except Exception as exc:
        problems.add("target_meta_unreadable", scene=scene, path=str(meta_path), detail=repr(exc))
        return set(target_values), index_meta, {}

    target_protocol = meta.get("protocol", meta.get("canonical_protocol"))
    if target_protocol != protocol_id:
        problems.add(
            "target_meta_protocol",
            scene=scene,
            path=str(meta_path),
            expected=protocol_id,
            actual=target_protocol,
        )
    if (
        meta.get("protocol") is not None
        and meta.get("canonical_protocol") is not None
        and meta["protocol"] != meta["canonical_protocol"]
    ):
        problems.add("target_meta_protocol_alias_mismatch", scene=scene, path=str(meta_path))
    target_obs_len = meta.get("obs_len", meta.get("canonical_obs_len"))
    try:
        target_obs_len_int = int(target_obs_len)
    except (TypeError, ValueError):
        target_obs_len_int = None
    if target_obs_len_int != canonical_obs_len:
        problems.add(
            "target_meta_obs_len",
            scene=scene,
            path=str(meta_path),
            expected=canonical_obs_len,
            actual=target_obs_len,
        )
    _check_feature_generation_fields(
        meta,
        expected_namespace=expected_namespace,
        vfm=vfm,
        kind="target_meta",
        problems=problems,
        detail={"scene": scene, "path": str(meta_path)},
    )
    if (
        meta.get("obs_len") is not None
        and meta.get("canonical_obs_len") is not None
    ):
        try:
            obs_aliases_match = int(meta["obs_len"]) == int(meta["canonical_obs_len"])
        except (TypeError, ValueError):
            obs_aliases_match = False
        if not obs_aliases_match:
            problems.add("target_meta_obs_len_alias_mismatch", scene=scene, path=str(meta_path))
    target_protocol_hash = _protocol_hash(meta, canonical=False)
    if target_protocol_hash != DEFAULT_PROTOCOL_SHA256:
        problems.add(
            "target_meta_protocol_sha256",
            scene=scene,
            path=str(meta_path),
            expected=DEFAULT_PROTOCOL_SHA256,
            actual=meta.get("protocol_sha256", meta.get("canonical_protocol_sha256")),
        )
    index_protocol_hash = _protocol_hash(index_meta, canonical=True)
    if (
        _valid_sha256(target_protocol_hash)
        and _valid_sha256(index_protocol_hash)
        and target_protocol_hash != index_protocol_hash
    ):
        problems.add(
            "target_protocol_sha256_mismatch",
            scene=scene,
            target_meta=target_protocol_hash,
            index_sidecar=index_protocol_hash,
        )
    try:
        actual_prefixes = _int_list(meta.get("prefixes"), label="target meta prefixes")
    except Exception as exc:
        actual_prefixes = []
        problems.add("target_meta_prefixes", scene=scene, path=str(meta_path), detail=repr(exc))
    if actual_prefixes != list(prefixes):
        problems.add(
            "target_meta_prefixes",
            scene=scene,
            path=str(meta_path),
            expected=list(prefixes),
            actual=actual_prefixes,
        )
    try:
        actual_horizons = _int_list(meta.get("horizons"), label="target meta horizons")
    except Exception as exc:
        actual_horizons = []
        problems.add("target_meta_horizons", scene=scene, path=str(meta_path), detail=repr(exc))
    if actual_horizons != list(horizons):
        problems.add(
            "target_meta_horizons",
            scene=scene,
            path=str(meta_path),
            expected=list(horizons),
            actual=actual_horizons,
        )
    if meta.get("include_anchors") is not True:
        problems.add(
            "target_meta_include_anchors",
            scene=scene,
            path=str(meta_path),
            actual=meta.get("include_anchors"),
        )
    if meta.get("target_count") != len(target_values):
        problems.add(
            "target_meta_count",
            scene=scene,
            path=str(meta_path),
            expected=len(target_values),
            actual=meta.get("target_count"),
        )
    actual_hash = sha256_file(path)
    if meta.get("target_indices_sha256") != actual_hash:
        problems.add(
            "target_meta_sha256",
            scene=scene,
            path=str(meta_path),
            expected=actual_hash,
            actual=meta.get("target_indices_sha256"),
        )
    return set(target_values), index_meta, meta


def audit_alignment(
    *,
    manifest: Path,
    streaming_index_root: Path,
    target_index_root: Path,
    vfm: str,
    source: str = "scannetpp",
    protocol_id: str = DEFAULT_PROTOCOL_ID,
    canonical_obs_len: int = DEFAULT_CANONICAL_OBS_LEN,
    prefixes: Sequence[int] = DEFAULT_PREFIXES,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    scene_ids: Iterable[str] = (),
    streaming_namespace: str = STREAMING_NAMESPACE,
    target_namespace: str = TARGET_NAMESPACE,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    streaming_index_root = streaming_index_root.resolve()
    target_index_root = target_index_root.resolve()
    prefixes = tuple(int(value) for value in prefixes)
    horizons = tuple(int(value) for value in horizons)
    selected_scene_ids = {str(value) for value in scene_ids if str(value)}
    problems = ProblemCollector()
    for error in namespace_pair_errors(streaming_namespace, target_namespace, vfm):
        problems.add("namespace_pair_invalid", detail=error)

    if canonical_obs_len <= 0:
        problems.add("canonical_obs_len_invalid", actual=canonical_obs_len)
    if not prefixes or sorted(set(prefixes)) != list(prefixes):
        problems.add("prefix_contract_invalid", actual=list(prefixes))
    if not horizons or sorted(set(horizons)) != list(horizons) or any(value <= 0 for value in horizons):
        problems.add("horizon_contract_invalid", actual=list(horizons))

    try:
        manifest_hash = sha256_file(manifest)
        all_rows = _load_manifest(manifest, problems)
    except Exception as exc:
        manifest_hash = ""
        all_rows = []
        problems.add("manifest_unreadable", path=str(manifest), detail=repr(exc))

    manifest_scenes: set[str] = set()
    for row in all_rows:
        try:
            manifest_scenes.add(_scene_id(row))
        except Exception as exc:
            problems.add("manifest_scene", line=row.get("__line__"), detail=repr(exc))

    manifest_meta = _check_manifest_sidecar(
        manifest,
        manifest_sha256=manifest_hash,
        rows=all_rows,
        scenes=manifest_scenes,
        protocol_id=protocol_id,
        canonical_obs_len=canonical_obs_len,
        horizons=horizons,
        problems=problems,
    )

    unknown_scenes = sorted(selected_scene_ids - manifest_scenes)
    for scene in unknown_scenes:
        problems.add("requested_scene_missing", scene=scene)
    rows = [row for row in all_rows if not selected_scene_ids or _scene_id(row) in selected_scene_ids]
    scope = "canary" if selected_scene_ids else "full"
    if not rows:
        problems.add("audit_selection_empty", selected_scenes=sorted(selected_scene_ids))

    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            rows_by_scene[_scene_id(row)].append(row)
        except Exception:
            # The precise malformed-row problem was recorded above.
            continue

    stream_scene_root = _scene_root(streaming_index_root, vfm, source)
    target_scene_root = _scene_root(target_index_root, vfm, source)
    if not _root_has_namespace(
        streaming_index_root, streaming_namespace, vfm, source
    ):
        problems.add(
            "streaming_index_root_namespace",
            expected=streaming_namespace,
            actual=str(streaming_index_root),
        )
    if not _root_has_namespace(target_index_root, target_namespace, vfm, source):
        problems.add(
            "target_index_root_namespace",
            expected=target_namespace,
            actual=str(target_index_root),
        )
    prefix_indexes_checked = 0
    target_indexes_checked = 0
    samples_checked = 0
    target_memberships_checked = 0
    required_target_values: dict[str, set[int]] = defaultdict(set)

    for scene, scene_rows in sorted(rows_by_scene.items()):
        prefix_path = stream_scene_root / scene / "prefix_index.npy"
        target_path = target_scene_root / scene / "target_indices.npy"
        target_meta_path = target_path.with_name("target_indices.meta.json")
        prefix_sidecar = _read_index_sidecar(
            prefix_path,
            scene=scene,
            protocol_id=protocol_id,
            canonical_obs_len=canonical_obs_len,
            expected_namespace=streaming_namespace,
            vfm=vfm,
            kind="prefix_index",
            problems=problems,
        )
        records = _load_prefix_records(prefix_path, scene, problems)
        if records:
            prefix_indexes_checked += 1
        target_values, target_sidecar, _target_meta = _read_target_index(
            target_path,
            target_meta_path,
            scene=scene,
            protocol_id=protocol_id,
            canonical_obs_len=canonical_obs_len,
            expected_namespace=target_namespace,
            vfm=vfm,
            prefixes=prefixes,
            horizons=horizons,
            problems=problems,
        )
        prefix_protocol_hash = _protocol_hash(prefix_sidecar, canonical=True)
        target_protocol_hash = _protocol_hash(target_sidecar, canonical=True)
        if (
            _valid_sha256(prefix_protocol_hash)
            and _valid_sha256(target_protocol_hash)
            and prefix_protocol_hash != target_protocol_hash
        ):
            problems.add(
                "stream_target_protocol_sha256_mismatch",
                scene=scene,
                streaming=prefix_protocol_hash,
                target=target_protocol_hash,
            )
        if (
            target_path.is_file()
            and Path(f"{target_path}.manifest.json").is_file()
            and target_meta_path.is_file()
        ):
            target_indexes_checked += 1

        for row in scene_rows:
            line = row.get("__line__")
            sample_id = str(row.get("sample_id", ""))
            detail = {"scene": scene, "sample_id": sample_id, "line": line}
            _check_protocol_fields(
                row,
                expected_protocol=protocol_id,
                expected_obs_len=canonical_obs_len,
                kind="manifest",
                problems=problems,
                detail=detail,
            )
            try:
                source_value = str(row["source"])
                window_id = int(row["streaming_window_id"])
                prefix_length = int(row.get("prefix_length", 0))
                tail_value = row.get("streaming_tail")
                tail = int(tail_value) if tail_value is not None else prefix_length - 1
                manifest_indices = _int_list(row["streaming_indices"], label="streaming_indices")
                manifest_window = _int_list(
                    row.get("window_indices", row.get("streaming_window_indices")),
                    label="streaming_window_indices",
                )
            except Exception as exc:
                problems.add("manifest_contract", detail=repr(exc), **detail)
                continue
            if source_value != source:
                problems.add("manifest_source", expected=source, actual=source_value, **detail)
            if row.get("streaming_feature_root") != STREAMING_FEATURE_ROOT:
                problems.add(
                    "manifest_streaming_feature_root",
                    expected=STREAMING_FEATURE_ROOT,
                    actual=row.get("streaming_feature_root"),
                    **detail,
                )
            if row.get("target_feature_root") != TARGET_FEATURE_ROOT:
                problems.add(
                    "manifest_target_feature_root",
                    expected=TARGET_FEATURE_ROOT,
                    actual=row.get("target_feature_root"),
                    **detail,
                )
            key = (window_id, tail)
            if prefix_length == 0:
                prefix_length = len(manifest_indices)
            if prefix_length != tail + 1 or prefix_length != len(manifest_indices):
                problems.add(
                    "manifest_prefix_length",
                    expected=tail + 1,
                    actual=row.get("prefix_length"),
                    streaming_indices_length=len(manifest_indices),
                    **detail,
                )
            if row.get("future_horizon") is not None:
                try:
                    future_horizon = int(row["future_horizon"])
                except Exception:
                    future_horizon = None
                if future_horizon not in horizons:
                    problems.add(
                        "manifest_future_horizon",
                        expected=list(horizons),
                        actual=row.get("future_horizon"),
                        **detail,
                    )
            if len(manifest_window) != canonical_obs_len:
                problems.add(
                    "manifest_window_length",
                    expected=canonical_obs_len,
                    actual=len(manifest_window),
                    **detail,
                )
            if tail < 0 or tail >= len(manifest_window):
                problems.add("manifest_tail_range", tail=tail, window_length=len(manifest_window), **detail)
                continue
            expected_indices = manifest_window[: tail + 1]
            if manifest_indices != expected_indices:
                problems.add(
                    "manifest_prefix_not_window_prefix",
                    expected=expected_indices,
                    actual=manifest_indices,
                    **detail,
                )
            if row.get("streaming_tail_raw") is not None:
                try:
                    tail_raw = int(row["streaming_tail_raw"])
                except Exception:
                    tail_raw = None
                if tail_raw != manifest_window[tail]:
                    problems.add(
                        "manifest_tail_raw",
                        expected=manifest_window[tail],
                        actual=row.get("streaming_tail_raw"),
                        **detail,
                    )

            record = records.get(key)
            if record is None:
                problems.add("prefix_record_missing", window=window_id, tail=tail, **detail)
            else:
                _check_protocol_fields(
                    record,
                    expected_protocol=protocol_id,
                    expected_obs_len=canonical_obs_len,
                    kind="prefix_record",
                    problems=problems,
                    detail={**detail, "window": window_id, "tail": tail},
                )
                record_protocol_hash = _protocol_hash(record, canonical=True)
                if not _valid_sha256(record_protocol_hash):
                    problems.add(
                        "prefix_record_protocol_sha256",
                        actual=record.get("canonical_protocol_sha256"),
                        **detail,
                    )
                elif (
                    _valid_sha256(prefix_protocol_hash)
                    and record_protocol_hash != prefix_protocol_hash
                ):
                    problems.add(
                        "prefix_protocol_sha256_mismatch",
                        record=record_protocol_hash,
                        index_sidecar=prefix_protocol_hash,
                        **detail,
                    )
                try:
                    record_indices = _int_list(
                        record.get("streaming_indices", record.get("indices")),
                        label="record streaming_indices",
                    )
                except Exception as exc:
                    record_indices = []
                    problems.add("prefix_record_indices", detail=repr(exc), **detail)
                try:
                    record_window = _int_list(record.get("window_indices"), label="record window_indices")
                except Exception as exc:
                    record_window = []
                    problems.add("prefix_record_window_indices", detail=repr(exc), **detail)
                if record_indices != manifest_indices:
                    problems.add(
                        "prefix_indices_mismatch",
                        expected=manifest_indices,
                        actual=record_indices,
                        **detail,
                    )
                if record_window != manifest_window:
                    problems.add(
                        "prefix_window_mismatch",
                        expected=manifest_window,
                        actual=record_window,
                        **detail,
                    )
                record_prefix_length = record.get("prefix_length")
                if record_prefix_length is not None and int(record_prefix_length) != prefix_length:
                    problems.add(
                        "prefix_record_prefix_length",
                        expected=prefix_length,
                        actual=record_prefix_length,
                        **detail,
                    )
                if record.get("valid_length") is not None:
                    try:
                        valid_length = int(record["valid_length"])
                    except Exception:
                        valid_length = None
                    if valid_length != len(manifest_indices):
                        problems.add(
                            "prefix_valid_length",
                            expected=len(manifest_indices),
                            actual=record.get("valid_length"),
                            **detail,
                        )
                if row.get("streaming_obs_start") is not None and record.get("obs_start") is not None:
                    if int(row["streaming_obs_start"]) != int(record["obs_start"]):
                        problems.add(
                            "prefix_obs_start_mismatch",
                            expected=int(row["streaming_obs_start"]),
                            actual=int(record["obs_start"]),
                            **detail,
                        )

            requested_positions = [("anchor", 0, tail)] + [
                ("horizon", horizon, tail + horizon) for horizon in horizons
            ]
            for target_kind, horizon, position in requested_positions:
                if position < 0 or position >= len(manifest_window):
                    problems.add(
                        "target_position_out_of_window",
                        target_kind=target_kind,
                        horizon=horizon,
                        position=position,
                        window_length=len(manifest_window),
                        **detail,
                    )
                    continue
                raw_id = manifest_window[position]
                required_target_values[scene].add(raw_id)
                target_memberships_checked += 1
                if raw_id not in target_values:
                    problems.add(
                        "target_frame_missing",
                        target_kind=target_kind,
                        horizon=horizon,
                        raw_id=raw_id,
                        **detail,
                    )
            samples_checked += 1

    clean_rows = [{key: value for key, value in row.items() if key != "__line__"} for row in all_rows]
    manifest_protocol_values = sorted(
        {str(row.get("canonical_protocol")) for row in clean_rows if row.get("canonical_protocol") is not None}
    )
    manifest_obs_len_values: list[int | str] = sorted(
        {int(row["canonical_obs_len"]) for row in clean_rows if row.get("canonical_obs_len") is not None}
    ) if all(
        not isinstance(row.get("canonical_obs_len"), bool)
        and str(row.get("canonical_obs_len", "")).lstrip("-").isdigit()
        for row in clean_rows
    ) else ["invalid"]

    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_type": AUDIT_TYPE,
        "ok": problems.count == 0,
        "status": "pass" if problems.count == 0 else "fail",
        "scope": scope,
        "protocol_id": protocol_id,
        "protocol": protocol_id,
        "pass": problems.count == 0,
        "canonical_obs_len": canonical_obs_len,
        "prefixes": list(prefixes),
        "horizons": list(horizons),
        "include_anchors": True,
        "vfm": vfm,
        "source": source,
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "manifest_meta": str(Path(f"{manifest}.meta.json")),
        "manifest_version": manifest_meta.get("manifest_version"),
        "manifest_sample_count": len(all_rows),
        "manifest_scene_count": len(manifest_scenes),
        "manifest_protocol_values": manifest_protocol_values,
        "manifest_obs_len_values": manifest_obs_len_values,
        "manifest_streaming_feature_root": STREAMING_FEATURE_ROOT,
        "manifest_target_feature_root": TARGET_FEATURE_ROOT,
        "canonical_protocol_sha256": DEFAULT_PROTOCOL_SHA256,
        "streaming_namespace": streaming_namespace,
        "target_namespace": target_namespace,
        "feature_generation_contract": FEATURE_GENERATION_CONTRACT,
        "selected_scenes": sorted(selected_scene_ids),
        "samples_checked": samples_checked,
        "scenes_checked": len(rows_by_scene),
        "prefix_indexes_checked": prefix_indexes_checked,
        "target_indexes_checked": target_indexes_checked,
        "target_memberships_checked": target_memberships_checked,
        "required_unique_target_frames": {
            scene: len(values) for scene, values in sorted(required_target_values.items())
        },
        "streaming_index_root": str(streaming_index_root),
        "target_index_root": str(target_index_root),
        "problem_count": problems.count,
        "problem_counts": dict(sorted(problems.counts.items())),
        "problems_truncated": problems.count > len(problems.items),
        "problems": problems.items,
    }
    return report


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        # SCO workers run as ``nobody`` while campaign submission and matrix
        # construction run as the workspace owner.  mkstemp defaults to 0600,
        # which makes an otherwise valid audit artifact unreadable to the
        # fail-closed submit gate after the worker exits.  Set the final public
        # metadata mode on the still-private temporary inode before the atomic
        # rename; feature tensors themselves are not affected.
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def verify_audit_artifact(
    *,
    artifact: Path,
    manifest: Path,
    vfm: str,
    source: str,
    protocol_id: str,
    canonical_obs_len: int,
    prefix: int,
    expected_artifact_sha256: str = "",
    require_full: bool = True,
    streaming_namespace: str = STREAMING_NAMESPACE,
    target_namespace: str = TARGET_NAMESPACE,
) -> list[str]:
    errors: list[str] = list(namespace_pair_errors(streaming_namespace, target_namespace, vfm))
    artifact = artifact.resolve()
    manifest = manifest.resolve()
    try:
        artifact_hash = sha256_file(artifact)
        report = json.loads(artifact.read_text(encoding="utf-8"))
    except Exception as exc:
        return errors + [f"unreadable audit artifact {artifact}: {exc!r}"]
    if expected_artifact_sha256 and artifact_hash != expected_artifact_sha256:
        errors.append("audit artifact SHA256 does not match groups.tsv")
    expected: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_type": AUDIT_TYPE,
        "ok": True,
        "status": "pass",
        "protocol_id": protocol_id,
        "protocol": protocol_id,
        "pass": True,
        "canonical_obs_len": canonical_obs_len,
        "vfm": vfm,
        "source": source,
        "manifest": str(manifest),
        "manifest_version": 2,
        "problem_count": 0,
        "include_anchors": True,
        "manifest_streaming_feature_root": STREAMING_FEATURE_ROOT,
        "manifest_target_feature_root": TARGET_FEATURE_ROOT,
        "canonical_protocol_sha256": DEFAULT_PROTOCOL_SHA256,
        "streaming_namespace": streaming_namespace,
        "target_namespace": target_namespace,
        "feature_generation_contract": FEATURE_GENERATION_CONTRACT,
    }
    if require_full:
        expected["scope"] = "full"
    for key, value in expected.items():
        if report.get(key) != value:
            errors.append(f"audit field {key!r}: expected {value!r}, got {report.get(key)!r}")
    try:
        current_manifest_hash = sha256_file(manifest)
    except Exception as exc:
        errors.append(f"cannot hash manifest: {exc!r}")
        current_manifest_hash = ""
    if report.get("manifest_sha256") != current_manifest_hash:
        errors.append("audit manifest SHA256 is stale or mismatched")
    if report.get("samples_checked") != report.get("manifest_sample_count"):
        errors.append("audit did not check every manifest sample")
    if report.get("scenes_checked") != report.get("manifest_scene_count"):
        errors.append("audit did not check every manifest scene")
    if report.get("prefix_indexes_checked") != report.get("manifest_scene_count"):
        errors.append("audit did not load one prefix index per manifest scene")
    if report.get("target_indexes_checked") != report.get("manifest_scene_count"):
        errors.append("audit did not load one target index contract per manifest scene")
    expected_memberships = int(report.get("manifest_sample_count", 0)) * (
        1 + len(DEFAULT_HORIZONS)
    )
    if report.get("target_memberships_checked") != expected_memberships:
        errors.append(
            f"audit target membership count is not sample_count*4 ({expected_memberships})"
        )
    if list(report.get("horizons", [])) != list(DEFAULT_HORIZONS):
        errors.append(f"audit horizons are not {list(DEFAULT_HORIZONS)}")
    if list(report.get("prefixes", [])) != list(DEFAULT_PREFIXES):
        errors.append(f"audit prefix contract is not {list(DEFAULT_PREFIXES)}")
    if int(prefix) not in report.get("prefixes", []):
        errors.append(f"group prefix {prefix} is absent from audit prefix contract")
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="produce a strict alignment audit artifact")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--streaming-index-root", required=True)
    audit.add_argument("--target-index-root", required=True)
    audit.add_argument("--vfm", required=True)
    audit.add_argument("--source", default="scannetpp")
    audit.add_argument("--protocol-id", default=DEFAULT_PROTOCOL_ID)
    audit.add_argument("--canonical-obs-len", type=int, default=DEFAULT_CANONICAL_OBS_LEN)
    audit.add_argument("--prefixes", default=" ".join(map(str, DEFAULT_PREFIXES)))
    audit.add_argument("--horizons", default=" ".join(map(str, DEFAULT_HORIZONS)))
    audit.add_argument(
        "--scene-ids",
        default="",
        help="optional comma/space-separated scene canary; omitted means full audit",
    )
    audit.add_argument("--streaming-namespace", default=STREAMING_NAMESPACE)
    audit.add_argument("--target-namespace", default=TARGET_NAMESPACE)
    audit.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify-artifact", help="verify a full audit before training")
    verify.add_argument("--artifact", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--vfm", required=True)
    verify.add_argument("--source", default="scannetpp")
    verify.add_argument("--protocol-id", default=DEFAULT_PROTOCOL_ID)
    verify.add_argument("--canonical-obs-len", type=int, default=DEFAULT_CANONICAL_OBS_LEN)
    verify.add_argument("--prefix", type=int, required=True)
    verify.add_argument("--expected-artifact-sha256", default="")
    verify.add_argument("--streaming-namespace", default=STREAMING_NAMESPACE)
    verify.add_argument("--target-namespace", default=TARGET_NAMESPACE)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "audit":
        report = audit_alignment(
            manifest=Path(args.manifest),
            streaming_index_root=Path(args.streaming_index_root),
            target_index_root=Path(args.target_index_root),
            vfm=args.vfm,
            source=args.source,
            protocol_id=args.protocol_id,
            canonical_obs_len=args.canonical_obs_len,
            prefixes=_ints(args.prefixes),
            horizons=_ints(args.horizons),
            scene_ids=_words(args.scene_ids),
            streaming_namespace=args.streaming_namespace,
            target_namespace=args.target_namespace,
        )
        write_json_atomic(Path(args.output), report)
        print(json.dumps({key: report[key] for key in (
            "ok", "scope", "vfm", "samples_checked", "scenes_checked", "problem_count", "problem_counts"
        )}, indent=2, sort_keys=True))
        raise SystemExit(0 if report["ok"] else 1)

    errors = verify_audit_artifact(
        artifact=Path(args.artifact),
        manifest=Path(args.manifest),
        vfm=args.vfm,
        source=args.source,
        protocol_id=args.protocol_id,
        canonical_obs_len=args.canonical_obs_len,
        prefix=args.prefix,
        expected_artifact_sha256=args.expected_artifact_sha256,
        streaming_namespace=args.streaming_namespace,
        target_namespace=args.target_namespace,
    )
    if errors:
        for error in errors:
            print(f"[strict-audit] {error}")
        raise SystemExit(1)
    print(f"[strict-audit] PASS artifact={Path(args.artifact).resolve()}")


if __name__ == "__main__":
    main()
